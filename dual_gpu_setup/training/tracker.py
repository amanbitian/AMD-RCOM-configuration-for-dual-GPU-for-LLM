"""Stdlib-only training tracker client.

Imported by the training script; it never blocks or fails training. Records are batched by a
background thread. If the dashboard is unreachable they are appended to a local JSONL spool and
replayed in order once it returns. Token counters stay on the device between logging windows, so
there is no per-microbatch GPU->CPU synchronisation.

Multi-GPU (DDP/torchrun): every rank constructs a tracker with the same run id. Rank 0 owns the
run (create/finish and the global, all-reduced token totals); every rank reports its own timing,
memory and communication rows so stragglers and per-card VRAM peaks are visible.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import math
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterator

from .checkpoints import inspect_resume_state, read_manifest, validate_manifest, write_manifest
from .environment import collect_environment


def _numel(value: Any) -> int:
    numel = getattr(value, "numel", None)
    if callable(numel):
        return int(numel())
    size = getattr(value, "size", None)
    if isinstance(size, int):
        return size
    return sum(len(row) for row in value)


def _torch() -> Any:
    return sys.modules.get("torch")


def _cuda_ready(torch: Any) -> bool:
    try:
        return torch is not None and torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _dist(torch: Any) -> Any:
    try:
        dist = torch.distributed
        return dist if dist.is_available() and dist.is_initialized() else None
    except Exception:  # noqa: BLE001
        return None


def resolve_run_id(run_id: str | None = None) -> str:
    """Share one run id across ranks: explicit value, then DGPU_RUN_ID, then a broadcast from rank 0."""
    if run_id:
        return run_id
    if os.environ.get("DGPU_RUN_ID"):
        return os.environ["DGPU_RUN_ID"]
    torch = _torch()
    dist = _dist(torch) if torch is not None else None
    generated = uuid.uuid4().hex
    if dist is not None and dist.get_world_size() > 1:
        holder = [generated]
        dist.broadcast_object_list(holder, src=0)
        return holder[0]
    return generated


class LeaseConflictError(RuntimeError):
    """The requested GPU lanes are leased by another job or in use by inference."""


class TrainingTracker:
    def __init__(
        self,
        server: str | None = None,
        name: str | None = None,
        lanes: list[str] | tuple[str, ...] = (),
        config: dict[str, Any] | None = None,
        dataset: dict[str, Any] | None = None,
        log_every: int | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
        parent_run_id: str | None = None,
        run_id: str | None = None,
        spool_dir: str | Path = ".tracker_spool",
        heartbeat_s: float = 10.0,
        flush_s: float = 2.0,
        timeout_s: float = 5.0,
        warmup_windows: int = 1,
        capture_environment: bool = True,
        enabled: bool = True,
        require_lease: bool = False,
        unload_inference: bool = False,
        lease_ttl_s: float = 300.0,
        rank: int | None = None,
        world_size: int | None = None,
        tags: dict[str, Any] | None = None,
        pci_bus: str | None = None,
    ):
        self.rank = int(os.environ.get("RANK", 0)) if rank is None else int(rank)
        self.world_size = int(os.environ.get("WORLD_SIZE", 1)) if world_size is None else int(world_size)
        self.is_primary = self.rank == 0
        env = os.environ
        self.server = (server or env.get("DGPU_TRACKER_SERVER") or "http://127.0.0.1:8090").rstrip("/")
        self.name = name or env.get("DGPU_RUN_NAME") or "training-run"
        self.lanes = list(lanes) or [x for x in env.get("DGPU_LANES", "").split(",") if x]
        self.config = config
        self.dataset = dataset
        self.log_every = log_every
        self.max_steps = max_steps
        self.seed = seed
        self.parent_run_id = parent_run_id
        self.tags = dict(tags or {})
        if env.get("DGPU_BENCH_ID"):  # launched by the benchmark runner: group the trial with its sweep
            self.tags.setdefault("benchmark_id", env["DGPU_BENCH_ID"])
            try:
                self.tags.setdefault("benchmark_params", json.loads(env.get("DGPU_PARAMS", "{}")))
            except ValueError:
                pass
        self.tags = self.tags or None
        self.run_id = resolve_run_id(run_id) if (enabled and self.world_size > 1) else (
            run_id or env.get("DGPU_RUN_ID") or uuid.uuid4().hex)
        self.pci_bus = pci_bus or os.environ.get("DGPU_PCI_BUS")
        self.heartbeat_s = heartbeat_s
        self.flush_s = flush_s
        self.timeout_s = timeout_s
        self.warmup_windows = warmup_windows
        self.capture_environment = capture_environment
        self.enabled = enabled
        self.require_lease = require_lease
        self.unload_inference = unload_inference
        self.lease_ttl_s = lease_ttl_s
        self.lease: dict[str, Any] | None = None
        self.artifact_dir: str | None = None
        suffix = "" if self.is_primary else f".rank{self.rank}"
        self.spool_path = Path(spool_dir) / f"{self.run_id}{suffix}.jsonl"
        self.last_error: str | None = None
        self.dropped_records = 0

        self._queue: queue.Queue = queue.Queue(maxsize=50000)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        # Non-blocking checkpoint() hashes a manifest off-thread then enqueues it; track those
        # threads so finish() can wait for the enqueue before draining -- otherwise a fast
        # finish can stop the worker first and silently drop the checkpoint.
        self._manifest_threads: list[threading.Thread] = []
        self._started = False
        self._finished = False
        self._failed = False
        self._last_status: int | None = None
        self._spool_pending = self.spool_path.exists() and self.spool_path.stat().st_size > 0
        self._last_replay = 0.0
        self._last_send = 0.0

        self._last_step = 0
        self._total_nonpad = 0
        self._windows_sent = 0
        self._reset_counters()
        self._window_scalars: dict[str, Any] = {}
        self._window_nan = 0
        self._ready: dict[str, Any] | None = None
        self._phase_stack: list[str] = []
        self._pending_resume: dict[str, Any] | None = None
        self.enqueue_ms_total = 0.0
        self.comm_events: list[tuple[Any, Any]] = []

    # ------------------------------------------------------------ lifecycle

    def start(self) -> "TrainingTracker":
        if not self.enabled or self._started:
            return self
        self._started = True
        if self.pci_bus is None:
            self.pci_bus = self._detect_pci_bus()
        if self.is_primary:
            self._create_run()
        self._worker = threading.Thread(target=self._run, name="training-tracker", daemon=True)
        self._worker.start()
        atexit.register(self._at_exit)
        if self._pending_resume:
            self.event("resumed", severity="info", **self._pending_resume)
        return self

    @staticmethod
    def _detect_pci_bus() -> str | None:
        """PCI address of this process's current GPU, when PyTorch exposes it."""
        torch = _torch()
        try:
            if not _cuda_ready(torch):
                return None
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            domain, bus, device = (getattr(props, a, None) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
            if bus is None:
                return None
            return f"{domain or 0:04x}:{bus:02x}:{device or 0:02x}.0"
        except Exception:  # noqa: BLE001
            return None

    def _create_run(self) -> None:
        environment = collect_environment() if self.capture_environment else None
        git = (environment or {}).get("git") or {}
        create = {
            "id": self.run_id, "name": self.name, "lanes": self.lanes, "config": self.config,
            "dataset": self.dataset, "env": environment, "seed": self.seed,
            "max_steps": self.max_steps, "parent_run_id": self.parent_run_id, "tags": self.tags,
            "git_commit": git.get("commit"), "started_at": time.time(), "pid": os.getpid(),
            "unload_inference": self.unload_inference, "lease_ttl_s": self.lease_ttl_s,
            "world_size": self.world_size,
        }
        if self.require_lease and self.lanes:
            # Fail fast (before any GPU work) if the lanes are taken; fall back to async if the
            # dashboard is simply unreachable so training is never blocked by telemetry.
            outcome = self._post("create", create)
            if outcome == "drop" and self._last_status == 409:
                self._finished = True
                raise LeaseConflictError(self.last_error or "GPU lane unavailable.")
            if outcome == "retry":
                self._enqueue("create", create)
        else:
            self._enqueue("create", create)

    def resume_from(self, checkpoint_dir: str | Path, verify_hashes: bool = False) -> dict[str, Any]:
        """Link this run to the run that wrote `checkpoint_dir`, validating its manifest first.
        Returns the validation report (also logged as a `resumed` event once started)."""
        directory = Path(checkpoint_dir)
        ok, reason = validate_manifest(directory, verify_hashes)
        manifest = read_manifest(directory) or {}
        report = {
            "checkpoint": str(directory), "manifest_valid": ok, "reason": reason,
            "from_step": manifest.get("step"), "tokens_seen": manifest.get("tokens_seen"),
            "parent_run_id": manifest.get("run_id"), **inspect_resume_state(directory),
        }
        if manifest.get("run_id"):
            self.parent_run_id = manifest["run_id"]
        if manifest.get("tokens_seen"):
            self._total_nonpad = int(manifest["tokens_seen"])
        self._pending_resume = report
        if self._started:
            self.event("resumed", severity="info" if ok else "warning", **report)
        return report

    def __enter__(self) -> "TrainingTracker":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finish("completed")
        else:
            self.finish("failed", error="".join(traceback.format_exception(exc_type, exc, tb))[-8000:])

    def finish(self, status: str = "completed", error: str | None = None, summary: dict[str, Any] | None = None) -> None:
        if not self.enabled or self._finished:
            return
        self._finished = True
        self._failed = status != "completed"
        self._close_window()
        self._flush_ready()
        # Let pending checkpoint-manifest threads enqueue before we stop the worker, so a
        # checkpoint saved just before finish() is never dropped.
        for thread in self._manifest_threads:
            thread.join(timeout=15)
        self._manifest_threads = []
        if self.is_primary:
            self._enqueue("finish", {"status": status, "error": error, "summary": summary})
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=15)

    def _at_exit(self) -> None:
        if not self._finished:
            self.finish("failed", error="Process exited before tracker.finish() was called.")

    # ------------------------------------------------------------ recording

    def _reset_counters(self) -> None:
        self._processed = 0
        self._nonpad: Any = 0
        self._supervised: Any = 0
        self._base_processed = 0
        self._base_nonpad = 0
        self._base_supervised = 0
        self._window_steps = 0
        self._window_start = time.perf_counter()
        self._last_step_time = self._window_start
        self._excluded_ms = 0.0
        self._excluded_at_step_end = 0.0
        self._durations: list[float] = []
        self._data_wait_ms = 0.0
        self._gpu_events: dict[str, list[tuple[Any, Any]]] = {}
        self._step_event: Any = None
        self._first_step_seen = False

    def microbatch(self, input_ids: Any = None, attention_mask: Any = None, labels: Any = None) -> None:
        """Count one microbatch. Tensors stay on the device; nothing is read back here."""
        if not self.enabled:
            return
        if self._ready is not None:
            self._flush_ready()
        if input_ids is not None:
            elements = _numel(input_ids)
            self._processed += elements
            if attention_mask is None:
                self._nonpad = self._nonpad + elements
        if attention_mask is not None:
            self._nonpad = self._nonpad + attention_mask.sum()
        if labels is not None:
            self._supervised = self._supervised + (labels[..., 1:] != -100).sum()

    def microbatch_counts(self, processed: int, nonpad: int, supervised: int) -> None:
        """For custom loops that already know their (host-side) token counts."""
        if not self.enabled:
            return
        if self._ready is not None:
            self._flush_ready()
        self._processed += int(processed)
        self._nonpad = self._nonpad + int(nonpad)
        self._supervised = self._supervised + int(supervised)

    def step_begin(self) -> None:
        """Optional: call at the start of each optimizer step. Measures how long the loop waited
        for data (gap since the previous step ended, minus eval/checkpoint time) and anchors a
        CUDA event so GPU step time can be split into forward / backward / optimizer."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._first_step_seen:
            gap = (now - self._last_step_time) * 1000.0 - (self._excluded_ms - self._excluded_at_step_end)
            self._data_wait_ms += max(0.0, gap)
        self._first_step_seen = True
        torch = _torch()
        if _cuda_ready(torch):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._step_event = event

    def step_completed(self, step: int | None = None) -> None:
        """Call once per optimizer step (after gradient accumulation)."""
        if not self.enabled:
            return
        if self._ready is not None:
            self._flush_ready()
        now = time.perf_counter()
        self._last_step = self._last_step + 1 if step is None else int(step)
        self._durations.append((now - self._last_step_time) * 1000.0)
        self._last_step_time = now
        self._excluded_at_step_end = self._excluded_ms
        self._window_steps += 1
        torch = _torch()
        if self._step_event is not None and _cuda_ready(torch):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._gpu_events.setdefault("step", []).append((self._step_event, end))
            self._step_event = None
        if self.log_every is None:
            self.log_every = 20
        if self._window_steps >= self.log_every:
            self._close_window()

    def gpu_phase(self, name: str) -> "contextlib.AbstractContextManager[None]":
        """Time a region on the GPU stream with CUDA events (read once per window, never synced
        per call). Use for 'forward', 'optimizer', or any custom region."""
        return self._gpu_phase(name)

    @contextlib.contextmanager
    def _gpu_phase(self, name: str) -> Iterator[None]:
        torch = _torch()
        if not self.enabled or not _cuda_ready(torch):
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._gpu_events.setdefault(name, []).append((start, end))

    def gpu_mark(self) -> Any:
        """Record a CUDA event now (or None without CUDA). Pair with gpu_record()."""
        torch = _torch()
        if not self.enabled or not _cuda_ready(torch):
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def gpu_record(self, name: str, start: Any) -> None:
        if start is None:
            return
        torch = _torch()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._gpu_events.setdefault(name, []).append((start, end))

    def wrap_loader(self, loader: Any) -> Iterator[Any]:
        """Iterate a DataLoader while measuring time blocked waiting for the next batch."""
        iterator = iter(loader)
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self._data_wait_ms += (time.perf_counter() - started) * 1000.0
            yield batch

    def log_scalars(self, step: int | None = None, loss: float | None = None, lr: float | None = None,
                    grad_norm: float | None = None, **extra: Any) -> None:
        """Attach host-side scalars (e.g. from the Trainer's log dict) to the current window."""
        if not self.enabled:
            return
        if loss is not None:
            if not math.isfinite(loss):
                self._window_nan += 1
                self.event("nan_loss", step=step, severity="error", loss=str(loss))
            else:
                self._window_scalars["loss"] = float(loss)
        if lr is not None and math.isfinite(lr):
            self._window_scalars["lr"] = float(lr)
        if grad_norm is not None:
            if math.isfinite(grad_norm):
                self._window_scalars["grad_norm"] = float(grad_norm)
            else:
                self._window_nan += 1
                self.event("nan_grad_norm", step=step, severity="error", grad_norm=str(grad_norm))
        if extra:
            self._window_scalars.setdefault("extra", {}).update(extra)

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time spent inside (eval, checkpoint, ...) is excluded from steady-state step time."""
        started = time.perf_counter()
        self._phase_stack.append(name)
        try:
            yield
        finally:
            self._phase_stack.pop()
            self.add_excluded_ms((time.perf_counter() - started) * 1000.0)

    def add_excluded_ms(self, milliseconds: float) -> None:
        self._excluded_ms += float(milliseconds)

    def _close_window(self) -> None:
        if self._window_steps == 0:
            return
        if self._ready is not None:
            self._flush_ready()
        self._ready = {
            "step": self._last_step, "ts": time.time(), "window_steps": self._window_steps,
            "wall_ms": (self._last_step_time - self._window_start) * 1000.0,
            "excluded_ms": self._excluded_ms, "durations": self._durations,
            "data_wait_ms": self._data_wait_ms, "events": self._gpu_events, "comm": self.comm_events,
            "processed": self._processed, "nonpad": self._nonpad, "supervised": self._supervised,
        }
        self._window_steps = 0
        self._window_start = self._last_step_time
        self._excluded_ms = 0.0
        self._excluded_at_step_end = 0.0
        self._durations = []
        self._data_wait_ms = 0.0
        self._gpu_events = {}
        self.comm_events = []

    @staticmethod
    def _sum_events(pairs: list[tuple[Any, Any]]) -> float | None:
        if not pairs:
            return None
        try:
            return float(sum(start.elapsed_time(end) for start, end in pairs))
        except Exception:  # noqa: BLE001 - never let a timing read affect training
            return None

    def _global_counts(self, local: list[int]) -> list[int]:
        """Sum token counters over all ranks (one all-reduce per window, only when torch.distributed
        is initialised and the run is healthy). Falls back to local counts otherwise."""
        torch = _torch()
        dist = _dist(torch) if torch is not None else None
        if dist is None or self.world_size <= 1 or self._failed:
            return local
        try:
            backend = str(dist.get_backend()).lower()
            device = torch.device("cuda", torch.cuda.current_device()) if "nccl" in backend or "rccl" in backend \
                else torch.device("cpu")
            tensor = torch.tensor(local, dtype=torch.int64, device=device)
            dist.all_reduce(tensor)
            return [int(v) for v in tensor.tolist()]
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"global token all_reduce failed: {exc}"
            return local

    def _flush_ready(self) -> None:
        """Convert the closed window. This is the single device->host read per window; it is
        deferred until the next microbatch so scalars logged for the same step can attach."""
        window, self._ready = self._ready, None
        if window is None:
            return
        started = time.perf_counter()
        nonpad_total = int(window["nonpad"])
        supervised_total = int(window["supervised"])
        processed_total = int(window["processed"])
        local = [processed_total - self._base_processed, nonpad_total - self._base_nonpad,
                 supervised_total - self._base_supervised, self._window_nan]
        self._base_nonpad, self._base_supervised, self._base_processed = nonpad_total, supervised_total, processed_total
        processed, nonpad, supervised, nan_count = self._global_counts(local)
        self._total_nonpad += nonpad
        active_ms = max(0.0, window["wall_ms"] - window["excluded_ms"])
        steps = window["window_steps"]
        durations = sorted(window["durations"])
        events = window["events"]
        gpu_step = self._sum_events(events.get("step", []))
        forward = self._sum_events(events.get("forward", []))
        optimizer = self._sum_events(events.get("optimizer", []))
        backward = None if gpu_step is None else max(0.0, gpu_step - (forward or 0.0) - (optimizer or 0.0))
        comm = self._sum_events(window["comm"])
        per_step = (lambda v: None if v is None or not steps else v / steps)
        phase = "warmup" if self._windows_sent < self.warmup_windows else "train"
        step_time_ms = active_ms / steps if steps else None
        local_nonpad = local[1]
        memory = self._memory_stats()
        if self.is_primary:
            record: dict[str, Any] = {
                "step": window["step"], "ts": window["ts"], "window_steps": steps, "phase": phase,
                "step_time_ms": step_time_ms, "excluded_ms": window["excluded_ms"],
                "processed_tokens": processed, "nonpad_tokens": nonpad, "supervised_tokens": supervised,
                "nan_count": nan_count, "data_wait_ms": per_step(window["data_wait_ms"] or None),
                "fwd_ms": per_step(forward), "bwd_ms": per_step(backward), "opt_ms": per_step(optimizer),
                "comm_ms": per_step(comm), **{k: v for k, v in self._window_scalars.items() if k != "extra"},
            }
            extra = dict(self._window_scalars.get("extra") or {})
            if durations:
                extra["step_time_median_ms"] = durations[len(durations) // 2]
                extra["step_time_max_ms"] = durations[-1]
            if self.world_size > 1:
                extra["world_size"] = self.world_size
            if extra:
                record["extra"] = extra
            record.update(memory)
            payload: dict[str, Any] = {"steps": [record]}
        else:
            payload = {"steps": []}
        if self.world_size > 1:
            payload["ranks"] = [{
                "step": window["step"], "rank": self.rank, "ts": window["ts"], "pci_bus": self.pci_bus,
                "window_steps": steps, "step_time_ms": step_time_ms,
                "data_wait_ms": per_step(window["data_wait_ms"] or None), "comm_ms": per_step(comm),
                "nonpad_tokens": local_nonpad, "mem_peak_bytes": memory.get("mem_peak_bytes"),
            }]
        self._window_scalars, self._window_nan = {}, 0
        self._windows_sent += 1
        self._enqueue("metrics", payload)
        self.enqueue_ms_total += (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _memory_stats() -> dict[str, Any]:
        torch = _torch()
        try:
            if not _cuda_ready(torch):
                return {}
            device = torch.cuda.current_device()
            stats = torch.cuda.memory_stats(device)
            return {
                "mem_alloc_bytes": int(torch.cuda.memory_allocated(device)),
                "mem_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "mem_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
                "alloc_retries": int(stats.get("num_alloc_retries", 0)),
            }
        except Exception:  # noqa: BLE001 - never let a stats read affect training
            return {}

    def event(self, kind: str, step: int | None = None, severity: str = "info", **payload: Any) -> None:
        if not self.enabled:
            return
        if self.world_size > 1:
            payload.setdefault("rank", self.rank)
        self._enqueue("events", {
            "eid": uuid.uuid4().hex, "kind": kind, "step": self._last_step if step is None else step,
            "severity": severity, "ts": time.time(), "payload": payload or None,
        })

    def checkpoint(self, step: int, path: str | Path, kind: str = "resume", block: bool = False) -> None:
        """Write the manifest (last) in a background thread, then register the checkpoint.
        Only rank 0 owns checkpoints."""
        if not self.enabled or not self.is_primary:
            return
        tokens_seen = self._total_nonpad
        extra = {"run_id": self.run_id, "world_size": self.world_size, "config_hash": self._config_hash()}

        def work() -> None:
            record: dict[str, Any] = {"step": step, "path": str(path), "kind": kind}
            try:
                manifest = write_manifest(path, step=step, kind=kind, tokens_seen=tokens_seen, extra=extra)
                ok, reason = validate_manifest(path)
                record.update(valid=ok, error=reason, size_bytes=manifest["total_bytes"],
                              save_ms=manifest["hash_ms"], manifest=manifest)
            except Exception as exc:  # noqa: BLE001
                record.update(valid=False, error=f"{type(exc).__name__}: {exc}")
                self.event("checkpoint_invalid", step=step, severity="error", path=str(path), error=str(exc))
            self._enqueue("checkpoints", record)

        thread = threading.Thread(target=work, name="checkpoint-manifest", daemon=True)
        thread.start()
        if block:
            thread.join()
        else:
            self._manifest_threads = [t for t in self._manifest_threads if t.is_alive()]
            self._manifest_threads.append(thread)

    def _config_hash(self) -> str | None:
        if self.config is None:
            return None
        import hashlib
        return hashlib.sha256(json.dumps(self.config, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def attach_artifact(self, path: str | Path, kind: str = "profile", **meta: Any) -> None:
        """Attach a file (e.g. a rocprofv3 trace) to the run. It is copied into the dashboard's
        artifact folder when that is reachable from this machine, so it becomes downloadable."""
        if not self.enabled or not self.is_primary:
            return
        source = Path(path)
        destination, downloadable = source, False
        if self.artifact_dir and source.exists():
            try:
                target = Path(self.artifact_dir)
                target.mkdir(parents=True, exist_ok=True)
                destination = target / source.name
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
                downloadable = destination.is_file()
            except OSError as exc:
                destination, downloadable = source, False
                self.last_error = f"artifact copy failed: {exc}"
        size = destination.stat().st_size if destination.is_file() else None
        self._enqueue("artifacts", {"kind": kind, "path": str(destination), "size_bytes": size,
                                    "downloadable": downloadable, "meta": meta or None})

    # ------------------------------------------------------------ transport

    def _enqueue(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((kind, payload))
        except queue.Full:
            self.dropped_records += 1

    def _url(self, kind: str) -> str:
        base = f"{self.server}/api/training/runs"
        return base if kind == "create" else f"{base}/{self.run_id}/{kind}"

    def _post(self, kind: str, payload: dict[str, Any]) -> str:
        """Returns 'ok', 'retry' (server unreachable / 5xx) or 'drop' (rejected, will never succeed)."""
        body = json.dumps(payload, default=str).encode("utf-8")
        request = urllib.request.Request(self._url(kind), data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                if kind == "create":
                    try:
                        reply = json.loads(response.read()) or {}
                        self.lease = reply.get("lease")
                        self.artifact_dir = reply.get("artifact_dir")
                    except ValueError:
                        pass
                return "ok"
        except urllib.error.HTTPError as exc:
            self._last_status = exc.code
            try:
                detail = json.loads(exc.read()).get("error", "")
            except (ValueError, OSError, AttributeError):
                detail = ""
            self.last_error = f"HTTP {exc.code} for {kind}" + (f": {detail}" if detail else "")
            return "retry" if exc.code >= 500 else "drop"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self.last_error = str(exc)
            return "retry"

    def _spool_append(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self.spool_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.spool_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": kind, "payload": payload}, default=str) + "\n")
            self._spool_pending = True
        except OSError as exc:
            self.dropped_records += 1
            self.last_error = f"spool write failed: {exc}"

    def _replay_spool(self) -> None:
        self._last_replay = time.monotonic()
        try:
            lines = self.spool_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            self._spool_pending = False
            return
        remaining: list[str] = []
        blocked = False
        for line in lines:
            if blocked:
                remaining.append(line)
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if self._post(record["kind"], record["payload"]) == "retry":
                blocked = True
                remaining.append(line)
        if remaining:
            self.spool_path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
        else:
            with contextlib.suppress(OSError):
                self.spool_path.unlink()
            self._spool_pending = False

    def _deliver(self, kind: str, payload: dict[str, Any], droppable: bool = False) -> None:
        if self._spool_pending:
            if not droppable:
                self._spool_append(kind, payload)
            return
        if self._post(kind, payload) == "retry" and not droppable:
            self._spool_append(kind, payload)

    def _heartbeat(self) -> dict[str, Any]:
        beat: dict[str, Any] = {"last_step": self._last_step}
        if self.world_size > 1:
            beat.update(rank=self.rank, pci_bus=self.pci_bus)
        return beat

    def _run(self) -> None:
        while True:
            items: list[tuple[str, dict[str, Any]]] = []
            try:
                items.append(self._queue.get(timeout=self.flush_s))
                while True:
                    items.append(self._queue.get_nowait())
            except queue.Empty:
                pass
            if self._spool_pending and time.monotonic() - self._last_replay >= 5.0:
                self._replay_spool()
            merged: list[tuple[str, dict[str, Any]]] = []
            for kind, payload in items:
                if kind == "metrics" and merged and merged[-1][0] == "metrics":
                    merged[-1][1]["steps"].extend(payload["steps"])
                    merged[-1][1].setdefault("ranks", []).extend(payload.get("ranks", []))
                else:
                    merged.append((kind, payload))
            for kind, payload in merged:
                if kind == "metrics":
                    payload["heartbeat"] = self._heartbeat()
                self._deliver(kind, payload)
                self._last_send = time.monotonic()
            if self._started and time.monotonic() - self._last_send >= self.heartbeat_s:
                self._deliver("metrics", {"steps": [], "heartbeat": self._heartbeat()}, droppable=True)
                self._last_send = time.monotonic()
            if self._stop.is_set() and self._queue.empty():
                if self._spool_pending:
                    self._replay_spool()
                return
