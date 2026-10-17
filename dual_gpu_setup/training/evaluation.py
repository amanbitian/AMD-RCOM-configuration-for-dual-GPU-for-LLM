"""Checkpoint evaluation: bounded queue, adapter compatibility, three-level pipeline.

Levels (see docs/TRAINING_TELEMETRY.md section 8):
  native      PyTorch + adapter                          -> training quality
  gguf_f16    LoRA converted to GGUF over the F16 base   -> conversion/engine differences
  gguf_quant  the same adapter over the quantized base   -> quantization loss (the deployed config)

A failed or unavailable level is recorded as `failed` with its reason; a level is never silently
skipped, so a gap in the scorecard always means something broke.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from dual_gpu_setup.config import EvaluationConfig
from dual_gpu_setup.gguf import GGUFParseError, read_metadata

from .checkpoints import read_manifest, sha256_file
from .db import Database
from .store import TrainingStore

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL,
    priority INTEGER NOT NULL,
    protected INTEGER NOT NULL DEFAULT 0,
    flags_json TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    UNIQUE(run_id, step)
);
CREATE INDEX IF NOT EXISTS idx_eval_jobs_state ON eval_jobs(state, priority DESC, created_at);
CREATE TABLE IF NOT EXISTS eval_counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""

PRIORITY = {"final": 100, "best": 80, "designated": 60, "latest": 40, "routine": 10}


# ----------------------------------------------------------------------- queue


class EvalQueue(Database):
    """Persistent, bounded, coalescing queue. Protected checkpoints (final, best candidate,
    designated every-Kth) are never dropped; routine ones are coalesced so that a fast trainer
    cannot bury the evaluator under stale checkpoints."""

    def __init__(self, path: Path, capacity: int = 16, clock: Callable[[], float] = time.time):
        super().__init__(path)
        self.capacity = capacity
        self.clock = clock
        self.ensure_schema(SCHEMA)
        self._lock = threading.Lock()

    @staticmethod
    def _priority(flags: dict[str, bool]) -> tuple[int, bool]:
        for name in ("final", "best", "designated"):
            if flags.get(name):
                return PRIORITY[name], True
        return (PRIORITY["latest"] if flags.get("latest") else PRIORITY["routine"]), False

    def submit(self, run_id: str, step: int, checkpoint_path: str, *, final: bool = False, best: bool = False,
               designated: bool = False) -> dict[str, Any]:
        flags = {"final": final, "best": best, "designated": designated, "latest": True}
        priority, protected = self._priority(flags)
        now = self.clock()
        dropped: list[int] = []
        with self._lock, self.connect() as connection:
            existing = connection.execute("SELECT * FROM eval_jobs WHERE run_id = ? AND step = ?", (run_id, step)).fetchone()
            if existing is not None:  # merge flags; a protected request upgrades a queued routine job
                merged = {**json.loads(existing["flags_json"]), **{k: v for k, v in flags.items() if v}}
                new_priority, new_protected = self._priority(merged)
                if existing["state"] == "queued":
                    connection.execute("UPDATE eval_jobs SET priority = ?, protected = ?, flags_json = ? WHERE id = ?",
                                       (new_priority, 1 if new_protected else 0, json.dumps(merged), existing["id"]))
                return {"accepted": True, "id": existing["id"], "duplicate": True, "dropped": []}
            if not protected:
                # Only the newest routine checkpoint per run is worth evaluating.
                stale = connection.execute(
                    "SELECT id FROM eval_jobs WHERE run_id = ? AND state = 'queued' AND protected = 0 AND step < ?",
                    (run_id, step)).fetchall()
                for row in stale:
                    dropped.append(row["id"])
                    connection.execute("DELETE FROM eval_jobs WHERE id = ?", (row["id"],))
            depth = connection.execute("SELECT COUNT(*) FROM eval_jobs WHERE state = 'queued'").fetchone()[0]
            while depth >= self.capacity:
                victim = connection.execute(
                    "SELECT id FROM eval_jobs WHERE state = 'queued' AND protected = 0 ORDER BY priority, created_at LIMIT 1"
                ).fetchone()
                if victim is None:
                    break
                dropped.append(victim["id"])
                connection.execute("DELETE FROM eval_jobs WHERE id = ?", (victim["id"],))
                depth -= 1
            if depth >= self.capacity and not protected:
                self._bump(connection, "rejected")
                return {"accepted": False, "reason": "queue is full of protected checkpoints", "dropped": dropped}
            if depth >= self.capacity * 2:  # hard cap even for protected work
                self._bump(connection, "rejected")
                return {"accepted": False, "reason": "queue is over its hard limit", "dropped": dropped}
            cursor = connection.execute(
                """INSERT INTO eval_jobs (run_id, step, checkpoint_path, priority, protected, flags_json, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (run_id, step, checkpoint_path, priority, 1 if protected else 0, json.dumps(flags), now),
            )
            if dropped:
                self._bump(connection, "dropped", len(dropped))
            return {"accepted": True, "id": cursor.lastrowid, "duplicate": False, "dropped": dropped}

    @staticmethod
    def _bump(connection: Any, name: str, amount: int = 1) -> None:
        connection.execute(
            "INSERT INTO eval_counters (name, value) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value = value + ?",
            (name, amount, amount))

    def claim(self) -> dict[str, Any] | None:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM eval_jobs WHERE state = 'queued' ORDER BY priority DESC, created_at LIMIT 1").fetchone()
            if row is None:
                return None
            connection.execute("UPDATE eval_jobs SET state = 'running', started_at = ?, attempts = attempts + 1 WHERE id = ?",
                               (self.clock(), row["id"]))
            return {**dict(row), "flags": json.loads(row["flags_json"]), "state": "running"}

    def finish(self, job_id: int, state: str, error: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE eval_jobs SET state = ?, error = ?, finished_at = ? WHERE id = ?",
                               (state, error, self.clock(), job_id))

    def requeue(self, job_id: int, reason: str) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE eval_jobs SET state = 'queued', error = ?, started_at = NULL WHERE id = ?", (reason, job_id))

    def recover_interrupted(self) -> int:
        """Jobs left `running` by a crashed worker go back to the queue."""
        with self.connect() as connection:
            return connection.execute("UPDATE eval_jobs SET state = 'queued', started_at = NULL WHERE state = 'running'").rowcount

    def stats(self, latest_training_step: int | None = None) -> dict[str, Any]:
        with self.connect() as connection:
            counts = {r["state"]: r["n"] for r in connection.execute("SELECT state, COUNT(*) AS n FROM eval_jobs GROUP BY state")}
            oldest = connection.execute(
                "SELECT MIN(created_at) AS t, MIN(step) AS s FROM eval_jobs WHERE state = 'queued'").fetchone()
            counters = {r["name"]: r["value"] for r in connection.execute("SELECT * FROM eval_counters")}
        now = self.clock()
        return {
            "capacity": self.capacity, "depth": counts.get("queued", 0), "running": counts.get("running", 0),
            "completed": counts.get("completed", 0), "failed": counts.get("failed", 0),
            "dropped_total": counters.get("dropped", 0), "rejected_total": counters.get("rejected", 0),
            "oldest_wait_s": None if oldest["t"] is None else max(0.0, now - oldest["t"]),
            "lag_steps": None if oldest["s"] is None or latest_training_step is None else max(0, latest_training_step - oldest["s"]),
        }

    def jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [{**dict(r), "flags": json.loads(r["flags_json"])} for r in connection.execute(
                "SELECT * FROM eval_jobs ORDER BY CASE state WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,"
                " priority DESC, created_at DESC LIMIT ?", (limit,))]


# ---------------------------------------------------------------- compatibility

SUPPORTED_LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                          "embed_tokens", "lm_head", "qkv_proj", "gate_up_proj", "wq", "wk", "wv", "wo",
                          "w1", "w2", "w3", "fc1", "fc2"}
_LAYER_RE = re.compile(r"\.layers?\.(\d+)\.")


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """The JSON header of a .safetensors file (tensor names/shapes) without loading any tensor."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("file too short to be safetensors")
        (length,) = struct.unpack("<Q", raw)
        if length > 100 * 1024 * 1024:
            raise ValueError("implausible safetensors header size")
        return json.loads(handle.read(length).decode("utf-8"))


@dataclass(slots=True)
class CompatReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    adapter: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "adapter": self.adapter}


def check_adapter_compat(adapter_dir: str | Path, *, base_gguf: str | Path | None = None,
                         expected: dict[str, Any] | None = None) -> CompatReport:
    """Validate base-model revision, tokenizer, LoRA target modules, layer count and converter
    version before spending GPU time. `expected` may hold base_model, revision, tokenizer_hash,
    converter_version and `converter_version_actual`."""
    expected = expected or {}
    adapter_dir = Path(adapter_dir)
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}
    config_path = adapter_dir / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return CompatReport(False, [f"{config_path} is missing or unreadable (not a PEFT adapter)."])
    if str(config.get("peft_type", "LORA")).upper() != "LORA":
        errors.append(f"peft_type {config.get('peft_type')!r} is not LORA; the converter only handles LoRA.")
    if config.get("use_dora"):
        errors.append("DoRA adapters are not supported by the llama.cpp LoRA converter.")
    if config.get("modules_to_save"):
        warnings.append(f"modules_to_save={config['modules_to_save']} are full-weight tensors the GGUF adapter may drop.")
    targets = config.get("target_modules") or []
    targets = [targets] if isinstance(targets, str) else list(targets)
    unsupported = sorted(t for t in targets if t.split(".")[-1] not in SUPPORTED_LORA_TARGETS)
    if unsupported:
        errors.append(f"Unsupported LoRA target modules for GGUF conversion: {', '.join(unsupported)}.")
    info.update(base_model=config.get("base_model_name_or_path"), revision=config.get("revision"),
                rank=config.get("r"), alpha=config.get("lora_alpha"), target_modules=targets)
    if expected.get("base_model") and config.get("base_model_name_or_path") not in (None, expected["base_model"]):
        errors.append(f"Adapter was trained on {config.get('base_model_name_or_path')!r}, expected {expected['base_model']!r}.")
    if expected.get("revision") and config.get("revision") not in (None, expected["revision"]):
        errors.append(f"Base revision {config.get('revision')!r} differs from expected {expected['revision']!r}.")
    weights = next((p for p in (adapter_dir / "adapter_model.safetensors",) if p.is_file()), None)
    if weights is None:
        errors.append("adapter_model.safetensors not found.")
    else:
        try:
            header = read_safetensors_header(weights)
            layers = {int(m.group(1)) for name in header if (m := _LAYER_RE.search(name))}
            info["adapter_layers"] = (max(layers) + 1) if layers else None
            info["tensor_count"] = len([k for k in header if k != "__metadata__"])
            if base_gguf:
                meta = read_metadata(Path(base_gguf), {"general.architecture"})
                arch = meta.get("general.architecture")
                blocks = read_metadata(Path(base_gguf), {f"{arch}.block_count"}).get(f"{arch}.block_count") if arch else None
                info["gguf_architecture"], info["gguf_block_count"] = arch, blocks
                if blocks and layers and max(layers) >= blocks:
                    errors.append(f"Adapter touches layer {max(layers)} but the base GGUF has only {blocks} layers.")
        except (OSError, ValueError, GGUFParseError) as exc:
            errors.append(f"Could not inspect adapter/base files: {exc}")
    tokenizer = next((adapter_dir / n for n in ("tokenizer.json", "tokenizer.model") if (adapter_dir / n).is_file()), None)
    if tokenizer is None:
        warnings.append("No tokenizer file saved with the adapter; chat template/vocab drift cannot be ruled out.")
    else:
        info["tokenizer_hash"] = sha256_file(tokenizer)[:16]
        if expected.get("tokenizer_hash") and info["tokenizer_hash"] != expected["tokenizer_hash"]:
            errors.append("Tokenizer differs from the one used for the baseline evaluation.")
    if expected.get("converter_version") and expected.get("converter_version_actual") \
            and expected["converter_version"] != expected["converter_version_actual"]:
        warnings.append(f"Converter version {expected['converter_version_actual']} differs from pinned {expected['converter_version']}.")
    return CompatReport(not errors, errors, warnings, info)


# ------------------------------------------------------------- server lifecycle


class ServerProcess(Protocol):
    base_url: str

    def start(self) -> None: ...

    def stop(self) -> None: ...


class AdapterServerManager:
    """Owns one llama-server on the evaluation lane. llama-server reads `--lora` only at startup,
    so a changed adapter (or base) means a restart; the complete changeover latency is returned."""

    def __init__(self, factory: Callable[[Path, Path], ServerProcess], clock: Callable[[], float] = time.perf_counter):
        self.factory = factory
        self.clock = clock
        self._process: ServerProcess | None = None
        self._fingerprint: tuple[str, str] | None = None

    def ensure(self, base_model: Path, adapter: Path) -> dict[str, Any]:
        fingerprint = (str(base_model), sha256_file(adapter))
        if self._process is not None and fingerprint == self._fingerprint:
            return {"base_url": self._process.base_url, "restarted": False, "changeover_ms": 0.0, "stop_ms": 0.0, "start_ms": 0.0}
        t0 = self.clock()
        if self._process is not None:
            self._process.stop()
            self._process = None
        t1 = self.clock()
        process = self.factory(base_model, adapter)
        try:
            process.start()
        except BaseException:
            self._fingerprint = None
            raise
        t2 = self.clock()
        self._process, self._fingerprint = process, fingerprint
        return {"base_url": process.base_url, "restarted": True, "stop_ms": (t1 - t0) * 1000.0,
                "start_ms": (t2 - t1) * 1000.0, "changeover_ms": (t2 - t0) * 1000.0}

    def close(self) -> None:
        if self._process is not None:
            try:
                self._process.stop()
            finally:
                self._process, self._fingerprint = None, None


# --------------------------------------------------------------------- pipeline

CommandRunner = Callable[[list[str], dict[str, str], float], "subprocess.CompletedProcess[str]"]


def _default_runner(command: list[str], env: dict[str, str], timeout: float) -> "subprocess.CompletedProcess[str]":
    import os

    return subprocess.run(command, env={**os.environ, **env}, capture_output=True, text=True, timeout=timeout)


def render_command(template: list[str], context: dict[str, str]) -> list[str]:
    return [part.format_map(context) for part in template]


class EvaluationPipeline:
    """Runs one checkpoint through the three levels and records every outcome in the store."""

    def __init__(self, settings: EvaluationConfig, store: TrainingStore, servers: AdapterServerManager,
                 runner: CommandRunner = _default_runner, workdir: Path | None = None):
        self.settings = settings
        self.store = store
        self.servers = servers
        self.runner = runner
        self.workdir = Path(workdir or settings.workdir or ".eval_work")

    def _expected(self, run: dict[str, Any]) -> dict[str, Any]:
        config = run.get("config") or {}
        base = self.settings.base_model or config.get("base_model") or config.get("model_name_or_path")
        return {"base_model": None if not base else base, "revision": config.get("revision"),
                "tokenizer_hash": (run.get("tags") or {}).get("tokenizer_hash"),
                "converter_version": self.settings.converter_version or None,
                "converter_version_actual": self.settings.converter_version or None}

    def _record(self, run_id: str, step: int, level: str, status: str, *, metrics: dict[str, Any] | None = None,
                latency: dict[str, Any] | None = None, versions: dict[str, Any] | None = None,
                error: str | None = None) -> None:
        self.store.record_eval(run_id, {"step": step, "level": level, "status": status, "metrics": metrics,
                                        "latency": latency, "versions": versions, "error": error})

    def _run_eval_command(self, template: list[str], context: dict[str, str], output: Path) -> dict[str, Any]:
        if not template:
            raise RuntimeError("no evaluation command is configured")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        started = time.perf_counter()
        result = self.runner(render_command(template, {**context, "output_json": str(output)}), {}, self.settings.timeout_s)
        if result.returncode != 0:
            raise RuntimeError(f"evaluation command exited {result.returncode}: {(result.stderr or result.stdout or '')[-500:]}")
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"evaluation command wrote no valid {output.name}: {exc}") from exc
        if not isinstance(payload.get("metrics"), dict):
            raise RuntimeError("evaluation output has no `metrics` object")
        payload.setdefault("latency", {})["eval_wall_ms"] = (time.perf_counter() - started) * 1000.0
        return payload

    def convert_adapter(self, adapter_dir: Path, out_file: Path) -> dict[str, Any]:
        script = self.settings.converter_script
        if not script:
            raise RuntimeError("evaluation.converter_script is not configured")
        command = [self.settings.converter_python or sys.executable, script, "--base", self.settings.base_model,
                   "--outfile", str(out_file), "--outtype", "f16", str(adapter_dir)]
        started = time.perf_counter()
        result = self.runner(command, {}, self.settings.timeout_s)
        if result.returncode != 0 or not out_file.is_file():
            raise RuntimeError(f"LoRA->GGUF conversion failed (exit {result.returncode}): {(result.stderr or result.stdout or '')[-500:]}")
        return {"conversion_ms": (time.perf_counter() - started) * 1000.0}

    def evaluate(self, job: dict[str, Any]) -> dict[str, Any]:
        """Returns {level: status}. Never raises for an evaluation failure: failures are recorded."""
        run_id, step = job["run_id"], job["step"]
        run = self.store.get_run(run_id)
        adapter_dir = Path(job["checkpoint_path"])
        manifest = read_manifest(adapter_dir) or {}
        work = self.workdir / run_id / f"step-{step}"
        work.mkdir(parents=True, exist_ok=True)
        versions = {"adapter_manifest_step": manifest.get("step"), "converter_version": self.settings.converter_version or None,
                    "base_f16_gguf": self.settings.base_f16_gguf, "base_quant_gguf": self.settings.base_quant_gguf}
        outcomes: dict[str, str] = {}

        compat = check_adapter_compat(adapter_dir, base_gguf=self.settings.base_f16_gguf or None, expected=self._expected(run))
        versions["adapter"] = compat.adapter
        if not compat.ok:
            reason = "adapter compatibility check failed: " + "; ".join(compat.errors)
            for level in ("native", "gguf_f16", "gguf_quant"):
                self._record(run_id, step, level, "failed", error=reason, versions=versions)
                outcomes[level] = "failed"
            return outcomes
        versions["compat_warnings"] = compat.warnings

        context = {"adapter_dir": str(adapter_dir), "base_model": self.settings.base_model, "parity_set": self.settings.parity_set,
                   "run_id": run_id, "step": str(step)}
        try:  # level 1: native
            payload = self._run_eval_command(self.settings.native_command, {**context, "level": "native", "base_url": "", "model_name": ""},
                                             work / "native.json")
            self._record(run_id, step, "native", "completed", metrics=payload["metrics"], latency=payload["latency"], versions=versions)
            outcomes["native"] = "completed"
        except Exception as exc:  # noqa: BLE001
            self._record(run_id, step, "native", "failed", error=str(exc), versions=versions)
            outcomes["native"] = "failed"

        adapter_gguf = work / "adapter.gguf"
        conversion: dict[str, Any] = {}
        try:
            conversion = self.convert_adapter(adapter_dir, adapter_gguf)
        except Exception as exc:  # noqa: BLE001
            for level in ("gguf_f16", "gguf_quant"):
                self._record(run_id, step, level, "failed", error=str(exc), versions=versions)
                outcomes[level] = "failed"
            return outcomes

        for level, base in (("gguf_f16", self.settings.base_f16_gguf), ("gguf_quant", self.settings.base_quant_gguf)):
            try:
                if not base:
                    raise RuntimeError(f"evaluation.base_{'f16' if level == 'gguf_f16' else 'quant'}_gguf is not configured")
                served = self.servers.ensure(Path(base), adapter_gguf)
                payload = self._run_eval_command(
                    self.settings.gguf_command,
                    {**context, "level": level, "base_url": served["base_url"], "model_name": Path(base).stem}, work / f"{level}.json")
                latency = {**payload["latency"], **conversion, "changeover_ms": served["changeover_ms"],
                           "server_restarted": served["restarted"]}
                self._record(run_id, step, level, "completed", metrics=payload["metrics"], latency=latency,
                             versions={**versions, "base_gguf": base})
                outcomes[level] = "completed"
            except Exception as exc:  # noqa: BLE001
                self._record(run_id, step, level, "failed", error=str(exc), versions=versions)
                outcomes[level] = "failed"
        shutil.rmtree(work, ignore_errors=True) if all(v == "completed" for v in outcomes.values()) else None
        return outcomes


class EvalWorker:
    """Background loop: claim the highest-priority job and run the pipeline on the evaluation lane.

    `lane_ready` lets the host refuse (job goes back to the queue) while inference occupies the lane;
    `on_done(job, outcomes)` lets the host update best-checkpoint selection."""

    def __init__(self, queue_: EvalQueue, pipeline: EvaluationPipeline, *, poll_s: float = 2.0,
                 lane_ready: Callable[[], bool] | None = None, on_done: Callable[[dict[str, Any], dict[str, str]], None] | None = None,
                 acquire_lane: Callable[[], Any] | None = None, release_lane: Callable[[Any], None] | None = None):
        self.queue = queue_
        self.pipeline = pipeline
        self.poll_s = poll_s
        self.lane_ready = lane_ready
        self.on_done = on_done
        self.acquire_lane = acquire_lane
        self.release_lane = release_lane
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def run_once(self) -> bool:
        if self.lane_ready is not None and not self.lane_ready():
            return False
        job = self.queue.claim()
        if job is None:
            return False
        lease = None
        try:
            if self.acquire_lane is not None:
                lease = self.acquire_lane()
            outcomes = self.pipeline.evaluate(job)
            state = "completed" if outcomes and all(v == "completed" for v in outcomes.values()) else "failed"
            self.queue.finish(job["id"], state, None if state == "completed" else "one or more levels failed; see the checkpoint scorecard")
            if self.on_done:
                self.on_done(job, outcomes)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.queue.finish(job["id"], "failed", self.last_error)
        finally:
            if lease is not None and self.release_lane is not None:
                self.release_lane(lease)
        return True

    def start(self) -> None:
        if self._thread is not None:
            return
        self.queue.recover_interrupted()

        def loop() -> None:
            while not self._stop.is_set():
                if not self.run_once():
                    self._stop.wait(self.poll_s)

        self._thread = threading.Thread(target=loop, name="eval-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.pipeline.servers.close()
