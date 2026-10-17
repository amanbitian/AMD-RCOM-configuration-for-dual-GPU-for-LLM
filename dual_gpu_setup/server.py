from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass, replace
import json
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from dual_gpu_setup.config import AppConfig, LaneConfig, ModelConfig
from dual_gpu_setup.hostos import get_host
from dual_gpu_setup.host_info import os_log_line
from dual_gpu_setup.lmstudio import (
    RuntimeInfo,
    backend_env,
    multi_gpu_config,
    resolve_device,
    resolve_model_path,
    runtime_dir,
    runtime_info,
    server_binary,
)

# Flags llama.cpp renamed upstream. A configured `extra_args` value is rewritten only when the
# running build lacks the old spelling and has the new one, so either build accepts the same config.
RENAMED_FLAGS: dict[str, list[str]] = {"--no-mmap": ["--load-mode", "none"]}


def compat_args(runtime: RuntimeInfo, args: list[str]) -> list[str]:
    out: list[str] = []
    for arg in args:
        replacement = RENAMED_FLAGS.get(arg)
        if replacement and runtime.flags and not runtime.supports(arg) and runtime.supports(replacement[0]):
            out.extend(replacement)
        else:
            out.append(arg)
    return out


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def _wait_port_free(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_is_free(port):
            return True
        time.sleep(0.5)
    return False


def gpu_process_metrics(pid: int) -> dict[str, float] | None:
    return get_host().gpu.process_metrics([pid]).get(pid)


def vram_placement(pid: int) -> tuple[float, float] | None:
    metrics = gpu_process_metrics(pid)
    if metrics is None:
        return None
    return metrics["dedicated_gb"], metrics["shared_gb"]


def warn_if_spilling(pid: int, model_name: str, lane_name: str) -> tuple[float, float] | None:
    placement = None
    for _ in range(4):
        time.sleep(4)
        placement = vram_placement(pid)
        if placement and placement[0] > 0.1:
            break
    if not placement or placement[0] <= 0.1:
        print(f"[{lane_name}] could not read GPU memory counters for {model_name}")
        return None
    dedicated, shared = placement
    if shared > 1.0 and shared > dedicated * 0.15:
        print(
            f"[{lane_name}] WARNING: {model_name} is spilling to system RAM "
            f"({dedicated:.1f} GB VRAM, {shared:.1f} GB shared)"
        )
    else:
        print(f"[{lane_name}] VRAM {dedicated:.1f} GB dedicated, {shared:.1f} GB shared")
    return placement


def _get_json(url: str, timeout: float) -> dict:
    with urlrequest.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body)


# A GPU-offloaded server must hold at least this share of its weight file in VRAM. llama-server
# still reports healthy when it silently falls back to CPU (seen under host memory pressure), so
# health alone cannot prove placement.
PLACEMENT_MIN_VRAM_FRACTION = 0.5


def placement_error(dedicated_gb: float, model_bytes: int, device: str) -> str | None:
    """Return a failure message when a GPU-offloaded server is not resident on its device."""
    expected_gb = model_bytes * PLACEMENT_MIN_VRAM_FRACTION / 1024**3
    if dedicated_gb >= expected_gb:
        return None
    return (
        f"placement check failed on {device}: process holds {dedicated_gb:.2f} GB in VRAM, "
        f"expected at least {expected_gb:.2f} GB for a {model_bytes / 1024**3:.2f} GB model "
        "(weights likely fell back to CPU)"
    )


@dataclass(slots=True)
class LlamaServerProcess:
    config: AppConfig
    backend: str
    device: str
    port: int
    model: ModelConfig
    model_path: Path
    ctx_size: int
    run_dir: Path
    host: str = "127.0.0.1"
    tensor_split: str = ""
    lane_extra_args: list[str] | None = None
    proc: subprocess.Popen | None = None
    log_path: Path | None = None
    _log_handle: object | None = None
    _runtime: RuntimeInfo | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def pid(self) -> int:
        if self.proc is None:
            raise RuntimeError("Server has not started.")
        return self.proc.pid

    @classmethod
    def for_single_lane(
        cls,
        config: AppConfig,
        lane: LaneConfig,
        model: ModelConfig,
        model_path: Path,
        ctx_size: int,
        run_dir: Path,
    ) -> "LlamaServerProcess":
        return cls(
            config=config,
            backend=lane.backend,
            device=resolve_device(config, lane),
            port=lane.port,
            model=model,
            model_path=model_path,
            ctx_size=ctx_size,
            host=model.host or lane.host or config.project.host,
            lane_extra_args=list(lane.extra_args),
            run_dir=run_dir,
        )

    @classmethod
    def for_multi_gpu(
        cls,
        config: AppConfig,
        lanes: list[LaneConfig],
        model: ModelConfig,
        model_path: Path,
        ctx_size: int,
        run_dir: Path,
    ) -> "LlamaServerProcess":
        config = multi_gpu_config(config)
        ordered = sorted(lanes, key=lambda item: item.vram_gb, reverse=True)
        backend = ordered[0].backend
        mismatched = [lane.key for lane in ordered if lane.backend != backend]
        if mismatched:
            raise RuntimeError(
                "Multi-GPU models require the same backend on every lane. "
                f"Expected '{backend}', found mismatches on: {', '.join(mismatched)}"
            )
        device = ",".join(resolve_device(config, lane) for lane in ordered)
        return cls(
            config=config,
            backend=backend,
            device=device,
            port=ordered[0].port,
            model=model,
            model_path=model_path,
            ctx_size=ctx_size,
            host=model.host or ordered[0].host or config.project.host,
            tensor_split=model.tensor_split,
            lane_extra_args=[],
            run_dir=run_dir,
        )

    @staticmethod
    def kill_orphan_on_port(port: int) -> None:
        control = get_host().control
        pid = control.pid_listening_on(port)
        if pid is None:
            return
        if not control.is_llama_server(pid):
            raise RuntimeError(f"Port {port} is held by PID {pid}, which is not {control.server_binary_name}.")
        control.terminate_tree(pid)
        _wait_port_free(port, timeout=30)

    def runtime(self) -> RuntimeInfo:
        """The llama-server build this process runs (probed once, then reused)."""
        if self._runtime is None:
            self._runtime = runtime_info(self.config, self.backend, server_binary(self.config, self.backend))
        return self._runtime

    def warm_slot(self) -> int | None:
        """Slot id reserved for residency warm-ups, or None.

        With one serving slot, the server runs a second slot over a unified KV pool (no extra KV
        VRAM; each slot keeps the full context). Real requests are pinned to slot 0, so a warm-up
        on slot 1 can never overwrite a conversation's cached prompt (measured: a warm-up on the
        only slot forced the next turn to reprocess 2,510 tokens; on slot 1, 10 tokens)."""
        policy = self.config.policy
        explicit = {"-np", "--parallel", "-kvu", "--kv-unified", "-no-kvu", "--no-kv-unified"}
        if (not policy.residency_guard or (self.model.parallel or policy.parallel) != 1
                or explicit & set([*(self.lane_extra_args or []), *self.model.extra_args])
                or not self.runtime().supports("--kv-unified")):
            return None
        return 1

    def build_command(self) -> list[str]:
        policy = self.config.policy
        runtime = self.runtime()
        warm_slot = self.warm_slot()
        args = [
            runtime.binary,
            "--model",
            str(self.model_path),
            "--alias",
            self.model.name,
            "--device",
            self.device,
            "--gpu-layers",
            str(self.model.gpu_layers if self.model.gpu_layers is not None else policy.gpu_layers),
            "--ctx-size",
            str(self.ctx_size),
            "--parallel",
            str(2 if warm_slot is not None else self.model.parallel or policy.parallel),
            *(["--kv-unified"] if warm_slot is not None else []),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--flash-attn",
            self.model.flash_attn or policy.flash_attn,
            "--cache-reuse",
            str(self.model.cache_reuse if self.model.cache_reuse is not None else policy.cache_reuse),
            "--reasoning-format",
            "deepseek",
            "--reasoning",
            self.model.reasoning_mode or policy.reasoning_mode,
            "--reasoning-budget",
            str(self.model.reasoning_budget if self.model.reasoning_budget is not None else policy.reasoning_budget),
        ]
        if policy.no_mmproj:
            args.append("--no-mmproj")
        if policy.jinja:
            args.append("--jinja")
        if self.tensor_split:
            args.extend(["--tensor-split", self.tensor_split])
        if self.model.ubatch_size:
            args.extend(["--ubatch-size", str(self.model.ubatch_size)])
        if self.model.threads:
            args.extend(["--threads", str(self.model.threads), "--threads-batch", str(self.model.threads)])
        args.extend(self._draft_args(policy))
        extra = compat_args(runtime, [*(self.lane_extra_args or []), *self.model.extra_args])
        if runtime.supports("--fit") and not {"--fit", "-fit"} & set(extra):
            # Newer builds re-size any setting left unset to fit free memory. This repo sizes
            # context and offload itself (capacity check + placement verification), so keep it off.
            args.extend(["--fit", "off"])
        args.extend(self._host_memory_args(runtime, extra))
        args.extend(extra)
        return args

    def _host_memory_args(self, runtime: RuntimeInfo, extra: list[str]) -> list[str]:
        """Keep host RAM to what llama.cpp needs: no resident copy of the weights, and the
        prompt cache capped (measured: default flags hold ~1x the model in RAM after load and
        grow ~4 GB per 20 distinct prompts; these hold ~0.6 GB flat at the same decode speed).
        Flags a model's extra_args already set win; builds that lack a flag get nothing."""
        policy = self.config.policy
        args: list[str] = []
        if policy.load_mode != "auto" and not {"--load-mode", "-lm", "--no-mmap", "--mmap"} & set(extra):
            if runtime.supports("--load-mode"):
                args.extend(["--load-mode", policy.load_mode])
            elif policy.load_mode == "none" and runtime.supports("--no-mmap"):
                args.append("--no-mmap")
        if runtime.supports("--cache-ram") and not {"--cache-ram", "-cram"} & set(extra):
            args.extend(["--cache-ram", str(policy.cache_ram_mib)])
        return args

    def _draft_args(self, policy: "PolicyConfig") -> list[str]:
        # Speculative decoding: run a small draft model on the same lane(s) as the
        # target. It is lossless -- the target verifies every proposed token -- so it
        # only changes throughput. Omitted knobs fall back to llama-server defaults.
        if not self.model.draft_model:
            return []
        draft_path = resolve_model_path(self.config, replace(self.model, path=self.model.draft_model))
        draft_layers = self.model.draft_gpu_layers if self.model.draft_gpu_layers is not None else policy.gpu_layers
        args = [
            "--model-draft", str(draft_path),
            "--gpu-layers-draft", str(draft_layers),
            "--device-draft", self.device,
        ]
        if self.model.draft_max:
            args.extend(["--draft-max", str(self.model.draft_max)])
        if self.model.draft_min:
            args.extend(["--draft-min", str(self.model.draft_min)])
        if self.model.draft_p_min > 0:
            args.extend(["--draft-p-min", str(self.model.draft_p_min)])
        return args

    def start(self) -> None:
        if self.config.policy.kill_busy_ports:
            self.kill_orphan_on_port(self.port)
        elif not _port_is_free(self.port):
            raise RuntimeError(f"Port {self.port} is busy.")

        log_dir = self.run_dir / "server_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.model.name.replace(" ", "_").replace("/", "_")
        self.log_path = log_dir / f"{safe_name}.server.log"
        args = self.build_command()
        self._log_handle = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log_handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(args)} ===\n")
        self._log_handle.write(f"host_os={os_log_line()}\n")
        self._log_handle.flush()
        self.proc = subprocess.Popen(
            args,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=backend_env(self.config, self.backend),
            cwd=str(runtime_dir(self.config, self.backend)),
            **get_host().control.spawn_kwargs(),
        )

        deadline = time.time() + self.config.project.start_timeout_seconds
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = self._log_tail()
                raise RuntimeError(
                    f"llama-server exited during startup with code {self.proc.returncode}"
                    f"{f' ({tail})' if tail else ''}: {self.log_path}"
                )
            try:
                payload = _get_json(f"{self.base_url}/health", timeout=3)
                if payload.get("status") == "ok":
                    self._verify_placement()
                    return
            except (urlerror.URLError, TimeoutError, ValueError, OSError):
                pass
            time.sleep(self.config.policy.health_poll_seconds)
        self.stop()
        raise RuntimeError(f"llama-server did not become healthy within {self.config.project.start_timeout_seconds}s: {self.log_path}")

    def _log_tail(self, lines: int = 2) -> str:
        """Last lines this launch wrote, e.g. `error: invalid argument: --foo`, for the start error."""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        launch = text.rsplit("\n=== ", 1)[-1].splitlines()[1:]  # drop this launch's command header
        output = [line.strip() for line in launch if line.strip() and not line.startswith("host_os=")]
        return " | ".join(output[-lines:])[-400:]

    def _verify_placement(self) -> None:
        """Fail the start (and stop the server) unless the weights are resident on the target GPU."""
        policy_layers = self.model.gpu_layers if self.model.gpu_layers is not None else self.config.policy.gpu_layers
        if policy_layers <= 0:
            return
        try:
            metrics = get_host().gpu.process_metrics([self.proc.pid]).get(self.proc.pid, {})
        except Exception as exc:  # fail closed: an unverifiable placement is not a ready lane
            self.stop()
            raise RuntimeError(f"placement check could not read GPU memory for {self.model.name}: {exc}") from exc
        error = placement_error(float(metrics.get("dedicated_gb", 0.0)), self.model_path.stat().st_size, self.device)
        if error:
            self.stop()
            raise RuntimeError(f"{self.model.name}: {error}: {self.log_path}")

    def actual_ctx(self) -> int:
        try:
            payload = _get_json(f"{self.base_url}/v1/models", timeout=5)
            meta = payload["data"][0].get("meta", {})
            n_ctx = int(meta.get("n_ctx") or 0)
            n_ctx_train = int(meta.get("n_ctx_train") or 0)
            if n_ctx and n_ctx_train:
                return min(n_ctx, n_ctx_train)
            return n_ctx or n_ctx_train or self.ctx_size
        except Exception:  # noqa: BLE001
            return self.ctx_size

    def restart(self) -> None:
        self.stop()
        time.sleep(self.config.policy.settle_seconds_after_restart)
        self.start()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                get_host().control.terminate_tree(self.proc.pid)
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        _wait_port_free(self.port, timeout=30)
        if self._log_handle and not self._log_handle.closed:
            self._log_handle.close()
