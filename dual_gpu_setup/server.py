from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
import json
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from dual_gpu_setup.config import AppConfig, LaneConfig, ModelConfig
from dual_gpu_setup.lmstudio import backend_env, resolve_device, resolve_model_path, runtime_dir, server_binary

# Every helper process spawned in this module (netstat/tasklist/taskkill/powershell for
# metrics, the llama-server process itself) is launched without an inherited console --
# app.py normally runs detached (see launch_service.py), so a plain subprocess.run/Popen on
# Windows would otherwise allocate and briefly show a brand-new console window for EACH call.
# gpu_process_metrics_batch alone runs every ResourceSampler tick (every 2s, for the whole
# life of the process -- see app.py's ResourceSampler), so without this flag that's a new
# console window popping up every 2 seconds, forever.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def _pid_listening_on(port: int) -> int | None:
    try:
        output = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3].upper() == "LISTENING":
            if parts[1].rsplit(":", 1)[-1] == str(port):
                try:
                    return int(parts[4])
                except ValueError:
                    return None
    return None


def _is_llama_server(pid: int) -> bool:
    try:
        output = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "llama-server.exe" in output.lower()


def _wait_port_free(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_is_free(port):
            return True
        time.sleep(0.5)
    return False


def gpu_process_metrics_batch(pids: list[int]) -> dict[int, dict[str, float]]:
    """Read GPU dedicated/shared memory and utilization for several PIDs in one PowerShell
    invocation. Get-Counter's PDH counter-set resolution is the dominant cost here (multiple
    seconds cold), roughly halved by requesting all three counter paths in a single Get-Counter
    call instead of three, and only paid once per tick by collapsing N per-server calls (the
    resource sampler and every chat request's before/after capture) into one per sampling tick.

    Instance names look like 'pid_1234_luid_...' -- the filter anchors on the trailing
    underscore so pid 123 can't accidentally match an instance for pid 1234."""
    unique_pids = sorted(set(pids))
    if not unique_pids:
        return {}
    pid_array = ",".join(str(pid) for pid in unique_pids)
    script = (
        f"$targets=@({pid_array}); "
        "$samples=(Get-Counter -Counter @("
        "'\\GPU Process Memory(*)\\Dedicated Usage',"
        "'\\GPU Process Memory(*)\\Shared Usage',"
        "'\\GPU Engine(*)\\Utilization Percentage'"
        ") -EA SilentlyContinue).CounterSamples; "
        "$d=$samples | Where-Object {$_.Path -like '*dedicated usage'}; "
        "$s=$samples | Where-Object {$_.Path -like '*shared usage'}; "
        "$u=$samples | Where-Object {$_.Path -like '*utilization percentage'}; "
        "foreach ($targetPid in $targets) { "
        "$pattern=\"pid_${targetPid}_*\"; "
        "$dd=($d | Where-Object {$_.InstanceName -like $pattern} | Measure-Object CookedValue -Sum).Sum; "
        "$ss=($s | Where-Object {$_.InstanceName -like $pattern} | Measure-Object CookedValue -Sum).Sum; "
        "$uu=($u | Where-Object {$_.InstanceName -like $pattern} | Measure-Object CookedValue -Sum).Sum; "
        "Write-Output \"$targetPid $([double]$dd) $([double]$ss) $([double]$uu)\" }"
    )
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    results: dict[int, dict[str, float]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            pid, dedicated, shared, utilization = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        results[pid] = {
            "dedicated_gb": dedicated / 1024**3,
            "shared_gb": shared / 1024**3,
            "utilization_pct": max(0.0, min(100.0, utilization)),
        }
    return results


def gpu_process_metrics(pid: int) -> dict[str, float] | None:
    return gpu_process_metrics_batch([pid]).get(pid)


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
        pid = _pid_listening_on(port)
        if pid is None:
            return
        if not _is_llama_server(pid):
            raise RuntimeError(f"Port {port} is held by PID {pid}, which is not llama-server.exe.")
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, text=True, timeout=30,
                        creationflags=_NO_WINDOW)
        _wait_port_free(port, timeout=30)

    def build_command(self) -> list[str]:
        policy = self.config.policy
        args = [
            str(server_binary(self.config, self.backend)),
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
            str(self.model.parallel or policy.parallel),
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
        args.extend(self.lane_extra_args or [])
        args.extend(self.model.extra_args)
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
        self._log_handle.flush()
        self.proc = subprocess.Popen(
            args,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=backend_env(self.config, self.backend),
            cwd=str(runtime_dir(self.config, self.backend)),
            creationflags=_NO_WINDOW,
        )

        deadline = time.time() + self.config.project.start_timeout_seconds
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup with code {self.proc.returncode}: {self.log_path}")
            try:
                payload = _get_json(f"{self.base_url}/health", timeout=3)
                if payload.get("status") == "ok":
                    return
            except (urlerror.URLError, TimeoutError, ValueError, OSError):
                pass
            time.sleep(self.config.policy.health_poll_seconds)
        self.stop()
        raise RuntimeError(f"llama-server did not become healthy within {self.config.project.start_timeout_seconds}s: {self.log_path}")

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
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    creationflags=_NO_WINDOW,
                )
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        _wait_port_free(self.port, timeout=30)
        if self._log_handle and not self._log_handle.closed:
            self._log_handle.close()
