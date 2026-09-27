from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
import json
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from dual_gpu_setup.config import AppConfig, LaneConfig, ModelConfig
from dual_gpu_setup.lmstudio import backend_env, resolve_device, runtime_dir, server_binary


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


def gpu_process_metrics(pid: int) -> dict[str, float] | None:
    script = (
        "$d=(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -EA SilentlyContinue)"
        f".CounterSamples | Where-Object {{$_.InstanceName -like 'pid_{pid}*'}} | "
        "Measure-Object CookedValue -Sum; "
        "$s=(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -EA SilentlyContinue)"
        f".CounterSamples | Where-Object {{$_.InstanceName -like 'pid_{pid}*'}} | "
        "Measure-Object CookedValue -Sum; "
        "$u=(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -EA SilentlyContinue)"
        f".CounterSamples | Where-Object {{$_.InstanceName -like 'pid_{pid}*'}} | "
        "Measure-Object CookedValue -Sum; "
        "Write-Output \"$([double]$d.Sum) $([double]$s.Sum) $([double]$u.Sum)\""
    )
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    if len(output) < 3:
        return None
    try:
        return {
            "dedicated_gb": float(output[0]) / 1024**3,
            "shared_gb": float(output[1]) / 1024**3,
            "utilization_pct": max(0.0, min(100.0, float(output[2]))),
        }
    except ValueError:
        return None


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
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, text=True, timeout=30)
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
        args.extend(self.lane_extra_args or [])
        args.extend(self.model.extra_args)
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
                )
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        _wait_port_free(self.port, timeout=30)
        if self._log_handle and not self._log_handle.closed:
            self._log_handle.close()
