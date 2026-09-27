from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(slots=True)
class ProjectConfig:
    name: str = "dual-gpu"
    suite_prefix: str = "dualgpu"
    host: str = "127.0.0.1"
    log_dir: str = "./runs"
    cleanup_ports_on_start: bool = True
    start_timeout_seconds: int = 900


@dataclass(slots=True)
class DiscoveryConfig:
    lmstudio_home: str = str(Path.home() / ".lmstudio")
    models_dir: str = ""
    backend_runtime_overrides: dict[str, str] = field(default_factory=dict)
    backend_vendor_bin_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyConfig:
    vram_safety_fraction: float = 0.90
    ctx_min: int = 16384
    ctx_mid: int = 32768
    ctx_max: int = 49152
    ctx_mid_headroom_gb: float = 5.0
    ctx_max_headroom_gb: float = 10.0
    reasoning_mode: str = "off"
    reasoning_budget: int = 0
    cache_reuse: int = 256
    flash_attn: str = "auto"
    gpu_layers: int = 999
    parallel: int = 1
    no_mmproj: bool = True
    jinja: bool = True
    kill_busy_ports: bool = True
    health_poll_seconds: float = 1.0
    settle_seconds_after_restart: float = 2.0


@dataclass(slots=True)
class LaneConfig:
    key: str
    display: str
    match: str
    vram_gb: float
    port: int
    backend: str = "rocm"
    host: str = "127.0.0.1"
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelConfig:
    name: str
    path: str
    size_gb: float = 0.0
    ctx_size: int = 0
    parallel: int = 0
    reasoning_mode: str = ""
    reasoning_budget: int | None = None
    cache_reuse: int | None = None
    flash_attn: str = ""
    gpu_layers: int | None = None
    pin_lane: str = ""
    multi_gpu: bool = False
    tensor_split: str = ""
    extra_args: list[str] = field(default_factory=list)
    threads: int = 0
    ubatch_size: int = 0
    host: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskConfig:
    name: str
    command: str | list[str]
    cwd: str = ""
    shell: bool = False
    continue_on_error: bool = False
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig
    discovery: DiscoveryConfig
    policy: PolicyConfig
    lanes: list[LaneConfig]
    models: list[ModelConfig]
    tasks: list[TaskConfig]
    config_path: Path

    @property
    def base_dir(self) -> Path:
        return self.config_path.parent


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _dict_of_strings(data: dict[str, Any] | None) -> dict[str, str]:
    if not data:
        return {}
    return {str(key): str(value) for key, value in data.items()}


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    project_raw = raw.get("project", {})
    discovery_raw = raw.get("discovery", {})
    policy_raw = raw.get("policy", {})

    discovery = DiscoveryConfig(
        lmstudio_home=_expand(discovery_raw.get("lmstudio_home", DiscoveryConfig.lmstudio_home)),
        models_dir=_expand(discovery_raw.get("models_dir", "")),
        backend_runtime_overrides=_dict_of_strings(discovery_raw.get("backend_runtime_overrides")),
        backend_vendor_bin_overrides=_dict_of_strings(discovery_raw.get("backend_vendor_bin_overrides")),
    )
    project = ProjectConfig(
        name=str(project_raw.get("name", ProjectConfig.name)),
        suite_prefix=str(project_raw.get("suite_prefix", ProjectConfig.suite_prefix)),
        host=str(project_raw.get("host", ProjectConfig.host)),
        log_dir=str(project_raw.get("log_dir", ProjectConfig.log_dir)),
        cleanup_ports_on_start=bool(project_raw.get("cleanup_ports_on_start", ProjectConfig.cleanup_ports_on_start)),
        start_timeout_seconds=int(project_raw.get("start_timeout_seconds", ProjectConfig.start_timeout_seconds)),
    )
    policy = PolicyConfig(
        vram_safety_fraction=float(policy_raw.get("vram_safety_fraction", PolicyConfig.vram_safety_fraction)),
        ctx_min=int(policy_raw.get("ctx_min", PolicyConfig.ctx_min)),
        ctx_mid=int(policy_raw.get("ctx_mid", PolicyConfig.ctx_mid)),
        ctx_max=int(policy_raw.get("ctx_max", PolicyConfig.ctx_max)),
        ctx_mid_headroom_gb=float(policy_raw.get("ctx_mid_headroom_gb", PolicyConfig.ctx_mid_headroom_gb)),
        ctx_max_headroom_gb=float(policy_raw.get("ctx_max_headroom_gb", PolicyConfig.ctx_max_headroom_gb)),
        reasoning_mode=str(policy_raw.get("reasoning_mode", PolicyConfig.reasoning_mode)),
        reasoning_budget=int(policy_raw.get("reasoning_budget", PolicyConfig.reasoning_budget)),
        cache_reuse=int(policy_raw.get("cache_reuse", PolicyConfig.cache_reuse)),
        flash_attn=str(policy_raw.get("flash_attn", PolicyConfig.flash_attn)),
        gpu_layers=int(policy_raw.get("gpu_layers", PolicyConfig.gpu_layers)),
        parallel=int(policy_raw.get("parallel", PolicyConfig.parallel)),
        no_mmproj=bool(policy_raw.get("no_mmproj", PolicyConfig.no_mmproj)),
        jinja=bool(policy_raw.get("jinja", PolicyConfig.jinja)),
        kill_busy_ports=bool(policy_raw.get("kill_busy_ports", PolicyConfig.kill_busy_ports)),
        health_poll_seconds=float(policy_raw.get("health_poll_seconds", PolicyConfig.health_poll_seconds)),
        settle_seconds_after_restart=float(
            policy_raw.get("settle_seconds_after_restart", PolicyConfig.settle_seconds_after_restart)
        ),
    )

    lanes = [
        LaneConfig(
            key=str(item["key"]),
            display=str(item["display"]),
            match=str(item.get("match", item["display"])),
            vram_gb=float(item["vram_gb"]),
            port=int(item["port"]),
            backend=str(item.get("backend", "rocm")),
            host=str(item.get("host", project.host)),
            extra_args=[str(value) for value in item.get("extra_args", [])],
        )
        for item in raw.get("lanes", [])
    ]
    models = [
        ModelConfig(
            name=str(item["name"]),
            path=str(item["path"]),
            size_gb=float(item.get("size_gb", 0.0)),
            ctx_size=int(item.get("ctx_size", 0)),
            parallel=int(item.get("parallel", 0)),
            reasoning_mode=str(item.get("reasoning_mode", "")),
            reasoning_budget=(
                int(item["reasoning_budget"]) if "reasoning_budget" in item and item["reasoning_budget"] is not None else None
            ),
            cache_reuse=int(item["cache_reuse"]) if "cache_reuse" in item and item["cache_reuse"] is not None else None,
            flash_attn=str(item.get("flash_attn", "")),
            gpu_layers=int(item["gpu_layers"]) if "gpu_layers" in item and item["gpu_layers"] is not None else None,
            pin_lane=str(item.get("pin_lane", "")),
            multi_gpu=bool(item.get("multi_gpu", False)),
            tensor_split=str(item.get("tensor_split", "")),
            extra_args=[str(value) for value in item.get("extra_args", [])],
            threads=int(item.get("threads", 0)),
            ubatch_size=int(item.get("ubatch_size", 0)),
            host=str(item.get("host", "")),
            tags=[str(value) for value in item.get("tags", [])],
        )
        for item in raw.get("models", [])
    ]
    tasks = [
        TaskConfig(
            name=str(item["name"]),
            command=item["command"],
            cwd=str(item.get("cwd", "")),
            shell=bool(item.get("shell", False)),
            continue_on_error=bool(item.get("continue_on_error", False)),
            env={str(key): str(value) for key, value in item.get("env", {}).items()},
        )
        for item in raw.get("tasks", [])
    ]

    app = AppConfig(
        project=project,
        discovery=discovery,
        policy=policy,
        lanes=lanes,
        models=models,
        tasks=tasks,
        config_path=config_path,
    )
    validate_config(app)
    return app


def validate_config(config: AppConfig) -> None:
    if len(config.lanes) != 2:
        raise SystemExit("This setup expects exactly 2 lanes for a dual-GPU scheduler.")
    if not config.models:
        raise SystemExit("Config must define at least one model.")

    lane_keys = {lane.key for lane in config.lanes}
    if len(lane_keys) != len(config.lanes):
        raise SystemExit("Lane keys must be unique.")

    seen_ports: set[int] = set()
    for lane in config.lanes:
        if lane.port in seen_ports:
            raise SystemExit("Lane ports must be unique.")
        seen_ports.add(lane.port)
    for model in config.models:
        if model.pin_lane and model.pin_lane not in lane_keys:
            raise SystemExit(f"Model '{model.name}' pins unknown lane '{model.pin_lane}'.")
