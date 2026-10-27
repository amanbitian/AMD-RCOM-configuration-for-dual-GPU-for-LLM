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
    # Full-conversation archive (opt-in: it retains complete prompts and responses). When on,
    # every request through /api/chat and the OpenAI gateway is logged -- content plus token
    # counts -- to a SEPARATE SQLite file, written off the request path. Empty path = default
    # next to the metrics DB. See CODING_AGENTS.md "Conversation archive".
    store_chat_history: bool = False
    chat_history_path: str = ""
    # Watertight, zero-config attribution: a client needs no API key or pre-registration. It
    # names itself via its Bearer key or an X-DGPU-Project header and the project is created on
    # first use; anything unnamed is tracked under default_project so nothing is ever untracked.
    default_project: str = "unassigned"


@dataclass(slots=True)
class DiscoveryConfig:
    lmstudio_home: str = str(Path.home() / ".lmstudio")
    models_dir: str = ""
    backend_runtime_overrides: dict[str, str] = field(default_factory=dict)
    backend_vendor_bin_overrides: dict[str, str] = field(default_factory=dict)
    # Explicit llama-server binary per backend (official llama.cpp releases or native Linux builds;
    # no LM Studio required).
    backend_server_binaries: dict[str, str] = field(default_factory=dict)
    # Which runtime serves a model split across lanes: "configured" (the binaries above) or
    # "lmstudio" (LM Studio's bundled runtime), for a configured build that splits incorrectly.
    multi_gpu_runtime: str = "configured"


@dataclass(slots=True)
class EvaluationConfig:
    """Checkpoint evaluation on a spare GPU lane (see docs/TRAINING_TELEMETRY.md section 8)."""

    enabled: bool = False
    lane: str = ""
    base_model: str = ""  # HF directory or id of the base model (LoRA converter input)
    base_f16_gguf: str = ""
    base_quant_gguf: str = ""
    converter_script: str = ""  # llama.cpp convert_lora_to_gguf.py
    converter_python: str = ""
    converter_version: str = ""
    native_command: list[str] = field(default_factory=list)
    gguf_command: list[str] = field(default_factory=list)
    parity_set: str = ""
    workdir: str = ""
    ctx_size: int = 8192
    timeout_s: int = 3600
    queue_capacity: int = 16
    designate_every: int = 0  # every Kth checkpoint is protected from queue coalescing (0 = none)
    primary_metric: str = "task_score"
    higher_is_better: bool = True
    max_native_to_f16_drop: float = 0.02
    max_f16_to_quant_drop: float = 0.03
    max_latency_regression_pct: float = 20.0
    peak_tflops: dict[str, float] = field(default_factory=dict)  # per lane key, for MFU estimates


@dataclass(slots=True)
class PolicyConfig:
    vram_safety_fraction: float = 0.90
    ctx_min: int = 16384
    ctx_mid: int = 32768
    ctx_max: int = 49152
    ctx_mid_headroom_gb: float = 5.0
    ctx_max_headroom_gb: float = 10.0
    safety_tokens: int = 256
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
    # How long to let a lane's in-flight requests finish before stopping its server during a
    # deployment change, so a hot-swap does not kill another client's request mid-generation.
    drain_timeout_seconds: float = 30.0
    # Host RAM. "none" reads the weights without memory-mapping the file, so the copy the GPU
    # driver keeps of uploaded weights is committed but not resident; "mmap"/"auto" keep a
    # resident copy the size of the model. cache_ram_mib caps llama.cpp's prompt cache in RAM
    # (0 = off; llama.cpp's own default is 8192 MiB per server and grows with every new prompt).
    load_mode: str = "none"
    cache_ram_mib: int = 0
    # GPU-only placement: refuse to start a server unless its weights + KV cache + headroom fit
    # the VRAM the GPU reports free, instead of letting the driver spill it into system RAM.
    vram_admission: bool = True
    vram_headroom_gb: float = 1.0
    # A gateway request naming a known model that is not loaded loads it, evicting the least
    # recently used idle model(s) when the GPUs are full.
    auto_load_models: bool = True
    # Windows can move an idle server's VRAM into system RAM; its next request then pages it
    # back (measured 10 s for 16.85 GB). Every residency_check_seconds the guard finds idle
    # servers no longer resident and pages them back with a 1-token request on a reserved slot.
    residency_guard: bool = True
    residency_check_seconds: float = 15.0


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
    pci_bus: str = ""


@dataclass(slots=True)
class ModelConfig:
    name: str
    path: str
    size_gb: float = 0.0
    ctx_size: int = 0
    parallel: int = 0
    reasoning_mode: str = ""
    reasoning_budget: int | None = None
    reasoning_effort: str = "default"
    # Speculative decoding (lossless): a small draft model proposes tokens that the
    # target model verifies, raising tokens/sec without changing the output. draft_model
    # is a path/glob resolved like `path`; empty disables it. The remaining knobs map to
    # llama-server --draft-max/--draft-min/--draft-p-min/--gpu-layers-draft (0/None = let
    # the backend default decide). See CODING_AGENTS.md.
    draft_model: str = ""
    draft_max: int = 0
    draft_min: int = 0
    draft_p_min: float = 0.0
    draft_gpu_layers: int | None = None
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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def base_dir(self) -> Path:
        return self.config_path.parent


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _config_path(value: Any, base_dir: Path) -> str:
    text = _expand(str(value))
    if not text:
        return ""
    path = Path(text)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _dict_of_paths(data: dict[str, Any] | None, base_dir: Path) -> dict[str, str]:
    if not data:
        return {}
    return {str(key): _config_path(value, base_dir) for key, value in data.items()}


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    project_raw = raw.get("project", {})
    discovery_raw = raw.get("discovery", {})
    policy_raw = raw.get("policy", {})

    # NOTE: defaults below are read from *instances*, not the classes themselves.
    # `@dataclass(slots=True)` replaces a class's own attribute with a slot descriptor,
    # so `PolicyConfig.reasoning_budget` (class access) is not the default value `0` --
    # it's a `member_descriptor` object. `PolicyConfig().reasoning_budget` (instance
    # access) is. Using the class directly here would crash with a confusing TypeError
    # the moment a config omits a field instead of falling back to its real default.
    project_defaults = ProjectConfig()
    discovery_defaults = DiscoveryConfig()
    policy_defaults = PolicyConfig()

    discovery = DiscoveryConfig(
        lmstudio_home=_config_path(
            discovery_raw.get("lmstudio_home", discovery_defaults.lmstudio_home), config_path.parent
        ),
        models_dir=_config_path(discovery_raw.get("models_dir", ""), config_path.parent),
        backend_runtime_overrides=_dict_of_paths(
            discovery_raw.get("backend_runtime_overrides"), config_path.parent
        ),
        backend_vendor_bin_overrides=_dict_of_paths(
            discovery_raw.get("backend_vendor_bin_overrides"), config_path.parent
        ),
        backend_server_binaries=_dict_of_paths(
            discovery_raw.get("backend_server_binaries"), config_path.parent
        ),
        multi_gpu_runtime=str(discovery_raw.get("multi_gpu_runtime", discovery_defaults.multi_gpu_runtime)),
    )
    if discovery.multi_gpu_runtime not in {"configured", "lmstudio"}:
        raise SystemExit(
            f"discovery.multi_gpu_runtime must be 'configured' or 'lmstudio', got {discovery.multi_gpu_runtime!r}"
        )
    project = ProjectConfig(
        name=str(project_raw.get("name", project_defaults.name)),
        suite_prefix=str(project_raw.get("suite_prefix", project_defaults.suite_prefix)),
        host=str(project_raw.get("host", project_defaults.host)),
        log_dir=str(project_raw.get("log_dir", project_defaults.log_dir)),
        cleanup_ports_on_start=bool(project_raw.get("cleanup_ports_on_start", project_defaults.cleanup_ports_on_start)),
        start_timeout_seconds=int(project_raw.get("start_timeout_seconds", project_defaults.start_timeout_seconds)),
        store_chat_history=bool(project_raw.get("store_chat_history", project_defaults.store_chat_history)),
        chat_history_path=_expand(str(project_raw.get("chat_history_path", ""))),
        default_project=str(project_raw.get("default_project", project_defaults.default_project)) or "unassigned",
    )
    policy = PolicyConfig(
        vram_safety_fraction=float(policy_raw.get("vram_safety_fraction", policy_defaults.vram_safety_fraction)),
        ctx_min=int(policy_raw.get("ctx_min", policy_defaults.ctx_min)),
        ctx_mid=int(policy_raw.get("ctx_mid", policy_defaults.ctx_mid)),
        ctx_max=int(policy_raw.get("ctx_max", policy_defaults.ctx_max)),
        ctx_mid_headroom_gb=float(policy_raw.get("ctx_mid_headroom_gb", policy_defaults.ctx_mid_headroom_gb)),
        ctx_max_headroom_gb=float(policy_raw.get("ctx_max_headroom_gb", policy_defaults.ctx_max_headroom_gb)),
        safety_tokens=int(policy_raw.get("safety_tokens", policy_defaults.safety_tokens)),
        reasoning_mode=str(policy_raw.get("reasoning_mode", policy_defaults.reasoning_mode)),
        reasoning_budget=int(policy_raw.get("reasoning_budget", policy_defaults.reasoning_budget)),
        cache_reuse=int(policy_raw.get("cache_reuse", policy_defaults.cache_reuse)),
        flash_attn=str(policy_raw.get("flash_attn", policy_defaults.flash_attn)),
        gpu_layers=int(policy_raw.get("gpu_layers", policy_defaults.gpu_layers)),
        parallel=int(policy_raw.get("parallel", policy_defaults.parallel)),
        no_mmproj=bool(policy_raw.get("no_mmproj", policy_defaults.no_mmproj)),
        jinja=bool(policy_raw.get("jinja", policy_defaults.jinja)),
        kill_busy_ports=bool(policy_raw.get("kill_busy_ports", policy_defaults.kill_busy_ports)),
        health_poll_seconds=float(policy_raw.get("health_poll_seconds", policy_defaults.health_poll_seconds)),
        settle_seconds_after_restart=float(
            policy_raw.get("settle_seconds_after_restart", policy_defaults.settle_seconds_after_restart)
        ),
        drain_timeout_seconds=float(
            policy_raw.get("drain_timeout_seconds", policy_defaults.drain_timeout_seconds)
        ),
        load_mode=str(policy_raw.get("load_mode", policy_defaults.load_mode)),
        cache_ram_mib=int(policy_raw.get("cache_ram_mib", policy_defaults.cache_ram_mib)),
        vram_admission=bool(policy_raw.get("vram_admission", policy_defaults.vram_admission)),
        vram_headroom_gb=float(policy_raw.get("vram_headroom_gb", policy_defaults.vram_headroom_gb)),
        auto_load_models=bool(policy_raw.get("auto_load_models", policy_defaults.auto_load_models)),
        residency_guard=bool(policy_raw.get("residency_guard", policy_defaults.residency_guard)),
        residency_check_seconds=float(policy_raw.get("residency_check_seconds", policy_defaults.residency_check_seconds)),
    )
    if policy.load_mode not in {"none", "mmap", "auto"}:
        raise SystemExit(f"policy.load_mode must be 'none', 'mmap' or 'auto', got {policy.load_mode!r}")

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
            pci_bus=str(item.get("pci_bus", "")).strip().lower(),
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
            reasoning_effort=str(item.get("reasoning_effort", "default")),
            reasoning_budget=(
                int(item["reasoning_budget"]) if "reasoning_budget" in item and item["reasoning_budget"] is not None else None
            ),
            draft_model=_expand(str(item.get("draft_model", ""))),
            draft_max=int(item.get("draft_max", 0)),
            draft_min=int(item.get("draft_min", 0)),
            draft_p_min=float(item.get("draft_p_min", 0.0)),
            draft_gpu_layers=(
                int(item["draft_gpu_layers"]) if "draft_gpu_layers" in item and item["draft_gpu_layers"] is not None else None
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
        evaluation=_load_evaluation(raw.get("evaluation", {})),
    )
    validate_config(app)
    return app


def _load_evaluation(raw: dict[str, Any]) -> EvaluationConfig:
    defaults = EvaluationConfig()

    def command(key: str) -> list[str]:
        value = raw.get(key, [])
        return [str(part) for part in value] if isinstance(value, list) else [str(value)]

    return EvaluationConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        lane=str(raw.get("lane", "")),
        base_model=_expand(str(raw.get("base_model", ""))),
        base_f16_gguf=_expand(str(raw.get("base_f16_gguf", ""))),
        base_quant_gguf=_expand(str(raw.get("base_quant_gguf", ""))),
        converter_script=_expand(str(raw.get("converter_script", ""))),
        converter_python=_expand(str(raw.get("converter_python", ""))),
        converter_version=str(raw.get("converter_version", "")),
        native_command=command("native_command"),
        gguf_command=command("gguf_command"),
        parity_set=_expand(str(raw.get("parity_set", ""))),
        workdir=_expand(str(raw.get("workdir", ""))),
        ctx_size=int(raw.get("ctx_size", defaults.ctx_size)),
        timeout_s=int(raw.get("timeout_s", defaults.timeout_s)),
        queue_capacity=int(raw.get("queue_capacity", defaults.queue_capacity)),
        designate_every=int(raw.get("designate_every", defaults.designate_every)),
        primary_metric=str(raw.get("primary_metric", defaults.primary_metric)),
        higher_is_better=bool(raw.get("higher_is_better", defaults.higher_is_better)),
        max_native_to_f16_drop=float(raw.get("max_native_to_f16_drop", defaults.max_native_to_f16_drop)),
        max_f16_to_quant_drop=float(raw.get("max_f16_to_quant_drop", defaults.max_f16_to_quant_drop)),
        max_latency_regression_pct=float(raw.get("max_latency_regression_pct", defaults.max_latency_regression_pct)),
        peak_tflops={str(k): float(v) for k, v in dict(raw.get("peak_tflops", {})).items()},
    )


def validate_config(config: AppConfig) -> None:
    if config.evaluation.enabled and config.evaluation.lane not in {lane.key for lane in config.lanes}:
        raise SystemExit(f"[evaluation] lane '{config.evaluation.lane}' is not a configured lane.")
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
