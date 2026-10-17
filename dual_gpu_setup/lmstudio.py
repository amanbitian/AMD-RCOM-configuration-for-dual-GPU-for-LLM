from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dual_gpu_setup.config import AppConfig, LaneConfig, ModelConfig
from dual_gpu_setup.hostos import get_host

BACKEND_SPECS = {
    "rocm": {"dir_re": r"llama\.cpp-(?:win|linux)-x86_64-amd-rocm-", "vendor_hint": "rocm", "prefix": "ROCm"},
    "vulkan": {"dir_re": r"llama\.cpp-(?:win|linux)-x86_64-vulkan-", "vendor_hint": "vulkan", "prefix": "Vulkan"},
}

SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
VERSION_RE = re.compile(r"version:\s*(\S+)\s*\(build\s*(\d+),\s*commit\s*([0-9a-fA-F]+)\)")
FLAG_RE = re.compile(r"(?<![\w-])--[a-z0-9][a-z0-9-]*")


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Which llama-server build a lane runs and the command-line flags it accepts.

    `flags` is empty when the binary could not be probed; callers then add no capability-dependent
    arguments, so the command is exactly what it was before probing existed.
    """

    binary: str
    source: str  # "llama.cpp" (a configured build) or "lmstudio" (LM Studio's bundled runtime)
    version: str = ""
    build: int | None = None
    commit: str = ""
    flags: frozenset[str] = frozenset()

    def supports(self, flag: str) -> bool:
        return flag in self.flags

    def public(self) -> dict[str, object]:
        return {"binary": self.binary, "source": self.source, "version": self.version,
                "build": self.build, "commit": self.commit}


_RUNTIME_CACHE: dict[tuple[str, int], RuntimeInfo] = {}
_RUNTIME_LOCK = threading.Lock()


def _version_key(path: Path) -> tuple[int, int, int]:
    match = re.search(r"-(\d+)\.(\d+)\.(\d+)$", path.name)
    return tuple(int(group) for group in match.groups()) if match else (0, 0, 0)


def backends_dir(config: AppConfig) -> Path:
    return Path(config.discovery.lmstudio_home) / "extensions" / "backends"


def native_server_binary(config: AppConfig, backend: str) -> Path | None:
    """An explicit llama-server path (no LM Studio needed): an official llama.cpp release or a
    native Linux build. When unset, the newest LM Studio runtime for the backend is used."""
    configured = config.discovery.backend_server_binaries.get(backend, "").strip()
    if not configured:
        return None
    path = Path(configured)
    if not path.is_file():
        raise SystemExit(f"Configured {backend} server binary does not exist: {path}")
    return path


def multi_gpu_config(config: AppConfig) -> AppConfig:
    """The config a lane-spanning server resolves its runtime from. With `multi_gpu_runtime =
    "lmstudio"` the configured binaries are dropped, so every lookup below lands on LM Studio's."""
    if config.discovery.multi_gpu_runtime != "lmstudio" or not config.discovery.backend_server_binaries:
        return config
    return replace(config, discovery=replace(config.discovery, backend_server_binaries={}))


def runtime_dir(config: AppConfig, backend: str) -> Path:
    native = native_server_binary(config, backend)
    if native is not None:
        return native.parent
    override = config.discovery.backend_runtime_overrides.get(backend, "").strip()
    if override:
        path = Path(override)
        if not path.is_dir():
            raise SystemExit(f"Configured {backend} runtime does not exist: {path}")
        return path

    spec = BACKEND_SPECS.get(backend)
    if spec is None:
        raise SystemExit(
            f"Unsupported backend '{backend}'. Built-in support exists for: {', '.join(sorted(BACKEND_SPECS))}."
        )

    root = backends_dir(config)
    if not root.is_dir():
        raise SystemExit(f"LM Studio backends directory not found: {root}")
    matches = [path for path in root.iterdir() if path.is_dir() and re.match(spec["dir_re"], path.name)]
    if not matches:
        raise SystemExit(
            f"No LM Studio '{backend}' runtime found under {root}. "
            f"Install it in LM Studio or set discovery.backend_runtime_overrides.{backend}."
        )
    return max(matches, key=_version_key)


def vendor_bin(config: AppConfig, backend: str) -> Path | None:
    override = config.discovery.backend_vendor_bin_overrides.get(backend, "").strip()
    if override:
        path = Path(override)
        return path if path.exists() else None
    # A configured llama.cpp build ships its own GPU runtime DLLs next to the binary; LM Studio's
    # vendor folder on PATH could only supply mismatched ones.
    if native_server_binary(config, backend) is not None:
        return None

    spec = BACKEND_SPECS.get(backend)
    if spec is None:
        return None
    vendor_root = backends_dir(config) / "vendor"
    if not vendor_root.is_dir():
        return None
    candidates = [path for path in vendor_root.iterdir() if spec["vendor_hint"] in path.name.lower()]
    if not candidates:
        return None
    best = max(candidates, key=lambda item: int(match.group(1)) if (match := re.search(r"v(\d+)$", item.name)) else 0)
    return best / "bin" if (best / "bin").is_dir() else best


def server_binary(config: AppConfig, backend: str) -> Path:
    native = native_server_binary(config, backend)
    if native is not None:
        return native
    name = get_host().control.server_binary_name
    path = runtime_dir(config, backend) / name
    if not path.exists():
        raise SystemExit(f"{name} not found in {backend} runtime: {path}")
    return path


def backend_env(config: AppConfig, backend: str) -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(runtime_dir(config, backend))]
    vendor = vendor_bin(config, backend)
    if vendor:
        parts.append(str(vendor))
    env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
    if sys.platform != "win32":
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts + ([existing] if existing else []))
    return env


def parse_runtime(binary: str, source: str, version_text: str, help_text: str) -> RuntimeInfo:
    match = VERSION_RE.search(version_text)
    return RuntimeInfo(
        binary=binary,
        source=source,
        version=match.group(1) if match else "",
        build=int(match.group(2)) if match else None,
        commit=match.group(3) if match else "",
        flags=frozenset(FLAG_RE.findall(help_text)),
    )


def runtime_info(config: AppConfig, backend: str, binary: Path | None = None) -> RuntimeInfo:
    """Probe `--version` and `--help` once per binary file (cached by path and mtime)."""
    binary = binary if binary is not None else server_binary(config, backend)
    source = "lmstudio" if backends_dir(config) in binary.parents else "llama.cpp"
    try:
        key = (str(binary), binary.stat().st_mtime_ns)
    except OSError:
        return RuntimeInfo(binary=str(binary), source=source)
    with _RUNTIME_LOCK:
        cached = _RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached
    outputs = []
    for flag in ("--version", "--help"):
        try:
            proc = subprocess.run(
                [str(binary), flag], capture_output=True, text=True, errors="replace", timeout=60,
                env=backend_env(config, backend), cwd=str(binary.parent), **get_host().control.spawn_kwargs(),
            )
            outputs.append((proc.stdout or "") + (proc.stderr or ""))
        except (OSError, subprocess.SubprocessError):
            return RuntimeInfo(binary=str(binary), source=source)
    info = parse_runtime(str(binary), source, *outputs)
    with _RUNTIME_LOCK:
        _RUNTIME_CACHE[key] = info
    return info


def list_devices(config: AppConfig, backend: str) -> list[tuple[str, str]]:
    proc = subprocess.run(
        [str(server_binary(config, backend)), "--list-devices"],
        capture_output=True,
        text=True,
        timeout=120,
        env=backend_env(config, backend),
        cwd=str(runtime_dir(config, backend)),
        **get_host().control.spawn_kwargs(),
    )
    prefix = BACKEND_SPECS[backend]["prefix"]
    found: list[tuple[str, str]] = []
    for line in (proc.stdout or "").splitlines():
        match = re.match(rf"\s*({prefix}\d+):\s*(.+)", line)
        if match:
            found.append((match.group(1), match.group(2).strip()))
    return found


DEVICE_MEMORY_RE = re.compile(r"\((\d+) MiB, (\d+) MiB free\)")


def lane_free_vram_gb(config: AppConfig, lanes: list[LaneConfig]) -> dict[str, float]:
    """Free VRAM per lane, counting every process on the card. Lanes whose device or figure
    cannot be read are omitted.

    Windows: total minus in-use as the OS accounts it (DXGI + GPU Adapter Memory counters). HIP's
    `--list-devices` figure there is the calling process's virtualized view -- measured reporting
    an R9700 holding a 16.9 GB model as 32,472 of 32,624 MiB free. Linux: `--list-devices`."""
    adapters = get_host().gpu.adapter_memory()
    if adapters:
        return free_from_adapters(lanes, adapters)
    return free_from_device_listing(config, lanes)


def free_from_adapters(lanes: list[LaneConfig], adapters: list[dict[str, Any]]) -> dict[str, float]:
    free: dict[str, float] = {}
    for lane in lanes:
        for adapter in adapters:
            if lane.match.lower() in adapter["name"].lower():
                free[lane.key] = max(0.0, adapter["total_gb"] - adapter["used_gb"])
                break
    return free


def free_from_device_listing(config: AppConfig, lanes: list[LaneConfig]) -> dict[str, float]:
    listings: dict[str, list[tuple[str, str]]] = {}
    free: dict[str, float] = {}
    for lane in lanes:
        if lane.backend not in listings:
            listings[lane.backend] = list_devices(config, lane.backend)
        for _device_id, description in listings[lane.backend]:
            if lane.match.lower() in description.lower():
                match = DEVICE_MEMORY_RE.search(description)
                if match:
                    free[lane.key] = int(match.group(2)) / 1024
                break
    return free


def resolve_device(config: AppConfig, lane: LaneConfig) -> str:
    devices = list_devices(config, lane.backend)
    for device_id, description in devices:
        if lane.match.lower() in description.lower():
            return device_id
    listing = "\n".join(f"  {device_id}: {description}" for device_id, description in devices) or "  (none)"
    raise SystemExit(
        f"Could not find lane match '{lane.match}' in backend '{lane.backend}' devices:\n{listing}"
    )


def models_root(config: AppConfig) -> Path:
    if config.discovery.models_dir:
        return Path(config.discovery.models_dir)

    settings_path = Path(config.discovery.lmstudio_home) / "settings.json"
    try:
        folder = json.loads(settings_path.read_text(encoding="utf-8")).get("downloadsFolder")
        if folder:
            return Path(folder)
    except (OSError, json.JSONDecodeError):
        pass
    return Path(config.discovery.lmstudio_home) / "models"


def resolve_model_path(config: AppConfig, model: ModelConfig) -> Path:
    raw_path = Path(os.path.expandvars(os.path.expanduser(model.path)))
    if raw_path.is_absolute():
        if any(char in model.path for char in "*?["):
            matches = sorted(raw_path.parent.glob(raw_path.name))
            matches = [path for path in matches if "mmproj" not in path.name.lower()]
            if matches:
                return max(matches, key=lambda path: path.stat().st_size)
        return raw_path

    root = models_root(config)
    if any(char in model.path for char in "*?["):
        matches = sorted(root.glob(model.path))
        matches = [path for path in matches if "mmproj" not in path.name.lower()]
        if matches:
            return max(matches, key=lambda path: path.stat().st_size)
        return root / model.path
    return root / model.path


def model_bytes(config: AppConfig, model: ModelConfig) -> int:
    path = resolve_model_path(config, model)
    if not path.exists():
        return 0
    match = SHARD_RE.search(path.name)
    if not match:
        return path.stat().st_size
    stem = path.name[: match.start()]
    total = match.group(2)
    return sum(candidate.stat().st_size for candidate in path.parent.glob(f"{stem}-*-of-{total}.gguf"))


def model_size_gb(config: AppConfig, model: ModelConfig) -> float:
    if model.size_gb > 0:
        return model.size_gb
    size_bytes = model_bytes(config, model)
    return round(size_bytes / 1024**3, 2) if size_bytes else 0.0
