from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dual_gpu_setup.config import AppConfig, LaneConfig, ModelConfig

# See server.py's own _NO_WINDOW for why: app.py runs detached (no console), so an
# unflagged subprocess.run on Windows pops a new console window per call.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

BACKEND_SPECS = {
    "rocm": {"dir_re": r"llama\.cpp-win-x86_64-amd-rocm-", "vendor_hint": "rocm", "prefix": "ROCm"},
    "vulkan": {"dir_re": r"llama\.cpp-win-x86_64-vulkan-", "vendor_hint": "vulkan", "prefix": "Vulkan"},
}

SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _version_key(path: Path) -> tuple[int, int, int]:
    match = re.search(r"-(\d+)\.(\d+)\.(\d+)$", path.name)
    return tuple(int(group) for group in match.groups()) if match else (0, 0, 0)


def backends_dir(config: AppConfig) -> Path:
    return Path(config.discovery.lmstudio_home) / "extensions" / "backends"


def runtime_dir(config: AppConfig, backend: str) -> Path:
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
    path = runtime_dir(config, backend) / "llama-server.exe"
    if not path.exists():
        raise SystemExit(f"llama-server.exe not found in {backend} runtime: {path}")
    return path


def backend_env(config: AppConfig, backend: str) -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(runtime_dir(config, backend))]
    vendor = vendor_bin(config, backend)
    if vendor:
        parts.append(str(vendor))
    env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
    return env


def list_devices(config: AppConfig, backend: str) -> list[tuple[str, str]]:
    proc = subprocess.run(
        [str(server_binary(config, backend)), "--list-devices"],
        capture_output=True,
        text=True,
        timeout=120,
        env=backend_env(config, backend),
        cwd=str(runtime_dir(config, backend)),
        creationflags=_NO_WINDOW,
    )
    prefix = BACKEND_SPECS[backend]["prefix"]
    found: list[tuple[str, str]] = []
    for line in (proc.stdout or "").splitlines():
        match = re.match(rf"\s*({prefix}\d+):\s*(.+)", line)
        if match:
            found.append((match.group(1), match.group(2).strip()))
    return found


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
