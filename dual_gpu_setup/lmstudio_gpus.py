"""Choose which GPUs LM Studio's own llama.cpp runtime may use.

LM Studio keeps that choice in `<lmstudio_home>/.internal/hardware-config.json`, per backend, under
`load.gpuSplitConfig.disabledGpus` (device indices in the order `llama-server --list-devices` prints
them). This module reads and rewrites only that list; every other field in the file is preserved.

It does not touch LM Studio's running process: a model that is already loaded keeps its placement, and
a running LM Studio may not re-read the file until it is restarted and the model reloaded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from dual_gpu_setup.config import AppConfig
from dual_gpu_setup.hostos import get_host
from dual_gpu_setup.lmstudio import BACKEND_SPECS, list_devices, runtime_dir

SPLIT_KEY = "load.gpuSplitConfig"
_CONFIG_LOCK = threading.Lock()  # one read-modify-write of hardware-config.json at a time
NOTE = (
    "Saved to LM Studio's hardware config. A model that is already loaded keeps its current GPUs, and "
    "a running LM Studio may not re-read this file until it is restarted - reload the model afterwards "
    "and check where it landed."
)


class GpuSelectionError(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def hardware_config_path(config: AppConfig) -> Path:
    return Path(config.discovery.lmstudio_home) / ".internal" / "hardware-config.json"


def _backend(backend: str) -> dict[str, str]:
    spec = BACKEND_SPECS.get(backend)
    if spec is None:
        raise GpuSelectionError(f"Unsupported backend '{backend}'. Use one of: {', '.join(sorted(BACKEND_SPECS))}.")
    return spec


def _devices(config: AppConfig, backend: str) -> list[dict[str, Any]]:
    try:
        found = list_devices(config, backend)
    except SystemExit as exc:
        raise GpuSelectionError(str(exc), status=409) from exc
    devices = []
    for device_id, description in found:
        index = int(re.search(r"(\d+)$", device_id).group(1))
        match = re.match(r"(.*?)\s*\(([^()]*MiB[^()]*)\)\s*$", description)
        devices.append({
            "index": index,
            "device": device_id,
            "name": match.group(1) if match else description,
            "memory": match.group(2) if match else None,
        })
    return devices


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"json": [], "meta": {"values": ["map"]}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuSelectionError(f"Cannot read LM Studio hardware config {path}: {exc}", status=409) from exc
    entries = doc.get("json") if isinstance(doc, dict) else None
    if not isinstance(entries, list) or any(
        not (isinstance(e, list) and len(e) == 2 and isinstance(e[0], str) and isinstance(e[1], dict)) for e in entries
    ):
        raise GpuSelectionError(f"Unrecognised LM Studio hardware config layout in {path}; refusing to edit it.", status=409)
    return doc


def _entries(doc: dict[str, Any], backend: str) -> list[list]:
    pattern = _backend(backend)["dir_re"]
    return [entry for entry in doc["json"] if re.match(pattern, entry[0])]


def _split_field(entry: list, create: bool = False) -> dict[str, Any] | None:
    fields = entry[1].setdefault("fields", [])
    for field in fields:
        if isinstance(field, dict) and field.get("key") == SPLIT_KEY and isinstance(field.get("value"), dict):
            return field
    if not create:
        return None
    field = {"key": SPLIT_KEY, "value": {"strategy": "evenly", "disabledGpus": [], "priority": [], "customRatio": []}}
    fields.insert(0, field)
    return field


def _disabled(doc: dict[str, Any], backend: str) -> list[int]:
    for entry in _entries(doc, backend):
        field = _split_field(entry)
        if field is not None:
            return sorted(int(i) for i in field["value"].get("disabledGpus", []))
    return []


def lms_binary(config: AppConfig) -> str | None:
    found = shutil.which("lms")
    if found:
        return found
    candidate = Path(config.discovery.lmstudio_home) / "bin" / ("lms.exe" if os.name == "nt" else "lms")
    return str(candidate) if candidate.exists() else None


def _lms(config: AppConfig, *args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    binary = lms_binary(config)
    if binary is None:
        raise GpuSelectionError("The lms CLI was not found; open LM Studio once or put it on PATH.", status=409)
    try:
        return subprocess.run([binary, *args], capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace", **get_host().control.spawn_kwargs())
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuSelectionError(f"lms {' '.join(args)} failed: {exc}", status=409) from exc


def lm_studio_state(config: AppConfig) -> dict[str, Any]:
    """Whether LM Studio answers `lms ps`, and which models it holds in memory (and if they are busy)."""
    try:
        proc = _lms(config, "ps", "--json", timeout=20)
        raw = json.loads(proc.stdout or "[]") if proc.returncode == 0 else None
    except (GpuSelectionError, ValueError) as exc:
        return {"reachable": False, "models": [], "error": str(exc)}
    if raw is None:
        return {"reachable": False, "models": [], "error": (proc.stderr or proc.stdout or "").strip()[-300:]}
    return {"reachable": True, "models": [
        {"identifier": m.get("identifier"), "status": m.get("status"), "queued": m.get("queued") or 0,
         "size_bytes": m.get("sizeBytes"), "context_length": m.get("contextLength")} for m in raw]}


def release(config: AppConfig, *, force: bool = False, stop_server: bool = False) -> dict[str, Any]:
    """Free the GPUs and RAM LM Studio holds: unload every model, and optionally stop its server.

    Refuses (409) while any model is generating or has queued requests, unless `force`."""
    before = lm_studio_state(config)
    busy = [m["identifier"] for m in before["models"] if m["status"] != "idle" or m["queued"]]
    if busy and not force:
        raise GpuSelectionError(
            f"LM Studio is busy with {', '.join(map(str, busy))}; retry when idle or pass force=true.", status=409)
    if before["models"]:
        proc = _lms(config, "unload", "--all", timeout=120)
        if proc.returncode != 0:
            raise GpuSelectionError(f"lms unload --all failed: {(proc.stderr or proc.stdout).strip()[-300:]}", status=409)
    if stop_server:
        proc = _lms(config, "server", "stop", timeout=60)
        if proc.returncode != 0:
            raise GpuSelectionError(f"lms server stop failed: {(proc.stderr or proc.stdout).strip()[-300:]}", status=409)
    return {"unloaded": [m["identifier"] for m in before["models"]], "server_stopped": stop_server,
            "lm_studio": lm_studio_state(config)}


def _view(config: AppConfig, backend: str, devices: list[dict[str, Any]], doc: dict[str, Any]) -> dict[str, Any]:
    disabled = _disabled(doc, backend)
    listed = [{**d, "enabled": d["index"] not in disabled} for d in devices]
    return {
        "backend": backend,
        "config_path": str(hardware_config_path(config)),
        "devices": listed,
        "enabled": [d["index"] for d in listed if d["enabled"]],
        "disabled_gpus": disabled,
        "lm_studio": lm_studio_state(config),
    }


def get_selection(config: AppConfig, backend: str = "rocm") -> dict[str, Any]:
    _backend(backend)
    return _view(config, backend, _devices(config, backend), _load(hardware_config_path(config)))


def _resolve(devices: list[dict[str, Any]], selectors: Any) -> set[int]:
    if isinstance(selectors, str) and selectors.strip().lower() == "all":
        return {d["index"] for d in devices}
    if not isinstance(selectors, list) or not selectors:
        raise GpuSelectionError('"gpus" must be "all" or a non-empty list of device indexes or names.')
    chosen: set[int] = set()
    for selector in selectors:
        if isinstance(selector, bool) or not isinstance(selector, (int, str)):
            raise GpuSelectionError(f"Unusable GPU selector: {selector!r}.")
        text = str(selector).strip().lower()
        if text.isdigit():
            matches = [d for d in devices if d["index"] == int(text)]
        else:
            matches = [d for d in devices if text == d["device"].lower() or text in d["name"].lower()]
        if len(matches) != 1:
            listing = ", ".join(f"{d['index']}={d['name']}" for d in devices) or "(none found)"
            reason = "matches no GPU" if not matches else "matches more than one GPU"
            raise GpuSelectionError(f"GPU selector {selector!r} {reason}. Available: {listing}.")
        chosen.add(matches[0]["index"])
    return chosen


def set_selection(config: AppConfig, gpus: Any, backend: str = "rocm") -> dict[str, Any]:
    """Allow only `gpus` (indexes, names, device ids, or "all") for LM Studio's `backend` runtime."""
    _backend(backend)
    devices = _devices(config, backend)
    chosen = _resolve(devices, gpus)
    disabled = sorted({d["index"] for d in devices} - chosen)

    path = hardware_config_path(config)
    with _CONFIG_LOCK:
        existed = path.exists()
        doc = _load(path)
        previous = _disabled(doc, backend)

        entries = _entries(doc, backend)
        if not entries:
            key = re.sub(r"-\d+\.\d+\.\d+$", "", runtime_dir(config, backend).name)
            entries = [[key, {"fields": []}]]
            doc["json"].append(entries[0])
        for entry in entries:
            _split_field(entry, create=True)["value"]["disabledGpus"] = disabled

        path.parent.mkdir(parents=True, exist_ok=True)
        backup = None
        if existed:
            backup = path.with_name(path.name + ".dgpu.bak")
            if not backup.exists():  # keep the very first original, not the latest edit
                backup.write_bytes(path.read_bytes())
        _atomic_write(path, json.dumps(doc, separators=(",", ":")))

    result = _view(config, backend, devices, doc)
    result.update(previous_disabled_gpus=previous, backup_path=str(backup) if backup else None, note=NOTE)
    return result


def _atomic_write(path: Path, text: str) -> None:
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except OSError as exc:
        Path(temp_name).unlink(missing_ok=True)
        raise GpuSelectionError(f"Could not write {path}: {exc}", status=409) from exc
