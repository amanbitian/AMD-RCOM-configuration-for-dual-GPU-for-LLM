"""Optional AMD SMI enrichment for the sysfs sampler.

AMD SMI coverage on consumer Radeon cards is narrower than on Instinct, and its Python API differs
between ROCm releases, so every call is wrapped: a missing function or an N/A value simply leaves the
field unset (and sysfs, or `null`, stands). Values only FILL fields sysfs left empty; the source of
every filled field is reported so a number is never ambiguous about where it came from.
"""

from __future__ import annotations

from typing import Any


def _call(module: Any, name: str, *args: Any) -> Any:
    function = getattr(module, name, None)
    if function is None:
        return None
    try:
        return function(*args)
    except Exception:  # noqa: BLE001 - N/A and unsupported sensors raise AmdSmiLibraryException
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        digits = value.split()[0] if value.split() else ""
        try:
            return float(digits)
        except ValueError:
            return None
    return None


class AmdSmiReader:
    def __init__(self, module: Any = None):
        self.available = False
        self.reason: str | None = None
        self._module = module
        self._handles: dict[str, Any] = {}
        if module is None:
            try:
                import amdsmi as module  # type: ignore[no-redef]
            except Exception as exc:  # noqa: BLE001
                self.reason = f"amdsmi not importable: {exc}"
                return
            self._module = module
        try:
            _call(self._module, "amdsmi_init")
            for handle in _call(self._module, "amdsmi_get_processor_handles") or []:
                bdf = _call(self._module, "amdsmi_get_gpu_device_bdf", handle)
                if isinstance(bdf, str):
                    self._handles[bdf.lower()] = handle
        except Exception as exc:  # noqa: BLE001
            self.reason = f"amdsmi initialisation failed: {exc}"
            return
        self.available = bool(self._handles)
        if not self.available:
            self.reason = self.reason or "amdsmi reported no processors"

    def _temperature(self, handle: Any, kind: str) -> float | None:
        m = self._module
        types, metrics = getattr(m, "AmdSmiTemperatureType", None), getattr(m, "AmdSmiTemperatureMetric", None)
        if types is None or metrics is None or not hasattr(types, kind):
            return None
        return _number(_call(m, "amdsmi_get_temp_metric", handle, getattr(types, kind), metrics.CURRENT))

    def read(self, pci_bus: str) -> dict[str, Any]:
        handle = self._handles.get(pci_bus.lower())
        if handle is None:
            return {}
        m = self._module
        out: dict[str, Any] = {}
        for field, kind in (("temp_edge_c", "EDGE"), ("temp_junction_c", "JUNCTION"), ("temp_memory_c", "VRAM")):
            value = self._temperature(handle, kind)
            if value is not None:
                out[field] = value
        power = _call(m, "amdsmi_get_power_info", handle)
        if isinstance(power, dict):
            watts = _number(power.get("average_socket_power") if _number(power.get("average_socket_power")) else power.get("current_socket_power"))
            if watts:
                out["power_w"] = watts
        activity = _call(m, "amdsmi_get_gpu_activity", handle)
        if isinstance(activity, dict) and _number(activity.get("gfx_activity")) is not None:
            out["gpu_busy_pct"] = _number(activity["gfx_activity"])
        clocks = getattr(m, "AmdSmiClkType", None)
        if clocks is not None:
            for field, name in (("sclk_mhz", "GFX"), ("mclk_mhz", "MEM")):
                if hasattr(clocks, name):
                    info = _call(m, "amdsmi_get_clock_info", handle, getattr(clocks, name))
                    value = _number(info.get("clk") if isinstance(info, dict) else None)
                    if value:
                        out[field] = value
        metrics = _call(m, "amdsmi_get_gpu_metrics_info", handle)
        if isinstance(metrics, dict):
            throttle = metrics.get("throttle_status")
            numeric = _number(throttle)
            if numeric is not None:
                out["throttle_status"] = int(numeric)
        return out

    def merge(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Fill only the fields sysfs left empty; record which ones came from AMD SMI."""
        extra = self.read(sample["pci_bus"])
        filled = []
        for field, value in extra.items():
            if field == "throttle_status" or sample.get(field) is None:
                sample[field] = value
                filled.append(field)
                sample.get("unavailable", {}).pop(field, None)
        if filled:
            sample["amd_smi_fields"] = filled
        return sample

    def close(self) -> None:
        if self.available:
            _call(self._module, "amdsmi_shut_down")

def _find_throttle(node: Any) -> Any:
    """First value stored under a key containing 'throttle', anywhere in nested JSON."""
    if isinstance(node, dict):
        for key, value in node.items():
            if "throttle" in str(key).lower():
                return value
        for value in node.values():
            found = _find_throttle(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_throttle(value)
            if found is not None:
                return found
    return None


def _throttle_flag(value: Any) -> int | None:
    """Non-zero when any throttle reason is active; None for N/A. The raw bit layout differs by ASIC,
    so only active / not active is reported."""
    if value is None or isinstance(value, bool):
        return None if value is None else int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "n/a", "na", "none"):
            return None if text != "none" else 0
        try:
            return int(text, 0)
        except ValueError:
            return 1 if text not in ("0", "inactive", "false", "no") else 0
    if isinstance(value, dict):
        flags = [_throttle_flag(v) for v in value.values()]
        known = [f for f in flags if f is not None]
        return None if not known else int(any(known))
    if isinstance(value, list):
        return int(any(_throttle_flag(v) for v in value))
    return None


class AmdSmiCli:
    """Fallback when the `amdsmi` Python module is missing: reads throttle status from the `amd-smi`
    command (installed with ROCm). The command is slow, so it runs on a background thread every
    `refresh_s` seconds and `merge` only reads the cached result."""

    def __init__(self, binary: str, runner: Any = None, refresh_s: float = 5.0, autostart: bool = True):
        import threading

        self.binary = binary
        self.available = True
        self.reason: str | None = None
        self.source = "amd-smi cli"
        self._runner = runner or self._run
        self._cache: dict[str, int | None] = {}
        self._refresh_s = refresh_s
        self._stop = threading.Event()
        if autostart:
            threading.Thread(target=self._loop, name="amd-smi-cli", daemon=True).start()

    def _run(self, *args: str) -> Any:
        import json
        import subprocess

        done = subprocess.run([self.binary, *args, "--json"], capture_output=True, text=True, timeout=30)
        if done.returncode != 0:
            raise RuntimeError((done.stderr or done.stdout or "amd-smi failed").strip()[:200])
        return json.loads(done.stdout)

    def refresh(self) -> None:
        try:
            listing = self._runner("list")
            metrics = self._runner("metric")
            if isinstance(metrics, dict):
                metrics = metrics.get("gpu_data") or list(metrics.values())
            by_index = {}
            for entry in metrics if isinstance(metrics, list) else []:
                if isinstance(entry, dict):
                    by_index[int(entry.get("gpu", len(by_index)))] = entry
            cache: dict[str, int | None] = {}
            for entry in listing if isinstance(listing, list) else []:
                bdf = str(entry.get("bdf", "")).lower()
                if bdf:
                    cache[bdf] = _throttle_flag(_find_throttle(by_index.get(int(entry.get("gpu", -1)), {})))
            self._cache = cache
            self.reason = None if any(v is not None for v in cache.values()) else "amd-smi reports throttle status as N/A on this GPU"
        except Exception as exc:  # noqa: BLE001
            self.reason = f"amd-smi query failed: {exc}"

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self._refresh_s)

    def read(self, pci_bus: str) -> dict[str, Any]:
        value = self._cache.get(pci_bus.lower())
        return {} if value is None else {"throttle_status": value}

    def merge(self, sample: dict[str, Any]) -> dict[str, Any]:
        extra = self.read(sample["pci_bus"])
        if extra:
            sample.update(extra)
            sample["amd_smi_fields"] = ["throttle_status"]
            sample.get("unavailable", {}).pop("throttle_status", None)
        return sample

    def close(self) -> None:
        self._stop.set()


def make_reader() -> Any:
    """The best available AMD SMI source: the Python module, else the `amd-smi` command, else the
    failed module reader (whose `reason` says exactly what is missing, for the health page)."""
    import shutil

    reader = AmdSmiReader()
    if reader.available:
        reader.source = "amdsmi python"
        return reader
    binary = shutil.which("amd-smi")
    if binary:
        fallback = AmdSmiCli(binary)
        fallback.module_reason = reader.reason
        return fallback
    reader.source = None
    return reader

