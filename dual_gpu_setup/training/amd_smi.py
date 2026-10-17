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
            if throttle is not None and not (isinstance(throttle, str) and throttle.upper() == "N/A"):
                out["throttle_status"] = throttle
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
