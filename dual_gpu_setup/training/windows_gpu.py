"""GPU hardware telemetry for AMD cards on Windows.

Linux reads these values from amdgpu sysfs; Windows has no such files, so they come from three places:

* **ADL**, AMD's driver library (`atiadlxx.dll`, installed with the Adrenalin driver): busy %, edge /
  junction / memory temperature, board power, clocks and fan, read in-process in well under a
  millisecond. This is what the 1 Hz sampler uses.
* **Windows performance counters** (`GPU Adapter Memory`): dedicated VRAM in use and shared (system)
  memory in use. Read through PowerShell, so it runs on a background thread every few seconds.
* **Windows device properties**: the PCIe link a card actually negotiated, and its driver version.
  Slow to query and almost constant, so refreshed about once a minute.

Which ADL sensor is which is not documented where this code can check it, so the ids below were
identified on an RX 9070 XT and an R9700 (RDNA4) by behaviour: edge matched Task Manager, the junction
sensor moved within a second of every change in power and activity while the memory sensor moved over
tens of seconds, and board power tracked each card's rated draw. Sensors that could not be identified
(ids 40 and 41, which are not the PCIe link) are not used. Other GPU generations may map differently.

Anything unreadable is returned as None with a reason, never as zero.
"""

from __future__ import annotations

import ctypes
import re
import sys
import threading
import time
from ctypes import c_char, c_int, c_size_t, c_void_p
from dataclasses import dataclass
from typing import Any, Callable

ADL_OK = 0
ADL_MAX_PATH = 256
ADL_PMLOG_MAX_SENSORS = 256
AMD_VENDOR_ID = 1002  # ADL reports the PCI vendor id (0x1002) in decimal digits

# ADL PMLog sensor ids (see the module docstring for how they were identified).
SENSOR_GFX_CLOCK = 1
SENSOR_MEMORY_CLOCK = 2
SENSOR_TEMP_EDGE = 8
SENSOR_TEMP_MEMORY = 9
SENSOR_FAN_RPM = 14
SENSOR_ACTIVITY_GFX = 19
SENSOR_TEMP_JUNCTION = 27
SENSOR_BOARD_POWER = 73

PCIE_GT_PER_S = {1: "2.5", 2: "5.0", 3: "8.0", 4: "16.0", 5: "32.0", 6: "64.0"}


class AdapterInfo(ctypes.Structure):
    _fields_ = [
        ("iSize", c_int), ("iAdapterIndex", c_int), ("strUDID", c_char * ADL_MAX_PATH), ("iBusNumber", c_int),
        ("iDeviceNumber", c_int), ("iFunctionNumber", c_int), ("iVendorID", c_int),
        ("strAdapterName", c_char * ADL_MAX_PATH), ("strDisplayName", c_char * ADL_MAX_PATH),
        ("iPresent", c_int), ("iExist", c_int), ("strDriverPath", c_char * ADL_MAX_PATH),
        ("strDriverPathExt", c_char * ADL_MAX_PATH), ("strPNPString", c_char * ADL_MAX_PATH),
        ("iOSDisplayIndex", c_int),
    ]


class SingleSensor(ctypes.Structure):
    _fields_ = [("supported", c_int), ("value", c_int)]


class PMLogOutput(ctypes.Structure):
    _fields_ = [("size", c_int), ("sensors", SingleSensor * ADL_PMLOG_MAX_SENSORS)]


class AdlUnavailable(RuntimeError):
    """ADL cannot be used on this machine; the message says why."""


@dataclass(frozen=True, slots=True)
class AdlAdapter:
    index: int
    name: str
    bus: int
    device: int
    function: int

    @property
    def pci_bus(self) -> str:
        return f"0000:{self.bus:02x}:{self.device:02x}.{self.function:x}"


class AdlLibrary:
    """The few ADL2 calls needed, over `atiadlxx.dll`. `dll` can be replaced in tests."""

    def __init__(self, dll: Any = None):
        if dll is None:
            if sys.platform != "win32":
                raise AdlUnavailable("ADL is a Windows driver library.")
            try:
                dll = ctypes.CDLL("atiadlxx.dll")
            except OSError as exc:
                raise AdlUnavailable(f"atiadlxx.dll not found (AMD Adrenalin driver not installed?): {exc}") from exc
        self.dll = dll
        self.context = c_void_p()
        self._malloc = None
        callback_type = ctypes.CFUNCTYPE(c_void_p, c_int)
        if sys.platform == "win32":
            libc = ctypes.CDLL("msvcrt")
            libc.malloc.restype = c_void_p
            libc.malloc.argtypes = [c_size_t]
            self._malloc = callback_type(lambda size: libc.malloc(size))
        else:  # tests with a fake dll
            self._malloc = callback_type(lambda size: None)
        code = dll.ADL2_Main_Control_Create(self._malloc, 1, ctypes.byref(self.context))
        if code != ADL_OK:
            raise AdlUnavailable(f"ADL2_Main_Control_Create returned {code}.")

    def adapters(self) -> list[AdlAdapter]:
        count = c_int()
        if self.dll.ADL2_Adapter_NumberOfAdapters_Get(self.context, ctypes.byref(count)) != ADL_OK or count.value <= 0:
            return []
        infos = (AdapterInfo * count.value)()
        if self.dll.ADL2_Adapter_AdapterInfo_Get(self.context, infos, ctypes.sizeof(infos)) != ADL_OK:
            return []
        found: dict[tuple[int, int, int], AdlAdapter] = {}
        for info in infos:  # one entry per display output; a card appears several times
            key = (info.iBusNumber, info.iDeviceNumber, info.iFunctionNumber)
            if info.iVendorID != AMD_VENDOR_ID or not info.iPresent or key in found:
                continue
            found[key] = AdlAdapter(info.iAdapterIndex, info.strAdapterName.decode(errors="replace").strip(), *key)
        return list(found.values())

    def sensors(self, adapter_index: int) -> dict[int, int] | None:
        """{sensor id: value} for every sensor the card supports, or None if the call failed."""
        out = PMLogOutput()
        out.size = ctypes.sizeof(PMLogOutput)
        if self.dll.ADL2_New_QueryPMLogData_Get(self.context, adapter_index, ctypes.byref(out)) != ADL_OK:
            return None
        return {i: out.sensors[i].value for i in range(ADL_PMLOG_MAX_SENSORS) if out.sensors[i].supported}

    def close(self) -> None:
        if self.context:
            self.dll.ADL2_Main_Control_Destroy(self.context)
            self.context = c_void_p()


_MEMORY_SCRIPT = (
    "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage','\\GPU Adapter Memory(*)\\Shared Usage' "
    "-ErrorAction SilentlyContinue).CounterSamples | ForEach-Object { "
    "\"mem|$($_.Path.Split('\\')[-1])|$($_.InstanceName)|$([long]$_.CookedValue)\" }"
)
_LINK_SCRIPT = (
    "Get-PnpDevice -Class Display -PresentOnly -ErrorAction SilentlyContinue | ForEach-Object { "
    "$i = $_.InstanceId; $g = { param($k) (Get-PnpDeviceProperty -InstanceId $i -KeyName $k -ErrorAction SilentlyContinue).Data }; "
    "\"link|$(& $g 'DEVPKEY_Device_LocationInfo')|$(& $g 'DEVPKEY_PciDevice_CurrentLinkSpeed')|"
    "$(& $g 'DEVPKEY_PciDevice_CurrentLinkWidth')|$(& $g 'DEVPKEY_PciDevice_MaxLinkSpeed')|"
    "$(& $g 'DEVPKEY_PciDevice_MaxLinkWidth')|$(& $g 'DEVPKEY_Device_DriverVersion')\" }"
)
_LOCATION_RE = re.compile(r"PCI bus (\d+), device (\d+), function (\d+)", re.I)


def parse_memory_lines(output: str) -> dict[str, dict[str, int]]:
    """`mem|<counter>|<instance>|<bytes>` lines -> {luid: {"dedicated": bytes, "shared": bytes}}."""
    usage: dict[str, dict[str, int]] = {}
    for line in output.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 4 or parts[0] != "mem" or "luid_" not in parts[2]:
            continue
        luid = parts[2].split("luid_", 1)[1].rsplit("_phys_", 1)[0]
        kind = "dedicated" if "dedicated" in parts[1].lower() else "shared" if "shared" in parts[1].lower() else None
        try:
            value = int(parts[3])
        except ValueError:
            continue
        if kind:
            usage.setdefault(luid, {})
            usage[luid][kind] = usage[luid].get(kind, 0) + value
    return usage


def parse_link_lines(output: str) -> dict[str, dict[str, Any]]:
    """`link|<location>|<cur speed>|<cur width>|<max speed>|<max width>|<driver>` -> {pci_bus: {...}}."""
    links: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 7 or parts[0] != "link":
            continue
        match = _LOCATION_RE.search(parts[1])
        if not match:
            continue
        bus, device, function = (int(group) for group in match.groups())

        def code(text: str) -> int | None:
            return int(text) if text.strip().isdigit() else None

        links[f"0000:{bus:02x}:{device:02x}.{function:x}"] = {
            "speed": code(parts[2]), "width": code(parts[3]), "max_speed": code(parts[4]), "max_width": code(parts[5]),
            "driver_version": parts[6].strip() or None,
        }
    return links


def speed_label(code: int | None) -> str | None:
    """Windows reports the PCIe generation; Linux sysfs reports `16.0 GT/s PCIe`. Use the Linux form."""
    return None if code is None or code not in PCIE_GT_PER_S else f"{PCIE_GT_PER_S[code]} GT/s PCIe"


class WindowsGpuReader:
    """Sensor source for HardwareSampler on Windows. See the module docstring."""

    MEMORY_INTERVAL_S = 5.0
    LINK_INTERVAL_S = 60.0

    def __init__(self, adl: Any, dxgi: Callable[[], list[dict[str, Any]]], powershell: Callable[[str], str]):
        self.adl = adl
        self._dxgi = dxgi
        self._powershell = powershell
        self._lock = threading.Lock()
        self._memory: dict[str, dict[str, int]] = {}
        self._links: dict[str, dict[str, Any]] = {}
        self._memory_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.refresh_errors = 0

    @classmethod
    def try_create(cls) -> tuple["WindowsGpuReader | None", str | None]:
        """(reader, None) on Windows with ADL, else (None, why not)."""
        if sys.platform != "win32":
            return None, "ADL telemetry is for Windows."
        try:
            adl = AdlLibrary()
            from dual_gpu_setup.hostos.windows import _run, dxgi_adapters

            return cls(adl, dxgi_adapters, lambda script: _run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=60)), None
        except AdlUnavailable as exc:
            return None, str(exc)

    # ---- discovery ----------------------------------------------------------------------------

    def discover(self) -> list[dict[str, Any]]:
        """One entry per AMD GPU: name, PCI address, ADL index, total VRAM and the LUID Windows counters use."""
        dxgi = [a for a in self._dxgi() if a.get("dedicated_bytes", 0) > 0 or "graphics" in a.get("name", "").lower()]
        by_name: dict[str, list[dict[str, Any]]] = {}
        for adapter in dxgi:
            by_name.setdefault(adapter["name"].strip().lower(), []).append(adapter)
        gpus = []
        for adl in self.adl.adapters():
            matches = by_name.get(adl.name.strip().lower(), [])
            match = matches[0] if len(matches) == 1 else None  # two identical cards cannot be told apart by name
            gpus.append({"name": adl.name, "pci_bus": adl.pci_bus, "adl_index": adl.index,
                         "vram_total_bytes": match["dedicated_bytes"] if match else None,
                         "luid": match["luid"] if match else None})
        return sorted(gpus, key=lambda g: g["pci_bus"])

    # ---- background refresh of the slow values ------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gpu-windows-slow", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self.adl.close()

    def _run(self) -> None:
        next_link = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                self.refresh_memory()
                if now >= next_link:
                    self.refresh_links()
                    next_link = now + self.LINK_INTERVAL_S
            except Exception:  # noqa: BLE001 - a failed read must not stop the thread
                self.refresh_errors += 1
            if self._stop.wait(self.MEMORY_INTERVAL_S):
                break

    def refresh_memory(self) -> None:
        memory = parse_memory_lines(self._powershell(_MEMORY_SCRIPT))
        with self._lock:
            self._memory = memory
            self._memory_at = time.monotonic()

    def refresh_links(self) -> None:
        links = parse_link_lines(self._powershell(_LINK_SCRIPT))
        with self._lock:
            self._links = links

    # ---- sampling -----------------------------------------------------------------------------

    def link(self, pci_bus: str) -> dict[str, Any] | None:
        with self._lock:
            return self._links.get(pci_bus)

    def sample(self, adl_index: int, pci_bus: str, luid: str | None, vram_total: int | None) -> dict[str, Any]:
        """Sensor fields for one GPU, in the same shape `sample_gpu` returns on Linux (without ts/identity)."""
        unavailable: dict[str, str] = {}

        def put(key: str, reading: Any, reason: str) -> Any:
            if reading is None:
                unavailable[key] = reason
            return reading

        sensors = self.adl.sensors(adl_index)
        failed = "ADL could not read this GPU's sensors"

        def sensor(sensor_id: int) -> int | None:
            if sensors is None:
                return None
            return sensors.get(sensor_id)

        def why(sensor_id: int) -> str:
            return failed if sensors is None else f"ADL sensor {sensor_id} not supported by this GPU"

        out: dict[str, Any] = {}
        out["gpu_busy_pct"] = put("gpu_busy_pct", sensor(SENSOR_ACTIVITY_GFX), why(SENSOR_ACTIVITY_GFX))
        for kind, sensor_id in (("edge", SENSOR_TEMP_EDGE), ("junction", SENSOR_TEMP_JUNCTION), ("memory", SENSOR_TEMP_MEMORY)):
            value = sensor(sensor_id)
            out[f"temp_{kind}_c"] = put(f"temp_{kind}_c", None if value is None else float(value), why(sensor_id))
        power = sensor(SENSOR_BOARD_POWER)
        out["power_w"] = put("power_w", None if power is None else float(power), why(SENSOR_BOARD_POWER))
        for key, sensor_id in (("sclk_mhz", SENSOR_GFX_CLOCK), ("mclk_mhz", SENSOR_MEMORY_CLOCK)):
            value = sensor(sensor_id)
            out[key] = put(key, None if value is None else float(value), why(sensor_id))
        out["fan_rpm"] = put("fan_rpm", sensor(SENSOR_FAN_RPM), why(SENSOR_FAN_RPM))

        with self._lock:
            memory = self._memory.get(luid or "")
            link = self._links.get(pci_bus)
            memory_known = self._memory_at is not None
        no_luid = "no Windows adapter matched this GPU by name"
        not_yet = "not read yet"
        out["vram_used_bytes"] = put("vram_used_bytes", memory.get("dedicated") if memory else None,
                                     no_luid if luid is None else (not_yet if not memory_known else "no GPU Adapter Memory counter"))
        out["vram_total_bytes"] = put("vram_total_bytes", vram_total, no_luid)
        out["gtt_used_bytes"] = put("gtt_used_bytes", memory.get("shared") if memory else None,
                                    no_luid if luid is None else (not_yet if not memory_known else "no GPU Adapter Memory counter"))
        link_reason = "Windows device properties not read yet" if not self._links else "no PCIe link properties for this GPU"
        out["pcie_speed"] = put("pcie_speed", speed_label(link["speed"]) if link else None, link_reason)
        out["pcie_width"] = put("pcie_width", link["width"] if link else None, link_reason)
        out["pcie_max_speed"] = speed_label(link["max_speed"]) if link else None
        out["pcie_max_width"] = link["max_width"] if link else None
        out["unavailable"] = unavailable
        return out
