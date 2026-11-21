"""Host-side sensors that sit beside the GPU telemetry: CPU temperature and power, RAM module and SSD
temperature, coolant temperature, pump and fan speeds, memory pressure and free disk space.

A missing sensor is `None` with a reason in `unavailable`, never 0. Linux reads kernel hwmon / RAPL files
(`/sys/class/hwmon`, `/sys/class/powercap`). Windows has no equivalent in the OS itself, so it reads the
WMI namespace published by a running LibreHardwareMonitor or OpenHardwareMonitor (the workstation's
hardware monitor app); without one every Windows host sensor is reported unavailable with that reason.
Both readers are constructed with injectable roots/runners so they are testable without the hardware.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

FIELDS = ("cpu_temp_c", "cpu_power_w", "ram_temp_c", "ssd_temp_c", "coolant_temp_c", "pump_rpm", "cpu_fan_rpm",
          "case_fan_rpm", "mem_pressure_pct", "disk_free_bytes")

_CPU_CHIPS = ("k10temp", "coretemp", "zenpower", "cpu_thermal")
_SSD_CHIPS = ("nvme", "drivetemp")
_RAM_CHIPS = ("spd5118", "jc42", "ee1004")
_COOLANT_WORDS = ("coolant", "liquid", "water", "fluid")


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return None


def _int(path: Path) -> int | None:
    text = _read(path)
    try:
        return int(text.split()[0]) if text else None
    except (ValueError, IndexError):
        return None


class LinuxHostSensors:
    def __init__(self, data_dir: Path | str = ".", hwmon_root: str | Path = "/sys/class/hwmon",
                 powercap_root: str | Path = "/sys/class/powercap", proc_root: str | Path = "/proc",
                 clock: Callable[[], float] = time.monotonic):
        self.data_dir = Path(data_dir)
        self.hwmon_root = Path(hwmon_root)
        self.powercap_root = Path(powercap_root)
        self.proc_root = Path(proc_root)
        self.clock = clock
        self._last_energy: tuple[float, float] | None = None  # (clock, microjoules)
        self._energy_reason: str | None = None

    def _chips(self) -> list[tuple[str, Path]]:
        if not self.hwmon_root.is_dir():
            return []
        return [(_read(p / "name") or "", p) for p in sorted(self.hwmon_root.iterdir())]

    @staticmethod
    def _temps(chip: Path) -> list[tuple[str, float]]:
        out = []
        for path in sorted(chip.glob("temp*_input")):
            value = _int(path)
            if value is not None and -40_000 < value < 150_000:
                out.append(((_read(path.with_name(path.name.replace("_input", "_label"))) or "").lower(), value / 1000.0))
        return out

    @staticmethod
    def _fans(chip: Path) -> list[tuple[str, float]]:
        out = []
        for path in sorted(chip.glob("fan*_input")):
            value = _int(path)
            if value is not None:
                out.append(((_read(path.with_name(path.name.replace("_input", "_label"))) or path.name.split("_")[0]).lower(), float(value)))
        return out

    def _cpu_power(self, chips: list[tuple[str, Path]]) -> tuple[float | None, str | None]:
        for name, chip in chips:
            if name == "zenpower":
                micro = _int(chip / "power1_input")
                if micro is not None:
                    return micro / 1e6, None
        energy: float | None = None
        for name, chip in chips:
            if name == "amd_energy":
                values = [v for p in sorted(chip.glob("energy*_input")) if (v := _int(p)) is not None]
                if values:
                    energy = float(sum(values))  # per-socket and per-core counters in microjoules
                    break
        if energy is None and self.powercap_root.is_dir():
            for domain in sorted(self.powercap_root.glob("*rapl*:[0-9]")):
                if (_read(domain / "name") or "").startswith("package"):
                    value = _int(domain / "energy_uj")
                    if value is None:
                        self._energy_reason = f"{domain}/energy_uj is not readable (needs root or a udev rule)"
                    else:
                        energy = float(value)
                    break
        if energy is None:
            return None, self._energy_reason or "no zenpower, amd_energy or RAPL package counter exposed"
        now = self.clock()
        previous, self._last_energy = self._last_energy, (now, energy)
        if previous is None or now <= previous[0] or energy < previous[1]:
            return None, "CPU energy counter needs two readings"
        return (energy - previous[1]) / 1e6 / (now - previous[0]), None

    def read(self) -> tuple[dict[str, Any], dict[str, str]]:
        values: dict[str, Any] = {k: None for k in FIELDS}
        why: dict[str, str] = {}
        chips = self._chips()
        cpu, ssd, ram, coolant = [], [], [], []
        pump, cpu_fans, case_fans = [], [], []
        for name, chip in chips:
            temps, fans = self._temps(chip), self._fans(chip)
            if name in _CPU_CHIPS:
                preferred = [v for label, v in temps if any(w in label for w in ("tctl", "tdie", "package"))]
                cpu.extend(preferred or [v for _l, v in temps[:1]])
            elif name in _SSD_CHIPS:
                ssd.extend(v for label, v in temps if "composite" in label or name == "drivetemp" or not label)
            elif name in _RAM_CHIPS:
                ram.extend(v for _l, v in temps)
            for label, value in temps:
                if any(w in label for w in _COOLANT_WORDS):
                    coolant.append(value)
            for label, rpm in fans:
                if "pump" in label:
                    pump.append(rpm)
                elif "cpu" in label:
                    cpu_fans.append(rpm)
                elif name not in ("amdgpu", "radeon") and rpm >= 0:
                    case_fans.append(rpm)
        for key, found, reason in (
            ("cpu_temp_c", cpu, "no k10temp/coretemp/zenpower hwmon chip"),
            ("ssd_temp_c", ssd, "no nvme/drivetemp hwmon chip"),
            ("ram_temp_c", ram, "no spd5118/jc42 hwmon chip (load the kernel module for the DIMM sensors)"),
            ("coolant_temp_c", coolant, "no hwmon temperature labelled coolant/liquid/water (cooler needs a hwmon driver)"),
            ("pump_rpm", pump, "no hwmon fan labelled pump"),
            ("cpu_fan_rpm", cpu_fans, "no hwmon fan labelled cpu"),
            ("case_fan_rpm", case_fans, "no other motherboard hwmon fan readings"),
        ):
            if found:
                values[key] = max(found)
            else:
                why[key] = reason
        values["cpu_power_w"], reason = self._cpu_power(chips)
        if reason:
            why["cpu_power_w"] = reason
        pressure = _read(self.proc_root / "pressure" / "memory")
        if pressure and pressure.startswith("some"):
            for part in pressure.splitlines()[0].split():
                if part.startswith("avg10="):
                    values["mem_pressure_pct"] = float(part.split("=")[1])
        else:
            why["mem_pressure_pct"] = "/proc/pressure/memory not available"
        try:
            values["disk_free_bytes"] = shutil.disk_usage(self.data_dir).free
        except OSError as exc:
            why["disk_free_bytes"] = str(exc)
        return values, why


class WindowsHostSensors:
    """Reads the sensors LibreHardwareMonitor / OpenHardwareMonitor publish through WMI. One PowerShell
    call every `refresh_s` seconds on a background thread (never per sample), cached in between."""

    QUERY = ("$o=@(); foreach($ns in 'root/LibreHardwareMonitor','root/OpenHardwareMonitor'){ try{ "
             "$o=Get-CimInstance -Namespace $ns -ClassName Sensor -ErrorAction Stop; break }catch{} }; "
             "$o | Select-Object Name,SensorType,Value,Identifier | ConvertTo-Json -Compress")

    def __init__(self, data_dir: Path | str = ".", refresh_s: float = 10.0,
                 runner: Callable[[], str] | None = None, autostart: bool = True):
        self.data_dir = Path(data_dir)
        self.refresh_s = refresh_s
        self.runner = runner or self._powershell
        self._cache: list[dict[str, Any]] | None = None
        self._reason = "LibreHardwareMonitor / OpenHardwareMonitor is not running (no WMI sensor namespace)"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if autostart:
            self.start()

    @staticmethod
    def _powershell() -> str:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", WindowsHostSensors.QUERY],
                              capture_output=True, text=True, timeout=20, creationflags=flags)
        return done.stdout

    def refresh(self) -> None:
        try:
            text = (self.runner() or "").strip()
            data = json.loads(text) if text else []
            if isinstance(data, dict):
                data = [data]
            self._cache = data
        except Exception as exc:  # noqa: BLE001
            self._cache = None
            self._reason = f"sensor query failed: {type(exc).__name__}: {exc}"

    def start(self) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                self.refresh()
                self._stop.wait(self.refresh_s)

        self._thread = threading.Thread(target=loop, name="host-sensors", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def read(self) -> tuple[dict[str, Any], dict[str, str]]:
        values: dict[str, Any] = {k: None for k in FIELDS}
        why: dict[str, str] = {}
        sensors = self._cache
        if not sensors:
            reason = self._reason
            why = {k: reason for k in FIELDS if k not in ("mem_pressure_pct", "disk_free_bytes")}
        else:
            def pick(kind: str, test: Callable[[str, str], bool]) -> list[float]:
                return [float(s["Value"]) for s in sensors if s.get("SensorType") == kind and s.get("Value") is not None
                        and test((s.get("Identifier") or "").lower(), (s.get("Name") or "").lower())]

            def first(items: list[float], preferred: list[float] | None = None) -> float | None:
                items = preferred or items
                return max(items) if items else None

            cpu = pick("Temperature", lambda i, n: "/cpu" in i or "/amdcpu" in i or "/intelcpu" in i)
            package = pick("Temperature", lambda i, n: ("/cpu" in i or "cpu" in i) and any(w in n for w in ("package", "tctl", "tdie")))
            values["cpu_temp_c"] = first(cpu, package)
            values["cpu_power_w"] = first(pick("Power", lambda i, n: "cpu" in i and "package" in n))
            values["ssd_temp_c"] = first(pick("Temperature", lambda i, n: "/nvme" in i or "/hdd" in i or "/ssd" in i))
            values["ram_temp_c"] = first(pick("Temperature", lambda i, n: "/ram" in i or "/memory" in i or "dimm" in n))
            values["coolant_temp_c"] = first(pick("Temperature", lambda i, n: any(w in n for w in _COOLANT_WORDS)))
            values["pump_rpm"] = first(pick("Fan", lambda i, n: "pump" in n))
            values["cpu_fan_rpm"] = first(pick("Fan", lambda i, n: "cpu" in n and "pump" not in n))
            values["case_fan_rpm"] = first(pick("Fan", lambda i, n: "pump" not in n and "cpu" not in n and "gpu" not in n and "/gpu" not in i))
            for key, value in values.items():
                if value is None and key not in ("mem_pressure_pct", "disk_free_bytes"):
                    why[key] = "sensor not published by the hardware monitor"
        why["mem_pressure_pct"] = "no memory-pressure counter on Windows"
        try:
            values["disk_free_bytes"] = shutil.disk_usage(self.data_dir).free
            why.pop("disk_free_bytes", None)
        except OSError as exc:
            why["disk_free_bytes"] = str(exc)
        return values, why


def make_host_sensors(data_dir: Path | str, platform_name: str | None = None) -> Any:
    platform_name = platform_name or sys.platform
    if platform_name == "win32":
        return WindowsHostSensors(data_dir)
    return LinuxHostSensors(data_dir)
