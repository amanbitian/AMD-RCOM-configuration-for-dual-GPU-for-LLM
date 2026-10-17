"""Windows implementation: ctypes (kernel32/psapi), PDH GPU counters via one PowerShell call per
tick, netstat/tasklist/taskkill. Moved here unchanged in behaviour from app.py and server.py.

Every helper process is launched with CREATE_NO_WINDOW: the app normally runs detached, so an
unflagged subprocess would flash a console window on every call (GPU metrics run every tick).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from typing import Any

from .base import HostOS, ProcessIdentity

NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    def value(self) -> int:
        return (self.high << 32) | self.low


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("memory_load", ctypes.c_uint32),
        ("total_physical", ctypes.c_uint64),
        ("available_physical", ctypes.c_uint64),
        ("total_page_file", ctypes.c_uint64),
        ("available_page_file", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("available_virtual", ctypes.c_uint64),
        ("available_extended_virtual", ctypes.c_uint64),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("page_fault_count", ctypes.c_uint32),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


_PROCESS_QUERY_LIMITED = 0x1000


def _run(args: list[str], timeout: float = 30) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


class WindowsProcessProbe:
    def __init__(self) -> None:
        self._last: dict[int, tuple[int, float]] = {}

    def _open(self, pid: int):
        return ctypes.windll.kernel32.OpenProcess(_PROCESS_QUERY_LIMITED, False, pid)

    def identity(self, pid: int) -> ProcessIdentity | None:
        kernel32 = ctypes.windll.kernel32
        handle = self._open(pid)
        if not handle:
            return None
        try:
            created, exited, kernel, user = _FileTime(), _FileTime(), _FileTime(), _FileTime()
            if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                            ctypes.byref(kernel), ctypes.byref(user)):
                return None
            # A process that has exited but is still held open has a non-zero exit time.
            if exited.value() != 0:
                return None
            size = ctypes.c_uint32(1024)
            buffer = ctypes.create_unicode_buffer(size.value)
            exe = None
            if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                exe = os.path.basename(buffer.value)
            return ProcessIdentity(pid, created.value(), exe)
        finally:
            kernel32.CloseHandle(handle)

    def metrics(self, pid: int, now: float) -> dict[str, Any]:
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        handle = self._open(pid)
        if not handle:
            return {}
        try:
            created, exited, kernel, user = _FileTime(), _FileTime(), _FileTime(), _FileTime()
            total_100ns: int | None = None
            if kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                        ctypes.byref(kernel), ctypes.byref(user)):
                total_100ns = kernel.value() + user.value()
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            memory_ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            cpu_pct = None
            previous = self._last.get(pid)
            if total_100ns is not None and previous:
                delta_cpu_seconds = (total_100ns - previous[0]) / 10_000_000
                delta_wall = max(now - previous[1], 0.001)
                cpu_pct = max(0.0, min(100.0, delta_cpu_seconds / delta_wall / (os.cpu_count() or 1) * 100.0))
            if total_100ns is not None:
                self._last[pid] = (total_100ns, now)
            return {
                "cpu_pct": round(cpu_pct, 2) if cpu_pct is not None else None,
                "rss_bytes": int(counters.working_set_size) if memory_ok else None,
                "private_bytes": int(counters.pagefile_usage) if memory_ok else None,
                "peak_rss_bytes": int(counters.peak_working_set_size) if memory_ok else None,
                "peak_private_bytes": int(counters.peak_pagefile_usage) if memory_ok else None,
                "page_fault_count": int(counters.page_fault_count) if memory_ok else None,
            }
        finally:
            kernel32.CloseHandle(handle)

    def forget(self, pid: int) -> None:
        self._last.pop(pid, None)


class WindowsSystemProbe:
    def __init__(self) -> None:
        self._last: tuple[int, int, int] | None = None

    def metrics(self) -> dict[str, Any]:
        kernel32 = ctypes.windll.kernel32
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        cpu_pct = None
        if kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            current = (idle.value(), kernel.value(), user.value())
            if self._last:
                idle_delta = current[0] - self._last[0]
                total_delta = (current[1] - self._last[1]) + (current[2] - self._last[2])
                if total_delta > 0:
                    cpu_pct = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
            self._last = current
        memory = _MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        memory_ok = kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        return {
            "cpu_pct": round(cpu_pct, 2) if cpu_pct is not None else None,
            "ram_total_bytes": int(memory.total_physical) if memory_ok else None,
            "ram_available_bytes": int(memory.available_physical) if memory_ok else None,
            "ram_used_bytes": int(memory.total_physical - memory.available_physical) if memory_ok else None,
            "ram_used_pct": float(memory.memory_load) if memory_ok else None,
            "pagefile_used_bytes": int(memory.total_page_file - memory.available_page_file) if memory_ok else None,
            "pagefile_total_bytes": int(memory.total_page_file) if memory_ok else None,
        }


def parse_gpu_counter_output(output: str) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            pid, dedicated, shared, utilization = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        results[pid] = {
            "dedicated_gb": dedicated / 1024**3,
            "shared_gb": shared / 1024**3,
            "utilization_pct": max(0.0, min(100.0, utilization)),
        }
    return results


class _Luid(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("description", ctypes.c_wchar * 128),
        ("vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("subsys_id", ctypes.c_uint32),
        ("revision", ctypes.c_uint32),
        ("dedicated_video_memory", ctypes.c_size_t),
        ("dedicated_system_memory", ctypes.c_size_t),
        ("shared_system_memory", ctypes.c_size_t),
        ("luid", _Luid),
        ("flags", ctypes.c_uint32),
    ]


class _Guid(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16), ("data3", ctypes.c_uint16),
                ("data4", ctypes.c_ubyte * 8)]


_IID_IDXGI_FACTORY1 = _Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
# COM vtable slots: IUnknown 0-2, IDXGIObject 3-6, IDXGIFactory 7-11, IDXGIFactory1.EnumAdapters1 = 12;
# IDXGIAdapter 7-9, IDXGIAdapter1.GetDesc1 = 10.
_RELEASE, _ENUM_ADAPTERS1, _GET_DESC1 = 2, 12, 10


def _com(obj: ctypes.c_void_p, slot: int, *argtypes):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    method = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])
    return lambda *args: method(obj, *args)


def luid_key(high: int, low: int) -> str:
    """The `0x<high>_0x<low>` form PDH GPU counter instance names use."""
    return f"0x{high & 0xFFFFFFFF:08x}_0x{low & 0xFFFFFFFF:08x}"


def dxgi_adapters() -> list[dict[str, Any]]:
    """Every display adapter's name, LUID and dedicated VRAM, from DXGI (no extra process)."""
    factory = ctypes.c_void_p()
    if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGI_FACTORY1), ctypes.byref(factory)) != 0:
        return []
    adapters: list[dict[str, Any]] = []
    try:
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            if _com(factory, _ENUM_ADAPTERS1, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))(index, ctypes.byref(adapter)) != 0:
                break  # DXGI_ERROR_NOT_FOUND: no more adapters
            try:
                desc = _AdapterDesc1()
                if _com(adapter, _GET_DESC1, ctypes.POINTER(_AdapterDesc1))(ctypes.byref(desc)) == 0:
                    adapters.append({"name": desc.description, "luid": luid_key(desc.luid.high, desc.luid.low),
                                     "dedicated_bytes": int(desc.dedicated_video_memory)})
            finally:
                _com(adapter, _RELEASE)()
            index += 1
    finally:
        _com(factory, _RELEASE)()
    return adapters


def parse_adapter_usage(output: str) -> dict[str, float]:
    """`<instance> <bytes>` lines from GPU Adapter Memory counters -> bytes per LUID."""
    usage: dict[str, float] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2 or "luid_" not in parts[0]:
            continue
        luid = parts[0].split("luid_", 1)[1].rsplit("_phys_", 1)[0]
        try:
            usage[luid] = usage.get(luid, 0.0) + float(parts[1])
        except ValueError:
            continue
    return usage


def parse_process_adapter_usage(output: str, pids: list[int]) -> dict[int, dict[str, float]]:
    """`pid_<pid>_luid_<luid>_phys_<n> <bytes>` lines -> GB per LUID for each requested pid."""
    wanted = set(pids)
    usage: dict[int, dict[str, float]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].startswith("pid_") or "_luid_" not in parts[0]:
            continue
        pid_text, rest = parts[0][4:].split("_luid_", 1)
        try:
            pid, value = int(pid_text), float(parts[1])
        except ValueError:
            continue
        if pid in wanted:
            luid = rest.rsplit("_phys_", 1)[0]
            usage.setdefault(pid, {})
            usage[pid][luid] = usage[pid].get(luid, 0.0) + value / 1024**3
    return usage


class WindowsGpuProbe:
    def process_adapter_memory(self, pids: list[int]) -> dict[int, dict[str, float]]:
        """Dedicated VRAM each pid holds, per adapter LUID: what is actually resident on the GPU
        now (VRAM the OS has moved out to system RAM is not counted)."""
        if not pids:
            return {}
        output = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                       "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -EA SilentlyContinue)"
                       ".CounterSamples | ForEach-Object { \"$($_.InstanceName) $($_.CookedValue)\" }"])
        return parse_process_adapter_usage(output, pids)

    def adapter_memory(self) -> list[dict[str, Any]]:
        """Per physical GPU: name, total and in-use dedicated VRAM as Windows accounts it, which
        counts every process on the card. (HIP's free-memory figure on Windows describes only the
        calling process's virtualized view and reports an occupied card as empty.)"""
        try:
            adapters = [adapter for adapter in dxgi_adapters() if adapter["dedicated_bytes"] > 0]
        except (OSError, AttributeError):
            return []
        if not adapters:
            return []
        usage = parse_adapter_usage(_run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                                          "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -EA SilentlyContinue)"
                                          ".CounterSamples | ForEach-Object { \"$($_.InstanceName) $($_.CookedValue)\" }"]))
        return [{"name": adapter["name"], "luid": adapter["luid"], "total_gb": adapter["dedicated_bytes"] / 1024**3,
                 "used_gb": usage[adapter["luid"]] / 1024**3}
                for adapter in adapters if adapter["luid"] in usage]

    def process_metrics(self, pids: list[int]) -> dict[int, dict[str, float]]:
        """One PowerShell call for every PID: Get-Counter's counter-set resolution dominates the
        cost, so the three counter paths are requested together and N servers share one call.
        The instance filter anchors on the trailing underscore so pid 123 cannot match 1234."""
        unique = sorted(set(pids))
        if not unique:
            return {}
        script = (
            f"$targets=@({','.join(str(p) for p in unique)}); "
            "$samples=(Get-Counter -Counter @("
            "'\\GPU Process Memory(*)\\Dedicated Usage',"
            "'\\GPU Process Memory(*)\\Shared Usage',"
            "'\\GPU Engine(*)\\Utilization Percentage'"
            ") -EA SilentlyContinue).CounterSamples; "
            "$d=$samples | Where-Object {$_.Path -like '*dedicated usage'}; "
            "$s=$samples | Where-Object {$_.Path -like '*shared usage'}; "
            "$u=$samples | Where-Object {$_.Path -like '*utilization percentage'}; "
            "foreach ($targetPid in $targets) { "
            "$pattern=\"pid_${targetPid}_*\"; "
            "$dd=($d | Where-Object {$_.InstanceName -like $pattern} | Measure-Object CookedValue -Sum).Sum; "
            "$ss=($s | Where-Object {$_.InstanceName -like $pattern} | Measure-Object CookedValue -Sum).Sum; "
            "$uu=($u | Where-Object {$_.InstanceName -like $pattern} | Measure-Object CookedValue -Sum).Sum; "
            "Write-Output \"$targetPid $([double]$dd) $([double]$ss) $([double]$uu)\" }"
        )
        return parse_gpu_counter_output(_run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script]))


class WindowsProcessControl:
    server_binary_name = "llama-server.exe"

    def pid_listening_on(self, port: int) -> int | None:
        for line in _run(["netstat", "-ano", "-p", "TCP"]).splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "TCP" and parts[3].upper() == "LISTENING":
                if parts[1].rsplit(":", 1)[-1] == str(port):
                    try:
                        return int(parts[4])
                    except ValueError:
                        return None
        return None

    def is_llama_server(self, pid: int) -> bool:
        return "llama-server.exe" in _run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"]).lower()

    def terminate_tree(self, pid: int, grace_s: float = 10.0) -> None:
        _run(["taskkill", "/F", "/T", "/PID", str(pid)])

    def spawn_kwargs(self) -> dict[str, Any]:
        return {"creationflags": NO_WINDOW}


def build() -> HostOS:
    return HostOS(
        name="windows",
        process=WindowsProcessProbe(),
        system=WindowsSystemProbe(),
        gpu=WindowsGpuProbe(),
        control=WindowsProcessControl(),
    )
