"""Independent GPU hardware telemetry from Linux amdgpu sysfs.

Runs in the dashboard process (never inside the training process). Every field that a card does
not expose is returned as None with a reason in `unavailable`, so a missing sensor is never
mistaken for a zero reading. On non-Linux hosts discovery returns no GPUs and says why.
"""

from __future__ import annotations

import collections
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from dual_gpu_setup.hostos import get_host

AMD_VENDOR = "0x1002"
_CARD_RE = re.compile(r"^card\d+$")
_PCI_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$")


def normalize_pci_bus(value: str) -> str:
    value = (value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]", value):
        value = "0000:" + value
    return value


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return None


def _read_int(path: Path) -> int | None:
    text = _read(path)
    if text is None:
        return None
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return None


def _pci_slot(device: Path) -> str | None:
    """PCI address of a drm device: `PCI_SLOT_NAME` from its uevent, else the resolved path name."""
    for line in (_read(device / "uevent") or "").splitlines():
        if line.startswith("PCI_SLOT_NAME="):
            slot = normalize_pci_bus(line.split("=", 1)[1])
            if _PCI_RE.match(slot):
                return slot
    try:
        slot = normalize_pci_bus(device.resolve().name)
    except OSError:
        return None
    return slot if _PCI_RE.match(slot) else None


@dataclass(slots=True)
class GpuDevice:
    pci_bus: str
    card: str
    device_path: Path
    hwmon_path: Path | None
    name: str
    lane: str | None = None
    vram_total_bytes: int | None = None
    temp_files: dict[str, Path] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        return {"pci_bus": self.pci_bus, "card": self.card, "name": self.name, "lane": self.lane,
                "vram_total_bytes": self.vram_total_bytes}


def discover_gpus(sysfs_root: str | Path = "/sys", platform_name: str | None = None) -> tuple[list[GpuDevice], str | None]:
    """Returns (gpus, unavailable_reason). Reason is set when discovery itself is not possible."""
    platform_name = platform_name or sys.platform
    if not platform_name.startswith("linux"):
        return [], f"GPU sysfs telemetry requires Linux (running on {platform_name})."
    drm = Path(sysfs_root) / "class" / "drm"
    if not drm.is_dir():
        return [], f"{drm} not found."
    devices: list[GpuDevice] = []
    for entry in sorted(drm.iterdir(), key=lambda p: p.name):
        if not _CARD_RE.match(entry.name):
            continue
        device = entry / "device"
        if _read(device / "vendor") != AMD_VENDOR:
            continue
        pci_bus = _pci_slot(device)
        if pci_bus is None:
            continue
        if not _PCI_RE.match(pci_bus):
            continue
        hwmon = None
        hwmon_root = device / "hwmon"
        if hwmon_root.is_dir():
            candidates = sorted(hwmon_root.iterdir(), key=lambda p: p.name)
            hwmon = candidates[0] if candidates else None
        name = _read(device / "product_name") or _read(device / "device") or entry.name
        gpu = GpuDevice(pci_bus=pci_bus, card=entry.name, device_path=device, hwmon_path=hwmon, name=name,
                        vram_total_bytes=_read_int(device / "mem_info_vram_total"))
        if hwmon is not None:
            gpu.temp_files = _temperature_files(hwmon)
        devices.append(gpu)
    return devices, None if devices else "No AMD GPUs found under /sys/class/drm."


def _temperature_files(hwmon: Path) -> dict[str, Path]:
    """Map edge/junction/memory to temp*_input by reading temp*_label."""
    found: dict[str, Path] = {}
    for label_file in sorted(hwmon.glob("temp*_label")):
        label = (_read(label_file) or "").lower()
        input_file = hwmon / label_file.name.replace("_label", "_input")
        if not input_file.exists():
            continue
        if label.startswith("edge"):
            found["edge"] = input_file
        elif label.startswith("junction") or label.startswith("hotspot"):
            found["junction"] = input_file
        elif label.startswith("mem"):
            found["memory"] = input_file
    return found


def assign_lanes(gpus: list[GpuDevice], lanes: Iterable[Any]) -> None:
    """Map each GPU to a configured lane: exact pci_bus, then name match, then unique VRAM size.
    Name matching is only a fallback because Linux marketing names are not reliable."""
    lanes = list(lanes)
    unassigned = [g for g in gpus]
    claimed: set[str] = set()

    def claim(gpu: GpuDevice, lane: Any) -> None:
        gpu.lane = lane.key
        claimed.add(lane.key)
        if gpu.name == gpu.card or gpu.name.startswith("0x"):
            gpu.name = getattr(lane, "display", None) or gpu.name

    for gpu in list(unassigned):
        for lane in lanes:
            bus = normalize_pci_bus(getattr(lane, "pci_bus", ""))
            if bus and bus == gpu.pci_bus and lane.key not in claimed:
                claim(gpu, lane)
                unassigned.remove(gpu)
                break
    for gpu in list(unassigned):
        for lane in lanes:
            if lane.key in claimed or getattr(lane, "pci_bus", ""):
                continue
            needle = (lane.match or "").lower()
            if needle and needle in gpu.name.lower():
                claim(gpu, lane)
                unassigned.remove(gpu)
                break
    for gpu in list(unassigned):
        if not gpu.vram_total_bytes:
            continue
        gpu_gb = gpu.vram_total_bytes / 1024**3
        matches = [
            lane for lane in lanes
            if lane.key not in claimed and not getattr(lane, "pci_bus", "")
            and abs(lane.vram_gb - gpu_gb) <= max(1.5, 0.12 * lane.vram_gb)
        ]
        if len(matches) == 1:
            claim(gpu, matches[0])
            unassigned.remove(gpu)


def torch_pci_address(device: dict[str, Any]) -> str | None:
    """Rebuild 'dddd:bb:dd.0' from the PCI fields PyTorch exposes on a device (None if absent)."""
    bus = device.get("pci_bus_id")
    if isinstance(bus, str) and _PCI_RE.match(bus.lower()):
        return bus.lower()
    if bus is None:
        return None
    return f"{int(device.get('pci_domain_id') or 0):04x}:{int(bus):02x}:{int(device.get('pci_device_id') or 0):02x}.0"


def map_torch_devices(torch_devices: list[dict[str, Any]], gpus: list[GpuDevice]) -> list[dict[str, Any]]:
    """Join PyTorch device indices to physical GPUs and lanes. PCI address is authoritative; a
    unique VRAM-size match is the fallback (HIP_VISIBLE_DEVICES can renumber indices)."""
    by_bus = {g.pci_bus: g for g in gpus}
    mapping = []
    for device in torch_devices:
        address = torch_pci_address(device)
        gpu, how = (by_bus.get(address), "pci_bus") if address else (None, None)
        if gpu is None:
            total = device.get("total_memory")
            candidates = [g for g in gpus if total and g.vram_total_bytes
                          and abs(g.vram_total_bytes - total) <= 0.06 * g.vram_total_bytes]
            if len(candidates) == 1:
                gpu, how = candidates[0], "vram_size"
        mapping.append({"torch_index": device.get("index"), "torch_name": device.get("name"),
                        "pci_bus": gpu.pci_bus if gpu else address, "lane": gpu.lane if gpu else None,
                        "matched_by": how})
    return mapping


def sample_gpu(gpu: GpuDevice, now: float | None = None) -> dict[str, Any]:
    device = gpu.device_path
    unavailable: dict[str, str] = {}

    def value(field_name: str, reading: Any, reason: str) -> Any:
        if reading is None:
            unavailable[field_name] = reason
        return reading

    sample: dict[str, Any] = {
        "ts": time.time() if now is None else now,
        "pci_bus": gpu.pci_bus,
        "lane": gpu.lane,
        "name": gpu.name,
    }
    sample["gpu_busy_pct"] = value("gpu_busy_pct", _read_int(device / "gpu_busy_percent"), "gpu_busy_percent not exposed")
    sample["vram_used_bytes"] = value("vram_used_bytes", _read_int(device / "mem_info_vram_used"), "mem_info_vram_used not exposed")
    sample["vram_total_bytes"] = value("vram_total_bytes", _read_int(device / "mem_info_vram_total"), "mem_info_vram_total not exposed")
    sample["gtt_used_bytes"] = value("gtt_used_bytes", _read_int(device / "mem_info_gtt_used"), "mem_info_gtt_used not exposed")
    for kind in ("edge", "junction", "memory"):
        path = gpu.temp_files.get(kind)
        millidegrees = _read_int(path) if path else None
        sample[f"temp_{kind}_c"] = value(
            f"temp_{kind}_c", None if millidegrees is None else millidegrees / 1000.0,
            f"{kind} temperature sensor not exposed by hwmon",
        )
    power_w = None
    if gpu.hwmon_path is not None:
        for name in ("power1_average", "power1_input"):
            microwatts = _read_int(gpu.hwmon_path / name)
            if microwatts is not None:
                power_w = microwatts / 1_000_000.0
                break
    sample["power_w"] = value("power_w", power_w, "hwmon power1_average/power1_input not exposed")
    sclk = _read_int(gpu.hwmon_path / "freq1_input") if gpu.hwmon_path else None
    mclk = _read_int(gpu.hwmon_path / "freq2_input") if gpu.hwmon_path else None
    sample["sclk_mhz"] = value("sclk_mhz", None if sclk is None else sclk / 1_000_000.0, "hwmon freq1_input not exposed")
    sample["mclk_mhz"] = value("mclk_mhz", None if mclk is None else mclk / 1_000_000.0, "hwmon freq2_input not exposed")
    sample["fan_rpm"] = value("fan_rpm", _read_int(gpu.hwmon_path / "fan1_input") if gpu.hwmon_path else None, "hwmon fan1_input not exposed")
    sample["pcie_speed"] = value("pcie_speed", _read(device / "current_link_speed"), "current_link_speed not exposed")
    width = _read(device / "current_link_width")
    sample["pcie_width"] = value("pcie_width", int(width) if width and width.isdigit() else None, "current_link_width not exposed")
    sample["pcie_max_speed"] = _read(device / "max_link_speed")
    sample["pcie_max_width"] = _read(device / "max_link_width")
    sample["unavailable"] = unavailable
    return sample


class HardwareSampler:
    """1 Hz background sampler with a live ring buffer and self-measured overhead."""

    def __init__(
        self,
        lanes: Iterable[Any] = (),
        interval_s: float = 1.0,
        history_seconds: int = 900,
        sysfs_root: str | Path = "/sys",
        platform_name: str | None = None,
        on_samples: Callable[[list[dict[str, Any]]], None] | None = None,
        on_system: Callable[[dict[str, Any]], None] | None = None,
        smi: Any = None,
    ):
        self.interval_s = interval_s
        self.on_samples = on_samples
        self.on_system = on_system
        self.smi = smi
        self.system_latest: dict[str, Any] | None = None
        self.gpus, self.unavailable_reason = discover_gpus(sysfs_root, platform_name)
        assign_lanes(self.gpus, lanes)
        self._history: dict[str, collections.deque] = {
            g.pci_bus: collections.deque(maxlen=max(10, int(history_seconds / interval_s))) for g in self.gpus
        }
        self._latest: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ticks = 0
        self.dropped = 0
        self.last_duration_ms: float | None = None
        self.max_duration_ms = 0.0
        self.total_duration_ms = 0.0
        self.callback_errors = 0
        self.read_errors = 0
        self.last_error: str | None = None

    def sample_once(self) -> list[dict[str, Any]]:
        started = time.perf_counter()
        samples = [sample_gpu(g) for g in self.gpus]
        if self.smi is not None and getattr(self.smi, "available", False):
            for sample in samples:
                self.smi.merge(sample)
        system = self._system_sample()
        duration_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            for s in samples:
                self._latest[s["pci_bus"]] = s
                self._history[s["pci_bus"]].append(s)
            self.ticks += 1
            self.last_duration_ms = duration_ms
            self.max_duration_ms = max(self.max_duration_ms, duration_ms)
            self.total_duration_ms += duration_ms
        for callback, payload in ((self.on_samples, samples if samples else None), (self.on_system, system)):
            if callback and payload:
                try:
                    callback(payload)
                except Exception:  # noqa: BLE001 - telemetry must never take the dashboard down
                    self.callback_errors += 1
        return samples

    def _system_sample(self) -> dict[str, Any] | None:
        try:
            metrics = get_host().system.metrics()
        except Exception:  # noqa: BLE001
            return None
        self.system_latest = {"ts": time.time(), "cpu_pct": metrics.get("cpu_pct"),
                              "ram_used_pct": metrics.get("ram_used_pct"),
                              "swap_used_bytes": metrics.get("pagefile_used_bytes")}
        return self.system_latest

    def hardware_environment(self) -> dict[str, Any]:
        """Identity and link state recorded with each run (read now, not cached): physical GPUs by
        PCI address, the PCIe link they actually negotiated, driver and kernel versions."""
        import platform as stdlib_platform

        def text(path: str) -> str | None:
            return _read(Path(path))

        gpus = []
        for gpu in self.gpus:
            device = gpu.device_path
            gpus.append({**gpu.identity(), "pcie_speed": _read(device / "current_link_speed"),
                         "pcie_width": _read(device / "current_link_width"), "pcie_max_speed": _read(device / "max_link_speed"),
                         "pcie_max_width": _read(device / "max_link_width"), "vbios_version": _read(device / "vbios_version")})
        return {"gpus": gpus, "kernel": stdlib_platform.release(), "amdgpu_driver": text("/sys/module/amdgpu/version"),
                "rocm_version": text("/opt/rocm/.info/version"), "unavailable_reason": self.unavailable_reason,
                "visibility": {k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES") if os.environ.get(k)}}

    def start(self) -> None:
        if not self.gpus or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception as exc:  # noqa: BLE001 - a bad sensor read must never kill the sampler thread
                self.read_errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            next_tick += self.interval_s
            delay = next_tick - time.monotonic()
            if delay < 0:
                # Overran the interval: count the skipped ticks and resynchronise.
                missed = int(-delay // self.interval_s) + 1
                self.dropped += missed
                next_tick = time.monotonic()
                delay = 0
            if self._stop.wait(delay):
                break

    def snapshot(self, history_points: int = 120) -> dict[str, Any]:
        with self._lock:
            gpus = []
            for gpu in self.gpus:
                latest = self._latest.get(gpu.pci_bus)
                history = list(self._history[gpu.pci_bus])[-history_points:]
                gpus.append({
                    **gpu.identity(),
                    "latest": latest,
                    "history": [
                        {k: s.get(k) for k in ("ts", "gpu_busy_pct", "vram_used_bytes", "temp_junction_c",
                                               "temp_memory_c", "temp_edge_c", "power_w")}
                        for s in history
                    ],
                })
            avg = self.total_duration_ms / self.ticks if self.ticks else None
            return {
                "available": bool(self.gpus),
                "unavailable_reason": self.unavailable_reason,
                "interval_s": self.interval_s,
                "gpus": gpus,
                "system": self.system_latest,
                "amd_smi": None if self.smi is None else {"available": self.smi.available, "reason": self.smi.reason},
                "sampler": {
                    "ticks": self.ticks, "dropped_ticks": self.dropped,
                    "last_sample_ms": self.last_duration_ms, "avg_sample_ms": avg,
                    "max_sample_ms": self.max_duration_ms if self.ticks else None,
                    "callback_errors": self.callback_errors, "read_errors": self.read_errors,
                    "last_error": self.last_error,
                },
            }


KERNEL_FAULT_PATTERNS = (
    (re.compile(r"amdgpu.*(gpu reset|GPU reset begin|GPU reset succeeded)", re.I), "gpu_reset", "error"),
    (re.compile(r"amdgpu.*ring .* timeout", re.I), "ring_timeout", "error"),
    (re.compile(r"amdgpu.*(page fault|VM_L2_PROTECTION_FAULT|GCVM_L2_PROTECTION_FAULT)", re.I), "page_fault", "error"),
    (re.compile(r"amdgpu.*(failed to allocate|out of memory|VRAM is lost)", re.I), "gpu_memory_error", "error"),
    (re.compile(r"amdgpu.*(thermal|throttl|over.?temperature)", re.I), "thermal", "warning"),
    (re.compile(r"amdgpu.*(PCIE error|AER|link .* down)", re.I), "pcie_error", "error"),
)


def classify_kernel_line(message: str) -> tuple[str, str] | None:
    """Returns (kind, severity) for an amdgpu kernel log line worth recording, else None."""
    for pattern, kind, severity in KERNEL_FAULT_PATTERNS:
        if pattern.search(message):
            return kind, severity
    return None


class KernelFaultWatcher:
    """Follows the kernel journal for amdgpu faults. Best effort: needs Linux, journalctl, and
    membership in systemd-journal/adm; otherwise it records why it is inactive and does nothing."""

    def __init__(self, on_fault: Callable[[str, str, dict[str, Any]], None]):
        self.on_fault = on_fault
        self.active = False
        self.inactive_reason: str | None = None
        self._process: Any = None

    def start(self) -> None:
        import json
        import shutil
        import subprocess

        if not sys.platform.startswith("linux"):
            self.inactive_reason = "Kernel log follower requires Linux."
            return
        journalctl = shutil.which("journalctl")
        if journalctl is None:
            self.inactive_reason = "journalctl not found."
            return
        try:
            self._process = subprocess.Popen(
                [journalctl, "-k", "-f", "-n", "0", "-o", "json"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            )
        except OSError as exc:
            self.inactive_reason = str(exc)
            return
        self.active = True

        def follow() -> None:
            assert self._process is not None and self._process.stdout is not None
            for line in self._process.stdout:
                try:
                    message = str(json.loads(line).get("MESSAGE", ""))
                except ValueError:
                    continue
                hit = classify_kernel_line(message)
                if hit:
                    try:
                        self.on_fault(hit[0], hit[1], {"line": message[:2000]})
                    except Exception:  # noqa: BLE001
                        pass
            self.active = False

        threading.Thread(target=follow, name="kernel-faults", daemon=True).start()

    def stop(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
            except OSError:
                pass
