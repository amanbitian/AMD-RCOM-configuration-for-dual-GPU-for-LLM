"""Linux implementation: /proc, amdgpu DRM fdinfo, and POSIX signals. No subprocess per tick.

Every reader takes a `proc_root` so tests can point it at a fake /proc tree.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

from .base import HostOS, ProcessIdentity

_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PAGE = 4096


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def parse_stat(text: str) -> dict[str, int] | None:
    """Fields after the parenthesised comm (which may itself contain spaces or parens)."""
    close = text.rfind(")")
    if close < 0:
        return None
    rest = text[close + 1:].split()
    try:
        return {
            "state": ord(rest[0][0]),
            "minflt": int(rest[7]), "majflt": int(rest[9]),
            "utime": int(rest[11]), "stime": int(rest[12]),
            "starttime": int(rest[19]),
        }
    except (IndexError, ValueError):
        return None


def parse_kv_kb(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1)
    return values


class LinuxProcessProbe:
    def __init__(self, proc_root: str | Path = "/proc"):
        self.root = Path(proc_root)
        self._last: dict[int, tuple[int, float]] = {}
        self._peak_private: dict[int, int] = {}

    def _stat(self, pid: int) -> dict[str, int] | None:
        text = _read(self.root / str(pid) / "stat")
        return parse_stat(text) if text else None

    def identity(self, pid: int) -> ProcessIdentity | None:
        stat = self._stat(pid)
        if stat is None or stat["state"] == ord("Z"):
            return None
        try:
            exe = os.path.basename(os.readlink(self.root / str(pid) / "exe"))
        except OSError:
            exe = None
        return ProcessIdentity(pid, stat["starttime"], exe)

    def metrics(self, pid: int, now: float) -> dict[str, Any]:
        stat = self._stat(pid)
        if stat is None:
            return {}
        status = parse_kv_kb(_read(self.root / str(pid) / "status") or "")
        ticks = stat["utime"] + stat["stime"]
        cpu_pct = None
        previous = self._last.get(pid)
        if previous:
            delta_cpu = (ticks - previous[0]) / _CLK_TCK
            delta_wall = max(now - previous[1], 0.001)
            cpu_pct = max(0.0, min(100.0, delta_cpu / delta_wall / (os.cpu_count() or 1) * 100.0))
        self._last[pid] = (ticks, now)
        private = status.get("RssAnon", status.get("VmData"))
        if private is not None:
            self._peak_private[pid] = max(self._peak_private.get(pid, 0), private)
        return {
            "cpu_pct": round(cpu_pct, 2) if cpu_pct is not None else None,
            "rss_bytes": status.get("VmRSS"),
            "private_bytes": private,
            "peak_rss_bytes": status.get("VmHWM"),
            "peak_private_bytes": self._peak_private.get(pid),
            "page_fault_count": stat["minflt"] + stat["majflt"],
        }

    def forget(self, pid: int) -> None:
        self._last.pop(pid, None)
        self._peak_private.pop(pid, None)


class LinuxSystemProbe:
    def __init__(self, proc_root: str | Path = "/proc"):
        self.root = Path(proc_root)
        self._last: tuple[int, int] | None = None

    def metrics(self) -> dict[str, Any]:
        cpu_pct = None
        first = (_read(self.root / "stat") or "").splitlines()[:1]
        if first and first[0].startswith("cpu "):
            values = [int(v) for v in first[0].split()[1:9]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            if self._last is not None and total > self._last[1]:
                cpu_pct = max(0.0, min(100.0, (1.0 - (idle - self._last[0]) / (total - self._last[1])) * 100.0))
            self._last = (idle, total)
        mem = parse_kv_kb(_read(self.root / "meminfo") or "")
        total_ram, available = mem.get("MemTotal"), mem.get("MemAvailable")
        swap_total, swap_free = mem.get("SwapTotal"), mem.get("SwapFree")
        used = None if total_ram is None or available is None else total_ram - available
        return {
            "cpu_pct": round(cpu_pct, 2) if cpu_pct is not None else None,
            "ram_total_bytes": total_ram,
            "ram_available_bytes": available,
            "ram_used_bytes": used,
            "ram_used_pct": None if used is None or not total_ram else round(used / total_ram * 100.0, 1),
            "pagefile_used_bytes": None if swap_total is None or swap_free is None else swap_total - swap_free,
            "pagefile_total_bytes": swap_total,
        }


def parse_fdinfo(text: str) -> dict[str, Any] | None:
    """amdgpu DRM client accounting from /proc/<pid>/fdinfo/<fd>; None if not a DRM fd."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    if "drm-driver" not in fields:
        return None

    def number(key: str) -> int:
        parts = fields.get(key, "").split()
        if not parts or not parts[0].isdigit():
            return 0
        scale = 1024 if len(parts) > 1 and parts[1].lower() == "kib" else 1
        return int(parts[0]) * scale

    return {
        "client": (fields.get("drm-driver"), fields.get("drm-pdev"), fields.get("drm-client-id")),
        "vram": number("drm-memory-vram"),
        "gtt": number("drm-memory-gtt"),
        "engine_ns": number("drm-engine-gfx") + number("drm-engine-compute"),
    }


class LinuxGpuProbe:
    """Per-process VRAM/GTT and engine busy % from DRM fdinfo (shared client ids deduplicated)."""

    def __init__(self, proc_root: str | Path = "/proc"):
        self.root = Path(proc_root)
        self._last: dict[int, tuple[int, float]] = {}

    def adapter_memory(self) -> list[dict[str, Any]]:
        return []  # HIP's free-memory figure on Linux already covers every process on the device

    def process_adapter_memory(self, pids: list[int]) -> dict[int, dict[str, float]]:
        return {}  # amdgpu does not move a process's VRAM to system RAM behind its back

    def process_metrics(self, pids: list[int]) -> dict[int, dict[str, float]]:
        results: dict[int, dict[str, float]] = {}
        for pid in sorted(set(pids)):
            fdinfo = self.root / str(pid) / "fdinfo"
            try:
                entries = list(fdinfo.iterdir())
            except OSError:
                continue
            clients: dict[Any, dict[str, Any]] = {}
            for entry in entries:
                text = _read(entry)
                parsed = parse_fdinfo(text) if text and "drm-driver" in text else None
                if parsed:
                    clients[parsed["client"]] = parsed
            if not clients:
                continue
            now = time.monotonic()
            engine_ns = sum(c["engine_ns"] for c in clients.values())
            utilization = 0.0
            previous = self._last.get(pid)
            if previous and now > previous[1]:
                utilization = (engine_ns - previous[0]) / ((now - previous[1]) * 1e9) * 100.0
            self._last[pid] = (engine_ns, now)
            results[pid] = {
                "dedicated_gb": sum(c["vram"] for c in clients.values()) / 1024**3,
                "shared_gb": sum(c["gtt"] for c in clients.values()) / 1024**3,
                "utilization_pct": max(0.0, min(100.0, utilization)),
            }
        return results


def _listening_inodes(root: Path, port: int) -> set[str]:
    inodes: set[str] = set()
    for name in ("tcp", "tcp6"):
        for line in (_read(root / "net" / name) or "").splitlines()[1:]:
            parts = line.split()
            if len(parts) > 9 and parts[3] == "0A" and int(parts[1].rsplit(":", 1)[-1], 16) == port:
                inodes.add(parts[9])
    return inodes


class LinuxProcessControl:
    server_binary_name = "llama-server"

    def __init__(self, proc_root: str | Path = "/proc"):
        self.root = Path(proc_root)

    def pid_listening_on(self, port: int) -> int | None:
        inodes = _listening_inodes(self.root, port)
        if not inodes:
            return None
        wanted = {f"socket:[{inode}]" for inode in inodes}
        for entry in self.root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                for fd in (entry / "fd").iterdir():
                    try:
                        if os.readlink(fd) in wanted:
                            return int(entry.name)
                    except OSError:
                        continue
            except OSError:
                continue
        return None

    def is_llama_server(self, pid: int) -> bool:
        try:
            exe = os.path.basename(os.readlink(self.root / str(pid) / "exe"))
        except OSError:
            exe = (_read(self.root / str(pid) / "comm") or "").strip()
        return exe.lower().startswith("llama-server")

    def _children(self, pid: int) -> list[int]:
        found = []
        for entry in self.root.iterdir():
            if entry.name.isdigit():
                text = _read(entry / "stat")
                if text:
                    rest = text[text.rfind(")") + 1:].split()
                    if len(rest) > 1 and rest[1] == str(pid):
                        found.append(int(entry.name))
        return found

    def _alive(self, pid: int) -> bool:
        text = _read(self.root / str(pid) / "stat")
        stat = parse_stat(text) if text else None
        return stat is not None and stat["state"] != ord("Z")

    def terminate_tree(self, pid: int, grace_s: float = 10.0) -> None:
        # Servers are spawned with start_new_session=True, so the pid is its own group leader and
        # killpg takes the whole tree. Otherwise fall back to walking children.
        try:
            group = os.getpgid(pid)
        except OSError:
            return
        targets = [pid] if group != pid else []
        if group != pid:
            targets += self._children(pid)

        def send(sig: int) -> None:
            try:
                if group == pid:
                    os.killpg(pid, sig)
                else:
                    for target in targets:
                        os.kill(target, sig)
            except OSError:
                pass

        send(signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline and self._alive(pid):
            time.sleep(0.1)
        if self._alive(pid):
            send(signal.SIGKILL)

    def spawn_kwargs(self) -> dict[str, Any]:
        return {"start_new_session": True}


def build(proc_root: str | Path = "/proc") -> HostOS:
    return HostOS(
        name="linux",
        process=LinuxProcessProbe(proc_root),
        system=LinuxSystemProbe(proc_root),
        gpu=LinuxGpuProbe(proc_root),
        control=LinuxProcessControl(proc_root),
    )
