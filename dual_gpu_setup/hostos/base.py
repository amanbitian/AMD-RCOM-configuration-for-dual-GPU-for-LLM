"""Host-OS abstraction: process/system/GPU probes and process control behind one interface.

Inference and training code call these instead of touching ctypes, PowerShell, netstat or /proc
directly, so the same behaviour is available on Windows and Linux.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """A PID alone can be recycled. The start token (creation time on Windows, `starttime`
    ticks on Linux) makes the pair unique for the life of the machine's uptime."""

    pid: int
    start_token: int
    exe: str | None = None


class ProcessProbe(Protocol):
    def metrics(self, pid: int, now: float) -> dict[str, Any]: ...

    def identity(self, pid: int) -> ProcessIdentity | None: ...

    def forget(self, pid: int) -> None: ...


class SystemProbe(Protocol):
    def metrics(self) -> dict[str, Any]: ...


class GpuProbe(Protocol):
    def process_metrics(self, pids: list[int]) -> dict[int, dict[str, float]]: ...

    def adapter_memory(self) -> list[dict[str, Any]]:
        """[{name, luid, total_gb, used_gb}] per GPU as the OS accounts VRAM, or [] when the GPU
        runtime's own free-memory figure is already device-wide (Linux)."""
        ...

    def process_adapter_memory(self, pids: list[int]) -> dict[int, dict[str, float]]:
        """{pid: {luid: resident GB}} -- empty where the OS never pages VRAM out (Linux)."""
        ...


class ProcessControl(Protocol):
    server_binary_name: str

    def pid_listening_on(self, port: int) -> int | None: ...

    def is_llama_server(self, pid: int) -> bool: ...

    def terminate_tree(self, pid: int, grace_s: float = 10.0) -> None: ...

    def spawn_kwargs(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class HostOS:
    name: str
    process: ProcessProbe
    system: SystemProbe
    gpu: GpuProbe
    control: ProcessControl

    def terminate_verified(self, identity: ProcessIdentity, grace_s: float = 10.0) -> bool:
        """Terminate only if the PID still names the same process (same start token).

        Returns False without touching anything when the PID was recycled or has exited, so an
        expired lease can never take down an unrelated process."""
        current = self.process.identity(identity.pid)
        if current is None or current.start_token != identity.start_token:
            return False
        self.control.terminate_tree(identity.pid, grace_s)
        return True
