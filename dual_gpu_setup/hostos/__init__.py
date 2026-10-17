"""Host-OS abstraction. `get_host()` returns the implementation for the running platform."""

from __future__ import annotations

import sys
import threading

from .base import HostOS, ProcessIdentity

__all__ = ["HostOS", "ProcessIdentity", "get_host", "set_host"]

_host: HostOS | None = None
_lock = threading.Lock()


def get_host() -> HostOS:
    global _host
    with _lock:
        if _host is None:
            if sys.platform == "win32":
                from . import windows as impl
            else:
                from . import linux as impl
            _host = impl.build()
        return _host


def set_host(host: HostOS | None) -> None:
    """Install a specific implementation (tests, or a fake /proc root)."""
    global _host
    with _lock:
        _host = host
