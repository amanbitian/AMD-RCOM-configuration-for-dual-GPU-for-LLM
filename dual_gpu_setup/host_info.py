"""Small, dependency-free host metadata and process-launch helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import Any


def os_info() -> dict[str, str]:
    """Return stable, non-identifying OS metadata suitable for persisted run logs."""
    info = {
        "system": platform.system() or sys.platform,
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }
    if sys.platform.startswith("linux"):
        try:
            release = platform.freedesktop_os_release()
        except OSError:
            release = {}
        if release:
            info.update({
                "distribution": release.get("PRETTY_NAME") or release.get("NAME", ""),
                "distribution_id": release.get("ID", ""),
                "distribution_version": release.get("VERSION_ID", ""),
            })
    return {key: value for key, value in info.items() if value}


def os_log_line() -> str:
    """Compact JSON used in plain-text log headers."""
    return json.dumps(os_info(), ensure_ascii=False, sort_keys=True)


def default_config_name(platform_name: str | None = None) -> str:
    """Use the native llama.cpp example automatically on Linux."""
    platform_name = platform_name or sys.platform
    return "example.dual_gpu.ubuntu.toml" if platform_name.startswith("linux") else "example.dual_gpu.toml"


def detached_process_kwargs() -> dict[str, Any]:
    """Return Popen options for a child that must survive its short-lived launcher."""
    if sys.platform == "win32":
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            "close_fds": True,
        }
    # A new session prevents terminal SIGHUP/process-group signals from taking down an Ubuntu
    # service when launch_service.py (or the old process during a UI restart) exits.
    return {"start_new_session": True, "close_fds": True}
