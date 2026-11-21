"""Remembers which llama-server processes THIS service started, so a restart can tell a leftover of its own
(safe to stop) from a server someone else started on a lane port (never stopped automatically).

Entries carry the OS start token, so a recycled PID is never mistaken for one of ours. The file is rewritten
atomically; a corrupt or missing file just means "nothing is known to be ours".
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

from dual_gpu_setup.hostos import get_host


class ProcessRegistry:
    def __init__(self, path: Path | str, identity: Callable[[int], Any] | None = None):
        self.path = Path(path)
        self._identity = identity or (lambda pid: get_host().process.identity(pid))
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, entries: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        os.replace(temporary, self.path)

    def record(self, pid: int, port: int, model: str = "", lanes: list[str] | None = None) -> None:
        identity = self._identity(pid)
        with self._lock:
            entries = self._load()
            entries[str(pid)] = {"pid": pid, "port": port, "model": model, "lanes": lanes or [],
                                 "start_token": None if identity is None else identity.start_token}
            self._save(entries)

    def forget(self, pid: int) -> None:
        with self._lock:
            entries = self._load()
            if entries.pop(str(pid), None) is not None:
                self._save(entries)

    def owns(self, pid: int) -> bool:
        entry = self._load().get(str(pid))
        if not entry or entry.get("start_token") is None:
            return False
        current = self._identity(pid)
        return current is not None and current.start_token == entry["start_token"]

    def entries(self) -> list[dict[str, Any]]:
        return list(self._load().values())
