"""Crash recovery policy for llama-server processes: restart with back-off, give up on a crash loop.

A crashed server used to stay down until someone noticed. Restarting is opt-in (`policy.auto_restart`), always
on the SAME lane with the same model (a restart never places a model on a GPU nobody chose), waits longer after
each crash, and stops trying when a server keeps dying (`auto_restart_max` crashes within `auto_restart_window_s`),
so a model that cannot load does not spin the GPU forever. Crashes are recorded either way.
"""

from __future__ import annotations

import time
from typing import Callable

BACKOFF_S = (5.0, 15.0, 45.0, 120.0)


class CrashTracker:
    def __init__(self, max_crashes: int = 3, window_s: float = 600.0, clock: Callable[[], float] = time.time):
        self.max_crashes = max(1, int(max_crashes))
        self.window_s = float(window_s)
        self.clock = clock
        self._crashes: dict[str, list[float]] = {}

    def record_crash(self, target: str) -> tuple[str, float]:
        """('restart', delay_seconds) or ('give_up', 0.0) after this crash of `target`."""
        now = self.clock()
        recent = [t for t in self._crashes.get(target, []) if now - t < self.window_s] + [now]
        self._crashes[target] = recent
        if len(recent) > self.max_crashes:
            return "give_up", 0.0
        return "restart", BACKOFF_S[min(len(recent), len(BACKOFF_S)) - 1]

    def reset(self, target: str) -> None:
        self._crashes.pop(target, None)

    def count(self, target: str) -> int:
        now = self.clock()
        return len([t for t in self._crashes.get(target, []) if now - t < self.window_s])
