"""Training telemetry: tracker client, run store, GPU sampler, checkpoint manifests.

Names are resolved lazily so importing this package (e.g. from the dashboard server) never
imports `transformers`/`torch`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["HFTrackerCallback", "LeaseConflictError", "TrainingTracker", "find_latest_valid", "validate_manifest", "write_manifest"]

_LOCATIONS = {
    "HFTrackerCallback": ".hf_callback",
    "TrainingTracker": ".tracker",
    "LeaseConflictError": ".tracker",
    "find_latest_valid": ".checkpoints",
    "validate_manifest": ".checkpoints",
    "write_manifest": ".checkpoints",
}


def __getattr__(name: str) -> Any:
    if name in _LOCATIONS:
        import importlib

        return getattr(importlib.import_module(_LOCATIONS[name], __name__), name)
    raise AttributeError(name)
