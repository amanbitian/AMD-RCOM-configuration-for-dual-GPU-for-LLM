from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "example.dual_gpu.toml"


class _FakeProc:
    def __init__(self, alive: bool):
        self._alive = alive

    def poll(self):
        return None if self._alive else 1


def _handle(key: str, alive: bool):
    return SimpleNamespace(
        process=SimpleNamespace(proc=_FakeProc(alive)),
        lanes=[key],
        public=lambda: {"target": key, "alive": alive},
    )


def _ready_manager(tmp_path: Path, handles: dict):
    config = app.load_config(EXAMPLE_CONFIG)
    store = app.RunStore(tmp_path / "runs.sqlite3")
    manager = app.DeploymentManager(config, tmp_path / "data", store)
    manager.state = "ready"
    manager.mode = "parallel_models"
    manager.error = None
    manager.handles = handles
    return manager


def test_ready_deployment_with_exited_lane_reports_degraded(tmp_path: Path):
    manager = _ready_manager(tmp_path, {"r9700": _handle("r9700", True), "9070xt": _handle("9070xt", False)})

    status = manager.status()

    assert status["state"] == "degraded"
    assert "9070xt" in status["error"]
    assert "load the deployment again" in status["error"]


def test_ready_deployment_with_all_lanes_alive_stays_ready(tmp_path: Path):
    manager = _ready_manager(tmp_path, {"r9700": _handle("r9700", True), "9070xt": _handle("9070xt", True)})

    assert manager.status()["state"] == "ready"


def test_degraded_state_is_not_overwritten_back_to_ready(tmp_path: Path):
    manager = _ready_manager(tmp_path, {"r9700": _handle("r9700", False)})
    manager.status()

    manager.handles["r9700"] = _handle("r9700", True)
    # Once degraded, only a new deploy may make the deployment ready again.
    assert manager.status()["state"] == "degraded"
