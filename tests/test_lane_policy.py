from __future__ import annotations

from types import SimpleNamespace
import threading

import pytest

import app
from dual_gpu_setup.config import LaneConfig, ModelConfig


def _manager(fit_by_model_and_lane):
    manager = object.__new__(app.DeploymentManager)
    manager.config = SimpleNamespace(lanes=[
        LaneConfig("r9700", "R9700", "R9700", 32, 8081),
        LaneConfig("9070xt", "9070 XT", "9070 XT", 16, 8082),
    ])
    manager.check_fit = lambda model, lane, context, input_tokens, max_output_tokens: (
        fit_by_model_and_lane[(model, lane)]
    )
    manager.deploy = lambda profile: profile
    return manager


def test_parallel_rejects_two_models_that_both_fit_9070():
    manager = _manager({
        ("small-a", "9070xt"): {"fits": True, "reason": None},
        ("small-b", "9070xt"): {"fits": True, "reason": None},
    })
    with pytest.raises(app.AppError, match="Run these models sequentially"):
        manager.deploy_parallel_for_client(["small-a", "small-b"], 51200)


def test_parallel_pairs_9070_fit_with_9700_required_model():
    manager = _manager({
        ("small", "9070xt"): {"fits": True, "reason": None},
        ("large", "9070xt"): {"fits": False, "reason": "too large"},
        ("large", "r9700"): {"fits": True, "reason": None},
    })
    profile = manager.deploy_parallel_for_client(["large", "small"], 51200)
    assert profile == {
        "mode": "parallel_models",
        "models": [
            {"model": "large", "lane": "r9700", "context_window": 51200},
            {"model": "small", "lane": "9070xt", "context_window": 51200},
        ],
    }


def test_auto_lane_uses_full_context_aware_fit_verdict():
    manager = _manager({
        ("context-heavy", "9070xt"): {"fits": False, "reason": "KV cache"},
        ("context-heavy", "r9700"): {"fits": True, "reason": None},
    })
    lane = manager._auto_lane_for_model("context-heavy", 51200)
    assert lane.key == "r9700"


def test_auto_lane_keeps_every_fitting_model_on_9070():
    manager = _manager({
        ("small", "9070xt"): {"fits": True, "reason": None},
    })
    lane = manager._auto_lane_for_model("small", 51200)
    assert lane.key == "9070xt"


def test_lane_replace_preserves_other_gpu(tmp_path):
    class FakeProcess:
        def __init__(self, ctx=51200):
            self.ctx_size = ctx
            self.device = "fake"
            self.base_url = "http://127.0.0.1:1"
            self.pid = 1
            self.proc = None
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    class FakeStore:
        def begin_deployment(self, *_args):
            pass

        def update_deployment(self, *_args, **_kwargs):
            pass

    lanes = [
        LaneConfig("r9700", "R9700", "R9700", 32, 8081),
        LaneConfig("9070xt", "9070 XT", "9070 XT", 16, 8082),
    ]
    old_large_process = FakeProcess()
    preserved_process = FakeProcess()
    old_large = app.ServerHandle(
        "r9700", ModelConfig(name="old-large", path="unused.gguf"), [lanes[0]], old_large_process
    )
    preserved = app.ServerHandle(
        "9070xt", ModelConfig(name="still-running", path="unused.gguf"), [lanes[1]], preserved_process
    )
    new_model = ModelConfig(name="new-large", path="unused.gguf")
    manager = object.__new__(app.DeploymentManager)
    manager.config = SimpleNamespace(lanes=lanes)
    manager.data_dir = tmp_path
    manager.store = FakeStore()
    manager.sampler = SimpleNamespace(latest=lambda: {})
    manager.lock = threading.RLock()
    manager.transition_lock = threading.Lock()
    manager.state = "ready"
    manager.mode = "parallel_models"
    manager.error = None
    manager.active_profile = None
    manager.deployment_id = "old"
    manager.handles = {"r9700": old_large, "9070xt": preserved}
    spec = {"target": "r9700", "model": new_model, "lanes": [lanes[0]], "context": 51200, "multi": False}
    manager._validate_profile = lambda _mode, _profile: [spec]
    new_process = FakeProcess()
    manager._make_process = lambda _spec, _run_dir: new_process

    status = manager.deploy_lane_for_client("new-large", "r9700", 51200)

    assert old_large_process.stopped is True
    assert preserved_process.stopped is False
    assert new_process.started is True
    assert manager.handles["9070xt"] is preserved
    assert manager.handles["r9700"].model.name == "new-large"
    assert {server["target"] for server in status["servers"]} == {"9070xt", "r9700"}


def _lane_replace_manager(tmp_path, active_profile, handles_spec, new_model_name):
    """Build a DeploymentManager wired to fake processes, ready for one lane replacement."""
    class FakeProcess:
        def __init__(self, ctx=51200):
            self.ctx_size = ctx
            self.device = "fake"
            self.base_url = "http://127.0.0.1:1"
            self.pid = 1
            self.proc = None
            self.stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    class FakeStore:
        def begin_deployment(self, *_args):
            pass

        def update_deployment(self, *_args, **_kwargs):
            pass

    lanes = [
        LaneConfig("r9700", "R9700", "R9700", 32, 8081),
        LaneConfig("9070xt", "9070 XT", "9070 XT", 16, 8082),
    ]
    manager = object.__new__(app.DeploymentManager)
    manager.config = SimpleNamespace(lanes=lanes)
    manager.data_dir = tmp_path
    manager.store = FakeStore()
    manager.sampler = SimpleNamespace(latest=lambda: {})
    manager.lock = threading.RLock()
    manager.transition_lock = threading.Lock()
    manager.state = "ready"
    manager.mode = "parallel_models" if len(handles_spec) > 1 else "single_gpu"
    manager.error = None
    manager.active_profile = active_profile
    manager.deployment_id = "old"
    lane_by_key = {lane.key: lane for lane in lanes}
    manager.handles = {
        key: app.ServerHandle(
            key, ModelConfig(name=resolved, path="unused.gguf"), [lane_by_key[key]], FakeProcess()
        )
        for key, resolved in handles_spec.items()
    }
    spec = {
        "target": "r9700",
        "model": ModelConfig(name=new_model_name, path="unused.gguf"),
        "lanes": [lanes[0]],
        "context": 51200,
        "multi": False,
    }
    manager._validate_profile = lambda _mode, _profile: [spec]
    manager._make_process = lambda _spec, _run_dir: FakeProcess()
    return manager


def test_lane_replace_profile_echoes_caller_model_strings(tmp_path):
    """profile.models[] is the documented join key, so it must carry the caller's exact
    request strings -- for the replaced lane AND for the lane left running."""
    requested_small = "lmstudio:lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf"
    requested_large = "lmstudio:some-org/next-large-GGUF/next-large-Q4_K_M.gguf"
    manager = _lane_replace_manager(
        tmp_path,
        active_profile={
            "mode": "parallel_models",
            "models": [
                {"model": "lmstudio:old/large-GGUF/large-Q4_K_M.gguf", "lane": "r9700"},
                {"model": requested_small, "lane": "9070xt"},
            ],
        },
        # Both handles carry resolved SHORT display names, which never equal a catalog id.
        handles_spec={"r9700": "large-Q4_K_M", "9070xt": "gemma-3-4b-it-Q4_K_M"},
        new_model_name="next-large-Q4_K_M",
    )

    manager.deploy_lane_for_client(requested_large, "r9700", 51200)

    by_lane = {entry["lane"]: entry["model"] for entry in manager.active_profile["models"]}
    assert by_lane["r9700"] == requested_large, "replaced lane must echo this call's input"
    assert by_lane["9070xt"] == requested_small, "preserved lane must keep its original input"


def test_lane_replace_from_idle_needs_no_prior_deploy(tmp_path):
    """A per-lane queue starts both lanes with deploy_lane and never calls deploy_parallel."""
    requested = "lmstudio:some-org/first-GGUF/first-Q4_K_M.gguf"
    manager = _lane_replace_manager(
        tmp_path, active_profile=None, handles_spec={}, new_model_name="first-Q4_K_M"
    )
    manager.state = "idle"
    manager.mode = "idle"

    status = manager.deploy_lane_for_client(requested, "r9700", 51200)

    assert manager.state == "ready"
    assert manager.mode == "single_gpu"
    assert status["profile"]["models"] == [
        {"model": requested, "lane": "r9700", "context_window": 51200}
    ]


def test_lane_replace_falls_back_to_resolved_name_for_unknown_lane_history(tmp_path):
    """A lane loaded outside the client API (no recorded request string) still appears."""
    manager = _lane_replace_manager(
        tmp_path,
        active_profile={"mode": "parallel_models", "models": [{"model": "x"}]},  # no lane key
        handles_spec={"r9700": "old-large", "9070xt": "mystery-model"},
        new_model_name="next-large",
    )

    manager.deploy_lane_for_client("next-large-request", "r9700", 51200)

    by_lane = {entry["lane"]: entry["model"] for entry in manager.active_profile["models"]}
    assert by_lane["r9700"] == "next-large-request"
    assert by_lane["9070xt"] == "mystery-model"
