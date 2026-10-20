"""On-demand model manager: GPU-only placement, LRU eviction of idle models, gateway routing."""
from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import app
from dual_gpu_setup import lmstudio
from dual_gpu_setup.config import LaneConfig, ModelConfig, load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "example.dual_gpu.toml"
R9700 = LaneConfig("r9700", "R9700", "R9700", 32, 8081)
N9070 = LaneConfig("9070xt", "9070 XT", "9070 XT", 16, 8082)


class FakeProcess:
    def __init__(self, ctx: int = 16384):
        self.ctx_size = ctx
        self.device = "fake"
        self.base_url = "http://127.0.0.1:1"
        self.pid = 1
        self.proc = SimpleNamespace(poll=lambda: None)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeStore:
    def __init__(self):
        self.deployments = []

    def begin_deployment(self, deployment_id, mode, profile):
        self.deployments.append((mode, profile))

    def update_deployment(self, *_args, **_kwargs):
        pass


def _manager(tmp_path, single_fits, multi_fits=(), drain=0.0):
    """single_fits: {(model, lane)}; multi_fits: models that only fit across both lanes."""
    manager = object.__new__(app.DeploymentManager)
    manager.config = SimpleNamespace(
        lanes=[R9700, N9070],
        policy=SimpleNamespace(drain_timeout_seconds=drain, vram_admission=False, auto_load_models=True,
                              kill_busy_ports=False),
        project=SimpleNamespace(start_timeout_seconds=5),
    )
    manager.data_dir = tmp_path
    manager.store = FakeStore()
    manager.sampler = SimpleNamespace(latest=lambda: {})
    manager.lock = threading.RLock()
    manager.transition_lock = threading.Lock()
    manager.state, manager.mode, manager.error = "idle", "idle", None
    manager.active_profile = manager.deployment_id = None
    manager.handles = {}
    manager.lease_guard = None
    manager._abort_drain = threading.Event()
    names = {name for name, _lane in single_fits} | set(multi_fits)
    manager.models_by_id = {name: ModelConfig(name, f"{name}.gguf") for name in names}
    lanes = {"r9700": R9700, "9070xt": N9070}

    def validate(mode, profile):
        raw = profile["model"]
        model = manager.models_by_id[raw["model"]]
        if mode == "single_gpu":
            if (raw["model"], raw["lane"]) not in single_fits:
                raise app.AppError(f"{raw['model']} does not fit {raw['lane']}")
            return [{"target": raw["lane"], "model": model, "lanes": [lanes[raw["lane"]]], "context": 16384, "multi": False}]
        if raw["model"] not in multi_fits:
            raise app.AppError(f"{raw['model']} does not fit both GPUs")
        return [{"target": "primary", "model": model, "lanes": [R9700, N9070], "context": 8192, "multi": True}]

    manager._validate_profile = validate
    manager.processes = []

    def make_process(_spec, _run_dir):
        process = FakeProcess()
        manager.processes.append(process)
        return process

    manager._make_process = make_process
    return manager


def _release(handle):
    """Turn the claim ensure_model made into a finished request (idle handle)."""
    with handle.serving(claimed=True):
        pass


# --- ServerHandle request tracking -----------------------------------------------------------

def test_claim_keeps_a_routed_handle_busy_until_it_has_served():
    handle = app.ServerHandle("r9700", ModelConfig("m", "m.gguf"), [R9700], FakeProcess())
    assert not handle.busy
    handle.claim()
    assert handle.busy
    with handle.serving(claimed=True):
        assert handle.busy and handle.inflight == 1
    assert not handle.busy


def test_unclaimed_serving_does_not_consume_another_requests_claim():
    handle = app.ServerHandle("r9700", ModelConfig("m", "m.gguf"), [R9700], FakeProcess())
    handle.claim()
    with handle.serving():  # e.g. /api/chat, which never claims
        pass
    assert handle.busy  # the gateway request's claim still protects it


def test_abandoned_claim_expires(monkeypatch):
    handle = app.ServerHandle("r9700", ModelConfig("m", "m.gguf"), [R9700], FakeProcess())
    handle.claim()
    monkeypatch.setattr(app.ServerHandle, "CLAIM_TTL_SECONDS", 0.0)
    time.sleep(0.01)
    assert not handle.busy


# --- placement and eviction --------------------------------------------------------------------

def test_first_model_goes_to_the_smallest_free_gpu_it_fits(tmp_path):
    manager = _manager(tmp_path, {("small", "9070xt"), ("small", "r9700")})
    target, handle = manager.ensure_model("small")
    assert target == "9070xt" and handle.busy and manager.processes[0].started
    assert manager.state == "ready" and manager.mode == "single_gpu"
    # The gateway loads in coding-agent mode; the profile keeps that so a reload reproduces it.
    assert manager.active_profile["models"] == [
        {"model": "small", "lane": "9070xt", "context_window": 16384, "workload_kind": "coding_agent"}]


def test_loaded_model_is_reused_not_reloaded(tmp_path):
    manager = _manager(tmp_path, {("small", "9070xt")})
    _target, first = manager.ensure_model("small")
    _release(first)
    _target, again = manager.ensure_model("small")
    assert again is first and len(manager.processes) == 1


def test_second_model_uses_the_free_gpu_instead_of_evicting(tmp_path):
    manager = _manager(tmp_path, {("a", "9070xt"), ("a", "r9700"), ("b", "9070xt"), ("b", "r9700")})
    _release(manager.ensure_model("a")[1])
    target, _handle = manager.ensure_model("b")
    assert target == "r9700"
    assert not any(process.stopped for process in manager.processes)
    assert manager.mode == "parallel_models"


def test_full_gpus_evict_the_least_recently_used_idle_model(tmp_path):
    fits = {(name, lane) for name in "abc" for lane in ("9070xt", "r9700")}
    manager = _manager(tmp_path, fits)
    _release(manager.ensure_model("a")[1])  # 9070xt
    _release(manager.ensure_model("b")[1])  # r9700
    manager.handles["9070xt"].last_used = 1.0  # a: least recently used
    manager.handles["r9700"].last_used = 2.0
    old_a = manager.handles["9070xt"].process

    target, handle = manager.ensure_model("c")

    assert target == "9070xt" and handle.model.name == "c"
    assert old_a.stopped and not manager.handles["r9700"].process.stopped
    assert {entry["model"] for entry in manager.active_profile["models"]} == {"b", "c"}


def test_a_busy_model_is_never_evicted_while_an_idle_one_exists(tmp_path):
    fits = {(name, lane) for name in "abc" for lane in ("9070xt", "r9700")}
    manager = _manager(tmp_path, fits)
    _release(manager.ensure_model("a")[1])
    _release(manager.ensure_model("b")[1])
    manager.handles["9070xt"].last_used = 1.0  # LRU, but serving a request
    manager.handles["r9700"].last_used = 2.0
    manager.handles["9070xt"].claim()

    target, _handle = manager.ensure_model("c")

    assert target == "r9700"
    assert not manager.handles["9070xt"].process.stopped


def test_all_gpus_busy_returns_503_and_stops_nothing(tmp_path):
    manager = _manager(tmp_path, {("a", "9070xt"), ("b", "r9700"), ("c", "9070xt"), ("c", "r9700")})
    manager.ensure_model("a")  # claims stay held: both models are serving
    manager.ensure_model("b")
    with pytest.raises(app.AppError) as raised:
        manager.ensure_model("c")
    assert raised.value.status == HTTPStatus.SERVICE_UNAVAILABLE
    assert not any(process.stopped for process in manager.processes)
    assert set(manager.handles) == {"9070xt", "r9700"}


def test_model_too_big_for_one_gpu_spans_both_after_evicting_them(tmp_path):
    manager = _manager(tmp_path, {("a", "9070xt"), ("b", "r9700")}, multi_fits={"huge"})
    _release(manager.ensure_model("a")[1])
    _release(manager.ensure_model("b")[1])

    target, handle = manager.ensure_model("huge")

    assert target == "primary" and [lane.key for lane in handle.lanes] == ["r9700", "9070xt"]
    assert all(process.stopped for process in manager.processes[:2])
    assert set(manager.handles) == {"primary"} and manager.mode == "single_large_model"
    assert manager.active_profile["model"]["model"] == "huge"


def test_single_gpu_model_replaces_a_split_model(tmp_path):
    manager = _manager(tmp_path, {("small", "9070xt")}, multi_fits={"huge"})
    _release(manager.ensure_model("huge")[1])
    target, _handle = manager.ensure_model("small")
    assert target == "9070xt" and set(manager.handles) == {"9070xt"}
    assert manager.processes[0].stopped


def test_requested_lane_is_honoured(tmp_path):
    manager = _manager(tmp_path, {("a", "9070xt"), ("a", "r9700")})
    target, _handle = manager.ensure_model("a", lane_key="r9700")
    assert target == "r9700"


def test_model_that_fits_nowhere_is_refused_without_touching_loaded_models(tmp_path):
    manager = _manager(tmp_path, {("a", "9070xt")})
    _release(manager.ensure_model("a")[1])
    manager.models_by_id["giant"] = ModelConfig("giant", "giant.gguf")
    with pytest.raises(app.AppError, match="cannot be placed"):
        manager.ensure_model("giant")
    assert not manager.processes[0].stopped


def test_unknown_model_is_reported():
    manager = _manager(Path("."), {("a", "9070xt")})
    assert manager.knows_model("a") and not manager.knows_model("gpt-4o")


# --- GPU-only admission ------------------------------------------------------------------------

def test_admission_refuses_a_model_the_gpu_cannot_hold(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "9070xt")})
    manager.config.policy.vram_admission = True
    manager._vram_need_gb = lambda spec: {"9070xt": 12.0}
    monkeypatch.setattr(app, "lane_free_vram_gb", lambda _config, _lanes: {"9070xt": 4.0})
    monkeypatch.setattr(app.time, "sleep", lambda _s: None)
    spec = {"target": "9070xt", "model": ModelConfig("a", "a.gguf"), "lanes": [N9070], "context": 8192, "multi": False}

    with pytest.raises(app.AppError, match="needs 12.0 GB, 4.0 GB free") as raised:
        manager._admit_vram(spec, wait_seconds=0)
    assert raised.value.status == HTTPStatus.SERVICE_UNAVAILABLE


def test_admission_passes_when_vram_frees_up(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "9070xt")})
    manager.config.policy.vram_admission = True
    manager._vram_need_gb = lambda spec: {"9070xt": 12.0}
    readings = iter([{"9070xt": 4.0}, {"9070xt": 14.0}])  # previous server still releasing
    monkeypatch.setattr(app, "lane_free_vram_gb", lambda _config, _lanes: next(readings))
    monkeypatch.setattr(app.time, "sleep", lambda _s: None)
    spec = {"target": "9070xt", "model": ModelConfig("a", "a.gguf"), "lanes": [N9070], "context": 8192, "multi": False}
    manager._admit_vram(spec, wait_seconds=60)


def test_planner_prefers_the_gpu_with_room_over_the_smaller_one(tmp_path, monkeypatch):
    """The 9070 XT is the smaller fit, but another app holds most of it: go to the R9700."""
    manager = _manager(tmp_path, {("m", "9070xt"), ("m", "r9700")})
    manager.config.policy.vram_admission = True
    manager._vram_need_gb = lambda spec: {lane.key: 6.0 for lane in spec["lanes"]}
    monkeypatch.setattr(app, "lane_free_vram_gb", lambda _config, _lanes: {"9070xt": 3.0, "r9700": 20.0})
    target, _handle = manager.ensure_model("m")
    assert target == "r9700"


def test_planner_counts_vram_our_idle_model_would_release(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "9070xt"), ("b", "9070xt")})
    _release(manager.ensure_model("a")[1])
    manager.config.policy.vram_admission = True
    manager._vram_need_gb = lambda spec: {lane.key: 6.0 for lane in spec["lanes"]}
    # 1 GB free now, but evicting "a" (6 GB) makes room: allowed.
    monkeypatch.setattr(app, "lane_free_vram_gb", lambda _config, _lanes: {"9070xt": 1.0, "r9700": 0.0})
    manager._admit_vram = lambda spec: None
    target, _handle = manager.ensure_model("b")
    assert target == "9070xt" and manager.processes[0].stopped


def test_planner_refuses_before_evicting_when_nothing_can_fit(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "9070xt"), ("big", "9070xt"), ("big", "r9700")})
    _release(manager.ensure_model("a")[1])
    manager.config.policy.vram_admission = True
    manager._vram_need_gb = lambda spec: {lane.key: (2.0 if spec["model"].name == "a" else 20.0) for lane in spec["lanes"]}
    monkeypatch.setattr(app, "lane_free_vram_gb", lambda _config, _lanes: {"9070xt": 3.0, "r9700": 10.0})
    with pytest.raises(app.AppError, match="even after unloading idle models") as raised:
        manager.ensure_model("big")
    assert raised.value.status == HTTPStatus.SERVICE_UNAVAILABLE
    assert not manager.processes[0].stopped and set(manager.handles) == {"9070xt"}


def test_unreadable_vram_does_not_block(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "9070xt")})
    manager.config.policy.vram_admission = True
    manager._vram_need_gb = lambda spec: {"9070xt": 12.0}

    def boom(_config, _lanes):
        raise SystemExit("llama-server not found")

    monkeypatch.setattr(app, "lane_free_vram_gb", boom)
    spec = {"target": "9070xt", "model": ModelConfig("a", "a.gguf"), "lanes": [N9070], "context": 8192, "multi": False}
    manager._admit_vram(spec)


def test_split_need_follows_tensor_split_in_launch_order(tmp_path, monkeypatch):
    manager = object.__new__(app.DeploymentManager)
    manager.config = load_config(EXAMPLE)
    model = ModelConfig("big", "big.gguf", size_gb=30.0, tensor_split="2,1")
    monkeypatch.setattr(app, "estimate_kv_cache_gb", lambda _path, _ctx: (2.0, ""))
    spec = {"target": "primary", "model": model, "lanes": [N9070, R9700], "context": 8192, "multi": True}
    need = manager._vram_need_gb(spec)
    total = 30.0 + 2.0 + manager.config.policy.vram_headroom_gb
    assert need["r9700"] == pytest.approx(total * 2 / 3) and need["9070xt"] == pytest.approx(total / 3)


def test_linux_free_vram_comes_from_the_device_listing(monkeypatch):
    devices = [("ROCm0", "AMD Radeon AI PRO R9700 (32624 MiB, 16000 MiB free)"),
               ("ROCm1", "AMD Radeon RX 9070 XT (16304 MiB, 4096 MiB free)"),
               ("ROCm2", "AMD Radeon(TM) Graphics (59564 MiB, 59413 MiB free)")]
    monkeypatch.setattr(lmstudio, "list_devices", lambda _config, _backend: devices)
    monkeypatch.setattr(lmstudio, "get_host", lambda: SimpleNamespace(gpu=SimpleNamespace(adapter_memory=lambda: [])))
    free = lmstudio.lane_free_vram_gb(SimpleNamespace(), [R9700, N9070])
    assert free == {"r9700": pytest.approx(15.625), "9070xt": 4.0}


def test_windows_free_vram_uses_the_os_view_not_hip(monkeypatch):
    """HIP on Windows reported an R9700 holding a 16.9 GB model as empty; the OS view does not."""
    adapters = [{"name": "AMD Radeon RX 9070 XT", "luid": "0x0_0x158fb", "total_gb": 15.8, "used_gb": 12.75},
                {"name": "AMD Radeon AI PRO R9700", "luid": "0x0_0x19cb5", "total_gb": 31.7, "used_gb": 16.85}]
    monkeypatch.setattr(lmstudio, "get_host", lambda: SimpleNamespace(gpu=SimpleNamespace(adapter_memory=lambda: adapters)))
    monkeypatch.setattr(lmstudio, "list_devices", Mock(side_effect=AssertionError("HIP view must not be used")))
    free = lmstudio.lane_free_vram_gb(SimpleNamespace(), [R9700, N9070])
    assert free == {"r9700": pytest.approx(14.85), "9070xt": pytest.approx(3.05)}


def test_adapter_usage_counters_are_summed_per_luid():
    from dual_gpu_setup.hostos.windows import luid_key, parse_adapter_usage

    output = ("luid_0x00000000_0x000158fb_phys_0 1073741824\n"
              "luid_0x00000000_0x000158fb_phys_1 1073741824\n"
              "luid_0x00000000_0x00019cb5_phys_0 0\nnoise line here\n")
    assert parse_adapter_usage(output) == {"0x00000000_0x000158fb": 2 * 1024**3, "0x00000000_0x00019cb5": 0.0}
    assert luid_key(0, 0x19CB5) == "0x00000000_0x00019cb5" and luid_key(-1, 1) == "0xffffffff_0x00000001"


# --- residency guard (Windows pages idle VRAM out to RAM) ---------------------------------------

class WarmProcess(FakeProcess):
    def __init__(self, pid, slot=1):
        super().__init__()
        self.pid = pid
        self._slot = slot

    def warm_slot(self):
        return self._slot


ADAPTERS = [{"name": "AMD Radeon AI PRO R9700", "luid": "L_R9700", "total_gb": 31.7, "used_gb": 0.0},
            {"name": "AMD Radeon RX 9070 XT", "luid": "L_9070", "total_gb": 15.8, "used_gb": 8.0}]


def _guarded(tmp_path, monkeypatch, usage, handles):
    manager = _manager(tmp_path, {("m", "r9700")})
    manager.handles = handles
    host = SimpleNamespace(gpu=SimpleNamespace(adapter_memory=lambda: ADAPTERS,
                                               process_adapter_memory=lambda pids: {p: usage[p] for p in pids if p in usage}))
    monkeypatch.setattr(app, "get_host", lambda: host)
    monkeypatch.setattr(app, "model_size_gb", lambda _config, model: 16.0)
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(app.urlrequest, "urlopen", lambda request, timeout: sent.append(
        (request.full_url, json.loads(request.data))) or Response())
    return manager, sent


def test_evicted_idle_model_is_paged_back_in_on_its_warm_up_slot(tmp_path, monkeypatch):
    handle = app.ServerHandle("r9700", ModelConfig("qwen", "q.gguf"), [R9700], WarmProcess(7864))
    manager, sent = _guarded(tmp_path, monkeypatch, {7864: {"L_R9700": 0.0, "L_9070": 0.18}}, {"r9700": handle})

    assert manager.check_residency() == ["r9700"]
    assert sent == [("http://127.0.0.1:1/completion", {"prompt": "ok", "n_predict": 1, "cache_prompt": False, "id_slot": 1})]
    assert handle.evictions == 1 and handle.rewarmed_at and handle.resident_gb == 0.0
    assert not handle.busy  # the warm-up's in-flight count was released


def test_resident_busy_or_unreported_models_are_left_alone(tmp_path, monkeypatch):
    resident = app.ServerHandle("r9700", ModelConfig("a", "a.gguf"), [R9700], WarmProcess(1))
    busy = app.ServerHandle("9070xt", ModelConfig("b", "b.gguf"), [N9070], WarmProcess(2))
    busy.claim()
    unreported = app.ServerHandle("x", ModelConfig("c", "c.gguf"), [R9700], WarmProcess(3))
    no_slot = app.ServerHandle("y", ModelConfig("d", "d.gguf"), [R9700], WarmProcess(4, slot=None))
    manager, sent = _guarded(tmp_path, monkeypatch, {1: {"L_R9700": 16.85}, 2: {"L_9070": 0.0}, 4: {"L_R9700": 0.0}},
                             {"r9700": resident, "9070xt": busy, "x": unreported, "y": no_slot})

    assert manager.check_residency() == [] and sent == []
    assert resident.resident_gb == 16.85 and resident.evictions == 0


def test_split_model_residency_counts_both_gpus(tmp_path, monkeypatch):
    split = app.ServerHandle("primary", ModelConfig("big", "big.gguf"), [R9700, N9070], WarmProcess(5))
    manager, sent = _guarded(tmp_path, monkeypatch, {5: {"L_R9700": 6.0, "L_9070": 3.0}}, {"primary": split})
    assert manager.check_residency() == []  # 9 GB resident >= half of 16 GB
    assert split.resident_gb == 9.0


def test_real_requests_are_pinned_to_slot_zero_only_with_a_warm_up_slot():
    warm = app.ServerHandle("r9700", ModelConfig("a", "a.gguf"), [R9700], WarmProcess(1))
    plain = app.ServerHandle("r9700", ModelConfig("a", "a.gguf"), [R9700], FakeProcess())
    assert warm.pin_serving_slot({"model": "a"}) == {"model": "a", "id_slot": 0}
    assert plain.pin_serving_slot({"model": "a"}) == {"model": "a"}


def test_process_adapter_usage_parsing():
    from dual_gpu_setup.hostos.windows import parse_process_adapter_usage

    output = ("pid_7864_luid_0x00000000_0x00019cb5_phys_0 18092408832\n"
              "pid_7864_luid_0x00000000_0x000158fb_phys_0 193273528\n"
              "pid_999_luid_0x00000000_0x00019cb5_phys_0 1\nluid_0x00000000_0x00019cb5_phys_0 5\n")
    usage = parse_process_adapter_usage(output, [7864])
    assert set(usage) == {7864}
    assert usage[7864]["0x00000000_0x00019cb5"] == pytest.approx(16.85, abs=0.01)


# --- gateway routing ---------------------------------------------------------------------------

def _handler(manager):
    handler = object.__new__(app.RequestHandler)
    handler.server = SimpleNamespace(app_state=SimpleNamespace(config=manager.config, manager=manager))
    handler.headers = {}
    return handler


def test_gateway_loads_a_named_catalog_model_on_demand():
    handle = object()
    manager = SimpleNamespace(config=load_config(EXAMPLE), knows_model=lambda name: name == "Qwen3-8B",
                              ensure_model=Mock(return_value=("9070xt", handle)))
    assert _handler(manager)._resolve_lane_handle("/v1/chat/completions", {"model": "Qwen3-8B"}) == ("9070xt", handle)
    manager.ensure_model.assert_called_once_with("Qwen3-8B", lane_key=None)


def test_gateway_lane_url_pins_the_on_demand_load():
    manager = SimpleNamespace(config=load_config(EXAMPLE), knows_model=lambda _name: True,
                              ensure_model=Mock(return_value=("r9700", object())))
    _handler(manager)._resolve_lane_handle("/v1/r9700/developer/chat/completions", {"model": "Qwen3-8B"})
    manager.ensure_model.assert_called_once_with("Qwen3-8B", lane_key="r9700")


def test_gateway_lane_aliases_keep_plain_routing():
    loaded = app.ServerHandle("r9700", ModelConfig("m", "m.gguf"), [R9700], FakeProcess())
    manager = SimpleNamespace(config=load_config(EXAMPLE), knows_model=lambda _name: True, ensure_model=Mock(),
                              lock=threading.RLock(), state="ready", handles={"r9700": loaded},
                              _mark_dead_lanes=lambda: None)
    target, handle = _handler(manager)._resolve_lane_handle("/v1/chat/completions", {"model": "dgpu:r9700"})
    assert (target, handle) == ("r9700", loaded) and loaded.busy
    manager.ensure_model.assert_not_called()


def test_gateway_auto_load_can_be_switched_off():
    config = load_config(EXAMPLE)
    config.policy.auto_load_models = False
    manager = SimpleNamespace(config=config, knows_model=lambda _name: True, ensure_model=Mock(),
                              lock=threading.RLock(), state="idle", handles={}, _mark_dead_lanes=lambda: None)
    with pytest.raises(app.AppError, match="Load a deployment"):
        _handler(manager)._resolve_lane_handle("/v1/chat/completions", {"model": "Qwen3-8B"})
    manager.ensure_model.assert_not_called()


# --- orphaned servers and per-lane unload ---------------------------------------------------------

class FakeControl:
    """The process-control slice of the host: which pid listens on which port, and what gets killed."""

    def __init__(self, listeners, llama=None):
        self.listeners = dict(listeners)
        self.llama = set(llama if llama is not None else listeners.values())
        self.terminated = []

    def pid_listening_on(self, port):
        return self.listeners.get(port)

    def is_llama_server(self, pid):
        return pid in self.llama

    def terminate_tree(self, pid, grace_s=10.0):
        self.terminated.append(pid)
        self.listeners = {port: owner for port, owner in self.listeners.items() if owner != pid}


def _with_host(monkeypatch, control):
    monkeypatch.setattr(app, "get_host", lambda: SimpleNamespace(control=control))
    return control


def _give_pid(handle, pid):
    handle.process.proc = SimpleNamespace(poll=lambda: None, pid=pid)


def test_orphans_are_unmanaged_llama_servers_on_lane_ports_only(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")})
    _release(manager.ensure_model("a")[1])
    _give_pid(manager.handles["r9700"], 100)
    _with_host(monkeypatch, FakeControl({8081: 100, 8082: 200}, llama={100, 200}))
    assert manager.lane_orphans() == [{"lane": "9070xt", "port": 8082, "pid": 200}]

    _with_host(monkeypatch, FakeControl({8081: 100, 8082: 300}, llama={100}))  # 300 is some other program
    assert manager.lane_orphans() == []


def test_orphan_scan_is_cached_between_status_polls(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")})
    scans = []
    manager.lane_orphans = lambda lanes=None: scans.append(1) or [{"lane": "r9700", "port": 8081, "pid": 5}]
    assert manager.orphans_cached() == manager.orphans_cached() and len(scans) == 1
    manager._orphan_cache = None
    manager.orphans_cached()
    assert len(scans) == 2


def test_unload_all_also_stops_orphans_and_goes_idle(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")})
    _release(manager.ensure_model("a")[1])
    _give_pid(manager.handles["r9700"], 100)
    control = _with_host(monkeypatch, FakeControl({8081: 100, 8082: 200}))
    process = manager.processes[0]

    def stop():  # a real server releases its port when it stops
        process.stopped = True
        control.listeners.pop(8081, None)

    process.stop = stop
    status = manager.unload()
    assert control.terminated == [200]  # the managed one is stopped through its handle, not as an orphan
    assert manager.processes[0].stopped and status["state"] == "idle" and manager.deployment_id is None


def test_unload_orphans_only_keeps_loaded_models(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")})
    _release(manager.ensure_model("a")[1])
    _give_pid(manager.handles["r9700"], 100)
    control = _with_host(monkeypatch, FakeControl({8081: 100, 8082: 200}))
    manager.unload(orphans_only=True)
    assert control.terminated == [200] and not manager.processes[0].stopped and set(manager.handles) == {"r9700"}


def test_unload_one_lane_keeps_the_other_and_updates_the_profile(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "9070xt"), ("b", "r9700")})
    _release(manager.ensure_model("a")[1])
    _release(manager.ensure_model("b")[1])
    _with_host(monkeypatch, FakeControl({}))
    status = manager.unload(lane="9070xt")
    assert [p.stopped for p in manager.processes] == [True, False]
    assert set(manager.handles) == {"r9700"} and status["state"] == "ready" and manager.mode == "single_gpu"
    assert [entry["lane"] for entry in manager.active_profile["models"]] == ["r9700"]

    status = manager.unload(lane="r9700")
    assert status["state"] == "idle" and manager.active_profile is None and manager.deployment_id is None


def test_unloading_a_lane_of_a_split_model_frees_both_gpus(tmp_path, monkeypatch):
    manager = _manager(tmp_path, set(), multi_fits={"huge"})
    _release(manager.ensure_model("huge")[1])
    _with_host(monkeypatch, FakeControl({}))
    manager.unload(lane="9070xt")
    assert manager.handles == {} and manager.processes[0].stopped and manager.state == "idle"


def test_unloading_an_unknown_lane_is_rejected(tmp_path):
    manager = _manager(tmp_path, {("a", "r9700")})
    manager._lane = lambda key: (_ for _ in ()).throw(app.AppError(f"Unknown lane: {key}"))
    with pytest.raises(app.AppError, match="Unknown lane"):
        manager.unload(lane="nope")


def test_loading_stops_a_leftover_server_before_the_vram_check(tmp_path, monkeypatch):
    """The leftover's VRAM would make the check refuse a load that is fine once it is gone."""
    manager = _manager(tmp_path, {("a", "r9700")})
    manager.config.policy.kill_busy_ports = True
    manager.config.policy.vram_admission = True
    control = _with_host(monkeypatch, FakeControl({8081: 777}))
    manager._vram_need_gb = lambda spec: {"r9700": 10.0}
    # free VRAM only appears once the leftover has been terminated
    monkeypatch.setattr(app, "lane_free_vram_gb", lambda _config, _lanes: {"r9700": 20.0 if control.terminated else 1.0})
    monkeypatch.setattr(app.time, "sleep", lambda _s: None)
    manager.ensure_model("a")
    assert control.terminated == [777] and manager.handles["r9700"].model.name == "a"


# --- Stop now: unload waits for running requests, force ends the wait ----------------------------------

def _hold_request(handle):
    """Keep a request 'in flight' on `handle` until the returned event is set."""
    started, release = threading.Event(), threading.Event()

    def hold():
        with handle.serving():
            started.set()
            release.wait(10)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert started.wait(2)
    return release, thread


def _run_unload(manager, **kwargs):
    done = {}
    thread = threading.Thread(target=lambda: done.update(result=manager.unload(**kwargs)), daemon=True)
    thread.start()
    return thread, done


def test_unload_waits_for_a_running_request_then_stops(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")}, drain=5.0)
    _with_host(monkeypatch, FakeControl({}))
    handle = manager.ensure_model("a")[1]
    _release(handle)
    release, holder = _hold_request(handle)
    thread, done = _run_unload(manager)
    time.sleep(0.4)
    assert thread.is_alive() and not manager.processes[0].stopped  # still waiting for the running request
    started = time.monotonic()
    release.set()
    thread.join(3)
    assert not thread.is_alive() and manager.processes[0].stopped and done["result"]["state"] == "idle"
    assert time.monotonic() - started < 2


def test_stop_now_ends_the_wait_of_an_unload_in_progress(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")}, drain=30.0)
    _with_host(monkeypatch, FakeControl({}))
    handle = manager.ensure_model("a")[1]
    _release(handle)
    release, holder = _hold_request(handle)
    thread, done = _run_unload(manager)
    time.sleep(0.4)
    assert thread.is_alive()

    started = time.monotonic()
    manager.unload(force=True)  # the lock is held by the waiting unload: this only cuts its wait short
    thread.join(3)
    assert not thread.is_alive() and manager.processes[0].stopped and done["result"]["state"] == "idle"
    assert time.monotonic() - started < 2.5  # not the 30 s drain
    release.set()


def test_force_on_a_quiet_service_stops_at_once_without_draining(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")}, drain=30.0)
    _with_host(monkeypatch, FakeControl({}))
    handle = manager.ensure_model("a")[1]
    _release(handle)
    release, holder = _hold_request(handle)
    started = time.monotonic()
    status = manager.unload(force=True)
    assert time.monotonic() - started < 2 and status["state"] == "idle" and manager.processes[0].stopped
    release.set()


def test_a_stale_stop_now_does_not_shorten_a_later_swap(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "9070xt"), ("b", "9070xt")}, drain=5.0)
    _with_host(monkeypatch, FakeControl({}))
    _release(manager.ensure_model("a")[1])
    manager._abort_drain.set()  # left over from an earlier "Stop now"
    manager.ensure_model("b")
    assert not manager._abort_drain.is_set()
    assert manager.unload()["state"] == "idle" and not manager._abort_drain.is_set()


def test_the_server_reports_how_many_requests_it_is_answering(tmp_path):
    handle = app.ServerHandle("r9700", ModelConfig("m", "m.gguf"), [R9700], FakeProcess())
    assert handle.public()["inflight"] == 0
    release, holder = _hold_request(handle)
    assert handle.public()["inflight"] == 1
    release.set()
    holder.join(2)
    assert handle.public()["inflight"] == 0


# --- releasing "my" model when a task is done ---------------------------------------------------------

def test_lane_unload_reports_whether_it_stopped_anything(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")})
    _with_host(monkeypatch, FakeControl({}))
    _release(manager.ensure_model("a")[1])
    assert manager.unload(lane="9070xt")["released"] is False and not manager.processes[0].stopped  # nothing there
    assert manager.unload(lane="r9700")["released"] is True and manager.processes[0].stopped
    assert manager.unload(lane="r9700")["released"] is False  # already empty: a harmless no-op


def test_unload_with_a_model_only_stops_that_model(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700"), ("b", "r9700")})
    _with_host(monkeypatch, FakeControl({}))
    _release(manager.ensure_model("b", lane_key="r9700")[1])
    status = manager.unload(lane="r9700", model="a")  # a client releasing "a" after someone loaded "b" there
    assert status["released"] is False and not manager.processes[0].stopped and "r9700" in manager.handles
    status = manager.unload(lane="r9700", model="b")
    assert status["released"] is True and manager.processes[0].stopped and status["state"] == "idle"


def test_unload_with_a_model_matches_the_name_the_caller_originally_used(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")})
    _with_host(monkeypatch, FakeControl({}))
    _release(manager.ensure_model("a")[1])
    manager.active_profile["models"][0]["model"] = "lmstudio:org/repo/a.gguf"  # the catalog id the client sent
    assert manager.unload(lane="r9700", model="lmstudio:org/repo/other.gguf")["released"] is False
    assert manager.unload(lane="r9700", model="lmstudio:org/repo/a.gguf")["released"] is True


def test_a_split_model_is_released_through_either_of_its_lanes(tmp_path, monkeypatch):
    manager = _manager(tmp_path, set(), multi_fits={"huge"})
    _with_host(monkeypatch, FakeControl({}))
    _release(manager.ensure_model("huge")[1])
    assert manager.unload(lane="9070xt", model="huge")["released"] is True and manager.handles == {}


# --- the service's own server is not an orphan while it is being started or stopped ----------------------

def test_no_orphans_are_reported_during_a_transition_and_the_scan_is_not_cached(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")})
    scans = []
    manager.lane_orphans = lambda lanes=None: scans.append(1) or [{"lane": "r9700", "port": 8081, "pid": 5}]
    manager.transition_lock.acquire()  # a load, swap or unload is running
    try:
        assert manager.orphans_cached() == [] and scans == []  # not even scanned
    finally:
        manager.transition_lock.release()
    assert manager.orphans_cached() == [{"lane": "r9700", "port": 8081, "pid": 5}] and len(scans) == 1  # nothing stale was cached


def test_a_server_draining_during_an_unload_is_never_reported_as_unmanaged(tmp_path, monkeypatch):
    """Reproduces the real bug: the handle leaves `handles` first, the process keeps running while it drains."""
    manager = _manager(tmp_path, {("a", "r9700")}, drain=30.0)
    control = _with_host(monkeypatch, FakeControl({8081: 100}))
    handle = manager.ensure_model("a")[1]
    _release(handle)
    _give_pid(handle, 100)
    release, holder = _hold_request(handle)
    thread, done = _run_unload(manager)
    time.sleep(0.4)
    assert thread.is_alive() and manager.handles == {}  # handle gone, server (pid 100) still listening and draining
    assert [o["pid"] for o in manager.lane_orphans()] == [100]  # a raw scan would flag it ...
    assert manager.orphans_cached() == []                       # ... but the status path does not
    manager.unload(force=True)
    thread.join(3)
    release.set()
    assert not thread.is_alive()


def test_a_second_unload_while_one_is_waiting_explains_the_wait(tmp_path, monkeypatch):
    manager = _manager(tmp_path, {("a", "r9700")}, drain=30.0)
    _with_host(monkeypatch, FakeControl({}))
    handle = manager.ensure_model("a")[1]
    _release(handle)
    release, holder = _hold_request(handle)
    thread, done = _run_unload(manager)
    time.sleep(0.4)
    manager.config.policy.drain_timeout_seconds = 30.0
    with pytest.raises(app.AppError, match="already in progress.*waits up to 30 s.*Use Stop now") as raised:
        manager.unload()
    assert raised.value.status == HTTPStatus.CONFLICT
    manager.unload(force=True)
    thread.join(3)
    release.set()
