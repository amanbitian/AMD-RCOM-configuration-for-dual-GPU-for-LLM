from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dual_gpu_setup import lmstudio_gpus as gpus

ROCM_KEY = "llama.cpp-win-x86_64-amd-rocm-avx2"
VULKAN_KEY = "llama.cpp-win-x86_64-vulkan-avx2"
DEVICES = [
    ("ROCm0", "AMD Radeon AI PRO R9700 (32624 MiB, 32472 MiB free)"),
    ("ROCm1", "AMD Radeon RX 9070 XT (16304 MiB, 16152 MiB free)"),
    ("ROCm2", "AMD Radeon(TM) Graphics (59564 MiB, 59413 MiB free)"),
]
ORIGINAL = {
    "json": [
        [VULKAN_KEY, {"fields": [{"key": gpus.SPLIT_KEY, "value": {
            "strategy": "evenly", "disabledGpus": [0, 1], "priority": [], "customRatio": []}}]}],
        [ROCM_KEY, {"fields": [
            {"key": gpus.SPLIT_KEY, "value": {"strategy": "evenly", "disabledGpus": [], "priority": [], "customRatio": []}},
            {"key": "llm.load.offloadKVCacheToGpu", "value": True},
        ]}],
    ],
    "meta": {"values": ["map"]},
}


@pytest.fixture
def lm_home(tmp_path, monkeypatch):
    monkeypatch.setattr(gpus, "list_devices", lambda config, backend: list(DEVICES))
    monkeypatch.setattr(gpus, "runtime_dir", lambda config, backend: Path(f"{ROCM_KEY}-2.55.0"))
    monkeypatch.setattr(gpus, "lm_studio_state", lambda config: {"reachable": False, "models": []})
    config = SimpleNamespace(discovery=SimpleNamespace(lmstudio_home=str(tmp_path)))
    path = gpus.hardware_config_path(config)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(ORIGINAL), encoding="utf-8")
    return config, path


def _disabled(path: Path, key: str = ROCM_KEY) -> list[int]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    entry = next(e for e in doc["json"] if e[0] == key)
    return next(f for f in entry[1]["fields"] if f["key"] == gpus.SPLIT_KEY)["value"]["disabledGpus"]


def test_get_reports_devices_in_lm_studio_order(lm_home):
    config, _ = lm_home
    view = gpus.get_selection(config)
    assert [(d["index"], d["name"], d["enabled"]) for d in view["devices"]] == [
        (0, "AMD Radeon AI PRO R9700", True), (1, "AMD Radeon RX 9070 XT", True), (2, "AMD Radeon(TM) Graphics", True)]
    assert view["enabled"] == [0, 1, 2] and view["disabled_gpus"] == []


@pytest.mark.parametrize("selector", [[0], ["R9700"], ["rocm0"], ["0"], ["r9700", 0]])
def test_selecting_the_r9700_disables_the_other_two(lm_home, selector):
    config, path = lm_home
    result = gpus.set_selection(config, selector)
    assert _disabled(path) == [1, 2]
    assert result["enabled"] == [0] and result["previous_disabled_gpus"] == []


def test_only_the_disabled_list_changes_and_the_first_original_is_kept(lm_home):
    config, path = lm_home
    gpus.set_selection(config, [0])
    gpus.set_selection(config, [1])
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert _disabled(path) == [0, 2]
    assert _disabled(path, VULKAN_KEY) == [0, 1]  # the other backend is untouched
    rocm = next(e for e in doc["json"] if e[0] == ROCM_KEY)
    assert {"key": "llm.load.offloadKVCacheToGpu", "value": True} in rocm[1]["fields"]
    assert doc["meta"] == ORIGINAL["meta"]
    assert json.loads(path.with_name(path.name + ".dgpu.bak").read_text(encoding="utf-8")) == ORIGINAL


def test_all_re_enables_every_gpu(lm_home):
    config, path = lm_home
    gpus.set_selection(config, [0])
    assert gpus.set_selection(config, "all")["enabled"] == [0, 1, 2]
    assert _disabled(path) == []


@pytest.mark.parametrize("selector,match", [
    (None, "non-empty list"), ([], "non-empty list"), ([7], "matches no GPU"),
    (["radeon"], "more than one GPU"), ([True], "Unusable"), (["nvidia"], "matches no GPU"),
])
def test_bad_selectors_are_rejected_without_writing(lm_home, selector, match):
    config, path = lm_home
    before = path.read_text(encoding="utf-8")
    with pytest.raises(gpus.GpuSelectionError, match=match):
        gpus.set_selection(config, selector)
    assert path.read_text(encoding="utf-8") == before


def test_missing_backend_entry_is_created_from_the_installed_runtime(lm_home):
    config, path = lm_home
    path.write_text(json.dumps({"json": [], "meta": {"values": ["map"]}}), encoding="utf-8")
    gpus.set_selection(config, [0])
    assert _disabled(path) == [1, 2]


def test_unreadable_or_foreign_config_is_never_overwritten(lm_home):
    config, path = lm_home
    for text in ("not json", json.dumps({"json": "nope"})):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(gpus.GpuSelectionError) as caught:
            gpus.set_selection(config, [0])
        assert caught.value.status == 409 and path.read_text(encoding="utf-8") == text


def test_unknown_backend_is_rejected(lm_home):
    config, _ = lm_home
    with pytest.raises(gpus.GpuSelectionError, match="Unsupported backend"):
        gpus.set_selection(config, [0], backend="cuda")


# ---------------------------------------------------------------- HTTP API


def test_api_reads_and_sets_the_selection(live_app, lm_home):
    config, path = lm_home
    live = live_app()
    object.__setattr__(live.state.config.discovery, "lmstudio_home", config.discovery.lmstudio_home)

    status, body = live.get("/api/lmstudio/gpus")
    assert status == 200 and body["enabled"] == [0, 1, 2]

    status, body = live.post("/api/lmstudio/gpus", {"gpus": ["R9700"]})
    assert status == 200 and body["enabled"] == [0] and _disabled(path) == [1, 2]
    assert live.get("/api/lmstudio/gpus")[1]["disabled_gpus"] == [1, 2]

    status, body = live.post("/api/lmstudio/gpus", {"gpus": ["nvidia"]})
    assert status == 400 and "matches no GPU" in body["error"]
    assert live.post("/api/lmstudio/gpus", {})[0] == 400
    assert live.get("/api/lmstudio/gpus?backend=cuda")[0] == 400


def test_content_type_guard_blocks_blind_form_posts(live_app, lm_home):
    from urllib import error as urlerror, request as urlrequest

    config, path = lm_home
    live = live_app()
    object.__setattr__(live.state.config.discovery, "lmstudio_home", config.discovery.lmstudio_home)
    before = path.read_text(encoding="utf-8")
    for route in ("/api/lmstudio/gpus", "/api/lmstudio/release"):
        req = urlrequest.Request(live.base + route, data=b'{"gpus": [0]}', method="POST",
                                 headers={"Content-Type": "text/plain"})
        with pytest.raises(urlerror.HTTPError) as caught:
            urlrequest.urlopen(req, timeout=10)
        assert caught.value.code == 415
    assert path.read_text(encoding="utf-8") == before


# ------------------------------------------------------------------ release


class FakeLms:
    def __init__(self, models, fail=()):
        self.models, self.calls, self.fail = models, [], set(fail)

    def state(self, config):
        return {"reachable": True, "models": [dict(m) for m in self.models]}

    def run(self, config, *args, timeout=60):
        self.calls.append(args)
        code = 1 if args[0] in self.fail else 0
        if args[:2] == ("unload", "--all") and code == 0:
            self.models = []
        return SimpleNamespace(returncode=code, stdout="", stderr="boom" if code else "")


def _model(status="idle", queued=0):
    return {"identifier": "qwen", "status": status, "queued": queued, "size_bytes": 1, "context_length": 1}


def _fake(monkeypatch, fake):
    monkeypatch.setattr(gpus, "lm_studio_state", fake.state)
    monkeypatch.setattr(gpus, "_lms", fake.run)


def test_release_unloads_idle_models_and_can_stop_the_server(monkeypatch):
    fake = FakeLms([_model()])
    _fake(monkeypatch, fake)
    result = gpus.release(SimpleNamespace(), stop_server=True)
    assert result["unloaded"] == ["qwen"] and result["server_stopped"] and result["lm_studio"]["models"] == []
    assert fake.calls == [("unload", "--all"), ("server", "stop")]


@pytest.mark.parametrize("model", [_model("generating"), _model("idle", queued=2)])
def test_release_refuses_a_busy_model_unless_forced(monkeypatch, model):
    fake = FakeLms([model])
    _fake(monkeypatch, fake)
    with pytest.raises(gpus.GpuSelectionError, match="busy") as caught:
        gpus.release(SimpleNamespace())
    assert caught.value.status == 409 and fake.calls == []
    assert gpus.release(SimpleNamespace(), force=True)["unloaded"] == ["qwen"]


def test_release_with_nothing_loaded_runs_no_unload(monkeypatch):
    fake = FakeLms([])
    _fake(monkeypatch, fake)
    assert gpus.release(SimpleNamespace())["unloaded"] == []
    assert fake.calls == []


def test_release_surfaces_a_failed_unload(monkeypatch):
    _fake(monkeypatch, FakeLms([_model()], fail={"unload"}))
    with pytest.raises(gpus.GpuSelectionError, match="unload --all failed"):
        gpus.release(SimpleNamespace())


def test_api_release_endpoint(live_app, lm_home, monkeypatch):
    config, _ = lm_home
    fake = FakeLms([_model("generating")])
    _fake(monkeypatch, fake)
    live = live_app()
    object.__setattr__(live.state.config.discovery, "lmstudio_home", config.discovery.lmstudio_home)
    assert live.post("/api/lmstudio/release", {})[0] == 409
    status, body = live.post("/api/lmstudio/release", {"force": True})
    assert status == 200 and body["unloaded"] == ["qwen"]
    assert live.post("/api/lmstudio/release", {"force": "yes"})[0] == 200  # non-true is not force; nothing left to unload
