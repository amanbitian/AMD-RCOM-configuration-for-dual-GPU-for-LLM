from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib import error, request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402
from dual_gpu_setup.hostos import ProcessIdentity  # noqa: E402
from dual_gpu_setup.hostos.base import HostOS  # noqa: E402
from dual_gpu_setup.scheduling.leases import LeaseError, LeaseStore  # noqa: E402
from test_hostos import _Control, _Proc  # noqa: E402


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def make(tmp_path, identity=None):
    clock = Clock()
    control = _Control()
    host = HostOS("fake", _Proc(identity), None, None, control)
    return LeaseStore(tmp_path / "l.sqlite3", clock=clock, host=host), clock, control


def test_multi_lane_acquire_is_all_or_nothing(tmp_path):
    store, _, _ = make(tmp_path)
    store.acquire("inference:a", ["9070xt"])
    with pytest.raises(LeaseError) as excinfo:
        store.acquire("training:1", ["r9700", "9070xt"])
    assert excinfo.value.status == 409 and excinfo.value.conflicts[0]["holder"] == "inference:a"
    assert store.acquire("training:2", ["r9700"])["lanes"] == ["r9700"]  # r9700 was NOT left half-claimed


def test_expired_leases_free_their_lanes_and_stale_tokens_cannot_act(tmp_path):
    store, clock, _ = make(tmp_path)
    first = store.acquire("training:1", ["r9700"], ttl_s=60)
    clock.now += 61
    second = store.acquire("training:2", ["r9700"], ttl_s=60)
    assert second["holder"] == "training:2"
    with pytest.raises(LeaseError) as excinfo:  # the expired holder cannot release the new holder's lease
        store.release(second["id"], first["token"])
    assert excinfo.value.status == 403
    with pytest.raises(LeaseError):  # and its own lease cannot be revived while the lane is taken
        store.renew(first["id"], first["token"])
    assert store.lane_holders()["r9700"]["holder"] == "training:2"


def test_renew_extends_and_revives_an_expired_lease_when_the_lane_is_still_free(tmp_path):
    store, clock, _ = make(tmp_path)
    lease = store.acquire("training:1", ["r9700"], ttl_s=60)
    clock.now += 50
    assert store.renew(lease["id"], lease["token"])["remaining_s"] == pytest.approx(60)
    clock.now += 200  # expired, nobody else took the lane
    assert store.renew(lease["id"], lease["token"])["state"] == "active"
    with pytest.raises(LeaseError):
        store.renew(lease["id"], "wrong-token")


def test_acquire_is_idempotent_per_holder_and_rejects_changed_lanes(tmp_path):
    store, _, _ = make(tmp_path)
    a = store.acquire("training:1", ["r9700"])
    b = store.acquire("training:1", ["r9700"])
    assert a["id"] == b["id"] and a["token"] == b["token"]
    with pytest.raises(LeaseError):
        store.acquire("training:1", ["r9700", "9070xt"])


def test_release_frees_lanes_and_holder_helpers(tmp_path):
    store, _, _ = make(tmp_path)
    store.acquire("training:1", ["r9700"])
    assert store.renew_holder("training:1") == "renewed" and store.renew_holder("training:nope") == "none"
    assert store.conflicts(["r9700"]) and not store.conflicts(["r9700"], ignore_holder="training:1")
    assert store.release_holder("training:1") is True and store.release_holder("training:1") is False
    assert store.conflicts(["r9700"]) == [] and store.renew_holder("training:1") == "none"
    store.acquire("training:2", ["r9700"])
    assert store.lane_holders()["r9700"]["holder"] == "training:2"


def test_heartbeat_reports_lost_when_lane_was_taken_after_expiry(tmp_path):
    store, clock, _ = make(tmp_path)
    store.acquire("training:1", ["r9700"], ttl_s=10)
    clock.now += 11
    store.acquire("training:2", ["r9700"])
    assert store.renew_holder("training:1") == "lost"


def test_preempt_terminates_only_the_recorded_process(tmp_path):
    store, _, control = make(tmp_path, identity=ProcessIdentity(4321, start_token=77, exe="python"))
    lease = store.acquire("training:1", ["r9700"], pid=4321)
    result = store.terminate_holder(lease["id"])
    assert result["process"] == "terminated" and control.terminated == [4321] and result["lease"]["state"] == "released"

    store2, _, control2 = make(tmp_path / "b", identity=ProcessIdentity(4321, start_token=77))
    lease2 = store2.acquire("training:2", ["r9700"], pid=4321)
    store2._host = HostOS("fake", _Proc(ProcessIdentity(4321, start_token=78)), None, None, control2)  # PID recycled
    result2 = store2.terminate_holder(lease2["id"])
    assert result2["process"] == "pid_mismatch_or_gone" and control2.terminated == []
    assert store2.lane_holders() == {}


def test_validation(tmp_path):
    store, _, _ = make(tmp_path)
    for kwargs in ({"holder": "", "lanes": ["a"]}, {"holder": "x", "lanes": []}, {"holder": "x", "lanes": ["a"], "ttl_s": 0}):
        with pytest.raises(LeaseError) as excinfo:
            store.acquire(**kwargs)
        assert excinfo.value.status == 400


# ----------------------------------------------------------- HTTP + deploy gating


@pytest.fixture
def live(tmp_path):
    config_path = Path(__file__).resolve().parent.parent / "example.dual_gpu.toml"
    state = app.ApplicationState(config_path, tmp_path / "data")
    server = app.ChatbotHTTPServer(("127.0.0.1", 0), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        state.manager.shutdown()
        thread.join(timeout=5)


def call(method, url, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"} if body else {})
    try:
        with request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read())
    except error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_lease_http_lifecycle(live):
    _state, base = live
    status, lease = call("POST", f"{base}/api/leases", {"holder": "manual", "lanes": ["r9700"], "ttl_s": 60})
    assert status == 200 and lease["token"]
    assert call("POST", f"{base}/api/leases", {"holder": "other", "lanes": ["r9700", "9070xt"]})[0] == 409
    assert call("POST", f"{base}/api/leases", {"holder": "x", "lanes": ["nope"]})[0] == 400
    listing = call("GET", f"{base}/api/leases")[1]
    assert listing["lane_holders"]["r9700"]["holder"] == "manual" and "9070xt" not in listing["lane_holders"]
    assert call("POST", f"{base}/api/leases/{lease['id']}/renew", {"token": "bad"})[0] == 403
    assert call("POST", f"{base}/api/leases/{lease['id']}/renew", {"token": lease["token"]})[0] == 200
    assert call("DELETE", f"{base}/api/leases/{lease['id']}?token=bad")[0] == 403
    assert call("DELETE", f"{base}/api/leases/{lease['id']}?token={lease['token']}")[0] == 200
    assert call("GET", f"{base}/api/leases")[1]["lane_holders"] == {}


def test_inference_cannot_displace_training_and_training_cannot_displace_inference(live):
    state, base = live
    status, run = call("POST", f"{base}/api/training/runs", {"name": "t", "lanes": ["r9700"]})
    assert status == 200 and run["lease"]["holder"] == f"training:{run['id']}"
    with pytest.raises(app.AppError) as excinfo:  # every deploy path calls this guard before touching servers
        state._lease_guard({"r9700"})
    assert excinfo.value.status == 409 and "training:" in str(excinfo.value)
    state._lease_guard({"9070xt"})  # the other lane is free for inference/eval

    call("POST", f"{base}/api/training/runs/{run['id']}/finish", {"status": "completed"})
    state._lease_guard({"r9700"})  # released on finish

    class _Lane:
        key = "9070xt"

    class _Handle:
        lanes = [_Lane()]
        model = type("M", (), {"name": "m"})()

    state.manager.handles = {"x": _Handle()}  # inference running on the 9070 XT
    status, body = call("POST", f"{base}/api/training/runs", {"name": "t2", "lanes": ["9070xt"]})
    assert status == 409 and "Inference is loaded" in body["error"]
    state.manager.handles = {}


def test_training_lease_is_renewed_by_heartbeats_and_expires_without_them(live):
    state, base = live
    _status, run = call("POST", f"{base}/api/training/runs", {"name": "t", "lanes": ["r9700"], "lease_ttl_s": 30})
    before = state.leases.active_holder(f"training:{run['id']}")["expires_at"]
    call("POST", f"{base}/api/training/runs/{run['id']}/metrics", {"steps": [], "heartbeat": {"last_step": 1}})
    assert state.leases.active_holder(f"training:{run['id']}")["expires_at"] >= before
    state.leases.clock = lambda: before + 100  # trainer died: no heartbeats
    assert state.leases.lane_holders() == {}
