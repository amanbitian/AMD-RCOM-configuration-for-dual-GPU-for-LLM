"""UI-controlled service: restart/stop, editable settings saved to the config file, launcher behaviour."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

import app
import launch_service
from dual_gpu_setup import settings
from dual_gpu_setup.config import load_config

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example.dual_gpu.toml"


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


# --- settings: validation ---------------------------------------------------------------------------

@pytest.mark.parametrize("update,message", [
    ({"nope": 1}, "Unknown setting"),
    ({"load_mode": "lazy"}, "one of: none, mmap, auto"),
    ({"vram_admission": "yes"}, "true or false"),
    ({"cache_ram_mib": -1}, "between 0 and 65536"),
    ({"cache_ram_mib": 1.5}, "whole number"),
    ({"cache_ram_mib": True}, "must be a number"),
    ({"residency_check_seconds": 1}, "between 5 and 600"),
    ({"vram_headroom_gb": float("nan")}, "must be a number"),
    ({}, "object of setting names"),
    ("load_mode", "object of setting names"),
])
def test_invalid_settings_are_rejected(update, message):
    with pytest.raises(ValueError, match=message):
        settings.validate(update)


def test_valid_settings_are_coerced_to_their_type():
    assert settings.validate({"cache_ram_mib": 2048.0, "vram_headroom_gb": 2, "load_mode": "mmap", "kill_busy_ports": False}) == {
        "cache_ram_mib": 2048, "vram_headroom_gb": 2.0, "load_mode": "mmap", "kill_busy_ports": False}


def test_every_editable_setting_is_a_real_policy_field():
    policy = load_config(EXAMPLE).policy
    assert {spec["key"] for spec in settings.SETTINGS} <= set(policy.__slots__)
    assert all(spec["apply"] in {"live", "next_load"} for spec in settings.SETTINGS)
    assert {item["key"]: item["value"] for item in settings.describe(policy)}["load_mode"] == "none"


# --- settings: saving into the TOML ----------------------------------------------------------------

SAMPLE = """\
# top comment
[project]
name = "x"

[policy]
vram_safety_fraction = 0.90   # keep 10% free
load_mode = "none"        # none | mmap | auto
cache_ram_mib = 0         # 0 = no prompt cache

[[lanes]]
key = "r9700"
"""


def test_editing_keeps_comments_order_and_other_tables():
    edited = settings.edit_policy_text(SAMPLE, {"load_mode": "mmap", "cache_ram_mib": 2048, "vram_headroom_gb": 1.5})
    assert 'load_mode = "mmap"        # none | mmap | auto' in edited
    assert "cache_ram_mib = 2048         # 0 = no prompt cache" in edited
    assert "vram_safety_fraction = 0.90   # keep 10% free" in edited and "# top comment" in edited
    # a missing key lands at the end of [policy], before the next table
    assert edited.index("vram_headroom_gb = 1.5") < edited.index("[[lanes]]")
    assert edited.index("vram_headroom_gb = 1.5") > edited.index("cache_ram_mib")


def test_editing_creates_the_policy_table_when_missing():
    edited = settings.edit_policy_text('[project]\nname = "x"\n', {"cache_ram_mib": 64})
    assert edited.endswith('[policy]\ncache_ram_mib = 64\n')


def test_editing_keeps_windows_line_endings():
    edited = settings.edit_policy_text(SAMPLE.replace("\n", "\r\n"), {"load_mode": "auto"})
    assert "\r\n" in edited and "\n" not in edited.replace("\r\n", "")
    assert 'load_mode = "auto"' in edited


def test_write_backs_up_once_and_the_result_loads(tmp_path):
    path = tmp_path / "config.toml"
    shutil.copy(EXAMPLE, path)
    original = path.read_bytes()
    first = settings.write_policy(path, {"load_mode": "mmap", "cache_ram_mib": 2048, "auto_load_models": False})
    assert first == tmp_path / "config.toml.bak" and first.read_bytes() == original
    policy = load_config(path).policy
    assert (policy.load_mode, policy.cache_ram_mib, policy.auto_load_models) == ("mmap", 2048, False)

    assert settings.write_policy(path, {"cache_ram_mib": 4096}) is None  # the backup keeps the true original
    assert (tmp_path / "config.toml.bak").read_bytes() == original
    assert load_config(path).policy.cache_ram_mib == 4096
    assert not (tmp_path / "config.toml.tmp").exists()


def test_write_refuses_and_leaves_the_file_when_the_edit_would_not_parse(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    shutil.copy(EXAMPLE, path)
    before = path.read_bytes()
    monkeypatch.setattr(settings, "edit_policy_text", lambda text, values: text + "\n[[[broken")
    with pytest.raises(ValueError, match="would not parse"):
        settings.write_policy(path, {"load_mode": "mmap"})
    assert path.read_bytes() == before and not (tmp_path / "config.toml.bak").exists()

    monkeypatch.setattr(settings, "edit_policy_text", lambda text, values: text)  # silently ignores the change
    with pytest.raises(ValueError, match="could not set load_mode"):
        settings.write_policy(path, {"load_mode": "mmap"})


# --- the suite must never touch real servers ---------------------------------------------------------

def test_the_suite_cannot_see_or_stop_real_servers():
    """These tests use the real lane ports. An earlier version of the restart test found a developer's
    llama-server on 8081 through unload() and stopped it; conftest now blocks that for every test."""
    control = app.get_host().control
    assert control.pid_listening_on(8081) is None and control.pid_listening_on(8082) is None
    with pytest.raises(AssertionError, match="terminate real process"):
        control.terminate_tree(os.getpid())


def test_unload_through_the_http_api_does_not_reach_a_real_lane_port(live_app):
    live = live_app()
    status, body = live.post("/api/unload", {})
    assert status == 200 and body["state"] == "idle"


def test_unload_accepts_force_over_http(live_app):
    live = live_app()
    status, body = live.post("/api/unload", {"force": True})
    assert status == 200 and body["state"] == "idle"


# --- service control ---------------------------------------------------------------------------------

class StubManager:
    def __init__(self, profile=None):
        self.active_profile = profile
        self.calls = []
        self.fail = None
        self.deployed = []

    def unload(self):
        if self.fail:
            raise self.fail
        self.calls.append("unload")

    def deploy(self, profile):
        self.deployed.append(profile)


def _control(tmp_path, profile=None, with_server=True):
    state = SimpleNamespace(manager=StubManager(profile), data_dir=tmp_path, config_path=tmp_path / "c.toml")
    control = app.ServiceControl(state, argparse.Namespace(host="127.0.0.1", port=8090))
    if with_server:
        control.server = SimpleNamespace(shutdown=Mock())
    control._spawn_replacement = Mock()
    return control, state


PROFILE = {"mode": "single_gpu", "models": [{"model": "m", "lane": "r9700", "context_window": 16384}]}


def test_restart_unloads_then_starts_a_replacement_then_stops(tmp_path):
    control, state = _control(tmp_path, PROFILE)
    order = []
    state.manager.unload = lambda: order.append("unload")
    control._spawn_replacement = lambda: order.append("spawn")
    control.server.shutdown = lambda: order.append("shutdown")

    assert control.request("restart")["action"] == "restart" and order == ["unload"]  # models are down before the reply
    assert wait_for(lambda: order == ["unload", "spawn", "shutdown"])
    assert not control.restore_path.exists()


def test_restart_with_reload_saves_the_deployment_for_the_replacement(tmp_path):
    control, _state = _control(tmp_path, PROFILE)
    assert control.request("restart", reload_models=True)["reload_models"] is True
    saved = json.loads(control.restore_path.read_text(encoding="utf-8"))
    assert saved["profile"] == PROFILE and abs(saved["created"] - time.time()) < 5


def test_reload_is_skipped_when_nothing_is_loaded(tmp_path):
    control, _state = _control(tmp_path, None)
    assert control.request("restart", reload_models=True)["reload_models"] is False
    assert not control.restore_path.exists()


def test_a_plain_restart_discards_an_old_restore_file(tmp_path):
    control, _state = _control(tmp_path, PROFILE)
    control.restore_path.write_text("{}", encoding="utf-8")
    control.request("restart")
    assert not control.restore_path.exists()


def test_stop_shuts_down_without_a_replacement(tmp_path):
    control, _state = _control(tmp_path, PROFILE)
    control.request("stop", reload_models=True)
    assert wait_for(lambda: control.server.shutdown.called)
    control._spawn_replacement.assert_not_called()
    assert not control.restore_path.exists()  # reload only applies to a restart


def test_a_second_request_while_one_is_pending_is_a_409(tmp_path):
    control, _state = _control(tmp_path, PROFILE)
    control.request("restart")
    with pytest.raises(app.AppError) as raised:
        control.request("stop")
    assert raised.value.status == 409


def test_a_busy_deployment_aborts_the_restart_cleanly(tmp_path):
    control, state = _control(tmp_path, PROFILE)
    state.manager.fail = app.AppError("Another deployment change is already in progress.", 409)
    with pytest.raises(app.AppError, match="already in progress"):
        control.request("restart", reload_models=True)
    assert control.pending is None and not control.restore_path.exists()
    time.sleep(0.7)
    assert not control.server.shutdown.called
    control.request  # the service is still usable: a retry is accepted
    state.manager.fail = None
    assert control.request("restart")["ok"]


def test_a_failed_spawn_keeps_the_service_running(tmp_path):
    control, _state = _control(tmp_path, PROFILE)
    control._spawn_replacement = Mock(side_effect=OSError("cannot start python"))
    control.request("restart", reload_models=True)
    assert wait_for(lambda: control.pending is None)
    assert not control.server.shutdown.called and not control.restore_path.exists()


def test_a_process_that_was_not_started_by_app_cannot_restart_itself(tmp_path):
    control, _state = _control(tmp_path, PROFILE, with_server=False)
    with pytest.raises(app.AppError, match="cannot restart itself") as raised:
        control.request("restart")
    assert raised.value.status == 409 and control.info()["can_restart"] is False


def test_unknown_actions_are_rejected(tmp_path):
    control, _state = _control(tmp_path)
    with pytest.raises(app.AppError, match="Unknown service action"):
        control.request("reboot")


def test_the_replacement_command_waits_for_this_process_and_stays_quiet(tmp_path):
    control, state = _control(tmp_path)
    command = control.replacement_command()
    assert command[0] == sys.executable and command[1].endswith("app.py")
    assert command[command.index("--replace-pid") + 1] == str(os.getpid())
    assert command[command.index("--config") + 1] == str(state.config_path)
    assert command[command.index("--port") + 1] == "8090" and command[command.index("--data-dir") + 1] == str(tmp_path)
    assert "--no-browser" in command and "--prune-older-than-days" not in command


def test_info_reports_the_running_process(tmp_path):
    control, state = _control(tmp_path)
    info = control.info()
    assert info["pid"] == os.getpid() and info["port"] == 8090 and info["config_path"] == str(state.config_path)
    assert info["can_restart"] is True and info["pending"] is None and info["restore_pending"] is False
    assert info["host_os"]["system"]


def test_restore_reloads_a_fresh_saved_deployment_once(tmp_path):
    control, state = _control(tmp_path)
    control.restore_path.write_text(json.dumps({"created": time.time(), "profile": PROFILE}), encoding="utf-8")
    assert control.restore_after_restart() is True
    # PROFILE lists one model under `models` (how an on-demand load records it); deploy() gets `model`.
    assert wait_for(lambda: state.manager.deployed == [{"mode": "single_gpu", "model": PROFILE["models"][0]}])
    assert not control.restore_path.exists() and control.restore_after_restart() is False


def test_a_profile_from_an_on_demand_load_is_made_deployable_before_the_reload(tmp_path):
    """On-demand loads record `models` even for one GPU; deploy() wants `model` there."""
    control, state = _control(tmp_path)
    on_demand = {"mode": "single_gpu", "models": [{"model": "m", "lane": "r9700", "context_window": 16384,
                                                  "workload_kind": "coding_agent"}]}
    control.restore_path.write_text(json.dumps({"created": time.time(), "profile": on_demand}), encoding="utf-8")
    assert control.restore_after_restart() is True
    assert wait_for(lambda: len(state.manager.deployed) == 1)
    assert state.manager.deployed[0] == {"mode": "single_gpu", "model": on_demand["models"][0]}


def test_deployable_profile_leaves_other_shapes_alone():
    parallel = {"mode": "parallel_models", "models": [{"model": "a", "lane": "r9700"}, {"model": "b", "lane": "9070xt"}]}
    large = {"mode": "single_large_model", "model": {"model": "huge"}}
    plain = {"mode": "single_gpu", "model": {"model": "m", "lane": "r9700"}}
    assert app.deployable_profile(parallel) is parallel and app.deployable_profile(large) is large
    assert app.deployable_profile(plain) is plain


@pytest.mark.parametrize("content", [
    json.dumps({"created": 0, "profile": PROFILE}),            # stale
    json.dumps({"created": time.time(), "profile": "nope"}),   # not a deployment
    "not json",
])
def test_restore_ignores_stale_or_damaged_files(tmp_path, content):
    control, state = _control(tmp_path)
    control.restore_path.write_text(content, encoding="utf-8")
    assert control.restore_after_restart() is False
    time.sleep(0.1)
    assert state.manager.deployed == [] and not control.restore_path.exists()


def test_wait_for_exit_returns_once_the_old_process_is_gone(monkeypatch):
    alive = iter([object(), object(), None])
    monkeypatch.setattr(app, "get_host", lambda: SimpleNamespace(process=SimpleNamespace(identity=lambda pid: next(alive))))
    monkeypatch.setattr(app.time, "sleep", lambda _s: None)
    assert app.wait_for_exit(123, timeout=5) is True


def test_wait_for_exit_gives_up_after_the_timeout(monkeypatch):
    monkeypatch.setattr(app, "get_host", lambda: SimpleNamespace(process=SimpleNamespace(identity=lambda pid: object())))
    assert app.wait_for_exit(123, timeout=0.2) is False


# --- HTTP ------------------------------------------------------------------------------------------

def _raw_post(live, path, content_type):
    request = urlrequest.Request(live.base + path, data=b"{}", method="POST", headers={"Content-Type": content_type})
    try:
        with urlrequest.urlopen(request, timeout=10) as response:
            return response.status
    except urlerror.HTTPError as exc:
        return exc.code


def test_settings_endpoint_lists_applies_and_saves(live_app):
    live = live_app()
    status, body = live.get("/api/settings")
    by_key = {item["key"]: item for item in body["settings"]}
    assert status == 200 and by_key["load_mode"]["value"] == "none" and by_key["cache_ram_mib"]["apply"] == "next_load"
    assert body["config_path"] == str(live.state.config_path)

    status, body = live.post("/api/settings", {"values": {"cache_ram_mib": 2048, "vram_admission": False, "load_mode": "none"}})
    assert status == 200 and body["changed"] == ["cache_ram_mib", "vram_admission"]
    assert body["next_load"] == ["cache_ram_mib"]  # vram_admission is live; load_mode did not change
    assert live.state.config.policy.cache_ram_mib == 2048 and live.state.config.policy.vram_admission is False
    saved = load_config(live.state.config_path).policy
    assert saved.cache_ram_mib == 2048 and saved.vram_admission is False

    status, body = live.post("/api/settings", {"values": {"cache_ram_mib": 2048}})
    assert status == 200 and body["changed"] == []  # nothing to write


def test_settings_endpoint_rejects_bad_input_and_non_json_posts(live_app):
    live = live_app()
    before = live.state.config_path.read_bytes()
    assert live.post("/api/settings", {"values": {"nope": 1}})[0] == 400
    assert live.post("/api/settings", {"values": {"load_mode": "lazy"}})[0] == 400
    assert _raw_post(live, "/api/settings", "text/plain") == 415
    assert _raw_post(live, "/api/service/restart", "application/x-www-form-urlencoded") == 415
    assert _raw_post(live, "/api/service/stop", "text/plain") == 415
    assert live.state.config_path.read_bytes() == before and live.state.config.policy.load_mode == "none"


def test_service_endpoints_report_and_control_the_process(live_app, tmp_path):
    live = live_app()
    assert live.get("/api/service")[0] == 501  # a bare test server has no service control
    control = app.ServiceControl(live.state, argparse.Namespace(host="127.0.0.1", port=live.server.server_address[1]))
    control.server = SimpleNamespace(shutdown=Mock())
    control._spawn_replacement = Mock()
    live.state.control = control

    status, info = live.get("/api/service")
    assert status == 200 and info["pid"] == os.getpid() and info["can_restart"] is True

    status, body = live.post("/api/service/restart", {"reload_models": True})
    assert status == 200 and body["ok"] and body["action"] == "restart" and body["reload_models"] is False  # nothing loaded
    assert wait_for(lambda: control.server.shutdown.called)
    control._spawn_replacement.assert_called_once()
    assert live.post("/api/service/stop", {})[0] == 409  # a restart is already under way


def test_status_reports_orphaned_servers_and_unload_accepts_a_lane(live_app):
    live = live_app()
    live.state.manager.lane_orphans = lambda lanes=None: [{"lane": "r9700", "port": 8081, "pid": 4242}]
    status, body = live.get("/api/status")
    assert status == 200 and body["orphans"] == [{"lane": "r9700", "port": 8081, "pid": 4242}]
    assert live.post("/api/unload", {"lane": "nope"})[0] == 400


# --- launcher ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("flags,interactive,expected", [
    ({}, True, True), ({}, False, False),                       # a person at a terminal vs a program
    ({"open_browser": True}, False, True), ({"no_browser": True}, True, False),
])
def test_launcher_opens_the_ui_only_for_people(flags, interactive, expected):
    args = launch_service.build_parser().parse_args([])
    for name, value in flags.items():
        setattr(args, name, value)
    assert launch_service.wants_browser(args, interactive) is expected


def test_launcher_opens_the_ui_when_the_service_is_already_running(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(launch_service.webbrowser, "open", opened.append)
    monkeypatch.setattr(launch_service, "interactive", lambda: True)
    monkeypatch.setattr(launch_service, "_probe", lambda base_url, timeout=2.0: {"state": "idle"})
    monkeypatch.setattr(sys, "argv", ["launch_service.py", "--port", "8123"])
    assert launch_service.main() == 0
    assert opened == ["http://127.0.0.1:8123"]
    assert json.loads(capsys.readouterr().out)["browser_opened"] is True


def test_launcher_stays_silent_for_programs(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(launch_service.webbrowser, "open", opened.append)
    monkeypatch.setattr(launch_service, "interactive", lambda: False)
    monkeypatch.setattr(launch_service, "_probe", lambda base_url, timeout=2.0: {"state": "idle"})
    monkeypatch.setattr(sys, "argv", ["launch_service.py"])
    assert launch_service.main() == 0 and opened == []


# --- dashboard script --------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_dashboard_scripts_parse(tmp_path):
    html = (REPO / "dashboard.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert scripts
    for index, source in enumerate(scripts):
        path = tmp_path / f"dashboard_{index}.js"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_unload_model_needs_a_lane_and_reports_released(live_app):
    live = live_app()
    status, body = live.post("/api/unload", {"model": "x"})
    assert status == 400 and "together with `lane`" in body["error"]
    status, body = live.post("/api/unload", {"lane": "r9700", "model": "x"})
    assert status == 200 and body["released"] is False and body["state"] == "idle"
    status, body = live.post("/api/unload", {"lane": "r9700"})
    assert status == 200 and body["released"] is False
