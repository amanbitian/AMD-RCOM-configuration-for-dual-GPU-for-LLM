"""Hardware log, alerts, archives, host sensors, PCIe slot link, backups and interrupted runs."""
from __future__ import annotations

import gzip
import json

import pytest
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_gpu_setup import datasafe  # noqa: E402
from dual_gpu_setup.archive import HardwareArchive, archive_is_complete, import_archive, read_archive  # noqa: E402
from dual_gpu_setup.hardware_log import HardwareLog  # noqa: E402
from dual_gpu_setup.host_sensors import LinuxHostSensors, WindowsHostSensors  # noqa: E402
from dual_gpu_setup.training.amd_smi import AmdSmiCli  # noqa: E402
from dual_gpu_setup.training.hardware import pcie_slot_link  # noqa: E402
from dual_gpu_setup.training.store import TrainingStore  # noqa: E402


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def gpu_sample(ts, lane="r9700", bus="0000:03:00.0", **extra):
    return {"ts": ts, "pci_bus": bus, "lane": lane, "gpu_busy_pct": 90.0, "vram_used_bytes": 10, "vram_total_bytes": 32,
            "temp_junction_c": 80.0, "temp_memory_c": 70.0, "temp_edge_c": 60.0, "power_w": 300.0, "fan_rpm": 3000.0,
            "unavailable": {}, **extra}


def make_log(tmp_path, in_use, clock=None, sensors=None, **kwargs):
    clock = clock or Clock()
    store = TrainingStore(tmp_path / "t.sqlite3", clock=clock)
    state = {"in_use": in_use}
    log = HardwareLog(store, "Linux", lambda: state["in_use"], host_sensors=sensors, clock=clock, **kwargs)
    return log, store, state, clock


def count(store, table):
    with store.connect() as c:
        return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ------------------------------------------------------------- in-use gating


def test_idle_gpus_and_the_integrated_gpu_write_nothing(tmp_path):
    log, store, state, clock = make_log(tmp_path, {})
    log.on_gpu_samples([gpu_sample(clock.now)])
    assert count(store, "hardware_samples") == 0 and count(store, "host_samples") == 0 and not log.logging


def test_only_lanes_in_use_are_logged_with_model_and_os_context(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {"model": "qwen", "kind": "inference"}})
    other = gpu_sample(clock.now, lane="9070xt", bus="0000:07:00.0")
    igpu = gpu_sample(clock.now, lane=None, bus="0000:0f:00.0")
    log.on_gpu_samples([gpu_sample(clock.now), other, igpu])
    with store.connect() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM hardware_samples")]
    assert [r["lane"] for r in rows] == ["r9700"]           # one model on one GPU: the other card is not logged
    assert rows[0]["host_os"] == "Linux" and rows[0]["model"] == "qwen"
    assert json.loads(rows[0]["context_json"]) == {"kind": "inference"}
    assert count(store, "host_samples") == 1 and log.logging


def test_both_gpus_in_use_are_both_logged(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {"model": "a"}, "9070xt": {"model": "b"}})
    log.on_gpu_samples([gpu_sample(clock.now), gpu_sample(clock.now, lane="9070xt", bus="0000:07:00.0")])
    assert count(store, "hardware_samples") == 2


def test_system_power_estimate_is_gpu_plus_cpu_and_flags_when_cpu_power_is_missing(tmp_path):
    class Sensors:
        def read(self):
            return {"cpu_temp_c": 70.0, "cpu_power_w": 100.0}, {}

    log, store, state, clock = make_log(tmp_path, {"r9700": {}}, sensors=Sensors())
    log.on_gpu_samples([gpu_sample(clock.now)])
    with store.connect() as c:
        row = dict(c.execute("SELECT * FROM host_samples").fetchone())
    assert row["gpu_power_w"] == 300.0 and row["est_system_power_w"] == 400.0 and row["cpu_temp_c"] == 70.0
    assert row["ssd_temp_c"] is None   # a missing sensor is NULL, never 0


# ------------------------------------------------------------------- alerts


def test_alert_is_saved_with_start_end_and_peak_and_survives_a_restart(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    for temp in (99.0, 101.0, 104.0, 102.0):
        clock.now += 6
        log.on_gpu_samples([gpu_sample(clock.now, temp_junction_c=temp)])
    open_now = log.alerts_between(None, None, open_only=True)
    assert len(open_now) == 1 and open_now[0]["kind"] == "gpu_junction_hot" and open_now[0]["peak_value"] == 104.0
    for _ in range(3):  # three cool samples close it
        clock.now += 1
        log.on_gpu_samples([gpu_sample(clock.now, temp_junction_c=80.0)])
    alert = log.alerts_between(None, None)[0]
    assert alert["ended_at"] is not None and alert["peak_value"] == 104.0 and alert["host_os"] == "Linux"
    assert alert["ended_at"] > alert["started_at"] and alert["threshold"] == 100.0


def test_open_alert_is_closed_when_the_gpu_goes_idle_and_orphans_are_recovered_after_a_crash(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    log.on_gpu_samples([gpu_sample(clock.now, temp_memory_c=95.0)])
    assert log.alerts_between(None, None, open_only=True)
    state["in_use"] = {}
    clock.now += 1
    log.on_gpu_samples([gpu_sample(clock.now)])
    assert not log.alerts_between(None, None, open_only=True)
    # crash case: an alert row left open by a dead process is closed by the next start
    log.on_gpu_samples([gpu_sample(clock.now)])
    state["in_use"] = {"r9700": {}}
    log.on_gpu_samples([gpu_sample(clock.now, temp_memory_c=96.0)])
    assert log.alerts_between(None, None, open_only=True)
    reborn = HardwareLog(store, "Linux", lambda: {}, clock=clock)
    assert not reborn.alerts_between(None, None, open_only=True)


def test_throttle_and_degraded_pcie_link_alerts(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    log.on_gpu_samples([gpu_sample(clock.now, throttle_status=4, pcie_slot_width=4, pcie_slot_max_width=8)])
    kinds = {a["kind"] for a in log.alerts_between(None, None, open_only=True)}
    assert kinds == {"gpu_throttle", "pcie_link_degraded"}


def test_healthy_x8_slot_does_not_raise_a_link_alert(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    log.on_gpu_samples([gpu_sample(clock.now, pcie_slot_width=8, pcie_slot_max_width=8, throttle_status=0)])
    assert log.alerts_between(None, None) == []


# ---------------------------------------------------- summary / history / export


def test_per_run_summary_has_min_avg_peak_energy_fan_and_throttle(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    for i, (temp, power, throttle) in enumerate(((70.0, 200.0, 0), (80.0, 300.0, 1), (90.0, 400.0, 0))):
        log.on_gpu_samples([gpu_sample(clock.now + i, temp_junction_c=temp, power_w=power, throttle_status=throttle)])
    summary = log.summary(clock.now - 1, clock.now + 10)["gpus"]["0000:03:00.0"]
    assert summary["temp_junction_c"] == {"min": 70.0, "avg": 80.0, "max": 90.0}
    assert summary["power_w"]["max"] == 400.0 and summary["peak_fan_rpm"] == 3000.0
    assert summary["energy_wh"] == (200 + 300 + 400) / 3600.0 and summary["throttled_seconds"] == 1.0


def test_history_thins_the_series_but_never_the_stored_rows(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    for i in range(200):
        log.on_gpu_samples([gpu_sample(clock.now + i)])
    history = log.history(clock.now - 1, clock.now + 500, max_points=20)["gpus"]["0000:03:00.0"]
    assert history["samples"] == 200 and history["stride"] == 10 and len(history["points"]) == 20
    assert count(store, "hardware_samples") == 200


def test_export_csv_and_jsonl(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {"model": "qwen"}})
    log.on_gpu_samples([gpu_sample(clock.now)])
    ctype, body = log.export(clock.now - 1, clock.now + 1, "csv")
    assert ctype.startswith("text/csv") and body.splitlines()[0].startswith("ts,pci_bus") and "qwen" in body
    ctype, body = log.export(clock.now - 1, clock.now + 1, "jsonl", "host_samples")
    assert json.loads(body.splitlines()[0])["host_os"] == "Linux"


def test_old_samples_are_never_deleted_by_the_store(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    log.on_gpu_samples([gpu_sample(clock.now - 90 * 86400)])
    assert store.prune_hardware(older_than_days=30) == 0
    assert count(store, "hardware_samples") == 1


# ------------------------------------------------------------------ archives


def test_archive_is_verified_idempotent_and_leaves_the_database_alone(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    for i in range(5):
        log.on_gpu_samples([gpu_sample(clock.now + i)])
    log.record_event("gpu_reset", "error", {"line": "x"}, lane="r9700")
    archive = HardwareArchive(tmp_path / "archive", store.connect, "Linux", clock=clock)
    first = archive.archive_all()
    assert first["written"] >= 1 and not first["failed"]
    files = archive.files()
    assert all(archive_is_complete(f) for f in files)
    rows, counts = read_archive(files[0])
    assert counts["hardware_samples"] == 5 and counts["hardware_events"] == 1
    again = archive.archive_all()
    assert again["written"] == 0                       # unchanged days are not rewritten
    assert count(store, "hardware_samples") == 5       # the archive never removes rows
    log.on_gpu_samples([gpu_sample(clock.now + 9)])
    assert archive.archive_all()["written"] == 1       # a day that gained rows is refreshed


def test_torn_archive_is_rejected_and_rewritten(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    log.on_gpu_samples([gpu_sample(clock.now)])
    archive = HardwareArchive(tmp_path / "archive", store.connect, "Linux", clock=clock)
    archive.archive_all()
    path = archive.files()[0]
    path.write_bytes(path.read_bytes()[:-12])          # truncate the gzip stream
    assert not archive_is_complete(path)
    archive.archive_all()
    assert archive_is_complete(path)


def test_archive_import_adds_missing_rows_once_and_tags_the_os(tmp_path):
    log, store, state, clock = make_log(tmp_path, {"r9700": {}})
    for i in range(3):
        log.on_gpu_samples([gpu_sample(clock.now + i)])
    HardwareArchive(tmp_path / "archive", store.connect, "Windows", clock=clock).archive_all()
    path = HardwareArchive(tmp_path / "archive", store.connect, "Windows").files()[0]
    other = TrainingStore(tmp_path / "other.sqlite3", clock=clock)
    with other.connect() as c:
        first = import_archive(path, c)
    with other.connect() as c:
        second = import_archive(path, c)
    assert first["hardware_samples"] == 3 and second == {}
    with other.connect() as c:
        assert {r[0] for r in c.execute("SELECT host_os FROM hardware_samples")} == {"Linux"}  # rows keep the OS they were logged on


# ------------------------------------------------------------ host sensors


def fake_hwmon(root: Path, name: str, **files):
    chip = root / name
    chip.mkdir(parents=True)
    (chip / "name").write_text(name.split("_", 1)[1] + "\n")
    for filename, value in files.items():
        (chip / filename).write_text(f"{value}\n")
    return chip


def test_linux_host_sensors_map_hwmon_chips_and_report_what_is_missing(tmp_path):
    hwmon = tmp_path / "hwmon"
    fake_hwmon(hwmon, "hwmon0_k10temp", temp1_input=61000, temp1_label="Tctl")
    fake_hwmon(hwmon, "hwmon1_nvme", temp1_input=47000, temp1_label="Composite")
    fake_hwmon(hwmon, "hwmon2_spd5118", temp1_input=39500)
    fake_hwmon(hwmon, "hwmon3_nct6798", fan1_input=900, fan1_label="CPU Fan", fan2_input=1100, fan2_label="Pump",
               fan3_input=700, temp2_input=33000, temp2_label="Coolant")
    fake_hwmon(hwmon, "hwmon4_amdgpu", fan1_input=5000)
    values, why = LinuxHostSensors(tmp_path, hwmon, tmp_path / "none", tmp_path / "noproc").read()
    assert values["cpu_temp_c"] == 61.0 and values["ssd_temp_c"] == 47.0 and values["ram_temp_c"] == 39.5
    assert values["cpu_fan_rpm"] == 900 and values["pump_rpm"] == 1100 and values["coolant_temp_c"] == 33.0
    assert values["case_fan_rpm"] == 700            # the GPU's own fan is not a case fan
    assert values["cpu_power_w"] is None and "cpu_power_w" in why and values["disk_free_bytes"] > 0


def test_cpu_power_comes_from_the_rapl_energy_delta(tmp_path):
    cap = tmp_path / "powercap" / "intel-rapl:0"
    cap.mkdir(parents=True)
    (cap / "name").write_text("package-0\n")
    ticks = iter([0.0, 2.0])
    sensors = LinuxHostSensors(tmp_path, tmp_path / "hwmon", tmp_path / "powercap", tmp_path / "proc", clock=lambda: next(ticks))
    (cap / "energy_uj").write_text("1000000\n")
    assert sensors.read()[0]["cpu_power_w"] is None             # needs two readings
    (cap / "energy_uj").write_text("201000000\n")
    assert sensors.read()[0]["cpu_power_w"] == 100.0            # 200 J over 2 s


def test_windows_host_sensors_read_the_hardware_monitor_and_degrade_with_a_reason(tmp_path):
    payload = json.dumps([
        {"Name": "CPU Package", "SensorType": "Temperature", "Value": 66.0, "Identifier": "/amdcpu/0/temperature/2"},
        {"Name": "CPU Package", "SensorType": "Power", "Value": 88.0, "Identifier": "/amdcpu/0/power/0"},
        {"Name": "Samsung SSD", "SensorType": "Temperature", "Value": 52.0, "Identifier": "/nvme/0/temperature/0"},
        {"Name": "Pump", "SensorType": "Fan", "Value": 2100.0, "Identifier": "/lpc/nct/fan/2"},
    ])
    live = WindowsHostSensors(tmp_path, runner=lambda: payload, autostart=False)
    live.refresh()
    values, why = live.read()
    assert (values["cpu_temp_c"], values["cpu_power_w"], values["ssd_temp_c"], values["pump_rpm"]) == (66.0, 88.0, 52.0, 2100.0)
    assert "coolant_temp_c" in why
    absent = WindowsHostSensors(tmp_path, runner=lambda: "", autostart=False)
    absent.refresh()
    values, why = absent.read()
    assert values["cpu_temp_c"] is None and "LibreHardwareMonitor" in why["cpu_temp_c"]


# ----------------------------------------------- PCIe slot link and throttle


def test_pcie_slot_link_uses_the_narrowest_link_on_the_path(tmp_path):
    root = tmp_path / "pci0000:00"
    port = root / "0000:00:01.1"
    switch = port / "0000:01:00.0"
    gpu = switch / "0000:02:00.0"
    gpu.mkdir(parents=True)
    for node, width, max_width, speed in ((port, 8, 8, "32.0 GT/s PCIe"), (switch, 8, 16, "32.0 GT/s PCIe"), (gpu, 16, 16, "32.0 GT/s PCIe")):
        (node / "current_link_width").write_text(f"{width}\n")
        (node / "max_link_width").write_text(f"{max_width}\n")
        (node / "current_link_speed").write_text(speed + "\n")
        (node / "max_link_speed").write_text("32.0 GT/s PCIe\n")
    link = pcie_slot_link(gpu)
    assert link["width"] == 8 and link["max_width"] == 8     # the card reports x16; the slot is x8 and that is normal here
    (port / "current_link_width").write_text("4\n")
    assert pcie_slot_link(gpu)["width"] == 4                 # a real downgrade is visible against max 8


def test_amd_smi_cli_fallback_fills_throttle_status():
    listing = [{"gpu": 0, "bdf": "0000:03:00.0"}, {"gpu": 1, "bdf": "0000:07:00.0"}]
    metric = [{"gpu": 0, "throttle": {"throttle_status": "N/A"}}, {"gpu": 1, "throttle": {"throttle_status": {"thermal": 1, "power": 0}}}]
    cli = AmdSmiCli("amd-smi", runner=lambda arg, *_: listing if arg == "list" else metric, autostart=False)
    cli.refresh()
    quiet = cli.merge({"pci_bus": "0000:03:00.0", "unavailable": {"throttle_status": "x"}})
    hot = cli.merge({"pci_bus": "0000:07:00.0", "unavailable": {"throttle_status": "x"}})
    assert "throttle_status" not in quiet and quiet["unavailable"]    # N/A stays unavailable, never 0
    assert hot["throttle_status"] == 1 and hot["amd_smi_fields"] == ["throttle_status"]


# --------------------------------------------------------- backups, migration


def make_db(path, rows=3):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE runs (id TEXT PRIMARY KEY, status TEXT)")
    connection.executemany("INSERT INTO runs VALUES (?, 'completed')", [(f"r{i}",) for i in range(rows)])
    connection.commit()
    connection.close()


def test_backup_is_verified_counted_never_overwritten_and_survives_a_missing_source(tmp_path):
    db = tmp_path / "chatbot.sqlite3"
    make_db(db, rows=4)
    result = datasafe.backup_sqlite(db, tmp_path / "backups", "pre-test")
    assert result.ok and Path(result.path).exists() and result.counts == {"runs": 4}
    again = datasafe.backup_sqlite(db, tmp_path / "backups", "pre-test")
    assert again.skipped and again.path == result.path
    assert datasafe.backup_sqlite(tmp_path / "missing.sqlite3", tmp_path / "backups", "x").skipped
    assert datasafe.verify_counts_not_lower({"runs": 4}, {"runs": 3}) == ["runs: 4 rows before, 3 after"]


def test_runs_left_running_by_a_dead_service_are_marked_interrupted_not_deleted(tmp_path):
    import app

    store = app.RunStore(tmp_path / "chatbot.sqlite3")
    store.begin_run({"id": "stuck", "created_at": app.utc_now(), "mode": "single_gpu", "target": "primary", "model_name": "m",
                     "lane_keys": ["r9700"], "request": {}, "host_os": {"system": "Linux"}})
    assert store.mark_interrupted() == 1
    run = store.get("stuck")
    assert run["status"] == "interrupted" and run["finished_at"] and "service stopped" in run["error_text"]
    assert run["origin_os"] == "Linux" and run["os"] == "Linux"
    assert store.mark_interrupted() == 0


def test_untagged_history_is_marked_windows_explicitly_and_inferred_flag_is_kept(tmp_path):
    import app

    path = tmp_path / "chatbot.sqlite3"
    app.RunStore(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO runs (id, created_at, status, mode, target, model_name, lane_keys_json, request_json) "
        "VALUES ('old', '2026-10-01T00:00:00Z', 'completed', 'm', 't', 'x', '[]', '{}')")
    connection.commit()
    connection.close()
    app.RunStore(path)                                 # the next start fills the new column
    connection = sqlite3.connect(path)
    row = connection.execute("SELECT origin_os, origin_os_inferred, host_os_json FROM runs WHERE id='old'").fetchone()
    connection.close()
    assert row == ("Windows", 1, None)                 # tagged, flagged as a guess, original columns untouched


# ------------------------------------------- checkpoints a trainer never reported


def test_unreported_checkpoints_on_disk_are_discovered_and_registered_once(tmp_path):
    from dual_gpu_setup.training.checkpoints import discover_checkpoints

    run_dir = tmp_path / "out" / "my_run"
    for step, weights in ((100, True), (200, True), (300, False)):
        d = run_dir / f"step_{step:07d}"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        if weights:
            (d / "model.safetensors").write_bytes(b"x" * 10)
    (run_dir / ".tmp_step_0000400").mkdir()           # an in-progress write is not a checkpoint
    (run_dir / "latest").write_text("step_0000200")
    found = discover_checkpoints(run_dir)
    assert [(c["step"], c["valid"]) for c in found] == [(100, True), (200, True), (300, False)]
    assert found[0]["manifest"]["kind"] == "discovered" and found[0]["manifest"]["hashed"] is False
    assert not (run_dir / "step_0000100" / "manifest.json").exists()    # nothing is written into the trainer's folder


def test_service_scans_the_runs_own_out_dir_so_a_finished_run_lists_its_checkpoints(live_app, tmp_path):
    live = live_app()
    run_dir = tmp_path / "out" / "stage1"
    for step in (10, 20):
        d = run_dir / f"step_{step:07d}"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"w")
    status, body = live.post("/api/training/runs", {"id": "r1", "name": "stage1", "config": {"out_dir": str(tmp_path / "out"), "run_name": "stage1"}, "lanes": []})
    assert status == 200, body
    assert live.get("/api/training/runs/r1")[1]["checkpoints"] == []     # the trainer never reported any
    status, scan = live.post("/api/training/runs/r1/checkpoints/scan", {})
    assert status == 200 and [a["step"] for a in scan["added"]] == [10, 20]
    assert [c["step"] for c in live.get("/api/training/runs/r1")[1]["checkpoints"]] == [20, 10] or \
        {c["step"] for c in live.get("/api/training/runs/r1")[1]["checkpoints"]} == {10, 20}
    assert live.post("/api/training/runs/r1/checkpoints/scan", {})[1]["added"] == []   # idempotent
    assert live.state.service.rescan_unreported_checkpoints() == {}


# -------------------------------------------- only stop servers this service started


def test_registry_recognises_only_its_own_processes_and_ignores_recycled_pids(tmp_path):
    from types import SimpleNamespace

    from dual_gpu_setup.process_registry import ProcessRegistry

    tokens = {100: 11, 200: 22}
    registry = ProcessRegistry(tmp_path / "managed.json", identity=lambda pid: SimpleNamespace(start_token=tokens[pid]) if pid in tokens else None)
    registry.record(100, 8081, "qwen")
    assert registry.owns(100) and not registry.owns(200)       # 200 is someone else's server
    tokens[100] = 99                                           # the PID was recycled by an unrelated process
    assert not registry.owns(100)
    tokens[100] = 11
    registry.forget(100)
    assert not registry.owns(100) and registry.entries() == []
    (tmp_path / "managed.json").write_text("{not json")
    assert not registry.owns(100)                              # a corrupt file means "nothing is known to be ours"


def test_automatic_reaping_leaves_foreign_servers_but_explicit_stop_still_works(tmp_path, monkeypatch):
    import app
    from dual_gpu_setup.server import LlamaServerProcess
    from test_model_manager import FakeControl, _manager, _with_host

    manager = _manager(tmp_path, set())
    control = _with_host(monkeypatch, FakeControl({8081: 777, 8082: 888}))

    class Registry:
        def owns(self, pid):
            return pid == 888

    LlamaServerProcess.registry = Registry()
    manager._reap_orphans_safely()
    assert control.terminated == [888]                         # ours is stopped, the foreign 777 is left alone
    manager.reap_orphans()                                     # the explicit action is unrestricted
    assert 777 in control.terminated


# ------------------------------------------------ cross-OS history, logs, schema guards


def make_runs_db(path, os_name, ids, deployed=True):
    import app

    store = app.RunStore(path)
    for rid in ids:
        store.begin_run({"id": rid, "created_at": "2026-11-01T10:00:00.000+00:00", "mode": "single_gpu", "target": "primary",
                         "model_name": "m", "lane_keys": ["r9700"], "request": {}, "host_os": {"system": os_name}})
        store.finish_run(rid, {"status": "completed", "output_text": f"out-{rid}"})
        store.save_samples(rid, [{"sampled_at": "x", "system": {"cpu_pct": 1}}], 0.0)
    return store


def test_runs_exchange_files_flow_both_ways_without_ping_pong_or_duplicates(tmp_path):
    from dual_gpu_setup.exchange import ExchangeSync

    shared = tmp_path / "shared"
    win = make_runs_db(tmp_path / "win" / "chatbot.sqlite3", "Windows", ["w1", "w2"])
    lin = make_runs_db(tmp_path / "lin" / "chatbot.sqlite3", "Linux", ["l1"])
    a = ExchangeSync(shared, "Windows", tmp_path / "win", win.path, tmp_path / "win" / "training.sqlite3")
    b = ExchangeSync(shared, "Linux", tmp_path / "lin", lin.path, tmp_path / "lin" / "training.sqlite3")
    assert a.sync(include_snapshot=False)["export_runs"]["written"] == 1
    result = b.sync(include_snapshot=False)
    assert result["import"]["imported_rows"]["runs"] == 2 and not result["import"]["errors"]
    a.sync(include_snapshot=False)                      # the Windows install now sees the Linux run
    assert {r["id"] for r in win.recent(50)} == {"w1", "w2", "l1"} and {r["id"] for r in lin.recent(50)} == {"w1", "w2", "l1"}
    assert lin.get("w1")["os"] == "Windows" and win.get("l1")["os"] == "Linux"
    assert lin.get("w1")["output_text"] == "out-w1"    # full content travels, not just a summary
    # nothing ping-pongs: the Linux install exports only its own run, and a second sync imports nothing
    b.sync(include_snapshot=False)
    own = sorted(p.name for p in (shared / "linux").iterdir())
    _h, rows, _c = __import__("dual_gpu_setup.exchange", fromlist=["read_runs_file"]).read_runs_file(shared / "linux" / own[0])
    assert [r["id"] for r in rows["runs"]] == ["l1"]
    assert a.sync(include_snapshot=False)["import"]["imported_rows"] == {}
    with lin.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM metric_samples").fetchone()[0] == 3     # samples imported once, not duplicated


def test_damaged_exchange_file_is_reported_and_does_not_stop_the_others(tmp_path):
    from dual_gpu_setup.exchange import ExchangeSync

    shared = tmp_path / "shared"
    win = make_runs_db(tmp_path / "win" / "chatbot.sqlite3", "Windows", ["w1"])
    lin = make_runs_db(tmp_path / "lin" / "chatbot.sqlite3", "Linux", ["l1"])
    ExchangeSync(shared, "Windows", tmp_path / "win", win.path, tmp_path / "w.sqlite3").sync(include_snapshot=False)
    (shared / "windows" / "runs-2026-01-01.windows.jsonl.gz").write_bytes(b"not gzip")
    result = ExchangeSync(shared, "Linux", tmp_path / "lin", lin.path, tmp_path / "l.sqlite3").sync(include_snapshot=False)
    assert len(result["import"]["errors"]) == 1 and result["import"]["imported_rows"]["runs"] == 1


def test_training_snapshot_is_a_consistent_single_file_and_merges_with_os_tags(tmp_path):
    from dual_gpu_setup.exchange import ExchangeSync
    from dual_gpu_setup.training.merge import merge_training_db

    log, store, state, clock = make_log(tmp_path / "w", {"r9700": {}})
    log.host_os = "Windows"
    log.on_gpu_samples([gpu_sample(clock.now)])
    exporter = ExchangeSync(tmp_path / "shared", "Windows", tmp_path / "w", tmp_path / "w" / "chatbot.sqlite3", store.path)
    assert exporter.export_training_snapshot(force=True)["written"]
    assert exporter.export_training_snapshot()["written"] is False          # not re-exported within hours
    snapshot = tmp_path / "shared" / "windows" / "training-windows.sqlite3"
    assert not (tmp_path / "shared" / "windows" / "training-windows.sqlite3-wal").exists()
    dest = tmp_path / "linux" / "training.sqlite3"
    report = merge_training_db(dest, snapshot, all_telemetry=True)
    assert report.ok and report.rows.get("hardware_samples") == 1 and report.rows.get("host_samples") == 1


def test_full_history_import_from_another_installs_database(tmp_path):
    import app
    from dual_gpu_setup.exchange import import_runs_db

    src = make_runs_db(tmp_path / "windows" / "chatbot.sqlite3", "Windows", ["a", "b"])
    with src.connect() as c:                            # a pre-capture row: no host_os at all
        c.execute("UPDATE runs SET host_os_json = NULL, origin_os = NULL WHERE id = 'b'")
    dest = make_runs_db(tmp_path / "linux" / "chatbot.sqlite3", "Linux", ["l"])
    before = datasafe.table_counts(dest.path)
    result = import_runs_db(dest.path, src.path)
    assert result["ok"] and result["imported"]["runs"] == 2 and result["imported"]["metric_samples"] == 2
    assert import_runs_db(dest.path, src.path)["imported"]["runs"] == 0       # repeatable
    assert datasafe.verify_counts_not_lower(before, datasafe.table_counts(dest.path)) == []
    assert dest.get("a")["os"] == "Windows" and dest.get("a")["origin_os_inferred"] == 0
    assert dest.get("b")["os"] == "Windows" and dest.get("b")["origin_os_inferred"] == 1   # a guess, and flagged as one
    assert "out-a" == dest.get("a")["output_text"] and app.RunStore(src.path).get("a")["output_text"] == "out-a"  # source untouched


def test_import_never_touches_a_missing_or_foreign_source(tmp_path):
    from dual_gpu_setup.exchange import import_runs_db

    dest = make_runs_db(tmp_path / "d.sqlite3", "Linux", ["l"])
    assert import_runs_db(dest.path, tmp_path / "nope.sqlite3")["ok"] is False
    other = tmp_path / "other.sqlite3"
    sqlite3.connect(other).execute("CREATE TABLE x (a)").connection.commit()
    assert "not an inference run database" in import_runs_db(dest.path, other)["error"]


def test_a_database_from_newer_code_is_refused_not_downgraded(tmp_path):
    import pytest

    import app
    from dual_gpu_setup.training.db import SCHEMA_VERSION

    run_db = tmp_path / "chatbot.sqlite3"
    app.RunStore(run_db)
    c = sqlite3.connect(run_db)
    c.execute(f"PRAGMA user_version = {app.RUN_SCHEMA_VERSION + 1}")
    c.close()
    with pytest.raises(RuntimeError, match="newer than this code"):
        app.RunStore(run_db)
    t = tmp_path / "t.sqlite3"
    TrainingStore(t)
    c = sqlite3.connect(t)
    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    c.close()
    with pytest.raises(RuntimeError, match="newer than this code"):
        TrainingStore(t)


def test_model_fingerprint_ignores_the_path(tmp_path):
    from dual_gpu_setup.gguf import fingerprint

    data = bytes(range(256)) * 20000
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "m.gguf").write_bytes(data)
    (tmp_path / "b" / "renamed.gguf").write_bytes(data)
    (tmp_path / "b" / "other.gguf").write_bytes(data[:-1] + b"x")
    assert fingerprint(tmp_path / "a" / "m.gguf") == fingerprint(tmp_path / "b" / "renamed.gguf")
    assert fingerprint(tmp_path / "a" / "m.gguf") != fingerprint(tmp_path / "b" / "other.gguf")
    assert fingerprint(tmp_path / "missing.gguf") is None


def test_storage_warning_for_a_database_on_an_ntfs_partition(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text(f"/dev/sda1 / ext4 rw 0 0\n/dev/sdb2 {tmp_path} fuseblk rw 0 0\n")
    assert "fuseblk" in datasafe.storage_warnings(tmp_path / "data", str(mounts))[0]
    mounts.write_text("/dev/sda1 / ext4 rw 0 0\n")
    assert datasafe.storage_warnings(tmp_path / "data", str(mounts)) == []


def test_service_logs_rotate_into_compressed_files_and_nothing_is_deleted(tmp_path):
    import logging

    from dual_gpu_setup.logs import KeepEverythingRotatingHandler, is_routine_poll

    handler = KeepEverythingRotatingHandler(tmp_path / "service.log", max_bytes=400)
    logger = logging.getLogger("rot-test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    for i in range(60):
        logger.info("line %03d %s", i, "x" * 30)
    handler.close()
    packed = sorted(tmp_path.glob("service.log.*.gz"))
    text = "".join(gzip.open(p, "rt").read() for p in packed) + (tmp_path / "service.log").read_text()
    assert len(packed) >= 2 and all(f"line {i:03d}" in text for i in range(60))     # every line survives rotation
    assert is_routine_poll('GET /api/status HTTP/1.1', 200) and not is_routine_poll('POST /api/chat HTTP/1.1', 200)
    assert not is_routine_poll('GET /api/runs/x HTTP/1.1', 500)                       # failures are always logged


def test_backup_is_a_single_file_with_no_wal_side_files(tmp_path):
    db = tmp_path / "chatbot.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("CREATE TABLE runs (id TEXT)")
    connection.execute("INSERT INTO runs VALUES ('a')")
    connection.commit()
    result = datasafe.backup_sqlite(db, tmp_path / "backups", "v1")
    connection.close()
    assert result.ok and sorted(p.name for p in (tmp_path / "backups").iterdir()) == ["chatbot.sqlite3.v1.bak"]


# ------------------------------------------------------------ training quality metrics


def _run_with_windows(live, run_id="q1", out_dir=None, name="q1"):
    cfg = {"out_dir": str(out_dir), "run_name": name} if out_dir else {}
    assert live.post("/api/training/runs", {"id": run_id, "name": name, "config": cfg, "lanes": []})[0] == 200
    rows = [{"step": i * 10, "window_steps": 10, "phase": "train", "loss": 4.0 - i * 0.1, "lr": 1e-4, "grad_norm": 1.0,
             "nonpad_tokens": 100, "step_time_ms": 500} for i in range(1, 11)]
    for r in rows:
        r["extra"] = {"train/z_loss_term": 0.01 * r["step"], "train/objective_total": r["loss"] + 0.01}
    assert live.post(f"/api/training/runs/{run_id}/metrics", {"steps": rows, "heartbeat": {"last_step": 100}})[0] == 200


def test_rolling_mean_is_step_weighted_and_windowed():
    from dual_gpu_setup.training.quality import rolling_mean

    rows = [(10, 4.0, 10), (20, 2.0, 10), (30, 6.0, 10)]
    out = rolling_mean(rows, window_steps=20)
    assert out[0] == [10, 4.0] and out[1] == [20, 3.0]
    assert out[2] == [30, 4.0]            # the step-10 row has left the 20-step window; (2 + 6) / 2


def test_validation_losses_come_from_the_trainers_metrics_file_per_set_and_ingest_is_idempotent(live_app, tmp_path):
    live = live_app()
    run_dir = tmp_path / "out" / "q1"
    run_dir.mkdir(parents=True)
    import time as _time

    now = _time.time()
    lines = [{"step": 50, "time": now, "train/loss": 2.0},                                        # no val keys: skipped
             {"step": 50, "time": now, "val/pretrain_loss": 3.1, "val/pretrain_ppl": 22.2, "val/resume_loss": 1.9},
             {"step": 100, "time": now, "val/pretrain_loss": 2.9, "val/pretrain_ppl": 18.1, "val/jd_loss": 2.4}]
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    _run_with_windows(live, out_dir=tmp_path / "out")
    first = live.post("/api/training/runs/q1/metrics/ingest", {})[1]
    assert first["read"] == 2 and first["stored"] == 2
    assert live.post("/api/training/runs/q1/metrics/ingest", {})[1]["stored"] == 0
    status, q = live.get("/api/training/runs/q1/quality?window=30")
    assert status == 200 and q["window"] == {"size": 30, "unit": "steps"}
    sets = q["validation"]["sets"]
    assert set(sets) == {"pretrain", "resume", "jd"}                         # sets are discovered, not hard-coded
    assert sets["pretrain"]["loss"] == [[50, 3.1], [100, 2.9]] and sets["pretrain"]["ppl"][-1] == [100, 18.1]
    assert q["validation"]["latest"]["jd"]["loss"] == [100, 2.4]
    assert len(q["loss"]) == 10 and len(q["loss_rolling"]) == 10 and q["loss_rolling"][-1][1] > q["loss"][-1][1]
    assert q["z_loss_term"][-1] == [100, 1.0] and q["objective_total"][0][0] == 10   # from the stored step scalars
    assert q["grad_norm"] and q["lr"]


def test_service_start_fills_validation_history_for_runs_that_finished_earlier(live_app, tmp_path):
    live = live_app()
    run_dir = tmp_path / "out" / "q1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 10, "val/pretrain_loss": 3.0}) + "\n")
    _run_with_windows(live, out_dir=tmp_path / "out")
    live.state.service.rescan_unreported_checkpoints()
    assert live.get("/api/training/runs/q1/quality")[1]["validation"]["sets"]["pretrain"]["loss"] == [[10, 3.0]]


def test_checkpoints_in_a_shared_output_dir_go_only_to_the_run_that_wrote_them(live_app, tmp_path):
    import os

    live = live_app()
    out = tmp_path / "out"
    run_dir = out / "shared"
    for step in (50, 500):
        d = run_dir / f"step_{step:07d}"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"w")
    # an early run that was cancelled at step 60 and a later run that resumed in the same directory
    assert live.post("/api/training/runs", {"id": "early", "name": "shared", "config": {"out_dir": str(out), "run_name": "shared"}, "lanes": []})[0] == 200
    live.post("/api/training/runs/early/metrics", {"steps": [{"step": 60, "window_steps": 10, "phase": "train", "loss": 3.0}], "heartbeat": {"last_step": 60}})
    live.post("/api/training/runs/early/finish", {"status": "cancelled"})
    assert live.post("/api/training/runs", {"id": "late", "name": "shared", "config": {"out_dir": str(out), "run_name": "shared"}, "lanes": []})[0] == 200
    live.post("/api/training/runs/late/metrics", {"steps": [{"step": 500, "window_steps": 10, "phase": "train", "loss": 2.0}], "heartbeat": {"last_step": 500}})
    early = live.post("/api/training/runs/early/checkpoints/scan", {})[1]
    late = live.post("/api/training/runs/late/checkpoints/scan", {})[1]
    assert [a["step"] for a in early["added"]] == [50] and early["belong_to_another_run"] == 1   # step 500 is past its last step
    assert sorted(a["step"] for a in late["added"]) == [50, 500]   # both inside its window and not past its last step
    old = out / "shared" / "step_0000050"                                                         # a checkpoint from long before the run
    os.utime(old / "model.safetensors", (1_000_000, 1_000_000))
    forced = live.post("/api/training/runs/late/checkpoints/scan", {"all": True})[1]
    assert forced["belong_to_another_run"] == 0


# ------------------------------------------------- GPU modes: lane switch, kernel VRAM, one card


@pytest.mark.real_sysfs
def test_free_vram_comes_from_kernel_files_without_starting_llama_server(tmp_path):
    from dual_gpu_setup.config import LaneConfig
    from dual_gpu_setup.lmstudio import free_from_sysfs

    for card, bus, total, used in (("card0", "0000:03:00.0", 32, 8), ("card1", "0000:07:00.0", 16, 15)):
        dev = tmp_path / "class" / "drm" / card / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text("0x1002\n")
        (dev / "uevent").write_text(f"PCI_SLOT_NAME={bus}\n")
        (dev / "mem_info_vram_total").write_text(f"{total * 1024**3}\n")
        (dev / "mem_info_vram_used").write_text(f"{used * 1024**3}\n")
    lanes = [LaneConfig("r9700", "R9700", "R9700", 32.0, 8081, pci_bus="0000:03:00.0"),
             LaneConfig("9070xt", "9070 XT", "9070", 16.0, 8082, pci_bus="0000:07:00.0")]
    assert free_from_sysfs(lanes, str(tmp_path)) == {"r9700": 24.0, "9070xt": 1.0}


def test_lane_switched_off_for_automatic_use_is_never_chosen_but_can_still_be_named(tmp_path, monkeypatch):
    from test_model_manager import _manager

    manager = _manager(tmp_path, {("a", "r9700"), ("a", "9070xt")})
    for lane in manager.config.lanes:      # the lanes are shared module-level objects: restore them after the test
        monkeypatch.setattr(lane, "auto_use", True)
    assert {l.key for l in manager.automatic_lanes()} == {"r9700", "9070xt"}
    assert manager.set_lane_auto_use("9070xt", False) == {"lane": "9070xt", "auto_use": False}
    assert [l.key for l in manager.automatic_lanes()] == ["r9700"]
    assert manager._lane("9070xt").key == "9070xt"                      # an explicit request can still name it
    manager.set_lane_auto_use("r9700", False)
    import app
    import pytest

    with pytest.raises(app.AppError, match="switched off for automatic use"):
        manager.automatic_lanes()


def test_a_single_lane_config_starts(tmp_path):
    from dual_gpu_setup.config import load_config

    text = (Path(__file__).resolve().parent.parent / "example.dual_gpu.ubuntu.toml").read_text(encoding="utf-8")
    lines, kept, skipping, seen = text.splitlines(), [], False, 0
    for line in lines:                      # drop the second [[lanes]] table
        if line.strip() == "[[lanes]]":
            seen += 1
            skipping = seen == 2
        elif line.startswith("[") and not line.startswith("[[lanes"):
            skipping = False
        if not skipping:
            kept.append(line)
    cfg = tmp_path / "one.toml"
    cfg.write_text("\n".join(kept), encoding="utf-8")
    assert len(load_config(cfg).lanes) == 1


def test_lane_switch_is_available_over_http_and_shown_in_the_catalog(live_app):
    live = live_app()
    assert live.post("/api/lanes/auto-use", {"lane": "9070xt", "enabled": False})[1] == {"lane": "9070xt", "auto_use": False}
    lanes = {l["key"]: l["auto_use"] for l in live.state.manager.catalog()["lanes"]}
    assert lanes["9070xt"] is False and lanes["r9700"] is True
    assert live.post("/api/lanes/auto-use", {"lane": "9070xt", "enabled": "no"})[0] == 400


# ------------------------------------------------------------------- split-model canary


class _FakeLlama:
    """A llama-server stand-in whose answers a test scripts."""

    def __init__(self, answers):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.answers, self.calls = list(answers), 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                text = outer.answers[min(outer.calls, len(outer.answers) - 1)]
                outer.calls += 1
                body = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.mark.parametrize("answers,ok", [
    (["27", "27"], True),
    (["The answer is 27.", "The answer is 27."], True),
    (["", ""], False),                                       # empty
    (["\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09", "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"], False),   # not text
    (["the the the the the the the the the the the the", "the the the the the the the the the the the the"], False),
    (["27", "31"], False),                                   # greedy decoding disagreed with itself
])
def test_split_canary_verdicts(answers, ok):
    from dual_gpu_setup.canary import run_split_canary

    fake = _FakeLlama(answers)
    try:
        result = run_split_canary(fake.url, "m", timeout=5)
    finally:
        fake.close()
    assert result["ok"] is ok and result["detail"]
    assert result["expected_answer_seen"] is ("27" in answers[0])


def test_split_canary_fails_when_the_server_is_unreachable():
    from dual_gpu_setup.canary import run_split_canary

    assert run_split_canary("http://127.0.0.1:9", "m", timeout=1)["ok"] is False


def _check_split_manager(tmp_path, mode, answers):
    from types import SimpleNamespace

    import app

    manager = object.__new__(app.DeploymentManager)
    manager.config = SimpleNamespace(policy=SimpleNamespace(split_canary=mode))
    fake = _FakeLlama(answers)
    stopped = []
    process = SimpleNamespace(base_url=fake.url, stop=lambda: stopped.append(True))
    spec = {"multi": True, "target": "primary", "model": SimpleNamespace(name="big")}
    return manager, spec, process, stopped, fake


def test_a_corrupt_split_is_refused_in_refuse_mode_and_only_recorded_in_warn_mode(tmp_path):
    import app

    manager, spec, process, stopped, fake = _check_split_manager(tmp_path, "refuse", ["", ""])
    try:
        with pytest.raises(app.AppError, match="failed the correctness canary"):
            manager._check_split(spec, process)
        assert stopped == [True] and manager.split_canary_results["primary"]["ok"] is False
    finally:
        fake.close()
    manager, spec, process, stopped, fake = _check_split_manager(tmp_path, "warn", ["", ""])
    try:
        manager._check_split(spec, process)                            # recorded, not refused
        assert stopped == [] and manager.split_canary_results["primary"]["ok"] is False
        manager.config.policy.split_canary = "off"
        manager.split_canary_results.clear()
        manager._check_split(spec, process)
        assert manager.split_canary_results == {}                      # off: no request is made
        manager.config.policy.split_canary = "refuse"
        manager._check_split({**spec, "multi": False}, process)        # single-GPU loads are never canaried
        assert manager.split_canary_results == {}
    finally:
        fake.close()


def test_isolation_is_off_by_default_and_hides_the_other_gpu_when_enabled():
    from types import SimpleNamespace

    from dual_gpu_setup.server import isolated_device

    off = SimpleNamespace(policy=SimpleNamespace(isolate_single_lane=False))
    on = SimpleNamespace(policy=SimpleNamespace(isolate_single_lane=True))
    assert isolated_device(off, "rocm", "ROCm1") == ("ROCm1", {})
    assert isolated_device(on, "rocm", "ROCm1") == ("ROCm0", {"HIP_VISIBLE_DEVICES": "1"})
    assert isolated_device(on, "vulkan", "Vulkan1") == ("Vulkan1", {})
    assert isolated_device(on, "rocm", "weird") == ("weird", {})


# ------------------------------------------------------- inference benchmark baseline


def _load_bench():
    import importlib.util

    spec = importlib.util.spec_from_file_location("benchmark_inference", Path(__file__).resolve().parent.parent / "scripts" / "benchmark_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_baseline_records_speed_and_energy_and_compare_flags_regressions(live_app, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Fake(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"usage": {"completion_tokens": 100, "prompt_tokens": 20},
                               "timings": {"predicted_per_second": 50.0, "prompt_per_second": 800.0}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            body = json.dumps({"gpus": {"b": {"energy_wh": 0.5, "temp_junction_c": {"max": 88.0}}}, "host": {}}).encode() \
                if "summary" in self.path else json.dumps({"servers": []}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    bench = _load_bench()
    try:
        result = bench.run_benchmark(f"http://127.0.0.1:{server.server_port}", "m", "r9700", reps=2, prompts=bench.PROMPTS[:1])
    finally:
        server.shutdown()
        server.server_close()
    s = result["summary"]
    assert s["generation_tokens_per_s"] == 50.0 and s["prompt_tokens_per_s"] == 800.0
    assert s["tokens_per_watt_hour"] == pytest.approx(400.0)      # 2 x 100 tokens over 0.5 Wh
    assert s["peak_junction_c"] == 88.0 and result["host_os"] and len(result["samples"]) == 2
    same = bench.compare(result, result)
    assert same["ok"] and not same["problems"]
    slower = json.loads(json.dumps(result))
    slower["summary"]["generation_tokens_per_s"] = 45.0           # -10 %
    verdict = bench.compare(result, slower, tolerance=0.03)
    assert not verdict["ok"] and "generation_tokens_per_s fell 10.0%" in verdict["problems"][0]
    assert bench.compare(result, {**slower, "host_os": "Elsewhere"}, 0.5)["problems"][0].startswith("host_os differs")


# ----------------------------------------------------- crash recovery with back-off


def _supervised(auto_restart, max_crashes=3):
    import threading
    from types import SimpleNamespace

    import app

    manager = object.__new__(app.DeploymentManager)
    manager.config = SimpleNamespace(policy=SimpleNamespace(auto_restart=auto_restart, auto_restart_max=max_crashes, auto_restart_window_s=600))
    manager.lock = threading.RLock()
    manager.transition_lock = threading.Lock()
    manager.state, manager.error = "ready", None
    events, restarts = [], []
    manager.on_event = lambda kind, severity, payload, lane=None: events.append((kind, lane))
    proc = SimpleNamespace(pid=100, returncode=None, poll=lambda: proc.returncode)
    process = SimpleNamespace(proc=proc, restart=lambda: (restarts.append(1), setattr(proc, "returncode", None), setattr(proc, "pid", proc.pid + 1)))
    handle = SimpleNamespace(process=process, lanes=[SimpleNamespace(key="r9700")], model=SimpleNamespace(name="qwen"))
    manager.handles = {"r9700": handle}
    return manager, proc, events, restarts


def test_a_crash_is_recorded_even_when_auto_restart_is_off():
    manager, proc, events, restarts = _supervised(auto_restart=False)
    assert manager.supervise_once(now=0) == [] and events == []        # healthy: nothing happens
    proc.returncode = -9
    manager.supervise_once(now=1)
    manager.supervise_once(now=2)                                      # the same crash is not recorded twice
    assert events == [("server_crash", "r9700")] and restarts == []


def test_auto_restart_waits_backs_off_and_restarts_on_the_same_lane():
    manager, proc, events, restarts = _supervised(auto_restart=True)
    proc.returncode = 1
    assert manager.supervise_once(now=100) == ["restart of r9700 in 5s"]
    assert restarts == []                                              # not before the back-off
    assert manager.supervise_once(now=104) == [] and restarts == []
    assert manager.supervise_once(now=106) == ["restarted r9700"] and restarts == [1]
    assert [e[0] for e in events] == ["server_crash", "server_restarted"] and all(e[1] == "r9700" for e in events)
    proc.returncode = 1                                                # it dies again: longer back-off
    assert manager.supervise_once(now=200) == ["restart of r9700 in 15s"]


def test_a_crash_loop_stops_automatic_restarts():
    manager, proc, events, restarts = _supervised(auto_restart=True, max_crashes=2)
    now = 0.0
    for expected in ("restart of r9700 in 5s", "restart of r9700 in 15s", "gave up on r9700"):
        proc.returncode = 1
        assert manager.supervise_once(now=now) == [expected]
        now += 30
        manager.supervise_once(now=now)                                # let any due restart run
        now += 1
    assert ("crash_loop", "r9700") in events and len(restarts) == 2
    proc.returncode = 1
    proc.pid += 1
    assert manager.supervise_once(now=now + 5) == ["gave up on r9700"]  # still inside the window: no more restarts


def test_restart_is_skipped_while_a_deploy_holds_the_transition_lock_and_a_failed_restart_is_recorded():
    manager, proc, events, restarts = _supervised(auto_restart=True)
    proc.returncode = 1
    manager.supervise_once(now=0)
    manager.transition_lock.acquire()
    assert manager.supervise_once(now=60) == [] and restarts == []     # a deploy is running: wait
    manager.transition_lock.release()
    manager.handles["r9700"].process.restart = lambda: (_ for _ in ()).throw(RuntimeError("port busy"))
    assert manager.supervise_once(now=61) == ["restart of r9700 failed"]
    assert ("restart_failed", "r9700") in events


def test_connections_beyond_the_cap_get_503_instead_of_piling_up_threads(live_app):
    import socket
    import threading
    import time

    import app

    live = live_app()
    server = app.ChatbotHTTPServer(("127.0.0.1", 0), live.state, max_connections=2)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        held = []
        for _ in range(2):                                   # two silent connections occupy both slots
            sock = socket.create_connection(("127.0.0.1", port))
            held.append(sock)
        time.sleep(0.3)
        third = socket.create_connection(("127.0.0.1", port))
        third.sendall(b"GET /api/health HTTP/1.1\r\nHost: x\r\n\r\n")
        third.settimeout(3)
        assert b"503" in third.recv(200).split(b"\r\n")[0]
        assert server.rejected_connections == 1
        for sock in held:
            sock.close()
        time.sleep(0.5)                                      # slots are released when the clients go away
        fresh = socket.create_connection(("127.0.0.1", port))
        fresh.sendall(b"GET /api/health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        fresh.settimeout(3)
        assert b"200" in fresh.recv(200).split(b"\r\n")[0]
    finally:
        server.shutdown()
        server.server_close()


def test_control_posts_refuse_a_form_content_type(live_app):
    import urllib.request

    live = live_app()
    for path in ("/api/deploy", "/api/unload", "/api/models/refresh", "/api/lanes/auto-use"):
        req = urllib.request.Request(live.base + path, data=b"lane=r9700", method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError(f"{path} accepted a form POST")
        except urllib.error.HTTPError as exc:
            assert exc.code == 415, path
    assert live.post("/api/models/refresh", {})[0] == 200          # the same call as JSON still works


def test_validation_metrics_in_a_shared_metrics_file_go_only_to_the_run_that_logged_them(live_app, tmp_path):
    import time as _time

    live = live_app()
    out = tmp_path / "out"
    (out / "shared").mkdir(parents=True)
    now = _time.time()
    lines = [{"step": 50, "time": now - 5, "val/pretrain_loss": 3.0}, {"step": 500, "time": now - 1, "val/pretrain_loss": 2.0}]
    (out / "shared" / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    cfg = {"out_dir": str(out), "run_name": "shared"}
    live.post("/api/training/runs", {"id": "early", "name": "shared", "config": cfg, "lanes": []})
    live.post("/api/training/runs/early/metrics", {"steps": [{"step": 60, "window_steps": 10, "phase": "train", "loss": 3.0}], "heartbeat": {"last_step": 60}})
    live.post("/api/training/runs/early/finish", {"status": "cancelled"})
    live.post("/api/training/runs", {"id": "late", "name": "shared", "config": cfg, "lanes": []})
    live.post("/api/training/runs/late/metrics", {"steps": [{"step": 500, "window_steps": 10, "phase": "train", "loss": 2.0}], "heartbeat": {"last_step": 500}})
    # simulate the earlier bug: every line already stored against the early run
    with live.state.training.connect() as c:
        for step, value in ((50, 3.0), (500, 2.0)):
            c.execute("INSERT INTO training_eval_points (run_id, step, ts, source, metrics_json) VALUES ('early', ?, ?, 'metrics.jsonl', ?)",
                      (step, now, json.dumps({"val/pretrain_loss": value})))
    assert live.post("/api/training/runs/early/metrics/ingest", {})[1]["stored"] == 0
    early = live.get("/api/training/runs/early/quality")[1]["validation"]["sets"]["pretrain"]["loss"]
    late = live.post("/api/training/runs/late/metrics/ingest", {})[1]
    assert early == [[50, 3.0]]                        # the step-500 copy (past its last step 60) was removed
    assert late["stored"] == 2
