from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib import request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402
from dual_gpu_setup.training.checkpoints import find_latest_valid, validate_manifest, write_manifest  # noqa: E402
from dual_gpu_setup.training.hardware import (  # noqa: E402
    HardwareSampler, assign_lanes, classify_kernel_line, discover_gpus, sample_gpu,
)
from dual_gpu_setup.training.store import TrainingError, TrainingStore  # noqa: E402
from dual_gpu_setup.training.tracker import TrainingTracker  # noqa: E402


class Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_store(tmp_path: Path, clock: Clock | None = None) -> tuple[TrainingStore, Clock]:
    clock = clock or Clock()
    return TrainingStore(tmp_path / "t.sqlite3", clock=clock), clock


def window(step: int, **overrides) -> dict:
    record = {
        "step": step, "window_steps": 10, "phase": "train", "loss": 2.0, "lr": 1e-4, "grad_norm": 0.5,
        "processed_tokens": 10000, "nonpad_tokens": 8000, "supervised_tokens": 4000,
        "step_time_ms": 500.0, "mem_peak_bytes": 20 * 1024**3,
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------- store


def test_create_run_is_idempotent_and_requires_name(tmp_path):
    store, _ = make_store(tmp_path)
    with pytest.raises(TrainingError):
        store.create_run({})
    first = store.create_run({"id": "abc", "name": "lora", "max_steps": 100})
    again = store.create_run({"id": "abc", "name": "lora"})
    assert first["id"] == again["id"] == "abc"
    assert store.list_runs()["total"] == 1


def test_derived_rates_and_unmeasured_values_stay_null(tmp_path):
    store, _ = make_store(tmp_path)
    store.create_run({"id": "r", "name": "x", "max_steps": 100})
    store.record_metrics("r", {"steps": [window(10, phase="warmup"), window(20), window(30, loss=None, nonpad_tokens=None)]})
    run = store.get_run("r")
    # window: 10 steps * 500 ms = 5 s; 8000 nonpad tokens -> 1600 tok/s. Warmup is excluded.
    assert run["throughput"]["tokens_per_sec"] == pytest.approx(1600.0)
    assert run["throughput"]["supervised_tokens_per_sec"] == pytest.approx(800.0)
    assert run["tokens"]["padding_ratio"] == pytest.approx(0.2)
    points = store.series("r")["points"]
    assert points[-1]["tokens_per_sec"] is None and points[-1]["loss"] is None
    assert run["latest_scalars"]["step"] == 20


def test_non_finite_numbers_are_stored_as_null(tmp_path):
    store, _ = make_store(tmp_path)
    store.create_run({"id": "r", "name": "x"})
    store.record_metrics("r", {"steps": [window(5, loss=float("nan"), grad_norm=float("inf"))]})
    point = store.series("r")["points"][0]
    assert point["loss"] is None and point["grad_norm"] is None


def test_metric_replay_is_idempotent(tmp_path):
    store, _ = make_store(tmp_path)
    store.create_run({"id": "r", "name": "x"})
    for _ in range(3):
        store.record_metrics("r", {"steps": [window(10)]})
    assert store.series("r")["total_windows"] == 1


def test_stall_is_judged_on_step_progress_not_heartbeats(tmp_path):
    store, clock = make_store(tmp_path)
    store.create_run({"id": "r", "name": "x", "max_steps": 1000})
    store.record_metrics("r", {"steps": [window(10)], "heartbeat": {"last_step": 10}})
    assert store.get_run("r")["status"] == "running"
    # Heartbeats keep arriving but the step never advances -> stalled once past the floor.
    for _ in range(40):
        clock.now += 20
        store.record_metrics("r", {"steps": [], "heartbeat": {"last_step": 10}})
    assert store.get_run("r")["status"] == "stalled"
    store.record_metrics("r", {"steps": [], "heartbeat": {"last_step": 11}})
    assert store.get_run("r")["status"] == "running"


def test_missing_heartbeats_mark_the_run_lost_and_finish_is_terminal(tmp_path):
    store, clock = make_store(tmp_path)
    store.create_run({"id": "r", "name": "x"})
    clock.now += 600
    assert store.get_run("r")["status"] == "lost"
    run = store.finish_run("r", {"status": "failed", "error": "OOM"})
    assert run["status"] == "failed" and run["error"] == "OOM"
    with pytest.raises(TrainingError):
        store.finish_run("r", {"status": "bogus"})
    with pytest.raises(TrainingError) as missing:
        store.get_run("nope")
    assert missing.value.status == 404


def test_eta_uses_rolling_step_time(tmp_path):
    store, _ = make_store(tmp_path)
    store.create_run({"id": "r", "name": "x", "max_steps": 110})
    store.record_metrics("r", {"steps": [window(10)], "heartbeat": {"last_step": 10}})
    assert store.get_run("r")["eta_s"] == pytest.approx(100 * 0.5)


def test_checkpoint_requires_manifest_to_be_valid(tmp_path):
    store, _ = make_store(tmp_path)
    store.create_run({"id": "r", "name": "x"})
    store.add_checkpoint("r", {"step": 10, "path": "/a", "valid": True})
    store.add_checkpoint("r", {"step": 20, "path": "/b", "valid": True, "manifest": {"files": [1], "tokens_seen": 5}})
    by_step = {c["step"]: c for c in store.checkpoints("r")}
    assert by_step[10]["valid"] is False and by_step[20]["valid"] is True
    assert store.get_run("r")["checkpoints"] == {"total": 2, "valid": 1}


def test_hardware_summary_and_threshold_exposure(tmp_path):
    store, clock = make_store(tmp_path)
    store.create_run({"id": "r", "name": "x"})
    base = clock.now
    samples = []
    for i in range(10):
        samples.append({"ts": base + i, "pci_bus": "0000:03:00.0", "lane": "r9700", "gpu_busy_pct": 90,
                        "vram_used_bytes": 1000 + i, "temp_memory_c": 85 + i, "power_w": 300.0,
                        "throttle_status": 4 if i >= 8 else 0, "amd_smi_fields": ["throttle_status"],
                        "unavailable": {"temp_junction_c": "not exposed"}})
    samples.append({"ts": base + 1, "pci_bus": "0000:04:00.0", "lane": "9070xt"})  # nothing measured
    store.add_hardware_samples(samples)
    clock.now = base + 20
    summary = store.hardware_summary("r")["gpus"]
    gpu = summary["0000:03:00.0"]
    assert gpu["max_temp_memory_c"] == 94
    assert gpu["seconds_memory_over_alert"] == 5  # 90..94
    assert gpu["energy_wh"] == pytest.approx(300 * 10 / 3600)
    assert gpu["seconds_throttled"] == 2
    assert gpu["max_temp_junction_c"] is None and gpu["seconds_junction_over_alert"] is None
    assert summary["0000:04:00.0"]["avg_power_w"] is None
    point = store.hardware_series("r")["gpus"]["0000:03:00.0"]["points"][0]
    assert point["vram_used_bytes"] == 1000
    assert point["throttle_status"] == 0 and point["telemetry_sources"] == ["throttle_status"]
    assert point["unavailable"] == {"temp_junction_c": "not exposed"}


def test_samples_are_only_persisted_while_a_run_is_active(tmp_path):
    store, clock = make_store(tmp_path)
    assert store.has_active_runs() is False
    store.create_run({"id": "r", "name": "x"})
    assert store.has_active_runs() is True
    store.finish_run("r", {"status": "completed"})
    assert store.has_active_runs() is False


# ------------------------------------------------------------------ hardware


def build_sysfs(tmp_path: Path) -> Path:
    """Fake /sys tree. `device` is a real directory (no symlinks, no ':' in names) so it also
    works on Windows; the PCI address comes from uevent, exactly as on a real system."""
    root = tmp_path / "sys"
    drm = root / "class" / "drm"
    for card, bus, gb, product, mem in (
        ("card0", "0000:03:00.0", 32, "AMD Radeon AI PRO R9700", True),
        ("card1", "0000:08:00.0", 16, None, False),
    ):
        device = drm / card / "device"
        hw = device / "hwmon" / "hwmon3"
        hw.mkdir(parents=True)
        (device / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME={bus}\n")
        (device / "vendor").write_text("0x1002\n")
        (device / "gpu_busy_percent").write_text("37\n")
        (device / "mem_info_vram_total").write_text(str(gb * 1024**3) + "\n")
        (device / "mem_info_vram_used").write_text(str(2 * 1024**3) + "\n")
        (device / "current_link_speed").write_text("32.0 GT/s PCIe\n")
        (device / "current_link_width").write_text("8\n")
        if product:
            (device / "product_name").write_text(product + "\n")
        (hw / "temp1_label").write_text("edge\n")
        (hw / "temp1_input").write_text("55000\n")
        (hw / "temp2_label").write_text("junction\n")
        (hw / "temp2_input").write_text("78000\n")
        if mem:
            (hw / "temp3_label").write_text("mem\n")
            (hw / "temp3_input").write_text("82000\n")
        (hw / "power1_average").write_text("250000000\n")
    (drm / "card0-DP-1").mkdir()  # connector entries must be ignored
    return root

def test_discovery_reads_sensors_and_marks_missing_ones_unavailable(tmp_path):
    root = build_sysfs(tmp_path)
    gpus, reason = discover_gpus(root, platform_name="linux")
    assert reason is None and [g.pci_bus for g in gpus] == ["0000:03:00.0", "0000:08:00.0"]
    first, second = (sample_gpu(g, now=1.0) for g in gpus)
    assert first["gpu_busy_pct"] == 37 and first["temp_memory_c"] == 82.0 and first["power_w"] == 250.0
    assert first["pcie_speed"] == "32.0 GT/s PCIe" and first["pcie_width"] == 8
    assert second["temp_memory_c"] is None and "temp_memory_c" in second["unavailable"]
    assert second["fan_rpm"] is None  # null, never 0


def test_lane_assignment_prefers_pci_bus_then_name_then_vram(tmp_path):
    root = build_sysfs(tmp_path)
    gpus, _ = discover_gpus(root, platform_name="linux")
    lane = lambda key, match, gb, bus="": SimpleNamespace(key=key, match=match, vram_gb=gb, pci_bus=bus)  # noqa: E731
    assign_lanes(gpus, [lane("a", "9070 XT", 16), lane("b", "R9700", 32)])
    assert {g.pci_bus: g.lane for g in gpus} == {"0000:03:00.0": "b", "0000:08:00.0": "a"}  # name, then VRAM
    swapped, _ = discover_gpus(root, platform_name="linux")
    assign_lanes(swapped, [lane("x", "R9700", 32, "0000:08:00.0"), lane("y", "none", 99, "0000:03:00.0")])
    assert {g.pci_bus: g.lane for g in swapped} == {"0000:08:00.0": "x", "0000:03:00.0": "y"}  # pci_bus wins


def test_non_linux_reports_why_and_sampler_is_inert():
    sampler = HardwareSampler(platform_name="win32")
    sampler.start()
    snapshot = sampler.snapshot()
    assert snapshot["available"] is False and "Linux" in snapshot["unavailable_reason"]


def test_sampler_tracks_overhead_and_isolates_callback_failures(tmp_path):
    root = build_sysfs(tmp_path)

    def boom(_samples):
        raise RuntimeError("db down")

    sampler = HardwareSampler(sysfs_root=root, platform_name="linux", on_samples=boom)
    sampler.sample_once()
    sampler.sample_once()
    snapshot = sampler.snapshot()
    assert snapshot["sampler"]["ticks"] == 2 and snapshot["sampler"]["callback_errors"] == 2
    assert snapshot["sampler"]["avg_sample_ms"] is not None
    assert len(snapshot["gpus"][0]["history"]) == 2


def test_kernel_fault_classification():
    assert classify_kernel_line("amdgpu 0000:03:00.0: amdgpu: GPU reset begin!")[0] == "gpu_reset"
    assert classify_kernel_line("amdgpu 0000:03:00.0: [gfxhub] page fault (src_id:0 ring:0)")[0] == "page_fault"
    assert classify_kernel_line("amdgpu: ring gfx_0.0.0 timeout, signaled seq=1")[0] == "ring_timeout"
    assert classify_kernel_line("usb 1-1: new high-speed USB device") is None


# --------------------------------------------------------------- checkpoints


def make_checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(b"weights" * 100)
    (directory / "optimizer.pt").write_bytes(b"opt" * 100)
    return directory


def test_checkpoint_without_manifest_is_incomplete_and_never_selected(tmp_path):
    older = make_checkpoint(tmp_path / "checkpoint-100")
    write_manifest(older, step=100)
    newer = make_checkpoint(tmp_path / "checkpoint-200")  # killed mid-save: no manifest
    assert validate_manifest(newer)[0] is False
    assert find_latest_valid(tmp_path) == older


def test_manifest_detects_truncation_and_corruption(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "checkpoint-5")
    write_manifest(checkpoint, step=5, tokens_seen=123)
    assert validate_manifest(checkpoint, verify_hashes=True) == (True, None)
    target = checkpoint / "optimizer.pt"
    target.write_bytes(b"XYZ" * 100)  # same size, different bytes
    assert validate_manifest(checkpoint)[0] is True
    assert "hash mismatch" in validate_manifest(checkpoint, verify_hashes=True)[1]
    target.write_bytes(b"short")
    assert "size mismatch" in validate_manifest(checkpoint)[1]


# ------------------------------------------------------- tracker end-to-end


@pytest.fixture
def live_server(tmp_path):
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


def get_json(url: str) -> dict:
    with request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def run_short_training(tracker: TrainingTracker, steps: int = 6) -> None:
    for step in range(1, steps + 1):
        for _ in range(2):  # two microbatches per optimizer step (gradient accumulation)
            tracker.microbatch_counts(processed=400, nonpad=300, supervised=100)
        tracker.step_completed(step)
        if step % tracker.log_every == 0:
            tracker.log_scalars(step, loss=2.0 / step, lr=1e-4, grad_norm=0.7)


def test_tracker_streams_a_run_to_the_dashboard(live_server, tmp_path):
    state, base = live_server
    tracker = TrainingTracker(base, "e2e", lanes=["r9700"], log_every=3, max_steps=6, spool_dir=tmp_path / "spool",
                              flush_s=0.1, warmup_windows=0, capture_environment=False).start()
    run_short_training(tracker)
    checkpoint = make_checkpoint(tmp_path / "out" / "checkpoint-6")
    tracker.checkpoint(6, checkpoint, block=True)
    tracker.finish("completed")

    detail = get_json(f"{base}/api/training/runs/{tracker.run_id}")
    run = detail["run"]
    assert run["status"] == "completed" and run["last_step"] == 6
    # 6 steps * 2 microbatches * 300 non-padding tokens; counters accumulate across accumulation.
    assert run["tokens"]["nonpad"] == 3600 and run["tokens"]["supervised"] == 1200 and run["tokens"]["processed"] == 4800
    assert run["windows"] == 2 and run["latest_scalars"]["loss"] == pytest.approx(2.0 / 6)
    assert detail["checkpoints"][0]["valid"] is True
    assert {e["kind"] for e in detail["events"]} >= {"started", "finished"}
    series = get_json(f"{base}/api/training/runs/{tracker.run_id}/series")
    assert [p["step"] for p in series["steps"]["points"]] == [3, 6]
    assert get_json(f"{base}/api/training/runs")["total"] == 1
    assert "gpus" in get_json(f"{base}/api/hardware")["hardware"]


def test_tracker_spools_while_server_is_down_and_replays_in_order(tmp_path):
    spool = tmp_path / "spool"
    tracker = TrainingTracker("http://127.0.0.1:9", "offline", log_every=2, spool_dir=spool, flush_s=0.05,
                              timeout_s=0.3, warmup_windows=0, capture_environment=False).start()
    run_short_training(tracker, steps=4)  # must not raise or stall even though nothing is reachable
    deadline = time.time() + 5
    kinds: list[str] = []
    while time.time() < deadline and "metrics" not in kinds:
        time.sleep(0.05)
        if tracker.spool_path.exists():
            kinds = [json.loads(line)["kind"] for line in tracker.spool_path.read_text().splitlines()]
    assert kinds[0] == "create" and "metrics" in kinds

    config_path = Path(__file__).resolve().parent.parent / "example.dual_gpu.toml"
    state = app.ApplicationState(config_path, tmp_path / "data")
    server = app.ChatbotHTTPServer(("127.0.0.1", 0), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tracker.server = f"http://127.0.0.1:{server.server_address[1]}"
        tracker._last_replay = 0  # allow an immediate replay attempt
        tracker.finish("completed")
        run = state.training.get_run(tracker.run_id)
        assert run["status"] == "completed" and run["last_step"] == 4 and run["tokens"]["nonpad"] == 4 * 2 * 300
        assert not tracker.spool_path.exists()
    finally:
        server.shutdown()
        server.server_close()
        state.manager.shutdown()
        thread.join(timeout=5)


def test_tracker_disabled_is_a_noop(tmp_path):
    tracker = TrainingTracker("http://127.0.0.1:9", "off", enabled=False, log_every=3, spool_dir=tmp_path).start()
    run_short_training(tracker)
    tracker.finish()
    assert not list(tmp_path.iterdir())


def test_unknown_run_returns_404(live_server):
    _state, base = live_server
    with pytest.raises(Exception) as excinfo:
        get_json(f"{base}/api/training/runs/missing")
    assert "404" in str(excinfo.value)
