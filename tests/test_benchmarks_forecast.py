from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_gpu_setup.training import memory_forecast  # noqa: E402
from dual_gpu_setup.training.benchmarks import BenchmarkRunner, build_report, expand_grid  # noqa: E402
from dual_gpu_setup.training.store import TrainingStore  # noqa: E402

GIB = 1024**3
REPO = str(Path(__file__).resolve().parent.parent)

TRIAL_SCRIPT = '''
import os, sys, time
sys.path.insert(0, sys.argv[1])
from dual_gpu_setup.training.tracker import TrainingTracker

batch = int(os.environ["DGPU_PARAM_BATCH_SIZE"])
if batch >= 16:
    print("HIP out of memory. Tried to allocate 2.00 GiB", file=sys.stderr)
    sys.exit(1)
TrainingTracker._memory_stats = staticmethod(lambda: {"mem_peak_bytes": batch * 1024**3})  # stands in for torch.cuda
tracker = TrainingTracker(log_every=5, warmup_windows=0, capture_environment=False, flush_s=0.05).start()
for step in range(1, 21):
    tracker.microbatch_counts(batch * 100, batch * 80, batch * 30)
    time.sleep(0.01)
    tracker.step_completed(step)
tracker.finish("completed")
'''


# ---------------------------------------------------------------- benchmarks


def test_plan_validation_and_grid_expansion():
    assert expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})[0] == {"a": 1, "b": "x"} and len(expand_grid({"a": [1, 2], "b": [1, 2, 3]})) == 6
    for bad in ({"kind": "nope", "command": ["x"], "grid": {"a": [1]}}, {"command": [], "grid": {"a": [1]}},
                {"command": ["x"], "grid": {}}, {"command": ["x"], "grid": {"a": []}},
                {"command": ["x"], "grid": {"a": list(range(9)), "b": list(range(9))}}):
        with pytest.raises(ValueError):
            BenchmarkRunner.validate(bad)


def test_batch_size_sweep_recommends_the_fastest_config_that_fits_and_maps_the_oom_boundary(live_app, tmp_path):
    live = live_app()
    script = tmp_path / "trial.py"
    script.write_text(TRIAL_SCRIPT)
    status, created = live.post("/api/training/benchmarks", {
        "name": "bs-sweep", "kind": "batch_size", "command": [sys.executable, str(script), REPO],
        "grid": {"batch_size": [1, 2, 4, 16]}, "steps": 20, "timeout_s": 120, "vram_budget_bytes": 3 * GIB, "headroom_fraction": 0.0})
    assert status == 200
    deadline = time.time() + 120
    while time.time() < deadline:
        data = live.get(f"/api/training/benchmarks/{created['id']}")[1]
        if data["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.5)
    assert data["state"] == "completed", data
    states = {t["params"]["batch_size"]: t["state"] for t in data["trials"]}
    assert states == {1: "completed", 2: "completed", 4: "completed", 16: "oom"}
    report = data["report"]
    assert report["recommended"]["params"] == {"batch_size": 2}  # batch 4 is faster but needs 4 GiB > the 3 GiB budget
    assert report["oom_boundary"] == [{"batch_size": 16}] and "never changed automatically" in report["note"]
    rates = {r["params"]["batch_size"]: r["tokens_per_sec"] for r in report["ranking"]}
    assert rates[2] > rates[1]
    # Every trial is an ordinary run grouped by its benchmark id, and none left its GPU lease behind.
    runs = live.get("/api/training/runs")[1]["runs"]
    assert {r["tags"].get("benchmark_id") for r in runs} == {created["id"]}
    assert live.state.leases.lane_holders() == {}


def test_gradient_checkpointing_report_shows_the_memory_vs_time_trade():
    def trial(flag, tps, mem):
        return {"state": "completed", "params": {"gradient_checkpointing": flag}, "run_id": str(flag),
                "result": {"tokens_per_sec": tps, "peak_mem_bytes": mem, "step_time_ms": 1.0}}
    report = build_report([trial(False, 5000.0, 20 * GIB), trial(True, 3500.0, 11 * GIB)], {"kind": "gradient_checkpointing"})
    ckpt = report["gradient_checkpointing"]
    assert ckpt["memory_saved_pct"] == pytest.approx(45.0) and ckpt["time_added_pct"] == pytest.approx((5000 / 3500 - 1) * 100)


def test_trial_that_never_reports_telemetry_is_flagged(live_app, tmp_path):
    live = live_app()
    runner = live.state.benchmarks
    plan = {"kind": "custom", "command": [sys.executable, "-c", "print('no tracker here')"], "grid": {"x": [1]}, "steps": 5}
    benchmark_id = runner.create(plan)
    runner.start(benchmark_id, background=False)
    trial = runner.get(benchmark_id)["trials"][0]
    assert trial["state"] == "no_telemetry" and "TrainingTracker" in trial["error"]


# ------------------------------------------------------------------ forecast

CONFIG = {"base_model": "org/3b", "max_seq_length": 4096, "per_device_train_batch_size": 2, "bf16": True, "lora_r": 16,
          "gradient_checkpointing": False, "total_params": 3e9, "trainable_params": 30e6, "hidden_size": 3072,
          "num_hidden_layers": 28, "num_attention_heads": 24, "vocab_size": 128256}


def test_estimate_has_a_component_breakdown_and_checkpointing_saves_activations():
    plain = memory_forecast.estimate(CONFIG)
    checkpointed = memory_forecast.estimate({**CONFIG, "gradient_checkpointing": True})
    assert plain["estimate"] and plain["complete"]
    assert plain["components"]["weights"] == pytest.approx(3e9 * 2)  # bf16 base
    assert plain["components"]["optimizer_state"] == pytest.approx(30e6 * 8)  # adapter-only Adam state
    assert checkpointed["components"]["activations"] < plain["components"]["activations"] / 5
    assert 8 * GIB < plain["total_bytes"] < 80 * GIB
    assert memory_forecast.estimate({"total_params": 3e9})["missing"] == ["seq_len", "microbatch"]


def test_measured_profile_replaces_the_estimate_and_gates_the_16gb_card(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    lanes = {"big": {"key": "r9700", "vram_gb": 32}, "small": {"key": "9070xt", "vram_gb": 16}}
    before = memory_forecast.forecast(store, CONFIG, lanes["small"])
    assert before["source"] == "estimate" and before["fits"] is not None
    store.create_run({"id": "r", "name": "x", "config": CONFIG, "lanes": ["9070xt"]})
    store.record_metrics("r", {"steps": [{"step": 10, "window_steps": 10, "phase": "train", "nonpad_tokens": 1000, "step_time_ms": 100,
                                          "mem_peak_bytes": int(15.5 * GIB)}]})
    store.finish_run("r", {"status": "completed"})
    assert memory_forecast.record_run_profile(store, store.get_run("r"), "9070xt") is True
    after = memory_forecast.forecast(store, CONFIG, lanes["small"])
    assert after["source"] == "measured" and after["peak_bytes"] == int(15.5 * GIB)
    assert after["fits"] is False  # 15.5 GiB exceeds 90% of 16 GiB: the measurement vetoes what the estimate allowed
    assert memory_forecast.forecast(store, CONFIG, lanes["big"])["source"] == "estimate"  # different GPU = different key
    failed = {**store.get_run("r"), "status": "failed"}
    assert memory_forecast.record_run_profile(store, failed, "9070xt") is False
