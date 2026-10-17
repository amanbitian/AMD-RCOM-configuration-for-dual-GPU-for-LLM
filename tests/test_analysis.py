from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_gpu_setup.training import analysis  # noqa: E402
from dual_gpu_setup.training.store import TrainingStore  # noqa: E402


BASE = 10_000.0


class Clock:
    def __init__(self, now=BASE):
        self.now = now

    def __call__(self):
        return self.now


def make_store(tmp_path, clock=None, **kwargs):
    clock = clock or Clock()
    return TrainingStore(tmp_path / "t.sqlite3", clock=clock, **kwargs), clock


def add_run(store, run_id, *, windows=40, loss_fn=lambda i: 3.0 * math.exp(-i / 15) + 0.5, tokens=1000, step_ms=500,
            start=BASE, config=None, lanes=("r9700",), seed=1, finish=True, tokens_fn=None, step_ms_fn=None, **create):
    store.create_run({"id": run_id, "name": run_id, "started_at": start, "config": config or {"lr": 1e-4}, "lanes": list(lanes),
                      "seed": seed, **create})
    rows = []
    for i in range(1, windows + 1):
        rows.append({"step": i * 10, "ts": start + i * 5, "window_steps": 10, "phase": "train", "loss": loss_fn(i),
                     "processed_tokens": 1200, "nonpad_tokens": (tokens_fn(i) if tokens_fn else tokens),
                     "supervised_tokens": 400, "step_time_ms": step_ms_fn(i) if step_ms_fn else step_ms,
                     "mem_peak_bytes": 20 * 1024**3})
    store.record_metrics(run_id, {"steps": rows, "heartbeat": {"last_step": windows * 10}})
    if finish:
        store.finish_run(run_id, {"status": "completed"})


# --------------------------------------------------------------- comparison


def test_compare_uses_an_equal_token_budget_not_equal_steps(tmp_path):
    store, _ = make_store(tmp_path)
    # B packs twice the tokens per step, so at equal STEPS it has seen more data; at equal TOKENS it is slower to learn.
    add_run(store, "a", loss_fn=lambda i: 3.0 * math.exp(-i / 10) + 0.5, tokens=1000)
    add_run(store, "b", loss_fn=lambda i: 3.0 * math.exp(-i / 20) + 0.5, tokens=2000)
    result = analysis.compare_runs(store, ["a", "b"])
    assert result["budget_tokens"] == 40 * 1000  # the smaller total: the largest budget BOTH reached
    rows = {r["run_id"]: r for r in result["runs"]}
    assert rows["a"]["reached_budget"] and rows["b"]["reached_budget"]
    # at 40k tokens: a is at its last window (40), b only at window 20 -> b's loss is still higher
    assert rows["b"]["loss_at_budget"] > rows["a"]["loss_at_budget"]
    explicit = {r["run_id"]: r for r in analysis.compare_runs(store, ["a", "b"], token_budget=10_000)["runs"]}
    assert explicit["a"]["loss_at_budget"] < explicit["b"]["loss_at_budget"]  # a has had 10 windows of data, b only 5


def test_seed_noise_gates_the_verdict_and_config_diff_ignores_seeds(tmp_path):
    store, _ = make_store(tmp_path)
    for seed, bump in enumerate((0.00, 0.02, -0.02)):
        add_run(store, f"x{seed}", seed=seed, config={"lr": 1e-4, "seed": seed}, loss_fn=lambda i, b=bump: 3.0 * math.exp(-i / 15) + 0.5 + b)
    for seed, bump in enumerate((0.01, 0.03, -0.01)):
        add_run(store, f"y{seed}", seed=seed, config={"lr": 2e-4, "seed": seed}, loss_fn=lambda i, b=bump: 3.0 * math.exp(-i / 15) + 0.5 + b + 0.001)
    result = analysis.compare_runs(store, ["x0", "x1", "x2", "y0", "y1", "y2"])
    assert len(result["seed_groups"]) == 2 and all(g["trustworthy"] for g in result["seed_groups"])
    assert result["verdicts"][0]["verdict"] == "within noise"  # a 0.001 gap is far below the 0.02-level seed spread
    assert list(result["config_diff"]) == ["lr"]
    lone = analysis.compare_runs(store, ["x0", "y0"])
    assert lone["verdicts"][0]["verdict"] == "insufficient seeds"


# ---------------------------------------------------------- hardware analyses


def add_hw(store, bus, lane, start, seconds, *, busy=95, temp_fn=lambda t: 70.0, clock_fn=lambda t: 2400.0, power=300.0, mem_fn=lambda t: 80.0):
    store.add_hardware_samples([{
        "ts": start + t, "pci_bus": bus, "lane": lane, "gpu_busy_pct": busy(t) if callable(busy) else busy,
        "temp_junction_c": temp_fn(t), "temp_memory_c": mem_fn(t), "sclk_mhz": clock_fn(t), "power_w": power,
    } for t in range(seconds)])


def test_thermal_analysis_detects_throttling_and_throughput_degradation(tmp_path):
    store, clock = make_store(tmp_path)
    # first half healthy, then the card heats up, clocks fall, throughput falls with them
    add_run(store, "r", windows=40, step_ms_fn=lambda i: 500 if i <= 20 else 700, finish=False)
    add_hw(store, "0000:03:00.0", "r9700", BASE, 200, temp_fn=lambda t: 70 if t < 100 else 96, clock_fn=lambda t: 2400 if t < 100 else 1900,
           mem_fn=lambda t: 80 if t < 100 else 91)
    clock.now = BASE + 400
    result = analysis.thermal_analysis(store, "r", bucket_s=20)
    gpu = result["gpus"]["0000:03:00.0"]
    assert result["throttle_suspected"] and gpu["throttle_buckets"] and min(gpu["throttle_buckets"]) >= 100
    assert result["throughput_degradation_pct"] > 20
    healthy = [b for b in gpu["buckets"] if b["t"] < 100]
    assert not any(b["throttle_suspected"] for b in healthy)


def test_contention_compares_throughput_with_the_other_gpu_busy_vs_quiet(tmp_path):
    store, clock = make_store(tmp_path)
    # steps 1-20 run while the 9070 XT is idle; steps 21-40 while it evaluates -> 25% slower
    add_run(store, "r", windows=40, step_ms_fn=lambda i: 400 if i <= 20 else 500, finish=False)
    add_hw(store, "0000:03:00.0", "r9700", BASE, 200)
    add_hw(store, "0000:08:00.0", "9070xt", BASE, 200, busy=lambda t: 2 if t < 100 else 90)
    clock.now = BASE + 400
    result = analysis.contention_analysis(store, "r")["other_gpu"]
    assert result["busy_windows"] > 5 and result["quiet_windows"] > 5
    assert result["slowdown_pct"] == pytest.approx(20.0, abs=4)  # 1000/0.4 vs 1000/0.5 tokens/s


def test_energy_per_million_tokens(tmp_path):
    store, clock = make_store(tmp_path)
    add_run(store, "r", windows=10, tokens=1000, finish=False)
    add_hw(store, "0000:03:00.0", "r9700", BASE, 101, power=300.0)  # 100 s at 300 W = 30 kJ
    add_hw(store, "0000:08:00.0", "9070xt", BASE, 101, power=40.0)  # other lane: excluded from the run's energy
    clock.now = BASE + 200
    result = analysis.energy_analysis(store, "r")
    assert result["joules"] == pytest.approx(30_000, rel=0.01) and result["wh"] == pytest.approx(30_000 / 3600, rel=0.01)
    assert result["joules_per_million_input_tokens"] == pytest.approx(30_000 / 10_000 * 1e6, rel=0.01)
    assert analysis.energy_analysis(*make_empty(tmp_path))["joules"] is None  # no power samples -> null, not 0


def make_empty(tmp_path):
    store = TrainingStore(tmp_path / "e.sqlite3")
    add_run(store, "q", windows=5)
    return store, "q"


def test_mfu_is_labelled_estimate_and_null_without_inputs():
    run = {"config": {"trainable_params": 40e6, "total_params": 3e9}, "throughput": {"tokens_per_sec": 2000.0}}
    out = analysis.mfu_estimate(run, 100.0)
    assert out["estimate"] is True
    assert out["mfu_trainable_basis"] == pytest.approx(6 * 40e6 * 2000 / 100e12)
    assert out["mfu_frozen_base_basis"] == pytest.approx(4 * 3e9 * 2000 / 100e12)  # LoRA: no weight grads for the frozen base
    assert analysis.mfu_estimate(run, None)["mfu_trainable_basis"] is None


# ------------------------------------------------------------------- DDP


def test_straggler_and_scaling_analysis(tmp_path):
    store, _ = make_store(tmp_path)
    store.create_run({"id": "d", "name": "ddp", "world_size": 2})
    store.add_hardware_samples([{"ts": BASE, "pci_bus": "0000:03:00.0", "lane": "r9700", "vram_total_bytes": 32 * 1024**3},
                                {"ts": BASE, "pci_bus": "0000:08:00.0", "lane": "9070xt", "vram_total_bytes": 16 * 1024**3}])
    ranks = []
    for step in (10, 20, 30):
        ranks.append({"step": step, "rank": 0, "pci_bus": "0000:03:00.0", "window_steps": 10, "step_time_ms": 500, "comm_ms": 50, "mem_peak_bytes": 12 * 1024**3})
        ranks.append({"step": step, "rank": 1, "pci_bus": "0000:08:00.0", "window_steps": 10, "step_time_ms": 650, "comm_ms": 20, "mem_peak_bytes": 15 * 1024**3})
    store.record_metrics("d", {"ranks": ranks, "heartbeat": {"last_step": 30, "rank": 0}})
    result = analysis.straggler_analysis(store, "d")
    by_rank = {r["rank"]: r for r in result["ranks"]}
    assert by_rank[0]["mean_wait_ms"] == pytest.approx(150) and by_rank[1]["mean_wait_ms"] == pytest.approx(0)
    assert by_rank[1]["slowest_windows"] == 3 and by_rank[0]["comm_share"] == pytest.approx(0.1)
    assert by_rank[1]["vram_headroom_bytes"] == 1 * 1024**3  # the 16 GB card sets the limit

    add_run(store, "solo_a", tokens=1000, step_ms=500)  # 2000 tok/s
    add_run(store, "solo_b", tokens=1000, step_ms=500)  # 2000 tok/s
    add_run(store, "ddp_fast", tokens=2000, step_ms=500)  # 4000 tok/s  -> perfect scaling
    scaling = analysis.scaling_efficiency(store, "ddp_fast", ["solo_a", "solo_b"])
    assert scaling["scaling_efficiency"] == pytest.approx(1.0) and scaling["speedup_vs_fastest_single"] == pytest.approx(2.0)


# ------------------------------------------------------------------ resume


def test_resume_report_flags_discontinuities(tmp_path):
    store, _ = make_store(tmp_path)
    add_run(store, "parent", windows=30, finish=False)
    store.add_event("parent", {"kind": "x"})
    # faithful resume from step 200: continues the curve
    store.create_run({"id": "good", "name": "good", "parent_run_id": "parent"})
    store.add_event("good", {"kind": "resumed", "payload": {"from_step": 200, "manifest_valid": True, "resume_state_complete": True, "tokens_seen": 20_000}})
    store.record_metrics("good", {"steps": [{"step": 210, "window_steps": 10, "phase": "train", "loss": 3.0 * math.exp(-21 / 15) + 0.5,
                                             "nonpad_tokens": 1000, "step_time_ms": 500}]})
    assert analysis.resume_report(store, "good")["verdict"] == "ok"
    # adapter-only checkpoint: weights restored but optimizer/scheduler/RNG are not, and the loss jumps back up
    store.create_run({"id": "bad", "name": "bad", "parent_run_id": "parent"})
    store.add_event("bad", {"kind": "resumed", "payload": {"from_step": 200, "manifest_valid": True, "resume_state_complete": False,
                                                           "resume_state_missing": ["optimizer", "rng"], "tokens_seen": 20_000}})
    store.record_metrics("bad", {"steps": [{"step": 210, "window_steps": 10, "phase": "train", "loss": 3.4, "nonpad_tokens": 1000, "step_time_ms": 500}]})
    report = analysis.resume_report(store, "bad")
    failed = {c["name"] for c in report["checks"] if c["ok"] is False}
    assert report["verdict"] == "fail" and {"optimizer_scheduler_rng_restored", "loss_continuity"} <= failed
    assert analysis.resume_report(store, "parent")["verdict"] == "n/a"


# ------------------------------------------------------------------ alerts


def test_alerts_cover_nan_spike_throughput_idle_heat_faults_and_queue(tmp_path):
    store, clock = make_store(tmp_path, stall_floor_s=1e6)
    losses = [2.0 - 0.01 * i for i in range(30)] + [9.0]
    add_run(store, "r", windows=31, loss_fn=lambda i: losses[i - 1], step_ms_fn=lambda i: 500 if i < 28 else 2000, finish=False, nan_count=0)
    store.record_metrics("r", {"steps": [{"step": 9999, "window_steps": 1, "phase": "train", "nan_count": 2, "step_time_ms": 500}]})
    store.add_event("r", {"kind": "gpu_reset", "severity": "error", "step": 100})
    store.add_event("r", {"kind": "lease_lost", "severity": "warning"})
    # last minute: training GPU nearly idle and memory too hot
    add_hw(store, "0000:03:00.0", "r9700", BASE, 400, busy=5, mem_fn=lambda t: 93)
    clock.now = BASE + 400
    store.record_metrics("r", {"steps": [], "heartbeat": {"last_step": 9999}})
    kinds = {a["kind"] for a in analysis.run_alerts(store, "r", eval_queue={"depth": 14, "capacity": 16})}
    assert {"nan_inf", "loss_spike", "throughput_drop", "gpu_idle", "temperature", "gpu_fault", "lease_lost", "eval_queue_saturated"} <= kinds


def test_stalled_run_alerts(tmp_path):
    store, clock = make_store(tmp_path)
    add_run(store, "s", windows=5, finish=False)
    clock.now += 100
    store.record_metrics("s", {"steps": [], "heartbeat": {"last_step": 50}})
    clock.now += 2000
    store.record_metrics("s", {"steps": [], "heartbeat": {"last_step": 50}})
    assert any(a["kind"] == "stalled" for a in analysis.run_alerts(store, "s"))
