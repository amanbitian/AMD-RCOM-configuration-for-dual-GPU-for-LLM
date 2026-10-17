from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_gpu_setup.training.privacy import sanitize  # noqa: E402
from dual_gpu_setup.training.store import TrainingError, TrainingStore  # noqa: E402


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def window(step, **kw):
    row = {"step": step, "window_steps": 10, "phase": "train", "loss": 2.0, "processed_tokens": 1000,
           "nonpad_tokens": 800, "supervised_tokens": 400, "step_time_ms": 500.0, "mem_peak_bytes": 5 * 1024**3}
    row.update(kw)
    return row


# ------------------------------------------------------------------- privacy


def test_sanitize_drops_content_keys_truncates_and_redacts():
    clean = sanitize({
        "resume": "John Doe, 12 Main St", "loss": 0.5, "note": "mail jane.doe@example.com or +1 (555) 123-4567 today",
        "nested": {"text": "secret", "ok": [1, 2]}, "long": "x" * 900, "date": "run 2026-10-11 17:28:26", "step": "v1.2.3",
    })
    assert clean["resume"] == "[redacted]" and clean["nested"]["text"] == "[redacted]" and clean["nested"]["ok"] == [1, 2]
    assert "jane.doe" not in clean["note"] and "[email]" in clean["note"] and "[phone]" in clean["note"]
    assert clean["long"].endswith("[truncated]") and len(clean["long"]) < 600
    assert clean["date"] == "run 2026-10-11 17:28:26" and clean["step"] == "v1.2.3"  # dates/versions are not phone numbers
    assert clean["loss"] == 0.5


def test_events_and_evals_never_store_raw_text(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    store.create_run({"id": "r", "name": "x"})
    store.add_event("r", {"kind": "bad_example", "payload": {"text": "full resume of a candidate", "example_id": "e1"}})
    store.record_eval("r", {"step": 1, "level": "native", "metrics": {"f1": 0.9, "examples": ["private resume"]}})
    event = next(e for e in store.events("r") if e["kind"] == "bad_example")
    assert event["payload"] == {"text": "[redacted]", "example_id": "e1"}
    metrics = store.evals("r")[1]["native"]["default"]["metrics"]
    assert metrics == {"f1": 0.9, "examples": "[redacted]"}


# ------------------------------------------------------------ incremental stats


def test_aggregates_are_incremental_and_replay_safe(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    store.create_run({"id": "r", "name": "x"})
    for _ in range(3):  # replays of the same windows must not double count
        store.record_metrics("r", {"steps": [window(10), window(20, nonpad_tokens=None)]})
    run = store.get_run("r")
    assert run["windows"] == 2 and run["tokens"]["nonpad"] == 800 and run["tokens"]["processed"] == 2000
    # padding ratio only over windows where both counters exist
    assert run["tokens"]["padding_ratio"] == pytest.approx(0.2)
    store.recompute_aggregates("r")
    assert store.get_run("r")["tokens"] == run["tokens"]


def test_old_database_is_upgraded_and_backfilled(tmp_path):
    path = tmp_path / "old.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE training_runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, parent_run_id TEXT,
            created_at REAL NOT NULL, started_at REAL NOT NULL, finished_at REAL, lanes_json TEXT, config_json TEXT,
            env_json TEXT, dataset_json TEXT, git_commit TEXT, seed INTEGER, max_steps INTEGER,
            last_step INTEGER NOT NULL DEFAULT 0, last_step_at REAL, last_heartbeat_at REAL, step_time_ema_ms REAL,
            error TEXT, summary_json TEXT);
        CREATE TABLE training_steps (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, step INTEGER NOT NULL,
            ts REAL NOT NULL, phase TEXT, window_steps INTEGER NOT NULL DEFAULT 1, loss REAL, lr REAL, grad_norm REAL,
            processed_tokens INTEGER, nonpad_tokens INTEGER, supervised_tokens INTEGER, step_time_ms REAL,
            excluded_ms REAL, mem_alloc_bytes INTEGER, mem_reserved_bytes INTEGER, mem_peak_bytes INTEGER,
            alloc_retries INTEGER, nan_count INTEGER, extra_json TEXT, UNIQUE(run_id, step));
        INSERT INTO training_runs (id, name, status, created_at, started_at) VALUES ('old', 'old run', 'completed', 1, 1);
        INSERT INTO training_steps (run_id, step, ts, phase, window_steps, nonpad_tokens, processed_tokens, step_time_ms)
            VALUES ('old', 10, 2, 'train', 10, 900, 1000, 100.0);
    """)
    connection.commit()
    connection.close()
    run = TrainingStore(path).get_run("old")
    assert run["tokens"]["nonpad"] == 900 and run["world_size"] == 1 and run["windows"] == 1
    assert run["host_os"] is None


# ----------------------------------------------------------------- ranks/stall


def test_rank_heartbeats_and_per_rank_rows(tmp_path):
    clock = Clock()
    store = TrainingStore(tmp_path / "t.sqlite3", clock=clock)
    store.create_run({"id": "r", "name": "ddp", "world_size": 2})
    for rank in (0, 1):
        store.record_metrics("r", {"steps": [window(10)] if rank == 0 else [], "heartbeat": {"last_step": 10, "rank": rank, "pci_bus": f"0000:0{rank}:00.0"},
                                   "ranks": [{"step": 10, "rank": rank, "window_steps": 10, "step_time_ms": 500 + 50 * rank,
                                              "comm_ms": 40, "nonpad_tokens": 400, "mem_peak_bytes": 1000 * (rank + 1)}]})
    assert [r["rank"] for r in store.rank_steps("r")] == [0, 1]
    assert store.get_run("r")["status"] == "running" and len(store.get_run("r")["ranks"]) == 2
    # Rank 1 dies; rank 0 keeps heart-beating (it is blocked in a collective but its thread lives on).
    for _ in range(8):
        clock.now += 20
        store.record_metrics("r", {"steps": [], "heartbeat": {"last_step": 10, "rank": 0}})
    run = store.get_run("r")
    assert run["status"] == "stalled" and "rank 1" in run["stall_reason"]


def test_missing_rank_is_reported_after_grace(tmp_path):
    clock = Clock()
    store = TrainingStore(tmp_path / "t.sqlite3", clock=clock)
    store.create_run({"id": "r", "name": "ddp", "world_size": 2})
    for _ in range(10):
        clock.now += 20
        store.record_metrics("r", {"steps": [], "heartbeat": {"last_step": 0, "rank": 0}})
    assert "never reported" in store.get_run("r")["stall_reason"]


# ------------------------------------------------------- evals, artifacts, etc


def test_eval_validation_and_flags(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    store.create_run({"id": "r", "name": "x"})
    store.add_checkpoint("r", {"step": 100, "path": "/c/100", "valid": True, "manifest": {"files": [1], "tokens_seen": 5}})
    for bad in ({"step": 1, "level": "bogus"}, {"step": 1, "level": "native"}, {"level": "native", "metrics": {}}):
        with pytest.raises(TrainingError):
            store.record_eval("r", bad)
    store.record_eval("r", {"step": 100, "level": "gguf_quant", "status": "failed", "error": "conversion failed"})
    ckpt = store.checkpoints("r")[0]
    assert ckpt["evals"]["gguf_quant"]["default"]["status"] == "failed"
    store.set_checkpoint_flags("r", 100, best=True, designated=True)
    assert store.checkpoints("r")[0]["best"] and store.checkpoints("r")[0]["designated"]
    with pytest.raises(TrainingError):
        store.set_checkpoint_flags("r", 999, best=True)


def test_notes_example_stats_artifacts_and_profiles(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    store.create_run({"id": "r", "name": "x"})
    with pytest.raises(TrainingError):
        store.add_note("r", {})
    store.add_note("r", {"author": "me", "hypothesis": "higher rank helps", "change": "r=32", "outcome": "mail me@x.com"})
    assert "[email]" in store.notes("r")[0]["outcome"]
    store.add_example_stats("r", 100, [{"example_id": "a", "loss": 0.1, "correct": True}, {"example_id": "b", "loss": 2.5, "correct": False}, {"loss": 1}])
    assert [e["example_id"] for e in store.hard_examples("r")] == ["b", "a"]
    art = store.add_artifact("r", "rocprof_trace", "/x/trace.pftrace", size_bytes=10, downloadable=False)
    assert store.get_artifact("r", art["id"])["kind"] == "rocprof_trace"
    store.record_profile("h", {"model": "m"}, gpu_pci_bus="0000:03:00.0", peak_mem_bytes=100, tokens_per_sec=5.0, run_id="r")
    store.record_profile("h", {"model": "m"}, gpu_pci_bus="0000:03:00.0", peak_mem_bytes=50, tokens_per_sec=6.0, run_id="r")
    profile = store.get_profile("h")
    assert profile["peak_mem_bytes"] == 100 and profile["tokens_per_sec"] == 6.0  # peaks only ratchet up


def test_series_downsamples_in_sql_and_keeps_last_window(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    store.create_run({"id": "r", "name": "x"})
    store.record_metrics("r", {"steps": [window(i * 10) for i in range(1, 501)]})
    series = store.series("r", max_points=50)
    assert series["total_windows"] == 500 and series["stride"] == 10 and len(series["points"]) <= 51
    assert series["points"][-1]["step"] == 5000 and series["points"][0]["step"] == 10


def test_listing_many_runs_is_cheap(tmp_path):
    import time

    store = TrainingStore(tmp_path / "t.sqlite3")
    for i in range(60):
        store.create_run({"id": f"r{i}", "name": f"run {i}"})
        store.record_metrics(f"r{i}", {"steps": [window(s * 10) for s in range(1, 40)]})
    started = time.perf_counter()
    assert store.list_runs(100)["total"] == 60
    assert time.perf_counter() - started < 1.0


def test_a_failed_attempt_never_overwrites_a_completed_evaluation(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    store.create_run({"id": "r", "name": "x"})
    store.add_checkpoint("r", {"step": 5, "path": "/c", "valid": True, "manifest": {"files": [1]}})
    store.record_eval("r", {"step": 5, "level": "native", "metrics": {"f1": 0.9}})
    assert store.record_eval("r", {"step": 5, "level": "native", "status": "failed", "error": "worker crashed"}) == {"ok": True, "kept_completed": True}
    kept = store.evals("r")[5]["native"]["default"]
    assert kept["status"] == "completed" and kept["metrics"] == {"f1": 0.9}
    store.record_eval("r", {"step": 5, "level": "native", "metrics": {"f1": 0.95}})  # a newer success still replaces it
    assert store.evals("r")[5]["native"]["default"]["metrics"] == {"f1": 0.95}
