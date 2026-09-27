from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402


def make_run(store: "app.RunStore", run_id: str, model_name: str, task_label, tokens_per_second, latency_ms, client_id=None):
    store.begin_run({
        "id": run_id, "comparison_id": None, "deployment_id": None,
        "created_at": app.utc_now(), "mode": "single_gpu", "target": "primary",
        "model_name": model_name, "model_path": "x.gguf", "lane_keys": ["primary"],
        "device": "rocm:0", "context_window": 8192, "reasoning_budget": 0,
        "request": {}, "configuration": {}, "client_id": client_id, "task_label": task_label,
    })
    store.finish_run(run_id, {
        "status": "completed",
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        "timing": {"tokens_per_second": tokens_per_second, "end_to_end_duration_ms": latency_ms},
        "resources": {},
    })


def test_migrations_create_expected_schema(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    with store.connect() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "client_id" in columns
    assert "task_label" in columns
    assert "clients" in tables


def test_client_registration_round_trip(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    registered = store.register_client("MyRepo", "github.com/user/repo", "/tmp/out.jsonl")
    fetched = store.get_client(registered["client_id"])
    assert fetched["project_name"] == "MyRepo"
    assert fetched["output_path"] == "/tmp/out.jsonl"
    assert store.get_client("does-not-exist") is None


def test_dashboard_categories_group_by_task_label(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    make_run(store, "r1", "modelA", "relevance", 60.0, 2000)
    make_run(store, "r2", "modelB", "relevance", 40.0, 3000)
    make_run(store, "r3", "modelA", "resume_extraction", 30.0, 20000)
    make_run(store, "r4", "modelA", None, 55.0, 1500)  # uncategorized

    dashboard = store.dashboard()

    categories = {c["category"]: c for c in dashboard["categories"]}
    assert set(categories) == {"relevance", "resume_extraction"}
    assert categories["relevance"]["runs"] == 2
    assert categories["relevance"]["avg_tokens_per_second"] == 50.0
    assert categories["resume_extraction"]["runs"] == 1

    pairs = {(c["category"], c["model"]): c for c in dashboard["category_models"]}
    assert pairs[("relevance", "modelA")]["avg_tokens_per_second"] == 60.0
    assert pairs[("relevance", "modelB")]["avg_tokens_per_second"] == 40.0
    assert ("resume_extraction", "modelA") in pairs

    # uncategorized run still counts in the model-level breakdown (3 runs for modelA:
    # relevance + resume_extraction + the uncategorized one), just not in categories.
    models = {m["model"]: m for m in dashboard["models"]}
    assert models["modelA"]["runs"] == 3


def test_dashboard_categories_empty_when_no_task_labels(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    make_run(store, "r1", "modelA", None, 60.0, 2000)
    dashboard = store.dashboard()
    assert dashboard["categories"] == []
    assert dashboard["category_models"] == []
    assert dashboard["models"][0]["runs"] == 1


def test_prune_older_than_removes_old_runs_and_cascades_samples(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    old_at = (app.datetime.now(app.UTC) - app.timedelta(days=10)).isoformat(timespec="milliseconds")
    store.begin_run({
        "id": "old", "comparison_id": None, "deployment_id": None,
        "created_at": old_at, "mode": "single_gpu", "target": "primary",
        "model_name": "modelA", "model_path": "x.gguf", "lane_keys": ["primary"],
        "device": "rocm:0", "context_window": 8192, "reasoning_budget": 0,
        "request": {}, "configuration": {}, "client_id": None, "task_label": None,
    })
    store.finish_run("old", {"status": "completed", "usage": {}, "timing": {}, "resources": {}})
    store.save_samples("old", [{"sampled_at": app.utc_now(), "monotonic": 0.0}], 0.0)
    make_run(store, "recent", "modelA", None, 60.0, 2000)

    result = store.prune_older_than(days=1)
    assert result["deleted_runs"] == 1

    remaining_ids = {run["id"] for run in store.recent(100)}
    assert remaining_ids == {"recent"}
    with store.connect() as connection:
        sample_count = connection.execute("SELECT COUNT(*) FROM metric_samples WHERE run_id = 'old'").fetchone()[0]
    assert sample_count == 0  # cascaded via the FK ON DELETE CASCADE


def test_prune_older_than_rejects_non_positive_days(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    with pytest.raises(ValueError):
        store.prune_older_than(days=0)


def test_runs_for_client_only_returns_that_clients_runs(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    client = store.register_client("Repo", None, "/tmp/out.jsonl")
    make_run(store, "r1", "modelA", None, 60.0, 2000, client_id=client["client_id"])
    make_run(store, "r2", "modelA", None, 60.0, 2000, client_id=None)
    runs = store.runs_for_client(client["client_id"])
    assert len(runs) == 1
    assert runs[0]["id"] == "r1"
