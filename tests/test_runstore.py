from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402


def make_run(
    store: "app.RunStore",
    run_id: str,
    model_name: str,
    task_label,
    tokens_per_second,
    latency_ms,
    client_id=None,
    project_id=None,
    workload_kind="evaluation",
    agent_role=None,
):
    store.begin_run({
        "id": run_id, "comparison_id": None, "deployment_id": None,
        "created_at": app.utc_now(), "mode": "single_gpu", "target": "primary",
        "model_name": model_name, "model_path": "x.gguf", "lane_keys": ["primary"],
        "device": "rocm:0", "context_window": 8192, "reasoning_budget": 0,
        "request": {}, "configuration": {}, "client_id": client_id, "task_label": task_label,
        "project_id": project_id, "workload_kind": workload_kind, "agent_role": agent_role,
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
    assert "project_id" in columns
    assert "workload_kind" in columns
    assert "host_os_json" in columns
    assert "clients" in tables
    assert "projects" in tables
    assert "agent_sessions" in tables
    assert "coding_tasks" in tables
    assert "tool_events" in tables


def test_run_records_host_os_by_default(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    make_run(store, "r1", "modelA", None, 60.0, 2000)
    host_os = store.get("r1")["host_os"]
    assert host_os["system"]
    assert host_os["release"]


def test_client_registration_round_trip(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    registered = store.register_client("MyRepo", "github.com/user/repo", "/tmp/out.jsonl")
    fetched = store.get_client(registered["client_id"])
    assert fetched["project_name"] == "MyRepo"
    assert fetched["project_id"] == registered["project_id"]
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


def test_infer_backend_prefers_probed_source_then_falls_back_to_date():
    assert app.infer_backend({"configuration": {"runtime": {"source": "llama.cpp"}}}) == "llama.cpp"
    assert app.infer_backend({"configuration": {"runtime": {"source": "lmstudio"}}}) == "LM Studio"
    # No probed source: created_at decides against the migration cutoff (2026-10-17T06:06:35Z).
    assert app.infer_backend({"created_at": "2026-09-01T10:00:00.000+00:00"}) == "LM Studio"
    assert app.infer_backend({"created_at": "2026-10-18T10:00:00.000+00:00"}) == "llama.cpp"
    # A malformed/absent timestamp is treated as an old (LM Studio) run, never misread as llama.cpp.
    assert app.infer_backend({}) == "LM Studio"
    assert app.infer_backend({"created_at": "not-a-date"}) == "LM Studio"


def test_infer_os_defaults_to_windows_for_backdated_runs():
    assert app.infer_os({"host_os": {"system": "Linux", "release": "6.8.0"}}) == "Linux"
    assert app.infer_os({"host_os": {"system": "Windows"}}) == "Windows"
    assert app.infer_os({}) == "Windows"  # back-dated run with no captured host_os
    assert app.infer_os({"host_os": {}}) == "Windows"


def test_runs_carry_backend_and_os_labels(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    store.begin_run({
        "id": "native", "comparison_id": None, "deployment_id": None,
        "created_at": app.utc_now(), "mode": "single_gpu", "target": "primary",
        "model_name": "modelA", "model_path": "x.gguf", "lane_keys": ["primary"],
        "device": "rocm:0", "context_window": 8192, "reasoning_budget": 0,
        "request": {}, "configuration": {"runtime": {"source": "llama.cpp", "build": 11699}},
        "host_os": {"system": "Windows", "release": "11"},
        "client_id": None, "task_label": None, "project_id": None,
        "workload_kind": "evaluation", "agent_role": None,
    })
    store.finish_run("native", {"status": "completed", "usage": {}, "timing": {}, "resources": {}})

    detail = store.get("native")
    assert detail["backend"] == "llama.cpp" and detail["os"] == "Windows"
    assert store.recent(10)[0]["backend"] == "llama.cpp"
    # The lean dashboard query also surfaces the probed source without loading full config.
    assert store._dashboard_runs()[0]["backend"] == "llama.cpp"


def test_dashboard_groups_by_backend_and_os(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    # Two llama.cpp runs and one legacy run with no runtime source (-> LM Studio by date).
    for run_id in ("a", "b"):
        store.begin_run({
            "id": run_id, "comparison_id": None, "deployment_id": None,
            "created_at": app.utc_now(), "mode": "single_gpu", "target": "primary",
            "model_name": "modelA", "model_path": "x.gguf", "lane_keys": ["primary"],
            "device": "rocm:0", "context_window": 8192, "reasoning_budget": 0,
            "request": {}, "configuration": {"runtime": {"source": "llama.cpp"}},
            "host_os": {"system": "Windows"}, "client_id": None, "task_label": None,
            "project_id": None, "workload_kind": "evaluation", "agent_role": None,
        })
        store.finish_run(run_id, {"status": "completed", "usage": {}, "timing": {}, "resources": {}})
    store.begin_run({
        "id": "legacy", "comparison_id": None, "deployment_id": None,
        "created_at": "2026-09-01T10:00:00.000+00:00", "mode": "single_gpu", "target": "primary",
        "model_name": "modelA", "model_path": "x.gguf", "lane_keys": ["primary"],
        "device": "rocm:0", "context_window": 8192, "reasoning_budget": 0,
        "request": {}, "configuration": {}, "host_os": {"system": "Windows"},
        "client_id": None, "task_label": None, "project_id": None,
        "workload_kind": "evaluation", "agent_role": None,
    })
    store.finish_run("legacy", {"status": "completed", "usage": {}, "timing": {}, "resources": {}})

    dashboard = store.dashboard()
    backends = {row["backend"]: row["runs"] for row in dashboard["backends"]}
    assert backends == {"llama.cpp": 2, "LM Studio": 1}
    operating_systems = {row["os"]: row["runs"] for row in dashboard["operating_systems"]}
    assert operating_systems == {"Windows": 3}


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


def test_dashboard_uses_complete_history_beyond_recent_cap(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    for index in range(505):
        make_run(store, f"r{index}", "modelA", None, 50.0, 1000)

    assert len(store.recent(1000)) == 500
    dashboard = store.dashboard()
    assert dashboard["summary"]["total_runs"] == 505
    assert dashboard["summary"]["total_tokens"] == 505 * 150


def test_dashboard_reads_metrics_without_loading_content_or_taking_writer_lock(tmp_path):
    store = app.RunStore(tmp_path / "runs.db")
    make_run(store, "a", "qwen", "code", 25, 1000, workload_kind="coding_agent")
    with store.connect() as connection:
        connection.execute("UPDATE runs SET request_json=?, configuration_json=? WHERE id='a'", (
            '{"messages":[{"content":"large private prompt"}]}',
            '{"reasoning":{"effective_effort":"low","budget_policy":"model_controlled"}}',
        ))
    class NoWriterLock:
        def __enter__(self):
            raise AssertionError("Dashboard reader blocked inference writer")
        def __exit__(self, *args):
            pass
    original = store.lock
    store.lock = NoWriterLock()
    rows = store._dashboard_runs()
    store.lock = original
    assert rows[0]["request"] is None
    assert rows[0]["configuration"]["reasoning"]["effective_effort"] == "low"
    assert store.get("a")["request"]["messages"][0]["content"] == "large private prompt"


def test_decode_aggregate_excludes_runs_with_missing_decode_time(tmp_path):
    store = app.RunStore(tmp_path / "runs.db")
    for run_id in ("timed", "untimed"):
        make_run(store, run_id, "qwen", "code", 25, 1000, workload_kind="coding_agent")
    store.finish_run("timed", {"status": "completed", "usage": {"output_tokens": 50, "thinking_tokens": 30},
                               "timing": {"decode_duration_ms": 2000}})
    dashboard = store.dashboard()
    assert dashboard["summary"]["aggregate_tokens_per_second"] == 25
    level = dashboard["reasoning_levels"][0]
    assert level["thinking_tokens_reported_runs"] == 1
    assert level["avg_thinking_tokens"] == 30


def test_summary_separates_estimated_thinking_from_unreported(tmp_path):
    store = app.RunStore(tmp_path / "runs.db")
    for run_id in ("plain", "reasoned"):
        make_run(store, run_id, "qwen", "code", 25, 1000)
    store.finish_run("reasoned", {"status": "completed", "usage": {
        "output_tokens": 80, "thinking_tokens": None, "thinking_tokens_estimated": 30}})
    summary = store.dashboard()["summary"]
    assert summary["thinking_tokens"] == 0
    assert summary["thinking_tokens_reported_runs"] == 0
    assert summary["thinking_tokens_estimated"] == 30
    assert summary["thinking_tokens_estimated_runs"] == 1


def test_project_filtered_dashboard_and_gpu_breakdown(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    project_a = store.upsert_project("Project A", "F:/Projects/A")
    project_b = store.upsert_project("Project B", "F:/Projects/B")
    make_run(store, "a", "modelA", "edit", 60.0, 1000, project_id=project_a["id"], workload_kind="coding_agent", agent_role="developer")
    make_run(store, "b", "modelB", "test", 30.0, 2000, project_id=project_b["id"], workload_kind="coding_agent", agent_role="qa")

    dashboard = store.dashboard(project_id=project_a["id"])
    assert dashboard["summary"]["total_runs"] == 1
    assert dashboard["projects"][0]["project"] == "Project A"
    assert dashboard["agents"][0]["agent_role"] == "developer"
    assert dashboard["gpus"][0]["gpu"] == "primary"


def test_tool_events_are_aggregated_per_project(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    project = store.upsert_project("Project")
    store.record_tool_event({"project_id": project["id"], "tool_type": "test", "status": "completed", "exit_code": 0})
    store.record_tool_event({"project_id": project["id"], "tool_type": "build", "status": "failed", "exit_code": 1})

    tools = store.dashboard(project_id=project["id"])["tool_summary"]
    assert tools == {"events": 2, "failed": 1, "success_rate": 50.0}


def _run_on_day(store, run_id, day):
    store.begin_run({
        "id": run_id, "comparison_id": None, "deployment_id": None,
        "created_at": f"{day}T12:00:00+00:00", "mode": "single_gpu", "target": "primary",
        "model_name": "m", "model_path": "x.gguf", "lane_keys": ["primary"], "device": "rocm:0",
        "context_window": 8192, "reasoning_budget": 0, "request": {}, "configuration": {},
        "client_id": None, "task_label": None, "project_id": None, "workload_kind": "evaluation",
        "agent_role": None,
    })
    store.finish_run(run_id, {"status": "completed", "usage": {"total_tokens": 100},
                              "timing": {}, "resources": {}})


def test_dashboard_date_range_filters_runs_and_daily(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    for run_id, day in [("r1", "2026-10-01"), ("r2", "2026-10-05"), ("r3", "2026-10-09")]:
        _run_on_day(store, run_id, day)

    scoped = store.dashboard(start_date="2026-10-05", end_date="2026-10-09")  # inclusive both ends
    assert scoped["summary"]["total_runs"] == 2
    assert {bucket["date"] for bucket in scoped["daily"]} == {"2026-10-05", "2026-10-09"}
    assert scoped["filters"] == {"start_date": "2026-10-05", "end_date": "2026-10-09"}
    # End date includes its whole day; start/end may be given alone; no filter sees everything.
    assert store.dashboard(end_date="2026-10-01")["summary"]["total_runs"] == 1
    assert store.dashboard(start_date="2026-10-05")["summary"]["total_runs"] == 2
    assert store.dashboard()["summary"]["total_runs"] == 3


def test_day_after_bound_and_validation():
    assert app.RunStore._day_after("2026-10-09") == "2026-10-10"
    assert app.RunStore._day_after("2026-12-31") == "2027-01-01"
    with pytest.raises(ValueError):
        app.RunStore._day_after("09-10-2026")


def test_usage_ledger_disabled_is_a_noop(tmp_path: Path):
    ledger = app.UsageLedger(tmp_path / "u.sqlite3", enabled=False)
    ledger.record("p1", "autocomplete", 10, 5)
    ledger.flush()
    assert ledger.query() == []
    ledger.close()


def test_usage_ledger_rolls_up_per_project_and_connection_type(tmp_path: Path):
    ledger = app.UsageLedger(tmp_path / "u.sqlite3", enabled=True, flush_interval=100)
    ledger.record("p1", "coding_agent", 100, 20)
    ledger.record("p1", "coding_agent", 50, 10)   # same bucket -> accumulates
    ledger.record("p1", "autocomplete", 30, 5)
    ledger.record(None, "inference", 1, 1)         # unattributed -> ""
    rows = ledger.query()                           # query() flushes pending first

    coding = next(r for r in rows if r["project_id"] == "p1" and r["connection_type"] == "coding_agent")
    assert (coding["requests"], coding["input_tokens"], coding["output_tokens"]) == (2, 150, 30)
    assert next(r for r in rows if r["connection_type"] == "autocomplete")["requests"] == 1
    assert next(r for r in rows if r["connection_type"] == "inference")["project_id"] == ""

    assert all(r["connection_type"] == "autocomplete" for r in ledger.query(connection_type="autocomplete"))
    assert all(r["project_id"] == "p1" for r in ledger.query(project_id="p1"))
    ledger.close()


def test_chat_history_disabled_is_a_noop_with_no_file(tmp_path: Path):
    history = app.ChatHistoryStore(tmp_path / "hist.sqlite3", enabled=False)
    history.enqueue({"id": "x", "created_at": "2026-10-09T00:00:00+00:00"})
    history.flush()
    assert history.query() == []
    assert not (tmp_path / "hist.sqlite3").exists()  # disabled -> nothing created
    history.close()


def test_chat_history_archives_content_and_scalar_tokens(tmp_path: Path):
    history = app.ChatHistoryStore(tmp_path / "hist.sqlite3", enabled=True)
    history.enqueue({
        "id": "r1", "created_at": "2026-10-09T10:00:00+00:00", "project_id": "p1",
        "model_name": "m", "target": "r9700", "workload_kind": "coding_agent",
        "usage": {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20,
                  "thinking_tokens": 5, "total_tokens": 120},
        "timing": {"tokens_per_second": 25.0, "end_to_end_duration_ms": 900.0},
        "finish_reason": "stop", "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function"}], "output_text": "hello", "reasoning_text": "think",
        "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}],
    })
    history.flush()

    light = history.query()
    assert len(light) == 1
    assert light[0]["input_tokens"] == 100 and light[0]["output_tokens"] == 20
    assert light[0]["total_tokens"] == 120 and light[0]["latency_ms"] == 900.0
    assert "messages_json" not in light[0] and "output_text" not in light[0]  # light view = metadata only

    full = history.query(include_content=True)[0]
    assert full["messages"] == [{"role": "user", "content": "hi"}]
    assert full["output_text"] == "hello" and full["reasoning_text"] == "think"
    assert full["tool_calls"][0]["function"]["name"] == "read"

    assert history.query(project_id="p1") and history.query(project_id="nope") == []
    assert history.query(start_date="2026-10-09", end_date="2026-10-09")  # end date inclusive
    assert history.query(end_date="2026-10-08") == []
    history.close()


def test_serving_reports_concurrency_including_the_new_request():
    handle = app.ServerHandle("primary", None, [], None)
    with handle.serving() as alone:
        assert alone == 1
        with handle.serving() as second:
            assert second == 2  # the first request is still in flight
    with handle.serving() as again:
        assert again == 1  # counts reset once requests finish


def test_wall_clock_throughput_counts_overlapping_decode_once(tmp_path: Path):
    store = app.RunStore(tmp_path / "t.sqlite3")
    base = app.datetime(2026, 10, 18, 12, 0, 0, tzinfo=app.UTC)
    # Two requests share one server: A decodes 0-10 s, B decodes 5-15 s, 100 tokens each.
    for run_id, start, end in [("a", 0, 10), ("b", 5, 15)]:
        store.begin_run({
            "id": run_id, "comparison_id": None, "deployment_id": None,
            "created_at": base.isoformat(timespec="milliseconds"), "mode": "single_gpu", "target": "9070xt",
            "model_name": "modelA", "model_path": "x.gguf", "lane_keys": ["9070xt"], "device": "rocm:0",
            "context_window": 8192, "reasoning_budget": 0, "request": {}, "configuration": {},
            "client_id": None, "task_label": None, "project_id": None, "workload_kind": "evaluation",
            "agent_role": None,
        })
        store.finish_run(run_id, {
            "status": "completed", "usage": {"output_tokens": 100, "total_tokens": 100},
            "timing": {"decode_duration_ms": (end - start) * 1000, "tokens_per_second": 10.0,
                       "concurrent_requests_at_start": 2},
            "resources": {},
        })
        with store.connect() as connection:
            connection.execute("UPDATE runs SET finished_at = ? WHERE id = ?",
                               ((base + app.timedelta(seconds=end)).isoformat(timespec="milliseconds"), run_id))

    dashboard = store.dashboard()
    summary = dashboard["summary"]
    assert summary["wall_clock_tokens_per_second"] == round(200 / 15, 2)  # 15 s of union, not 20 s summed
    assert summary["aggregate_tokens_per_second"] == 10.0  # the old sum-based figure undercounts
    assert summary["avg_concurrent_requests"] == 2
    assert dashboard["gpus"][0]["wall_clock_tokens_per_second"] == round(200 / 15, 2)
