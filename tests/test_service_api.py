from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_TOML = """
[evaluation]
enabled = true
lane = "9070xt"
designate_every = 2
queue_capacity = 4
primary_metric = "task_score"
[evaluation.peak_tflops]
r9700 = 96.0
"""


def window(step, loss, tokens=1000, step_ms=500):
    return {"step": step, "window_steps": 10, "phase": "train", "loss": loss, "processed_tokens": 1200,
            "nonpad_tokens": tokens, "supervised_tokens": 400, "step_time_ms": step_ms, "mem_peak_bytes": 4 * 1024**3}


def make_run(live, run_id, *, windows=20, config=None, lanes=None, step_ms=500):
    status, body = live.post("/api/training/runs", {"id": run_id, "name": run_id, "config": config or {"lr": 1e-4}, "lanes": lanes or []})
    assert status == 200, body
    rows = [window(i * 10, 3.0 * math.exp(-i / 10) + 0.5, step_ms=step_ms) for i in range(1, windows + 1)]
    assert live.post(f"/api/training/runs/{run_id}/metrics", {"steps": rows, "heartbeat": {"last_step": windows * 10}})[0] == 200
    return body


def add_checkpoint(live, run_id, step, tokens=None):
    return live.post(f"/api/training/runs/{run_id}/checkpoints", {
        "step": step, "path": f"/ckpt/{run_id}/checkpoint-{step}", "valid": True, "size_bytes": 10,
        "manifest": {"files": [{"path": "a"}], "tokens_seen": tokens if tokens is not None else step * 100}})


# ------------------------------------------------------ evaluation workflow


def test_checkpoints_are_queued_designated_and_coalesced_then_final_is_protected(live_app):
    live = live_app(EVAL_TOML)
    make_run(live, "r")
    for step in (100, 200, 300):
        add_checkpoint(live, "r", step)
    queue = live.get("/api/training/eval-queue")[1]
    jobs = {j["step"]: j for j in queue["jobs"]}
    assert set(jobs) == {200, 300} and jobs[200]["protected"] == 1  # every 2nd checkpoint designated; stale routine one coalesced
    assert [c["designated"] for c in live.get("/api/training/runs/r")[1]["checkpoints"] if c["step"] == 200] == [True]
    live.post("/api/training/runs/r/finish", {"status": "completed"})
    jobs = {j["step"]: j for j in live.get("/api/training/eval-queue")[1]["jobs"]}
    assert jobs[300]["flags"]["final"] is True and jobs[300]["priority"] == 100
    assert queue["settings"]["lane"] == "9070xt" and queue["stats"]["capacity"] == 4


def test_no_evaluation_jobs_when_evaluation_is_disabled(live_app):
    live = live_app()
    make_run(live, "r")
    add_checkpoint(live, "r", 100)
    assert live.get("/api/training/eval-queue")[1]["jobs"] == []


def test_recorded_evals_drive_best_checkpoint_selection_and_regressions(live_app):
    live = live_app(EVAL_TOML)
    make_run(live, "r", windows=30)
    scores = {100: (0.70, 0.70, 0.69), 200: (0.80, 0.79, 0.77), 300: (0.84, 0.83, 0.60)}
    for step, (native, f16, quant) in scores.items():
        add_checkpoint(live, "r", step)
        for level, score in (("native", native), ("gguf_f16", f16), ("gguf_quant", quant)):
            assert live.post("/api/training/runs/r/evals", {"step": step, "level": level, "metrics": {"task_score": score},
                                                            "latency": {"eval_wall_ms": 800}})[0] == 200
    assert live.post("/api/training/runs/r/evals", {"step": 100, "level": "bogus", "metrics": {}})[0] == 400
    body = live.get("/api/training/runs/r/evals")[1]
    assert body["selection"]["selected_step"] == 200 and body["selection"]["differs_from_loss_only_choice"]
    assert any(f["kind"] == "quality_regression" and f["step"] == 300 for f in body["regressions"])
    assert live.post("/api/training/runs/r/select", {"apply": True})[1]["applied"] is True
    flags = {c["step"]: c["best"] for c in live.get("/api/training/runs/r")[1]["checkpoints"]}
    assert flags == {100: False, 200: True, 300: False}
    assert live.post("/api/training/runs/r/checkpoints/300/promote")[0] == 200  # manual override is still possible
    assert live.post("/api/training/runs/r/checkpoints/999/promote")[0] == 404


def test_eval_queue_submission_validation(live_app):
    live = live_app(EVAL_TOML)
    make_run(live, "r")
    assert live.post("/api/training/eval-queue", {"run_id": "r"})[0] == 400
    assert live.post("/api/training/eval-queue", {"run_id": "nope", "step": 1, "path": "/x"})[0] == 404
    assert live.post("/api/training/eval-queue", {"run_id": "r", "step": 5, "path": "/x", "final": True})[1]["accepted"] is True


# ------------------------------------------------------------ compare/analysis


def test_compare_endpoint_and_validation(live_app):
    live = live_app()
    make_run(live, "a")
    make_run(live, "b", config={"lr": 2e-4})
    status, body = live.get("/api/training/compare?ids=a,b")
    assert status == 200 and body["budget_tokens"] == 20_000 and body["config_diff"] == {"lr": {"a": 0.0001, "b": 0.0002}}
    assert live.get("/api/training/compare?ids=a")[0] == 400
    assert live.get("/api/training/scaling?ddp=a")[0] == 400
    scaling = live.get("/api/training/scaling?ddp=a&baselines=b")[1]
    assert scaling["scaling_efficiency"] == pytest.approx(1.0)


def test_analysis_alerts_and_run_detail_endpoints(live_app):
    live = live_app(EVAL_TOML)
    make_run(live, "r", lanes=["r9700"], config={"trainable_params": 30e6, "total_params": 3e9})
    analysis = live.get("/api/training/runs/r/analysis")[1]
    assert analysis["mfu"]["estimate"] is True and analysis["mfu"]["peak_tflops"] == 96.0 and analysis["mfu"]["mfu_trainable_basis"]
    assert analysis["energy"]["joules"] is None  # no power samples on this host: null, never 0
    detail = live.get("/api/training/runs/r")[1]
    assert {"run", "events", "checkpoints", "hardware_summary", "artifacts", "notes", "alerts"} <= set(detail)
    live.post("/api/training/runs/r/events", {"kind": "gpu_reset", "severity": "error", "step": 5})
    kinds = {a["kind"] for a in live.get("/api/training/alerts")[1]["alerts"]}
    assert "gpu_fault" in kinds
    assert live.get("/api/training/runs/missing/analysis")[0] == 404
    assert live.get("/api/training/runs/r/ranks")[1]["ranks"] == []


def test_resume_report_endpoint(live_app):
    live = live_app()
    make_run(live, "parent")
    live.post("/api/training/runs", {"id": "child", "name": "child", "parent_run_id": "parent"})
    live.post("/api/training/runs/child/events", {"kind": "resumed", "payload": {"from_step": 100, "manifest_valid": True, "resume_state_complete": False,
                                                                                 "resume_state_missing": ["optimizer"]}})
    report = live.get("/api/training/runs/child/resume-report")[1]
    assert report["verdict"] in {"warning", "fail"} and report["checks"][0]["name"] == "manifest_valid"


# ------------------------------------------------------------------ artifacts


def test_artifacts_download_only_from_the_artifact_folder(live_app, tmp_path):
    live = live_app()
    body = make_run(live, "r")
    inside = Path(body["artifact_dir"])
    inside.mkdir(parents=True)
    trace = inside / "trace.pftrace"
    trace.write_bytes(b"ROCPROF-TRACE-BYTES")
    outside = tmp_path / "secret.txt"
    outside.write_text("do not serve me")
    good = live.post("/api/training/runs/r/artifacts", {"kind": "rocprof_trace", "path": str(trace), "downloadable": True})[1]
    bad = live.post("/api/training/runs/r/artifacts", {"kind": "log", "path": str(outside), "downloadable": True})[1]
    traversal = live.post("/api/training/runs/r/artifacts", {"kind": "log", "path": str(inside / ".." / ".." / "secret.txt")})[1]
    assert good["downloadable"] is True and bad["downloadable"] is False and traversal["downloadable"] is False  # the client's claim is ignored
    status, payload = live.get(f"/api/training/runs/r/artifacts/{good['id']}")
    assert status == 200 and payload["raw"] == b"ROCPROF-TRACE-BYTES"
    assert live.get(f"/api/training/runs/r/artifacts/{bad['id']}")[0] == 404
    assert live.get(f"/api/training/runs/r/artifacts/{traversal['id']}")[0] == 404
    assert live.post("/api/training/runs/r/artifacts", {"kind": "x"})[0] == 400
    assert len(live.get("/api/training/runs/r/artifacts")[1]["artifacts"]) == 3


def test_tracker_attach_artifact_copies_into_the_dashboard_folder(live_app, tmp_path):
    from dual_gpu_setup.training.tracker import TrainingTracker

    live = live_app()
    tracker = TrainingTracker(live.base, "t", run_id="art", log_every=2, capture_environment=False, flush_s=0.05,
                              spool_dir=tmp_path / "s").start()
    deadline = __import__("time").time() + 5
    while tracker.artifact_dir is None and __import__("time").time() < deadline:
        __import__("time").sleep(0.05)
    source = tmp_path / "rocprof.json"
    source.write_text("{}")
    tracker.attach_artifact(source, kind="rocprof_trace", note="first 20 steps")
    tracker.finish("completed")
    artifact = live.get("/api/training/runs/art/artifacts")[1]["artifacts"][0]
    assert artifact["downloadable"] and artifact["meta"] == {"note": "first 20 steps"} and Path(artifact["path"]).parent == Path(tracker.artifact_dir)


# ----------------------------------------------------- forecast, notes, examples


def test_forecast_notes_examples_and_profiles(live_app):
    live = live_app()
    config = {"max_seq_length": 2048, "per_device_train_batch_size": 1, "bf16": True, "total_params": 1e9, "trainable_params": 10e6,
              "hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 16, "vocab_size": 32000}
    forecast = live.post("/api/training/forecast", {"lane": "9070xt", "config": config})[1]
    assert forecast["source"] == "estimate" and forecast["fits"] is True and forecast["budget_bytes"] == pytest.approx(16 * 1024**3 * 0.9)
    assert live.post("/api/training/forecast", {"lane": "nope", "config": config})[0] == 400
    make_run(live, "r", config=config, lanes=["9070xt"])
    live.post("/api/training/runs/r/finish", {"status": "completed"})
    assert len(live.get("/api/training/profiles")[1]["profiles"]) == 1
    assert live.post("/api/training/forecast", {"lane": "9070xt", "config": config})[1]["source"] == "measured"
    assert live.post("/api/training/runs/r/notes", {"author": "me", "hypothesis": "h", "outcome": "mail a@b.co"})[0] == 200
    assert live.post("/api/training/notes", {"hypothesis": "global"})[0] == 200 and len(live.get("/api/training/notes")[1]["notes"]) == 2
    assert live.post("/api/training/notes", {})[0] == 400
    assert live.post("/api/training/runs/r/examples", {"step": 100, "items": [{"example_id": "e1", "loss": 3.0}, {"example_id": "e2", "loss": 0.1}]})[1] == {"stored": 2}
    assert live.get("/api/training/runs/r/hard-examples")[1]["examples"][0]["example_id"] == "e1"


def test_unknown_routes_and_run_ids_return_404(live_app):
    live = live_app()
    assert live.get("/api/training/nope")[0] == 404
    assert live.get("/api/training/runs/missing")[0] == 404
    assert live.post("/api/training/runs/missing/metrics", {})[0] == 404
    assert live.get("/api/training/benchmarks/unknown")[0] == 404
    assert live.post("/api/training/benchmarks", {"kind": "x"})[0] == 400


def test_create_run_attaches_server_hardware_and_gpu_map(live_app):
    live = live_app()
    payload = {"name": "gpu-map", "env": {"torch": {"devices": [{"index": 0, "name": "gfx", "total_memory": 32 * 1024**3}]}}}
    run = live.post("/api/training/runs", payload)[1]
    detail = live.get(f"/api/training/runs/{run['id']}")[1]["run"]
    assert "server_hardware" in detail["env"] and detail["env"]["gpu_map"][0]["torch_index"] == 0
