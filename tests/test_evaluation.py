from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import write_gguf  # noqa: E402
from dual_gpu_setup.config import EvaluationConfig  # noqa: E402
from dual_gpu_setup.training import selection  # noqa: E402
from dual_gpu_setup.training.evaluation import (  # noqa: E402
    AdapterServerManager, EvalQueue, EvalWorker, EvaluationPipeline, check_adapter_compat, read_safetensors_header,
)
from dual_gpu_setup.training.store import TrainingStore  # noqa: E402


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


# ------------------------------------------------------------------- queue


def test_queue_orders_by_priority_and_merges_duplicates(tmp_path):
    q = EvalQueue(tmp_path / "q.sqlite3", capacity=8)
    q.submit("r", 100, "/c/100")
    q.submit("r", 200, "/c/200", designated=True)
    q.submit("r", 300, "/c/300", final=True)
    again = q.submit("r", 100, "/c/100", best=True)  # a later request upgrades the queued job
    assert again["duplicate"] is True
    order = [q.claim()["step"] for _ in range(3)]
    assert order == [300, 100, 200]  # final(100) > best(80) > designated(60)
    assert q.claim() is None


def test_queue_coalesces_routine_checkpoints_and_never_drops_protected(tmp_path):
    q = EvalQueue(tmp_path / "q.sqlite3", capacity=3)
    q.submit("r", 10, "/c/10", designated=True)
    for step in (20, 30, 40):  # only the newest routine checkpoint per run survives
        q.submit("r", step, f"/c/{step}")
    jobs = {j["step"] for j in q.jobs()}
    assert jobs == {10, 40} and q.stats()["dropped_total"] == 2
    q.submit("r", 50, "/c/50", designated=True)
    q.submit("r", 60, "/c/60", final=True)  # full of protected work + one routine: the routine one is dropped
    assert {j["step"] for j in q.jobs()} == {10, 50, 60}
    rejected = q.submit("r", 70, "/c/70")
    assert rejected["accepted"] is False and "protected" in rejected["reason"]
    assert q.submit("r", 80, "/c/80", final=True)["accepted"] is True  # protected work may exceed capacity (to the hard cap)
    assert q.stats()["depth"] == 4


def test_queue_stats_lag_and_crash_recovery(tmp_path):
    clock = Clock()
    q = EvalQueue(tmp_path / "q.sqlite3", clock=clock)
    q.submit("r", 100, "/c/100", designated=True)
    clock.now += 30
    stats = q.stats(latest_training_step=400)
    assert stats["oldest_wait_s"] == 30 and stats["lag_steps"] == 300
    job = q.claim()
    assert q.stats()["running"] == 1
    assert q.recover_interrupted() == 1 and q.stats()["depth"] == 1
    q.finish(job["id"], "completed")


# ------------------------------------------------------------ compatibility


def make_adapter(directory: Path, *, layers=4, targets=("q_proj", "v_proj"), base="org/base", revision="abc", tokenizer=True, **config):
    directory.mkdir(parents=True, exist_ok=True)
    cfg = {"peft_type": "LORA", "base_model_name_or_path": base, "revision": revision, "r": 16, "lora_alpha": 32,
           "target_modules": list(targets), **config}
    (directory / "adapter_config.json").write_text(json.dumps(cfg))
    tensors = {f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight": {"dtype": "F32", "shape": [16, 64], "data_offsets": [0, 1]}
               for i in range(layers)}
    header = json.dumps(tensors).encode()
    (directory / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\x00")
    if tokenizer:
        (directory / "tokenizer.json").write_text('{"v": 1}')
    return directory


def make_base_gguf(path: Path, blocks=4):
    return write_gguf(path, {"general.architecture": "llama", "llama.block_count": blocks})


def test_safetensors_header_is_read_without_tensors(tmp_path):
    adapter = make_adapter(tmp_path / "a", layers=2)
    header = read_safetensors_header(adapter / "adapter_model.safetensors")
    assert len(header) == 2 and all("lora_A" in k for k in header)


def test_compat_accepts_a_good_adapter_and_reports_details(tmp_path):
    adapter = make_adapter(tmp_path / "a")
    gguf = make_base_gguf(tmp_path / "base.gguf", blocks=4)
    report = check_adapter_compat(adapter, base_gguf=gguf, expected={"base_model": "org/base", "revision": "abc"})
    assert report.ok and report.adapter["adapter_layers"] == 4 and report.adapter["gguf_block_count"] == 4
    assert report.adapter["rank"] == 16 and report.adapter["tokenizer_hash"]


@pytest.mark.parametrize("kwargs,expected,needle", [
    ({"targets": ("q_proj", "conv1d")}, {}, "Unsupported LoRA target"),
    ({"use_dora": True}, {}, "DoRA"),
    ({"base": "org/other"}, {"base_model": "org/base"}, "expected 'org/base'"),
    ({"revision": "zzz"}, {"revision": "abc"}, "Base revision"),
    ({"layers": 9}, {}, "only 4 layers"),
])
def test_compat_rejects_incompatible_adapters(tmp_path, kwargs, expected, needle):
    adapter = make_adapter(tmp_path / "a", **kwargs)
    report = check_adapter_compat(adapter, base_gguf=make_base_gguf(tmp_path / "b.gguf", blocks=4), expected=expected)
    assert not report.ok and any(needle in e for e in report.errors)


def test_compat_tokenizer_drift_and_missing_files(tmp_path):
    adapter = make_adapter(tmp_path / "a")
    assert not check_adapter_compat(adapter, expected={"tokenizer_hash": "deadbeef"}).ok
    no_tok = make_adapter(tmp_path / "b", tokenizer=False)
    assert check_adapter_compat(no_tok).warnings
    assert not check_adapter_compat(tmp_path / "missing").ok
    (adapter / "adapter_model.safetensors").unlink()
    assert any("not found" in e for e in check_adapter_compat(adapter).errors)


# ------------------------------------------------------------ server lifecycle


class FakeServer:
    instances: list["FakeServer"] = []

    def __init__(self, base, adapter, clock):
        self.base_url, self.stopped, self.clock = f"http://fake/{base.name}", False, clock
        FakeServer.instances.append(self)

    def start(self):
        self.clock.now += 4.0  # model load

    def stop(self):
        self.stopped = True
        self.clock.now += 1.5


def test_server_restarts_only_when_the_adapter_or_base_changes_and_times_the_changeover(tmp_path):
    FakeServer.instances.clear()
    clock = Clock(0.0)
    manager = AdapterServerManager(lambda base, adapter: FakeServer(base, adapter, clock), clock=clock)
    a1, a2 = tmp_path / "a1.gguf", tmp_path / "a2.gguf"
    a1.write_bytes(b"adapter-one")
    a2.write_bytes(b"adapter-two")
    base = tmp_path / "base.gguf"
    first = manager.ensure(base, a1)
    assert first["restarted"] and first["changeover_ms"] == pytest.approx(4000)
    assert manager.ensure(base, a1) == {**manager.ensure(base, a1)} and not manager.ensure(base, a1)["restarted"]
    second = manager.ensure(base, a2)  # new adapter -> stop old (1.5 s) + start new (4 s)
    assert second["restarted"] and second["stop_ms"] == pytest.approx(1500) and second["changeover_ms"] == pytest.approx(5500)
    assert FakeServer.instances[0].stopped
    a1.write_bytes(b"adapter-one-CHANGED-IN-PLACE")  # same path, new content: must still restart
    assert manager.ensure(base, a1)["restarted"]
    manager.close()
    assert FakeServer.instances[-1].stopped


# ------------------------------------------------------------------- pipeline


def settings(tmp_path, **kw):
    base = dict(enabled=True, lane="9070xt", base_model="org/base", base_f16_gguf=str(make_base_gguf(tmp_path / "f16.gguf")),
                base_quant_gguf=str(make_base_gguf(tmp_path / "q4.gguf")), converter_script="convert_lora_to_gguf.py",
                native_command=["native", "{adapter_dir}", "{output_json}"], gguf_command=["gguf", "{level}", "{base_url}", "{output_json}"],
                workdir=str(tmp_path / "work"))
    base.update(kw)
    return EvaluationConfig(**base)


class Runner:
    """Stands in for subprocess: fakes the converter and the eval harness."""

    def __init__(self, scores, fail_convert=False, fail_level=None, latency=900.0):
        self.scores, self.fail_convert, self.fail_level, self.latency, self.calls = scores, fail_convert, fail_level, latency, []

    def __call__(self, command, env, timeout):
        self.calls.append(command)
        if "convert_lora_to_gguf.py" in command:
            if self.fail_convert:
                return subprocess.CompletedProcess(command, 1, "", "unsupported tensor name")
            Path(command[command.index("--outfile") + 1]).write_bytes(b"gguf-adapter")
            return subprocess.CompletedProcess(command, 0, "", "")
        level = "native" if command[0] == "native" else command[1]
        if level == self.fail_level:
            return subprocess.CompletedProcess(command, 2, "", "harness crashed")
        Path(command[-1]).write_text(json.dumps({"metrics": {"task_score": self.scores[level]}, "latency": {"tokens_per_second": self.latency}}))
        return subprocess.CompletedProcess(command, 0, "", "")


def pipeline_setup(tmp_path, runner):
    store = TrainingStore(tmp_path / "t.sqlite3")
    store.create_run({"id": "r", "name": "run", "config": {"base_model": "org/base"}})
    adapter = make_adapter(tmp_path / "ckpt-100")
    store.add_checkpoint("r", {"step": 100, "path": str(adapter), "valid": True, "manifest": {"files": [1], "tokens_seen": 5}})
    FakeServer.instances.clear()
    clock = Clock(0.0)
    servers = AdapterServerManager(lambda base, ad: FakeServer(base, ad, clock), clock=clock)
    cfg = settings(tmp_path)
    return store, EvaluationPipeline(cfg, store, servers, runner=runner, workdir=tmp_path / "work"), adapter


def test_pipeline_records_all_three_levels_with_latency_and_versions(tmp_path):
    runner = Runner({"native": 0.82, "gguf_f16": 0.81, "gguf_quant": 0.78})
    store, pipeline, adapter = pipeline_setup(tmp_path, runner)
    outcomes = pipeline.evaluate({"run_id": "r", "step": 100, "checkpoint_path": str(adapter)})
    assert outcomes == {"native": "completed", "gguf_f16": "completed", "gguf_quant": "completed"}
    evals = store.evals("r")[100]
    assert evals["native"]["default"]["metrics"]["task_score"] == 0.82
    quant = evals["gguf_quant"]["default"]
    assert quant["metrics"]["task_score"] == 0.78 and quant["latency"]["server_restarted"] is True
    assert quant["latency"]["changeover_ms"] > 0 and "conversion_ms" in quant["latency"]
    assert quant["versions"]["base_gguf"].endswith("q4.gguf") and evals["gguf_f16"]["default"]["versions"]["adapter"]["rank"] == 16
    assert len(FakeServer.instances) == 2  # one restart per base model


def test_failed_conversion_is_an_explicit_failure_not_a_silent_skip(tmp_path):
    runner = Runner({"native": 0.8}, fail_convert=True)
    store, pipeline, adapter = pipeline_setup(tmp_path, runner)
    outcomes = pipeline.evaluate({"run_id": "r", "step": 100, "checkpoint_path": str(adapter)})
    assert outcomes == {"native": "completed", "gguf_f16": "failed", "gguf_quant": "failed"}
    evals = store.evals("r")[100]
    assert "conversion failed" in evals["gguf_f16"]["default"]["error"] and evals["gguf_quant"]["default"]["status"] == "failed"
    assert FakeServer.instances == []  # never started a server for an adapter that did not convert


def test_failure_at_one_level_does_not_hide_the_others(tmp_path):
    runner = Runner({"native": 0.8, "gguf_f16": 0.79, "gguf_quant": 0.7}, fail_level="gguf_f16")
    store, pipeline, adapter = pipeline_setup(tmp_path, runner)
    outcomes = pipeline.evaluate({"run_id": "r", "step": 100, "checkpoint_path": str(adapter)})
    assert outcomes == {"native": "completed", "gguf_f16": "failed", "gguf_quant": "completed"}
    assert "harness crashed" in store.evals("r")[100]["gguf_f16"]["default"]["error"]


def test_incompatible_adapter_fails_every_level_before_using_a_gpu(tmp_path):
    runner = Runner({})
    store, pipeline, adapter = pipeline_setup(tmp_path, runner)
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    cfg["target_modules"] = ["conv1d"]
    (adapter / "adapter_config.json").write_text(json.dumps(cfg))
    outcomes = pipeline.evaluate({"run_id": "r", "step": 100, "checkpoint_path": str(adapter)})
    assert set(outcomes.values()) == {"failed"} and runner.calls == []
    assert "compatibility" in store.evals("r")[100]["native"]["default"]["error"]


def test_worker_honours_lane_readiness_and_updates_selection(tmp_path):
    runner = Runner({"native": 0.8, "gguf_f16": 0.79, "gguf_quant": 0.77})
    store, pipeline, adapter = pipeline_setup(tmp_path, runner)
    queue = EvalQueue(tmp_path / "q.sqlite3")
    queue.submit("r", 100, str(adapter), final=True)
    ready, done, leases = {"lane": False}, [], []
    worker = EvalWorker(queue, pipeline, lane_ready=lambda: ready["lane"], acquire_lane=lambda: leases.append("acquire") or "L",
                        release_lane=lambda lease: leases.append(f"release:{lease}"),
                        on_done=lambda job, outcomes: done.append((job["step"], outcomes)))
    assert worker.run_once() is False and queue.stats()["depth"] == 1  # inference holds the lane: job waits
    ready["lane"] = True
    assert worker.run_once() is True
    assert queue.stats()["completed"] == 1 and done[0][0] == 100 and leases == ["acquire", "release:L"]


# ----------------------------------------------------------------- selection


def seed_scorecard(store, rows):
    """rows: step -> (native, f16, quant, latency, loss)"""
    store.create_run({"id": "s", "name": "s"})
    windows = []
    for step, (native, f16, quant, latency, loss) in rows.items():
        store.add_checkpoint("s", {"step": step, "path": f"/c/{step}", "valid": True, "manifest": {"files": [1], "tokens_seen": step * 100}})
        for level, score in (("native", native), ("gguf_f16", f16), ("gguf_quant", quant)):
            if score is not None:
                store.record_eval("s", {"step": step, "level": level, "metrics": {"task_score": score},
                                        "latency": {"eval_wall_ms": latency}})
        windows.append({"step": step, "window_steps": 10, "phase": "train", "loss": loss, "nonpad_tokens": 100, "step_time_ms": 100})
    store.record_metrics("s", {"steps": windows})


def test_selection_uses_the_deployed_score_not_training_loss(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    seed_scorecard(store, {100: (0.70, 0.70, 0.68, 800, 1.0), 200: (0.80, 0.79, 0.77, 800, 0.6), 300: (0.84, 0.83, 0.70, 800, 0.3)})
    cfg = EvaluationConfig(max_f16_to_quant_drop=0.03)
    result = selection.select_best(store, "s", cfg, apply=True)
    assert result["selected_step"] == 200 and result["deployable"] and result["level"] == "gguf_quant"
    assert result["loss_only_choice"] == 300 and result["differs_from_loss_only_choice"]
    flagged = {r["step"]: r["flags"] for r in result["scorecard"]}
    assert any("quantization hurts" in f for f in flagged[300]) and not flagged[200]
    assert [c["step"] for c in store.checkpoints("s") if c["best"]] == [200]


def test_selection_without_deployment_scores_falls_back_and_says_so(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    seed_scorecard(store, {100: (0.7, None, None, 0, 1.0), 200: (0.8, None, None, 0, 0.9)})
    result = selection.select_best(store, "s", EvaluationConfig())
    assert result["selected_step"] == 200 and result["deployable"] is False and result["level"] == "native"
    empty = TrainingStore(tmp_path / "e.sqlite3")
    empty.create_run({"id": "x", "name": "x"})
    assert selection.select_best(empty, "x", EvaluationConfig())["selected_step"] is None


def test_conversion_drop_and_failures_are_flagged(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    seed_scorecard(store, {100: (0.90, 0.80, None, 800, 1.0)})
    store.record_eval("s", {"step": 100, "level": "gguf_quant", "status": "failed", "error": "OOM on 9070 XT"})
    flags = selection.scorecard(store, "s", EvaluationConfig())[0]["flags"]
    assert any("conversion" in f for f in flags) and any("OOM" in f for f in flags)


def test_regression_detection_for_quality_and_inference_speed(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    seed_scorecard(store, {100: (0.7, 0.7, 0.70, 800, 1.0), 200: (0.8, 0.8, 0.78, 800, 0.8),
                           300: (0.8, 0.8, 0.70, 800, 0.7), 400: (0.8, 0.8, 0.79, 1200, 0.6)})
    findings = selection.detect_regressions(store, "s", EvaluationConfig(max_f16_to_quant_drop=0.03, max_latency_regression_pct=20))
    kinds = {(f["step"], f["kind"]) for f in findings}
    assert (300, "quality_regression") in kinds and (400, "speed_regression") in kinds
    assert (200, "quality_regression") not in kinds
