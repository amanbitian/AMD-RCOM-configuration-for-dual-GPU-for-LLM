"""Exercises HFTrackerCallback against a fake `transformers`/model (the real packages are not needed:
the callback only uses the TrainerCallback base class, a model's hook registration, and the Trainer's
`args` / `state` objects)."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_gpu_setup.training.tracker import TrainingTracker  # noqa: E402


class FakeTensor:
    """Just enough tensor: numel(), sum(), slicing, != comparison, + with ints."""

    def __init__(self, rows):
        self.rows = rows

    def numel(self):
        return sum(len(r) for r in self.rows)

    def sum(self):
        return sum(sum(int(v) for v in r) for r in self.rows)

    def __getitem__(self, key):
        _, cols = key  # (Ellipsis, slice)
        return FakeTensor([r[cols] for r in self.rows])

    def __ne__(self, other):
        return FakeTensor([[v != other for v in r] for r in self.rows])


class FakeModel:
    def __init__(self):
        self.training, self.pre, self.post = True, [], []

    def parameters(self):
        return [SimpleNamespace(numel=lambda: 1000, requires_grad=False), SimpleNamespace(numel=lambda: 50, requires_grad=True)]

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        self.pre.append(hook)
        return SimpleNamespace(remove=lambda: self.pre.remove(hook))

    def register_forward_hook(self, hook, with_kwargs=False):
        self.post.append(hook)
        return SimpleNamespace(remove=lambda: self.post.remove(hook))

    def forward(self, **kwargs):
        for hook in list(self.pre):
            hook(self, (), kwargs)
        for hook in list(self.post):
            hook(self, (), kwargs, None)


@pytest.fixture
def callback_module(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", fake)
    import dual_gpu_setup.training.hf_callback as module

    module = importlib.reload(module)
    yield module
    monkeypatch.delitem(sys.modules, "transformers")
    importlib.reload(module)


def test_callback_counts_tokens_across_accumulation_and_reports_the_run(callback_module, live_app, tmp_path):
    live = live_app()
    tracker = TrainingTracker(live.base, "hf-run", run_id="hf1", log_every=2, warmup_windows=0, flush_s=0.05,
                              capture_environment=False, spool_dir=tmp_path / "spool")
    callback = callback_module.HFTrackerCallback(tracker)
    model = FakeModel()
    args = SimpleNamespace(logging_steps=2, output_dir=str(tmp_path / "out"), to_dict=lambda: {"learning_rate": 1e-4})
    state = SimpleNamespace(max_steps=8, global_step=0)
    callback.on_train_begin(args, state, None, model=model)
    assert len(model.pre) == 1 and len(model.post) == 1 and tracker.max_steps == 8 and tracker.log_every == 2
    assert tracker.config["trainable_params"] == 50 and tracker.config["total_params"] == 1050

    (tmp_path / "out" / "checkpoint-4").mkdir(parents=True)
    (tmp_path / "out" / "checkpoint-4" / "adapter_model.safetensors").write_bytes(b"w" * 64)
    labels = FakeTensor([[-100, 5, 6, 7], [-100, -100, 9, 9]])  # shifted supervised tokens: [5,6,7]->3? see below
    mask = FakeTensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    ids = FakeTensor([[1, 2, 3, 4], [0, 6, 7, 8]])
    for step in range(1, 5):
        callback.on_step_begin(args, state, None)
        for _ in range(2):  # gradient accumulation: two microbatches per optimizer step
            model.forward(input_ids=ids, attention_mask=mask, labels=labels)
        model.training = False
        model.forward(input_ids=ids, attention_mask=mask, labels=labels)  # an eval forward must NOT be counted
        model.training = True
        state.global_step = step
        callback.on_step_end(args, state, None)
        if step % 2 == 0:
            callback.on_log(args, state, None, logs={"loss": 1.0 / step, "learning_rate": 1e-4, "grad_norm": 0.5})
    callback.on_evaluate(args, state, None, metrics={"eval_loss": 0.9, "eval_runtime": 1.5})
    state.global_step = 4
    callback.on_save(args, state, None)
    callback.on_train_end(args, state, None)

    run = live.state.training.get_run("hf1")
    assert run["recorded_status"] == "completed" and run["max_steps"] == 8 and run["config"]["trainable_params"] == 50
    assert run["tokens"]["processed"] == 4 * 2 * 8       # input_ids.numel() = 8, two microbatches per step, four steps
    assert run["tokens"]["nonpad"] == 4 * 2 * 7          # attention_mask.sum() = 7
    assert run["tokens"]["supervised"] == 4 * 2 * 5      # labels[..., 1:] != -100 -> [5,6,7] and [9,9] = 5
    assert run["latest_scalars"]["loss"] == pytest.approx(0.25)
    assert model.pre == [] and model.post == []          # hooks removed at train end
    kinds = [e["kind"] for e in live.state.training.events("hf1")]
    assert "eval" in kinds
    ckpts = live.state.training.checkpoints("hf1")
    assert ckpts and ckpts[0]["step"] == 4 and ckpts[0]["valid"]  # manifest written after on_save


def test_callback_requires_transformers():
    sys.modules.pop("transformers", None)
    import dual_gpu_setup.training.hf_callback as module

    module = importlib.reload(module)
    if module._Base is object:
        with pytest.raises(ImportError):
            module.HFTrackerCallback(TrainingTracker(enabled=False))
