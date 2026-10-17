"""Hugging Face Trainer integration. `transformers` is imported lazily so the rest of the
package works without it."""

from __future__ import annotations

import os
from typing import Any

from .tracker import TrainingTracker

try:
    from transformers import TrainerCallback as _Base
except ImportError:  # pragma: no cover - exercised only without transformers installed
    _Base = object


class HFTrackerCallback(_Base):  # type: ignore[misc, valid-type]
    """Feeds a TrainingTracker from a Trainer without replacing the training loop.

    Tokens are counted by a forward pre-hook on the model: it sees the tensors already on the
    device (after any collator/padding and across gradient accumulation) and only accumulates
    device-side sums, so there is no per-microbatch synchronisation. Forward and optimizer time
    use CUDA events read once per window; backward is the remainder of the GPU step. Set the
    Trainer's `logging_steps` equal to the tracker's `log_every` so loss/lr/grad_norm line up with
    windows.
    """

    def __init__(self, tracker: TrainingTracker, profile_gpu_phases: bool = True):
        if _Base is object:
            raise ImportError("HFTrackerCallback requires the `transformers` package.")
        self.tracker = tracker
        self.profile = profile_gpu_phases
        self._hooks: list[Any] = []
        self._forward_start: Any = None
        self._optimizer_start: Any = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        tracker = self.tracker
        if tracker.log_every is None:
            tracker.log_every = max(1, int(getattr(args, "logging_steps", 20) or 20))
        if tracker.max_steps is None and state.max_steps and state.max_steps > 0:
            tracker.max_steps = int(state.max_steps)
        config = dict(tracker.config) if tracker.config else (args.to_dict() if hasattr(args, "to_dict") else {})
        if model is not None:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            config.setdefault("trainable_params", trainable)
            config.setdefault("total_params", total)
        tracker.config = config
        tracker.start()
        resumed = getattr(state, "global_step", 0)
        if resumed and tracker._pending_resume is None:
            tracker.event("resumed", step=resumed, from_step=resumed)
        if model is not None:
            def pre(module, hook_args, hook_kwargs):
                if module.training:
                    tracker.microbatch(
                        hook_kwargs.get("input_ids"), hook_kwargs.get("attention_mask"), hook_kwargs.get("labels")
                    )
                    if self.profile:
                        self._forward_start = tracker.gpu_mark()

            def post(module, hook_args, hook_kwargs, output):
                if self.profile and module.training and self._forward_start is not None:
                    tracker.gpu_record("forward", self._forward_start)
                    self._forward_start = None

            self._hooks.append(model.register_forward_pre_hook(pre, with_kwargs=True))
            self._hooks.append(model.register_forward_hook(post, with_kwargs=True))

    def on_step_begin(self, args, state, control, **kwargs):
        self.tracker.step_begin()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.profile:
            self._optimizer_start = self.tracker.gpu_mark()

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.profile and self._optimizer_start is not None:
            self.tracker.gpu_record("optimizer", self._optimizer_start)
            self._optimizer_start = None

    def on_step_end(self, args, state, control, **kwargs):
        self.tracker.step_completed(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "eval_loss" in logs:
            return
        self.tracker.log_scalars(
            state.global_step,
            loss=logs.get("loss"),
            lr=logs.get("learning_rate"),
            grad_norm=logs.get("grad_norm"),
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        runtime = metrics.get("eval_runtime")
        if isinstance(runtime, (int, float)):
            self.tracker.add_excluded_ms(runtime * 1000.0)
        self.tracker.event("eval", step=state.global_step, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    def on_save(self, args, state, control, **kwargs):
        path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(path):
            self.tracker.checkpoint(state.global_step, path, kind="resume")

    def on_train_end(self, args, state, control, **kwargs):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.tracker.finish("completed")
