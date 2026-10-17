"""Checkpoint selection and regression detection.

The best checkpoint is the one that scores best in the DEPLOYED configuration (quantized GGUF), not
the one with the lowest training loss. Native -> F16 -> quant drops are reported per checkpoint so a
conversion or quantization problem is separated from a genuinely worse model.
"""

from __future__ import annotations

from typing import Any

from dual_gpu_setup.config import EvaluationConfig

from .store import TrainingStore

LEVELS = ("native", "gguf_f16", "gguf_quant")


def _metric(evals: dict[str, Any], level: str, metric: str, eval_set: str) -> float | None:
    entry = (evals.get(level) or {}).get(eval_set)
    if not entry or entry["status"] != "completed":
        return None
    value = (entry["metrics"] or {}).get(metric)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _latency(evals: dict[str, Any], level: str, eval_set: str, key: str) -> float | None:
    entry = (evals.get(level) or {}).get(eval_set)
    value = ((entry or {}).get("latency") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def scorecard(store: TrainingStore, run_id: str, settings: EvaluationConfig, eval_set: str = "default",
              latency_key: str = "eval_wall_ms") -> list[dict[str, Any]]:
    """One row per checkpoint: per-level score, adjacent-level drops, failures, and latency."""
    sign = 1.0 if settings.higher_is_better else -1.0
    metric = settings.primary_metric
    rows = []
    for ckpt in sorted(store.checkpoints(run_id), key=lambda c: c["step"]):
        evals = ckpt["evals"]
        scores = {level: _metric(evals, level, metric, eval_set) for level in LEVELS}
        failed = {level: (evals.get(level) or {}).get(eval_set, {}).get("error")
                  for level in LEVELS if ((evals.get(level) or {}).get(eval_set) or {}).get("status") == "failed"}
        drop_conv = None if scores["native"] is None or scores["gguf_f16"] is None else sign * (scores["native"] - scores["gguf_f16"])
        drop_quant = None if scores["gguf_f16"] is None or scores["gguf_quant"] is None else sign * (scores["gguf_f16"] - scores["gguf_quant"])
        flags = []
        if drop_conv is not None and drop_conv > settings.max_native_to_f16_drop:
            flags.append(f"native->GGUF-F16 drop {drop_conv:.3f} exceeds {settings.max_native_to_f16_drop}: suspect conversion/prompt-format/engine")
        if drop_quant is not None and drop_quant > settings.max_f16_to_quant_drop:
            flags.append(f"F16->quant drop {drop_quant:.3f} exceeds {settings.max_f16_to_quant_drop}: quantization hurts this checkpoint")
        for level, reason in failed.items():
            flags.append(f"{level} evaluation failed: {reason}")
        rows.append({
            "step": ckpt["step"], "valid": ckpt["valid"], "tokens_seen": ckpt["tokens_seen"], "best": ckpt["best"],
            "designated": ckpt["designated"], "scores": scores, "drop_native_to_f16": drop_conv,
            "drop_f16_to_quant": drop_quant, "flags": flags, "latency_ms": _latency(evals, "gguf_quant", eval_set, latency_key),
            "deploy_latency": {k: _latency(evals, "gguf_quant", eval_set, k) for k in ("eval_wall_ms", "tokens_per_second", "changeover_ms")},
        })
    return rows


def training_loss_choice(store: TrainingStore, run_id: str, steps: list[int]) -> int | None:
    """The checkpoint a loss-only rule would pick (shown to prove selection did not rely on it)."""
    windows = [w for w in store.step_windows(run_id, phase=None) if w["loss"] is not None]
    best_step, best_loss = None, None
    for step in steps:
        near = [w for w in windows if w["step"] <= step][-3:]
        if near:
            loss = sum(w["loss"] for w in near) / len(near)
            if best_loss is None or loss < best_loss:
                best_step, best_loss = step, loss
    return best_step


def select_best(store: TrainingStore, run_id: str, settings: EvaluationConfig, eval_set: str = "default",
                apply: bool = False) -> dict[str, Any]:
    card = scorecard(store, run_id, settings, eval_set)
    sign = 1.0 if settings.higher_is_better else -1.0
    eligible = [r for r in card if r["valid"] and r["scores"]["gguf_quant"] is not None]
    deployable = True
    if not eligible:  # no deployment-level score yet: rank by native quality but say so
        eligible = [r for r in card if r["valid"] and r["scores"]["native"] is not None]
        deployable = False
    key_level = "gguf_quant" if deployable else "native"
    chosen = None
    if eligible:
        # Highest deployed score; ties go to lower latency, then the later (more trained) step.
        chosen = max(eligible, key=lambda r: (sign * r["scores"][key_level], -(r["latency_ms"] or 0.0), r["step"]))
    loss_choice = training_loss_choice(store, run_id, [r["step"] for r in card]) if card else None
    result = {
        "metric": settings.primary_metric, "level": key_level if chosen else None,
        "selected_step": chosen["step"] if chosen else None, "deployable": bool(chosen) and deployable,
        "loss_only_choice": loss_choice, "differs_from_loss_only_choice": bool(chosen) and loss_choice != chosen["step"],
        "flags": chosen["flags"] if chosen else [],
        "reason": None if chosen else "no checkpoint has a completed evaluation yet",
        "scorecard": card,
    }
    if apply and chosen:
        store.set_checkpoint_flags(run_id, chosen["step"], best=True)
        result["applied"] = True
    return result


def detect_regressions(store: TrainingStore, run_id: str, settings: EvaluationConfig, eval_set: str = "default",
                       latency_key: str = "eval_wall_ms") -> list[dict[str, Any]]:
    """Compare each checkpoint with the best deployable one before it: quality and speed regressions."""
    sign = 1.0 if settings.higher_is_better else -1.0
    card = scorecard(store, run_id, settings, eval_set, latency_key)
    findings: list[dict[str, Any]] = []
    best_prev: dict[str, Any] | None = None
    for row in card:
        score = row["scores"]["gguf_quant"]
        if score is None:
            continue
        if best_prev is not None:
            quality_drop = sign * (best_prev["scores"]["gguf_quant"] - score)
            if quality_drop > settings.max_f16_to_quant_drop:
                findings.append({"step": row["step"], "kind": "quality_regression", "vs_step": best_prev["step"],
                                 "detail": f"{settings.primary_metric} fell by {quality_drop:.3f} against step {best_prev['step']}"})
            prev_latency, latency = best_prev["latency_ms"], row["latency_ms"]
            if prev_latency and latency and latency > prev_latency * (1 + settings.max_latency_regression_pct / 100.0):
                findings.append({"step": row["step"], "kind": "speed_regression", "vs_step": best_prev["step"],
                                 "detail": f"{latency_key} rose {((latency / prev_latency) - 1) * 100:.0f}% against step {best_prev['step']}"})
        if best_prev is None or sign * score > sign * best_prev["scores"]["gguf_quant"]:
            best_prev = row
    return findings
