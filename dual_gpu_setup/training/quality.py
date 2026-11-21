"""Learning-quality series for the dashboard: raw and rolling-mean training loss, validation loss and perplexity
per validation set, the z-loss term, the total optimisation objective, gradient norm and learning rate.

Sources, all stored (nothing here is computed only for display):
  * `training_steps` rows: loss, lr, grad_norm and, in `extra_json`, whatever scalars the trainer attached
    (`train/z_loss_term`, `train/objective_total`, and any `val/<set>_loss` it forwarded);
  * `training_eval_points`: validation metrics read from the trainer's own `metrics.jsonl`, for trainers that log
    validation locally but do not forward it to the tracker (the stage-1 trainer does exactly that).

Validation sets are discovered, not hard-coded: every `val/<set>_loss` / `val/<set>_ppl` key becomes a set, so a run
with resume / JD / knowledge / general-text sets shows each one and a run with one combined set shows that one.
The rolling window is measured in optimiser STEPS (windows weighted by the steps they cover), default 100.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

_VAL = re.compile(r"^val/(?P<set>.+?)_(?P<kind>loss|ppl)$")
_ALIASES = {"z_loss_term": ("train/z_loss_term", "z_loss_term"),
            "objective_total": ("train/objective_total", "objective_total"),
            "ce_objective": ("train/ce_objective", "ce_objective")}
DEFAULT_WINDOW_STEPS = 100


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _stride(points: list[list[float]], max_points: int) -> list[list[float]]:
    if len(points) <= max_points:
        return points
    step = math.ceil(len(points) / max_points)
    thinned = points[::step]
    if thinned[-1] != points[-1]:
        thinned.append(points[-1])
    return thinned


def rolling_mean(rows: list[tuple[int, float, int]], window_steps: int) -> list[list[float]]:
    """rows = (step, value, steps covered by that row), ordered. Mean of the values whose step lies in
    (step - window_steps, step], weighted by the steps each row covers."""
    out, start, total, weight = [], 0, 0.0, 0.0
    for end, (step, value, span) in enumerate(rows):
        total += value * span
        weight += span
        while rows[start][0] <= step - window_steps and start < end:
            total -= rows[start][1] * rows[start][2]
            weight -= rows[start][2]
            start += 1
        out.append([step, total / weight if weight else value])
    return out


def ingest_metrics_jsonl(store: Any, run_id: str, path: str | Path) -> dict[str, int]:
    """Copy validation metrics from a trainer's `metrics.jsonl` into `training_eval_points`. Idempotent (one row
    per run, step and source) and read-only on the file; lines that are not JSON, or have no `val/` keys, are skipped."""
    path = Path(path)
    if not path.is_file():
        return {"read": 0, "stored": 0}
    # Runs that resume in one output folder share one metrics.jsonl: a line belongs to the run alive when it was
    # written (inside its time window, not past its last step), never to every run that used the folder.
    run = store.get_run(run_id)
    window_start = float(run["started_at"]) - 300
    window_end = float(run.get("finished_at") or store.clock()) + 900
    last_step = int(run.get("last_step") or 0)
    read = stored = 0
    with store.connect() as connection:
        # earlier versions attributed every line to every run: drop only this derived, mis-attributed copy
        connection.execute(
            "DELETE FROM training_eval_points WHERE run_id = ? AND source = 'metrics.jsonl' "
            "AND ((? > 0 AND step > ?) OR (ts IS NOT NULL AND (ts < ? OR ts > ?)))",
            (run_id, last_step, last_step, window_start, window_end))
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                metrics = {k: v for k, v in item.items() if isinstance(k, str) and k.startswith("val/") and _finite(v) is not None}
                step = item.get("step")
                if not metrics or not isinstance(step, (int, float)):
                    continue
                stamp = _finite(item.get("time"))
                if (last_step and step > last_step) or (stamp is not None and not window_start <= stamp <= window_end):
                    continue
                read += 1
                stored += connection.execute(
                    "INSERT OR IGNORE INTO training_eval_points (run_id, step, ts, source, metrics_json) VALUES (?,?,?,?,?)",
                    (run_id, int(step), _finite(item.get("time")), "metrics.jsonl", json.dumps(metrics))).rowcount
    return {"read": read, "stored": stored}


def quality_series(store: Any, run_id: str, window_steps: int = DEFAULT_WINDOW_STEPS, max_points: int = 800) -> dict[str, Any]:
    store.get_run(run_id)  # 404 for an unknown run
    window_steps = max(1, int(window_steps))
    max_points = max(10, min(int(max_points), 5000))
    raw: list[list[float]] = []
    spans: list[tuple[int, float, int]] = []
    series: dict[str, list[list[float]]] = {"grad_norm": [], "lr": [], **{k: [] for k in _ALIASES}}
    val: dict[str, dict[str, dict[int, float]]] = {}

    def add_val(step: int, key: str, value: Any) -> None:
        match = _VAL.match(key)
        number = _finite(value)
        if match and number is not None:
            val.setdefault(match["set"], {}).setdefault(match["kind"], {})[step] = number

    with store.connect() as connection:
        for row in connection.execute(
                "SELECT step, loss, lr, grad_norm, window_steps, extra_json FROM training_steps WHERE run_id = ? "
                "AND phase != 'warmup' ORDER BY step", (run_id,)):
            step = row["step"]
            if _finite(row["loss"]) is not None:
                raw.append([step, row["loss"]])
                spans.append((step, row["loss"], max(1, row["window_steps"] or 1)))
            for name, key in (("grad_norm", "grad_norm"), ("lr", "lr")):
                if _finite(row[key]) is not None:
                    series[name].append([step, row[key]])
            if row["extra_json"]:
                try:
                    extra = json.loads(row["extra_json"])
                except ValueError:
                    extra = {}
                for name, keys in _ALIASES.items():
                    value = next((_finite(extra[k]) for k in keys if k in extra), None)
                    if value is not None:
                        series[name].append([step, value])
                for key, value in extra.items():
                    if key.startswith("val/"):
                        add_val(step, key, value)
        for row in connection.execute("SELECT step, metrics_json FROM training_eval_points WHERE run_id = ? ORDER BY step", (run_id,)):
            for key, value in (json.loads(row["metrics_json"]) or {}).items():
                add_val(row["step"], key, value)
    sets = {name: {kind: _stride([[s, v] for s, v in sorted(points.items())], max_points) for kind, points in kinds.items()}
            for name, kinds in sorted(val.items())}
    latest = {name: {kind: (pts[-1][0], pts[-1][1]) for kind, pts in kinds.items() if pts} for name, kinds in sets.items()}
    return {
        "window": {"size": window_steps, "unit": "steps"},
        "loss": _stride(raw, max_points),
        "loss_rolling": _stride(rolling_mean(spans, window_steps), max_points),
        "validation": {"sets": sets, "latest": latest},
        **{name: _stride(points, max_points) for name, points in series.items()},
        "points": len(raw),
    }
