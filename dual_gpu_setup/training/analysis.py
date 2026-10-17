"""Derived analyses over stored telemetry. Pure functions of a TrainingStore: nothing here writes.

Every result keeps the raw numbers next to the verdict, and returns None (never 0) for anything the
data cannot support, so the dashboard can show "n/a" honestly.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import statistics
from typing import Any, Iterable

from .store import TrainingStore

EXCLUDED_CONFIG_KEYS = {"seed", "data_seed", "run_name", "name", "output_dir", "logging_dir", "tags", "id"}


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _window_tps(window: dict[str, Any], key: str = "nonpad_tokens") -> float | None:
    if window.get(key) is None or not window.get("step_time_ms"):
        return None
    seconds = window["step_time_ms"] * (window["window_steps"] or 1) / 1000.0
    return window[key] / seconds if seconds > 0 else None


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


def config_hash(config: dict[str, Any] | None) -> str:
    flat = {k: v for k, v in _flatten(config or {}).items() if k.split(".")[-1] not in EXCLUDED_CONFIG_KEYS}
    return hashlib.sha256(json.dumps(flat, sort_keys=True, default=str).encode()).hexdigest()[:12]


def config_diff(configs: dict[str, dict[str, Any] | None]) -> dict[str, dict[str, Any]]:
    """Keys whose value differs between runs (seeds and names excluded as noise)."""
    flats = {rid: _flatten(cfg or {}) for rid, cfg in configs.items()}
    keys = sorted({k for flat in flats.values() for k in flat if k.split(".")[-1] not in EXCLUDED_CONFIG_KEYS})
    diff = {}
    for key in keys:
        values = {rid: flat.get(key) for rid, flat in flats.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) > 1:
            diff[key] = values
    return diff


# --------------------------------------------------------------- comparison


def _smoothed(points: list[tuple[float, float]], alpha: float = 0.3) -> list[tuple[float, float]]:
    out, ema = [], None
    for x, y in points:
        ema = y if ema is None else alpha * y + (1 - alpha) * ema
        out.append((x, ema))
    return out


def _interpolate(points: list[tuple[float, float]], x: float) -> float | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    if x > xs[-1]:
        return None
    index = bisect.bisect_left(xs, x)
    if index == 0:
        return points[0][1]
    (x0, y0), (x1, y1) = points[index - 1], points[min(index, len(points) - 1)]
    return y1 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _cumulative(windows: list[dict[str, Any]]) -> tuple[list[tuple[float, float]], float]:
    total, points = 0.0, []
    for w in windows:
        total += w["nonpad_tokens"] or 0
        if w["loss"] is not None:
            points.append((total, w["loss"]))
    return points, total


def compare_runs(store: TrainingStore, run_ids: list[str], token_budget: float | None = None,
                 min_seeds_for_noise: int = 3) -> dict[str, Any]:
    """Compare runs at an equal training-token budget (default: the largest budget every run
    reached). Comparing at equal steps or wall time would favour whichever run is cheaper per step."""
    runs: dict[str, dict[str, Any]] = {}
    curves: dict[str, list[tuple[float, float]]] = {}
    totals: dict[str, float] = {}
    for run_id in run_ids:
        run = store.get_run(run_id)
        runs[run_id] = run
        points, total = _cumulative(store.step_windows(run_id, phase=None))
        curves[run_id] = _smoothed(points)
        totals[run_id] = total
    budget = token_budget if token_budget else (min(totals.values()) if totals and min(totals.values()) > 0 else None)
    rows = []
    for run_id, run in runs.items():
        checkpoints = [c for c in store.checkpoints(run_id) if c["valid"] and c["tokens_seen"] is not None
                       and budget is not None and c["tokens_seen"] <= budget]
        latest_checkpoint = max(checkpoints, key=lambda c: c["tokens_seen"]) if checkpoints else None
        rows.append({
            "run_id": run_id, "name": run["name"], "status": run["status"], "tokens_total": totals[run_id],
            "reached_budget": budget is not None and totals[run_id] >= budget,
            "loss_at_budget": None if budget is None else _interpolate(curves[run_id], budget),
            "tokens_per_sec": run["throughput"]["tokens_per_sec"],
            "supervised_tokens_per_sec": run["throughput"]["supervised_tokens_per_sec"],
            "peak_mem_bytes": run["peak_mem_bytes"], "seed": run["seed"],
            "config_hash": config_hash(run.get("config")),
            "checkpoint_step": latest_checkpoint["step"] if latest_checkpoint else None,
            "evals": latest_checkpoint["evals"] if latest_checkpoint else {},
        })
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["config_hash"], []).append(row)
    seed_stats = []
    for digest, members in groups.items():
        losses = [m["loss_at_budget"] for m in members if m["loss_at_budget"] is not None]
        seed_stats.append({
            "config_hash": digest, "runs": [m["run_id"] for m in members], "n": len(losses),
            "mean_loss": _mean(losses), "std_loss": statistics.stdev(losses) if len(losses) > 1 else None,
            "trustworthy": len(losses) >= min_seeds_for_noise,
        })
    verdicts = []
    for i, a in enumerate(seed_stats):
        for b in seed_stats[i + 1:]:
            if a["mean_loss"] is None or b["mean_loss"] is None:
                continue
            delta = a["mean_loss"] - b["mean_loss"]
            noise = max(s for s in (a["std_loss"], b["std_loss"], 0.0) if s is not None)
            if not (a["trustworthy"] and b["trustworthy"]):
                label = "insufficient seeds"
            else:
                label = "within noise" if abs(delta) <= 2 * noise else ("A lower" if delta < 0 else "B lower")
            verdicts.append({"a": a["config_hash"], "b": b["config_hash"], "delta_loss": delta,
                             "noise": noise or None, "verdict": label})
    return {
        "budget_tokens": budget, "runs": rows, "seed_groups": seed_stats, "verdicts": verdicts,
        "config_diff": config_diff({rid: r.get("config") for rid, r in runs.items()}),
    }


# ------------------------------------------------------- hardware correlations


def _train_buses(store: TrainingStore, run: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    lanes = set(run.get("lanes") or [])
    buses = {r["pci_bus"] for r in rows}
    train = {r["pci_bus"] for r in rows if r.get("lane") in lanes} if lanes else set()
    return (train or buses), buses - (train or buses)


def thermal_analysis(store: TrainingStore, run_id: str, bucket_s: float = 60.0) -> dict[str, Any]:
    """Temperature, clocks, power and throughput over time per training GPU, with a degradation
    verdict (early vs late throughput) and suspected thermal-throttling buckets."""
    run = store.get_run(run_id)
    rows = store.hardware_rows(run_id, columns=("ts", "pci_bus", "lane", "gpu_busy_pct", "temp_junction_c", "temp_memory_c",
                                                "sclk_mhz", "power_w"))
    train_buses, _ = _train_buses(store, run, rows)
    windows = store.step_windows(run_id)
    start = run["started_at"]
    result: dict[str, Any] = {"buckets": [], "gpus": {}, "throughput_degradation_pct": None, "throttle_suspected": False}
    if not rows:
        return result
    tps_by_bucket: dict[int, list[float]] = {}
    for w in windows:
        tps = _window_tps(w)
        if tps is not None:
            tps_by_bucket.setdefault(int((w["ts"] - start) // bucket_s), []).append(tps)
    memory_alert, junction_alert = store.memory_temp_alert_c, store.junction_temp_alert_c
    for bus in sorted(train_buses):
        by_bucket: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            if r["pci_bus"] == bus:
                by_bucket.setdefault(int((r["ts"] - start) // bucket_s), []).append(r)
        buckets = []
        for index in sorted(by_bucket):
            samples = by_bucket[index]
            buckets.append({
                "t": index * bucket_s, "tokens_per_sec": _mean(tps_by_bucket.get(index, [])),
                "gpu_busy_pct": _mean(s["gpu_busy_pct"] for s in samples),
                "temp_junction_c": _mean(s["temp_junction_c"] for s in samples),
                "temp_memory_c": _mean(s["temp_memory_c"] for s in samples),
                "sclk_mhz": _mean(s["sclk_mhz"] for s in samples), "power_w": _mean(s["power_w"] for s in samples),
            })
        quarter = max(1, len(buckets) // 4)
        baseline_clock = _median(b["sclk_mhz"] for b in buckets[:quarter])
        suspected = []
        for b in buckets:
            hot = (b["temp_junction_c"] is not None and b["temp_junction_c"] >= junction_alert - 10) or \
                  (b["temp_memory_c"] is not None and b["temp_memory_c"] >= memory_alert - 5)
            slow = baseline_clock and b["sclk_mhz"] is not None and b["sclk_mhz"] < 0.9 * baseline_clock
            b["throttle_suspected"] = bool(hot and slow)
            if b["throttle_suspected"]:
                suspected.append(b["t"])
        result["gpus"][bus] = {"lane": next((r["lane"] for r in rows if r["pci_bus"] == bus), None),
                               "buckets": buckets, "throttle_buckets": suspected}
        result["throttle_suspected"] = result["throttle_suspected"] or bool(suspected)
    ordered = [(i, statistics.fmean(v)) for i, v in sorted(tps_by_bucket.items())]
    if len(ordered) >= 4:
        quarter = max(1, len(ordered) // 4)
        early = statistics.fmean(v for _, v in ordered[:quarter])
        late = statistics.fmean(v for _, v in ordered[-quarter:])
        if early > 0:
            result["throughput_degradation_pct"] = (early - late) / early * 100.0
    return result


def contention_analysis(store: TrainingStore, run_id: str, busy_threshold: float = 30.0, cpu_threshold: float = 85.0) -> dict[str, Any]:
    """Training throughput while another GPU (or the host) is busy versus quiet. A real gap points
    at shared PCIe lanes, host RAM/CPU, disk, or chassis cooling rather than the GPU itself."""
    run = store.get_run(run_id)
    rows = store.hardware_rows(run_id, columns=("ts", "pci_bus", "lane", "gpu_busy_pct"))
    _, other = _train_buses(store, run, rows)
    system = store.system_rows(run_id)
    windows = store.step_windows(run_id)
    other_samples = sorted((r["ts"], r["gpu_busy_pct"]) for r in rows if r["pci_bus"] in other and r["gpu_busy_pct"] is not None)
    other_ts = [t for t, _ in other_samples]
    system_ts = [s["ts"] for s in system]

    def mean_between(timestamps: list[float], values: list[float | None], lo: float, hi: float) -> float | None:
        i, j = bisect.bisect_left(timestamps, lo), bisect.bisect_right(timestamps, hi)
        return _mean(values[i:j])

    other_values = [v for _, v in other_samples]
    cpu_values = [s["cpu_pct"] for s in system]
    groups: dict[str, list[float]] = {"contended": [], "quiet": [], "host_loaded": [], "host_quiet": []}
    for w in windows:
        tps = _window_tps(w)
        if tps is None:
            continue
        span = (w["step_time_ms"] or 0) * (w["window_steps"] or 1) / 1000.0 + (w["excluded_ms"] or 0) / 1000.0
        lo, hi = w["ts"] - span, w["ts"]
        busy = mean_between(other_ts, other_values, lo, hi)
        if busy is not None:
            groups["contended" if busy >= busy_threshold else "quiet"].append(tps)
        cpu = mean_between(system_ts, cpu_values, lo, hi)
        if cpu is not None:
            groups["host_loaded" if cpu >= cpu_threshold else "host_quiet"].append(tps)

    def slowdown(busy_key: str, quiet_key: str) -> dict[str, Any]:
        busy, quiet = _mean(groups[busy_key]), _mean(groups[quiet_key])
        return {"busy_windows": len(groups[busy_key]), "quiet_windows": len(groups[quiet_key]),
                "busy_tokens_per_sec": busy, "quiet_tokens_per_sec": quiet,
                "slowdown_pct": None if not busy or not quiet else (quiet - busy) / quiet * 100.0}

    return {"other_gpu": slowdown("contended", "quiet"), "host_cpu": slowdown("host_loaded", "host_quiet"),
            "thresholds": {"other_gpu_busy_pct": busy_threshold, "host_cpu_pct": cpu_threshold}}


def energy_analysis(store: TrainingStore, run_id: str, max_gap_s: float = 5.0) -> dict[str, Any]:
    """Board energy integrated over the run's training GPUs (trapezoid, gaps capped), expressed per
    million tokens. Board power only: wall power also includes CPU, fans, and the PSU."""
    run = store.get_run(run_id)
    rows = store.hardware_rows(run_id, columns=("ts", "pci_bus", "lane", "power_w"))
    train_buses, _ = _train_buses(store, run, rows)
    joules, covered = 0.0, False
    per_gpu = {}
    for bus in sorted(train_buses):
        samples = [(r["ts"], r["power_w"]) for r in rows if r["pci_bus"] == bus and r["power_w"] is not None]
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            energy += (p0 + p1) / 2.0 * min(t1 - t0, max_gap_s)
        if samples:
            covered = True
        per_gpu[bus] = {"joules": energy if samples else None, "wh": energy / 3600.0 if samples else None}
        joules += energy
    tokens = run["tokens"]
    result: dict[str, Any] = {"per_gpu": per_gpu, "joules": joules if covered else None,
                              "wh": joules / 3600.0 if covered else None,
                              "joules_per_million_input_tokens": None, "joules_per_million_supervised_tokens": None,
                              "wh_per_million_input_tokens": None}
    if covered and tokens["nonpad"]:
        result["joules_per_million_input_tokens"] = joules / tokens["nonpad"] * 1e6
        result["wh_per_million_input_tokens"] = joules / 3600.0 / tokens["nonpad"] * 1e6
    if covered and tokens["supervised"]:
        result["joules_per_million_supervised_tokens"] = joules / tokens["supervised"] * 1e6
    return result


def mfu_estimate(run: dict[str, Any], peak_tflops: float | None) -> dict[str, Any]:
    """Model FLOPs utilisation, always an ESTIMATE. The 6*N*tokens/s rule counts a dense model's
    forward+backward; with LoRA the frozen base skips weight gradients, so both bases are shown."""
    config = run.get("config") or {}
    trainable, total = config.get("trainable_params"), config.get("total_params")
    tps = run["throughput"]["tokens_per_sec"]
    out = {"estimate": True, "peak_tflops": peak_tflops, "mfu_trainable_basis": None, "mfu_frozen_base_basis": None}
    if not (tps and peak_tflops):
        return out
    peak = peak_tflops * 1e12
    if trainable:
        out["mfu_trainable_basis"] = 6.0 * trainable * tps / peak
    if total:
        flops_per_token = 4.0 * total if trainable and trainable < total else 6.0 * total
        out["mfu_frozen_base_basis"] = flops_per_token * tps / peak
    return out


# ------------------------------------------------------------------- DDP


def straggler_analysis(store: TrainingStore, run_id: str, vram_totals: dict[str, int] | None = None) -> dict[str, Any]:
    """Per-rank step time, wait on the slowest rank, communication share, and VRAM headroom.
    `vram_totals` (pci_bus -> bytes) backs up the run's own samples when a card was not sampled."""
    rows = store.rank_steps(run_id)
    summary = store.hardware_summary(run_id)["gpus"]
    by_step: dict[int, dict[int, dict[str, Any]]] = {}
    for r in rows:
        by_step.setdefault(r["step"], {})[r["rank"]] = r
    ranks: dict[int, dict[str, Any]] = {}
    slowest_counts: dict[int, int] = {}
    complete = [(step, group) for step, group in by_step.items() if len(group) > 1 and all(g["step_time_ms"] for g in group.values())]
    for _, group in complete:
        slowest = max(group, key=lambda rk: group[rk]["step_time_ms"])
        slowest_counts[slowest] = slowest_counts.get(slowest, 0) + 1
        worst = group[slowest]["step_time_ms"]
        for rank, r in group.items():
            ranks.setdefault(rank, {"wait": [], "step": [], "comm": [], "data": []})
            ranks[rank]["wait"].append(worst - r["step_time_ms"])
    for r in rows:
        bucket = ranks.setdefault(r["rank"], {"wait": [], "step": [], "comm": [], "data": []})
        for key, column in (("step", "step_time_ms"), ("comm", "comm_ms"), ("data", "data_wait_ms")):
            if r[column] is not None:
                bucket[key].append(r[column])
    peaks: dict[int, int] = {}
    buses: dict[int, str | None] = {}
    for r in rows:
        if r["mem_peak_bytes"] is not None:
            peaks[r["rank"]] = max(peaks.get(r["rank"], 0), r["mem_peak_bytes"])
        buses[r["rank"]] = r["pci_bus"] or buses.get(r["rank"])
    out = []
    for rank in sorted(ranks):
        data = ranks[rank]
        step_ms, comm_ms = _mean(data["step"]), _mean(data["comm"])
        bus = buses.get(rank) or ""
        capacity = (summary.get(bus) or {}).get("vram_total_bytes") or (vram_totals or {}).get(bus)
        peak = peaks.get(rank)
        out.append({
            "rank": rank, "pci_bus": buses.get(rank), "mean_step_ms": step_ms,
            "mean_wait_ms": _mean(data["wait"]), "idle_pct": None if not step_ms or not data["wait"] else _mean(data["wait"]) / (step_ms + _mean(data["wait"])) * 100.0,
            "mean_comm_ms": comm_ms, "comm_share": None if not step_ms or comm_ms is None else comm_ms / step_ms,
            "mean_data_wait_ms": _mean(data["data"]), "peak_mem_bytes": peak, "vram_total_bytes": capacity,
            "vram_headroom_bytes": None if peak is None or not capacity else capacity - peak,
            "slowest_windows": slowest_counts.get(rank, 0),
        })
    return {"ranks": out, "windows_compared": len(complete)}


def scaling_efficiency(store: TrainingStore, ddp_run_id: str, baseline_run_ids: list[str]) -> dict[str, Any]:
    """T_combined / (T_A_alone + T_B_alone) with matched single-GPU baselines. 1.0 is perfect linear
    scaling; the ratio uses steady-state tokens/sec so warm-up cannot distort it."""
    ddp = store.get_run(ddp_run_id)["throughput"]["tokens_per_sec"]
    baselines = {rid: store.get_run(rid)["throughput"]["tokens_per_sec"] for rid in baseline_run_ids}
    total = sum(v for v in baselines.values() if v)
    return {
        "ddp_tokens_per_sec": ddp, "baseline_tokens_per_sec": baselines,
        "scaling_efficiency": None if not ddp or not total or None in baselines.values() else ddp / total,
        "speedup_vs_fastest_single": None if not ddp or not total else ddp / max(v for v in baselines.values() if v),
    }


# ------------------------------------------------------------------ resume


def resume_report(store: TrainingStore, run_id: str) -> dict[str, Any]:
    """Did a resumed run continue the parent faithfully? Checks step, LR schedule, loss, token
    counters, and whether the checkpoint carried optimizer/scheduler/RNG state."""
    run = store.get_run(run_id)
    report: dict[str, Any] = {"run_id": run_id, "parent_run_id": run["parent_run_id"], "checks": [], "verdict": "n/a"}
    if not run["parent_run_id"]:
        return report
    parent_windows = store.step_windows(run["parent_run_id"], phase=None)
    child_windows = store.step_windows(run_id, phase=None)
    event = next((e for e in reversed(store.events(run_id)) if e["kind"] == "resumed"), None)
    payload = (event or {}).get("payload") or {}
    from_step = payload.get("from_step")
    checks = report["checks"]

    def add(name: str, ok: bool | None, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add("manifest_valid", payload.get("manifest_valid"), payload.get("reason") or "checkpoint manifest validated")
    if "resume_state_complete" in payload:
        add("optimizer_scheduler_rng_restored", payload["resume_state_complete"],
            "present" if payload["resume_state_complete"] else f"missing: {', '.join(payload.get('resume_state_missing') or [])}")
    if parent_windows and child_windows:
        base = [w for w in parent_windows if from_step is None or w["step"] <= from_step] or parent_windows
        last_parent, first_child = base[-1], child_windows[0]
        gap = first_child["step"] - last_parent["step"]
        window = max(first_child["window_steps"] or 1, 1)
        add("step_continuity", 0 <= gap <= 2 * window,
            f"parent last step {last_parent['step']}, child first window ends at {first_child['step']}")
        losses = [w["loss"] for w in base[-10:] if w["loss"] is not None]
        child_loss = next((w["loss"] for w in child_windows if w["loss"] is not None), None)
        if len(losses) >= 3 and child_loss is not None:
            spread = statistics.pstdev(losses) or max(abs(statistics.fmean(losses)) * 0.02, 1e-6)
            z = abs(child_loss - statistics.fmean(losses[-3:])) / spread
            add("loss_continuity", z <= 4.0, f"first resumed loss {child_loss:.4f} vs parent tail {statistics.fmean(losses[-3:]):.4f} ({z:.1f} sigma)")
        tokens_parent = sum(w["nonpad_tokens"] or 0 for w in base)
        tokens_child = sum(w["nonpad_tokens"] or 0 for w in child_windows[:1])
        if tokens_parent and tokens_child:
            seen = payload.get("tokens_seen")
            add("token_counter_continuity", seen is None or abs(seen - tokens_parent) <= 0.05 * tokens_parent,
                f"checkpoint tokens_seen={seen} vs parent windows total={tokens_parent}")
    decided = [c["ok"] for c in checks if c["ok"] is not None]
    report["verdict"] = "n/a" if not decided else ("ok" if all(decided) else "warning" if sum(not d for d in decided) == 1 else "fail")
    return report


# -------------------------------------------------------------------- alerts


def run_alerts(store: TrainingStore, run_id: str, eval_queue: dict[str, Any] | None = None,
               idle_busy_pct: float = 30.0) -> list[dict[str, Any]]:
    """Active alerts for a run: NaN/Inf, loss spike, throughput drop, idle GPU, hot GPU, stalled or
    lost run, GPU reset, lost lease, and a saturated evaluation queue."""
    run = store.get_run(run_id)
    alerts: list[dict[str, Any]] = []

    def add(severity: str, kind: str, message: str, step: int | None = None) -> None:
        alerts.append({"run_id": run_id, "severity": severity, "kind": kind, "message": message, "step": step})

    if run["status"] in {"stalled", "lost"}:
        add("error", run["status"], run.get("stall_reason") or f"Run is {run['status']}: no heartbeat or step progress.")
    if run["status"] == "failed":
        add("error", "failed", (run.get("error") or "Run failed.").strip().splitlines()[-1][:200])
    if run["nan_count"]:
        add("error", "nan_inf", f"{run['nan_count']} NaN/Inf loss or gradient event(s).")
    windows = store.step_windows(run_id, tail=400)  # alerts run on every poll: bounded, recent history only
    losses = [w["loss"] for w in windows if w["loss"] is not None]
    if len(losses) >= 12:
        history, latest = losses[-12:-1], losses[-1]
        median = statistics.median(history)
        mad = statistics.median(abs(x - median) for x in history) or max(abs(median) * 0.02, 1e-6)
        if latest > median + 6 * 1.4826 * mad and latest > 1.5 * median:
            add("warning", "loss_spike", f"Loss {latest:.3f} is far above the recent median {median:.3f}.")
    rates = [(_window_tps(w), w) for w in windows]
    rates = [(r, w) for r, w in rates if r is not None]
    if len(rates) >= 10:
        baseline = statistics.median(r for r, _ in rates[:-3])
        recent = statistics.fmean(r for r, _ in rates[-3:])
        if baseline > 0 and recent < 0.7 * baseline:
            add("warning", "throughput_drop", f"Tokens/s fell to {recent:.0f} from a baseline of {baseline:.0f}.")
    rows = []
    if run["status"] == "running":
        horizon = store.clock() - 120  # only the last couple of minutes are needed for idle/heat checks
        rows = store.hardware_rows(run_id, since=horizon,
                                   columns=("ts", "pci_bus", "lane", "gpu_busy_pct", "temp_memory_c", "temp_junction_c"))
    if rows:
        train_buses, _ = _train_buses(store, run, rows)
        recent_rows = [r for r in rows if r["pci_bus"] in train_buses and r["ts"] >= rows[-1]["ts"] - 60]
        busy = _mean(r["gpu_busy_pct"] for r in recent_rows)
        if busy is not None and busy < idle_busy_pct and run["status"] == "running" and len(recent_rows) >= 20:
            add("warning", "gpu_idle", f"Training GPU averaged {busy:.0f}% busy over the last minute (likely input-bound).")
        latest_by_bus = {}
        for r in rows:
            latest_by_bus[r["pci_bus"]] = r
        for bus, r in latest_by_bus.items():
            if r["temp_memory_c"] is not None and r["temp_memory_c"] >= store.memory_temp_alert_c:
                add("error", "temperature", f"{r.get('lane') or bus} memory temperature {r['temp_memory_c']:.0f} °C.")
            if r["temp_junction_c"] is not None and r["temp_junction_c"] >= store.junction_temp_alert_c:
                add("error", "temperature", f"{r.get('lane') or bus} junction temperature {r['temp_junction_c']:.0f} °C.")
    for event in store.events(run_id, limit=100):
        if event["kind"] in {"gpu_reset", "ring_timeout", "page_fault", "gpu_memory_error", "pcie_error"}:
            add("error", "gpu_fault", f"{event['kind']} recorded by the kernel log.", event["step"])
        elif event["kind"] == "lease_lost":
            add("warning", "lease_lost", "GPU lease expired and was taken by another job.")
        elif event["kind"] == "oom":
            add("error", "oom", "Out-of-memory recorded.", event["step"])
    if eval_queue and eval_queue.get("capacity") and eval_queue.get("depth", 0) >= 0.8 * eval_queue["capacity"]:
        add("warning", "eval_queue_saturated", f"Evaluation queue at {eval_queue['depth']}/{eval_queue['capacity']}.")
    seen, unique = set(), []
    for a in alerts:
        key = (a["kind"], a["message"])
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def analyse_run(store: TrainingStore, run_id: str, peak_tflops: float | None = None,
                vram_totals: dict[str, int] | None = None) -> dict[str, Any]:
    run = store.get_run(run_id)
    result = {
        "energy": energy_analysis(store, run_id), "thermal": thermal_analysis(store, run_id),
        "contention": contention_analysis(store, run_id), "mfu": mfu_estimate(run, peak_tflops),
    }
    if run["world_size"] > 1:
        result["ranks"] = straggler_analysis(store, run_id, vram_totals)
    if run["parent_run_id"]:
        result["resume"] = resume_report(store, run_id)
    return result
