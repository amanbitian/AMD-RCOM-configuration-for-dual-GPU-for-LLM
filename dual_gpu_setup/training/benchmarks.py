"""Controlled benchmark sweeps (batch size, context length, LoRA config, gradient checkpointing,
precision) that RECOMMEND a configuration and never rewrite training settings on their own.

Each trial launches the training project's own command with one parameter combination. That command
reports through the normal TrainingTracker (the runner passes DGPU_* environment variables), so a
trial is an ordinary run with the same metrics, and the report is derived from those runs.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .db import Database
from .store import TrainingStore

SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    state TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    report_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS benchmark_trials (
    benchmark_id TEXT NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    run_id TEXT NOT NULL,
    state TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    PRIMARY KEY (benchmark_id, idx)
);
"""
KINDS = {"batch_size", "context", "lora", "gradient_checkpointing", "precision", "custom"}
OOM_MARKERS = ("out of memory", "hip out of memory", "cuda out of memory", "hiperroroutofmemory", "alloc_failed")

Launcher = Callable[[list[str], dict[str, str], float], "subprocess.CompletedProcess[str]"]


def _default_launcher(command: list[str], env: dict[str, str], timeout: float) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(command, env={**os.environ, **env}, capture_output=True, text=True, timeout=timeout)


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


class BenchmarkRunner(Database):
    def __init__(self, path: Path, store: TrainingStore, server_url: str,
                 launcher: Launcher = _default_launcher,
                 on_trial_end: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.time):
        super().__init__(path)
        self.store = store
        self.server_url = server_url
        self.launcher = launcher
        self.on_trial_end = on_trial_end
        self.clock = clock
        self.ensure_schema(SCHEMA)
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}

    # ----------------------------------------------------------------- plans

    @staticmethod
    def validate(plan: dict[str, Any]) -> dict[str, Any]:
        kind = str(plan.get("kind") or "custom")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        command = plan.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(p, str) for p in command):
            raise ValueError("command must be a non-empty list of strings (placeholders like {batch_size} are filled per trial).")
        grid = plan.get("grid")
        if not isinstance(grid, dict) or not grid or not all(isinstance(v, list) and v for v in grid.values()):
            raise ValueError("grid must map each parameter to a non-empty list of values.")
        trials = expand_grid(grid)
        if len(trials) > 64:
            raise ValueError(f"{len(trials)} trials is too many (limit 64); narrow the grid.")
        return {"name": str(plan.get("name") or f"{kind}-benchmark")[:100], "kind": kind, "command": command, "grid": grid,
                "steps": int(plan.get("steps") or 60), "lanes": [str(x) for x in plan.get("lanes") or []],
                "timeout_s": float(plan.get("timeout_s") or 1800), "cwd": plan.get("cwd"),
                "vram_budget_bytes": plan.get("vram_budget_bytes"), "headroom_fraction": float(plan.get("headroom_fraction", 0.05)),
                "trials": trials}

    def create(self, plan: dict[str, Any]) -> str:
        clean = self.validate(plan)
        benchmark_id = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO benchmarks (id, name, kind, state, plan_json, created_at) VALUES (?,?,?,?,?,?)",
                (benchmark_id, clean["name"], clean["kind"], "queued", json.dumps(clean), self.clock()))
            for index, params in enumerate(clean["trials"]):
                connection.execute(
                    "INSERT INTO benchmark_trials (benchmark_id, idx, params_json, run_id, state) VALUES (?,?,?,?,?)",
                    (benchmark_id, index, json.dumps(params), uuid.uuid4().hex, "pending"))
        return benchmark_id

    def start(self, benchmark_id: str, background: bool = True) -> None:
        self._cancel[benchmark_id] = threading.Event()
        if not background:
            self._run(benchmark_id)
            return
        thread = threading.Thread(target=self._run, args=(benchmark_id,), name=f"benchmark-{benchmark_id[:6]}", daemon=True)
        self._threads[benchmark_id] = thread
        thread.start()

    def cancel(self, benchmark_id: str) -> None:
        if benchmark_id in self._cancel:
            self._cancel[benchmark_id].set()

    def join(self, benchmark_id: str, timeout: float | None = None) -> None:
        thread = self._threads.get(benchmark_id)
        if thread:
            thread.join(timeout)

    # ------------------------------------------------------------- execution

    def _run(self, benchmark_id: str) -> None:
        with self.connect() as connection:
            row = connection.execute("SELECT plan_json FROM benchmarks WHERE id = ?", (benchmark_id,)).fetchone()
            connection.execute("UPDATE benchmarks SET state = 'running' WHERE id = ?", (benchmark_id,))
            trials = connection.execute("SELECT * FROM benchmark_trials WHERE benchmark_id = ? ORDER BY idx", (benchmark_id,)).fetchall()
        plan = json.loads(row["plan_json"])
        cancel = self._cancel.get(benchmark_id) or threading.Event()
        try:
            for trial in trials:
                if cancel.is_set():
                    self._set_trial(benchmark_id, trial["idx"], "cancelled")
                    continue
                self._run_trial(benchmark_id, plan, trial["idx"], json.loads(trial["params_json"]), trial["run_id"])
            report = self.report(benchmark_id)
            with self.connect() as connection:
                connection.execute("UPDATE benchmarks SET state = ?, report_json = ?, finished_at = ? WHERE id = ?",
                                   ("cancelled" if cancel.is_set() else "completed", json.dumps(report), self.clock(), benchmark_id))
        except Exception as exc:  # noqa: BLE001
            with self.connect() as connection:
                connection.execute("UPDATE benchmarks SET state = 'failed', error = ?, finished_at = ? WHERE id = ?",
                                   (f"{type(exc).__name__}: {exc}", self.clock(), benchmark_id))

    def _set_trial(self, benchmark_id: str, idx: int, state: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE benchmark_trials SET state = ?, result_json = ?, error = ? WHERE benchmark_id = ? AND idx = ?",
                               (state, json.dumps(result) if result is not None else None, error, benchmark_id, idx))

    def _run_trial(self, benchmark_id: str, plan: dict[str, Any], idx: int, params: dict[str, Any], run_id: str) -> None:
        context = {k: str(v) for k, v in params.items()}
        context.update(steps=str(plan["steps"]), run_id=run_id)
        command = [part.format_map(context) for part in plan["command"]]
        env = {"DGPU_TRACKER_SERVER": self.server_url, "DGPU_RUN_ID": run_id,
               "DGPU_RUN_NAME": f"{plan['name']}#{idx}", "DGPU_BENCH_ID": benchmark_id, "DGPU_BENCH_STEPS": str(plan["steps"]),
               "DGPU_PARAMS": json.dumps(params), "DGPU_LANES": ",".join(plan["lanes"]),
               **{f"DGPU_PARAM_{k.upper()}": str(v) for k, v in params.items()}}
        self._set_trial(benchmark_id, idx, "running")
        outcome, detail, started = "completed", "", time.perf_counter()
        try:
            result = self.launcher(command, env, plan["timeout_s"])
            text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
            if result.returncode != 0:
                outcome = "oom" if any(m in text for m in OOM_MARKERS) else "failed"
                detail = (result.stderr or result.stdout or "")[-400:]
        except subprocess.TimeoutExpired:
            outcome, detail = "timeout", f"exceeded {plan['timeout_s']}s"
        wall_s = time.perf_counter() - started
        if self.on_trial_end:
            self.on_trial_end(run_id)
        measured: dict[str, Any] = {}
        try:
            run = self.store.get_run(run_id)
            measured = {"tokens_per_sec": run["throughput"]["tokens_per_sec"],
                        "supervised_tokens_per_sec": run["throughput"]["supervised_tokens_per_sec"],
                        "peak_mem_bytes": run["peak_mem_bytes"], "step_time_ms": run["step_time_ema_ms"],
                        "last_step": run["last_step"], "nan_count": run["nan_count"], "run_status": run["status"]}
            if outcome == "completed" and run["nan_count"]:
                outcome, detail = "unstable", f"{run['nan_count']} NaN/Inf events"
            if outcome == "completed" and run["throughput"]["tokens_per_sec"] is None:
                outcome, detail = "no_telemetry", "the command finished but reported no steady-state windows"
            events = {e["kind"] for e in self.store.events(run_id)}
            if outcome == "failed" and "oom" in events:
                outcome = "oom"
        except Exception:  # noqa: BLE001 - run never registered
            if outcome == "completed":
                outcome, detail = "no_telemetry", "no TrainingTracker run was created by the command"
        self._set_trial(benchmark_id, idx, outcome, {**measured, "wall_s": wall_s, "params": params}, detail or None)

    # --------------------------------------------------------------- reports

    def get(self, benchmark_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM benchmarks WHERE id = ?", (benchmark_id,)).fetchone()
            if row is None:
                raise KeyError(benchmark_id)
            trials = connection.execute("SELECT * FROM benchmark_trials WHERE benchmark_id = ? ORDER BY idx", (benchmark_id,)).fetchall()
        plan = json.loads(row["plan_json"])
        return {"id": row["id"], "name": row["name"], "kind": row["kind"], "state": row["state"], "error": row["error"],
                "created_at": row["created_at"], "finished_at": row["finished_at"], "steps": plan["steps"],
                "trials": [{"idx": t["idx"], "params": json.loads(t["params_json"]), "run_id": t["run_id"], "state": t["state"],
                            "result": json.loads(t["result_json"]) if t["result_json"] else None, "error": t["error"]} for t in trials],
                "report": json.loads(row["report_json"]) if row["report_json"] else None}

    def list(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [{"id": r["id"], "name": r["name"], "kind": r["kind"], "state": r["state"], "created_at": r["created_at"]}
                    for r in connection.execute("SELECT id, name, kind, state, created_at FROM benchmarks ORDER BY created_at DESC")]

    def report(self, benchmark_id: str) -> dict[str, Any]:
        data = self.get(benchmark_id)
        with self.connect() as connection:
            plan = json.loads(connection.execute("SELECT plan_json FROM benchmarks WHERE id = ?", (benchmark_id,)).fetchone()[0])
        return build_report(data["trials"], plan)


def build_report(trials: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    """Recommendation only: the best stable configuration under the VRAM budget, plus the boundary
    where memory ran out and, for gradient checkpointing, the memory-vs-time trade."""
    ok = [t for t in trials if t["state"] == "completed" and t["result"] and t["result"]["tokens_per_sec"]]
    budget = plan.get("vram_budget_bytes")
    headroom = plan.get("headroom_fraction", 0.05)
    limit = None if not budget else budget * (1 - headroom)
    fitting = [t for t in ok if limit is None or (t["result"]["peak_mem_bytes"] or 0) <= limit]
    ranked = sorted(fitting, key=lambda t: t["result"]["tokens_per_sec"], reverse=True)
    report: dict[str, Any] = {
        "kind": plan["kind"], "trials": len(trials),
        "outcomes": {s: sum(1 for t in trials if t["state"] == s) for s in sorted({t["state"] for t in trials})},
        "recommended": None, "ranking": [{"params": t["params"], **{k: t["result"][k] for k in ("tokens_per_sec", "peak_mem_bytes", "step_time_ms")}} for t in ranked],
        "headroom_fraction": headroom, "vram_budget_bytes": budget,
        "note": "Recommendation only; training settings are never changed automatically.",
    }
    if ranked:
        best = ranked[0]
        report["recommended"] = {"params": best["params"], "tokens_per_sec": best["result"]["tokens_per_sec"],
                                 "peak_mem_bytes": best["result"]["peak_mem_bytes"], "run_id": best["run_id"]}
    ooms = [t for t in trials if t["state"] == "oom"]
    if ooms:
        report["oom_boundary"] = [t["params"] for t in ooms]
    if plan["kind"] == "gradient_checkpointing":
        by_flag = {str(t["params"].get("gradient_checkpointing")).lower(): t for t in ok}
        on, off = by_flag.get("true"), by_flag.get("false")
        if on and off and off["result"]["peak_mem_bytes"] and off["result"]["tokens_per_sec"]:
            saved = off["result"]["peak_mem_bytes"] - (on["result"]["peak_mem_bytes"] or 0)
            report["gradient_checkpointing"] = {
                "memory_saved_bytes": saved, "memory_saved_pct": saved / off["result"]["peak_mem_bytes"] * 100.0,
                "time_added_pct": (off["result"]["tokens_per_sec"] / on["result"]["tokens_per_sec"] - 1) * 100.0,
            }
    return report
