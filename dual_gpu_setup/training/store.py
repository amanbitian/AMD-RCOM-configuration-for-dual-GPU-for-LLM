"""SQLite persistence for training runs, window metrics, events, checkpoints, evaluations, and GPU
samples.

Conventions follow docs/TELEMETRY_AND_STORAGE.md: raw counters are stored and rates are derived at
read time, timestamps are epoch seconds, and anything unmeasured is NULL (never 0). Per-run
aggregates are maintained incrementally at ingest so reading a run never rescans its windows.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .db import Database, placeholders
from .privacy import sanitize

SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    parent_run_id TEXT,
    created_at REAL NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    lanes_json TEXT,
    config_json TEXT,
    env_json TEXT,
    dataset_json TEXT,
    git_commit TEXT,
    seed INTEGER,
    max_steps INTEGER,
    last_step INTEGER NOT NULL DEFAULT 0,
    last_step_at REAL,
    last_heartbeat_at REAL,
    step_time_ema_ms REAL,
    error TEXT,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS training_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    ts REAL NOT NULL,
    phase TEXT,
    window_steps INTEGER NOT NULL DEFAULT 1,
    loss REAL,
    lr REAL,
    grad_norm REAL,
    processed_tokens INTEGER,
    nonpad_tokens INTEGER,
    supervised_tokens INTEGER,
    step_time_ms REAL,
    excluded_ms REAL,
    mem_alloc_bytes INTEGER,
    mem_reserved_bytes INTEGER,
    mem_peak_bytes INTEGER,
    alloc_retries INTEGER,
    nan_count INTEGER,
    extra_json TEXT,
    UNIQUE(run_id, step)
);
CREATE TABLE IF NOT EXISTS training_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    eid TEXT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    step INTEGER,
    severity TEXT NOT NULL DEFAULT 'info',
    payload_json TEXT,
    UNIQUE(run_id, eid)
);
CREATE INDEX IF NOT EXISTS idx_training_events_run ON training_events(run_id, ts);
CREATE TABLE IF NOT EXISTS training_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    valid INTEGER NOT NULL,
    size_bytes INTEGER,
    save_ms REAL,
    manifest_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    UNIQUE(run_id, path)
);
CREATE TABLE IF NOT EXISTS hardware_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    pci_bus TEXT NOT NULL,
    lane TEXT,
    gpu_busy_pct REAL,
    vram_used_bytes INTEGER,
    vram_total_bytes INTEGER,
    gtt_used_bytes INTEGER,
    temp_edge_c REAL,
    temp_junction_c REAL,
    temp_memory_c REAL,
    power_w REAL,
    sclk_mhz REAL,
    mclk_mhz REAL,
    fan_rpm REAL,
    pcie_speed TEXT,
    pcie_width INTEGER
);
CREATE INDEX IF NOT EXISTS idx_hardware_samples ON hardware_samples(pci_bus, ts);
CREATE TABLE IF NOT EXISTS system_samples (
    ts REAL PRIMARY KEY,
    cpu_pct REAL,
    ram_used_pct REAL,
    swap_used_bytes INTEGER
);
CREATE TABLE IF NOT EXISTS training_rank_steps (
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    ts REAL NOT NULL,
    pci_bus TEXT,
    window_steps INTEGER NOT NULL DEFAULT 1,
    step_time_ms REAL,
    data_wait_ms REAL,
    comm_ms REAL,
    nonpad_tokens INTEGER,
    mem_peak_bytes INTEGER,
    PRIMARY KEY (run_id, step, rank)
);
CREATE TABLE IF NOT EXISTS training_ranks (
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    pci_bus TEXT,
    last_step INTEGER NOT NULL DEFAULT 0,
    last_step_at REAL,
    last_heartbeat_at REAL,
    PRIMARY KEY (run_id, rank)
);
CREATE TABLE IF NOT EXISTS checkpoint_evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    level TEXT NOT NULL,
    eval_set TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL DEFAULT 'completed',
    metrics_json TEXT,
    latency_json TEXT,
    versions_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    UNIQUE(run_id, step, level, eval_set)
);
CREATE TABLE IF NOT EXISTS training_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER,
    downloadable INTEGER NOT NULL DEFAULT 0,
    meta_json TEXT,
    created_at REAL NOT NULL,
    UNIQUE(run_id, path)
);
CREATE TABLE IF NOT EXISTS training_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES training_runs(id) ON DELETE CASCADE,
    ts REAL NOT NULL,
    author TEXT,
    hypothesis TEXT,
    change TEXT,
    outcome TEXT
);
CREATE TABLE IF NOT EXISTS training_example_stats (
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    example_id TEXT NOT NULL,
    loss REAL,
    correct INTEGER,
    extra_json TEXT,
    PRIMARY KEY (run_id, step, example_id)
);
CREATE TABLE IF NOT EXISTS hardware_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash TEXT NOT NULL UNIQUE,
    key_json TEXT NOT NULL,
    gpu_pci_bus TEXT,
    peak_mem_bytes INTEGER,
    tokens_per_sec REAL,
    measured_at REAL NOT NULL,
    run_id TEXT
);
"""

ADDITIONS = {
    "training_runs": {
        "world_size": "INTEGER NOT NULL DEFAULT 1",
        "tags_json": "TEXT",
        "agg_windows": "INTEGER NOT NULL DEFAULT 0",
        "agg_processed": "INTEGER NOT NULL DEFAULT 0",
        "agg_nonpad": "INTEGER NOT NULL DEFAULT 0",
        "agg_supervised": "INTEGER NOT NULL DEFAULT 0",
        "agg_pad_processed": "INTEGER NOT NULL DEFAULT 0",
        "agg_pad_nonpad": "INTEGER NOT NULL DEFAULT 0",
        "agg_sup_nonpad": "INTEGER NOT NULL DEFAULT 0",
        "agg_sup_supervised": "INTEGER NOT NULL DEFAULT 0",
        "agg_nonpad_ms": "REAL NOT NULL DEFAULT 0",
        "agg_timed_nonpad": "INTEGER NOT NULL DEFAULT 0",
        "agg_supervised_ms": "REAL NOT NULL DEFAULT 0",
        "agg_timed_supervised": "INTEGER NOT NULL DEFAULT 0",
        "agg_peak_mem": "INTEGER",
        "agg_nan": "INTEGER NOT NULL DEFAULT 0",
    },
    "training_steps": {
        "data_wait_ms": "REAL",
        "fwd_ms": "REAL",
        "bwd_ms": "REAL",
        "opt_ms": "REAL",
        "comm_ms": "REAL",
    },
    "training_checkpoints": {
        "best": "INTEGER NOT NULL DEFAULT 0",
        "designated": "INTEGER NOT NULL DEFAULT 0",
    },
}

STATUS_RUNNING = "running"
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
HARDWARE_FIELDS = (
    "gpu_busy_pct", "vram_used_bytes", "vram_total_bytes", "gtt_used_bytes", "temp_edge_c",
    "temp_junction_c", "temp_memory_c", "power_w", "sclk_mhz", "mclk_mhz", "fan_rpm",
    "pcie_speed", "pcie_width",
)
STEP_NUMERIC = ("loss", "lr", "grad_norm", "step_time_ms", "excluded_ms", "data_wait_ms", "fwd_ms", "bwd_ms", "opt_ms", "comm_ms")
STEP_INT = (
    "processed_tokens", "nonpad_tokens", "supervised_tokens", "mem_alloc_bytes",
    "mem_reserved_bytes", "mem_peak_bytes", "alloc_retries", "nan_count",
)
STEP_COLUMNS = (
    "run_id", "step", "ts", "phase", "window_steps", *STEP_NUMERIC, *STEP_INT, "extra_json",
)
MAX_TEXT = 20000
EVAL_LEVELS = ("native", "gguf_f16", "gguf_quant")


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    number = _num(value)
    return None if number is None else int(number)


def _text(value: Any, limit: int = MAX_TEXT) -> str | None:
    if value is None:
        return None
    return str(value)[:limit]


def _dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _load(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


class TrainingError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class TrainingStore(Database):
    def __init__(
        self,
        path: Path,
        clock: Callable[[], float] = time.time,
        heartbeat_lost_s: float = 90.0,
        stall_floor_s: float = 300.0,
        stall_factor: float = 20.0,
        memory_temp_alert_c: float = 90.0,
        junction_temp_alert_c: float = 100.0,
        hardware_interval_s: float = 1.0,
    ):
        super().__init__(path)
        self.clock = clock
        self.heartbeat_lost_s = heartbeat_lost_s
        self.stall_floor_s = stall_floor_s
        self.stall_factor = stall_factor
        self.memory_temp_alert_c = memory_temp_alert_c
        self.junction_temp_alert_c = junction_temp_alert_c
        self.hardware_interval_s = hardware_interval_s
        self.ensure_schema(SCHEMA, ADDITIONS)
        self._backfill_aggregates()
        self._active_cache: tuple[float, bool] = (0.0, False)

    # ------------------------------------------------------------------ runs

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise TrainingError("name is required.")
        run_id = str(payload.get("id") or uuid.uuid4().hex)
        now = self.clock()
        with self.connect() as connection:
            existing = connection.execute("SELECT id FROM training_runs WHERE id = ?", (run_id,)).fetchone()
            if existing is None:
                connection.execute(
                    """INSERT INTO training_runs
                    (id, name, status, parent_run_id, created_at, started_at, last_heartbeat_at,
                     lanes_json, config_json, env_json, dataset_json, git_commit, seed, max_steps,
                     world_size, tags_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id, name[:200], STATUS_RUNNING, _text(payload.get("parent_run_id"), 100),
                        now, _num(payload.get("started_at")) or now, now,
                        _dump(payload.get("lanes") or []),
                        _dump(sanitize(payload.get("config"), drop_content_keys=False, max_string=4000)),
                        _dump(payload.get("env")),
                        _dump(sanitize(payload.get("dataset"), drop_content_keys=False, max_string=4000)),
                        _text(payload.get("git_commit"), 100), _int(payload.get("seed")),
                        _int(payload.get("max_steps")), max(1, _int(payload.get("world_size")) or 1),
                        _dump(sanitize(payload.get("tags"), drop_content_keys=False, max_string=200)),
                    ),
                )
                connection.execute(
                    "INSERT INTO training_events (run_id, eid, ts, kind, severity, payload_json)"
                    " VALUES (?,?,?,?,?,?)",
                    (run_id, "created", now, "started", "info", _dump({"name": name})),
                )
        self._active_cache = (0.0, False)
        return self.get_run(run_id)

    def _row(self, run_id: str) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise TrainingError("Training run not found.", 404)
        return row

    def _judge(self, row: sqlite3.Row | dict[str, Any], now: float,
               ranks: list[sqlite3.Row] | None = None) -> tuple[str, str | None]:
        """Returns (status, reason). `stalled` is judged on step progress, not heartbeat arrival: a
        live heartbeat thread keeps beating while the main thread hangs in a collective. In a
        multi-rank run a rank that stops heartbeating also marks the job stalled (the surviving
        ranks block in the collective, so waiting for the run-level signal would take longer)."""
        if row["status"] != STATUS_RUNNING:
            return str(row["status"]), None
        heartbeat = row["last_heartbeat_at"] or row["started_at"]
        if now - heartbeat > self.heartbeat_lost_s:
            return "lost", None
        if row["world_size"] > 1 and ranks is not None:
            seen = {r["rank"]: r for r in ranks}
            for rank, info in sorted(seen.items()):
                age = now - (info["last_heartbeat_at"] or row["started_at"])
                if age > self.heartbeat_lost_s:
                    return "stalled", f"rank {rank} stopped reporting {age:.0f}s ago"
            missing = [r for r in range(row["world_size"]) if r not in seen]
            if missing and now - row["started_at"] > 2 * self.heartbeat_lost_s:
                return "stalled", f"rank(s) {', '.join(map(str, missing))} never reported"
        reference = row["last_step_at"] or row["started_at"]
        ema_s = (row["step_time_ema_ms"] or 0.0) / 1000.0
        threshold = max(self.stall_floor_s, self.stall_factor * ema_s)
        if now - reference > threshold:
            return "stalled", f"step has not advanced for {now - reference:.0f}s (threshold {threshold:.0f}s)"
        return STATUS_RUNNING, None

    def effective_status(self, row: sqlite3.Row | dict[str, Any], now: float | None = None) -> str:
        return self._judge(row, self.clock() if now is None else now)[0]

    def has_active_runs(self) -> bool:
        now = self.clock()
        checked_at, cached = self._active_cache
        if now - checked_at < 5.0:
            return cached
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT last_heartbeat_at, started_at FROM training_runs WHERE status = ?",
                (STATUS_RUNNING,),
            ).fetchall()
        active = any(now - (r["last_heartbeat_at"] or r["started_at"]) <= self.heartbeat_lost_s for r in rows)
        self._active_cache = (now, active)
        return active

    # -------------------------------------------------------------- aggregates

    @staticmethod
    def _delta(values: dict[str, Any], ints: dict[str, Any], window: int, phase: str) -> dict[str, float]:
        processed, nonpad, supervised = ints["processed_tokens"], ints["nonpad_tokens"], ints["supervised_tokens"]
        step_time = values["step_time_ms"]
        train_timed = phase == "train" and step_time is not None
        delta = {
            "agg_windows": 1, "agg_processed": processed or 0, "agg_nonpad": nonpad or 0,
            "agg_supervised": supervised or 0,
            "agg_pad_processed": processed if processed is not None and nonpad is not None else 0,
            "agg_pad_nonpad": nonpad if processed is not None and nonpad is not None else 0,
            "agg_sup_nonpad": nonpad if supervised is not None and nonpad is not None else 0,
            "agg_sup_supervised": supervised if supervised is not None and nonpad is not None else 0,
            "agg_nonpad_ms": step_time * window if train_timed and nonpad is not None else 0,
            "agg_timed_nonpad": nonpad if train_timed and nonpad is not None else 0,
            "agg_supervised_ms": step_time * window if train_timed and supervised is not None else 0,
            "agg_timed_supervised": supervised if train_timed and supervised is not None else 0,
            "agg_nan": ints["nan_count"] or 0,
        }
        return delta

    def recompute_aggregates(self, run_id: str) -> None:
        with self.connect() as connection:
            totals = {key: 0 for key in ADDITIONS["training_runs"] if key.startswith("agg_") and key != "agg_peak_mem"}
            peak = None
            for r in connection.execute(
                "SELECT phase, window_steps, step_time_ms, processed_tokens, nonpad_tokens, supervised_tokens,"
                " nan_count, mem_peak_bytes FROM training_steps WHERE run_id = ?", (run_id,)
            ):
                delta = self._delta(
                    {"step_time_ms": r["step_time_ms"]},
                    {"processed_tokens": r["processed_tokens"], "nonpad_tokens": r["nonpad_tokens"],
                     "supervised_tokens": r["supervised_tokens"], "nan_count": r["nan_count"]},
                    r["window_steps"], r["phase"],
                )
                for key, value in delta.items():
                    totals[key] += value
                if r["mem_peak_bytes"] is not None:
                    peak = r["mem_peak_bytes"] if peak is None else max(peak, r["mem_peak_bytes"])
            sets = ", ".join(f"{key} = ?" for key in totals)
            connection.execute(f"UPDATE training_runs SET {sets}, agg_peak_mem = ? WHERE id = ?",
                               (*totals.values(), peak, run_id))

    def _backfill_aggregates(self) -> None:
        with self.connect() as connection:
            stale = [r["id"] for r in connection.execute(
                "SELECT id FROM training_runs r WHERE agg_windows = 0 AND EXISTS"
                " (SELECT 1 FROM training_steps s WHERE s.run_id = r.id)"
            )]
        for run_id in stale:
            self.recompute_aggregates(run_id)

    # ------------------------------------------------------------ presentation

    def _present(self, row: sqlite3.Row, connection: sqlite3.Connection, include_blobs: bool) -> dict[str, Any]:
        run_id = row["id"]
        now = self.clock()
        ranks = connection.execute("SELECT * FROM training_ranks WHERE run_id = ? ORDER BY rank", (run_id,)).fetchall() \
            if row["world_size"] > 1 else []
        status, stall_reason = self._judge(row, now, ranks)
        latest = connection.execute(
            "SELECT * FROM training_steps WHERE run_id = ? ORDER BY step DESC LIMIT 1", (run_id,)
        ).fetchone()
        latest_loss = connection.execute(
            "SELECT loss, lr, grad_norm, step FROM training_steps WHERE run_id = ? AND loss IS NOT NULL"
            " ORDER BY step DESC LIMIT 1", (run_id,)
        ).fetchone()
        checkpoints = connection.execute(
            "SELECT COUNT(*) AS total, SUM(valid) AS valid FROM training_checkpoints WHERE run_id = ?", (run_id,)
        ).fetchone()
        end = row["finished_at"] or now
        max_steps, ema_ms = row["max_steps"], row["step_time_ema_ms"]
        eta = None
        if status == STATUS_RUNNING and max_steps and ema_ms and row["last_step"] < max_steps:
            eta = (max_steps - row["last_step"]) * ema_ms / 1000.0
        run = {
            "id": run_id, "name": row["name"], "status": status, "stall_reason": stall_reason,
            "recorded_status": row["status"], "parent_run_id": row["parent_run_id"],
            "started_at": row["started_at"], "finished_at": row["finished_at"],
            "elapsed_s": max(0.0, end - row["started_at"]),
            "lanes": _load(row["lanes_json"]) or [], "git_commit": row["git_commit"], "seed": row["seed"],
            "world_size": row["world_size"], "tags": _load(row["tags_json"]) or {},
            "max_steps": max_steps, "last_step": row["last_step"], "last_step_at": row["last_step_at"],
            "last_heartbeat_at": row["last_heartbeat_at"],
            "seconds_since_step": None if not row["last_step_at"] else max(0.0, now - row["last_step_at"]),
            "progress": None if not max_steps else min(1.0, row["last_step"] / max_steps),
            "eta_s": eta, "step_time_ema_ms": ema_ms, "error": row["error"],
            "tokens": {
                "processed": row["agg_processed"], "nonpad": row["agg_nonpad"], "supervised": row["agg_supervised"],
                "padding_ratio": _ratio_complement(row["agg_pad_nonpad"], row["agg_pad_processed"]),
                "supervision_ratio": _ratio(row["agg_sup_supervised"], row["agg_sup_nonpad"]),
            },
            "throughput": {
                "tokens_per_sec": _rate(row["agg_timed_nonpad"], row["agg_nonpad_ms"] / 1000.0),
                "supervised_tokens_per_sec": _rate(row["agg_timed_supervised"], row["agg_supervised_ms"] / 1000.0),
            },
            "latest": None if latest is None else {
                "step": latest["step"], "ts": latest["ts"],
                "tokens_per_sec": _step_rate(latest, "nonpad_tokens"),
                "supervised_tokens_per_sec": _step_rate(latest, "supervised_tokens"),
                "step_time_ms": latest["step_time_ms"], "mem_peak_bytes": latest["mem_peak_bytes"],
            },
            "latest_scalars": None if latest_loss is None else dict(latest_loss),
            "peak_mem_bytes": row["agg_peak_mem"], "nan_count": row["agg_nan"], "windows": row["agg_windows"],
            "checkpoints": {"total": checkpoints["total"] or 0, "valid": checkpoints["valid"] or 0},
        }
        if row["world_size"] > 1:
            run["ranks"] = [
                {"rank": r["rank"], "pci_bus": r["pci_bus"], "last_step": r["last_step"],
                 "heartbeat_age_s": None if r["last_heartbeat_at"] is None else max(0.0, now - r["last_heartbeat_at"])}
                for r in ranks
            ]
        if include_blobs:
            run.update(config=_load(row["config_json"]), env=_load(row["env_json"]),
                       dataset=_load(row["dataset_json"]), summary=_load(row["summary_json"]))
        return run

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise TrainingError("Training run not found.", 404)
            return self._present(row, connection, include_blobs=True)

    def list_runs(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        limit = max(1, min(int(limit), 500))
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ? OFFSET ?", (limit, max(0, int(offset)))
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            return {"runs": [self._present(r, connection, include_blobs=False) for r in rows], "total": total}

    def find_runs(self, tag_key: str, tag_value: Any) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute("SELECT id, tags_json FROM training_runs ORDER BY started_at").fetchall()
        return [r["id"] for r in rows if (_load(r["tags_json"]) or {}).get(tag_key) == tag_value]

    # --------------------------------------------------------------- ingestion

    def record_metrics(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self._row(run_id)
        steps = payload.get("steps") or []
        ranks = payload.get("ranks") or []
        if not isinstance(steps, list) or not isinstance(ranks, list):
            raise TrainingError("steps and ranks must be lists.")
        now = self.clock()
        accepted = 0
        max_step = row["last_step"]
        ema = row["step_time_ema_ms"]
        totals: dict[str, float] = {}
        peak = row["agg_peak_mem"]
        columns = ", ".join(STEP_COLUMNS)
        marks = placeholders(len(STEP_COLUMNS))
        with self.connect() as connection:
            for item in steps:
                if not isinstance(item, dict) or _int(item.get("step")) is None:
                    continue
                step = _int(item["step"])
                window = max(1, _int(item.get("window_steps")) or 1)
                phase = _text(item.get("phase"), 40) or "train"
                values = {k: _num(item.get(k)) for k in STEP_NUMERIC}
                ints = {k: _int(item.get(k)) for k in STEP_INT}
                extra = item.get("extra")
                cursor = connection.execute(
                    f"INSERT OR IGNORE INTO training_steps ({columns}) VALUES ({marks})",
                    (run_id, step, _num(item.get("ts")) or now, phase, window,
                     *(values[k] for k in STEP_NUMERIC), *(ints[k] for k in STEP_INT),
                     _dump(sanitize(extra)) if isinstance(extra, dict) else None),
                )
                max_step = max(max_step, step)
                if cursor.rowcount != 1:
                    continue  # replayed window: already counted
                accepted += 1
                for key, value in self._delta(values, ints, window, phase).items():
                    totals[key] = totals.get(key, 0) + value
                if ints["mem_peak_bytes"] is not None:
                    peak = ints["mem_peak_bytes"] if peak is None else max(peak, ints["mem_peak_bytes"])
                if values["step_time_ms"] and values["step_time_ms"] > 0 and phase == "train":
                    ema = values["step_time_ms"] if ema is None else 0.7 * ema + 0.3 * values["step_time_ms"]
            for item in ranks:
                if isinstance(item, dict) and _int(item.get("step")) is not None and _int(item.get("rank")) is not None:
                    connection.execute(
                        """INSERT OR IGNORE INTO training_rank_steps (run_id, step, rank, ts, pci_bus, window_steps,
                           step_time_ms, data_wait_ms, comm_ms, nonpad_tokens, mem_peak_bytes)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (run_id, _int(item["step"]), _int(item["rank"]), _num(item.get("ts")) or now,
                         _text(item.get("pci_bus"), 40), max(1, _int(item.get("window_steps")) or 1),
                         _num(item.get("step_time_ms")), _num(item.get("data_wait_ms")), _num(item.get("comm_ms")),
                         _int(item.get("nonpad_tokens")), _int(item.get("mem_peak_bytes"))),
                    )
            heartbeat = payload.get("heartbeat") if isinstance(payload.get("heartbeat"), dict) else {}
            reported = _int(heartbeat.get("last_step"))
            if reported is not None:
                max_step = max(max_step, reported)
            hb_rank = _int(heartbeat.get("rank"))
            if hb_rank is not None:
                connection.execute(
                    """INSERT INTO training_ranks (run_id, rank, pci_bus, last_step, last_step_at, last_heartbeat_at)
                       VALUES (?,?,?,?,?,?)
                       ON CONFLICT(run_id, rank) DO UPDATE SET
                         last_step_at = CASE WHEN excluded.last_step > last_step THEN excluded.last_step_at ELSE last_step_at END,
                         last_step = MAX(last_step, excluded.last_step), last_heartbeat_at = excluded.last_heartbeat_at,
                         pci_bus = COALESCE(excluded.pci_bus, pci_bus)""",
                    (run_id, hb_rank, _text(heartbeat.get("pci_bus"), 40), reported or 0, now, now),
                )
            advanced = max_step > row["last_step"]
            sets = "".join(f", {key} = {key} + ?" for key in totals)
            connection.execute(
                f"""UPDATE training_runs SET last_heartbeat_at = ?, last_step = ?,
                   last_step_at = CASE WHEN ? THEN ? ELSE last_step_at END, step_time_ema_ms = ?,
                   agg_peak_mem = ?{sets} WHERE id = ?""",
                (now, max_step, 1 if advanced else 0, now, ema, peak, *totals.values(), run_id),
            )
        self._active_cache = (0.0, False)
        return {"accepted": accepted, "last_step": max_step}

    def add_event(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._row(run_id)
        kind = str(payload.get("kind") or "").strip()
        if not kind:
            raise TrainingError("kind is required.")
        severity = str(payload.get("severity") or "info")
        if severity not in {"info", "warning", "error"}:
            severity = "info"
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO training_events (run_id, eid, ts, kind, step, severity, payload_json)"
                " VALUES (?,?,?,?,?,?,?)",
                (run_id, _text(payload.get("eid"), 100), _num(payload.get("ts")) or self.clock(),
                 kind[:60], _int(payload.get("step")), severity, _dump(sanitize(payload.get("payload")))),
            )
        return {"ok": True}

    def add_checkpoint(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._row(run_id)
        path = str(payload.get("path") or "").strip()
        step = _int(payload.get("step"))
        if not path or step is None:
            raise TrainingError("path and step are required.")
        manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else None
        valid = 1 if payload.get("valid") and manifest else 0
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO training_checkpoints
                (run_id, step, kind, path, valid, size_bytes, save_ms, manifest_json, error, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id, path) DO UPDATE SET step = excluded.step, kind = excluded.kind,
                  valid = excluded.valid, size_bytes = excluded.size_bytes, save_ms = excluded.save_ms,
                  manifest_json = excluded.manifest_json, error = excluded.error""",
                (run_id, step, str(payload.get("kind") or "resume")[:20], path[:1000], valid,
                 _int(payload.get("size_bytes")), _num(payload.get("save_ms")), _dump(manifest),
                 _text(payload.get("error"), 2000), self.clock()),
            )
        return {"ok": True, "valid": bool(valid)}

    def finish_run(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self._row(run_id)
        status = str(payload.get("status") or "completed")
        if status not in TERMINAL_STATUSES:
            raise TrainingError("status must be completed, failed, or cancelled.")
        now = self.clock()
        error = sanitize(payload.get("error"), drop_content_keys=False, max_string=8000)
        with self.connect() as connection:
            connection.execute(
                "UPDATE training_runs SET status = ?, finished_at = ?, error = ?, summary_json = ?,"
                " last_heartbeat_at = ? WHERE id = ?",
                (status, now, _text(error), _dump(sanitize(payload.get("summary"))), now, run_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO training_events (run_id, eid, ts, kind, step, severity, payload_json)"
                " VALUES (?,?,?,?,?,?,?)",
                (run_id, "finished", now, "finished", row["last_step"],
                 "error" if status == "failed" else "info", _dump({"status": status, "error": _text(error, 2000)})),
            )
        self._active_cache = (0.0, False)
        return self.get_run(run_id)

    def add_event_to_active_runs(self, kind: str, severity: str, payload: dict[str, Any]) -> int:
        """Attach a host-level event (e.g. an amdgpu fault from the kernel log) to every live run."""
        count = 0
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, last_step FROM training_runs WHERE status = ?", (STATUS_RUNNING,)
            ).fetchall()
            for row in rows:
                connection.execute(
                    "INSERT INTO training_events (run_id, ts, kind, step, severity, payload_json)"
                    " VALUES (?,?,?,?,?,?)",
                    (row["id"], self.clock(), kind, row["last_step"], severity, _dump(payload)),
                )
                count += 1
        return count

    # ------------------------------------------------------------------ reads

    def series(self, run_id: str, max_points: int = 600, since_step: int = 0) -> dict[str, Any]:
        self._row(run_id)
        max_points = max(10, min(int(max_points), 5000))
        with self.connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM training_steps WHERE run_id = ? AND step > ?", (run_id, since_step)
            ).fetchone()[0]
            stride = max(1, math.ceil(total / max_points))
            # Downsample in SQL (every stride-th window plus the newest) instead of shipping every row.
            rows = connection.execute(
                """SELECT * FROM (
                       SELECT *, ROW_NUMBER() OVER (ORDER BY step) AS rn, COUNT(*) OVER () AS n,
                              SUM(COALESCE(nonpad_tokens, 0)) OVER (ORDER BY step) AS cum_nonpad
                       FROM training_steps WHERE run_id = ? AND step > ?
                   ) WHERE (rn - 1) % ? = 0 OR rn = n ORDER BY step""",
                (run_id, since_step, stride),
            ).fetchall()
        points = []
        for r in rows:
            points.append({
                "step": r["step"], "ts": r["ts"], "phase": r["phase"], "window_steps": r["window_steps"],
                "loss": r["loss"], "lr": r["lr"], "grad_norm": r["grad_norm"], "cum_nonpad_tokens": r["cum_nonpad"],
                "tokens_per_sec": _step_rate(r, "nonpad_tokens"),
                "supervised_tokens_per_sec": _step_rate(r, "supervised_tokens"),
                "processed_tokens_per_sec": _step_rate(r, "processed_tokens"),
                "padding_ratio": _ratio_complement(r["nonpad_tokens"], r["processed_tokens"]),
                "step_time_ms": r["step_time_ms"], "data_wait_ms": r["data_wait_ms"], "fwd_ms": r["fwd_ms"],
                "bwd_ms": r["bwd_ms"], "opt_ms": r["opt_ms"], "comm_ms": r["comm_ms"],
                "mem_alloc_bytes": r["mem_alloc_bytes"], "mem_reserved_bytes": r["mem_reserved_bytes"],
                "mem_peak_bytes": r["mem_peak_bytes"], "alloc_retries": r["alloc_retries"],
                "nan_count": r["nan_count"],
            })
        return {"points": points, "stride": stride, "total_windows": total}

    def step_windows(self, run_id: str, phase: str | None = "train", tail: int | None = None) -> list[dict[str, Any]]:
        """Lightweight rows for analysis (no extras); oldest first. `tail` keeps only the newest N."""
        sql = ("SELECT step, ts, phase, window_steps, loss, nonpad_tokens, supervised_tokens, processed_tokens, "
               "step_time_ms, excluded_ms, data_wait_ms, fwd_ms, bwd_ms, opt_ms, comm_ms, mem_peak_bytes "
               "FROM training_steps WHERE run_id = ?")
        args: list[Any] = [run_id]
        if phase:
            sql += " AND phase = ?"
            args.append(phase)
        with self.connect() as connection:
            if tail:
                rows = connection.execute(sql + " ORDER BY step DESC LIMIT ?", [*args, tail]).fetchall()
                return [dict(r) for r in reversed(rows)]
            return [dict(r) for r in connection.execute(sql + " ORDER BY step", args)]

    def rank_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [dict(r) for r in connection.execute(
                "SELECT * FROM training_rank_steps WHERE run_id = ? ORDER BY step, rank", (run_id,))]

    def events(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        self._row(run_id)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM training_events WHERE run_id = ? ORDER BY ts DESC, id DESC LIMIT ?",
                (run_id, max(1, min(int(limit), 1000))),
            ).fetchall()
        return [
            {"ts": r["ts"], "kind": r["kind"], "step": r["step"], "severity": r["severity"],
             "payload": _load(r["payload_json"])}
            for r in rows
        ]

    def checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        self._row(run_id)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM training_checkpoints WHERE run_id = ? ORDER BY step DESC", (run_id,)
            ).fetchall()
            evals = self._evals(connection, run_id)
        result = []
        for r in rows:
            manifest = _load(r["manifest_json"]) or {}
            result.append({
                "step": r["step"], "kind": r["kind"], "path": r["path"], "valid": bool(r["valid"]),
                "size_bytes": r["size_bytes"], "save_ms": r["save_ms"], "error": r["error"],
                "created_at": r["created_at"], "tokens_seen": manifest.get("tokens_seen"),
                "file_count": len(manifest.get("files") or []),
                "best": bool(r["best"]), "designated": bool(r["designated"]),
                "evals": evals.get(r["step"], {}),
            })
        return result

    def set_checkpoint_flags(self, run_id: str, step: int, *, best: bool | None = None,
                             designated: bool | None = None, exclusive_best: bool = True) -> None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM training_checkpoints WHERE run_id = ? AND step = ?", (run_id, step)
            ).fetchone()
            if row is None:
                raise TrainingError("Checkpoint not found.", 404)
            if best is not None:
                if best and exclusive_best:
                    connection.execute("UPDATE training_checkpoints SET best = 0 WHERE run_id = ?", (run_id,))
                connection.execute("UPDATE training_checkpoints SET best = ? WHERE run_id = ? AND step = ?",
                                   (1 if best else 0, run_id, step))
            if designated is not None:
                connection.execute("UPDATE training_checkpoints SET designated = ? WHERE run_id = ? AND step = ?",
                                   (1 if designated else 0, run_id, step))

    # ------------------------------------------------------------ evaluations

    def record_eval(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._row(run_id)
        step = _int(payload.get("step"))
        level = str(payload.get("level") or "")
        if step is None or level not in EVAL_LEVELS:
            raise TrainingError(f"step and level ({', '.join(EVAL_LEVELS)}) are required.")
        status = str(payload.get("status") or "completed")
        if status not in {"completed", "failed", "skipped"}:
            raise TrainingError("status must be completed, failed, or skipped.")
        if status == "completed" and not isinstance(payload.get("metrics"), dict):
            raise TrainingError("A completed evaluation needs a metrics object.")
        with self.connect() as connection:
            if status != "completed":
                # A failed or skipped attempt must never replace a result that already completed
                # (e.g. one recorded by an external harness): keep it and note the failure instead.
                existing = connection.execute(
                    "SELECT status FROM checkpoint_evals WHERE run_id = ? AND step = ? AND level = ? AND eval_set = ?",
                    (run_id, step, level, str(payload.get("eval_set") or "default")[:80])).fetchone()
                if existing is not None and existing["status"] == "completed":
                    return {"ok": True, "kept_completed": True}
            connection.execute(
                """INSERT INTO checkpoint_evals (run_id, step, level, eval_set, status, metrics_json, latency_json,
                   versions_json, error, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id, step, level, eval_set) DO UPDATE SET status = excluded.status,
                   metrics_json = excluded.metrics_json, latency_json = excluded.latency_json,
                   versions_json = excluded.versions_json, error = excluded.error, created_at = excluded.created_at""",
                (run_id, step, level, str(payload.get("eval_set") or "default")[:80], status,
                 _dump(sanitize(payload.get("metrics"))), _dump(sanitize(payload.get("latency"))),
                 _dump(sanitize(payload.get("versions"), drop_content_keys=False)),
                 _text(sanitize(payload.get("error"), drop_content_keys=False, max_string=2000), 2000), self.clock()),
            )
        return {"ok": True}

    def _evals(self, connection: sqlite3.Connection, run_id: str) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for r in connection.execute("SELECT * FROM checkpoint_evals WHERE run_id = ? ORDER BY created_at", (run_id,)):
            result.setdefault(r["step"], {}).setdefault(r["level"], {})[r["eval_set"]] = {
                "status": r["status"], "metrics": _load(r["metrics_json"]), "latency": _load(r["latency_json"]),
                "versions": _load(r["versions_json"]), "error": r["error"], "created_at": r["created_at"],
            }
        return result

    def evals(self, run_id: str) -> dict[int, dict[str, Any]]:
        self._row(run_id)
        with self.connect() as connection:
            return self._evals(connection, run_id)

    # ------------------------------------------------- artifacts, notes, examples

    def add_artifact(self, run_id: str, kind: str, path: str, *, size_bytes: int | None, downloadable: bool,
                     meta: dict[str, Any] | None = None) -> dict[str, Any]:
        self._row(run_id)
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO training_artifacts (run_id, kind, path, size_bytes, downloadable, meta_json, created_at)
                   VALUES (?,?,?,?,?,?,?) ON CONFLICT(run_id, path) DO UPDATE SET kind = excluded.kind,
                   size_bytes = excluded.size_bytes, downloadable = excluded.downloadable, meta_json = excluded.meta_json""",
                (run_id, kind[:40], path[:1000], size_bytes, 1 if downloadable else 0,
                 _dump(sanitize(meta, drop_content_keys=False)), self.clock()),
            )
            row = connection.execute("SELECT id FROM training_artifacts WHERE run_id = ? AND path = ?", (run_id, path[:1000])).fetchone()
        return {"id": row["id"], "kind": kind, "path": path, "downloadable": downloadable}

    def artifacts(self, run_id: str) -> list[dict[str, Any]]:
        self._row(run_id)
        with self.connect() as connection:
            return [
                {"id": r["id"], "kind": r["kind"], "path": r["path"], "size_bytes": r["size_bytes"],
                 "downloadable": bool(r["downloadable"]), "meta": _load(r["meta_json"]), "created_at": r["created_at"]}
                for r in connection.execute("SELECT * FROM training_artifacts WHERE run_id = ? ORDER BY id", (run_id,))
            ]

    def get_artifact(self, run_id: str, artifact_id: int) -> dict[str, Any]:
        for artifact in self.artifacts(run_id):
            if artifact["id"] == artifact_id:
                return artifact
        raise TrainingError("Artifact not found.", 404)

    def add_note(self, run_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        if run_id:
            self._row(run_id)
        fields = {k: sanitize(str(payload.get(k) or ""), drop_content_keys=False, max_string=4000)
                  for k in ("author", "hypothesis", "change", "outcome")}
        if not any(fields.values()):
            raise TrainingError("A note needs a hypothesis, change, or outcome.")
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO training_notes (run_id, ts, author, hypothesis, change, outcome) VALUES (?,?,?,?,?,?)",
                (run_id, self.clock(), fields["author"], fields["hypothesis"], fields["change"], fields["outcome"]),
            )
        return {"id": cursor.lastrowid}

    def notes(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            sql, args = "SELECT * FROM training_notes", ()
            if run_id:
                sql, args = sql + " WHERE run_id = ?", (run_id,)
            return [dict(r) for r in connection.execute(sql + " ORDER BY ts DESC", args)]

    def add_example_stats(self, run_id: str, step: int, items: list[dict[str, Any]]) -> int:
        """Per-example loss/correctness keyed by an opaque example_id; never example content."""
        self._row(run_id)
        rows = []
        for item in items[:5000]:
            example_id = _text(item.get("example_id"), 100)
            if example_id:
                rows.append((run_id, step, example_id, _num(item.get("loss")),
                             None if item.get("correct") is None else (1 if item["correct"] else 0),
                             _dump(sanitize(item.get("extra")))))
        with self.connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO training_example_stats (run_id, step, example_id, loss, correct, extra_json)"
                " VALUES (?,?,?,?,?,?)", rows)
        return len(rows)

    def hard_examples(self, run_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Examples with the highest loss at the latest recorded step (data-map style triage)."""
        with self.connect() as connection:
            row = connection.execute("SELECT MAX(step) FROM training_example_stats WHERE run_id = ?", (run_id,)).fetchone()
            if row[0] is None:
                return []
            return [dict(r) for r in connection.execute(
                "SELECT example_id, step, loss, correct FROM training_example_stats WHERE run_id = ? AND step = ?"
                " ORDER BY loss DESC LIMIT ?", (run_id, row[0], max(1, min(int(limit), 500))))]

    # ---------------------------------------------------------- hardware / system

    def add_hardware_samples(self, samples: list[dict[str, Any]]) -> int:
        if not samples:
            return 0
        rows = [(s["ts"], s["pci_bus"], s.get("lane"), *(s.get(field) for field in HARDWARE_FIELDS)) for s in samples]
        columns = ", ".join(("ts", "pci_bus", "lane") + HARDWARE_FIELDS)
        with self.connect() as connection:
            connection.executemany(
                f"INSERT INTO hardware_samples ({columns}) VALUES ({placeholders(3 + len(HARDWARE_FIELDS))})", rows)
        return len(rows)

    def add_system_sample(self, ts: float, cpu_pct: float | None, ram_used_pct: float | None,
                          swap_used_bytes: int | None) -> None:
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO system_samples (ts, cpu_pct, ram_used_pct, swap_used_bytes)"
                               " VALUES (?,?,?,?)", (ts, cpu_pct, ram_used_pct, swap_used_bytes))

    def prune_hardware(self, older_than_days: float) -> int:
        cutoff = self.clock() - older_than_days * 86400
        with self.connect() as connection:
            connection.execute("DELETE FROM system_samples WHERE ts < ?", (cutoff,))
            return connection.execute("DELETE FROM hardware_samples WHERE ts < ?", (cutoff,)).rowcount

    def run_window(self, run_id: str) -> tuple[float, float]:
        row = self._row(run_id)
        return row["started_at"], row["finished_at"] or self.clock()

    def hardware_rows(self, run_id: str, bus: str | None = None, since: float | None = None,
                      columns: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        start, end = self.run_window(run_id)
        selected = ", ".join(columns) if columns else "*"
        sql, args = f"SELECT {selected} FROM hardware_samples WHERE ts BETWEEN ? AND ?", [max(start, since) if since else start, end]
        if bus:
            sql, args = sql + " AND pci_bus = ?", [*args, bus]
        with self.connect() as connection:
            return [dict(r) for r in connection.execute(sql + " ORDER BY ts", args)]

    def system_rows(self, run_id: str) -> list[dict[str, Any]]:
        start, end = self.run_window(run_id)
        with self.connect() as connection:
            return [dict(r) for r in connection.execute(
                "SELECT * FROM system_samples WHERE ts BETWEEN ? AND ? ORDER BY ts", (start, end))]

    def hardware_series(self, run_id: str, max_points: int = 600) -> dict[str, Any]:
        start, end = self.run_window(run_id)
        max_points = max(10, min(int(max_points), 5000))
        with self.connect() as connection:
            counts = {r["pci_bus"]: r["n"] for r in connection.execute(
                "SELECT pci_bus, COUNT(*) AS n FROM hardware_samples WHERE ts BETWEEN ? AND ? GROUP BY pci_bus",
                (start, end))}
            gpus = {}
            for bus, count in counts.items():
                stride = max(1, math.ceil(count / max_points))
                rows = connection.execute(
                    """SELECT * FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY ts) AS rn FROM hardware_samples
                       WHERE pci_bus = ? AND ts BETWEEN ? AND ?) WHERE (rn - 1) % ? = 0 OR rn = ? ORDER BY ts""",
                    (bus, start, end, stride, count),
                ).fetchall()
                gpus[bus] = {
                    "lane": rows[-1]["lane"] if rows else None, "stride": stride,
                    "points": [{"ts": r["ts"], **{f: r[f] for f in HARDWARE_FIELDS}} for r in rows],
                }
        return {"gpus": gpus}

    def hardware_summary(self, run_id: str) -> dict[str, Any]:
        """Per-GPU aggregates over the run window. Values are NULL when no sample reported them."""
        start, end = self.run_window(run_id)
        interval = self.hardware_interval_s
        result: dict[str, Any] = {}
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT pci_bus, MAX(lane) AS lane, COUNT(*) AS samples,
                          AVG(gpu_busy_pct) AS avg_busy, MAX(vram_used_bytes) AS peak_vram,
                          MAX(vram_total_bytes) AS vram_total, MAX(gtt_used_bytes) AS peak_gtt,
                          MAX(temp_edge_c) AS max_edge, MAX(temp_junction_c) AS max_junction,
                          MAX(temp_memory_c) AS max_memory, AVG(power_w) AS avg_power,
                          MAX(power_w) AS max_power, MIN(pcie_width) AS min_pcie_width,
                          SUM(CASE WHEN temp_memory_c >= ? THEN 1 ELSE 0 END) AS mem_over,
                          SUM(CASE WHEN temp_junction_c >= ? THEN 1 ELSE 0 END) AS junction_over,
                          SUM(power_w) AS power_sum
                   FROM hardware_samples WHERE ts BETWEEN ? AND ? GROUP BY pci_bus""",
                (self.memory_temp_alert_c, self.junction_temp_alert_c, start, end),
            ).fetchall()
        for r in rows:
            result[r["pci_bus"]] = {
                "lane": r["lane"], "samples": r["samples"], "avg_busy_pct": r["avg_busy"],
                "peak_vram_bytes": r["peak_vram"], "vram_total_bytes": r["vram_total"],
                "peak_gtt_bytes": r["peak_gtt"], "max_temp_edge_c": r["max_edge"],
                "max_temp_junction_c": r["max_junction"], "max_temp_memory_c": r["max_memory"],
                "avg_power_w": r["avg_power"], "max_power_w": r["max_power"],
                "min_pcie_width": r["min_pcie_width"],
                "seconds_memory_over_alert": None if r["max_memory"] is None else r["mem_over"] * interval,
                "seconds_junction_over_alert": None if r["max_junction"] is None else r["junction_over"] * interval,
                "energy_wh": None if r["power_sum"] is None else r["power_sum"] * interval / 3600.0,
            }
        return {
            "gpus": result,
            "alerts": {"memory_temp_c": self.memory_temp_alert_c, "junction_temp_c": self.junction_temp_alert_c},
        }

    # ------------------------------------------------------- hardware profiles

    def record_profile(self, key_hash: str, key: dict[str, Any], *, gpu_pci_bus: str | None,
                       peak_mem_bytes: int | None, tokens_per_sec: float | None, run_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO hardware_profiles (key_hash, key_json, gpu_pci_bus, peak_mem_bytes, tokens_per_sec,
                   measured_at, run_id) VALUES (?,?,?,?,?,?,?) ON CONFLICT(key_hash) DO UPDATE SET
                   peak_mem_bytes = MAX(COALESCE(peak_mem_bytes, 0), COALESCE(excluded.peak_mem_bytes, 0)),
                   tokens_per_sec = excluded.tokens_per_sec, measured_at = excluded.measured_at, run_id = excluded.run_id""",
                (key_hash, _dump(key), gpu_pci_bus, peak_mem_bytes, tokens_per_sec, self.clock(), run_id),
            )

    def get_profile(self, key_hash: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM hardware_profiles WHERE key_hash = ?", (key_hash,)).fetchone()
        return None if row is None else {**dict(row), "key": _load(row["key_json"])}

    def profiles(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [{**dict(r), "key": _load(r["key_json"])} for r in
                    connection.execute("SELECT * FROM hardware_profiles ORDER BY measured_at DESC")]


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _ratio_complement(part: Any, whole: Any) -> float | None:
    ratio = _ratio(part, whole)
    return None if ratio is None else max(0.0, 1.0 - ratio)


def _rate(tokens: Any, seconds: float) -> float | None:
    if tokens is None or seconds <= 0:
        return None
    return tokens / seconds


def _step_rate(row: sqlite3.Row, column: str) -> float | None:
    if row[column] is None or not row["step_time_ms"]:
        return None
    active_s = row["step_time_ms"] * (row["window_steps"] or 1) / 1000.0
    return row[column] / active_s if active_s > 0 else None
