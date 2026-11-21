"""Share history between a Windows and a Linux install through plain files in a shared folder.

A live SQLite file must never sit on a partition two operating systems write to (NTFS from Linux, WAL side
files, Windows Fast Startup are how databases get corrupted), so each install keeps its databases on its own
native disk and exchanges *files*:

    <exchange_dir>/<os>/runs-YYYY-MM-DD.jsonl.gz       inference runs created that UTC day on that OS (+ their metric samples)
    <exchange_dir>/<os>/training-<os>.sqlite3          a consistent snapshot (SQLite backup API) of that OS's training
                                                       database: training runs, hardware log, alerts, events

Each install exports only what originated on it (rows imported from the other OS are never re-exported, so
nothing ping-pongs), imports every other OS's files by id (rows already present are skipped, nothing existing
is changed), and tags every row with the OS it came from. Files are written to a temp name and renamed, so a
reader never sees a half-written file, and old files are never deleted.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from .osfilter import OS_NAMES

FORMAT_VERSION = 1
RUN_SNAPSHOT_MIN_INTERVAL_S = 6 * 3600


def _safe_os(name: str | None) -> str:
    return "".join(c for c in (name or "unknown").lower() if c.isalnum()) or "unknown"


def _write_gzip_lines(target: Path, lines: list[dict[str, Any]]) -> None:
    partial = target.with_name(target.name + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as handle:
        for line in lines:
            handle.write(json.dumps(line, separators=(",", ":"), default=str) + "\n")
    partial.replace(target)


def read_runs_file(path: Path) -> tuple[dict[str, Any] | None, dict[str, list[dict[str, Any]]], dict[str, int] | None]:
    header, rows, counts = None, {}, None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("t") == "_start":
                    header = item
                elif item.get("t") == "_end":
                    counts = item.get("counts") or {}
                elif "row" in item:
                    rows.setdefault(item["t"], []).append(item["row"])
    except (OSError, EOFError, ValueError):
        return header, rows, None
    return header, rows, counts


class ExchangeSync:
    def __init__(self, exchange_dir: Path | str, host_os: str, data_dir: Path | str, run_db_path: Path | str,
                 training_db_path: Path | str, merge_training: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.time):
        self.root = Path(exchange_dir)
        self.host_os = host_os
        self.os_key = _safe_os(host_os)
        self.data_dir = Path(data_dir)
        self.run_db = Path(run_db_path)
        self.training_db = Path(training_db_path)
        self.merge_training = merge_training
        self.clock = clock
        self.state_path = self.data_dir / "exchange_state.json"
        self.last: dict[str, Any] = {}

    # ------------------------------------------------------------------ state

    def _state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"imported": {}, "snapshot_exported_at": 0}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.state_path.with_name(self.state_path.name + ".tmp")
        partial.write_text(json.dumps(state, indent=1), encoding="utf-8")
        partial.replace(self.state_path)

    # ----------------------------------------------------------------- export

    @property
    def own_dir(self) -> Path:
        return self.root / self.os_key

    def export_runs(self) -> dict[str, Any]:
        """One file per UTC day for runs that originated on this OS. A day's file is rewritten only when its
        row counts changed (today's, while requests are still arriving)."""
        if not self.run_db.exists():
            return {"written": 0}
        self.own_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        connection = sqlite3.connect(f"file:{self.run_db.as_posix()}?mode=ro", uri=True, timeout=60)
        connection.row_factory = sqlite3.Row
        try:
            days = [r[0] for r in connection.execute(
                "SELECT DISTINCT substr(created_at, 1, 10) FROM runs WHERE origin_os = ? AND status != 'running' ORDER BY 1", (self.host_os,))]
            for day in days:
                runs = [dict(r) for r in connection.execute(
                    "SELECT * FROM runs WHERE origin_os = ? AND status != 'running' AND substr(created_at, 1, 10) = ? ORDER BY created_at",
                    (self.host_os, day))]
                target = self.own_dir / f"runs-{day}.{self.os_key}.jsonl.gz"
                if target.exists():
                    _h, existing, counts = read_runs_file(target)
                    if counts is not None and len(existing.get("runs", ())) == len(runs):
                        done = {r["id"]: r.get("finished_at") for r in existing.get("runs", ())}
                        if all(done.get(r["id"]) == r.get("finished_at") for r in runs):
                            continue
                ids = [r["id"] for r in runs]
                marks = ",".join("?" * len(ids))
                deployments = [dict(r) for r in connection.execute(
                    f"SELECT * FROM deployments WHERE id IN (SELECT deployment_id FROM runs WHERE id IN ({marks}))", ids)]
                samples = []
                for chunk in range(0, len(ids), 500):
                    part = ids[chunk:chunk + 500]
                    samples += [dict(r) for r in connection.execute(
                        f"SELECT * FROM metric_samples WHERE run_id IN ({','.join('?' * len(part))}) ORDER BY id", part)]
                for sample in samples:
                    sample.pop("id", None)
                lines: list[dict[str, Any]] = [{"t": "_start", "format": FORMAT_VERSION, "day": day, "host_os": self.host_os}]
                lines += [{"t": "deployments", "row": d} for d in deployments]
                lines += [{"t": "runs", "row": r} for r in runs]
                lines += [{"t": "metric_samples", "row": s} for s in samples]
                lines.append({"t": "_end", "counts": {"deployments": len(deployments), "runs": len(runs),
                                                      "metric_samples": len(samples)}})
                _write_gzip_lines(target, lines)
                written += 1
        finally:
            connection.close()
        return {"written": written}

    def export_training_snapshot(self, force: bool = False) -> dict[str, Any]:
        """Consistent copy of the training database (SQLite backup API), at most every few hours."""
        if not self.training_db.exists():
            return {"written": False, "reason": "no training database yet"}
        state = self._state()
        now = self.clock()
        if not force and now - state.get("snapshot_exported_at", 0) < RUN_SNAPSHOT_MIN_INTERVAL_S:
            return {"written": False, "reason": "exported recently"}
        self.own_dir.mkdir(parents=True, exist_ok=True)
        target = self.own_dir / f"training-{self.os_key}.sqlite3"
        partial = target.with_name(target.name + ".partial")
        free = shutil.disk_usage(self.own_dir).free
        if free < self.training_db.stat().st_size * 2:
            return {"written": False, "reason": "not enough free space in the exchange folder"}
        source = sqlite3.connect(f"file:{self.training_db.as_posix()}?mode=ro", uri=True, timeout=60)
        try:
            destination = sqlite3.connect(partial)
            try:
                source.backup(destination)
                destination.execute("PRAGMA journal_mode = DELETE")  # a single self-contained file, no WAL side files
            finally:
                destination.close()
        finally:
            source.close()
        partial.replace(target)
        state["snapshot_exported_at"] = now
        self._save_state(state)
        return {"written": True, "bytes": target.stat().st_size}

    # ----------------------------------------------------------------- import

    def _foreign_dirs(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return [p for p in sorted(self.root.iterdir()) if p.is_dir() and p.name != self.os_key]

    def import_runs_file(self, path: Path) -> dict[str, int]:
        header, rows, counts = read_runs_file(path)
        if header is None or counts is None or any(len(rows.get(t, ())) != n for t, n in counts.items()):
            raise ValueError(f"{path.name} is incomplete or damaged; not imported")
        if int(header.get("format", 0)) > FORMAT_VERSION:
            raise ValueError(f"{path.name} uses format {header['format']}, newer than this install ({FORMAT_VERSION}); update first")
        origin = header.get("host_os") or None
        inserted = {"deployments": 0, "runs": 0, "metric_samples": 0}
        connection = sqlite3.connect(self.run_db, timeout=60)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                def columns(table: str) -> set[str]:
                    return {r[1] for r in connection.execute(f"PRAGMA table_info({table})")}

                new_runs: set[str] = set()
                for table in ("deployments", "runs"):
                    known = columns(table)
                    for row in rows.get(table, ()):
                        if table == "runs" and not row.get("origin_os"):
                            row["origin_os"], row["origin_os_inferred"] = origin, 1
                        data = {k: v for k, v in row.items() if k in known}
                        cursor = connection.execute(
                            f"INSERT OR IGNORE INTO {table} ({', '.join(data)}) VALUES ({','.join('?' * len(data))})",
                            tuple(data.values()))
                        inserted[table] += cursor.rowcount
                        if table == "runs" and cursor.rowcount:
                            new_runs.add(row["id"])  # samples are only added for runs that were not here before
                known = columns("metric_samples")
                for sample in rows.get("metric_samples", ()):
                    if sample.get("run_id") in new_runs:
                        data = {k: v for k, v in sample.items() if k in known and k != "id"}
                        connection.execute(
                            f"INSERT INTO metric_samples ({', '.join(data)}) VALUES ({','.join('?' * len(data))})", tuple(data.values()))
                        inserted["metric_samples"] += 1
        finally:
            connection.close()
        return inserted

    def import_all(self) -> dict[str, Any]:
        """Import every foreign file not yet imported (by name + size + mtime). Failures are recorded per file
        and never stop the others; nothing is deleted from the exchange folder."""
        state = self._state()
        imported, errors = state.setdefault("imported", {}), []
        totals: dict[str, int] = {}
        for folder in self._foreign_dirs():
            for path in sorted(folder.glob("runs-*.jsonl.gz")) + sorted(folder.glob("training-*.sqlite3")):
                stamp = f"{path.stat().st_size}:{int(path.stat().st_mtime)}"
                key = f"{folder.name}/{path.name}"
                if imported.get(key) == stamp:
                    continue
                try:
                    if path.suffix == ".gz":
                        for table, n in self.import_runs_file(path).items():
                            totals[table] = totals.get(table, 0) + n
                    elif self.merge_training is not None:
                        report = self.merge_training(self.training_db, path, all_telemetry=True,
                                                     source_os=OS_NAMES.get(folder.name), dest_os=self.host_os)
                        if not report.ok:
                            raise ValueError("; ".join(report.errors))
                        for table, n in report.rows.items():
                            totals[table] = totals.get(table, 0) + n
                    imported[key] = stamp
                except Exception as exc:  # noqa: BLE001
                    errors.append({"file": key, "error": f"{type(exc).__name__}: {exc}"})
        self._save_state(state)
        return {"imported_rows": totals, "errors": errors}

    # ------------------------------------------------------------------ driver

    def sync(self, include_snapshot: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"at": self.clock()}
        try:
            result["export_runs"] = self.export_runs()
            if include_snapshot:
                result["export_training"] = self.export_training_snapshot()
            result["import"] = self.import_all()
        except Exception as exc:  # noqa: BLE001 - a missing share must never take the service down
            result["error"] = f"{type(exc).__name__}: {exc}"
        self.last = result
        return result

    def status(self) -> dict[str, Any]:
        others = [{"os": p.name, "files": len(list(p.iterdir()))} for p in self._foreign_dirs()]
        return {"dir": str(self.root), "host_os": self.host_os, "available": self.root.is_dir(), "other_installs": others,
                "last_sync": self.last or None}


# ---------------------------------------------------------------- whole-database import (one-off / --import-runs-from)

def import_runs_db(dest: Path | str, source: Path | str, default_os: str = "Windows") -> dict[str, Any]:
    """Copy inference history from another install's `chatbot.sqlite3` into this one. The source is opened
    read-only and never modified; runs whose id already exists here are skipped (so it is safe to repeat);
    imported rows are tagged with the OS they came from (a source row that carries no OS predates capture and
    is marked `default_os` with `origin_os_inferred = 1`). Rows are only ever added."""
    dest, source = Path(dest), Path(source)
    if not source.is_file():
        return {"ok": False, "error": f"source database not found: {source}"}
    if dest.exists() and dest.resolve() == source.resolve():
        return {"ok": False, "error": "source and destination are the same file"}
    connection = sqlite3.connect(dest, timeout=60, isolation_level=None)
    connection.execute("PRAGMA foreign_keys = OFF")  # tool_events may point at projects/sessions imported in the same pass
    counts: dict[str, int] = {}
    try:
        connection.execute("ATTACH DATABASE ? AS src", (source.resolve().as_uri() + "?mode=ro",))
        have = {r[0] for r in connection.execute("SELECT name FROM src.sqlite_master WHERE type = 'table'")}
        if "runs" not in have:
            return {"ok": False, "error": "source has no runs table: not an inference run database"}

        def common(table: str, drop: tuple[str, ...] = ()) -> list[str]:
            mine = [r[1] for r in connection.execute(f"PRAGMA main.table_info({table})")]
            theirs = {r[1] for r in connection.execute(f"PRAGMA src.table_info({table})")}
            return [c for c in mine if c in theirs and c not in drop]

        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("CREATE TEMP TABLE new_runs AS SELECT id FROM src.runs WHERE id NOT IN (SELECT id FROM main.runs)")
            for table in ("projects", "agent_sessions", "coding_tasks", "clients", "deployments"):
                if table in have:
                    cols = ", ".join(common(table))
                    counts[table] = connection.execute(
                        f"INSERT OR IGNORE INTO main.{table} ({cols}) SELECT {cols} FROM src.{table}").rowcount
            cols = common("runs", ("origin_os", "origin_os_inferred"))
            names, picks = ", ".join(cols), ", ".join(f"s.{c}" for c in cols)
            has_host = "host_os_json" in cols
            os_expr = ("COALESCE(CASE WHEN json_valid(s.host_os_json) THEN json_extract(s.host_os_json, '$.system') END, ?)"
                       if has_host else "?")
            inferred = ("CASE WHEN json_valid(s.host_os_json) AND json_extract(s.host_os_json, '$.system') IS NOT NULL THEN 0 ELSE 1 END"
                        if has_host else "1")
            counts["runs"] = connection.execute(
                f"INSERT INTO main.runs ({names}, origin_os, origin_os_inferred) "
                f"SELECT {picks}, {os_expr}, {inferred} FROM src.runs s WHERE s.id IN (SELECT id FROM new_runs)",
                (default_os,)).rowcount
            if "metric_samples" in have:
                cols = ", ".join(common("metric_samples", ("id",)))
                counts["metric_samples"] = connection.execute(
                    f"INSERT INTO main.metric_samples ({cols}) SELECT {cols} FROM src.metric_samples "
                    "WHERE run_id IN (SELECT id FROM new_runs)").rowcount
            if "tool_events" in have:
                cols = common("tool_events", ("id",))
                names, picks = ", ".join(cols), ", ".join(f"s.{c}" for c in cols)
                counts["tool_events"] = connection.execute(
                    f"INSERT INTO main.tool_events ({names}) SELECT {picks} FROM src.tool_events s WHERE NOT EXISTS ("
                    "SELECT 1 FROM main.tool_events d WHERE d.project_id IS s.project_id AND d.occurred_at IS s.occurred_at "
                    "AND d.tool_type IS s.tool_type AND d.run_id IS s.run_id AND d.session_id IS s.session_id)").rowcount
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            connection.execute("ROLLBACK")
            return {"ok": False, "error": f"import rolled back, nothing changed: {exc}"}
    finally:
        connection.close()
    return {"ok": True, "source": str(source), "imported": counts}


def import_usage_db(dest: Path | str, source: Path | str) -> dict[str, Any]:
    """Add another install's per-project daily usage rows (`project_usage.sqlite3`). A (project, connection type,
    day) row already here is kept as it is; the source is opened read-only."""
    dest, source = Path(dest), Path(source)
    if not source.is_file() or not dest.is_file():
        return {"ok": False, "error": f"usage database not found: {source if not source.is_file() else dest}"}
    connection = sqlite3.connect(dest, timeout=60, isolation_level=None)
    try:
        connection.execute("ATTACH DATABASE ? AS src", (source.resolve().as_uri() + "?mode=ro",))
        if not connection.execute("SELECT 1 FROM src.sqlite_master WHERE type='table' AND name='project_usage'").fetchone():
            return {"ok": False, "error": "source has no project_usage table"}
        mine = [r[1] for r in connection.execute("PRAGMA main.table_info(project_usage)")]
        theirs = {r[1] for r in connection.execute("PRAGMA src.table_info(project_usage)")}
        cols = ", ".join(c for c in mine if c in theirs)
        added = connection.execute(f"INSERT OR IGNORE INTO main.project_usage ({cols}) SELECT {cols} FROM src.project_usage").rowcount
    finally:
        connection.close()
    return {"ok": True, "source": str(source), "imported": {"project_usage": added}}
