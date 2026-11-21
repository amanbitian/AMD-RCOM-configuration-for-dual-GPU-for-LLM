"""Compressed, append-only daily telemetry archives. Nothing is ever deleted from the database.

Each UTC day is written to `<archive_dir>/telemetry-YYYY-MM-DD.<os>.jsonl.gz`: one JSON object per
line (`{"t": "<table>", "row": {...}}`) ending with a trailer `{"t": "_end", "counts": {...}}`. A file
is trusted only if its trailer matches the rows actually read back, so a torn write is never
mistaken for a complete archive. The archive is a safety copy, a portable export format and the
exchange format between a Windows and a Linux install (plain files are safe on a shared NTFS
partition, a live SQLite file is not).
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

FORMAT_VERSION = 1

# table -> (timestamp column, natural-key columns used to skip rows that already exist on import)
TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "hardware_samples": ("ts", ("pci_bus", "ts", "host_os")),
    "host_samples": ("ts", ("host_os", "ts")),
    "system_samples": ("ts", ("ts",)),
    "hardware_alerts": ("started_at", ("kind", "pci_bus", "metric", "started_at", "host_os")),
    "hardware_events": ("ts", ("ts", "kind", "lane", "host_os")),
}


def day_bounds(day: str) -> tuple[float, float]:
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")


def _safe_os(name: str | None) -> str:
    return "".join(c for c in (name or "unknown").lower() if c.isalnum()) or "unknown"


def archive_name(day: str, host_os: str) -> str:
    return f"telemetry-{day}.{_safe_os(host_os)}.jsonl.gz"


def read_archive(path: Path | str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int] | None]:
    """Rows by table plus the trailer counts (None when the trailer is missing: a torn file)."""
    rows: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] | None = None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("t") == "_end":
                    counts = item.get("counts") or {}
                elif "row" in item:
                    rows.setdefault(item["t"], []).append(item["row"])
    except (OSError, EOFError, ValueError):
        return rows, None
    return rows, counts


def archive_is_complete(path: Path | str) -> bool:
    rows, counts = read_archive(path)
    return counts is not None and all(len(rows.get(table, ())) == n for table, n in counts.items())


class HardwareArchive:
    """Writes and reads daily archives for one install. `connect` is the TrainingStore's pool lease."""

    def __init__(self, directory: Path | str, connect: Callable[[], Any], host_os: str, clock: Callable[[], float] = time.time):
        self.directory = Path(directory)
        self.connect = connect
        self.host_os = host_os
        self.clock = clock
        self.last_error: str | None = None
        self.last_run: dict[str, Any] | None = None

    # ------------------------------------------------------------------ writing

    def _days_with_data(self, connection: sqlite3.Connection) -> list[str]:
        days: set[str] = set()
        for table, (ts_col, _key) in TABLES.items():
            try:
                low, high = connection.execute(f"SELECT MIN({ts_col}), MAX({ts_col}) FROM {table}").fetchone()
            except sqlite3.Error:
                continue
            if low is None:
                continue
            cursor = datetime.fromtimestamp(low, UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            last = datetime.fromtimestamp(high, UTC)
            while cursor <= last:
                days.add(cursor.strftime("%Y-%m-%d"))
                cursor += timedelta(days=1)
        return sorted(days)

    def _day_counts(self, connection: sqlite3.Connection, day: str) -> dict[str, int]:
        start, end = day_bounds(day)
        counts = {}
        for table, (ts_col, _key) in TABLES.items():
            try:
                counts[table] = connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {ts_col} >= ? AND {ts_col} < ?", (start, end)).fetchone()[0]
            except sqlite3.Error:
                counts[table] = 0
        return {table: n for table, n in counts.items() if n}

    def archive_day(self, day: str) -> dict[str, Any]:
        """Write (or refresh) one day's archive. Skipped when a complete file already holds every row.
        The DB is only read; the file is written to a temp name and renamed once verified."""
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / archive_name(day, self.host_os)
        start, end = day_bounds(day)
        with self.connect() as connection:
            counts = self._day_counts(connection, day)
            if not counts:
                return {"day": day, "status": "empty"}
            if target.exists():
                existing, trailer = read_archive(target)
                if trailer is not None and trailer == counts and all(len(existing.get(t, ())) == n for t, n in counts.items()):
                    return {"day": day, "status": "current", "rows": sum(counts.values())}
            partial = target.with_name(target.name + ".partial")
            written: dict[str, int] = {}
            with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as handle:
                handle.write(json.dumps({"t": "_start", "format": FORMAT_VERSION, "day": day, "host_os": self.host_os}) + "\n")
                for table, (ts_col, _key) in TABLES.items():
                    if table not in counts:
                        continue
                    cursor = connection.execute(
                        f"SELECT * FROM {table} WHERE {ts_col} >= ? AND {ts_col} < ? ORDER BY {ts_col}", (start, end))
                    names = [d[0] for d in cursor.description]
                    for values in cursor:
                        row = dict(zip(names, values))
                        row.pop("id", None)  # the importing database assigns its own surrogate ids
                        if "host_os" in row and not row["host_os"]:
                            row["host_os"] = self.host_os
                        handle.write(json.dumps({"t": table, "row": row}, separators=(",", ":")) + "\n")
                        written[table] = written.get(table, 0) + 1
                handle.write(json.dumps({"t": "_end", "counts": written}) + "\n")
        if not archive_is_complete(partial) or written != counts:
            partial.unlink(missing_ok=True)
            return {"day": day, "status": "failed", "error": "verification after write did not match the database"}
        partial.replace(target)
        return {"day": day, "status": "written", "rows": sum(written.values()), "bytes": target.stat().st_size}

    def archive_all(self) -> dict[str, Any]:
        """Archive every day that has data, including today (refreshed when more rows arrived)."""
        results: list[dict[str, Any]] = []
        try:
            with self.connect() as connection:
                days = self._days_with_data(connection)
            for day in days:
                results.append(self.archive_day(day))
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 - archiving is best effort and must never stop the service
            self.last_error = f"{type(exc).__name__}: {exc}"
        summary = {"ran_at": self.clock(), "days": len(results),
                   "written": sum(1 for r in results if r["status"] == "written"),
                   "failed": [r for r in results if r["status"] == "failed"], "error": self.last_error}
        self.last_run = summary
        return summary

    # ------------------------------------------------------------------ reading

    def files(self) -> list[Path]:
        return sorted(self.directory.glob("telemetry-*.jsonl.gz")) if self.directory.is_dir() else []

    def list_days(self) -> list[dict[str, Any]]:
        out = []
        for path in self.files():
            parts = path.name.split(".")
            out.append({"file": path.name, "day": parts[0].removeprefix("telemetry-"), "host_os": parts[1] if len(parts) > 2 else None,
                        "bytes": path.stat().st_size})
        return out

    def iter_rows(self, table: str, start: float, end: float) -> Iterator[dict[str, Any]]:
        """Archived rows of `table` in [start, end) from every archive file whose day overlaps."""
        ts_col = TABLES[table][0]
        first, last = day_of(start), day_of(max(start, end - 1e-6))
        for path in self.files():
            day = path.name.split(".")[0].removeprefix("telemetry-")
            if not first <= day <= last:
                continue
            rows, counts = read_archive(path)
            if counts is None:
                continue
            for row in rows.get(table, ()):
                if start <= (row.get(ts_col) or 0) < end:
                    yield row


def export_rows(rows: Iterable[dict[str, Any]], fmt: str) -> tuple[str, str]:
    """Serialise rows as CSV or JSON lines. Returns (content_type, body)."""
    rows = list(rows)
    if fmt == "jsonl":
        return "application/x-ndjson", "".join(json.dumps(r, default=str) + "\n" for r in rows)
    import csv
    import io

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return "text/csv; charset=utf-8", buffer.getvalue()


def import_archive(path: Path | str, connection: sqlite3.Connection, default_os: str | None = None) -> dict[str, int]:
    """Insert rows from an archive file that this database does not already have (matched on each
    table's natural key). Never updates or deletes existing rows. Returns inserted counts."""
    rows, counts = read_archive(path)
    if counts is None or any(len(rows.get(t, ())) != n for t, n in counts.items()):
        raise ValueError(f"{path} is incomplete or damaged; not imported")
    inserted: dict[str, int] = {}
    for table, items in rows.items():
        if table not in TABLES or not items:
            continue
        existing_columns = [r[1] for r in connection.execute(f"PRAGMA table_info({table})")]
        ts_col, key = TABLES[table]
        key = tuple(k for k in key if k in existing_columns)
        for item in items:
            if "host_os" in existing_columns and not item.get("host_os"):
                item["host_os"] = default_os or ""
        stamps = [i.get(ts_col) or 0 for i in items]
        # keys already present in the file's time window, fetched once (a per-row probe would scan)
        present = {tuple("" if v is None else v for v in r) for r in connection.execute(
            f"SELECT {', '.join(key)} FROM {table} WHERE {ts_col} BETWEEN ? AND ?", (min(stamps), max(stamps)))}
        for item in items:
            identity = tuple("" if item.get(k) is None else item.get(k) for k in key)
            if identity in present:
                continue
            present.add(identity)
            data = {k: v for k, v in item.items() if k in existing_columns}
            connection.execute(
                f"INSERT INTO {table} ({', '.join(data)}) VALUES ({','.join('?' * len(data))})", tuple(data.values()))
            inserted[table] = inserted.get(table, 0) + 1
    return inserted
