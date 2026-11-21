"""Merge training telemetry written on another OS / machine into this machine's database, by run id.

Why: the Windows and the Ubuntu installs each keep their own local ``training.sqlite3``.  Run ids are random
UUIDs, so databases from different installs never collide and can be combined losslessly.  Sharing one *live* SQLite
file across operating systems (NTFS from Linux, WAL side files, Windows Fast Startup) is how databases get corrupted,
so this copies instead: the other database is opened **read-only** and is never modified.

    python -m dual_gpu_setup.training.merge --dest ~/dual-gpu-data/chat_runs/training.sqlite3 \
        --from /mnt/windows/Projects/Dual_gpu_setup/chat_runs/training.sqlite3 [--dry-run]
    python app.py --config dual_gpu.toml --import-from /path/to/other/training.sqlite3     # imports once at startup

Rules
  * a run whose id already exists locally is left untouched (never overwritten, so merging is idempotent);
  * each imported run is copied with all of its rows in ONE transaction (all or nothing);
  * per-GPU ``hardware_samples`` and ``system_samples`` carry no run id: the samples inside each imported run's time
    window are copied, skipping any (pci_bus, ts) already present;
  * global tables that are safe to share (hardware profiles, benchmarks) are merged with INSERT OR IGNORE;
  * leases, queued eval jobs and counters are machine-local live state and are NOT copied;
  * file paths inside imported rows (checkpoints, artifacts, configs) are kept verbatim: they are history, and an
    ``F:\\...`` path will not resolve on Linux.  Numbers (loss, throughput, power, temperatures) are unaffected;
  * timestamps are epoch seconds, so both installs must have correct system clocks for a combined timeline to line up;
  * every imported telemetry row carries the OS it was recorded on (`host_os`): a source row without one (an older
    schema, or recorded before rows were tagged) gets the source's OS. Before the first foreign row arrives, the
    destination's own untagged rows are tagged with the destination's OS (NULL -> value only), so the dashboard's
    OS filter can tell the two apart. Nothing else already stored is changed.
"""
from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .db import SCHEMA_VERSION

# child tables with an AUTOINCREMENT `id` (dropped on copy so the destination assigns its own)
AUTO_ID_CHILDREN = ("training_steps", "training_events", "training_checkpoints", "training_artifacts", "checkpoint_evals")
# child tables keyed by (run_id, ...) with no surrogate id
KEYED_CHILDREN = ("training_ranks", "training_rank_steps", "training_example_stats")
SAMPLE_PAD_S = 5.0


@dataclass
class MergeReport:
    source: str
    dest: str
    dry_run: bool = False
    runs_imported: list[dict[str, Any]] = field(default_factory=list)
    runs_skipped: list[dict[str, Any]] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)
    rows_tagged: dict[str, int] = field(default_factory=dict)
    source_os: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_rows(self, table: str, n: int) -> None:
        if n and n > 0:
            self.rows[table] = self.rows.get(table, 0) + n

    def summary(self) -> str:
        verb = "would import" if self.dry_run else "imported"
        lines = [f"{verb} {len(self.runs_imported)} run(s) from {self.source}; "
                 f"{len(self.runs_skipped)} already present"]
        for r in self.runs_imported:
            lines.append(f"  + {r['name']}  [{r['id'][:8]}]  status={r['status']}  steps={r['last_step']}")
        if self.source_os:
            lines.append(f"  recorded on: {self.source_os}")
        if self.rows:
            lines.append("  rows: " + ", ".join(f"{t}={n}" for t, n in sorted(self.rows.items())))
        if self.rows_tagged:
            lines.append("  local rows tagged with this OS: " + ", ".join(f"{t}={n}" for t, n in sorted(self.rows_tagged.items())))
        lines += [f"  warning: {w}" for w in self.warnings] + [f"  ERROR: {e}" for e in self.errors]
        return "\n".join(lines)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA {schema}.table_info({_q(table)})")]


def _common(conn: sqlite3.Connection, table: str, drop: Iterable[str] = ()) -> list[str]:
    """Columns present in BOTH databases (so an older source schema still merges), minus ``drop``."""
    drop = set(drop)
    dest = _columns(conn, "main", table)
    src = set(_columns(conn, "src", table))
    return [c for c in dest if c in src and c not in drop]


def _has_table(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    return bool(conn.execute(f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?", (table,)).fetchone())


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _copy_columns(conn: sqlite3.Connection, table: str, drop: Iterable[str], row_os: Optional[str]) -> tuple[list[str], list[str]]:
    """Destination columns and the matching `src` expressions. `host_os` is filled with `row_os` when the source
    row has none (an older source schema without the column, or a row recorded before rows were tagged)."""
    cols = _common(conn, table, drop=drop)
    picks = ["s." + _q(c) for c in cols]
    if row_os and "host_os" in _columns(conn, "main", table):
        if "host_os" in cols:
            picks[cols.index("host_os")] = f"COALESCE(NULLIF(s.host_os, ''), {_lit(row_os)})"
        else:
            cols.append("host_os")
            picks.append(_lit(row_os))
    return cols, picks


def _source_os(conn: sqlite3.Connection) -> Optional[str]:
    """The OS the source database was recorded on, when its rows say so unambiguously."""
    names = {r[0] for r in conn.execute(
        "SELECT DISTINCT json_extract(host_os_json, '$.system') FROM src.training_runs "
        "WHERE json_valid(host_os_json) AND json_extract(host_os_json, '$.system') IS NOT NULL")} \
        if "host_os_json" in _columns(conn, "src", "training_runs") else set()
    if not names and _has_table(conn, "src", "hardware_samples") and "host_os" in _columns(conn, "src", "hardware_samples"):
        names = {r[0] for r in conn.execute("SELECT DISTINCT host_os FROM src.hardware_samples WHERE host_os IS NOT NULL")}
    return names.pop() if len(names) == 1 else None


TAGGED_TABLES = ("hardware_samples", "host_samples", "hardware_alerts", "hardware_events")


def _tag_local_rows(conn: sqlite3.Connection, dest_os: str, report: MergeReport) -> None:
    """Untagged telemetry in the destination was recorded by this install (rows from another OS always arrive
    tagged), so give it this install's OS before foreign rows are added. Only NULL / '' is replaced. If runs from
    another OS are already here (merged by an older version, which did not tag), untagged rows are ambiguous
    and are left alone."""
    tables = [t for t in TAGGED_TABLES if _has_table(conn, "main", t) and "host_os" in _columns(conn, "main", t)]
    if not tables:
        return
    foreign = conn.execute(
        "SELECT COUNT(*) FROM main.training_runs WHERE json_valid(host_os_json) "
        "AND json_extract(host_os_json, '$.system') IS NOT NULL AND json_extract(host_os_json, '$.system') != ?",
        (dest_os,)).fetchone()[0] if "host_os_json" in _columns(conn, "main", "training_runs") else 0
    try:
        conn.execute("BEGIN IMMEDIATE")
        for table in tables:
            if foreign:
                if conn.execute(f"SELECT 1 FROM main.{_q(table)} WHERE host_os IS NULL OR host_os = '' LIMIT 1").fetchone():
                    report.warnings.append(f"{table}: untagged rows left untagged (runs from another OS are already "
                                           "merged, so their origin cannot be told apart)")
                continue
            n = conn.execute(f"UPDATE main.{_q(table)} SET host_os = ? WHERE host_os IS NULL OR host_os = ''", (dest_os,)).rowcount
            if n:
                report.rows_tagged[table] = report.rows_tagged.get(table, 0) + n
        conn.execute("COMMIT")
    except sqlite3.Error as exc:
        conn.execute("ROLLBACK")
        report.errors.append(f"local rows not tagged (rolled back): {exc}")


def _open(dest: Path, source: Path, report: MergeReport, dest_readonly: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(dest.resolve().as_uri() + ("?mode=ro" if dest_readonly else ""), uri=True, timeout=30,
                           isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    src_uri = source.resolve().as_uri()
    try:
        conn.execute("ATTACH DATABASE ? AS src", (src_uri + "?mode=ro",))
        conn.execute("SELECT count(*) FROM src.sqlite_master").fetchone()
    except sqlite3.OperationalError:
        # e.g. a read-only filesystem holding a WAL database whose -shm cannot be created: fall back to a
        # frozen read that ignores the WAL (rows committed but not yet checkpointed in the source are not seen)
        try:
            conn.execute("DETACH DATABASE src")
        except sqlite3.OperationalError:
            pass
        conn.execute("ATTACH DATABASE ? AS src", (src_uri + "?immutable=1",))
        conn.execute("SELECT count(*) FROM src.sqlite_master").fetchone()
        report.warnings.append("source opened immutable (read-only medium): any data still in its -wal file was not "
                               "read; stop the other install's service (or copy it with SQLite's backup) for a full import")
    return conn


def _import_run(conn: sqlite3.Connection, run: sqlite3.Row, report: MergeReport, source_os: Optional[str] = None) -> None:
    rid = run["id"]
    counts: dict[str, int] = {}

    cols = _common(conn, "training_runs")
    c = ", ".join(map(_q, cols))
    counts["training_runs"] = conn.execute(
        f"INSERT INTO main.training_runs ({c}) SELECT {c} FROM src.training_runs WHERE id = ?", (rid,)).rowcount

    for table in AUTO_ID_CHILDREN + KEYED_CHILDREN:
        if not _has_table(conn, "src", table):
            continue
        cols = _common(conn, table, drop=("id",) if table in AUTO_ID_CHILDREN else ())
        c = ", ".join(map(_q, cols))
        counts[table] = conn.execute(
            f"INSERT OR IGNORE INTO main.{_q(table)} ({c}) SELECT {c} FROM src.{_q(table)} WHERE run_id = ?", (rid,)).rowcount

    if _has_table(conn, "src", "training_notes"):
        cols = _common(conn, "training_notes", drop=("id",))
        c = ", ".join(map(_q, cols))
        counts["training_notes"] = conn.execute(
            f"INSERT INTO main.training_notes ({c}) SELECT {c} FROM src.training_notes WHERE run_id = ?", (rid,)).rowcount

    start = run["started_at"]
    end = (run["finished_at"] or run["last_heartbeat_at"] or run["last_step_at"] or start) + SAMPLE_PAD_S
    # imported samples carry the OS the run was recorded on
    system = None
    if "host_os_json" in run.keys() and run["host_os_json"]:
        try:
            system = (json.loads(run["host_os_json"]) or {}).get("system")
        except ValueError:
            system = None
    if _has_table(conn, "src", "hardware_samples"):
        cols, picks = _copy_columns(conn, "hardware_samples", ("id",), system or source_os)
        c = ", ".join(map(_q, cols))
        s = ", ".join(picks)
        counts["hardware_samples"] = conn.execute(
            f"INSERT INTO main.hardware_samples ({c}) SELECT {s} FROM src.hardware_samples s "
            f"WHERE s.ts BETWEEN ? AND ? AND NOT EXISTS (SELECT 1 FROM main.hardware_samples d "
            f"WHERE d.pci_bus = s.pci_bus AND d.ts = s.ts)", (start, end)).rowcount
    if _has_table(conn, "src", "system_samples"):
        cols = _common(conn, "system_samples")
        c = ", ".join(map(_q, cols))
        counts["system_samples"] = conn.execute(
            f"INSERT OR IGNORE INTO main.system_samples ({c}) SELECT {c} FROM src.system_samples WHERE ts BETWEEN ? AND ?",
            (start, end)).rowcount
    for table, n in counts.items():
        report.add_rows(table, n)


def _count_run(conn: sqlite3.Connection, run: sqlite3.Row, report: MergeReport) -> None:
    rid = run["id"]
    report.add_rows("training_runs", 1)
    for table in AUTO_ID_CHILDREN + KEYED_CHILDREN + ("training_notes",):
        if _has_table(conn, "src", table):
            report.add_rows(table, conn.execute(f"SELECT count(*) FROM src.{_q(table)} WHERE run_id = ?", (rid,)).fetchone()[0])
    start = run["started_at"]
    end = (run["finished_at"] or run["last_heartbeat_at"] or run["last_step_at"] or start) + SAMPLE_PAD_S
    if _has_table(conn, "src", "hardware_samples"):
        report.add_rows("hardware_samples", conn.execute(
            "SELECT count(*) FROM src.hardware_samples s WHERE s.ts BETWEEN ? AND ? AND NOT EXISTS ("
            "SELECT 1 FROM main.hardware_samples d WHERE d.pci_bus = s.pci_bus AND d.ts = s.ts)", (start, end)).fetchone()[0])


def merge_training_db(dest: Path | str, source: Path | str, *, dry_run: bool = False,
                      all_telemetry: bool = False, source_os: Optional[str] = None,
                      dest_os: Optional[str] = None) -> MergeReport:
    """Copy runs that exist in ``source`` but not in ``dest``. ``source`` is only ever opened read-only.

    ``source_os`` names the OS the source was recorded on (default: read from its runs); ``dest_os`` is this
    install's OS, used to tag the destination's own untagged telemetry before foreign rows arrive (skipped when
    not given)."""
    dest, source = Path(dest), Path(source)
    report = MergeReport(source=str(source), dest=str(dest), dry_run=dry_run)
    if not source.is_file():
        report.errors.append(f"source database not found: {source}")
        return report
    if dest.exists() and dest.resolve() == source.resolve():
        report.errors.append("source and destination are the same file")
        return report

    scratch: Optional[Path] = None
    if not dry_run:
        # create / upgrade the destination schema with the app's own code so columns and indexes always match
        from .store import TrainingStore
        TrainingStore(dest).close()
        dest_for_open = dest
    elif dest.exists():
        dest_for_open = dest                      # dry run: opened read-only below, never upgraded or written
    else:
        import tempfile
        from .store import TrainingStore
        report.warnings.append("destination does not exist yet: dry run assumes it is empty")
        scratch = Path(tempfile.mkdtemp(prefix="dgpu_merge_dry_")) / "dest.sqlite3"
        TrainingStore(scratch).close()
        dest_for_open = scratch

    conn = _open(dest_for_open, source, report, dest_readonly=dry_run and scratch is None)
    conn.row_factory = sqlite3.Row
    try:
        src_version = conn.execute("PRAGMA src.user_version").fetchone()[0]
        if src_version > SCHEMA_VERSION:
            report.errors.append(f"source schema v{src_version} is newer than this install (v{SCHEMA_VERSION}): "
                                 "update Dual-GPU Studio first")
            return report
        if not _has_table(conn, "src", "training_runs"):
            report.errors.append("source has no training_runs table: not a training telemetry database")
            return report
        report.source_os = source_os = source_os or _source_os(conn)
        if source_os is None:
            report.warnings.append("could not tell which OS the source was recorded on: its untagged telemetry rows "
                                   "are imported without an OS (pass source_os / --source-os)")
        if dest_os and not dry_run:
            _tag_local_rows(conn, dest_os, report)
            if not report.ok:
                return report
        have = {r[0] for r in conn.execute("SELECT id FROM main.training_runs")}
        runs = conn.execute("SELECT * FROM src.training_runs ORDER BY started_at").fetchall()
        for run in runs:
            info = {"id": run["id"], "name": run["name"], "status": run["status"], "last_step": run["last_step"],
                    "started_at": run["started_at"]}
            if run["id"] in have:
                report.runs_skipped.append(info)
                continue
            if dry_run:
                report.runs_imported.append(info)
                _count_run(conn, run, report)
                continue
            try:
                conn.execute("BEGIN IMMEDIATE")
                _import_run(conn, run, report, source_os)
                conn.execute("COMMIT")
                report.runs_imported.append(info)
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                report.errors.append(f"run {run['name']} [{run['id'][:8]}] not imported (rolled back): {exc}")
        if not dry_run:
            _merge_global(conn, report)
            _merge_telemetry(conn, report, all_telemetry, source_os)
    finally:
        conn.close()
        if scratch is not None:
            import shutil
            shutil.rmtree(scratch.parent, ignore_errors=True)
    return report


# natural keys used to skip rows already present (`IS` treats NULL = NULL)
_TELEMETRY_KEYS = {
    "host_samples": ("host_os", "ts"),
    "hardware_alerts": ("kind", "pci_bus", "metric", "started_at", "host_os"),
    "hardware_events": ("ts", "kind", "lane", "host_os"),
}
_ALL_SAMPLE_KEYS = {"hardware_samples": ("pci_bus", "ts"), "system_samples": ("ts",)}


def _merge_telemetry(conn: sqlite3.Connection, report: MergeReport, all_samples: bool,
                     source_os: Optional[str] = None) -> None:
    """Hardware log tables that belong to no training run: host samples, saved alerts and events always; and,
    with `all_samples`, every GPU/system sample (inference-time telemetry) so both installs show the union.
    Rows already present are skipped; nothing in the destination is changed."""
    plan = dict(_TELEMETRY_KEYS)
    if all_samples:
        plan.update(_ALL_SAMPLE_KEYS)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for table, key in plan.items():
            if not (_has_table(conn, "src", table) and _has_table(conn, "main", table)):
                continue
            cols, picks = _copy_columns(conn, table, ("id",), source_os)
            key = tuple(k for k in key if k in cols)
            c = ", ".join(map(_q, cols))
            s = ", ".join(picks)
            # compare on the value as it will be stored (a tagged host_os), so repeating a merge adds nothing
            same = " AND ".join(f"d.{_q(k)} IS {picks[cols.index(k)]}" for k in key)
            report.add_rows(table, conn.execute(
                f"INSERT INTO main.{_q(table)} ({c}) SELECT {s} FROM src.{_q(table)} s "
                f"WHERE NOT EXISTS (SELECT 1 FROM main.{_q(table)} d WHERE {same})").rowcount)
        conn.execute("COMMIT")
    except sqlite3.Error as exc:
        conn.execute("ROLLBACK")
        report.errors.append(f"telemetry tables not merged (rolled back): {exc}")


def _merge_global(conn: sqlite3.Connection, report: MergeReport) -> None:
    """Shared, machine-independent tables: INSERT OR IGNORE on their natural keys."""
    plan = (("hardware_profiles", ("id",)), ("benchmarks", ()), ("benchmark_trials", ()))
    try:
        conn.execute("BEGIN IMMEDIATE")
        for table, drop in plan:
            if not (_has_table(conn, "src", table) and _has_table(conn, "main", table)):
                continue
            cols = _common(conn, table, drop=drop)
            c = ", ".join(map(_q, cols))
            report.add_rows(table, conn.execute(
                f"INSERT OR IGNORE INTO main.{_q(table)} ({c}) SELECT {c} FROM src.{_q(table)}").rowcount)
        conn.execute("COMMIT")
    except sqlite3.Error as exc:
        conn.execute("ROLLBACK")
        report.errors.append(f"shared tables not merged (rolled back): {exc}")


def merge_many(dest: Path | str, sources: Iterable[Path | str], *, dry_run: bool = False,
               all_telemetry: bool = False, dest_os: Optional[str] = None) -> list[MergeReport]:
    return [merge_training_db(dest, s, dry_run=dry_run, all_telemetry=all_telemetry, dest_os=dest_os) for s in sources]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="this install's training.sqlite3 (created if missing)")
    ap.add_argument("--from", dest="sources", action="append", required=True, metavar="DB",
                    help="training.sqlite3 of the other install; repeatable. Opened read-only, never modified.")
    ap.add_argument("--dry-run", action="store_true", help="report what would be imported; change nothing")
    ap.add_argument("--all-telemetry", action="store_true",
                    help="also import every GPU/system sample, not only those inside imported runs' windows")
    ap.add_argument("--dest-os", default=platform.system(),
                    help="OS the destination was recorded on, used to tag its own untagged rows (default: this OS)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    reports = merge_many(args.dest, args.sources, dry_run=args.dry_run, all_telemetry=args.all_telemetry,
                         dest_os=args.dest_os or None)
    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2, default=str))
    else:
        for r in reports:
            print(r.summary())
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
