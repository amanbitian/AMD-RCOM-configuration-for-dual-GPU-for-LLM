"""Data-protection helpers: verified backups, row counts and a "nothing is ever deleted" policy.

Principle: no telemetry, log or run row is deleted by the service. Old data is *copied* to compressed
archives (see `archive.py`), backups are only ever added, and every schema change is additive.

`backup_sqlite` uses SQLite's online backup API, so it is safe on a live WAL database, then runs
`PRAGMA integrity_check` on the copy. A backup that fails verification is removed (it is our own,
unverified copy; the source is untouched) and the failure is reported to the caller.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Tables whose row counts are compared before and after a migration. Names that do not exist in a
# given database are skipped, so the same helper serves the run, training and chat databases.
WATCHED_TABLES = (
    "runs", "deployments", "metric_samples", "clients", "projects", "agent_sessions", "coding_tasks", "tool_events",
    "training_runs", "training_steps", "training_events", "training_checkpoints", "hardware_samples",
    "system_samples", "host_samples", "hardware_alerts", "checkpoint_evals",
)


@dataclass
class BackupResult:
    source: str
    path: str | None = None
    ok: bool = False
    skipped: str | None = None
    error: str | None = None
    size_bytes: int | None = None
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "path": self.path, "ok": self.ok, "skipped": self.skipped,
                "error": self.error, "size_bytes": self.size_bytes, "counts": self.counts}


def table_counts(path: Path | str, tables: tuple[str, ...] = WATCHED_TABLES) -> dict[str, int]:
    """Row count per existing watched table, read through a read-only connection."""
    path = Path(path)
    if not path.exists():
        return {}
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        existing = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {name: connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                for name in tables if name in existing}
    finally:
        connection.close()


def integrity_ok(path: Path | str) -> tuple[bool, str]:
    connection = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()
    return result == "ok", str(result)


def backup_sqlite(source: Path | str, backups_dir: Path | str, label: str = "") -> BackupResult:
    """Copy `source` to `<backups_dir>/<name>.<label|timestamp>.bak` and verify it.

    Never overwrites an existing backup and never deletes anything but its own failed copy."""
    source, backups_dir = Path(source), Path(backups_dir)
    result = BackupResult(source=str(source))
    if not source.exists():
        result.skipped = "database does not exist yet"
        result.ok = True
        return result
    backups_dir.mkdir(parents=True, exist_ok=True)
    tag = label or time.strftime("%Y%m%dT%H%M%S")
    target = backups_dir / f"{source.name}.{tag}.bak"
    if target.exists():
        result.path, result.ok, result.skipped = str(target), True, "backup with this label already exists"
        return result
    needed = source.stat().st_size * 2
    free = shutil.disk_usage(backups_dir).free
    if free < needed:
        result.error = f"not enough free space for a backup ({free} bytes free, {needed} needed)"
        return result
    partial = target.with_suffix(target.suffix + ".partial")
    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True, timeout=60)
    try:
        destination = sqlite3.connect(partial)
        try:
            source_connection.backup(destination)
            # a backup is one self-contained file: no -wal / -shm side files to lose or leave behind
            destination.execute("PRAGMA journal_mode = DELETE")
        finally:
            destination.close()
    except sqlite3.Error as exc:
        partial.unlink(missing_ok=True)
        result.error = f"backup failed: {exc}"
        return result
    finally:
        source_connection.close()
    good, detail = integrity_ok(partial)
    if not good:
        partial.unlink(missing_ok=True)
        result.error = f"backup integrity check failed: {detail}"
        return result
    partial.replace(target)
    result.path, result.ok = str(target), True
    result.size_bytes = target.stat().st_size
    result.counts = table_counts(target)
    return result


def backup_once_per_version(source: Path | str, backups_dir: Path | str, version: str) -> BackupResult:
    """One verified backup per (database, version): taken before the first start of a new version
    runs its migrations. Re-starting the same version does not copy a multi-GB file again."""
    return backup_sqlite(source, backups_dir, label=f"pre-{version}")


_FOREIGN_FS = {"ntfs", "ntfs3", "fuseblk", "exfat", "vfat", "msdos"}


def storage_warnings(path: Path | str, mounts_file: str = "/proc/mounts") -> list[str]:
    """A live SQLite database must sit on the install's own native disk. On Linux, warn when the data
    directory is on an NTFS/FAT partition (the shared Windows disk): that is where databases get corrupted."""
    try:
        resolved = str(Path(path).resolve())
        best = ("", "")
        for line in Path(mounts_file).read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mount = parts[1].replace("\\040", " ")
            if (resolved == mount or resolved.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best[0]):
                best = (mount, parts[2])
    except OSError:
        return []
    if best[1] in _FOREIGN_FS:
        return [f"{path} is on a {best[1]} partition ({best[0]}). Keep the databases on this OS's native disk and share "
                "history with --exchange-dir instead; a live SQLite file shared between Windows and Linux can be corrupted."]
    return []


def verify_counts_not_lower(before: dict[str, int], after: dict[str, int]) -> list[str]:
    """Problems found when comparing counts around a migration: any table that lost rows."""
    return [f"{table}: {count} rows before, {after.get(table, 0)} after"
            for table, count in before.items() if after.get(table, 0) < count]
