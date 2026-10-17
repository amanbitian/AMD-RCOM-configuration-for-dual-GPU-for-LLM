"""Shared SQLite plumbing for the training stores: a small connection pool and idempotent schema
upgrades.

`ThreadingHTTPServer` handles every request on a new thread, so thread-local connections would be
rebuilt per request. A pool of reusable connections (WAL, so readers never block the writer) avoids
re-opening the file and re-running PRAGMAs on each call.
"""

from __future__ import annotations

import queue
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3


def _new_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -16000")
    return connection


class _Lease:
    """`with db.connect() as c:` commits on success, rolls back on error, then returns the
    connection to the pool."""

    __slots__ = ("pool", "connection")

    def __init__(self, pool: "Database", connection: sqlite3.Connection):
        self.pool = pool
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        return self.connection

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.pool._release(self.connection)


class Database:
    def __init__(self, path: Path, pool_size: int = 6):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._idle: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue()
        self._pool_size = pool_size

    def connect(self) -> _Lease:
        try:
            connection = self._idle.get_nowait()
        except queue.Empty:
            connection = _new_connection(self.path)
        return _Lease(self, connection)

    def _release(self, connection: sqlite3.Connection) -> None:
        if self._idle.qsize() < self._pool_size:
            self._idle.put(connection)
        else:
            connection.close()

    def close(self) -> None:
        while True:
            try:
                self._idle.get_nowait().close()
            except queue.Empty:
                return

    def ensure_schema(self, script: str, additions: dict[str, dict[str, str]] | None = None) -> None:
        """Create missing tables and add missing columns. Safe to run on every start and on
        databases created by earlier versions."""
        with self.connect() as connection:
            connection.executescript(script)
            for table, columns in (additions or {}).items():
                existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
                for column, definition in columns.items():
                    if column not in existing:
                        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def placeholders(count: int) -> str:
    return ",".join("?" * count)


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]
