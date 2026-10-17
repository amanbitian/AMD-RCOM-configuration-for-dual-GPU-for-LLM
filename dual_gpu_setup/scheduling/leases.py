"""GPU leases: exclusive, expiring claims on one or more lanes.

A multi-lane request is all-or-nothing (one IMMEDIATE transaction). `lease_lanes.lane` is the
primary key, so SQLite itself rejects two active claims on a lane. Renew/release need the secret
token returned at acquire time, and process termination re-verifies PID identity (start token)
so an expired lease can never kill a recycled PID.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

from dual_gpu_setup.hostos import HostOS, ProcessIdentity, get_host

SCHEMA = """
CREATE TABLE IF NOT EXISTS leases (
    id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    kind TEXT NOT NULL,
    token TEXT NOT NULL,
    lanes_json TEXT NOT NULL,
    state TEXT NOT NULL,
    ttl_s REAL NOT NULL,
    acquired_at REAL NOT NULL,
    renewed_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    released_at REAL,
    pid INTEGER,
    pid_start_token INTEGER,
    exe TEXT
);
CREATE INDEX IF NOT EXISTS idx_leases_holder ON leases(holder, state);
CREATE TABLE IF NOT EXISTS lease_lanes (
    lane TEXT PRIMARY KEY,
    lease_id TEXT NOT NULL REFERENCES leases(id) ON DELETE CASCADE
);
"""


class LeaseError(RuntimeError):
    def __init__(self, message: str, status: int = 409, conflicts: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.status = status
        self.conflicts = conflicts or []


class LeaseStore:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time, host: HostOS | None = None):
        self.path = Path(path)
        self.clock = clock
        self._host = host
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as connection:
            connection.executescript(SCHEMA)

    @property
    def host(self) -> HostOS:
        return self._host or get_host()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: explicit BEGIN IMMEDIATE below, so acquire is a single write txn.
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _db(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _expire(connection: sqlite3.Connection, now: float) -> list[str]:
        ids = [r["id"] for r in connection.execute(
            "SELECT id FROM leases WHERE state = 'active' AND expires_at <= ?", (now,)
        ).fetchall()]
        for lease_id in ids:
            connection.execute("UPDATE leases SET state = 'expired' WHERE id = ?", (lease_id,))
            connection.execute("DELETE FROM lease_lanes WHERE lease_id = ?", (lease_id,))
        return ids

    def _present(self, row: sqlite3.Row, include_token: bool = False) -> dict[str, Any]:
        lease = {
            "id": row["id"], "holder": row["holder"], "kind": row["kind"], "state": row["state"],
            "lanes": json.loads(row["lanes_json"]), "ttl_s": row["ttl_s"],
            "acquired_at": row["acquired_at"], "renewed_at": row["renewed_at"],
            "expires_at": row["expires_at"], "released_at": row["released_at"],
            "pid": row["pid"], "exe": row["exe"],
            "remaining_s": max(0.0, row["expires_at"] - self.clock()) if row["state"] == "active" else 0.0,
        }
        if include_token:
            lease["token"] = row["token"]
        return lease

    def _row(self, connection: sqlite3.Connection, lease_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM leases WHERE id = ?", (lease_id,)).fetchone()
        if row is None:
            raise LeaseError("Lease not found.", 404)
        return row

    def _claim(self, connection: sqlite3.Connection, lease_id: str, lanes: list[str], now: float) -> None:
        self._expire(connection, now)
        marks = ",".join("?" * len(lanes))
        taken = connection.execute(
            f"SELECT ll.lane, l.holder, l.expires_at FROM lease_lanes ll JOIN leases l ON l.id = ll.lease_id "
            f"WHERE ll.lane IN ({marks}) AND ll.lease_id != ?", (*lanes, lease_id),
        ).fetchall()
        if taken:
            conflicts = [{"lane": r["lane"], "holder": r["holder"], "expires_at": r["expires_at"]} for r in taken]
            detail = ", ".join(f"{c['lane']} held by {c['holder']}" for c in conflicts)
            raise LeaseError(f"GPU lane(s) already leased: {detail}.", 409, conflicts)
        connection.executemany("INSERT OR IGNORE INTO lease_lanes (lane, lease_id) VALUES (?, ?)",
                               [(lane, lease_id) for lane in lanes])

    # ------------------------------------------------------------------ public API

    def acquire(
        self,
        holder: str,
        lanes: Iterable[str],
        kind: str = "training",
        ttl_s: float = 300.0,
        pid: int | None = None,
    ) -> dict[str, Any]:
        lanes = sorted(set(lanes))
        holder = holder.strip()
        if not holder or not lanes:
            raise LeaseError("holder and at least one lane are required.", 400)
        if ttl_s <= 0 or ttl_s > 7 * 86400:
            raise LeaseError("ttl_s must be between 0 and 7 days.", 400)
        identity = self.host.process.identity(pid) if pid else None
        now = self.clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._expire(connection, now)
                existing = connection.execute(
                    "SELECT * FROM leases WHERE holder = ? AND state = 'active'", (holder,)
                ).fetchone()
                if existing is not None:
                    if json.loads(existing["lanes_json"]) != lanes:
                        raise LeaseError(f"{holder} already holds a lease on different lanes.", 409)
                    connection.execute(
                        "UPDATE leases SET renewed_at = ?, expires_at = ?, ttl_s = ? WHERE id = ?",
                        (now, now + ttl_s, ttl_s, existing["id"]),
                    )
                    row = self._row(connection, existing["id"])
                else:
                    lease_id = uuid.uuid4().hex
                    connection.execute(
                        "INSERT INTO leases (id, holder, kind, token, lanes_json, state, ttl_s, acquired_at,"
                        " renewed_at, expires_at, pid, pid_start_token, exe) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (lease_id, holder, kind, secrets.token_urlsafe(24), json.dumps(lanes), "active", ttl_s,
                         now, now, now + ttl_s, identity.pid if identity else None,
                         identity.start_token if identity else None, identity.exe if identity else None),
                    )
                    self._claim(connection, lease_id, lanes, now)
                    row = self._row(connection, lease_id)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            return self._present(row, include_token=True)
        finally:
            connection.close()

    def _authorize(self, row: sqlite3.Row, token: str | None) -> None:
        if token is None or not secrets.compare_digest(str(token), row["token"]):
            raise LeaseError("Invalid lease token.", 403)

    def renew(self, lease_id: str, token: str | None, ttl_s: float | None = None, *, internal: bool = False) -> dict[str, Any]:
        """Extend a lease. An expired lease is revived only if its lanes are still free."""
        now = self.clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._row(connection, lease_id)
                if not internal:
                    self._authorize(row, token)
                if row["state"] == "released":
                    raise LeaseError("Lease was released.", 409)
                ttl = float(ttl_s) if ttl_s else row["ttl_s"]
                if row["state"] == "expired" or row["expires_at"] <= now:
                    self._claim(connection, lease_id, json.loads(row["lanes_json"]), now)
                else:
                    self._expire(connection, now)
                connection.execute(
                    "UPDATE leases SET state = 'active', renewed_at = ?, expires_at = ?, ttl_s = ? WHERE id = ?",
                    (now, now + ttl, ttl, lease_id),
                )
                row = self._row(connection, lease_id)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            return self._present(row)
        finally:
            connection.close()

    def release(self, lease_id: str, token: str | None, *, internal: bool = False) -> dict[str, Any]:
        with self._db() as connection:
            row = self._row(connection, lease_id)
            if not internal:
                self._authorize(row, token)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM lease_lanes WHERE lease_id = ?", (lease_id,))
            connection.execute(
                "UPDATE leases SET state = 'released', released_at = ? WHERE id = ? AND state != 'released'",
                (self.clock(), lease_id),
            )
            connection.execute("COMMIT")
            return self._present(self._row(connection, lease_id))

    def active_holder(self, holder: str) -> dict[str, Any] | None:
        """Most recent lease for a holder (any state), without the token."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT * FROM leases WHERE holder = ? ORDER BY acquired_at DESC LIMIT 1", (holder,)
            ).fetchone()
            return None if row is None else self._present(row)

    def renew_holder(self, holder: str) -> str:
        """Heartbeat path: renew by holder name. Returns 'renewed', 'none' (no such lease) or
        'lost' (expired and its lanes were taken by someone else)."""
        lease = self.active_holder(holder)
        if lease is None or lease["state"] == "released":
            return "none"
        try:
            self.renew(lease["id"], None, internal=True)
            return "renewed"
        except LeaseError:
            return "lost"

    def release_holder(self, holder: str) -> bool:
        lease = self.active_holder(holder)
        if lease is None or lease["state"] == "released":
            return False
        self.release(lease["id"], None, internal=True)
        return True

    def list_active(self) -> list[dict[str, Any]]:
        now = self.clock()
        with self._db() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire(connection, now)
            rows = connection.execute(
                "SELECT * FROM leases WHERE state = 'active' ORDER BY acquired_at"
            ).fetchall()
            connection.execute("COMMIT")
            return [self._present(r) for r in rows]

    def lane_holders(self) -> dict[str, dict[str, Any]]:
        holders: dict[str, dict[str, Any]] = {}
        for lease in self.list_active():
            for lane in lease["lanes"]:
                holders[lane] = {"holder": lease["holder"], "kind": lease["kind"], "lease_id": lease["id"],
                                 "expires_at": lease["expires_at"]}
        return holders

    def conflicts(self, lanes: Iterable[str], ignore_holder: str | None = None) -> list[dict[str, Any]]:
        wanted = set(lanes)
        return [
            {"lane": lane, **info} for lane, info in self.lane_holders().items()
            if lane in wanted and info["holder"] != ignore_holder
        ]

    def terminate_holder(self, lease_id: str, grace_s: float = 10.0) -> dict[str, Any]:
        """Preempt a lease: stop its process (only if the PID still has the recorded start
        token) and release the lanes. Never kills a process the lease did not start with."""
        with self._db() as connection:
            row = self._row(connection, lease_id)
        outcome = "no_process"
        if row["pid"] is not None and row["pid_start_token"] is not None:
            identity = ProcessIdentity(row["pid"], row["pid_start_token"], row["exe"])
            outcome = "terminated" if self.host.terminate_verified(identity, grace_s) else "pid_mismatch_or_gone"
        released = self.release(lease_id, None, internal=True)
        return {"process": outcome, "lease": released}
