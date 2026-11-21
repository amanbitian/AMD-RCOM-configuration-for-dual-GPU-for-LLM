"""The hardware log: GPU + host telemetry saved only while a GPU is in use, plus saved alerts.

"In use" means any GPU lane that has a loaded model, a running request, or a training / evaluation job
holding it. Nothing is written when idle, and the integrated GPU (no lane) is never logged. Samples carry
their OS, loaded model and training run, so a Windows and a Linux history combine into one self-describing
timeline. Alerts are saved with start, end and peak instead of being recomputed on request. Nothing here
deletes anything.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from dual_gpu_setup.archive import HardwareArchive, TABLES, export_rows

HOST_FIELDS = ("cpu_pct", "cpu_temp_c", "cpu_power_w", "ram_used_pct", "swap_used_bytes", "mem_pressure_pct", "ram_temp_c",
               "ssd_temp_c", "coolant_temp_c", "pump_rpm", "cpu_fan_rpm", "case_fan_rpm", "disk_free_bytes")

# (scope, metric column, kind, severity, threshold, recover-below margin)
DEFAULT_ALERTS = (
    ("gpu", "temp_junction_c", "gpu_junction_hot", "critical", 100.0, 3.0),
    ("gpu", "temp_memory_c", "gpu_memory_hot", "critical", 90.0, 3.0),
    ("gpu", "temp_edge_c", "gpu_edge_hot", "warning", 95.0, 3.0),
    ("host", "cpu_temp_c", "cpu_hot", "warning", 95.0, 3.0),
    ("host", "ssd_temp_c", "ssd_hot", "warning", 70.0, 3.0),
    ("host", "ram_temp_c", "ram_hot", "warning", 85.0, 3.0),
    ("host", "coolant_temp_c", "coolant_hot", "warning", 50.0, 3.0),
)
CLEAR_AFTER_SAMPLES = 3     # consecutive in-range samples before an alert is closed
TOUCH_EVERY_S = 5.0         # throttle how often an open alert's peak/last value is rewritten


@dataclass
class _Open:
    row_id: int
    peak: float
    last_written: float
    clear_streak: int = 0


@dataclass
class AlertTracker:
    """Opens an alert when a reading crosses its threshold, tracks the peak, closes it after the reading
    has been back below (threshold - margin) for CLEAR_AFTER_SAMPLES samples. Persists every transition."""

    connect: Callable[[], Any]
    host_os: str
    rules: tuple = DEFAULT_ALERTS
    _open: dict[tuple, _Open] = field(default_factory=dict)

    def recover_orphans(self, now: float) -> int:
        """Alerts left open by a crash: close them at the last time they were seen."""
        with self.connect() as c:
            return c.execute("UPDATE hardware_alerts SET ended_at = COALESCE(last_seen_at, started_at) "
                             "WHERE ended_at IS NULL AND host_os = ?", (self.host_os,)).rowcount

    def _insert(self, c: sqlite3.Connection, kind: str, severity: str, scope: str, bus: str | None, lane: str | None,
                metric: str, threshold: float | None, now: float, value: float, context: str | None) -> int:
        return c.execute(
            "INSERT INTO hardware_alerts (kind, severity, scope, pci_bus, lane, metric, threshold, started_at, last_seen_at,"
            " peak_value, last_value, host_os, context_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (kind, severity, scope, bus, lane, metric, threshold, now, now, value, value, self.host_os, context)).lastrowid

    def observe(self, key: tuple, active: bool, value: float | None, now: float, *, kind: str, severity: str, scope: str,
                bus: str | None, lane: str | None, metric: str, threshold: float | None, recovered: bool,
                context: str | None = None) -> None:
        state = self._open.get(key)
        with self.connect() as c:
            if state is None:
                if active and value is not None:
                    self._open[key] = _Open(self._insert(c, kind, severity, scope, bus, lane, metric, threshold, now, value, context),
                                            value, now)
                return
            if active:
                state.clear_streak = 0
                if value is not None:
                    state.peak = max(state.peak, value)
                if now - state.last_written >= TOUCH_EVERY_S:
                    c.execute("UPDATE hardware_alerts SET peak_value=?, last_value=?, last_seen_at=?, samples=samples+1 WHERE id=?",
                              (state.peak, value, now, state.row_id))
                    state.last_written = now
            elif recovered:
                state.clear_streak += 1
                if state.clear_streak >= CLEAR_AFTER_SAMPLES:
                    c.execute("UPDATE hardware_alerts SET peak_value=?, last_value=?, last_seen_at=?, ended_at=? WHERE id=?",
                              (state.peak, value, now, now, state.row_id))
                    del self._open[key]

    def close_all(self, now: float) -> None:
        with self.connect() as c:
            for state in self._open.values():
                c.execute("UPDATE hardware_alerts SET peak_value=?, last_seen_at=?, ended_at=? WHERE id=?",
                          (state.peak, now, now, state.row_id))
        self._open.clear()

    def open_alerts(self) -> list[int]:
        return [s.row_id for s in self._open.values()]


class HardwareLog:
    """Glue between the GPU sampler callbacks and storage. `lane_context()` returns
    {lane_key: {"model": ..., "kind": "inference|training|evaluation", "run_id": ...}} for every lane in use."""

    def __init__(self, store: Any, host_os: str, lane_context: Callable[[], dict[str, dict[str, Any]]],
                 host_sensors: Any = None, system_latest: Callable[[], dict[str, Any] | None] | None = None,
                 interval_s: float = 1.0, clock: Callable[[], float] = time.time,
                 alert_rules: tuple = DEFAULT_ALERTS):
        self.store = store
        self.host_os = host_os
        self.lane_context = lane_context
        self.host_sensors = host_sensors
        self.system_latest = system_latest or (lambda: None)
        self.interval_s = interval_s
        self.clock = clock
        self.alerts = AlertTracker(store.connect, host_os, alert_rules)
        self.alerts.recover_orphans(clock())
        self.rules = alert_rules
        self.logging = False
        self.rows_written = 0
        self.write_errors = 0
        self.last_error: str | None = None
        self._lock = threading.Lock()

    # -------------------------------------------------------------- recording

    def on_gpu_samples(self, samples: list[dict[str, Any]]) -> None:
        """Sampler callback (1 Hz). Writes only GPUs in use; idle ticks write nothing."""
        try:
            self._record(samples)
        except Exception as exc:  # noqa: BLE001 - logging must never break sampling
            self.write_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _record(self, samples: list[dict[str, Any]]) -> None:
        in_use = self.lane_context()
        used = [s for s in samples if s.get("lane") and s["lane"] in in_use]
        if not used:
            if self.logging:
                self.logging = False
                self.alerts.close_all(self.clock())  # the GPU went idle: end alerts, never leave them dangling
            return
        self.logging = True
        now = self.clock()
        models = sorted({str(in_use[s["lane"]].get("model")) for s in used if in_use[s["lane"]].get("model")})
        rows = []
        for s in used:
            ctx = in_use[s["lane"]]
            rows.append({**s, "host_os": self.host_os, "model": ctx.get("model"),
                         "context": {k: v for k, v in ctx.items() if k != "model" and v is not None}})
        with self._lock:
            self.store.add_hardware_samples(rows)
            host = self._host_row(now, used, models, in_use)
            self._write_host(host)
            self.rows_written += len(rows)
        self._evaluate(now, used, host)

    def _host_row(self, now: float, used: list[dict[str, Any]], models: list[str], in_use: dict[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {k: None for k in HOST_FIELDS}
        unavailable: dict[str, str] = {}
        if self.host_sensors is not None:
            read, why = self.host_sensors.read()
            values.update({k: v for k, v in read.items() if k in values})
            unavailable.update(why)
        system = self.system_latest() or {}
        for key in ("cpu_pct", "ram_used_pct", "swap_used_bytes"):
            if system.get(key) is not None:
                values[key] = system[key]
        gpu_w = [s.get("power_w") for s in used if s.get("power_w") is not None]
        gpu_power = sum(gpu_w) if gpu_w else None
        est = None if gpu_power is None else gpu_power + (values.get("cpu_power_w") or 0.0)
        if gpu_power is not None and values.get("cpu_power_w") is None:
            unavailable["est_system_power_w"] = "GPU power only: CPU power unavailable (an estimate, not wall power)"
        return {"ts": now, "host_os": self.host_os, **values, "gpu_power_w": gpu_power, "est_system_power_w": est,
                "model": ",".join(models) or None, "lanes_json": _json(sorted(in_use)), "context_json": None,
                "unavailable_json": _json(unavailable)}

    def _write_host(self, row: dict[str, Any]) -> None:
        with self.store.connect() as c:
            names = list(row)
            c.execute(f"INSERT OR REPLACE INTO host_samples ({', '.join(names)}) VALUES ({','.join('?' * len(names))})",
                      tuple(row.values()))
            # the existing per-run readers use system_samples; keep it fed while logging
            if row.get("cpu_pct") is not None or row.get("ram_used_pct") is not None:
                c.execute("INSERT OR REPLACE INTO system_samples (ts, cpu_pct, ram_used_pct, swap_used_bytes) VALUES (?,?,?,?)",
                          (row["ts"], row["cpu_pct"], row["ram_used_pct"], row["swap_used_bytes"]))

    # ----------------------------------------------------------------- alerts

    def _evaluate(self, now: float, used: list[dict[str, Any]], host: dict[str, Any]) -> None:
        for scope, metric, kind, severity, threshold, margin in self.rules:
            if scope == "gpu":
                for s in used:
                    value = s.get(metric)
                    self._observe((scope, s["pci_bus"], metric), value, threshold, margin, now, kind, severity,
                                  s["pci_bus"], s.get("lane"), metric)
            else:
                self._observe((scope, "host", metric), host.get(metric), threshold, margin, now, kind, severity,
                              None, None, metric)
        for s in used:  # throttling and a degraded slot link are state, not thresholds
            throttle = s.get("throttle_status")
            self.alerts.observe(("gpu", s["pci_bus"], "throttle"), bool(throttle), None if throttle is None else float(throttle),
                                now, kind="gpu_throttle", severity="warning", scope="gpu", bus=s["pci_bus"], lane=s.get("lane"),
                                metric="throttle_status", threshold=None, recovered=throttle is not None and not throttle)
            width, max_width = s.get("pcie_slot_width"), _to_int(s.get("pcie_slot_max_width"))
            degraded = width is not None and max_width is not None and width < max_width
            self.alerts.observe(("gpu", s["pci_bus"], "pcie"), degraded, None if width is None else float(width), now,
                                kind="pcie_link_degraded", severity="warning", scope="gpu", bus=s["pci_bus"], lane=s.get("lane"),
                                metric="pcie_slot_width", threshold=None if max_width is None else float(max_width),
                                recovered=width is not None and not degraded)

    def _observe(self, key: tuple, value: float | None, threshold: float, margin: float, now: float, kind: str, severity: str,
                 bus: str | None, lane: str | None, metric: str) -> None:
        active = value is not None and value >= threshold
        self.alerts.observe(key, active, value, now, kind=kind, severity=severity, scope=key[0], bus=bus, lane=lane,
                            metric=metric, threshold=threshold, recovered=value is not None and value < threshold - margin)

    # ----------------------------------------------------------------- events

    def record_event(self, kind: str, severity: str, payload: dict[str, Any], lane: str | None = None) -> None:
        """Kernel faults, server crashes and similar: saved whenever they happen, in use or not."""
        try:
            with self.store.connect() as c:
                c.execute("INSERT INTO hardware_events (ts, kind, severity, lane, host_os, payload_json) VALUES (?,?,?,?,?,?)",
                          (self.clock(), kind, severity, lane, self.host_os, _json(payload)))
        except Exception as exc:  # noqa: BLE001
            self.write_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        self.alerts.close_all(self.clock())

    # ----------------------------------------------------------------- reading

    def status(self) -> dict[str, Any]:
        return {"logging": self.logging, "rows_written": self.rows_written, "write_errors": self.write_errors,
                "last_error": self.last_error, "open_alerts": len(self.alerts.open_alerts()), "host_os": self.host_os,
                "host_sensors": type(self.host_sensors).__name__ if self.host_sensors is not None else None}

    def alerts_between(self, start: float | None, end: float | None, open_only: bool = False, limit: int = 500,
                       host_os: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM hardware_alerts WHERE 1=1", []
        if host_os:
            sql, args = sql + " AND host_os = ?", [host_os]
        if start is not None:
            sql, args = sql + " AND COALESCE(ended_at, ?) >= ?", [*args, self.clock(), start]
        if end is not None:
            sql, args = sql + " AND started_at <= ?", [*args, end]
        if open_only:
            sql += " AND ended_at IS NULL"
        with self.store.connect() as c:
            return [dict(r) for r in c.execute(sql + " ORDER BY started_at DESC LIMIT ?", [*args, max(1, min(limit, 5000))])]

    def events_between(self, start: float | None, end: float | None, limit: int = 500,
                       host_os: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM hardware_events WHERE ts >= ? AND ts <= ?", [start or 0, end or self.clock() + 1]
        if host_os:
            sql, args = sql + " AND host_os = ?", [*args, host_os]
        with self.store.connect() as c:
            return [dict(r) for r in c.execute(sql + " ORDER BY ts DESC LIMIT ?", [*args, max(1, min(limit, 5000))])]

    def summary(self, start: float, end: float, lane: str | None = None, host_os: str | None = None) -> dict[str, Any]:
        """Min / average / peak temperatures, average and peak watts, energy, peak fan speed and throttle time
        per GPU, plus host peaks, over [start, end]. NULL where no sample reported a value. `host_os` keeps only
        rows recorded on that OS ("Windows" / "Linux")."""
        interval = self.interval_s
        gpu_where, args = "ts BETWEEN ? AND ?", [start, end]
        if lane:
            gpu_where, args = gpu_where + " AND lane = ?", [*args, lane]
        os_where, os_args = ("", []) if not host_os else (" AND host_os = ?", [host_os])
        gpu_where, args = gpu_where + os_where, [*args, *os_args]
        with self.store.connect() as c:
            gpus = {}
            for r in c.execute(
                f"""SELECT pci_bus, MAX(lane) lane, COUNT(*) samples, MIN(ts) first_ts, MAX(ts) last_ts,
                    MIN(temp_junction_c) min_junction, AVG(temp_junction_c) avg_junction, MAX(temp_junction_c) max_junction,
                    MIN(temp_memory_c) min_memory, AVG(temp_memory_c) avg_memory, MAX(temp_memory_c) max_memory,
                    MIN(temp_edge_c) min_edge, AVG(temp_edge_c) avg_edge, MAX(temp_edge_c) max_edge,
                    AVG(power_w) avg_power, MAX(power_w) max_power, SUM(power_w) power_sum, MAX(fan_rpm) max_fan,
                    SUM(CASE WHEN throttle_status IS NOT NULL AND throttle_status != 0 THEN 1 ELSE 0 END) throttled,
                    SUM(CASE WHEN throttle_status IS NOT NULL THEN 1 ELSE 0 END) throttle_known
                    FROM hardware_samples WHERE {gpu_where} GROUP BY pci_bus""", args):
                gpus[r["pci_bus"]] = {
                    "lane": r["lane"], "samples": r["samples"], "first_ts": r["first_ts"], "last_ts": r["last_ts"],
                    "temp_junction_c": {"min": r["min_junction"], "avg": r["avg_junction"], "max": r["max_junction"]},
                    "temp_memory_c": {"min": r["min_memory"], "avg": r["avg_memory"], "max": r["max_memory"]},
                    "temp_edge_c": {"min": r["min_edge"], "avg": r["avg_edge"], "max": r["max_edge"]},
                    "power_w": {"avg": r["avg_power"], "max": r["max_power"]},
                    "energy_wh": None if r["power_sum"] is None else r["power_sum"] * interval / 3600.0,
                    "peak_fan_rpm": r["max_fan"],
                    "throttled_seconds": None if not r["throttle_known"] else r["throttled"] * interval,
                }
            host = c.execute(
                """SELECT COUNT(*) samples, MAX(cpu_pct) max_cpu, MAX(cpu_temp_c) max_cpu_temp, AVG(cpu_power_w) avg_cpu_power,
                   MAX(cpu_power_w) max_cpu_power, MAX(ram_used_pct) max_ram, MAX(ssd_temp_c) max_ssd, MAX(ram_temp_c) max_ram_temp,
                   MAX(coolant_temp_c) max_coolant, MAX(pump_rpm) max_pump, MAX(est_system_power_w) max_system_power,
                   AVG(est_system_power_w) avg_system_power, SUM(est_system_power_w) system_sum, MIN(disk_free_bytes) min_disk_free
                   FROM host_samples WHERE ts BETWEEN ? AND ?""" + os_where, (start, end, *os_args)).fetchone()
        return {"start": start, "end": end, "gpus": gpus,
                "host": {**dict(host), "est_system_energy_wh": None if host["system_sum"] is None else host["system_sum"] * interval / 3600.0}}

    def history(self, start: float, end: float, max_points: int = 600, host_os: str | None = None) -> dict[str, Any]:
        """Evenly thinned GPU and host series for charts. The stored rows are never thinned or removed."""
        max_points = max(10, min(int(max_points), 5000))
        out: dict[str, Any] = {"start": start, "end": end, "gpus": {}, "host": []}
        os_where, os_args = ("", []) if not host_os else (" AND host_os = ?", [host_os])
        with self.store.connect() as c:
            for bus, n in c.execute("SELECT pci_bus, COUNT(*) FROM hardware_samples WHERE ts BETWEEN ? AND ?"
                                    + os_where + " GROUP BY pci_bus", (start, end, *os_args)).fetchall():
                stride = max(1, -(-n // max_points))
                rows = c.execute(
                    """SELECT * FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY ts) rn FROM hardware_samples
                       WHERE pci_bus = ? AND ts BETWEEN ? AND ?""" + os_where + """) WHERE (rn - 1) % ? = 0 ORDER BY ts""",
                    (bus, start, end, *os_args, stride)).fetchall()
                out["gpus"][bus] = {"lane": rows[-1]["lane"] if rows else None, "stride": stride, "samples": n,
                                    "points": [{k: r[k] for k in r.keys() if k not in ("id", "rn")} for r in rows]}
            n = c.execute("SELECT COUNT(*) FROM host_samples WHERE ts BETWEEN ? AND ?" + os_where,
                          (start, end, *os_args)).fetchone()[0]
            if n:
                stride = max(1, -(-n // max_points))
                out["host_stride"] = stride
                out["host"] = [dict(r) for r in c.execute(
                    """SELECT * FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY ts) rn FROM host_samples
                       WHERE ts BETWEEN ? AND ?""" + os_where + """) WHERE (rn - 1) % ? = 0 ORDER BY ts""",
                    (start, end, *os_args, stride))]
        return out

    def data_range(self, host_os: str | None = None) -> dict[str, Any]:
        """First and last saved GPU sample, overall and per OS, so the dashboard can jump to where the data is."""
        with self.store.connect() as c:
            per_os = {(r[0] or "untagged"): {"first_ts": r[1], "last_ts": r[2], "samples": r[3]} for r in c.execute(
                "SELECT host_os, MIN(ts), MAX(ts), COUNT(*) FROM hardware_samples GROUP BY host_os")}
        picked = [v for k, v in per_os.items() if not host_os or k == host_os]
        return {"os": host_os, "per_os": per_os,
                "first_ts": min((v["first_ts"] for v in picked), default=None),
                "last_ts": max((v["last_ts"] for v in picked), default=None),
                "samples": sum(v["samples"] for v in picked)}

    def export(self, start: float, end: float, fmt: str = "csv", table: str = "hardware_samples",
               archive: HardwareArchive | None = None, host_os: str | None = None) -> tuple[str, str]:
        """CSV or JSON lines for a time range, from the database plus any archived rows it lacks."""
        if table not in TABLES:
            raise ValueError(f"unknown table {table!r}; choose one of {', '.join(TABLES)}")
        ts_col, key = TABLES[table]
        if "host_os" not in key:
            host_os = None  # system_samples has no OS column: it is exported whole
        os_where, os_args = ("", []) if not host_os else (" AND host_os = ?", [host_os])
        with self.store.connect() as c:
            rows = [dict(r) for r in c.execute(
                f"SELECT * FROM {table} WHERE {ts_col} >= ? AND {ts_col} < ?{os_where} ORDER BY {ts_col}", (start, end, *os_args))]
        if archive is not None:
            seen = {tuple("" if r.get(k) is None else r.get(k) for k in key) for r in rows}
            rows += [r for r in archive.iter_rows(table, start, end)
                     if tuple("" if r.get(k) is None else r.get(k) for k in key) not in seen
                     and (not host_os or r.get("host_os") == host_os)]
            rows.sort(key=lambda r: r.get(ts_col) or 0)
        for r in rows:
            r.pop("id", None)
        return export_rows(rows, fmt)


def _json(value: Any) -> str | None:
    import json

    return None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _to_int(value: Any) -> int | None:
    try:
        return int(str(value).split()[0])
    except (ValueError, IndexError):
        return None
