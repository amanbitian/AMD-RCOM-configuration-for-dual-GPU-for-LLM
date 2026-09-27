from __future__ import annotations

import argparse
import ctypes
import ipaddress
import json
import os
import re
import signal
import socket
import sqlite3
import threading
import time
import uuid
import webbrowser
from collections import deque
from dataclasses import asdict, replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse

from dual_gpu_setup.config import AppConfig, LaneConfig, ModelConfig, load_config
from dual_gpu_setup.lmstudio import model_size_gb, resolve_model_path
from dual_gpu_setup.orchestrator import ctx_for, lane_capacity_gb
from dual_gpu_setup.server import LlamaServerProcess, gpu_process_metrics


APP_VERSION = "0.1.0"
DEFAULT_REASONING_PRESETS = {"off": 0, "low": 1024, "medium": 4096, "high": 8192}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def local_network_addresses() -> list[str]:
    """Return private IPv4 addresses that another LAN device can use."""
    candidates: set[str] = set()
    try:
        candidates.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            candidates.add(str(probe.getsockname()[0]))
    except OSError:
        pass
    addresses: list[str] = []
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.version == 4 and address.is_private and not address.is_loopback and not address.is_link_local:
            addresses.append(candidate)
    return sorted(addresses, key=lambda value: tuple(int(part) for part in value.split(".")))


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    def value(self) -> int:
        return (self.high << 32) | self.low


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("memory_load", ctypes.c_uint32),
        ("total_physical", ctypes.c_uint64),
        ("available_physical", ctypes.c_uint64),
        ("total_page_file", ctypes.c_uint64),
        ("available_page_file", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("available_virtual", ctypes.c_uint64),
        ("available_extended_virtual", ctypes.c_uint64),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("page_fault_count", ctypes.c_uint32),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


class ResourceSampler:
    """Collect lightweight Windows process/system metrics and GPU memory counters."""

    def __init__(self, manager: "DeploymentManager", interval_seconds: float = 2.0):
        self.manager = manager
        self.interval_seconds = interval_seconds
        self.samples: deque[dict[str, Any]] = deque(maxlen=3600)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._last_process_times: dict[int, tuple[int, float]] = {}
        self._last_system_times: tuple[int, int, int] | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                sample = self._sample()
                with self.lock:
                    self.samples.append(sample)
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.samples.append({"sampled_at": utc_now(), "error": str(exc), "servers": []})
            self.stop_event.wait(self.interval_seconds)

    def latest(self) -> dict[str, Any] | None:
        with self.lock:
            return dict(self.samples[-1]) if self.samples else None

    def capture(self) -> dict[str, Any]:
        sample = self._sample()
        with self.lock:
            self.samples.append(sample)
        return sample

    def since(self, monotonic_start: float) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(item) for item in self.samples if item.get("monotonic", 0.0) >= monotonic_start]

    def _sample(self) -> dict[str, Any]:
        now = time.monotonic()
        with self.manager.lock:
            handles = list(self.manager.handles.items())
        servers = []
        for target, handle in handles:
            process = handle.process
            if process.proc is None:
                continue
            pid = process.pid
            process_metrics = self._process_metrics(pid, now)
            gpu = gpu_process_metrics(pid)
            servers.append(
                {
                    "target": target,
                    "model": handle.model.name,
                    "lane_keys": [lane.key for lane in handle.lanes],
                    "pid": pid,
                    "alive": process.proc.poll() is None,
                    "cpu_pct": process_metrics.get("cpu_pct"),
                    "rss_bytes": process_metrics.get("rss_bytes"),
                    "private_bytes": process_metrics.get("private_bytes"),
                    "gpu_utilization_pct": gpu.get("utilization_pct") if gpu else None,
                    "dedicated_vram_bytes": int(gpu["dedicated_gb"] * 1024**3) if gpu else None,
                    "shared_gpu_memory_bytes": int(gpu["shared_gb"] * 1024**3) if gpu else None,
                }
            )
        return {
            "sampled_at": utc_now(),
            "monotonic": now,
            "system": self._system_metrics(),
            "servers": servers,
        }

    def _process_metrics(self, pid: int, now: float) -> dict[str, Any]:
        if os.name != "nt":
            return {}
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        process_query = 0x1000
        handle = kernel32.OpenProcess(process_query, False, pid)
        if not handle:
            return {}
        try:
            created, exited, kernel, user = _FileTime(), _FileTime(), _FileTime(), _FileTime()
            total_100ns: int | None = None
            if kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                total_100ns = kernel.value() + user.value()
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            memory_ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            cpu_pct = None
            previous = self._last_process_times.get(pid)
            if total_100ns is not None and previous:
                delta_cpu_seconds = (total_100ns - previous[0]) / 10_000_000
                delta_wall = max(now - previous[1], 0.001)
                cpu_pct = max(0.0, min(100.0, delta_cpu_seconds / delta_wall / (os.cpu_count() or 1) * 100.0))
            if total_100ns is not None:
                self._last_process_times[pid] = (total_100ns, now)
            return {
                "cpu_pct": round(cpu_pct, 2) if cpu_pct is not None else None,
                "rss_bytes": int(counters.working_set_size) if memory_ok else None,
                "private_bytes": int(counters.pagefile_usage) if memory_ok else None,
            }
        finally:
            kernel32.CloseHandle(handle)

    def _system_metrics(self) -> dict[str, Any]:
        if os.name != "nt":
            return {}
        kernel32 = ctypes.windll.kernel32
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        cpu_pct = None
        if kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            current = (idle.value(), kernel.value(), user.value())
            if self._last_system_times:
                idle_delta = current[0] - self._last_system_times[0]
                total_delta = (current[1] - self._last_system_times[1]) + (current[2] - self._last_system_times[2])
                if total_delta > 0:
                    cpu_pct = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
            self._last_system_times = current
        memory = _MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        memory_ok = kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        return {
            "cpu_pct": round(cpu_pct, 2) if cpu_pct is not None else None,
            "ram_total_bytes": int(memory.total_physical) if memory_ok else None,
            "ram_available_bytes": int(memory.available_physical) if memory_ok else None,
            "ram_used_bytes": int(memory.total_physical - memory.available_physical) if memory_ok else None,
            "ram_used_pct": float(memory.memory_load) if memory_ok else None,
            "pagefile_used_bytes": int(memory.total_page_file - memory.available_page_file) if memory_ok else None,
        }


class RunStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self.lock, self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS deployments (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    ready_at TEXT,
                    stopped_at TEXT,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    servers_json TEXT,
                    error_text TEXT
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    comparison_id TEXT,
                    deployment_id TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    target TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_path TEXT,
                    lane_keys_json TEXT NOT NULL,
                    device_json TEXT,
                    context_window INTEGER,
                    reasoning_budget INTEGER,
                    request_json TEXT NOT NULL,
                    configuration_json TEXT,
                    output_text TEXT,
                    reasoning_text TEXT,
                    finish_reason TEXT,
                    usage_json TEXT,
                    timing_json TEXT,
                    resources_json TEXT,
                    error_text TEXT,
                    backend_response_json TEXT,
                    FOREIGN KEY (deployment_id) REFERENCES deployments(id)
                );
                CREATE TABLE IF NOT EXISTS metric_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sampled_at TEXT NOT NULL,
                    monotonic_offset_ms REAL,
                    sample_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_comparison_id ON runs(comparison_id);
                CREATE INDEX IF NOT EXISTS idx_metric_samples_run_id ON metric_samples(run_id, id);
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
            migrations = {
                "deployment_id": "TEXT",
                "model_path": "TEXT",
                "device_json": "TEXT",
                "context_window": "INTEGER",
                "reasoning_budget": "INTEGER",
                "configuration_json": "TEXT",
                "finish_reason": "TEXT",
            }
            for name, data_type in migrations.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {data_type}")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_deployment_id ON runs(deployment_id)")

    def begin_deployment(self, deployment_id: str, mode: str, profile: dict[str, Any]) -> None:
        with self.lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO deployments (id, created_at, status, mode, profile_json)
                VALUES (?, ?, 'loading', ?, ?)
                """,
                (deployment_id, utc_now(), mode, json.dumps(profile, ensure_ascii=False)),
            )

    def update_deployment(
        self,
        deployment_id: str,
        status: str,
        servers: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        ready_at = utc_now() if status == "ready" else None
        stopped_at = utc_now() if status in {"stopped", "failed"} else None
        with self.lock, self.connect() as connection:
            connection.execute(
                """
                UPDATE deployments SET status = ?,
                    ready_at = COALESCE(ready_at, ?),
                    stopped_at = COALESCE(stopped_at, ?),
                    servers_json = COALESCE(?, servers_json),
                    error_text = COALESCE(?, error_text)
                WHERE id = ?
                """,
                (
                    status,
                    ready_at,
                    stopped_at,
                    json.dumps(servers, ensure_ascii=False) if servers is not None else None,
                    error,
                    deployment_id,
                ),
            )

    def begin_run(self, record: dict[str, Any]) -> None:
        with self.lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, comparison_id, deployment_id, created_at, status, mode, target,
                    model_name, model_path, lane_keys_json, device_json, context_window,
                    reasoning_budget, request_json, configuration_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record.get("comparison_id"),
                    record.get("deployment_id"),
                    record["created_at"],
                    "running",
                    record["mode"],
                    record["target"],
                    record["model_name"],
                    record.get("model_path"),
                    json.dumps(record["lane_keys"]),
                    json.dumps(record.get("device"), ensure_ascii=False),
                    record.get("context_window"),
                    record.get("reasoning_budget"),
                    json.dumps(record["request"], ensure_ascii=False),
                    json.dumps(record.get("configuration"), ensure_ascii=False),
                ),
            )

    def finish_run(self, run_id: str, fields: dict[str, Any]) -> None:
        values = {
            "finished_at": utc_now(),
            "status": fields.get("status", "completed"),
            "output_text": fields.get("output_text"),
            "reasoning_text": fields.get("reasoning_text"),
            "finish_reason": fields.get("finish_reason"),
            "usage_json": json.dumps(fields.get("usage"), ensure_ascii=False),
            "timing_json": json.dumps(fields.get("timing"), ensure_ascii=False),
            "resources_json": json.dumps(fields.get("resources"), ensure_ascii=False),
            "error_text": fields.get("error_text"),
            "backend_response_json": json.dumps(fields.get("backend_response"), ensure_ascii=False),
        }
        with self.lock, self.connect() as connection:
            connection.execute(
                """
                UPDATE runs SET finished_at = :finished_at, status = :status,
                    output_text = :output_text, reasoning_text = :reasoning_text,
                    finish_reason = :finish_reason,
                    usage_json = :usage_json, timing_json = :timing_json,
                    resources_json = :resources_json, error_text = :error_text,
                    backend_response_json = :backend_response_json
                WHERE id = :id
                """,
                {"id": run_id, **values},
            )

    def save_samples(self, run_id: str, samples: list[dict[str, Any]], monotonic_start: float) -> None:
        rows = [
            (
                run_id,
                sample.get("sampled_at") or utc_now(),
                round((float(sample.get("monotonic", monotonic_start)) - monotonic_start) * 1000, 3),
                json.dumps(sample, ensure_ascii=False),
            )
            for sample in samples
        ]
        if not rows:
            return
        with self.lock, self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO metric_samples (run_id, sampled_at, monotonic_offset_ms, sample_json)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode(row) for row in rows]

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self.lock, self.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            samples = connection.execute(
                "SELECT sampled_at, monotonic_offset_ms, sample_json FROM metric_samples WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        if row is None:
            return None
        item = self._decode(row)
        item["metric_samples"] = [
            {
                "sampled_at": sample["sampled_at"],
                "offset_ms": sample["monotonic_offset_ms"],
                "sample": json.loads(sample["sample_json"]),
            }
            for sample in samples
        ]
        return item

    def dashboard(self) -> dict[str, Any]:
        runs = self.recent(1000)

        def nums(path: tuple[str, str], selected: list[dict[str, Any]] = runs) -> list[float]:
            parent, child = path
            return [
                float(run[parent][child])
                for run in selected
                if isinstance(run.get(parent), dict) and run[parent].get(child) is not None
            ]

        def total(path: tuple[str, str], selected: list[dict[str, Any]] = runs) -> int:
            return int(sum(nums(path, selected)))

        def average(path: tuple[str, str], selected: list[dict[str, Any]] = runs) -> float | None:
            values = nums(path, selected)
            return round(sum(values) / len(values), 2) if values else None

        def percentile(path: tuple[str, str], percentile_value: float) -> float | None:
            values = sorted(nums(path))
            if not values:
                return None
            index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile_value)))
            return round(values[index], 2)

        completed = [run for run in runs if run.get("status") == "completed"]
        models: list[dict[str, Any]] = []
        for model_name in sorted({run["model_name"] for run in runs}):
            group = [run for run in runs if run["model_name"] == model_name]
            group_completed = [run for run in group if run.get("status") == "completed"]
            models.append(
                {
                    "model": model_name,
                    "runs": len(group),
                    "success_rate": round(len(group_completed) / len(group) * 100, 1),
                    "input_tokens": total(("usage", "input_tokens"), group),
                    "output_tokens": total(("usage", "output_tokens"), group),
                    "thinking_tokens": total(("usage", "thinking_tokens"), group),
                    "avg_tokens_per_second": average(("timing", "tokens_per_second"), group_completed),
                    "avg_prefill_tokens_per_second": average(
                        ("timing", "prefill_tokens_per_second"), group_completed
                    ),
                    "avg_latency_ms": average(("timing", "end_to_end_duration_ms"), group_completed),
                    "avg_gpu_utilization_pct": average(
                        ("resources", "avg_gpu_utilization_pct"), group_completed
                    ),
                    "peak_vram_bytes": max(nums(("resources", "peak_dedicated_vram_bytes"), group) or [0]),
                }
            )
        daily: dict[str, dict[str, Any]] = {}
        for run in reversed(runs):
            day = str(run.get("created_at", ""))[:10]
            bucket = daily.setdefault(day, {"date": day, "runs": 0, "tokens": 0})
            bucket["runs"] += 1
            if isinstance(run.get("usage"), dict):
                bucket["tokens"] += int(run["usage"].get("total_tokens") or 0)
        return {
            "generated_at": utc_now(),
            "summary": {
                "total_runs": len(runs),
                "completed_runs": len(completed),
                "failed_runs": len(runs) - len(completed),
                "success_rate": round(len(completed) / len(runs) * 100, 1) if runs else None,
                "models_used": len({run["model_name"] for run in runs}),
                "input_tokens": total(("usage", "input_tokens")),
                "output_tokens": total(("usage", "output_tokens")),
                "thinking_tokens": total(("usage", "thinking_tokens")),
                "visible_output_tokens": total(("usage", "visible_output_tokens")),
                "total_tokens": total(("usage", "total_tokens")),
                "avg_tokens_per_second": average(("timing", "tokens_per_second"), completed),
                "avg_prefill_tokens_per_second": average(
                    ("timing", "prefill_tokens_per_second"), completed
                ),
                "avg_latency_ms": average(("timing", "end_to_end_duration_ms"), completed),
                "p95_latency_ms": percentile(("timing", "end_to_end_duration_ms"), 0.95),
                "avg_ttft_ms": average(("timing", "time_to_first_token_ms"), completed),
                "peak_process_ram_bytes": max(nums(("resources", "peak_process_rss_bytes")) or [0]),
                "peak_vram_bytes": max(nums(("resources", "peak_dedicated_vram_bytes")) or [0]),
                "peak_gpu_utilization_pct": max(
                    nums(("resources", "peak_gpu_utilization_pct")) or [0]
                ),
                "avg_gpu_utilization_pct": average(
                    ("resources", "avg_gpu_utilization_pct"), completed
                ),
                "peak_shared_gpu_memory_bytes": max(
                    nums(("resources", "peak_shared_gpu_memory_bytes")) or [0]
                ),
                "spill_runs": sum(
                    1
                    for run in runs
                    if isinstance(run.get("resources"), dict) and run["resources"].get("spill_suspected")
                ),
            },
            "models": models,
            "daily": list(daily.values())[-14:],
            "recent_runs": runs[:20],
        }

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in (
            "lane_keys_json",
            "device_json",
            "request_json",
            "configuration_json",
            "usage_json",
            "timing_json",
            "resources_json",
            "backend_response_json",
        ):
            raw = item.pop(key, None)
            output_key = key.removesuffix("_json")
            try:
                item[output_key] = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                item[output_key] = None
        return item


class AppError(RuntimeError):
    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


class ServerHandle:
    def __init__(self, target: str, model: ModelConfig, lanes: list[LaneConfig], process: LlamaServerProcess):
        self.target = target
        self.model = model
        self.lanes = lanes
        self.process = process

    def public(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "model": self.model.name,
            "lane_keys": [lane.key for lane in self.lanes],
            "device": self.process.device,
            "base_url": self.process.base_url,
            "pid": self.process.pid if self.process.proc else None,
            "alive": bool(self.process.proc and self.process.proc.poll() is None),
            "context_window": self.process.ctx_size,
            "reasoning_budget": self.model.reasoning_budget,
        }


class DeploymentManager:
    VALID_MODES = {"single_large_model", "parallel_models", "single_gpu"}

    def __init__(self, config: AppConfig, data_dir: Path, store: RunStore):
        self.config = config
        self.data_dir = data_dir
        self.store = store
        self.lock = threading.RLock()
        self.transition_lock = threading.Lock()
        self.state = "idle"
        self.mode = "idle"
        self.error: str | None = None
        self.handles: dict[str, ServerHandle] = {}
        self.active_profile: dict[str, Any] | None = None
        self.deployment_id: str | None = None
        self.models_by_id: dict[str, ModelConfig] = {}
        self.model_records: list[dict[str, Any]] = []
        self.model_roots: list[Path] = []
        self.refresh_models()
        self.sampler = ResourceSampler(self)
        self.sampler.start()

    def _candidate_model_roots(self) -> list[Path]:
        candidates: list[Path] = []
        if self.config.discovery.models_dir:
            candidates.append(Path(self.config.discovery.models_dir).expanduser())
        settings_path = Path(self.config.discovery.lmstudio_home) / "settings.json"
        try:
            downloads_folder = json.loads(settings_path.read_text(encoding="utf-8")).get("downloadsFolder")
            if downloads_folder:
                candidates.append(Path(os.path.expandvars(os.path.expanduser(str(downloads_folder)))))
        except (OSError, json.JSONDecodeError):
            pass
        candidates.append(Path(self.config.discovery.lmstudio_home) / "models")
        roots: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate.absolute()
            key = os.path.normcase(str(resolved))
            if key not in seen:
                seen.add(key)
                roots.append(resolved)
        return roots

    @staticmethod
    def _is_primary_gguf(path: Path) -> bool:
        name = path.name.lower()
        if not name.endswith(".gguf") or "mmproj" in name:
            return False
        shard = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", name)
        return shard is None or shard.group(1) == "00001"

    @staticmethod
    def _model_identity(path: Path, root: Path) -> tuple[str, str, str]:
        relative = path.relative_to(root)
        repository = relative.parent.name or path.parent.name
        family = re.sub(r"-gguf$", "", repository, flags=re.IGNORECASE)
        stem = re.sub(r"-\d{5}-of-\d{5}$", "", path.stem, flags=re.IGNORECASE)
        quant_tokens = re.findall(
            r"(?i)(?:IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|MXFP\d+(?:_[A-Z0-9]+)*|BF16|FP16|F16|FP32|F32)",
            stem,
        )
        quantization = "-".join(dict.fromkeys(token.upper() for token in quant_tokens)) or "Unspecified"
        publisher = relative.parts[0] if len(relative.parts) > 1 else "Local"
        return family, quantization, publisher

    def refresh_models(self) -> None:
        roots = self._candidate_model_roots()
        registry: dict[str, ModelConfig] = {}
        records: list[dict[str, Any]] = []
        configured_paths: set[str] = set()

        for configured in self.config.models:
            model = configured
            raw_path = Path(os.path.expandvars(os.path.expanduser(configured.path)))
            if not raw_path.is_absolute():
                for root in roots:
                    candidate = root / raw_path
                    if candidate.exists():
                        model = replace(configured, path=str(candidate))
                        break
            path = resolve_model_path(self.config, model)
            if path.exists():
                configured_paths.add(os.path.normcase(str(path.resolve())))
            model_id = configured.name
            family, quantization, publisher = self._model_identity(path, next((root for root in roots if path.is_relative_to(root)), path.parent))
            registry[model_id] = model
            records.append(
                self._model_record(model_id, model, path, family, quantization, publisher, "configured")
            )

        for root in roots:
            if not root.is_dir():
                continue
            try:
                paths = sorted(root.rglob("*.gguf"), key=lambda item: str(item).lower())
            except OSError:
                continue
            for path in paths:
                if not self._is_primary_gguf(path):
                    continue
                try:
                    canonical = os.path.normcase(str(path.resolve()))
                except OSError:
                    canonical = os.path.normcase(str(path.absolute()))
                if canonical in configured_paths:
                    continue
                relative = path.relative_to(root).as_posix()
                model_id = f"lmstudio:{relative}"
                family, quantization, publisher = self._model_identity(path, root)
                model = ModelConfig(
                    name=path.stem,
                    path=str(path),
                    size_gb=0.0,
                    tags=["lm-studio", publisher, quantization],
                )
                registry[model_id] = model
                records.append(
                    self._model_record(model_id, model, path, family, quantization, publisher, "lm_studio")
                )

        records.sort(
            key=lambda item: (
                not item["exists"],
                item["family"].lower(),
                item["quantization"].lower(),
                item["path"].lower(),
            )
        )
        with self.lock:
            self.model_roots = roots
            self.models_by_id = registry
            self.model_records = records

    def _model_record(
        self,
        model_id: str,
        model: ModelConfig,
        path: Path,
        family: str,
        quantization: str,
        publisher: str,
        source: str,
    ) -> dict[str, Any]:
        return {
            "id": model_id,
            "name": model.name,
            "display_name": family,
            "family": family,
            "quantization": quantization,
            "publisher": publisher,
            "source": source,
            "path": str(path),
            "exists": path.exists(),
            "size_gb": model_size_gb(self.config, model),
            "multi_gpu": model.multi_gpu,
            "pin_lane": model.pin_lane or None,
            "default_context": model.ctx_size or None,
            "default_reasoning_budget": (
                model.reasoning_budget
                if model.reasoning_budget is not None
                else self.config.policy.reasoning_budget
            ),
            "tensor_split": model.tensor_split or None,
        }

    def catalog(self) -> dict[str, Any]:
        lanes = [
            {
                "key": lane.key,
                "display": lane.display,
                "match": lane.match,
                "vram_gb": lane.vram_gb,
                "capacity_gb": round(lane_capacity_gb(self.config, lane), 2),
                "port": lane.port,
                "backend": lane.backend,
            }
            for lane in self.config.lanes
        ]
        with self.lock:
            models = list(self.model_records)
            model_roots = [str(root) for root in self.model_roots]
        return {
            "project": self.config.project.name,
            "lanes": lanes,
            "models": models,
            "model_count": sum(1 for model in models if model["exists"]),
            "model_roots": model_roots,
            "context_presets": sorted(
                {4096, 8192, self.config.policy.ctx_min, self.config.policy.ctx_mid, self.config.policy.ctx_max}
            ),
            "reasoning_presets": DEFAULT_REASONING_PRESETS,
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "mode": self.mode,
                "error": self.error,
                "profile": self.active_profile,
                "deployment_id": self.deployment_id,
                "servers": [handle.public() for handle in self.handles.values()],
                "metrics": self.sampler.latest(),
            }

    def deploy(self, profile: dict[str, Any]) -> dict[str, Any]:
        if not self.transition_lock.acquire(blocking=False):
            raise AppError("Another deployment change is already in progress.", HTTPStatus.CONFLICT)
        try:
            mode = str(profile.get("mode", ""))
            if mode not in self.VALID_MODES:
                raise AppError(f"Unsupported mode: {mode}")
            specs = self._validate_profile(mode, profile)
            with self.lock:
                self.state = "switching"
                self.error = None
            self._stop_all()
            deployment_id = str(uuid.uuid4())
            self.store.begin_deployment(deployment_id, mode, profile)
            with self.lock:
                self.deployment_id = deployment_id
            run_dir = self.data_dir / "deployments" / datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            started: dict[str, ServerHandle] = {}
            errors: list[str] = []

            def start_one(spec: dict[str, Any]) -> None:
                try:
                    process = self._make_process(spec, run_dir)
                    process.start()
                    handle = ServerHandle(spec["target"], spec["model"], spec["lanes"], process)
                    with self.lock:
                        started[spec["target"]] = handle
                except BaseException as exc:  # include SystemExit raised by discovery helpers
                    with self.lock:
                        errors.append(f"{spec['target']}: {exc}")

            threads = [threading.Thread(target=start_one, args=(spec,), daemon=True) for spec in specs]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            if errors:
                for handle in started.values():
                    handle.process.stop()
                message = "; ".join(errors)
                with self.lock:
                    self.state = "error"
                    self.mode = "idle"
                    self.handles = {}
                    self.error = message
                    self.active_profile = None
                self.store.update_deployment(deployment_id, "failed", error=message)
                raise AppError(message, HTTPStatus.INTERNAL_SERVER_ERROR)

            with self.lock:
                self.handles = started
                self.state = "ready"
                self.mode = mode
                self.active_profile = profile
                self.error = None
            self.store.update_deployment(
                deployment_id,
                "ready",
                servers=[handle.public() for handle in started.values()],
            )
            return self.status()
        finally:
            self.transition_lock.release()

    def unload(self) -> dict[str, Any]:
        if not self.transition_lock.acquire(blocking=False):
            raise AppError("Another deployment change is already in progress.", HTTPStatus.CONFLICT)
        try:
            with self.lock:
                self.state = "switching"
            self._stop_all()
            with self.lock:
                self.state = "idle"
                self.mode = "idle"
                self.error = None
                self.active_profile = None
                self.deployment_id = None
            return self.status()
        finally:
            self.transition_lock.release()

    def shutdown(self) -> None:
        self.sampler.stop()
        self._stop_all()

    def _stop_all(self) -> None:
        with self.lock:
            handles = list(self.handles.values())
            self.handles = {}
            deployment_id = self.deployment_id
        for handle in handles:
            try:
                handle.process.stop()
            except Exception:  # noqa: BLE001
                pass
        if deployment_id and handles:
            self.store.update_deployment(deployment_id, "stopped")

    def _model(self, name: str) -> ModelConfig:
        with self.lock:
            if name in self.models_by_id:
                return self.models_by_id[name]
            for model in self.models_by_id.values():
                if model.name == name:
                    return model
        raise AppError(f"Unknown model: {name}")

    def _lane(self, key: str) -> LaneConfig:
        for lane in self.config.lanes:
            if lane.key == key:
                return lane
        raise AppError(f"Unknown lane: {key}")

    def _configured_model(self, raw: dict[str, Any], lane_vram_gb: float) -> tuple[ModelConfig, int]:
        source = self._model(str(raw.get("model", "")))
        context = int(raw.get("context_window") or source.ctx_size or ctx_for(self.config, model_size_gb(self.config, source), lane_vram_gb))
        if context < 256:
            raise AppError("Context window must be at least 256 tokens.")
        budget = int(raw.get("reasoning_budget", source.reasoning_budget if source.reasoning_budget is not None else self.config.policy.reasoning_budget))
        if budget < 0:
            raise AppError("Reasoning budget cannot be negative.")
        tensor_split = str(raw.get("tensor_split") or source.tensor_split)
        model = replace(source, ctx_size=context, reasoning_budget=budget, tensor_split=tensor_split)
        path = resolve_model_path(self.config, model)
        if not path.exists():
            raise AppError(f"Model file does not exist: {path}")
        return model, context

    def _validate_profile(self, mode: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
        if mode == "single_large_model":
            raw = profile.get("model") or {}
            model, context = self._configured_model(raw, sum(lane.vram_gb for lane in self.config.lanes))
            backends = {lane.backend for lane in self.config.lanes}
            if len(backends) != 1:
                raise AppError("Both lanes must use the same backend for multi-GPU mode.")
            return [{"target": "primary", "model": model, "lanes": list(self.config.lanes), "context": context, "multi": True}]

        if mode == "single_gpu":
            raw = profile.get("model") or {}
            lane = self._lane(str(raw.get("lane", "")))
            model, context = self._configured_model(raw, lane.vram_gb)
            self._validate_single_fit(model, lane)
            return [{"target": lane.key, "model": model, "lanes": [lane], "context": context, "multi": False}]

        raw_models = profile.get("models") or []
        if len(raw_models) != 2:
            raise AppError("Parallel mode requires exactly two model/lane selections.")
        specs = []
        seen_lanes: set[str] = set()
        for raw in raw_models:
            lane = self._lane(str(raw.get("lane", "")))
            if lane.key in seen_lanes:
                raise AppError("Parallel mode must use two different lanes.")
            seen_lanes.add(lane.key)
            model, context = self._configured_model(raw, lane.vram_gb)
            self._validate_single_fit(model, lane)
            specs.append({"target": lane.key, "model": model, "lanes": [lane], "context": context, "multi": False})
        return specs

    def _validate_single_fit(self, model: ModelConfig, lane: LaneConfig) -> None:
        size = model_size_gb(self.config, model)
        if model.pin_lane and model.pin_lane != lane.key:
            raise AppError(f"{model.name} is pinned to lane {model.pin_lane}, not {lane.key}.")
        if size > lane_capacity_gb(self.config, lane):
            raise AppError(
                f"{model.name} is {size:.2f} GB but {lane.key} has a safe model-file budget of "
                f"{lane_capacity_gb(self.config, lane):.2f} GB. Choose another GPU/model or multi-GPU mode."
            )

    def _make_process(self, spec: dict[str, Any], run_dir: Path) -> LlamaServerProcess:
        model = spec["model"]
        path = resolve_model_path(self.config, model)
        if spec["multi"]:
            return LlamaServerProcess.for_multi_gpu(
                self.config, spec["lanes"], model, path, spec["context"], run_dir
            )
        return LlamaServerProcess.for_single_lane(
            self.config, spec["lanes"][0], model, path, spec["context"], run_dir
        )


class ChatService:
    def __init__(self, manager: DeploymentManager, store: RunStore):
        self.manager = manager
        self.store = store

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.manager.lock:
            if self.manager.state != "ready" or not self.manager.handles:
                raise AppError("Load a deployment before sending a message.", HTTPStatus.CONFLICT)
            requested_targets = payload.get("targets")
            if requested_targets:
                targets = [str(value) for value in requested_targets]
            elif payload.get("target") == "compare":
                targets = list(self.manager.handles)
            elif payload.get("target"):
                targets = [str(payload["target"])]
            else:
                targets = [next(iter(self.manager.handles))]
            missing = [target for target in targets if target not in self.manager.handles]
            if missing:
                raise AppError(f"Unknown or inactive target(s): {', '.join(missing)}")
            handles = {target: self.manager.handles[target] for target in targets}

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                raise AppError("A prompt or messages array is required.")
            messages = [{"role": "user", "content": prompt}]
        comparison_id = str(uuid.uuid4()) if len(handles) > 1 else None
        results: dict[str, Any] = {}
        threads = []

        def run(target: str, handle: ServerHandle) -> None:
            results[target] = self._run_one(target, handle, messages, payload, comparison_id)

        for target, handle in handles.items():
            thread = threading.Thread(target=run, args=(target, handle), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return {"comparison_id": comparison_id, "results": results}

    def _run_one(
        self,
        target: str,
        handle: ServerHandle,
        messages: list[dict[str, Any]],
        payload: dict[str, Any],
        comparison_id: str | None,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        request_payload: dict[str, Any] = {
            "model": handle.model.name,
            "messages": messages,
            "stream": False,
            "temperature": float(payload.get("temperature", 0.7)),
            "top_p": float(payload.get("top_p", 0.95)),
            "max_tokens": int(payload.get("max_tokens", 1024)),
        }
        if payload.get("seed") is not None:
            request_payload["seed"] = int(payload["seed"])
        model_path = resolve_model_path(self.manager.config, handle.model)
        configuration = {
            "app_version": APP_VERSION,
            "deployment_id": self.manager.deployment_id,
            "deployment_profile": self.manager.active_profile,
            "backend": handle.process.backend,
            "endpoint": handle.process.base_url,
            "device": handle.process.device,
            "model_path": str(model_path),
            "model_size_gb": model_size_gb(self.manager.config, handle.model),
            "context_window": handle.process.ctx_size,
            "reasoning_mode": handle.model.reasoning_mode or self.manager.config.policy.reasoning_mode,
            "reasoning_budget": handle.model.reasoning_budget,
            "tensor_split": handle.process.tensor_split or None,
            "gpu_layers": (
                handle.model.gpu_layers
                if handle.model.gpu_layers is not None
                else self.manager.config.policy.gpu_layers
            ),
            "parallel_slots": handle.model.parallel or self.manager.config.policy.parallel,
            "threads": handle.model.threads or None,
            "ubatch_size": handle.model.ubatch_size or None,
            "flash_attention": handle.model.flash_attn or self.manager.config.policy.flash_attn,
            "cache_reuse": (
                handle.model.cache_reuse
                if handle.model.cache_reuse is not None
                else self.manager.config.policy.cache_reuse
            ),
            "lanes": [asdict(lane) for lane in handle.lanes],
            "server_command": handle.process.build_command(),
        }
        self.store.begin_run(
            {
                "id": run_id,
                "comparison_id": comparison_id,
                "deployment_id": self.manager.deployment_id,
                "created_at": utc_now(),
                "mode": self.manager.mode,
                "target": target,
                "model_name": handle.model.name,
                "model_path": str(model_path),
                "lane_keys": [lane.key for lane in handle.lanes],
                "device": handle.process.device,
                "context_window": handle.process.ctx_size,
                "reasoning_budget": handle.model.reasoning_budget,
                "request": request_payload,
                "configuration": configuration,
            }
        )
        baseline = self.manager.sampler.capture()
        started = time.monotonic()
        try:
            backend = self._post_json(
                f"{handle.process.base_url}/v1/chat/completions",
                request_payload,
                timeout=float(payload.get("timeout_seconds", 900)),
            )
            elapsed = time.monotonic() - started
            choice = (backend.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            output = message.get("content") or choice.get("text") or ""
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            usage = self._normalize_usage(backend)
            timing = self._normalize_timing(backend, usage, elapsed)
            self.manager.sampler.capture()
            samples = self._sample_window(started, baseline)
            resources = self._resource_summary(samples, target)
            final = {
                "status": "completed",
                "output_text": output,
                "reasoning_text": reasoning,
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "timing": timing,
                "resources": resources,
                "backend_response": backend,
            }
            self.store.finish_run(run_id, final)
            self.store.save_samples(run_id, samples, started)
            return {"run_id": run_id, "model": handle.model.name, **final, "backend_response": None}
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - started
            self.manager.sampler.capture()
            samples = self._sample_window(started, baseline)
            resources = self._resource_summary(samples, target)
            final = {
                "status": "failed",
                "output_text": None,
                "reasoning_text": None,
                "usage": {},
                "timing": {"end_to_end_duration_ms": round(elapsed * 1000, 2)},
                "resources": resources,
                "error_text": str(exc),
                "backend_response": None,
            }
            self.store.finish_run(run_id, final)
            self.store.save_samples(run_id, samples, started)
            return {"run_id": run_id, "model": handle.model.name, **final}

    def _sample_window(self, started: float, baseline: dict[str, Any] | None) -> list[dict[str, Any]]:
        samples = self.manager.sampler.since(started)
        if baseline:
            baseline_key = baseline.get("sampled_at")
            if not samples or samples[0].get("sampled_at") != baseline_key:
                samples.insert(0, baseline)
        return samples

    @staticmethod
    def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model server returned HTTP {exc.code}: {detail}") from exc

    @staticmethod
    def _normalize_usage(payload: dict[str, Any]) -> dict[str, Any]:
        usage = payload.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        thinking_tokens = details.get("reasoning_tokens")
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = prompt_details.get("cached_tokens")
        timings = payload.get("timings") or payload.get("timing") or {}
        prefill_tokens = timings.get("prompt_n") or timings.get("prompt_tokens") or input_tokens
        visible_tokens = output_tokens
        if output_tokens is not None and thinking_tokens is not None:
            visible_tokens = max(0, output_tokens - thinking_tokens)
        return {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "uncached_input_tokens": (
                max(0, input_tokens - cached_tokens)
                if input_tokens is not None and cached_tokens is not None
                else None
            ),
            "prefill_tokens": prefill_tokens,
            "thinking_tokens": thinking_tokens,
            "visible_output_tokens": visible_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens"),
        }

    @staticmethod
    def _normalize_timing(payload: dict[str, Any], usage: dict[str, Any], elapsed: float) -> dict[str, Any]:
        timings = payload.get("timings") or payload.get("timing") or {}
        prompt_ms = timings.get("prompt_ms") or timings.get("prompt_eval_time_ms")
        predicted_ms = timings.get("predicted_ms") or timings.get("generation_time_ms")
        prefill_rate = timings.get("prompt_per_second") or timings.get("prompt_tokens_per_second")
        token_rate = timings.get("predicted_per_second") or timings.get("tokens_per_second")
        if token_rate is None and usage.get("output_tokens") and elapsed > 0:
            token_rate = usage["output_tokens"] / elapsed
        return {
            "time_to_first_token_ms": timings.get("time_to_first_token_ms"),
            "prefill_duration_ms": prompt_ms,
            "decode_duration_ms": predicted_ms,
            "end_to_end_duration_ms": round(elapsed * 1000, 2),
            "prefill_tokens_per_second": round(float(prefill_rate), 3) if prefill_rate is not None else None,
            "tokens_per_second": round(float(token_rate), 3) if token_rate is not None else None,
            "backend_prompt_tokens": timings.get("prompt_n") or timings.get("prompt_tokens"),
            "backend_output_tokens": timings.get("predicted_n") or timings.get("generated_tokens"),
            "native_timing": timings,
        }

    @staticmethod
    def _resource_summary(samples: list[dict[str, Any]], target: str) -> dict[str, Any]:
        process_rows = [
            server
            for sample in samples
            for server in sample.get("servers", [])
            if server.get("target") == target
        ]
        system_rows = [sample.get("system", {}) for sample in samples]

        def values(rows: list[dict[str, Any]], key: str) -> list[float]:
            return [float(row[key]) for row in rows if row.get(key) is not None]

        def average(items: list[float]) -> float | None:
            return round(sum(items) / len(items), 2) if items else None

        cpu = values(process_rows, "cpu_pct")
        gpu_utilization = values(process_rows, "gpu_utilization_pct")
        rss = values(process_rows, "rss_bytes")
        dedicated = values(process_rows, "dedicated_vram_bytes")
        shared = values(process_rows, "shared_gpu_memory_bytes")
        system_cpu = values(system_rows, "cpu_pct")
        ram = values(system_rows, "ram_used_bytes")
        spill = bool(shared and dedicated and max(shared) > 1024**3 and max(shared) > max(dedicated) * 0.15)
        return {
            "sample_count": len(process_rows),
            "avg_process_cpu_pct": average(cpu),
            "peak_process_cpu_pct": max(cpu) if cpu else None,
            "avg_gpu_utilization_pct": average(gpu_utilization),
            "peak_gpu_utilization_pct": max(gpu_utilization) if gpu_utilization else None,
            "peak_process_rss_bytes": int(max(rss)) if rss else None,
            "peak_dedicated_vram_bytes": int(max(dedicated)) if dedicated else None,
            "peak_shared_gpu_memory_bytes": int(max(shared)) if shared else None,
            "spill_suspected": spill if shared and dedicated else None,
            "avg_system_cpu_pct": average(system_cpu),
            "peak_system_ram_used_bytes": int(max(ram)) if ram else None,
            "samples_started_at": samples[0].get("sampled_at") if samples else None,
            "samples_finished_at": samples[-1].get("sampled_at") if samples else None,
        }


class ChatbotHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app_state: "ApplicationState"):
        super().__init__(address, RequestHandler)
        self.app_state = app_state


class ApplicationState:
    def __init__(self, config_path: Path, data_dir: Path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.data_dir = data_dir
        self.store = RunStore(data_dir / "chatbot.sqlite3")
        self.manager = DeploymentManager(self.config, data_dir, self.store)
        self.chat = ChatService(self.manager, self.store)

    def bootstrap(self) -> dict[str, Any]:
        return {
            "app_version": APP_VERSION,
            "config_path": str(self.config_path),
            "catalog": self.manager.catalog(),
            "status": self.manager.status(),
            "runs": self.store.recent(20),
        }


class RequestHandler(BaseHTTPRequestHandler):
    server: ChatbotHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/bootstrap":
            self._send_json(HTTPStatus.OK, self.server.app_state.bootstrap())
            return
        if path == "/api/status":
            self._send_json(HTTPStatus.OK, self.server.app_state.manager.status())
            return
        if path == "/api/dashboard":
            self._send_json(HTTPStatus.OK, self.server.app_state.store.dashboard())
            return
        if path == "/api/runs":
            self._send_json(HTTPStatus.OK, {"runs": self.server.app_state.store.recent(100)})
            return
        if path.startswith("/api/runs/"):
            run_id = path.rsplit("/", 1)[-1]
            run = self.server.app_state.store.get(run_id)
            if run is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Run not found"})
            else:
                self._send_json(HTTPStatus.OK, run)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/deploy":
                result = self.server.app_state.manager.deploy(payload)
            elif path == "/api/models/refresh":
                self.server.app_state.manager.refresh_models()
                result = {"catalog": self.server.app_state.manager.catalog()}
            elif path == "/api/unload":
                result = self.server.app_state.manager.unload()
            elif path == "/api/chat":
                result = self.server.app_state.chat.chat(payload)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            self._send_json(HTTPStatus.OK, result)
        except AppError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body"})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 10 * 1024 * 1024:
            raise AppError("Request body is too large.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise AppError("JSON request body must be an object.")
        return value

    def _send_json(self, status: int, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


LEGACY_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dual GPU Chat</title>
  <style>
    :root { color-scheme: dark; --bg:#0a0e13; --panel:#111821; --line:#273446; --muted:#94a3b8; --text:#eef4fb; --accent:#59d8b5; --blue:#62a8ff; --danger:#ff6b75; }
    * { box-sizing:border-box; } body { margin:0; font:14px/1.45 Inter,Segoe UI,sans-serif; background:var(--bg); color:var(--text); }
    header { position:sticky; top:0; z-index:2; display:flex; justify-content:space-between; gap:16px; padding:14px 20px; border-bottom:1px solid var(--line); background:#0a0e13ee; backdrop-filter:blur(12px); }
    h1,h2,h3,p { margin-top:0; } h1 { font-size:18px; margin:0; } h2 { font-size:17px; } h3 { font-size:14px; color:var(--muted); }
    main { display:grid; grid-template-columns:minmax(310px,390px) minmax(420px,1fr); gap:16px; padding:16px; max-width:1500px; margin:auto; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px; box-shadow:0 12px 40px #0003; }
    .stack { display:grid; gap:12px; } .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; } .grow { flex:1; }
    label { display:grid; gap:5px; color:var(--muted); font-size:12px; }
    input,select,textarea,button { font:inherit; color:var(--text); border:1px solid var(--line); border-radius:9px; background:#0c131c; padding:9px 10px; }
    textarea { width:100%; min-height:100px; resize:vertical; } button { cursor:pointer; background:#182333; } button.primary { background:var(--accent); color:#06251d; border-color:transparent; font-weight:700; } button.danger { color:#ffd9dc; border-color:#6d3038; } button:disabled { opacity:.5; cursor:not-allowed; }
    .mode-tabs { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; } .mode-tabs button.active { color:#061c16; background:var(--accent); border-color:transparent; }
    .slot { display:grid; grid-template-columns:1fr 1.4fr; gap:8px; padding:10px; border:1px solid var(--line); border-radius:10px; }
    .badge { display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:12px; }
    .dot { width:8px; height:8px; border-radius:50%; background:var(--muted); } .dot.ready { background:var(--accent); box-shadow:0 0 12px var(--accent); } .dot.error { background:var(--danger); }
    .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(90px,1fr)); gap:8px; } .metric { padding:9px; background:#0c131c; border-radius:9px; } .metric b { display:block; font-size:15px; } .metric span { color:var(--muted); font-size:11px; }
    #messages { min-height:280px; max-height:55vh; overflow:auto; display:grid; gap:10px; padding-right:5px; }
    .message { padding:12px; border-radius:11px; background:#0c131c; white-space:pre-wrap; overflow-wrap:anywhere; } .message.user { border-left:3px solid var(--blue); } .message.assistant { border-left:3px solid var(--accent); }
    .response-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:10px; }
    .response-card { padding:12px; border:1px solid var(--line); border-radius:11px; background:#0c131c; } .response-card pre { white-space:pre-wrap; font-family:inherit; }
    .error { color:#ff9ca3; white-space:pre-wrap; } .muted { color:var(--muted); } .hidden { display:none !important; }
    table { width:100%; border-collapse:collapse; font-size:12px; } th,td { text-align:left; padding:7px; border-bottom:1px solid var(--line); vertical-align:top; } th { color:var(--muted); }
    details summary { cursor:pointer; color:var(--muted); } code { color:#b5e8dc; }
    @media (max-width:900px) { main { grid-template-columns:1fr; } .metric-grid { grid-template-columns:repeat(2,1fr); } }
  </style>
</head>
<body>
<header><div><h1>Dual GPU Chat</h1><span id="project" class="muted">Loading…</span></div><div class="row"><span class="badge"><i id="state-dot" class="dot"></i><span id="state">connecting</span></span><button class="danger" onclick="unload()">Unload</button></div></header>
<main>
  <section class="stack">
    <div class="panel stack">
      <div><h2>Deployment</h2><p class="muted">Choose how models use the two GPUs.</p></div>
      <div class="mode-tabs">
        <button data-mode="single_large_model" onclick="setMode(this.dataset.mode)">Both GPUs</button>
        <button data-mode="parallel_models" onclick="setMode(this.dataset.mode)">Two models</button>
        <button data-mode="single_gpu" onclick="setMode(this.dataset.mode)">One GPU</button>
      </div>
      <div id="deployment-fields" class="stack"></div>
      <button id="deploy-button" class="primary" onclick="deploy()">Load deployment</button>
      <div id="deploy-error" class="error"></div>
    </div>
    <div class="panel stack">
      <div class="row"><h2 class="grow">Live resources</h2><span id="sample-time" class="muted">—</span></div>
      <div id="system-metrics" class="metric-grid"></div>
      <div id="server-metrics" class="stack"></div>
    </div>
  </section>
  <section class="stack">
    <div class="panel stack">
      <div class="row"><div class="grow"><h2>Chat</h2><span id="active-models" class="muted">No models loaded</span></div><label>Target<select id="chat-target"></select></label></div>
      <div id="messages"></div>
      <textarea id="prompt" placeholder="Send a message to the active model…"></textarea>
      <div class="row">
        <label>Temperature<input id="temperature" type="number" min="0" max="2" step="0.1" value="0.7"></label>
        <label>Max output<input id="max-tokens" type="number" min="1" value="1024"></label>
        <label>Top P<input id="top-p" type="number" min="0" max="1" step="0.05" value="0.95"></label>
        <button id="send-button" class="primary grow" onclick="sendMessage()">Send</button>
      </div>
      <div id="chat-error" class="error"></div>
    </div>
    <div class="panel stack">
      <div class="row"><h2 class="grow">Run history</h2><button onclick="loadRuns()">Refresh</button></div>
      <div style="overflow:auto"><table><thead><tr><th>Time</th><th>Model</th><th>Status</th><th>Tokens</th><th>tok/s</th><th>RAM / VRAM peak</th></tr></thead><tbody id="run-rows"></tbody></table></div>
    </div>
  </section>
</main>
<script>
let bootstrap=null, mode='single_gpu', history=[];
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtBytes=n=>n==null?'—':n>=1073741824?(n/1073741824).toFixed(2)+' GB':(n/1048576).toFixed(0)+' MB';
const metric=(v,l)=>`<div class="metric"><b>${esc(v??'—')}</b><span>${esc(l)}</span></div>`;
async function api(path,options={}){const r=await fetch(path,{headers:{'Content-Type':'application/json'},...options}); const d=await r.json(); if(!r.ok) throw new Error(d.error||r.statusText); return d;}
function modelOptions(multi=false){return bootstrap.catalog.models.filter(m=>multi?true:!m.multi_gpu).map(m=>`<option value="${esc(m.name)}">${esc(m.name)} · ${m.size_gb.toFixed(2)} GB${m.exists?'':' · MISSING'}</option>`).join('');}
function laneOptions(){return bootstrap.catalog.lanes.map(l=>`<option value="${esc(l.key)}">${esc(l.display)} · ${l.vram_gb} GB</option>`).join('');}
function contextOptions(){return bootstrap.catalog.context_presets.map(v=>`<option value="${v}" label="${Math.round(v/1024)}K"></option>`).join('');}
function reasoningOptions(){return Object.entries(bootstrap.catalog.reasoning_presets).map(([k,v])=>`<option value="${v}">${k[0].toUpperCase()+k.slice(1)} · ${v}</option>`).join('');}
function slot(i,showLane=true,multi=false){const defaultCtx=bootstrap.catalog.context_presets.includes(16384)?16384:bootstrap.catalog.context_presets[0]; return `<div class="slot">${showLane?`<label>GPU<select id="lane-${i}">${laneOptions()}</select></label>`:''}<label>Model<select id="model-${i}">${modelOptions(multi)}</select></label><label>Context tokens<input id="ctx-${i}" type="number" min="256" step="256" list="ctx-list-${i}" value="${defaultCtx}"><datalist id="ctx-list-${i}">${contextOptions()}</datalist></label><label>Reasoning<select id="reason-${i}">${reasoningOptions()}</select></label>${multi?'<label>Tensor split<input id="split-0" placeholder="Auto / 36,12"></label>':''}</div>`;}
function setMode(next){mode=next; document.querySelectorAll('.mode-tabs button').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode)); let html=''; if(mode==='single_large_model') html=slot(0,false,true); if(mode==='parallel_models') html=slot(0,true,false)+slot(1,true,false); if(mode==='single_gpu') html=slot(0,true,false); $('deployment-fields').innerHTML=html; if(mode==='parallel_models'&&bootstrap.catalog.lanes[1]) $('lane-1').value=bootstrap.catalog.lanes[1].key;}
function readSlot(i,withLane=true){const value={model:$('model-'+i).value,context_window:Number($('ctx-'+i).value),reasoning_budget:Number($('reason-'+i).value)}; if(withLane)value.lane=$('lane-'+i).value; if($('split-'+i))value.tensor_split=$('split-'+i).value; return value;}
async function deploy(){try{$('deploy-error').textContent=''; $('deploy-button').disabled=true; let body={mode}; if(mode==='parallel_models')body.models=[readSlot(0),readSlot(1)]; else body.model=readSlot(0,mode!=='single_large_model'); await api('/api/deploy',{method:'POST',body:JSON.stringify(body)}); await refresh();}catch(e){$('deploy-error').textContent=e.message;}finally{$('deploy-button').disabled=false;}}
async function unload(){try{await api('/api/unload',{method:'POST',body:'{}'}); await refresh();}catch(e){$('deploy-error').textContent=e.message;}}
function renderStatus(s){$('state').textContent=s.state+' · '+s.mode; $('state-dot').className='dot '+(s.state==='ready'?'ready':s.state==='error'?'error':''); const servers=s.servers||[]; $('active-models').textContent=servers.length?servers.map(x=>`${x.model} on ${x.lane_keys.join('+')}`).join(' · '):'No models loaded'; const old=$('chat-target').value; $('chat-target').innerHTML=servers.map(x=>`<option value="${esc(x.target)}">${esc(x.model)} · ${esc(x.target)}</option>`).join('')+(servers.length>1?'<option value="compare">Both / Compare</option>':''); if([...$('chat-target').options].some(x=>x.value===old))$('chat-target').value=old; $('send-button').disabled=s.state!=='ready'; renderMetrics(s.metrics);}
function renderMetrics(m){if(!m){$('system-metrics').innerHTML=metric('—','Waiting for sample');return;} $('sample-time').textContent=new Date(m.sampled_at).toLocaleTimeString(); const s=m.system||{}; $('system-metrics').innerHTML=metric(s.cpu_pct==null?'—':s.cpu_pct.toFixed(1)+'%','System CPU')+metric(s.ram_used_pct==null?'—':s.ram_used_pct.toFixed(1)+'%','RAM used')+metric(fmtBytes(s.ram_used_bytes),'RAM')+metric(fmtBytes(s.pagefile_used_bytes),'Pagefile / commit'); $('server-metrics').innerHTML=(m.servers||[]).map(x=>`<div><h3>${esc(x.model)} · ${esc(x.target)}</h3><div class="metric-grid">${metric(x.cpu_pct==null?'—':x.cpu_pct.toFixed(1)+'%','Process CPU')}${metric(fmtBytes(x.rss_bytes),'Process RAM')}${metric(fmtBytes(x.dedicated_vram_bytes),'Dedicated VRAM')}${metric(fmtBytes(x.shared_gpu_memory_bytes),'Shared GPU memory')}</div></div>`).join('')||'<span class="muted">No model process is active.</span>';}
async function refresh(){try{renderStatus(await api('/api/status'));}catch(e){$('state').textContent='disconnected';}}
async function sendMessage(){const prompt=$('prompt').value.trim(); if(!prompt)return; $('chat-error').textContent=''; $('send-button').disabled=true; $('messages').insertAdjacentHTML('beforeend',`<div class="message user">${esc(prompt)}</div>`); $('prompt').value=''; history.push({role:'user',content:prompt}); try{const d=await api('/api/chat',{method:'POST',body:JSON.stringify({target:$('chat-target').value,messages:history,temperature:Number($('temperature').value),top_p:Number($('top-p').value),max_tokens:Number($('max-tokens').value)})}); const cards=Object.entries(d.results).map(([target,r])=>`<div class="response-card"><h3>${esc(r.model)} · ${esc(target)}</h3>${r.status==='completed'?`<pre>${esc(r.output_text)}</pre><span class="muted">${r.usage?.output_tokens??'—'} output tokens · ${r.timing?.tokens_per_second??'—'} tok/s · ${r.timing?.end_to_end_duration_ms??'—'} ms</span>`:`<div class="error">${esc(r.error_text)}</div>`}</div>`).join(''); $('messages').insertAdjacentHTML('beforeend',`<div class="response-grid">${cards}</div>`); if(Object.keys(d.results).length===1){const r=Object.values(d.results)[0]; if(r.status==='completed')history.push({role:'assistant',content:r.output_text});} $('messages').scrollTop=$('messages').scrollHeight; await loadRuns();}catch(e){$('chat-error').textContent=e.message;}finally{$('send-button').disabled=false;}}
async function loadRuns(){try{const d=await api('/api/runs'); $('run-rows').innerHTML=d.runs.map(r=>`<tr title="${esc(r.error_text||r.id)}"><td>${new Date(r.created_at).toLocaleString()}</td><td>${esc(r.model_name)}<br><span class="muted">${esc(r.target)}</span></td><td>${esc(r.status)}</td><td>${r.usage?.input_tokens??'—'} / ${r.usage?.output_tokens??'—'}${r.usage?.thinking_tokens!=null?'<br>thinking '+r.usage.thinking_tokens:''}</td><td>${r.timing?.tokens_per_second??'—'}</td><td>${fmtBytes(r.resources?.peak_process_rss_bytes)} / ${fmtBytes(r.resources?.peak_dedicated_vram_bytes)}${r.resources?.spill_suspected?'<br><span class="error">spill suspected</span>':''}</td></tr>`).join('');}catch(e){console.error(e);}}
async function init(){try{bootstrap=await api('/api/bootstrap'); $('project').textContent=bootstrap.catalog.project+' · '+bootstrap.config_path; setMode('single_gpu'); renderStatus(bootstrap.status); await loadRuns(); setInterval(refresh,2000);}catch(e){$('project').textContent=e.message;}}
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))sendMessage();}); init();
</script>
</body>
</html>
"""

# Keep the browser application in a standalone file so its layout can evolve
# independently while app.py remains the single command used to run the system.
HTML = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local dual-GPU chatbot web application.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("example.dual_gpu.toml")),
        help="Path to the dual-GPU TOML configuration.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web UI bind address. Defaults to all interfaces for local-network access.",
    )
    parser.add_argument("--port", type=int, default=8090, help="Web UI port. Defaults to 8090.")
    parser.add_argument("--data-dir", default="./chat_runs", help="Directory for SQLite data and server logs.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the UI in the default browser.")
    parser.add_argument("--check", action="store_true", help="Validate configuration and storage, then exit.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (config_path.parent / data_dir).resolve()
    state = ApplicationState(config_path, data_dir)
    if args.check:
        print(json.dumps({"status": "ok", "config": str(config_path), "data_dir": str(data_dir)}, indent=2))
        state.manager.shutdown()
        return 0

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: This app has no authentication. Binding outside localhost exposes prompts and model controls.")
    server = ChatbotHTTPServer((args.host, args.port), state)
    local_url = f"http://127.0.0.1:{args.port}"
    is_lan_enabled = args.host not in {"127.0.0.1", "localhost", "::1"}
    lan_urls = [f"http://{address}:{args.port}" for address in local_network_addresses()] if is_lan_enabled else []
    print("Dual GPU Studio is running")
    print(f"  Windows local: {local_url}")
    if lan_urls:
        for lan_url in lan_urls:
            print(f"  Local network: {lan_url}")
        print("  Open a Local network URL on your Mac.")
    elif is_lan_enabled:
        print(f"  Listening on: http://{args.host}:{args.port}")
        print("  No private IPv4 address was detected; run ipconfig to find the Windows address.")
    else:
        print("  Local network access is disabled by the selected --host value.")
    print(f"Configuration: {config_path}")
    print(f"Run database: {state.store.path}")
    print("Press Ctrl+C to stop and unload managed model servers.")
    if not args.no_browser:
        threading.Timer(0.75, lambda: webbrowser.open(local_url)).start()

    stopping = threading.Event()

    def stop_server(_signum: int | None = None, _frame: Any = None) -> None:
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_server)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        state.manager.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
