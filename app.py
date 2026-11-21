from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import re
import queue
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import parse_qs, urlparse

from dual_gpu_setup.config import AppConfig, LaneConfig, ModelConfig, load_config
from dual_gpu_setup import settings as service_settings
from dual_gpu_setup.lmstudio import lane_free_vram_gb, model_size_gb, resolve_model_path
from dual_gpu_setup.lmstudio_gpus import GpuSelectionError, get_selection, release, set_selection
from dual_gpu_setup.orchestrator import (
    cache_types_from_args, ctx_for, estimate_kv_cache_gb, lane_capacity_gb, offloaded_weights_gb, resolve_context_window,
)
from dual_gpu_setup.hostos import get_host
from dual_gpu_setup.host_info import default_config_name, detached_process_kwargs, os_info, os_log_line
from dual_gpu_setup.scheduling.leases import LeaseError, LeaseStore
from dual_gpu_setup.server import PLACEMENT_MIN_VRAM_FRACTION, LlamaServerProcess
from dual_gpu_setup.gguf import fingerprint as gguf_fingerprint, has_embedded_mtp
from dual_gpu_setup.osfilter import normalize_os
from dual_gpu_setup.reasoning import coding_request, model_reasoning
from dual_gpu_setup import datasafe
from dual_gpu_setup.canary import run_split_canary
from dual_gpu_setup.supervisor import CrashTracker
from dual_gpu_setup.logs import LOGGER_NAME, is_routine_poll, setup_service_logging
from dual_gpu_setup.process_registry import ProcessRegistry
from dual_gpu_setup.archive import HardwareArchive
from dual_gpu_setup.hardware_log import HardwareLog
from dual_gpu_setup.host_sensors import make_host_sensors
from dual_gpu_setup.training.amd_smi import make_reader as make_amd_smi_reader
from dual_gpu_setup.training.benchmarks import BenchmarkRunner
from dual_gpu_setup.training.evaluation import (
    AdapterServerManager, EvalQueue, EvalWorker, EvaluationPipeline,
)
from dual_gpu_setup.training.hardware import HardwareSampler, KernelFaultWatcher
from dual_gpu_setup.training.windows_gpu import WindowsGpuReader
from dual_gpu_setup.training.selection import select_best
from dual_gpu_setup.training.service import FileResponse, TrainingService
from dual_gpu_setup.training.store import TrainingError, TrainingStore


APP_VERSION = "0.1.0"
RUN_SCHEMA_VERSION = 2  # 1 = original; 2 adds origin_os / origin_os_inferred / model_fingerprint (add-only)
API_VERSION = "1"
DEFAULT_REASONING_PRESETS = {"off": 0, "low": 1024, "medium": 4096, "high": 8192}

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
CORS_ALLOWED_HEADERS = (
    "Content-Type, Authorization, X-DGPU-Project, X-DGPU-Agent-Role, X-DGPU-Session, "
    "X-DGPU-Task, X-DGPU-Task-Label, X-DGPU-Capture-Content, X-DGPU-No-History, "
    "X-DGPU-Connection-Type, X-DGPU-Target, X-Lease-Token"
)

# Documents exactly the fields ChatService already computes (see _normalize_usage,
# _normalize_timing, _resource_summary) and hands to clients unchanged via /api/chat's
# response and the client_id delivery file. Served at GET /api/clients/schema.
CLIENT_METRICS_SCHEMA = {
    "usage": {
        "input_tokens": "Prompt tokens sent to the model.",
        "cached_input_tokens": "Prompt tokens served from KV cache reuse, if reported by the backend.",
        "uncached_input_tokens": "input_tokens minus cached_input_tokens.",
        "prefill_tokens": "Tokens processed during the prefill/prompt phase.",
        "thinking_tokens": "Exact reasoning/thinking tokens as reported by the backend; null when it reports none.",
        "thinking_tokens_estimated": "Fallback ~4-chars/token estimate of thinking tokens, set only when the backend omits an exact count but reasoning text was observed; null otherwise.",
        "thinking_characters": "Observed reasoning-text characters (streamed deltas or a non-streamed message's reasoning); the basis for thinking_tokens_estimated.",
        "thinking_tokens_source": "backend (exact count), estimated_from_characters (fallback estimate), or unavailable (no reasoning observed).",
        "visible_output_tokens": "output_tokens minus thinking_tokens.",
        "tool_calls_total": "Number of tool calls the model emitted in this response.",
        "tool_calls_malformed": "Tool calls with no function name or unparseable JSON arguments; grammar/schema constraints reduce this.",
        "tool_call_valid_rate": "(tool_calls_total - tool_calls_malformed) / tool_calls_total; null when the response made no tool calls.",
        "output_tokens": "Total completion tokens generated.",
        "total_tokens": "input_tokens + output_tokens as reported by the backend.",
    },
    "timing": {
        "time_to_first_token_ms": "Latency before the first output token.",
        "prefill_duration_ms": "Wall time spent on the prefill/prompt phase.",
        "decode_duration_ms": "Wall time spent generating output tokens.",
        "end_to_end_duration_ms": "Total wall time for the request.",
        "prefill_tokens_per_second": "Prefill rate: prompt tokens / prefill duration.",
        "tokens_per_second": "Decode rate: output tokens / decode duration.",
        "end_to_end_tokens_per_second": "Output tokens / full request duration, including prefill; not decode speed.",
        "backend_prompt_tokens": "Prompt token count as reported natively by llama-server.",
        "backend_output_tokens": "Output token count as reported natively by llama-server.",
        "draft_tokens": "Tokens proposed by the speculative draft model; null when speculative decoding is off or unreported.",
        "draft_accepted_tokens": "Draft tokens the target model accepted; null when speculative decoding is off or unreported.",
        "draft_acceptance_rate": "draft_accepted_tokens / draft_tokens; higher means the draft is a better match and the speedup is larger. Null when unavailable.",
        "native_timing": "Raw timings object returned by the backend, unmodified.",
        "reasoning_duration_ms": "Observed time from first to last reasoning chunk in a streamed coding response; not native decode time.",
        "reasoning_duration_source": "observed_stream_span when reasoning chunks were observed, otherwise null.",
        "time_to_first_visible_token_ms": "Elapsed time until the first answer or tool-call chunk, excluding reasoning-only chunks.",
    },
    "resources": {
        "avg_process_cpu_pct": "Average CPU percent of the model process during the run.",
        "peak_process_cpu_pct": "Peak CPU percent of the model process during the run.",
        "avg_gpu_utilization_pct": "Average GPU utilization percent during the run.",
        "peak_gpu_utilization_pct": "Peak GPU utilization percent during the run.",
        "peak_process_rss_bytes": "Peak resident memory (working set) of the model process.",
        "peak_process_private_bytes": "Peak committed/private memory (pagefile usage) of the model process.",
        "process_page_fault_delta": "Page faults incurred by the model process during this run's window.",
        "peak_dedicated_vram_bytes": "Peak dedicated VRAM used by the model process.",
        "peak_shared_gpu_memory_bytes": "Peak shared/system GPU memory used (VRAM-spill indicator).",
        "spill_suspected": "True if shared GPU memory usage suggests VRAM spill.",
        "avg_system_cpu_pct": "Average whole-system CPU percent during the run.",
        "peak_system_ram_used_bytes": "Peak whole-system RAM used during the run.",
        "peak_system_pagefile_used_bytes": "Peak whole-system commit charge (pagefile usage) during the run.",
    },
}


# Endpoints that load, stop or reconfigure GPU work. They must carry a JSON content type, which makes a browser send a
# CORS preflight first, so another web page cannot trigger them with a hidden form POST.
JSON_ONLY_CONTROL_POSTS = frozenset({"/api/deploy", "/api/unload", "/api/models/refresh", "/api/lanes/auto-use"})


def kv_estimate(path: Path, context: int, *arg_lists: list[str] | None) -> tuple[float | None, str]:
    """KV-cache estimate that honours --cache-type-k/-v given on the model or lane command line."""
    k, v = cache_types_from_args(*arg_lists)
    return estimate_kv_cache_gb(path, context) if (k, v) == ("f16", "f16") else estimate_kv_cache_gb(path, context, k, v)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# This project replaced LM Studio's bundled runtime with a native llama.cpp build in commit
# 10650e3 ("replaced Lm studio with LLama.cpp"), committed 2026-10-17 11:36:35 +05:30 =
# 2026-10-17T06:06:35Z. Runs recorded since then carry an explicitly probed
# configuration.runtime.source ("llama.cpp" or "lmstudio"); older runs carry none, so they are
# attributed to LM Studio by their created_at timestamp. Likewise the reference machine ran
# Windows before host_os capture existed, so runs with no host_os default to Windows. Both are
# read-time labels only -- no stored run is rewritten.
LLAMACPP_MIGRATION_CUTOFF = datetime(2026, 10, 17, 6, 6, 35, tzinfo=UTC)


def _parse_created_at(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def infer_backend(run: dict[str, Any]) -> str:
    """Display label for the inference runtime a run used: 'llama.cpp' or 'LM Studio'.

    Prefers the runtime source probed at deploy time (present on every run since the migration).
    Runs predating that capture fall back to their created_at against the migration cutoff, which
    resolves every historical run to LM Studio."""
    source = ((run.get("configuration") or {}).get("runtime") or {}).get("source")
    if source == "llama.cpp":
        return "llama.cpp"
    if source == "lmstudio":
        return "LM Studio"
    created = _parse_created_at(run.get("created_at"))
    return "llama.cpp" if created and created >= LLAMACPP_MIGRATION_CUTOFF else "LM Studio"


def wall_clock_decode(runs: list[dict[str, Any]]) -> tuple[int, float]:
    """Output tokens and the wall-clock seconds spent decoding them, with overlapping decode
    windows counted once. Concurrent requests share one server, so summing their durations
    would count the same seconds several times and understate the lane's real throughput."""
    windows: list[tuple[float, float]] = []
    tokens = 0
    for run in runs:
        timing = run.get("timing") or {}
        output_tokens = (run.get("usage") or {}).get("output_tokens")
        decode_ms = timing.get("decode_duration_ms")
        finished = _parse_created_at(run.get("finished_at"))
        if run.get("status") != "completed" or not decode_ms or output_tokens is None or finished is None:
            continue
        end = finished.timestamp()
        windows.append((end - decode_ms / 1000, end))
        tokens += int(output_tokens)
    seconds = 0.0
    current_start = current_end = None
    for start, end in sorted(windows):
        if current_end is None or start > current_end:
            if current_end is not None:
                seconds += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        seconds += current_end - current_start
    return tokens, seconds


def infer_os(run: dict[str, Any]) -> str:
    """Host OS family a run executed on. Back-dated runs with no captured host_os default to
    Windows, the reference machine's OS before host_os capture existed."""
    system = (run.get("host_os") or {}).get("system")
    return str(system) if system else "Windows"


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def local_network_addresses() -> list[str]:
    """Return private IPv4 addresses that another LAN device can use."""
    candidates: set[str] = set()
    preferred: str | None = None
    try:
        candidates.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            preferred = str(probe.getsockname()[0])
            candidates.add(preferred)
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
    addresses.sort(key=lambda value: tuple(int(part) for part in value.split(".")))
    if preferred in addresses:
        addresses.remove(preferred)
        addresses.insert(0, preferred)
    return addresses


def network_metadata(bind_host: str, port: int) -> dict[str, Any]:
    """Describe how a remote application should reach this service.

    The llama-server ports intentionally remain loopback-only; the public integration surface is
    the gateway on ``port``.  Keeping this shape in the API means clients do not need to infer
    whether a deployment is local or LAN-backed from the process command line.
    """
    lan_enabled = bind_host not in LOOPBACK_HOSTS
    host_for_url = "127.0.0.1" if bind_host in {"0.0.0.0", "::"} else bind_host
    base_url = f"http://{host_for_url}:{port}"
    advertised = (
        [f"http://{address}:{port}" for address in local_network_addresses()]
        if lan_enabled
        else []
    )
    advertised_openai = [f"{url}/v1" for url in advertised]
    return {
        "bind_host": bind_host,
        "port": int(port),
        "lan_enabled": lan_enabled,
        "base_url": base_url,
        # Prefer a usable LAN address when one is available; otherwise fall back to loopback.
        "openai_base_url": advertised_openai[0] if advertised_openai else f"{base_url}/v1",
        "advertised_urls": advertised,
        "advertised_openai_base_urls": advertised_openai,
        "model_server_ports": "loopback_only",
    }


class ResourceSampler:
    """Collect lightweight process/system metrics and GPU memory counters via the host-OS layer."""

    def __init__(self, manager: "DeploymentManager", interval_seconds: float = 2.0):
        self.manager = manager
        self.interval_seconds = interval_seconds
        self.samples: deque[dict[str, Any]] = deque(maxlen=3600)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

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
        live = [(target, handle) for target, handle in handles if handle.process.proc is not None]
        # One batched GPU-metrics call for every active server's PID instead of one per
        # server (on Windows the PowerShell spawn + Get-Counter cost dominates this loop).
        gpu_by_pid = get_host().gpu.process_metrics([handle.process.pid for _, handle in live])
        servers = []
        for target, handle in live:
            process = handle.process
            pid = process.pid
            process_metrics = self._process_metrics(pid, now)
            gpu = gpu_by_pid.get(pid)
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
                    "peak_rss_bytes": process_metrics.get("peak_rss_bytes"),
                    "peak_private_bytes": process_metrics.get("peak_private_bytes"),
                    "page_fault_count": process_metrics.get("page_fault_count"),
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
        return get_host().process.metrics(pid, now)

    def _system_metrics(self) -> dict[str, Any]:
        return get_host().system.metrics()


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
            found = connection.execute("PRAGMA user_version").fetchone()[0]
            if found > RUN_SCHEMA_VERSION:
                raise RuntimeError(f"{self.path} is schema v{found}, newer than this code (v{RUN_SCHEMA_VERSION}). "
                                   "Update Dual-GPU Studio before opening it; nothing was changed.")
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
                    host_os_json TEXT,
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
                CREATE TABLE IF NOT EXISTS clients (
                    id TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    github_repo TEXT,
                    output_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT,
                    project_id TEXT
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    repo_path TEXT,
                    git_remote TEXT,
                    created_at TEXT NOT NULL,
                    archived_at TEXT
                );
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    runtime TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                );
                CREATE TABLE IF NOT EXISTS coding_tasks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    branch TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                );
                CREATE TABLE IF NOT EXISTS tool_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    session_id TEXT,
                    task_id TEXT,
                    run_id TEXT,
                    occurred_at TEXT NOT NULL,
                    tool_type TEXT NOT NULL,
                    duration_ms REAL,
                    exit_code INTEGER,
                    status TEXT NOT NULL,
                    detail_json TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects(id),
                    FOREIGN KEY (session_id) REFERENCES agent_sessions(id),
                    FOREIGN KEY (task_id) REFERENCES coding_tasks(id),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_comparison_id ON runs(comparison_id);
                CREATE INDEX IF NOT EXISTS idx_metric_samples_run_id ON metric_samples(run_id, id);
                CREATE INDEX IF NOT EXISTS idx_sessions_project_id ON agent_sessions(project_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON coding_tasks(project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tool_events_project_id ON tool_events(project_id, occurred_at DESC);
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
                "client_id": "TEXT",
                "task_label": "TEXT",
                "project_id": "TEXT",
                "agent_session_id": "TEXT",
                "coding_task_id": "TEXT",
                "agent_role": "TEXT",
                "workload_kind": "TEXT NOT NULL DEFAULT 'evaluation'",
                "host_os_json": "TEXT",
                # explicit OS tag so a combined Windows + Linux history never has to guess
                "origin_os": "TEXT",
                "origin_os_inferred": "INTEGER NOT NULL DEFAULT 0",
                # file identity independent of path, so models match across Windows and Linux
                "model_fingerprint": "TEXT",
            }
            for name, data_type in migrations.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {data_type}")
            self._tag_untagged_runs(connection)
            connection.execute(f"PRAGMA user_version = {RUN_SCHEMA_VERSION}")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_deployment_id ON runs(deployment_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_client_id ON runs(client_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_task_label ON runs(task_label)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id, created_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_origin_os ON runs(origin_os, created_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(agent_session_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_coding_task_id ON runs(coding_task_id)")
            client_columns = {row[1] for row in connection.execute("PRAGMA table_info(clients)")}
            if "project_id" not in client_columns:
                connection.execute("ALTER TABLE clients ADD COLUMN project_id TEXT")

    @staticmethod
    def _tag_untagged_runs(connection: sqlite3.Connection) -> None:
        """Fill the new `origin_os` column (NULL -> value only; nothing already stored is changed).
        A run that captured host_os keeps that system; older runs predate capture and ran on Windows,
        which is recorded with `origin_os_inferred = 1` so the guess stays distinguishable."""
        connection.execute(
            "UPDATE runs SET origin_os = json_extract(host_os_json, '$.system'), origin_os_inferred = 0 "
            "WHERE origin_os IS NULL AND host_os_json IS NOT NULL AND json_valid(host_os_json) "
            "AND json_extract(host_os_json, '$.system') IS NOT NULL"
        )
        connection.execute("UPDATE runs SET origin_os = 'Windows', origin_os_inferred = 1 WHERE origin_os IS NULL")

    def mark_interrupted(self) -> int:
        """Runs still 'running' when the service starts cannot be running: their process died with
        the previous service. Mark them 'interrupted' (kept, never deleted) instead of leaving them
        'running' forever. Call once at startup, before requests are served."""
        with self.lock, self.connect() as connection:
            return connection.execute(
                "UPDATE runs SET status = 'interrupted', finished_at = COALESCE(finished_at, ?), "
                "error_text = COALESCE(error_text, 'The service stopped before this request finished.') "
                "WHERE status = 'running'",
                (utc_now(),),
            ).rowcount

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
                    reasoning_budget, request_json, configuration_json, client_id, task_label,
                    project_id, agent_session_id, coding_task_id, agent_role, workload_kind,
                    host_os_json, origin_os, model_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.get("client_id"),
                    record.get("task_label"),
                    record.get("project_id"),
                    record.get("agent_session_id"),
                    record.get("coding_task_id"),
                    record.get("agent_role"),
                    record.get("workload_kind", "evaluation"),
                    json.dumps(record.get("host_os") or os_info(), ensure_ascii=False),
                    (record.get("host_os") or os_info()).get("system"),
                    gguf_fingerprint(record.get("model_path")),
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

    def upsert_project(
        self,
        name: str,
        repo_path: str | None = None,
        git_remote: str | None = None,
    ) -> dict[str, Any]:
        normalized = name.strip()
        if not normalized:
            raise ValueError("Project name is required.")
        with self.lock, self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE name = ? COLLATE NOCASE", (normalized,)
            ).fetchone()
            if row is None:
                project_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT INTO projects (id, name, repo_path, git_remote, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (project_id, normalized, repo_path, git_remote, utc_now()),
                )
            else:
                project_id = row["id"]
                connection.execute(
                    """
                    UPDATE projects SET repo_path = COALESCE(?, repo_path),
                        git_remote = COALESCE(?, git_remote)
                    WHERE id = ?
                    """,
                    (repo_path, git_remote, project_id),
                )
            result = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(result)

    def projects(self) -> list[dict[str, Any]]:
        with self.lock, self.connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*,
                    COUNT(r.id) AS run_count,
                    COALESCE(SUM(CAST(json_extract(r.usage_json, '$.total_tokens') AS INTEGER)), 0) AS total_tokens,
                    MAX(r.created_at) AS last_run_at
                FROM projects p
                LEFT JOIN runs r ON r.project_id = p.id
                WHERE p.archived_at IS NULL
                GROUP BY p.id
                ORDER BY COALESCE(MAX(r.created_at), p.created_at) DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, project_id_or_name: str) -> dict[str, Any] | None:
        with self.lock, self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE id = ? OR name = ? COLLATE NOCASE",
                (project_id_or_name, project_id_or_name),
            ).fetchone()
        return dict(row) if row else None

    def create_agent_session(self, project_id: str, agent_role: str, runtime: str) -> dict[str, Any]:
        session = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "agent_role": agent_role.strip() or "developer",
            "runtime": runtime.strip() or "unknown",
            "started_at": utc_now(),
            "status": "active",
        }
        with self.lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_sessions (id, project_id, agent_role, runtime, started_at, status)
                VALUES (:id, :project_id, :agent_role, :runtime, :started_at, :status)
                """,
                session,
            )
        return session

    def create_coding_task(
        self, project_id: str, title: str, branch: str | None = None
    ) -> dict[str, Any]:
        task = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "title": title.strip(),
            "status": "active",
            "branch": branch,
            "created_at": utc_now(),
        }
        if not task["title"]:
            raise ValueError("Task title is required.")
        with self.lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO coding_tasks (id, project_id, title, status, branch, created_at)
                VALUES (:id, :project_id, :title, :status, :branch, :created_at)
                """,
                task,
            )
        return task

    def record_tool_event(self, event: dict[str, Any]) -> dict[str, Any]:
        record = {
            "project_id": event["project_id"],
            "session_id": event.get("session_id"),
            "task_id": event.get("task_id"),
            "run_id": event.get("run_id"),
            "occurred_at": event.get("occurred_at") or utc_now(),
            "tool_type": str(event.get("tool_type") or "unknown"),
            "duration_ms": event.get("duration_ms"),
            "exit_code": event.get("exit_code"),
            "status": str(event.get("status") or "completed"),
            "detail_json": json.dumps(event.get("detail"), ensure_ascii=False),
        }
        with self.lock, self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tool_events (
                    project_id, session_id, task_id, run_id, occurred_at, tool_type,
                    duration_ms, exit_code, status, detail_json
                ) VALUES (
                    :project_id, :session_id, :task_id, :run_id, :occurred_at, :tool_type,
                    :duration_ms, :exit_code, :status, :detail_json
                )
                """,
                record,
            )
            record["id"] = cursor.lastrowid
        record["detail"] = json.loads(record.pop("detail_json"))
        return record

    def register_client(self, project_name: str, github_repo: str | None, output_path: str) -> dict[str, Any]:
        project = self.upsert_project(project_name, git_remote=github_repo)
        client_id = str(uuid.uuid4())
        created_at = utc_now()
        with self.lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO clients (
                    id, project_name, github_repo, output_path, created_at, last_seen_at, project_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (client_id, project_name, github_repo, output_path, created_at, created_at, project["id"]),
            )
        return {
            "client_id": client_id,
            "project_id": project["id"],
            "project_name": project_name,
            "github_repo": github_repo,
            "output_path": output_path,
            "created_at": created_at,
        }

    def get_client(self, client_id: str) -> dict[str, Any] | None:
        with self.lock, self.connect() as connection:
            row = connection.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()
        return dict(row) if row else None

    def touch_client(self, client_id: str) -> None:
        with self.lock, self.connect() as connection:
            connection.execute("UPDATE clients SET last_seen_at = ? WHERE id = ?", (utc_now(), client_id))

    def link_client_project(self, client_id: str, project_id: str) -> None:
        with self.lock, self.connect() as connection:
            connection.execute("UPDATE clients SET project_id = ? WHERE id = ?", (project_id, client_id))

    def runs_for_client(self, client_id: str, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs WHERE client_id = ? ORDER BY created_at DESC LIMIT ?",
                (client_id, limit),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def prune_older_than(self, days: int) -> dict[str, Any]:
        """Delete runs (and their metric_samples, via ON DELETE CASCADE) and deployments
        older than `days`. Opt-in only -- nothing calls this unless the operator passes
        --prune-older-than-days, so the default stays "keep everything forever". There is
        no automatic retention policy otherwise: this database grows without bound."""
        if days <= 0:
            raise ValueError("days must be positive.")
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="milliseconds")
        with self.lock, self.connect() as connection:
            deleted_runs = connection.execute("DELETE FROM runs WHERE created_at < ?", (cutoff,)).rowcount
            deleted_deployments = connection.execute(
                "DELETE FROM deployments WHERE created_at < ?", (cutoff,)
            ).rowcount
        return {"cutoff": cutoff, "deleted_runs": deleted_runs, "deleted_deployments": deleted_deployments}

    def recent(
        self,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
        workload_kind: str | None = None,
        host_os: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        conditions: list[str] = []
        params: list[Any] = []
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if workload_kind:
            conditions.append("workload_kind = ?")
            params.append(workload_kind)
        if host_os:
            conditions.append("origin_os = ?")
            params.append(host_os)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self.lock, self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def count_runs(self, project_id: str | None = None, workload_kind: str | None = None,
                   host_os: str | None = None) -> int:
        conditions: list[str] = []
        params: list[Any] = []
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if workload_kind:
            conditions.append("workload_kind = ?")
            params.append(workload_kind)
        if host_os:
            conditions.append("origin_os = ?")
            params.append(host_os)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self.lock, self.connect() as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM runs{where}", params).fetchone()[0])

    @staticmethod
    def _day_after(date_str: str) -> str:
        # Exclusive upper bound so an inclusive end date covers its whole day. Parsing
        # also validates the YYYY-MM-DD format (ValueError on anything else).
        return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    def _dashboard_runs(
        self, project_id: str | None = None, workload_kind: str | None = None,
        start_date: str | None = None, end_date: str | None = None, host_os: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if workload_kind:
            conditions.append("workload_kind = ?")
            params.append(workload_kind)
        if host_os:
            conditions.append("origin_os = ?")
            params.append(host_os)
        # created_at is ISO text, so lexicographic bounds match chronological ones and use
        # the created_at index. start is inclusive; end covers its whole day.
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at < ?")
            params.append(self._day_after(end_date))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        # WAL readers need not hold the writer lock. Do not copy/parse large prompt,
        # completion, or backend-response bodies for performance aggregates.
        with self.connect() as connection:
            rows = connection.execute(f"""
                SELECT id, created_at, status, mode, target, model_name, lane_keys_json,
                    context_window, reasoning_budget, project_id, agent_role, task_label,
                    workload_kind, usage_json, timing_json, resources_json, host_os_json, finished_at,
                    json_object(
                        'reasoning', json_extract(configuration_json, '$.reasoning'),
                        'runtime', json_object(
                            'source', json_extract(configuration_json, '$.runtime.source'),
                            'build', json_extract(configuration_json, '$.runtime.build')
                        )
                    ) AS configuration_json
                FROM runs{where} ORDER BY created_at DESC
            """, params).fetchall()
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

    def dashboard(
        self, project_id: str | None = None, workload_kind: str | None = None,
        start_date: str | None = None, end_date: str | None = None, host_os: str | None = None,
    ) -> dict[str, Any]:
        # Dashboard aggregates intentionally use the complete matching history (optionally
        # narrowed to a date range). The prior implementation called recent(1000), which
        # was silently capped to 500 records.
        runs = self._dashboard_runs(project_id, workload_kind, start_date, end_date, host_os)

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

        def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
            group_completed = [run for run in group if run.get("status") == "completed"]
            timed = [run for run in group_completed if (run.get("timing") or {}).get("decode_duration_ms", 0)
                     and (run.get("usage") or {}).get("output_tokens") is not None]
            output_tokens = total(("usage", "output_tokens"), timed)
            decode_ms = sum(nums(("timing", "decode_duration_ms"), timed))
            wall_tokens, wall_seconds = wall_clock_decode(group)
            return {
                "runs": len(group),
                "success_rate": round(len(group_completed) / len(group) * 100, 1) if group else None,
                "input_tokens": total(("usage", "input_tokens"), group),
                "output_tokens": total(("usage", "output_tokens"), group),
                "thinking_tokens": total(("usage", "thinking_tokens"), group),
                "thinking_tokens_reported_runs": len(nums(("usage", "thinking_tokens"), group_completed)),
                "avg_thinking_tokens": average(("usage", "thinking_tokens"), group_completed),
                "thinking_tokens_estimated_runs": len(nums(("usage", "thinking_tokens_estimated"), group_completed)),
                "avg_thinking_tokens_estimated": average(("usage", "thinking_tokens_estimated"), group_completed),
                "avg_thinking_characters": average(("usage", "thinking_characters"), group_completed),
                "avg_reasoning_duration_ms": average(("timing", "reasoning_duration_ms"), group_completed),
                "avg_tokens_per_second": average(("timing", "tokens_per_second"), group_completed),
                "avg_draft_acceptance_rate": average(("timing", "draft_acceptance_rate"), group_completed),
                "tool_calls_total": total(("usage", "tool_calls_total"), group_completed),
                "tool_calls_malformed": total(("usage", "tool_calls_malformed"), group_completed),
                "avg_tool_call_valid_rate": average(("usage", "tool_call_valid_rate"), group_completed),
                "aggregate_tokens_per_second": (
                    round(output_tokens / (decode_ms / 1000), 2) if decode_ms > 0 else None
                ),
                # Output tokens over wall-clock decode time with overlaps counted once. This is the
                # throughput the device delivered; the per-request figures above are each request's
                # own rate and fall as more requests share the server.
                "wall_clock_tokens_per_second": (
                    round(wall_tokens / wall_seconds, 2) if wall_seconds > 0 else None
                ),
                "avg_concurrent_requests": average(("timing", "concurrent_requests_at_start"), group_completed),
                "avg_prefill_tokens_per_second": average(
                    ("timing", "prefill_tokens_per_second"), group_completed
                ),
                "avg_latency_ms": average(("timing", "end_to_end_duration_ms"), group_completed),
                "avg_gpu_utilization_pct": average(
                    ("resources", "avg_gpu_utilization_pct"), group_completed
                ),
                "peak_vram_bytes": max(nums(("resources", "peak_dedicated_vram_bytes"), group) or [0]),
            }

        models: list[dict[str, Any]] = []
        for model_name in sorted({run["model_name"] for run in runs}):
            group = [run for run in runs if run["model_name"] == model_name]
            models.append({"model": model_name, **summarize(group)})

        # task_label is optional, freeform metadata a caller passes to /api/chat (see
        # CLIENT_API.md) to tag what kind of work a call was -- e.g. "relevance",
        # "resume_extraction", "fraud_d2_fusion". Runs that never set it are left out of
        # these two breakdowns (they still count in `models` and the overall `summary`).
        labeled_runs = [run for run in runs if run.get("task_label")]
        categories: list[dict[str, Any]] = []
        for label in sorted({run["task_label"] for run in labeled_runs}):
            group = [run for run in labeled_runs if run["task_label"] == label]
            categories.append({"category": label, **summarize(group)})

        category_models: list[dict[str, Any]] = []
        pairs = sorted({(run["task_label"], run["model_name"]) for run in labeled_runs})
        for label, model_name in pairs:
            group = [
                run for run in labeled_runs
                if run["task_label"] == label and run["model_name"] == model_name
            ]
            category_models.append({"category": label, "model": model_name, **summarize(group)})

        gpu_lanes: list[dict[str, Any]] = []
        for lane in sorted({lane for run in runs for lane in (run.get("lane_keys") or [])}):
            group = [run for run in runs if lane in (run.get("lane_keys") or [])]
            gpu_lanes.append({"gpu": lane, **summarize(group)})

        project_rows: list[dict[str, Any]] = []
        known_projects = {project["id"]: project for project in self.projects()}
        project_keys_set: set[str | None] = set(known_projects) if not project_id else {project_id}
        project_keys_set.update(run.get("project_id") for run in runs)
        project_keys = sorted(project_keys_set, key=lambda value: str(value or ""))
        for key in project_keys:
            group = [run for run in runs if run.get("project_id") == key]
            project = known_projects.get(key or "", {})
            project_rows.append(
                {
                    "project_id": key,
                    "project": project.get("name") or "Unassigned / Legacy",
                    **summarize(group),
                }
            )

        agent_rows: list[dict[str, Any]] = []
        for role in sorted({run.get("agent_role") for run in runs if run.get("agent_role")}):
            group = [run for run in runs if run.get("agent_role") == role]
            agent_rows.append({"agent_role": role, **summarize(group)})

        backend_rows: list[dict[str, Any]] = []
        for backend in sorted({run["backend"] for run in runs}):
            group = [run for run in runs if run["backend"] == backend]
            backend_rows.append({"backend": backend, **summarize(group)})

        os_rows: list[dict[str, Any]] = []
        for os_name in sorted({run["os"] for run in runs}):
            group = [run for run in runs if run["os"] == os_name]
            os_rows.append({"os": os_name, **summarize(group)})

        tool_summary = {"events": 0, "failed": 0, "success_rate": None}
        reasoning_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for run in runs:
            reasoning = (run.get("configuration") or {}).get("reasoning") or {}
            if run.get("workload_kind") == "coding_agent":
                level = reasoning.get("effective_effort") or reasoning.get("effort") or "unknown"
                reasoning_groups.setdefault((run["model_name"], level), []).append(run)
        reasoning_levels = [{"model": model, "effort": effort, **summarize(group)}
                            for (model, effort), group in sorted(reasoning_groups.items())]
        tool_conditions: list[str] = []
        tool_params: list[Any] = []
        if project_id:
            tool_conditions.append("project_id = ?")
            tool_params.append(project_id)
        if start_date:
            tool_conditions.append("occurred_at >= ?")
            tool_params.append(start_date)
        if end_date:
            tool_conditions.append("occurred_at < ?")
            tool_params.append(self._day_after(end_date))
        tool_where = f" WHERE {' AND '.join(tool_conditions)}" if tool_conditions else ""
        with self.lock, self.connect() as connection:
            tool_row = connection.execute(
                f"""
                SELECT COUNT(*) AS events,
                    SUM(CASE WHEN status IN ('failed', 'error', 'timeout') OR COALESCE(exit_code, 0) != 0 THEN 1 ELSE 0 END) AS failed
                FROM tool_events{tool_where}
                """,
                tool_params,
            ).fetchone()
        if tool_row:
            event_count = int(tool_row["events"] or 0)
            failed_count = int(tool_row["failed"] or 0)
            tool_summary = {
                "events": event_count,
                "failed": failed_count,
                "success_rate": round((event_count - failed_count) / event_count * 100, 1) if event_count else None,
            }

        daily: dict[str, dict[str, Any]] = {}
        for run in reversed(runs):
            day = str(run.get("created_at", ""))[:10]
            bucket = daily.setdefault(day, {"date": day, "runs": 0, "tokens": 0})
            bucket["runs"] += 1
            if isinstance(run.get("usage"), dict):
                bucket["tokens"] += int(run["usage"].get("total_tokens") or 0)
        # Default view keeps the last 14 days tidy; an explicit date range shows every day in it.
        daily_list = list(daily.values()) if (start_date or end_date) else list(daily.values())[-14:]
        return {
            "generated_at": utc_now(),
            "filters": {"start_date": start_date, "end_date": end_date, "os": host_os},
            "summary": {
                "total_runs": len(runs),
                "completed_runs": len(completed),
                "failed_runs": len(runs) - len(completed),
                "success_rate": round(len(completed) / len(runs) * 100, 1) if runs else None,
                "models_used": len({run["model_name"] for run in runs}),
                "input_tokens": total(("usage", "input_tokens")),
                "output_tokens": total(("usage", "output_tokens")),
                "thinking_tokens": total(("usage", "thinking_tokens")),
                # Exact sum above is 0 both when nothing reasoned and when the backend never
                # reports a count (llama.cpp), so expose coverage and the labeled estimate.
                "thinking_tokens_reported_runs": len(nums(("usage", "thinking_tokens"))),
                "thinking_tokens_estimated": total(("usage", "thinking_tokens_estimated")),
                "thinking_tokens_estimated_runs": len(nums(("usage", "thinking_tokens_estimated"))),
                "visible_output_tokens": total(("usage", "visible_output_tokens")),
                "total_tokens": total(("usage", "total_tokens")),
                "avg_tokens_per_second": average(("timing", "tokens_per_second"), completed),
                "aggregate_tokens_per_second": summarize(runs)["aggregate_tokens_per_second"],
                "wall_clock_tokens_per_second": summarize(runs)["wall_clock_tokens_per_second"],
                "avg_concurrent_requests": summarize(runs)["avg_concurrent_requests"],
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
            "gpus": gpu_lanes,
            "projects": project_rows,
            "agents": agent_rows,
            "backends": backend_rows,
            "operating_systems": os_rows,
            "reasoning_levels": reasoning_levels,
            "tool_summary": tool_summary,
            "categories": categories,
            "category_models": category_models,
            "daily": daily_list,
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
            "host_os_json",
        ):
            raw = item.pop(key, None)
            output_key = key.removesuffix("_json")
            try:
                item[output_key] = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                item[output_key] = None
        # Read-time labels derived from the runtime source and host OS already on the row, with
        # back-dated defaults (LM Studio / Windows) for runs predating that capture. See
        # infer_backend/infer_os. Nothing persisted is rewritten.
        item["backend"] = infer_backend(item)
        item["os"] = item.get("origin_os") or infer_os(item)
        return item


class ChatHistoryStore:
    """Append-only full-conversation archive in its OWN SQLite file.

    Design goals: (1) never slow the inference hot path, (2) keep the metrics RunStore lean.
    So content (full prompts, responses, reasoning, tool calls) plus token counts are written
    here, not into the metrics DB, and the write happens on a background worker: callers only
    pay an O(1) queue put, and JSON serialization + the DB commit (batched) happen off-thread.
    When disabled, no file, thread, or queue work exists -- `enqueue` is a no-op.
    """

    _SENTINEL = object()
    _COLUMNS = (
        "id", "created_at", "project_id", "client_id", "agent_role", "agent_session_id",
        "coding_task_id", "model_name", "target", "workload_kind",
        "input_tokens", "cached_input_tokens", "output_tokens", "thinking_tokens", "total_tokens",
        "tokens_per_second", "latency_ms", "finish_reason", "status",
        "request_json",
        "messages_json", "tools_json", "output_text", "reasoning_text", "tool_calls_json",
        "resources_json", "error_text",
    )

    def __init__(self, path: Path, enabled: bool = False):
        self.path = path
        self.enabled = enabled
        # Unbounded producer queue: enqueue stays O(1) and inference never waits on storage, while
        # every model request remains eligible for durable logging instead of being silently dropped.
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        # health counters: batches that fell back to the JSONL file, and records that could not be saved anywhere
        self.write_failures = 0
        self.records_lost = 0
        if not enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._worker = threading.Thread(target=self._run, name="chat-history", daemon=True)
        self._worker.start()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    project_id TEXT,
                    client_id TEXT,
                    agent_role TEXT,
                    agent_session_id TEXT,
                    coding_task_id TEXT,
                    model_name TEXT,
                    target TEXT,
                    workload_kind TEXT,
                    input_tokens INTEGER,
                    cached_input_tokens INTEGER,
                    output_tokens INTEGER,
                    thinking_tokens INTEGER,
                    total_tokens INTEGER,
                    tokens_per_second REAL,
                    latency_ms REAL,
                    finish_reason TEXT,
                    status TEXT,
                    request_json TEXT,
                    messages_json TEXT,
                    tools_json TEXT,
                    output_text TEXT,
                    reasoning_text TEXT,
                    tool_calls_json TEXT,
                    resources_json TEXT,
                    error_text TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_chat_turns_created_at ON chat_turns(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chat_turns_project ON chat_turns(project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chat_turns_model ON chat_turns(model_name, created_at DESC);
                """
            )
            existing = {row[1] for row in connection.execute("PRAGMA table_info(chat_turns)")}
            for column, kind in {
                "status": "TEXT",
                "request_json": "TEXT",
                "resources_json": "TEXT",
                "error_text": "TEXT",
            }.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE chat_turns ADD COLUMN {column} {kind}")

    def enqueue(self, record: dict[str, Any]) -> None:
        # Hot path: one non-blocking put; serialization and persistence stay on the worker.
        if not self.enabled:
            return
        self._queue.put_nowait(record)

    @staticmethod
    def _row(record: dict[str, Any]) -> tuple[Any, ...]:
        usage = record.get("usage") or {}
        timing = record.get("timing") or {}

        def dump(value: Any) -> str | None:
            return json.dumps(value, ensure_ascii=False) if value is not None else None

        return (
            record.get("id"), record.get("created_at"), record.get("project_id"),
            record.get("client_id"), record.get("agent_role"), record.get("agent_session_id"),
            record.get("coding_task_id"), record.get("model_name"), record.get("target"),
            record.get("workload_kind"),
            usage.get("input_tokens"), usage.get("cached_input_tokens"), usage.get("output_tokens"),
            usage.get("thinking_tokens"), usage.get("total_tokens"),
            timing.get("tokens_per_second"), timing.get("end_to_end_duration_ms"),
            record.get("finish_reason"), record.get("status", "completed"),
            dump(record.get("request")),
            dump(record.get("messages")), dump(record.get("tools")),
            record.get("output_text"), record.get("reasoning_text"), dump(record.get("tool_calls")),
            dump(record.get("resources")), record.get("error_text"),
        )

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        placeholders = ", ".join(["?"] * len(self._COLUMNS))
        statement = f"INSERT OR REPLACE INTO chat_turns ({', '.join(self._COLUMNS)}) VALUES ({placeholders})"
        for attempt in range(3):
            try:
                with self.connect() as connection:
                    connection.executemany(statement, [self._row(item) for item in batch])
                return
            except sqlite3.Error:
                if attempt < 2:
                    time.sleep(0.05 * (2**attempt))
        # Preserve the records even when SQLite remains unavailable. This worker is the only
        # writer, so appending JSONL needs no additional lock and never touches inference latency.
        self.write_failures += 1
        fallback = self.path.with_name(self.path.stem + ".fallback.jsonl")
        try:
            with fallback.open("a", encoding="utf-8") as stream:
                for item in batch:
                    stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            self.records_lost += len(batch)
            print(f"[history] could not persist {len(batch)} record(s): {exc}", flush=True)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                self._queue.task_done()
                return
            batch = [item]
            while len(batch) < 200:  # batch whatever else is waiting into one commit
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is self._SENTINEL:
                    self._flush(batch)
                    for _ in batch:
                        self._queue.task_done()
                    self._queue.task_done()
                    return
                batch.append(nxt)
            self._flush(batch)
            for _ in batch:
                self._queue.task_done()

    def flush(self) -> None:
        """Block until every queued record has been written (used by tests/shutdown)."""
        if self.enabled:
            self._queue.join()

    def close(self) -> None:
        if self._worker is None:
            return
        self._queue.put(self._SENTINEL)
        self._worker.join(timeout=10)

    def query(self, project_id: str | None = None, start_date: str | None = None,
              end_date: str | None = None, limit: int = 100, offset: int = 0,
              include_content: bool = False, model_name: str | None = None,
              target: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        content_columns = {
            "request_json", "messages_json", "tools_json", "output_text", "reasoning_text",
            "tool_calls_json", "resources_json",
        }
        columns = [col for col in self._COLUMNS if include_content or col not in content_columns]
        conditions: list[str] = []
        params: list[Any] = []
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if target:
            conditions.append("target = ?")
            params.append(target)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at < ?")
            params.append(RunStore._day_after(end_date))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(columns)} FROM chat_turns{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, int(limit), int(offset)],
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if include_content:
                for key in ("request_json", "messages_json", "tools_json", "tool_calls_json", "resources_json"):
                    if item.get(key) is not None:
                        item[key[:-5]] = json.loads(item.pop(key))
                    else:
                        item.pop(key, None)
            result.append(item)
        return result


class UsageLedger:
    """Per-project, per-connection-type usage rollup in its own SQLite file.

    Records requests and token totals for EVERY connection -- coding agents, model inference,
    inline autocomplete, and any client that self-tags via X-DGPU-Connection-Type (e.g. a
    training/data pipeline). To keep it optimised and bloat-free it is a DAILY ROLLUP, not a
    per-request row: counts accumulate in memory and a background worker flushes them, so even
    high-frequency autocomplete adds no DB write on the request path and the table stays small
    (one row per project x connection-type x day).
    """

    def __init__(self, path: Path, enabled: bool = True, flush_interval: float = 5.0):
        self.path = path
        self.enabled = enabled
        self._pending: dict[tuple[str, str, str], dict[str, int]] = {}
        self.write_failures = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._flush_interval = flush_interval
        self._worker: threading.Thread | None = None
        if not enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._worker = threading.Thread(target=self._run, name="usage-ledger", daemon=True)
        self._worker.start()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS project_usage (
                    project_id TEXT NOT NULL,
                    connection_type TEXT NOT NULL,
                    usage_date TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, connection_type, usage_date)
                );
                CREATE INDEX IF NOT EXISTS idx_project_usage_date ON project_usage(usage_date DESC);
                """
            )

    def record(self, project_id: str | None, connection_type: str,
               input_tokens: int | None = 0, output_tokens: int | None = 0, requests: int = 1) -> None:
        # Hot path: an O(1) in-memory accumulate under a short lock. No DB touch here.
        if not self.enabled:
            return
        key = (project_id or "", connection_type or "unknown", utc_now()[:10])
        with self._lock:
            slot = self._pending.setdefault(key, {"requests": 0, "input_tokens": 0, "output_tokens": 0})
            slot["requests"] += requests
            slot["input_tokens"] += int(input_tokens or 0)
            slot["output_tokens"] += int(output_tokens or 0)

    def _run(self) -> None:
        while not self._stop.wait(self._flush_interval):
            self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            pending, self._pending = self._pending, {}
        if not pending:
            return
        now = utc_now()
        rows = [(p, c, d, v["requests"], v["input_tokens"], v["output_tokens"], now)
                for (p, c, d), v in pending.items()]
        try:
            with self.connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO project_usage
                        (project_id, connection_type, usage_date, requests, input_tokens, output_tokens, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, connection_type, usage_date) DO UPDATE SET
                        requests = requests + excluded.requests,
                        input_tokens = input_tokens + excluded.input_tokens,
                        output_tokens = output_tokens + excluded.output_tokens,
                        updated_at = excluded.updated_at
                    """,
                    rows,
                )
        except sqlite3.Error:
            # Never lose counts over a transient write error: fold them back in.
            self.write_failures += 1
            with self._lock:
                for (p, c, d), v in pending.items():
                    slot = self._pending.setdefault((p, c, d), {"requests": 0, "input_tokens": 0, "output_tokens": 0})
                    for field, value in v.items():
                        slot[field] += value

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5)
        self.flush()

    def query(self, project_id: str | None = None, connection_type: str | None = None,
              start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        self.flush()  # include not-yet-flushed counts so reads are current
        conditions: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if connection_type:
            conditions.append("connection_type = ?")
            params.append(connection_type)
        if start_date:
            conditions.append("usage_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("usage_date <= ?")  # usage_date is date-only; end is inclusive
            params.append(end_date)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT project_id, connection_type, usage_date, requests, input_tokens, output_tokens, updated_at
                    FROM project_usage{where}
                    ORDER BY usage_date DESC, project_id, connection_type""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]


class AppError(RuntimeError):
    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


class ServerHandle:
    # A gateway request claims its handle when it is routed and converts the claim when it
    # starts serving; a claim that is never converted (the request failed in between) stops
    # protecting the handle from eviction after this long.
    CLAIM_TTL_SECONDS = 60.0

    def __init__(self, target: str, model: ModelConfig, lanes: list[LaneConfig], process: LlamaServerProcess,
                 workload_kind: str | None = None):
        self.target = target
        self.model = model
        self.lanes = lanes
        self.process = process
        self.workload_kind = workload_kind  # "coding_agent" when loaded for the gateway; kept so a reload matches
        self._reasoning_capabilities: dict[str, Any] | None = None
        self._inflight = 0
        self._claims: list[float] = []
        self._inflight_lock = threading.Lock()
        self.last_used = time.monotonic()
        # Residency guard bookkeeping (Windows can page an idle server's VRAM out to RAM).
        self.resident_gb: float | None = None
        self.evictions = 0
        self.rewarmed_at: str | None = None

    def warm_slot(self) -> int | None:
        probe = getattr(self.process, "warm_slot", None)
        return probe() if callable(probe) else None

    def pin_serving_slot(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Real traffic always runs on slot 0 when the server reserves a warm-up slot: a warm-up
        can then never overwrite a conversation's cached prompt, and requests still queue one at a
        time exactly as with a single slot."""
        if self.warm_slot() is not None:
            payload["id_slot"] = 0
        return payload

    def claim(self) -> None:
        """Reserve this server for a request that has been routed but not yet sent, so a
        concurrent on-demand load cannot evict it in between."""
        with self._inflight_lock:
            self._claims.append(time.monotonic())
            self.last_used = time.monotonic()

    @contextmanager
    def serving(self, claimed: bool = False):
        """Count an in-flight request so a deployment change can drain before stopping this
        server, instead of killing another client's request mid-generation. `claimed`: the
        request claimed this handle when it was routed, so convert that claim.

        Yields how many requests (this one included) were in flight on this server when it
        entered, so runs can record the concurrency they were served under."""
        with self._inflight_lock:
            if claimed and self._claims:
                self._claims.pop(0)
            self._inflight += 1
            concurrent_at_start = self._inflight
            self.last_used = time.monotonic()
        try:
            yield concurrent_at_start
        finally:
            with self._inflight_lock:
                self._inflight = max(0, self._inflight - 1)
                self.last_used = time.monotonic()

    @property
    def inflight(self) -> int:
        with self._inflight_lock:
            return self._inflight

    @property
    def busy(self) -> bool:
        """In flight, or routed and about to be (an unexpired claim)."""
        with self._inflight_lock:
            cutoff = time.monotonic() - self.CLAIM_TTL_SECONDS
            self._claims = [stamp for stamp in self._claims if stamp > cutoff]
            return self._inflight > 0 or bool(self._claims)

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while self.busy and time.monotonic() < deadline:
            time.sleep(0.05)
        return not self.busy

    def drain(self, timeout: float, abort: threading.Event | None = None) -> bool:
        """Wait up to `timeout` seconds for in-flight requests to finish. Returns True if the
        server went idle, False if it timed out or `abort` was set ("Stop now"); the caller
        stops it anyway to avoid hanging."""
        deadline = time.monotonic() + max(0.0, timeout)
        while self.inflight > 0 and time.monotonic() < deadline and not (abort is not None and abort.is_set()):
            time.sleep(0.05)
        return self.inflight == 0

    def reasoning_capabilities(self, config: AppConfig) -> dict[str, Any]:
        # The model file is fixed for the life of a deployment, so read its native
        # effort levels once per handle. Reloading the model makes a fresh handle.
        # Re-resolving/stat-ing the GGUF on every agent tool turn is pure latency.
        if self._reasoning_capabilities is None:
            resolved = getattr(self.process, "model_path", None)
            path = Path(resolved) if resolved is not None else resolve_model_path(config, self.model)
            self._reasoning_capabilities = model_reasoning(path)
        return self._reasoning_capabilities

    def runtime(self) -> dict[str, Any] | None:
        """The llama-server build serving this handle (probed when the server started)."""
        probe = getattr(self.process, "runtime", None)
        return probe().public() if callable(probe) else None

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
            "reasoning_mode": self.model.reasoning_mode,
            "reasoning_effort": self.model.reasoning_effort,
            "draft_model": self.model.draft_model or None,
            "spec_type": self.model.spec_type or None,
            "runtime": self.runtime(),
            "inflight": self.inflight,  # requests being answered right now (Unload waits for them)
            "resident_gb": self.resident_gb,
            "evictions": self.evictions,
            "rewarmed_at": self.rewarmed_at,
        }


class DeploymentManager:
    VALID_MODES = {"single_large_model", "parallel_models", "single_gpu"}
    # Called with the lane keys a deployment would occupy; raises AppError(409) when a lane is
    # leased (e.g. by a training run). Installed by ApplicationState.
    lease_guard: Any = None

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
        self._guard_stop = threading.Event()
        self._abort_drain = threading.Event()  # set by "Stop now": ends the wait for running requests
        # Always running: the loop reads policy.residency_guard each tick, so the UI can switch it live.
        threading.Thread(target=self._residency_loop, name="residency-guard", daemon=True).start()
        threading.Thread(target=self._supervise_loop, name="crash-supervisor", daemon=True).start()

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
            "coding_reasoning": model_reasoning(path),
            "tensor_split": model.tensor_split or None,
        }

    def catalog(self) -> dict[str, Any]:
        lanes = [
            {
                "key": lane.key,
                "display": lane.display,
                "match": lane.match,
                "vram_gb": lane.vram_gb,
                "auto_use": lane.auto_use,
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

    def busy_lane_keys(self) -> set[str]:
        with self.lock:
            return {lane.key for handle in self.handles.values() for lane in handle.lanes}

    def _check_leases(self, specs: list[dict[str, Any]]) -> None:
        if self.lease_guard is not None:
            self.lease_guard({lane.key for spec in specs for lane in spec["lanes"]})

    def _supervise_loop(self) -> None:
        while not self._guard_stop.wait(5.0):
            try:
                self.supervise_once()
            except Exception as exc:  # noqa: BLE001 - the supervisor must never take the service down
                print(f"[supervisor] error: {exc}", flush=True)

    def _emit(self, kind: str, severity: str, payload: dict[str, Any], lane: str | None = None) -> None:
        sink = self.__dict__.get("on_event")
        print(f"[supervisor] {kind}: {payload}", flush=True)
        if sink is not None:
            sink(kind, severity, payload, lane)

    def supervise_once(self, now: float | None = None) -> list[str]:
        """Find servers that died while still registered (a deliberate stop removes the handle first, so a dead
        handle is a crash). Each crash is recorded once; with `policy.auto_restart` the same server is restarted on
        the same lane after a back-off, until it crash-loops. Returns the notable things done, for tests."""
        now = time.time() if now is None else now
        tracker = self.__dict__.setdefault("_crash_tracker", CrashTracker(
            getattr(self.config.policy, "auto_restart_max", 3), getattr(self.config.policy, "auto_restart_window_s", 600)))
        tracker.clock = lambda: now
        seen = self.__dict__.setdefault("_crashes_seen", set())
        pending = self.__dict__.setdefault("_restart_due", {})
        done: list[str] = []
        with self.lock:
            snapshot = list(self.handles.items())
        for target, handle in snapshot:
            proc = handle.process.proc
            if proc is None or proc.poll() is None:
                continue
            key = (target, proc.pid)
            if key in seen:
                continue
            seen.add(key)
            lanes = [lane.key for lane in handle.lanes]
            lane = lanes[0] if lanes else None
            self._emit("server_crash", "error", {"target": target, "model": handle.model.name, "lanes": lanes,
                                                 "exit_code": proc.returncode, "pid": proc.pid}, lane)
            if not getattr(self.config.policy, "auto_restart", False):
                continue
            verdict, delay = tracker.record_crash(target)
            if verdict == "give_up":
                pending.pop(target, None)
                self._emit("crash_loop", "error", {"target": target, "model": handle.model.name, "lanes": lanes,
                                                   "detail": f"{tracker.count(target)} crashes within "
                                                             f"{tracker.window_s:.0f}s; automatic restart stopped"}, lane)
                done.append(f"gave up on {target}")
            else:
                pending[target] = now + delay
                done.append(f"restart of {target} in {delay:.0f}s")
        for target, due in list(pending.items()):
            if due > now:
                continue
            handle = self.handles.get(target)
            if handle is None or (handle.process.proc is not None and handle.process.proc.poll() is None):
                pending.pop(target, None)
                continue
            if not self.transition_lock.acquire(blocking=False):
                continue  # a deploy or swap is running: try again on the next tick
            try:
                pending.pop(target, None)
                lanes = [lane.key for lane in handle.lanes]
                lane = lanes[0] if lanes else None
                try:
                    handle.process.restart()   # same process object: same lane, device and model, never another GPU
                    self._emit("server_restarted", "info", {"target": target, "model": handle.model.name, "lanes": lanes}, lane)
                    with self.lock:
                        if self.state == "degraded" and all(h.process.proc is not None and h.process.proc.poll() is None
                                                            for h in self.handles.values()):
                            self.state, self.error = "ready", None
                    done.append(f"restarted {target}")
                except Exception as exc:  # noqa: BLE001
                    self._emit("restart_failed", "error", {"target": target, "error": str(exc)[:300]}, lane)
                    verdict, delay = tracker.record_crash(target)
                    if verdict == "restart":
                        pending[target] = now + delay
                    else:
                        self._emit("crash_loop", "error", {"target": target, "detail": "restart keeps failing; stopped trying"}, lane)
                    done.append(f"restart of {target} failed")
            finally:
                self.transition_lock.release()
        return done

    def _mark_dead_lanes(self) -> None:
        """Caller holds self.lock. A deployment whose lane server has exited is not ready, whatever
        the last successful deploy reported; without this, status and chat routing keep trusting it."""
        if self.state != "ready":
            return
        dead = sorted(
            key
            for key, handle in self.handles.items()
            if handle.process.proc is None or handle.process.proc.poll() is not None
        )
        if dead:
            self.state = "degraded"
            self.error = f"llama-server exited on lane(s) {', '.join(dead)}; load the deployment again."

    def status(self) -> dict[str, Any]:
        with self.lock:
            self._mark_dead_lanes()
            return {
                "state": self.state,
                "mode": self.mode,
                "error": self.error,
                "profile": self.active_profile,
                "deployment_id": self.deployment_id,
                "servers": [handle.public() for handle in self.handles.values()],
                "metrics": self.sampler.latest(),
                "split_canary": dict(self.__dict__.get("split_canary_results", {})),
            }

    def deploy(self, profile: dict[str, Any]) -> dict[str, Any]:
        if not self.transition_lock.acquire(blocking=False):
            raise AppError("Another deployment change is already in progress.", HTTPStatus.CONFLICT)
        try:
            mode = str(profile.get("mode", ""))
            if mode not in self.VALID_MODES:
                raise AppError(f"Unsupported mode: {mode}")
            specs = self._validate_profile(mode, profile)
            self._check_leases(specs)
            if "_abort_drain" in self.__dict__:
                self._abort_drain.clear()
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
                    self._admit_vram(spec)
                    process = self._make_process(spec, run_dir)
                    process.start()
                    self._check_split(spec, process)
                    handle = ServerHandle(spec["target"], spec["model"], spec["lanes"], process, spec.get("workload_kind"))
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

    def unload(self, lane: str | None = None, orphans_only: bool = False, force: bool = False,
               model: str | None = None) -> dict[str, Any]:
        """Free GPU memory. No arguments: stop every model this service runs and any leftover
        llama-server holding a lane port. `lane`: only that lane. `orphans_only`: only the
        leftovers this service does not manage.

        Running requests get `drain_timeout_seconds` to finish first. `force` ends that wait: when an
        unload or load is already waiting it is told to stop waiting (and this returns at once);
        otherwise it stops the models immediately, cutting any request in progress.

        A lane-scoped unload reports `released`: whether a model was stopped. With `model` (the name
        or catalog id the caller loaded) it unloads only if that is still what serves the lane, so a
        client releasing "its" model cannot stop one another client has loaded there since."""
        abort = self.__dict__.get("_abort_drain")
        if not self.transition_lock.acquire(blocking=False):
            if force and abort is not None:
                abort.set()
                return self.status()
            raise AppError("An unload or model change is already in progress; it waits up to "
                           f"{self.config.policy.drain_timeout_seconds:.0f} s for running requests. "
                           "Use Stop now to end the wait.", HTTPStatus.CONFLICT)
        if abort is not None:
            abort.set() if force else abort.clear()
        try:
            if orphans_only:
                self.reap_orphans()
                return self.status()
            if lane is not None:
                target = self._lane(lane)
                if model is not None and not self._lane_serves(target, model):
                    return {**self.status(), "released": False}
                released = self._unload_lane(target)
                return {**self.status(), "released": released}
            with self.lock:
                self.state = "switching"
            self._stop_all()
            self._reap_orphans_safely()
            with self.lock:
                self.state = "idle"
                self.mode = "idle"
                self.error = None
                self.active_profile = None
                self.deployment_id = None
            return self.status()
        finally:
            if abort is not None:
                abort.clear()
            self.transition_lock.release()

    def _lane_serves(self, lane: LaneConfig, model: str) -> bool:
        """Is `model` (its resolved name, or the string the caller originally asked for) what serves `lane`?"""
        with self.lock:
            requested = self._requested_names_locked()
            return any(model in {handle.model.name, requested.get(handle.target)}
                       for handle in self.handles.values() if lane.key in {item.key for item in handle.lanes})

    def _unload_lane(self, lane: LaneConfig) -> bool:
        """Stop whatever serves `lane` (a split model also frees its other GPU) and keep the rest.
        Returns whether a model was stopped."""
        with self.lock:
            requested = self._requested_names_locked()
            victims = [handle for handle in self.handles.values() if lane.key in {item.key for item in handle.lanes}]
            for victim in victims:
                del self.handles[victim.target]
                requested.pop(victim.target, None)
        for victim in victims:
            self._stop_handle(victim)
        self._reap_orphans_safely([lane])
        with self.lock:
            self._refresh_profile_locked(requested)
            self.state = "ready" if self.handles else "idle"
            self.error = None
            if not self.handles:
                self.deployment_id = None
        return bool(victims)

    _orphan_cache: tuple[float, list[dict[str, Any]]] | None = None

    def lane_orphans(self, lanes: list[LaneConfig] | None = None) -> list[dict[str, Any]]:
        """llama-server processes listening on a lane port that this service does not manage, e.g.
        left behind when a previous instance was killed. They keep holding VRAM and RAM."""
        control = get_host().control
        with self.lock:
            managed = {handle.process.proc.pid for handle in self.handles.values() if handle.process.proc is not None}
        found: list[dict[str, Any]] = []
        for lane in self.config.lanes if lanes is None else lanes:
            pid = control.pid_listening_on(lane.port)
            if pid is not None and pid not in managed and control.is_llama_server(pid):
                found.append({"lane": lane.key, "port": lane.port, "pid": pid})
        return found

    def orphans_cached(self, ttl: float = 10.0) -> list[dict[str, Any]]:
        """lane_orphans for the status poll: a port scan every 2 s would cost a netstat per lane.

        Nothing is reported while a load, swap or unload is in progress: its handle has already left
        `handles` while the server is still draining requests or still starting, so a scan then would
        flag the service's own server as a leftover. Such a scan is not cached either."""
        if self.transition_lock.locked():
            return []
        cached = self._orphan_cache
        if cached is not None and time.monotonic() - cached[0] < ttl:
            return cached[1]
        try:
            found = self.lane_orphans()
        except Exception:  # noqa: BLE001 -- a failed scan must not break status
            found = []
        self._orphan_cache = (time.monotonic(), found)
        return found

    def reap_orphans(self, lanes: list[LaneConfig] | None = None, only_ours: bool = False) -> list[dict[str, Any]]:
        """Stop leftover servers. `only_ours` (used by every automatic path) stops only servers this service
        recorded starting; a llama-server someone else started is left alone. The explicit "Stop orphan"
        action calls this without it."""
        control = get_host().control
        reaped = self.lane_orphans(lanes)
        registry = LlamaServerProcess.registry
        if only_ours and registry is not None:
            for orphan in reaped:
                if not registry.owns(orphan["pid"]):
                    print(f"[manager] leaving llama-server pid {orphan['pid']} on port {orphan['port']}: "
                          "this service did not start it", flush=True)
            reaped = [orphan for orphan in reaped if registry.owns(orphan["pid"])]
        for orphan in reaped:
            print(f"[manager] stopping orphaned llama-server pid {orphan['pid']} on port {orphan['port']}", flush=True)
            control.terminate_tree(orphan["pid"])
        self._orphan_cache = None
        return reaped

    def _reap_orphans_safely(self, lanes: list[LaneConfig] | None = None) -> None:
        try:
            self.reap_orphans(lanes, only_ours=True)
        except Exception as exc:  # noqa: BLE001 -- freeing leftovers is best effort
            print(f"[manager] could not stop orphaned servers: {exc}", flush=True)

    def shutdown(self) -> None:
        self._guard_stop.set()
        self.sampler.stop()
        self._stop_all()

    def _residency_loop(self) -> None:
        while not self._guard_stop.wait(max(5.0, self.config.policy.residency_check_seconds)):
            if not self.config.policy.residency_guard:
                continue
            try:
                self.check_residency()
            except Exception as exc:  # noqa: BLE001 -- the guard must never take the service down
                print(f"[residency] check failed: {exc}", flush=True)

    def check_residency(self) -> list[str]:
        """Keep loaded models in VRAM. Windows can move an idle server's VRAM into system RAM
        (measured: a 16.85 GB model at 0 GB resident; its next request took 10 s to page back in,
        and RAM held the 16.85 GB meanwhile). Each idle server whose weights are no longer resident
        gets a 1-token request on its reserved warm-up slot, which pages them back in now instead
        of during the next real request. Returns the targets re-warmed."""
        with self.lock:
            candidates = [
                handle for handle in self.handles.values()
                if not handle.busy and handle.warm_slot() is not None
                and handle.process.proc is not None and handle.process.proc.poll() is None
            ]
        if not candidates:
            return []
        gpu = get_host().gpu
        adapters = gpu.adapter_memory()
        luid_by_lane = {}
        for lane in self.config.lanes:
            luid_by_lane[lane.key] = next(
                (adapter["luid"] for adapter in adapters if lane.match.lower() in adapter["name"].lower()), None)
        usage = gpu.process_adapter_memory([handle.process.pid for handle in candidates])
        warmed: list[str] = []
        for handle in candidates:
            per_adapter = usage.get(handle.process.pid)
            luids = [luid_by_lane.get(lane.key) for lane in handle.lanes]
            if per_adapter is None or None in luids:
                continue  # the OS does not report it (Linux, or an unmatched adapter)
            resident = sum(per_adapter.get(luid, 0.0) for luid in luids)
            handle.resident_gb = round(resident, 2)
            expected = model_size_gb(self.config, handle.model) * PLACEMENT_MIN_VRAM_FRACTION
            if resident >= expected or handle.busy:
                continue
            handle.evictions += 1
            print(f"[residency] {handle.model.name} on {handle.target} is {resident:.2f} GB resident "
                  f"(weights {expected / PLACEMENT_MIN_VRAM_FRACTION:.2f} GB): paging it back into VRAM", flush=True)
            try:
                request = urlrequest.Request(
                    f"{handle.process.base_url}/completion",
                    data=json.dumps({"prompt": "ok", "n_predict": 1, "cache_prompt": False,
                                     "id_slot": handle.warm_slot()}).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
                    method="POST",
                )
                with handle.serving(), urlrequest.urlopen(request, timeout=120) as response:
                    response.read()
                handle.rewarmed_at = utc_now()
                warmed.append(handle.target)
            except (urlerror.URLError, TimeoutError, OSError) as exc:
                print(f"[residency] warm-up of {handle.target} failed: {exc}", flush=True)
        return warmed

    def _stop_handle(self, handle: ServerHandle) -> None:
        # Let in-flight requests finish (bounded) before killing the server, so a deployment
        # change never drops another client's request mid-generation. "Stop now" cuts the wait.
        abort = self.__dict__.get("_abort_drain")
        if not handle.drain(self.config.policy.drain_timeout_seconds, abort):
            print(f"[{handle.target}] {handle.inflight} request(s) still in flight; stopping anyway.", flush=True)
        handle.process.stop()

    def _stop_all(self) -> None:
        with self.lock:
            handles = list(self.handles.values())
            self.handles = {}
            deployment_id = self.deployment_id
        for handle in handles:
            try:
                self._stop_handle(handle)
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
        # A draft model shares the lane's VRAM, so include it in the footprint the
        # auto-sizer leaves headroom against (explicit context_window still wins).
        footprint_gb = model_size_gb(self.config, source) + self._draft_footprint_gb(source)
        context = int(raw.get("context_window") or source.ctx_size or ctx_for(self.config, footprint_gb, lane_vram_gb))
        if context < 256:
            raise AppError("Context window must be at least 256 tokens.")
        input_tokens = raw.get("input_tokens")
        max_output_tokens = raw.get("max_output_tokens")
        if input_tokens is not None or max_output_tokens is not None:
            if input_tokens is None or max_output_tokens is None:
                raise AppError("input_tokens and max_output_tokens must be provided together.")
            try:
                context = resolve_context_window(self.config, context, int(input_tokens), int(max_output_tokens))
            except ValueError as exc:
                raise AppError(str(exc)) from exc
        budget = int(raw.get("reasoning_budget", source.reasoning_budget if source.reasoning_budget is not None else self.config.policy.reasoning_budget))
        if budget < -1:
            raise AppError("Reasoning budget must be -1 (unlimited) or non-negative.")
        # The UI exposes a budget selector: choosing a positive budget must also
        # enable thinking, even when the base TOML policy defaults to off.
        reasoning_mode = raw.get("reasoning_mode")
        if reasoning_mode is None:
            reasoning_mode = (
                ("auto" if budget == -1 else "on" if budget > 0 else "off")
                if "reasoning_budget" in raw
                else source.reasoning_mode or self.config.policy.reasoning_mode
            )
        if reasoning_mode not in {"on", "off", "auto"}:
            raise AppError("reasoning_mode must be on, off, or auto.")
        effort = str(raw.get("reasoning_effort", source.reasoning_effort))
        if raw.get("workload_kind") == "coding_agent":
            budget, reasoning_mode = -1, "auto"
            capabilities = model_reasoning(resolve_model_path(self.config, source))
            if effort not in capabilities["efforts"]:
                raise AppError(f"Unsupported reasoning effort '{effort}'. Available: {', '.join(capabilities['efforts'])}.")
        tensor_split = str(raw.get("tensor_split") or source.tensor_split)
        model = replace(source, ctx_size=context, reasoning_budget=budget,
                        reasoning_mode=reasoning_mode, reasoning_effort=effort, tensor_split=tensor_split)
        path = resolve_model_path(self.config, model)
        if not path.exists():
            raise AppError(f"Model file does not exist: {path}")
        if "draft-mtp" in {part.strip() for part in model.spec_type.split(",")}:
            try:
                embedded_mtp = has_embedded_mtp(path)
            except (OSError, RuntimeError, ValueError) as exc:
                raise AppError(f"Could not inspect model for embedded MTP tensors: {exc}") from exc
            if not embedded_mtp:
                raise AppError(
                    f"Model '{model.name}' enables draft-mtp but its GGUF contains no nextn tensors."
                )
        if model.draft_model:
            draft_path = resolve_model_path(self.config, replace(model, path=model.draft_model))
            if not draft_path.exists():
                raise AppError(f"Draft model file does not exist: {draft_path}")
        return model, context

    def _draft_footprint_gb(self, model: ModelConfig) -> float:
        if not model.draft_model:
            return 0.0
        # size_gb=0 forces an actual file measurement of the DRAFT; without it model_size_gb
        # would return the target model's configured size_gb (replace keeps that field).
        return model_size_gb(self.config, replace(model, path=model.draft_model, size_gb=0.0))

    def _requested_model_names_by_lane(self) -> dict[str, str]:
        """Map lane key -> the exact model string the caller last asked for on that lane.

        `profile.models[]` is the documented join key a client uses to match its own
        requested model strings back to resolved lanes, so a lane replacement must not
        downgrade the lanes it preserves to `ModelConfig.name` (the resolved short display
        name, which never equals a catalog id a caller requests with). active_profile holds
        the caller's own raw profile as deploy() stored it -- either a single `model` dict
        (single_gpu / single_large_model) or a `models` list (parallel_models).
        """
        profile = self.active_profile or {}
        entries = list(profile.get("models") or [])
        single = profile.get("model")
        if single is not None:
            entries.append(single)
        requested: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            lane_key = entry.get("lane")
            name = entry.get("model")
            if lane_key and name:
                requested[str(lane_key)] = str(name)
        return requested

    def _validate_profile(self, mode: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
        if profile.get("workload_kind") == "coding_agent":
            profile = {**profile,
                       "model": {**(profile.get("model") or {}), "workload_kind": "coding_agent"},
                       "models": [{**raw, "workload_kind": "coding_agent"} for raw in profile.get("models", [])]}
        if mode == "single_large_model":
            raw = profile.get("model") or {}
            model, context = self._configured_model(raw, sum(lane.vram_gb for lane in self.config.lanes))
            backends = {lane.backend for lane in self.config.lanes}
            if len(backends) != 1:
                raise AppError("Both lanes must use the same backend for multi-GPU mode.")
            self._validate_capacity(model, list(self.config.lanes))
            return [{"target": "primary", "model": model, "lanes": list(self.config.lanes), "context": context, "multi": True, "workload_kind": raw.get("workload_kind")}]

        if mode == "single_gpu":
            raw = profile.get("model") or {}
            lane = self._lane(str(raw.get("lane", "")))
            model, context = self._configured_model(raw, lane.vram_gb)
            self._validate_capacity(model, [lane])
            return [{"target": lane.key, "model": model, "lanes": [lane], "context": context, "multi": False, "workload_kind": raw.get("workload_kind")}]

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
            self._validate_capacity(model, [lane])
            specs.append({"target": lane.key, "model": model, "lanes": [lane], "context": context, "multi": False, "workload_kind": raw.get("workload_kind")})
        return specs

    def _validate_capacity(self, model: ModelConfig, lanes: list[LaneConfig]) -> None:
        """Check the model actually fits the lane(s) it's headed for: file size always,
        plus a best-effort KV-cache estimate for the requested context when the model's
        GGUF metadata is one we're confident reading (see estimate_kv_cache_gb) -- this is
        the check that was missing for context sizes bumped up by input_tokens/
        max_output_tokens or requested explicitly, and was never run at all for
        single_large_model deploys before this."""
        size = model_size_gb(self.config, model)
        # A draft model (speculative decoding) is loaded on the same lane(s), so its weights
        # and KV cache share this budget. Omitting them here -- while _configured_model already
        # reserves the draft for auto-sizing -- would approve a deploy that then OOMs.
        draft_gb = self._draft_footprint_gb(model)
        if len(lanes) == 1 and model.pin_lane and model.pin_lane != lanes[0].key:
            raise AppError(f"{model.name} is pinned to lane {model.pin_lane}, not {lanes[0].key}.")
        capacity = sum(lane_capacity_gb(self.config, lane) for lane in lanes)
        lane_label = "+".join(lane.key for lane in lanes)
        draft_note = f" + {draft_gb:.2f} GB draft model" if draft_gb else ""
        path = resolve_model_path(self.config, model)
        lane_args = [arg for lane in lanes for arg in getattr(lane, "extra_args", [])]
        # weights kept on the CPU (--cpu-moe / --n-cpu-moe / -ngl) do not take VRAM
        offloaded = offloaded_weights_gb(path, size, model.extra_args, lane_args, gpu_layers=model.gpu_layers)
        if offloaded:
            draft_note += f" - {offloaded:.2f} GB held in system RAM"
        weights = size - offloaded + draft_gb
        if weights > capacity:
            raise AppError(
                f"{model.name} ({size:.2f} GB{draft_note}) needs {weights:.2f} GB but {lane_label} "
                f"has a safe model-file budget of {capacity:.2f} GB. Choose another GPU/model or multi-GPU mode."
            )
        kv_gb, _note = kv_estimate(path, model.ctx_size, model.extra_args, lane_args)
        if kv_gb is None:
            return
        draft_kv_gb = 0.0
        if model.draft_model:
            draft_path = resolve_model_path(self.config, replace(model, path=model.draft_model))
            draft_kv_est, _ = estimate_kv_cache_gb(draft_path, model.ctx_size)
            draft_kv_gb = draft_kv_est or 0.0
        total = weights + kv_gb + draft_kv_gb
        if total > capacity:
            kv_note = f" + ~{kv_gb:.2f} GB KV cache at {model.ctx_size} context tokens"
            if draft_gb or draft_kv_gb:
                kv_note += f" + {draft_gb:.2f} GB draft + ~{draft_kv_gb:.2f} GB draft KV"
            raise AppError(
                f"{model.name} needs an estimated {total:.2f} GB ({size:.2f} GB model file{kv_note}) "
                f"but {lane_label} has a safe budget of {capacity:.2f} GB. Lower the context window, "
                "choose another GPU, or use multi-GPU mode."
            )

    def _check_split(self, spec: dict[str, Any], process: Any) -> None:
        """After a model is split across GPUs, prove its output is sane (see canary.py). Policy `split_canary`:
        "off" skips it, "warn" records a failure in `split_canary_results` and the log, "refuse" also stops the
        server and fails the load."""
        mode = str(getattr(self.config.policy, "split_canary", "warn"))
        if not spec.get("multi") or mode == "off":
            return
        result = run_split_canary(process.base_url, spec["model"].name)
        self.__dict__.setdefault("split_canary_results", {})[spec["target"]] = {**result, "model": spec["model"].name}
        if result["ok"]:
            return
        print(f"[canary] split model {spec['model'].name}: {result['detail']}", flush=True)
        if mode == "refuse":
            process.stop()
            raise AppError(f"{spec['model'].name} was refused: it is split across GPUs and failed the correctness canary "
                           f"({result['detail']}). Use a single GPU, a different llama.cpp build, or set "
                           "policy.split_canary = \"warn\" to accept it anyway.", HTTPStatus.CONFLICT)

    def automatic_lanes(self) -> list[LaneConfig]:
        """Lanes automatic features may use. A lane switched to `auto_use = false` is never chosen by auto-placement,
        on-demand loading or routing; explicit requests naming it still work."""
        lanes = [lane for lane in self.config.lanes if getattr(lane, "auto_use", True)]
        if not lanes:
            raise AppError("Every GPU lane is switched off for automatic use (lane `auto_use = false`); name a lane explicitly.",
                           HTTPStatus.CONFLICT)
        return lanes

    def set_lane_auto_use(self, lane_key: str, enabled: bool) -> dict[str, Any]:
        lane = self._lane(lane_key)
        lane.auto_use = bool(enabled)
        return {"lane": lane.key, "auto_use": lane.auto_use}

    def _auto_lane_for_model(
        self,
        model_name: str,
        context_window: int | None,
        input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> LaneConfig:
        """Pick the smallest lane that passes the complete fit check.

        This intentionally uses ``check_fit`` rather than model file size alone: at a 51k
        context, KV cache can make a model that appears to fit the 9070 require the 9700.
        """
        candidates = sorted(self.automatic_lanes(), key=lambda lane: lane.vram_gb)
        for lane in candidates:
            verdict = self.check_fit(
                model_name, lane.key, context_window, input_tokens, max_output_tokens
            )
            if verdict["fits"]:
                return lane
        # Let deploy() produce the authoritative error against the largest lane when
        # nothing fits, instead of returning the smaller lane and hiding that it was tried.
        return candidates[-1]

    def deploy_parallel_for_client(
        self,
        model_names: list[str],
        context_window: int | None,
        input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Deploy one model per lane without promoting small models to the large lane.

        Each model is tested against the smallest lane at the requested context. A model that
        fits there belongs there; only a model that does not fit there may use the larger lane.
        Consequently, two small models (or two large-only models) are not a valid parallel pair
        and must be run sequentially on their common lane.
        """
        if len(model_names) != len(self.config.lanes):
            raise AppError(
                f"deploy_parallel needs exactly {len(self.config.lanes)} model(s) "
                f"(one per lane), got {len(model_names)}."
            )
        small_lane, large_lane = sorted(self.config.lanes, key=lambda lane: lane.vram_gb)
        assignments: list[tuple[str, LaneConfig]] = []
        for name in model_names:
            small_fit = self.check_fit(
                name, small_lane.key, context_window, input_tokens, max_output_tokens
            )
            if small_fit["fits"]:
                assignments.append((name, small_lane))
                continue
            large_fit = self.check_fit(
                name, large_lane.key, context_window, input_tokens, max_output_tokens
            )
            if not large_fit["fits"]:
                raise AppError(
                    f"{name} fits neither {small_lane.key} nor {large_lane.key} at the requested "
                    f"context: {large_fit['reason'] or small_fit['reason']}"
                )
            assignments.append((name, large_lane))

        assigned_keys = [lane.key for _, lane in assignments]
        if len(set(assigned_keys)) != len(assignments):
            raise AppError(
                "Parallel placement would promote a model to a larger GPU merely to fill both "
                f"lanes ({assigned_keys}). Run these models sequentially on their assigned lane."
            )

        raw_models = []
        for name, lane in assignments:
            raw: dict[str, Any] = {"model": name, "lane": lane.key}
            if context_window is not None:
                raw["context_window"] = context_window
            if input_tokens is not None:
                raw["input_tokens"] = input_tokens
            if max_output_tokens is not None:
                raw["max_output_tokens"] = max_output_tokens
            raw_models.append(raw)
        return self.deploy({"mode": "parallel_models", "models": raw_models})

    def check_fit(
        self,
        model_name: str,
        lane_key: str,
        context_window: int | None = None,
        input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Read-only feasibility check: "would this model fit this lane at this context window,"
        without loading anything. Runs the exact same resolution/validation deploy() would
        (_configured_model + _validate_capacity) so the answer is authoritative, not a client-side
        guess -- but never calls deploy() itself. For a client planning a schedule ahead of time
        (e.g. partitioning a model list by which lane each one fits) instead of discovering fit by
        trial deploy, which would actually load the model just to find out."""
        try:
            lane = self._lane(lane_key)
            raw: dict[str, Any] = {"model": model_name}
            if context_window is not None:
                raw["context_window"] = context_window
            if input_tokens is not None:
                raw["input_tokens"] = input_tokens
            if max_output_tokens is not None:
                raw["max_output_tokens"] = max_output_tokens
            model, resolved_context = self._configured_model(raw, lane.vram_gb)
            self._validate_capacity(model, [lane])
            return {"fits": True, "reason": None, "resolved_context_window": resolved_context}
        except AppError as exc:
            return {"fits": False, "reason": str(exc), "resolved_context_window": None}

    def deploy_for_client(
        self,
        model_name: str,
        gpu: str | None,
        context_window: int | None,
        input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Client-facing deploy: resolves a lane automatically when gpu is omitted, then delegates
        to deploy() so the feasibility verdict (fit/pin/context checks) is identical to the UI's.
        When the client reports its real input/output token budget, the allocated context window
        is sized to fit it (see resolve_context_window) instead of trusting a guessed default."""
        lane = self._lane(gpu) if gpu else self._auto_lane_for_model(
            model_name, context_window, input_tokens, max_output_tokens
        )
        raw: dict[str, Any] = {"model": model_name, "lane": lane.key}
        if context_window is not None:
            raw["context_window"] = context_window
        if input_tokens is not None:
            raw["input_tokens"] = input_tokens
        if max_output_tokens is not None:
            raw["max_output_tokens"] = max_output_tokens
        return self.deploy({"mode": "single_gpu", "model": raw})

    def deploy_lane_for_client(
        self,
        model_name: str,
        lane_key: str,
        context_window: int | None,
        input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Replace one GPU lane without stopping models on the other lanes.

        The caller must only replace a lane after its own outstanding chat call has
        completed. Deployment changes are serialized, but chat traffic on preserved
        lanes continues while this lane stops and loads its next model.
        """
        lane = self._lane(lane_key)
        raw: dict[str, Any] = {"model": model_name, "lane": lane.key}
        if context_window is not None:
            raw["context_window"] = context_window
        if input_tokens is not None:
            raw["input_tokens"] = input_tokens
        if max_output_tokens is not None:
            raw["max_output_tokens"] = max_output_tokens
        profile = {"mode": "single_gpu", "model": raw}
        spec = self._validate_profile("single_gpu", profile)[0]
        self._check_leases([spec])

        # Blocking is intentional: two lane workers can finish together, and the
        # second should wait for the first model load instead of failing with 409.
        self.transition_lock.acquire()
        if "_abort_drain" in self.__dict__:
            self._abort_drain.clear()
        with self.lock:
            requested_by_lane = self._requested_model_names_by_lane()
        requested_by_lane[lane.key] = model_name
        deployment_id = str(uuid.uuid4())
        self.store.begin_deployment(deployment_id, "lane_replace", profile)
        try:
            with self.lock:
                old_handle = self.handles.pop(lane.key, None)
                if not self.handles:
                    self.state = "switching"
                self.error = None
            if old_handle is not None:
                self._stop_handle(old_handle)

            run_dir = self.data_dir / "deployments" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._admit_vram(spec)
                process = self._make_process(spec, run_dir)
                process.start()
                handle = ServerHandle(spec["target"], spec["model"], spec["lanes"], process, spec.get("workload_kind"))
            except BaseException as exc:
                message = f"{lane.key}: {exc}"
                with self.lock:
                    self.state = "ready" if self.handles else "error"
                    self.mode = "parallel_models" if len(self.handles) > 1 else (
                        "single_gpu" if self.handles else "idle"
                    )
                    self.error = message
                self.store.update_deployment(deployment_id, "failed", error=message)
                raise AppError(message, HTTPStatus.INTERNAL_SERVER_ERROR) from exc

            with self.lock:
                self.handles[lane.key] = handle
                self.deployment_id = deployment_id
                self.state = "ready"
                self.mode = "parallel_models" if len(self.handles) > 1 else "single_gpu"
                self.active_profile = {
                    "mode": self.mode,
                    "models": [
                        {
                            "model": requested_by_lane.get(target, active.model.name),
                            "lane": target,
                            "context_window": active.process.ctx_size,
                        }
                        for target, active in self.handles.items()
                    ],
                }
                self.error = None
                servers = [active.public() for active in self.handles.values()]
            self.store.update_deployment(deployment_id, "ready", servers=servers)
            return self.status()
        finally:
            self.transition_lock.release()

    def _vram_need_gb(self, spec: dict[str, Any]) -> dict[str, float]:
        """VRAM a spec takes per lane: weights (+ draft) + estimated KV cache + headroom. A
        multi-GPU spec is split in --tensor-split proportion (else by lane VRAM), in the same
        largest-lane-first device order LlamaServerProcess.for_multi_gpu launches with."""
        model = spec["model"]
        path = resolve_model_path(self.config, model)
        lane_args = [arg for lane in spec["lanes"] for arg in getattr(lane, "extra_args", [])]
        kv_gb, _note = kv_estimate(path, spec["context"], model.extra_args, lane_args)
        offloaded = offloaded_weights_gb(path, model_size_gb(self.config, model), model.extra_args, lane_args,
                                         gpu_layers=model.gpu_layers)
        total = (model_size_gb(self.config, model) - offloaded + self._draft_footprint_gb(model)
                 + (kv_gb or 0.0) + self.config.policy.vram_headroom_gb)
        lanes = sorted(spec["lanes"], key=lambda lane: lane.vram_gb, reverse=True) if spec["multi"] else spec["lanes"]
        try:
            shares = [float(part) for part in model.tensor_split.split(",")] if spec["multi"] and model.tensor_split else []
        except ValueError:
            shares = []
        if len(shares) != len(lanes) or sum(shares) <= 0:
            shares = [lane.vram_gb for lane in lanes]
        return {lane.key: total * share / sum(shares) for lane, share in zip(lanes, shares)}

    def _admit_vram(self, spec: dict[str, Any], wait_seconds: float = 20.0) -> None:
        """GPU-only placement. On Windows a GPU that runs out of VRAM silently backs the rest
        with system RAM, so a server is started only when every lane it uses reports enough
        free VRAM. Waits briefly first, because a just-stopped server's VRAM is released
        asynchronously. A lane whose free VRAM cannot be read is not blocked.

        A leftover llama-server on one of the lanes' ports (a previous instance that was killed) is
        stopped first, as `kill_busy_ports` would do when the new server starts: its VRAM is not
        usable by this load, and would otherwise make the free-VRAM check refuse it."""
        if self.config.policy.kill_busy_ports:
            self._reap_orphans_safely(spec["lanes"])
        if not self.config.policy.vram_admission:
            return
        need = self._vram_need_gb(spec)
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                free = lane_free_vram_gb(self.config, spec["lanes"])
            except (Exception, SystemExit):  # noqa: BLE001 -- an unreadable probe must not block loading
                return
            short = {key: (gb, free[key]) for key, gb in need.items() if key in free and free[key] < gb}
            if not short:
                return
            if time.monotonic() >= deadline:
                detail = ", ".join(f"{key} needs {gb:.1f} GB, {avail:.1f} GB free" for key, (gb, avail) in short.items())
                raise AppError(
                    f"Not enough free VRAM to keep {spec['model'].name} on the GPU ({detail}). Unload a model, "
                    "lower its context window, or close other GPU applications.",
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
            time.sleep(1.0)

    def knows_model(self, name: str) -> bool:
        try:
            self._model(name)
        except AppError:
            return False
        return True

    def _claim_loaded(self, source: ModelConfig, lane_key: str | None) -> tuple[str, ServerHandle] | None:
        with self.lock:
            for target, handle in self.handles.items():
                if handle.model.name != source.name or handle.model.path != source.path:
                    continue
                if lane_key and lane_key not in {lane.key for lane in handle.lanes}:
                    continue
                if handle.process.proc is None or handle.process.proc.poll() is not None:
                    continue  # exited; the planner will replace it
                handle.claim()
                return target, handle
        return None

    def _occupants(self, lanes: list[LaneConfig]) -> list[ServerHandle]:
        keys = {lane.key for lane in lanes}
        return [handle for handle in self.handles.values() if keys & {lane.key for lane in handle.lanes}]

    def _plan_auto_load(self, requested: str, lane_key: str | None) -> tuple[dict[str, Any], list[ServerHandle]]:
        """Choose where `requested` goes and which loaded servers must stop for it.

        Candidates are the lanes it fits at its default context (smallest first), or -- only
        when no single GPU can hold it -- every lane together. Preference: a free lane, then a
        lane whose occupants are idle, least recently used first. Leased lanes are skipped."""
        raw = {"model": requested, "workload_kind": "coding_agent"}
        lanes = [self._lane(lane_key)] if lane_key else sorted(self.automatic_lanes(), key=lambda lane: lane.vram_gb)
        options: list[dict[str, Any]] = []
        reasons: list[str] = []
        for lane in lanes:
            try:
                spec = self._validate_profile("single_gpu", {"mode": "single_gpu", "model": {**raw, "lane": lane.key}})[0]
                self._check_leases([spec])
                options.append(spec)
            except AppError as exc:
                reasons.append(f"{lane.key}: {exc}")
        if not options and lane_key is None and len(self.automatic_lanes()) > 1 and all(getattr(l, "auto_use", True) for l in self.config.lanes):
            try:
                spec = self._validate_profile("single_large_model", {"mode": "single_large_model", "model": raw})[0]
                self._check_leases([spec])
                options.append(spec)
            except AppError as exc:
                reasons.append(f"all GPUs: {exc}")
        if not options:
            raise AppError(f"{requested} cannot be placed on the GPUs. " + " ".join(reasons))
        shortfall = self._vram_shortfall(options)
        if shortfall is not None and all(shortfall[id(spec)] for spec in options):
            detail = "; ".join(shortfall[id(spec)] for spec in options)
            raise AppError(f"Not enough free VRAM to keep {requested} on the GPU even after unloading idle "
                           f"models ({detail}). Close other GPU applications or use a smaller model.",
                           HTTPStatus.SERVICE_UNAVAILABLE)
        with self.lock:
            def rank(spec: dict[str, Any]) -> tuple[bool, bool, bool, float]:
                occupants = self._occupants(spec["lanes"])
                return (bool(shortfall and shortfall[id(spec)]), bool(occupants),
                        any(handle.busy for handle in occupants),
                        max((handle.last_used for handle in occupants), default=0.0))

            best = min(options, key=rank)  # min() keeps the first (smallest-lane) option on ties
            return best, self._occupants(best["lanes"])

    def _vram_shortfall(self, options: list[dict[str, Any]]) -> dict[int, str] | None:
        """Per option: "" when it fits the VRAM free now plus what evicting our own servers on
        its lanes would release, else a description of the shortfall. None when VRAM admission
        is off or free VRAM cannot be read (the per-start check stays the final word)."""
        if not self.config.policy.vram_admission:
            return None
        try:
            free = lane_free_vram_gb(self.config, list(self.config.lanes))
        except (Exception, SystemExit):  # noqa: BLE001
            return None
        if not free:
            return None
        with self.lock:
            released: dict[str, float] = {}
            for handle in self.handles.values():
                held = {"target": handle.target, "model": handle.model, "lanes": handle.lanes,
                        "context": handle.process.ctx_size, "multi": len(handle.lanes) > 1}
                for key, gb in self._vram_need_gb(held).items():
                    released[key] = released.get(key, 0.0) + gb
        result: dict[int, str] = {}
        for spec in options:
            short = [f"{key} needs {gb:.1f} GB, {free[key] + released.get(key, 0.0):.1f} GB obtainable"
                     for key, gb in self._vram_need_gb(spec).items()
                     if key in free and free[key] + released.get(key, 0.0) < gb]
            result[id(spec)] = ", ".join(short)
        return result

    def _requested_names_locked(self) -> dict[str, str]:
        requested = self._requested_model_names_by_lane()
        profile = self.active_profile or {}
        if profile.get("mode") == "single_large_model" and isinstance(profile.get("model"), dict):
            requested["primary"] = str(profile["model"].get("model") or "")
        return requested

    def _refresh_profile_locked(self, requested: dict[str, str]) -> None:
        """Rebuild mode/active_profile from the handles actually loaded (caller holds self.lock)."""
        def kind(handle: ServerHandle) -> dict[str, str]:
            return {"workload_kind": handle.workload_kind} if handle.workload_kind else {}

        if "primary" in self.handles:
            handle = self.handles["primary"]
            self.mode = "single_large_model"
            self.active_profile = {"mode": self.mode, "model": {
                "model": requested.get("primary") or handle.model.name, "context_window": handle.process.ctx_size,
                **kind(handle)}}
        elif self.handles:
            self.mode = "parallel_models" if len(self.handles) > 1 else "single_gpu"
            self.active_profile = {"mode": self.mode, "models": [
                {"model": requested.get(target) or handle.model.name, "lane": target,
                 "context_window": handle.process.ctx_size, **kind(handle)}
                for target, handle in self.handles.items()]}
        else:
            self.mode = "idle"
            self.active_profile = None

    def ensure_model(self, requested: str, lane_key: str | None = None) -> tuple[str, ServerHandle]:
        """Serve `requested` from the GPUs for an OpenAI-gateway request, loading it on demand.

        Already loaded (on `lane_key`, when given): returned as is. Otherwise it is placed per
        _plan_auto_load, evicting the least recently used idle model(s) it displaces, and only
        started when its weights + KV cache fit the VRAM the GPU reports free (_admit_vram), so
        it never lands in system RAM. A model that is serving is never stopped mid-request: if
        it stays busy past the drain timeout the caller gets 503 and should retry. The returned
        handle is already claimed for the caller's request."""
        source = self._model(requested)
        found = self._claim_loaded(source, lane_key)
        if found:
            return found
        if not self.transition_lock.acquire(timeout=self.config.project.start_timeout_seconds):
            raise AppError("Another model is still loading; retry shortly.", HTTPStatus.SERVICE_UNAVAILABLE)
        if "_abort_drain" in self.__dict__:
            self._abort_drain.clear()  # a stale "Stop now" must not cut this swap's wait short
        try:
            found = self._claim_loaded(source, lane_key)
            if found:
                return found
            if self.config.policy.kill_busy_ports:
                # A leftover llama-server on a lane port holds VRAM no plan can use; stop it first
                # (kill_busy_ports would do so at start anyway) so the free-VRAM check sees the truth.
                self._reap_orphans_safely()
            spec, victims = self._plan_auto_load(requested, lane_key)
            for victim in victims:
                victim.wait_idle(self.config.policy.drain_timeout_seconds)
            with self.lock:
                busy = [victim for victim in victims if victim.busy]
                if busy:
                    raise AppError(
                        f"Cannot load {source.name}: {', '.join(f'{v.model.name} on {v.target}' for v in busy)} "
                        "is still serving a request; retry shortly.",
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                requested_names = self._requested_names_locked()
                for victim in victims:
                    if self.handles.get(victim.target) is victim:
                        del self.handles[victim.target]
                    requested_names.pop(victim.target, None)
                self._refresh_profile_locked(requested_names)
                self.state = "switching" if not self.handles else self.state
                self.error = None
            for victim in victims:
                print(f"[manager] evicting {victim.model.name} from {victim.target} to load {source.name}", flush=True)
                victim.process.stop()

            profile = {"mode": "single_large_model" if spec["multi"] else "single_gpu", "workload_kind": "coding_agent",
                       "model": {"model": requested, **({} if spec["multi"] else {"lane": spec["target"]})}}
            deployment_id = str(uuid.uuid4())
            self.store.begin_deployment(deployment_id, "auto_load", profile)
            run_dir = self.data_dir / "deployments" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._admit_vram(spec)
                process = self._make_process(spec, run_dir)
                process.start()
                self._check_split(spec, process)
            except BaseException as exc:
                message = f"{spec['target']}: {exc}"
                with self.lock:
                    self.state = "ready" if self.handles else "error"
                    self.error = message
                self.store.update_deployment(deployment_id, "failed", error=message)
                if isinstance(exc, AppError):
                    raise
                raise AppError(message, HTTPStatus.INTERNAL_SERVER_ERROR) from exc

            handle = ServerHandle(spec["target"], spec["model"], spec["lanes"], process, "coding_agent")
            handle.claim()
            with self.lock:
                self.handles[spec["target"]] = handle
                requested_names[spec["target"]] = requested
                self._refresh_profile_locked(requested_names)
                self.deployment_id = deployment_id
                self.state = "ready"
                self.error = None
                servers = [active.public() for active in self.handles.values()]
            self.store.update_deployment(deployment_id, "ready", servers=servers)
            print(f"[manager] loaded {source.name} on {spec['target']}", flush=True)
            return spec["target"], handle
        finally:
            self.transition_lock.release()

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
    def __init__(self, manager: DeploymentManager, store: RunStore,
                 history: "ChatHistoryStore | None" = None, usage: "UsageLedger | None" = None):
        self.manager = manager
        self.store = store
        self.history = history or ChatHistoryStore(Path(), enabled=False)
        self.usage = usage or UsageLedger(Path(), enabled=False)
        self._client_output_lock = threading.Lock()

    def _coding_request(self, handle: ServerHandle, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        policy = self.manager.config.policy
        try:
            self._validate_structured_output(payload)
            return coding_request(
                payload, capabilities=handle.reasoning_capabilities(self.manager.config),
                default_effort=handle.model.reasoning_effort,
                server_budget=handle.model.reasoning_budget if handle.model.reasoning_budget is not None else policy.reasoning_budget,
                server_mode=handle.model.reasoning_mode or policy.reasoning_mode,
            )
        except ValueError as exc:
            raise AppError(str(exc)) from exc

    @staticmethod
    def _validate_structured_output(payload: dict[str, Any]) -> None:
        """Fail fast on malformed structured-output constraints so a coding harness gets a
        clear error instead of a cryptic backend failure. These fields pass through to
        llama-server untouched: `grammar` (GBNF), `json_schema`, and `response_format`.
        Grammar/schema-constrained decoding is the most reliable way to stop a local model
        emitting malformed tool-call JSON, which is the dominant tool-use failure mode."""
        grammar = payload.get("grammar")
        if grammar is not None and (not isinstance(grammar, str) or not grammar.strip()):
            raise ValueError("`grammar` must be a non-empty GBNF string.")
        json_schema = payload.get("json_schema")
        if json_schema is not None and not isinstance(json_schema, dict):
            raise ValueError("`json_schema` must be an object.")
        response_format = payload.get("response_format")
        if response_format is not None:
            if not isinstance(response_format, dict):
                raise ValueError("`response_format` must be an object.")
            kind = response_format.get("type")
            if kind not in {"text", "json_object", "json_schema"}:
                raise ValueError("`response_format.type` must be text, json_object, or json_schema.")
            if kind == "json_schema" and not isinstance(response_format.get("json_schema"), dict):
                raise ValueError("`response_format` of type json_schema requires a `json_schema` object.")
        if grammar is not None and response_format is not None:
            raise ValueError("Set `grammar` or `response_format`, not both.")

    @staticmethod
    def _tool_call_stats(tool_calls: list[dict[str, Any]] | None) -> dict[str, Any]:
        """Validity of the model's emitted tool calls. Malformed = a call with no function
        name, or whose JSON `arguments` string does not parse. Empty arguments are valid (a
        no-argument call). This is exactly what grammar/schema constraints are meant to fix,
        so the valid rate is the signal for whether constraining output is helping."""
        total = 0
        malformed = 0
        for call in tool_calls or []:
            function = (call or {}).get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments")
            total += 1
            if not name:
                malformed += 1
                continue
            if isinstance(arguments, str) and arguments.strip():
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    malformed += 1
        if total == 0:
            return {"tool_calls_total": 0, "tool_calls_malformed": 0, "tool_call_valid_rate": None}
        return {"tool_calls_total": total, "tool_calls_malformed": malformed,
                "tool_call_valid_rate": round((total - malformed) / total, 3)}

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        client: dict[str, Any] | None = None
        client_id = payload.get("client_id")
        if client_id:
            client = self.store.get_client(str(client_id))
            if client is None:
                raise AppError(f"Unknown client_id: {client_id}", HTTPStatus.NOT_FOUND)
            self.store.touch_client(client["id"])
        project_id = str(payload.get("project_id") or "").strip() or None
        if client:
            project_id = client.get("project_id") or project_id
            if not project_id:
                project = self.store.upsert_project(
                    client["project_name"], git_remote=client.get("github_repo")
                )
                project_id = project["id"]
                self.store.link_client_project(client["id"], project_id)
        if project_id and self.store.get_project(project_id) is None:
            raise AppError(f"Unknown project_id: {project_id}", HTTPStatus.NOT_FOUND)
        if not project_id:
            # Watertight: an unattributed call still lands in the default project.
            project_id = self.store.upsert_project(self.manager.config.project.default_project)["id"]
        workload_kind = str(payload.get("workload_kind") or "evaluation").strip().lower()
        if workload_kind not in {"evaluation", "coding_agent"}:
            raise AppError("workload_kind must be 'evaluation' or 'coding_agent'.")
        agent_role = str(payload.get("agent_role") or "").strip().lower() or None
        agent_session_id = str(payload.get("agent_session_id") or "").strip() or None
        coding_task_id = str(payload.get("coding_task_id") or "").strip() or None
        task_label = payload.get("task_label")
        # Normalized (lowercase + stripped) so "Resume_Extraction" and "resume_extraction"
        # land in the same dashboard category instead of silently fragmenting into two.
        task_label = str(task_label).strip().lower() or None if task_label else None
        with self.manager.lock:
            self.manager._mark_dead_lanes()
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
        if workload_kind == "coding_agent":
            # Validate before starting workers, so invalid settings return a proper
            # HTTP error instead of an uncaught thread exception and empty results.
            for handle in handles.values():
                self._coding_request(handle, payload)
        results: dict[str, Any] = {}
        threads = []

        def run(target: str, handle: ServerHandle) -> None:
            results[target] = self._run_one(
                target,
                handle,
                messages,
                payload,
                comparison_id,
                client,
                task_label,
                project_id,
                agent_session_id,
                coding_task_id,
                agent_role,
                workload_kind,
            )

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
        client: dict[str, Any] | None = None,
        task_label: str | None = None,
        project_id: str | None = None,
        agent_session_id: str | None = None,
        coding_task_id: str | None = None,
        agent_role: str | None = None,
        workload_kind: str = "evaluation",
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        request_payload: dict[str, Any] = {
            "model": handle.model.name,
            "messages": messages,
            "stream": False,
            "temperature": float(payload.get("temperature", 0.7)),
            "top_p": float(payload.get("top_p", 0.95)),
        }
        if payload.get("max_tokens") is not None:
            request_payload["max_tokens"] = int(payload["max_tokens"])
        elif workload_kind != "coding_agent":
            request_payload["max_tokens"] = 1024
        reasoning_settings = None
        if workload_kind == "coding_agent":
            for key in ("reasoning_effort", "chat_template_kwargs", "max_completion_tokens"):
                if key in payload:
                    request_payload[key] = payload[key]
            request_payload, reasoning_settings = self._coding_request(handle, request_payload)
        if payload.get("seed") is not None:
            request_payload["seed"] = int(payload["seed"])
        model_path = getattr(handle.process, "model_path", None) or resolve_model_path(self.manager.config, handle.model)
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
            "runtime": handle.runtime(),
            "host_os": os_info(),
            "reasoning": reasoning_settings,
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
                "client_id": client["id"] if client else None,
                "task_label": task_label,
                "project_id": project_id,
                "agent_session_id": agent_session_id,
                "coding_task_id": coding_task_id,
                "agent_role": agent_role,
                "workload_kind": workload_kind,
                "host_os": os_info(),
            }
        )
        baseline = self.manager.sampler.latest() if workload_kind == "coding_agent" else self.manager.sampler.capture()
        started = time.monotonic()
        try:
            with handle.serving() as concurrent_at_start:  # drained before a deployment change stops this server
                backend = self._post_json(
                    f"{handle.process.base_url}/v1/chat/completions",
                    handle.pin_serving_slot(dict(request_payload)),
                    timeout=float(payload.get("timeout_seconds", 900)),
                )
            elapsed = time.monotonic() - started
            choice = (backend.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            output = message.get("content") or choice.get("text") or ""
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            usage = self._normalize_usage(backend)
            usage.update(self._tool_call_stats(message.get("tool_calls")))
            timing = self._normalize_timing(backend, usage, elapsed)
            timing["concurrent_requests_at_start"] = concurrent_at_start
            if workload_kind != "coding_agent":
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
                "reasoning": reasoning_settings,
            }
            self.store.finish_run(run_id, final)
            self.store.save_samples(run_id, samples, started)
            if self.history.enabled and payload.get("store_history") is not False:
                self.history.enqueue({
                    "id": run_id, "created_at": utc_now(),
                    "project_id": project_id, "client_id": client["id"] if client else None,
                    "agent_role": agent_role, "agent_session_id": agent_session_id,
                    "coding_task_id": coding_task_id, "model_name": handle.model.name,
                    "target": target, "workload_kind": workload_kind, "usage": usage,
                    "timing": timing, "finish_reason": choice.get("finish_reason"),
                    "status": "completed", "request": request_payload, "resources": resources,
                    "messages": messages, "tools": payload.get("tools"), "output_text": output,
                    "reasoning_text": reasoning, "tool_calls": message.get("tool_calls"),
                })
            self.usage.record(project_id, str(payload.get("connection_type") or workload_kind),
                              usage.get("input_tokens"), usage.get("output_tokens"))
            delivery = self._deliver_to_client(client, run_id, target, handle, final)
            result = {"run_id": run_id, "model": handle.model.name, **final, "backend_response": None}
            if delivery is not None:
                result["client_delivery"] = delivery
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - started
            if workload_kind != "coding_agent":
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
            if self.history.enabled and payload.get("store_history") is not False:
                self.history.enqueue({
                    "id": run_id, "created_at": utc_now(),
                    "project_id": project_id, "client_id": client["id"] if client else None,
                    "agent_role": agent_role, "agent_session_id": agent_session_id,
                    "coding_task_id": coding_task_id, "model_name": handle.model.name,
                    "target": target, "workload_kind": workload_kind, "status": "failed",
                    "request": request_payload, "messages": messages, "tools": payload.get("tools"),
                    "usage": {}, "timing": final["timing"], "resources": resources,
                    "error_text": str(exc),
                })
            delivery = self._deliver_to_client(client, run_id, target, handle, final)
            result = {"run_id": run_id, "model": handle.model.name, **final}
            if delivery is not None:
                result["client_delivery"] = delivery
            return result

    def _deliver_to_client(
        self,
        client: dict[str, Any] | None,
        run_id: str,
        target: str,
        handle: ServerHandle,
        final: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Append this run's metrics as a JSON line to the client's own output_path, so a
        consuming codebase gets its own durable copy of the performance data alongside the
        response. Never fails the chat call itself if the client's path can't be written."""
        if client is None:
            return None
        record = {
            "run_id": run_id,
            "client_id": client["id"],
            "project_name": client["project_name"],
            "github_repo": client.get("github_repo"),
            "recorded_at": utc_now(),
            "target": target,
            "model": handle.model.name,
            "device": handle.process.device,
            "context_window": handle.process.ctx_size,
            "status": final.get("status"),
            "usage": final.get("usage"),
            "timing": final.get("timing"),
            "resources": final.get("resources"),
            "reasoning": final.get("reasoning"),
            "error_text": final.get("error_text"),
            "host_os": os_info(),
        }
        output_path = Path(client["output_path"]).expanduser()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with self._client_output_lock, output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            return {"written": False, "path": str(output_path), "error": str(exc)}
        return {"written": True, "path": str(output_path)}

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
        timings = payload.get("timings") or payload.get("timing") or {}
        details = usage.get("completion_tokens_details") or {}
        def first(*values: Any) -> Any:
            return next((value for value in values if value is not None), None)

        prefill_tokens = first(timings.get("prompt_n"), timings.get("prompt_tokens"))
        input_tokens = usage.get("prompt_tokens")
        if input_tokens is None and prefill_tokens is not None:
            input_tokens = prefill_tokens + (timings.get("cache_n") or 0)
        output_tokens = first(usage.get("completion_tokens"), timings.get("predicted_n"), timings.get("generated_tokens"))
        thinking_tokens = details.get("reasoning_tokens")
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = first(prompt_details.get("cached_tokens"), timings.get("cache_n"))
        if prefill_tokens is None and input_tokens is not None and cached_tokens is not None:
            prefill_tokens = max(0, input_tokens - cached_tokens)
        # Reasoning characters: supplied by the streaming gateway (deltas are not kept
        # on the payload), else summed from a non-streamed message's reasoning text.
        reasoning_characters = payload.get("reasoning_characters")
        if reasoning_characters is None:
            reasoning_characters = sum(
                len((choice.get("message") or {}).get("reasoning_content") or
                    (choice.get("message") or {}).get("reasoning") or "")
                for choice in payload.get("choices", [])
            )
        visible_tokens = output_tokens
        reasoning_observed = bool(payload.get("reasoning_observed") or reasoning_characters)
        # Native effort levels carry no fixed budget, so track what each run actually
        # spends on thinking. Prefer the backend's exact reasoning-token count; when it
        # is omitted, fall back to a labeled ~4-chars/token estimate (never mixed into
        # the exact-count aggregates, never passed off as a backend figure).
        thinking_tokens_estimated = None
        if thinking_tokens is None and reasoning_characters > 0:
            thinking_tokens_estimated = max(1, round(reasoning_characters / 4))
        if thinking_tokens is None and reasoning_observed:
            visible_tokens = None
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
            "thinking_tokens_estimated": thinking_tokens_estimated,
            "thinking_characters": reasoning_characters or None,
            "thinking_tokens_source": (
                "backend" if thinking_tokens is not None
                else "estimated_from_characters" if thinking_tokens_estimated is not None
                else "unavailable"
            ),
            "visible_output_tokens": visible_tokens,
            "output_tokens": output_tokens,
            "total_tokens": first(usage.get("total_tokens"), (
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            )),
        }

    @staticmethod
    def _normalize_timing(payload: dict[str, Any], usage: dict[str, Any], elapsed: float) -> dict[str, Any]:
        timings = payload.get("timings") or payload.get("timing") or {}
        def first(*values: Any) -> Any:
            return next((value for value in values if value is not None), None)

        prompt_ms = first(timings.get("prompt_ms"), timings.get("prompt_eval_time_ms"))
        predicted_ms = first(timings.get("predicted_ms"), timings.get("generation_time_ms"))
        prefill_rate = first(timings.get("prompt_per_second"), timings.get("prompt_tokens_per_second"))
        token_rate = first(timings.get("predicted_per_second"), timings.get("tokens_per_second"))
        if token_rate is None and predicted_ms is not None and predicted_ms > 0 and usage.get("output_tokens") is not None:
            token_rate = usage["output_tokens"] / (predicted_ms / 1000)
        end_to_end_rate = usage["output_tokens"] / elapsed if usage.get("output_tokens") is not None and elapsed > 0 else None
        # Speculative decoding stats (present only when a draft model is loaded). Key
        # names vary across llama.cpp builds, so probe timings and the top-level payload.
        draft_tokens = first(timings.get("draft_n"), timings.get("n_draft"), timings.get("draft_tokens"),
                             payload.get("draft_n"))
        draft_accepted = first(timings.get("draft_n_accepted"), timings.get("n_draft_accepted"),
                               timings.get("draft_accepted"), payload.get("draft_n_accepted"))
        acceptance = (round(draft_accepted / draft_tokens, 3)
                      if draft_tokens and draft_accepted is not None and draft_tokens > 0 else None)
        return {
            "time_to_first_token_ms": timings.get("time_to_first_token_ms"),
            "prefill_duration_ms": prompt_ms,
            "decode_duration_ms": predicted_ms,
            "end_to_end_duration_ms": round(elapsed * 1000, 2),
            "prefill_tokens_per_second": round(float(prefill_rate), 3) if prefill_rate is not None else None,
            "tokens_per_second": round(float(token_rate), 3) if token_rate is not None else None,
            "end_to_end_tokens_per_second": round(end_to_end_rate, 3) if end_to_end_rate is not None else None,
            "backend_prompt_tokens": first(timings.get("prompt_n"), timings.get("prompt_tokens")),
            "backend_output_tokens": first(timings.get("predicted_n"), timings.get("generated_tokens")),
            "draft_tokens": draft_tokens,
            "draft_accepted_tokens": draft_accepted,
            "draft_acceptance_rate": acceptance,
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
        private = values(process_rows, "private_bytes")
        dedicated = values(process_rows, "dedicated_vram_bytes")
        shared = values(process_rows, "shared_gpu_memory_bytes")
        page_faults = values(process_rows, "page_fault_count")
        system_cpu = values(system_rows, "cpu_pct")
        ram = values(system_rows, "ram_used_bytes")
        pagefile = values(system_rows, "pagefile_used_bytes")
        spill = bool(shared and dedicated and max(shared) > 1024**3 and max(shared) > max(dedicated) * 0.15)
        return {
            "sample_count": len(process_rows),
            "avg_process_cpu_pct": average(cpu),
            "peak_process_cpu_pct": max(cpu) if cpu else None,
            "avg_gpu_utilization_pct": average(gpu_utilization),
            "peak_gpu_utilization_pct": max(gpu_utilization) if gpu_utilization else None,
            "peak_process_rss_bytes": int(max(rss)) if rss else None,
            "peak_process_private_bytes": int(max(private)) if private else None,
            # page_fault_count is a cumulative counter since process start; the delta across
            # this run's window is what's actually informative (memory pressure during THIS run).
            "process_page_fault_delta": int(page_faults[-1] - page_faults[0]) if len(page_faults) >= 2 else None,
            "peak_dedicated_vram_bytes": int(max(dedicated)) if dedicated else None,
            "peak_shared_gpu_memory_bytes": int(max(shared)) if shared else None,
            "spill_suspected": spill if shared and dedicated else None,
            "avg_system_cpu_pct": average(system_cpu),
            "peak_system_ram_used_bytes": int(max(ram)) if ram else None,
            "peak_system_pagefile_used_bytes": int(max(pagefile)) if pagefile else None,
            "samples_started_at": samples[0].get("sampled_at") if samples else None,
            "samples_finished_at": samples[-1].get("sampled_at") if samples else None,
        }


class ChatbotHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # One thread per connection was unbounded: a burst of polls or a runaway client could exhaust threads. Above
    # this many in-flight connections new ones get an immediate 503 + Retry-After instead of queueing forever.
    max_connections = 128

    def __init__(
        self,
        address: tuple[str, int],
        app_state: "ApplicationState",
        cors_origins: list[str] | tuple[str, ...] | None = None,
        max_connections: int | None = None,
    ):
        if max_connections:
            self.max_connections = max_connections
        self._slots = threading.BoundedSemaphore(self.max_connections)
        self.rejected_connections = 0
        super().__init__(address, RequestHandler)
        self.app_state = app_state
        self.cors_origins = frozenset(
            origin.strip() for origin in (cors_origins or []) if origin and origin.strip()
        )
        if hasattr(app_state, "network"):
            app_state.network = network_metadata(self.server_address[0], self.server_address[1])

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            self.rejected_connections += 1
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Length: 0\r\n"
                                b"Connection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class ApplicationState:
    def __init__(self, config_path: Path, data_dir: Path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.network: dict[str, Any] = {
            "bind_host": "127.0.0.1",
            "port": 8090,
            "lan_enabled": False,
            "base_url": "http://127.0.0.1:8090",
            "openai_base_url": "http://127.0.0.1:8090/v1",
            "advertised_urls": [],
            "advertised_openai_base_urls": [],
            "model_server_ports": "loopback_only",
        }
        self.control: ServiceControl | None = None  # set by main(); the UI's restart/stop need it
        self.data_dir = data_dir
        self.host_os = os_info().get("system", sys.platform)
        self._ttl: dict[str, tuple[float, Any]] = {}
        self._ttl_lock = threading.Lock()
        LlamaServerProcess.registry = ProcessRegistry(data_dir / "managed_servers.json")
        self.backups: list[dict[str, Any]] = []  # filled by main() before the databases are opened
        self.store = RunStore(data_dir / "chatbot.sqlite3")
        self.interrupted_runs = self.store.mark_interrupted()
        history_path = (
            Path(self.config.project.chat_history_path)
            if self.config.project.chat_history_path
            else data_dir / "chat_history.sqlite3"
        )
        self.history = ChatHistoryStore(history_path, enabled=self.config.project.store_chat_history)
        self.usage = UsageLedger(data_dir / "project_usage.sqlite3")
        self.manager = DeploymentManager(self.config, data_dir, self.store)
        self.chat = ChatService(self.manager, self.store, self.history, self.usage)
        training_db = data_dir / "training.sqlite3"
        self.training = TrainingStore(training_db)
        self.leases = LeaseStore(training_db)
        self.manager.lease_guard = self._lease_guard
        windows_reader, _why = WindowsGpuReader.try_create() if sys.platform == "win32" else (None, None)
        self.hardware_log = HardwareLog(
            self.training, self.host_os, self.hardware_lane_context, host_sensors=make_host_sensors(data_dir),
            system_latest=lambda: self.hardware.system_latest,
        )
        self.manager.on_event = lambda kind, severity, payload, lane=None: self.hardware_log.record_event(kind, severity, payload, lane)
        self.hardware = HardwareSampler(
            self.config.lanes, on_samples=self.hardware_log.on_gpu_samples,
            smi=make_amd_smi_reader() if sys.platform.startswith("linux") else None, windows_reader=windows_reader,
        )
        self.archive = HardwareArchive(data_dir / "archive", self.training.connect, self.host_os)
        self._archive_stop = threading.Event()
        self.exchange = None  # set by main() when --exchange-dir is given
        self.kernel_faults = KernelFaultWatcher(self._record_kernel_fault)
        self.eval_queue = EvalQueue(training_db, capacity=self.config.evaluation.queue_capacity)
        self.benchmarks = BenchmarkRunner(
            training_db, self.training, "http://127.0.0.1:8090",
            on_trial_end=lambda run_id: self.service.cleanup_trial(run_id),
        )
        self.service = TrainingService(
            self.config, self.training, self.leases, self.eval_queue, self.benchmarks, self.hardware,
            data_dir / "artifacts", self.manager.busy_lane_keys, self.manager.unload,
        )
        self.eval_worker = self._build_eval_worker() if self.config.evaluation.enabled else None

    def _eval_server_factory(self, base: Path, adapter: Path) -> LlamaServerProcess:
        settings = self.config.evaluation
        model = ModelConfig(name=f"eval-{base.stem}", path=str(base), ctx_size=settings.ctx_size, parallel=1,
                            extra_args=["--lora", str(adapter)])
        return LlamaServerProcess.for_single_lane(
            self.config, self.manager._lane(settings.lane), model, base, settings.ctx_size,
            self.data_dir / "eval_servers",
        )

    def _build_eval_worker(self) -> EvalWorker:
        settings = self.config.evaluation
        holder = "evaluation:worker"
        pipeline = EvaluationPipeline(settings, self.training, AdapterServerManager(self._eval_server_factory),
                                      workdir=self.data_dir / "eval_work")
        return EvalWorker(
            self.eval_queue, pipeline,
            lane_ready=lambda: settings.lane not in self.manager.busy_lane_keys()
            and not self.leases.conflicts([settings.lane], ignore_holder=holder),
            acquire_lane=lambda: self.leases.acquire(holder, [settings.lane], kind="evaluation",
                                                     ttl_s=float(settings.timeout_s) * 3),
            release_lane=lambda _lease: self.leases.release_holder(holder),
            on_done=lambda job, _outcomes: select_best(self.training, job["run_id"], settings, apply=True),
        )

    def _lease_guard(self, lanes: set[str]) -> None:
        conflicts = self.leases.conflicts(lanes)
        if conflicts:
            detail = "; ".join(f"{c['lane']} is leased by {c['holder']}" for c in conflicts)
            raise AppError(f"GPU lane unavailable: {detail}. Finish or release that job first.", HTTPStatus.CONFLICT)

    def inference_leases(self) -> list[dict[str, Any]]:
        """Loaded inference servers shown as pseudo-leases (derived, not stored)."""
        with self.manager.lock:
            return [{"id": f"inference:{target}", "holder": f"inference:{target}", "kind": "inference",
                     "state": "active", "lanes": [lane.key for lane in handle.lanes],
                     "model": handle.model.name, "expires_at": None}
                    for target, handle in self.manager.handles.items()]

    def lane_holders(self) -> dict[str, dict[str, Any]]:
        holders = self.leases.lane_holders()
        for lease in self.inference_leases():
            for lane in lease["lanes"]:
                holders.setdefault(lane, {"holder": lease["holder"], "kind": "inference",
                                          "lease_id": lease["id"], "expires_at": None})
        return holders

    def cached(self, key: str, ttl: float, produce: Any) -> Any:
        """Short-lived answer cache for endpoints the dashboard polls (the full-history dashboard aggregate cost
        the service 9 h of CPU in three days). Entries expire after `ttl` seconds; data is never stored here."""
        now = time.monotonic()
        with self._ttl_lock:
            hit = self._ttl.get(key)
            if hit is not None and now - hit[0] < ttl:
                return hit[1]
        value = produce()
        with self._ttl_lock:
            self._ttl[key] = (time.monotonic(), value)
            if len(self._ttl) > 64:  # bounded: drop the oldest entries
                for stale in sorted(self._ttl, key=lambda k: self._ttl[k][0])[:16]:
                    del self._ttl[stale]
        return value

    def hardware_lane_context(self) -> dict[str, dict[str, Any]]:
        """Lanes a GPU is in use on right now -> what is using it. Idle lanes are absent, so the hardware
        log writes nothing for them. Read lock-free: the sampler thread must never wait on a model load."""
        context: dict[str, dict[str, Any]] = {}
        for target, handle in list(self.manager.handles.items()):
            for lane in handle.lanes:
                context[lane.key] = {"model": handle.model.name, "kind": "inference", "target": target}
        try:
            for lane, holder in self.leases.lane_holders().items():
                entry = context.setdefault(lane, {})
                entry.setdefault("kind", holder.get("kind"))
                entry["holder"] = holder.get("holder")
            for run in self.training.live_runs():
                for lane in run["lanes"]:
                    entry = context.setdefault(lane, {})
                    entry.setdefault("kind", "training")
                    entry["run_id"] = run["id"]
        except Exception:  # noqa: BLE001 - a locked database must not stop inference logging
            pass
        return context

    def _record_kernel_fault(self, kind: str, severity: str, payload: dict[str, Any]) -> None:
        self.training.add_event_to_active_runs(kind, severity, payload)
        self.hardware_log.record_event(kind, severity, payload)

    def _archive_loop(self) -> None:
        # first pass right away (writes every past day once), then hourly so today's file stays current
        while True:
            self.archive.archive_all()
            if self.exchange is not None:
                self.exchange.sync()
            if self._archive_stop.wait(3600):
                return

    def start_telemetry(self) -> None:
        # No retention delete: telemetry is history. Daily compressed archives are written beside the database.
        threading.Thread(target=self._archive_loop, name="telemetry-archive", daemon=True).start()
        try:  # checkpoints a trainer wrote but never reported (the service listed none for a finished run)
            recovered = self.service.rescan_unreported_checkpoints()
            if recovered:
                print(f"[checkpoints] registered unreported checkpoints from disk: {recovered}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[checkpoints] rescan failed: {exc}", flush=True)
        self.hardware.start()
        self.kernel_faults.start()
        if self.eval_worker is not None:
            self.eval_worker.start()

    def close(self) -> None:
        """Flush and stop the background usage/history writers (connections + threads)."""
        self.usage.close()
        self.history.close()

    def stop_telemetry(self) -> None:
        self.hardware.stop()
        self.hardware_log.close()
        self._archive_stop.set()
        self.archive.archive_all()
        if self.exchange is not None:
            try:
                self.exchange.export_runs()
                self.exchange.export_training_snapshot(force=True)
            except Exception as exc:  # noqa: BLE001 - an unplugged share must not block shutdown
                print(f"[exchange] final export failed: {exc}", flush=True)
        self.kernel_faults.stop()
        if self.eval_worker is not None:
            self.eval_worker.stop()
        for benchmark in self.benchmarks.list():
            self.benchmarks.cancel(benchmark["id"])

    def training_overview(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware.snapshot(),
            "lane_holders": self.lane_holders(),
            "kernel_faults": {"active": self.kernel_faults.active, "reason": self.kernel_faults.inactive_reason},
        }

    def bootstrap(self) -> dict[str, Any]:
        return {
            "app_version": APP_VERSION,
            "api_version": API_VERSION,
            "config_path": str(self.config_path),
            "network": self.network,
            "catalog": self.manager.catalog(),
            "status": self.manager.status(),
            "runs": self.store.recent(20),
            "projects": self.store.projects(),
            "coding_presets": {
                "temperature": 0.2,
                "top_p": 0.95,
                "max_tokens": None,
                "reasoning_budget": -1,
                "reasoning_effort": "default",
                "workload_kind": "coding_agent",
            },
        }

    def health(self) -> dict[str, Any]:
        status = self.manager.status()
        return {
            "status": "ok",
            "service": "dual-gpu-studio",
            "app_version": APP_VERSION,
            "api_version": API_VERSION,
            "ready": status.get("state") == "ready",
            "state": status.get("state"),
            "mode": status.get("mode"),
            "network": self.network,
            "host_os": self.host_os,
            "telemetry": self.telemetry_health(),
        }

    def telemetry_health(self) -> dict[str, Any]:
        """Everything that can silently stop telemetry, in one place: missing sensor sources, write errors,
        archive state, backups and interrupted runs."""
        snapshot = self.hardware.snapshot(history_points=1)
        smi = self.hardware.smi
        return {
            "gpu_sampler": {"available": snapshot["available"], "reason": snapshot["unavailable_reason"],
                            **snapshot["sampler"]},
            "throttle_source": getattr(smi, "source", None) if smi is not None and smi.available else None,
            "amd_smi": snapshot["amd_smi"],
            "hardware_log": self.hardware_log.status(),
            "archive": {"last_run": self.archive.last_run, "last_error": self.archive.last_error,
                        "files": len(self.archive.files())},
            "backups": self.backups,
            "interrupted_runs_marked": self.interrupted_runs,
            "exchange": self.exchange.status() if self.exchange is not None else None,
            "storage_warnings": datasafe.storage_warnings(self.data_dir),
            "archive_write_failures": {"history": self.history.write_failures, "history_records_lost": self.history.records_lost,
                                       "usage": self.usage.write_failures},
        }

    def register_client(self, payload: dict[str, Any]) -> dict[str, Any]:
        project_name = str(payload.get("project_name") or "").strip()
        if not project_name:
            raise AppError("project_name is required.")
        output_path = str(payload.get("output_path") or "").strip()
        if not output_path:
            raise AppError("output_path is required (where run metrics will be appended as JSON lines).")
        github_repo = payload.get("github_repo")
        github_repo = str(github_repo).strip() or None if github_repo else None
        return self.store.register_client(project_name, github_repo, output_path)

    def create_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.store.upsert_project(
                str(payload.get("name") or ""),
                str(payload.get("repo_path") or "").strip() or None,
                str(payload.get("git_remote") or "").strip() or None,
            )
        except ValueError as exc:
            raise AppError(str(exc)) from exc

    def create_agent_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        project_id = str(payload.get("project_id") or "").strip()
        if not project_id or self.store.get_project(project_id) is None:
            raise AppError("A valid project_id is required.")
        return self.store.create_agent_session(
            project_id,
            str(payload.get("agent_role") or "developer"),
            str(payload.get("runtime") or "unknown"),
        )

    def create_coding_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        project_id = str(payload.get("project_id") or "").strip()
        if not project_id or self.store.get_project(project_id) is None:
            raise AppError("A valid project_id is required.")
        try:
            return self.store.create_coding_task(
                project_id,
                str(payload.get("title") or ""),
                str(payload.get("branch") or "").strip() or None,
            )
        except ValueError as exc:
            raise AppError(str(exc)) from exc

    def record_tool_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        project_id = str(payload.get("project_id") or "").strip()
        if not project_id or self.store.get_project(project_id) is None:
            raise AppError("A valid project_id is required.")
        payload = {**payload, "project_id": project_id}
        return self.store.record_tool_event(payload)


def deployable_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """The current deployment as a request `deploy()` accepts. After an on-demand load or a lane
    replacement the profile lists `models` even for a single GPU, while deploy() wants `model`."""
    models = profile.get("models")
    if profile.get("mode") == "single_gpu" and "model" not in profile and isinstance(models, list) and len(models) == 1:
        return {**{key: value for key, value in profile.items() if key != "models"}, "model": models[0]}
    return profile


def wait_for_exit(pid: int, timeout: float = 90.0) -> bool:
    """Block until process `pid` has exited (a restarting predecessor releasing its port)."""
    host = get_host()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if host.process.identity(pid) is None:
            return True
        time.sleep(0.25)
    return False


class ServiceControl:
    """Restart or stop this service from the UI and API.

    A restart stops every model server first (and any leftover one on a lane port), starts a
    replacement process that waits for this one to exit before binding the port (`--replace-pid`),
    then shuts this one down. With `reload_models` the loaded deployment is saved and the
    replacement loads it again.
    """

    RESTORE_FILE = "restart_state.json"
    RESTORE_MAX_AGE_SECONDS = 600.0

    def __init__(self, state: "ApplicationState", args: argparse.Namespace):
        self.state = state
        self.args = args
        self.server: Any = None  # the listening HTTP server; set by main() once it is up
        self.started_at = utc_now()
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self.pending: str | None = None

    @property
    def restore_path(self) -> Path:
        return self.state.data_dir / self.RESTORE_FILE

    def info(self) -> dict[str, Any]:
        network = getattr(self.state, "network", None) or network_metadata(self.args.host, self.args.port)
        return {
            "pid": os.getpid(),
            "started_at": self.started_at,
            "uptime_seconds": round(time.monotonic() - self._started),
            "host": self.args.host,
            "port": self.args.port,
            "api_version": API_VERSION,
            "network": network,
            "config_path": str(self.state.config_path),
            "data_dir": str(self.state.data_dir),
            "app_version": APP_VERSION,
            "python": sys.executable,
            "host_os": os_info(),
            "can_restart": self.server is not None,
            "pending": self.pending,
            "restore_pending": self.restore_path.exists(),
        }

    def request(self, action: str, reload_models: bool = False) -> dict[str, Any]:
        if action not in {"restart", "stop"}:
            raise AppError(f"Unknown service action: {action}")
        if self.server is None:
            raise AppError("This process cannot restart itself (it was not started by app.py).", HTTPStatus.CONFLICT)
        with self._lock:
            if self.pending:
                raise AppError(f"A service {self.pending} is already in progress.", HTTPStatus.CONFLICT)
            self.pending = action
        manager = self.state.manager
        profile = manager.active_profile if reload_models and action == "restart" else None
        try:
            if profile:
                self.restore_path.write_text(json.dumps({"created": time.time(), "profile": profile}), encoding="utf-8")
            else:
                self.restore_path.unlink(missing_ok=True)
            manager.unload()  # models stop before the reply, so a busy deployment is reported to the caller
        except BaseException:
            with self._lock:
                self.pending = None
            self.restore_path.unlink(missing_ok=True)
            raise
        threading.Thread(target=self._finish, args=(action,), name="service-control", daemon=True).start()
        return {"ok": True, "action": action, "reload_models": bool(profile), "pid": os.getpid()}

    def _finish(self, action: str) -> None:
        time.sleep(0.5)  # let the HTTP reply reach the browser before the listener goes away
        try:
            if action == "restart":
                self._spawn_replacement()
        except Exception as exc:  # noqa: BLE001 -- keep serving rather than leave no service at all
            print(f"[service] restart failed, still running: {exc}", flush=True)
            with self._lock:
                self.pending = None
            self.restore_path.unlink(missing_ok=True)
            return
        self.server.shutdown()

    def replacement_command(self) -> list[str]:
        command = [
            sys.executable, str(Path(__file__).resolve()), "--config", str(self.state.config_path),
            "--host", self.args.host, "--port", str(self.args.port), "--data-dir", str(self.state.data_dir),
            "--no-browser", "--replace-pid", str(os.getpid()),
        ]
        for origin in getattr(self.args, "cors_origin", []) or []:
            command.extend(["--cors-origin", origin])
        return command

    def _spawn_replacement(self) -> None:
        command = self.replacement_command()
        with open(self.state.data_dir / "launcher.log", "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} :: restart from the UI :: {' '.join(command)} ===\n")
            log.write(f"host_os={os_log_line()}\n")
            log.flush()
            subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             **detached_process_kwargs())

    def restore_after_restart(self, allowed: bool = True) -> bool:
        """Reload a saved deployment only for this app's controlled replacement process.

        An ordinary/manual launch (including one after a Windows reboot) discards any
        leftover restore file so it cannot unexpectedly load a GPU model.
        """
        if not allowed:
            self.restore_path.unlink(missing_ok=True)
            return False
        try:
            raw = self.restore_path.read_text(encoding="utf-8")
        except OSError:
            return False
        self.restore_path.unlink(missing_ok=True)  # one shot: a crash must not replay it on a later start
        try:
            payload = json.loads(raw)
        except ValueError:
            return False
        profile = payload.get("profile") if isinstance(payload, dict) else None
        fresh = isinstance(payload, dict) and time.time() - float(payload.get("created", 0)) <= self.RESTORE_MAX_AGE_SECONDS
        if not isinstance(profile, dict) or not fresh:
            return False
        profile = deployable_profile(profile)

        def reload() -> None:
            try:
                self.state.manager.deploy(profile)
                print("[service] reloaded the models that were loaded before the restart", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[service] could not reload the previous models: {exc}", flush=True)

        threading.Thread(target=reload, name="restore-models", daemon=True).start()
        return True


class RequestHandler(BaseHTTPRequestHandler):
    server: ChatbotHTTPServer
    protocol_version = "HTTP/1.1"
    # Seconds a connection may sit silent (a half-open client, an idle keep-alive) before its thread is released.
    timeout = 300

    def log_message(self, format: str, *args: Any) -> None:
        # BaseHTTPRequestHandler.log_request passes (request line, status, size); errors pass other shapes.
        if len(args) >= 2 and is_routine_poll(str(args[0]), args[1]):
            return
        logging.getLogger(LOGGER_NAME).info(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")

    def _cors_origin(self) -> str | None:
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            return None
        allowed = getattr(self.server, "cors_origins", frozenset())
        if origin in allowed:
            return origin
        if "*" in allowed:
            return "*"
        return None

    def _write_cors_headers(self) -> None:
        origin = self._cors_origin()
        if origin is None:
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", CORS_ALLOWED_HEADERS)
        self.send_header("Access-Control-Expose-Headers", "X-DGPU-Run-ID, X-DGPU-API-Version")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Vary", "Origin")

    def _write_common_headers(self) -> None:
        self.send_header("X-DGPU-API-Version", API_VERSION)
        self._write_cors_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        # A configured origin gets a normal preflight response. An unconfigured origin receives
        # no allow headers, so browsers reject it without making state-changing requests.
        self.send_response(HTTPStatus.NO_CONTENT)
        self._write_common_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/":
            self._send_bytes(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/bootstrap":
            self._send_json(HTTPStatus.OK, self.server.app_state.bootstrap())
            return
        if path == "/api/health":
            state = self.server.app_state
            self._send_json(HTTPStatus.OK, self._cached("health", 2.0, state.health))
            return
        if path == "/api/status":
            manager = self.server.app_state.manager
            self._send_json(
                HTTPStatus.OK,
                {
                    **manager.status(),
                    "api_version": API_VERSION,
                    "network": self.server.app_state.network,
                    "orphans": manager.orphans_cached(),
                    "transitioning": manager.transition_lock.locked(),
                },
            )
            return
        if path == "/api/service":
            try:
                self._send_json(HTTPStatus.OK, self._service_control().info())
            except AppError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            return
        if path == "/api/settings":
            self._send_json(HTTPStatus.OK, self._settings_payload())
            return
        if path == "/api/dashboard":
            project_id = (query.get("project_id") or [None])[0]
            workload_kind = (query.get("workload_kind") or [None])[0]
            start_date = (query.get("start_date") or [None])[0]
            end_date = (query.get("end_date") or [None])[0]
            try:
                for value in (start_date, end_date):
                    if value:
                        datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST,
                                {"error": "start_date and end_date must be YYYY-MM-DD."})
                return
            try:
                host_os = normalize_os((query.get("os") or [None])[0])
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            state = self.server.app_state
            self._send_json(HTTPStatus.OK, self._cached(
                f"dashboard:{project_id}:{workload_kind}:{start_date}:{end_date}:{host_os}", 5.0,
                lambda: state.store.dashboard(project_id, workload_kind, start_date, end_date, host_os)))
            return
        if path == "/api/usage":
            start_date = (query.get("start_date") or [None])[0]
            end_date = (query.get("end_date") or [None])[0]
            try:
                for value in (start_date, end_date):
                    if value:
                        datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST,
                                {"error": "start_date and end_date must be YYYY-MM-DD."})
                return
            rows = self.server.app_state.usage.query(
                (query.get("project_id") or [None])[0], (query.get("connection_type") or [None])[0],
                start_date, end_date)
            totals: dict[str, dict[str, int]] = {}
            for row in rows:
                bucket = totals.setdefault(row["connection_type"],
                                           {"requests": 0, "input_tokens": 0, "output_tokens": 0})
                for field in bucket:
                    bucket[field] += row[field]
            self._send_json(HTTPStatus.OK, {"rows": rows, "by_connection_type": totals})
            return
        if path == "/api/runs":
            try:
                limit = int((query.get("limit") or [100])[0])
                offset = int((query.get("offset") or [0])[0])
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "limit and offset must be integers"})
                return
            try:
                host_os = normalize_os((query.get("os") or [None])[0])
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            project_id = (query.get("project_id") or [None])[0]
            workload_kind = (query.get("workload_kind") or [None])[0]
            store = self.server.app_state.store
            self._send_json(
                HTTPStatus.OK,
                {
                    "runs": store.recent(limit, offset, project_id, workload_kind, host_os),
                    "total": store.count_runs(project_id, workload_kind, host_os),
                    "limit": max(1, min(limit, 500)),
                    "offset": max(0, offset),
                },
            )
            return
        if path == "/api/chat-history":
            history = self.server.app_state.chat.history
            if not history.enabled:
                self._send_json(HTTPStatus.OK, {"enabled": False, "turns": [],
                                                "note": "Set project.store_chat_history = true to archive conversations."})
                return
            try:
                limit = int((query.get("limit") or [100])[0])
                offset = int((query.get("offset") or [0])[0])
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "limit and offset must be integers"})
                return
            start_date = (query.get("start_date") or [None])[0]
            end_date = (query.get("end_date") or [None])[0]
            try:
                for value in (start_date, end_date):
                    if value:
                        datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "start_date and end_date must be YYYY-MM-DD."})
                return
            include_content = (query.get("content") or ["0"])[0].lower() in {"1", "true", "yes"}
            self._send_json(HTTPStatus.OK, {
                "enabled": True,
                "turns": history.query(
                    project_id=(query.get("project_id") or [None])[0],
                    start_date=start_date,
                    end_date=end_date,
                    limit=min(max(1, limit), 500),
                    offset=max(0, offset),
                    include_content=include_content,
                    model_name=(query.get("model_name") or [None])[0],
                    target=(query.get("target") or [None])[0],
                    status=(query.get("status") or [None])[0],
                ),
                "limit": min(max(1, limit), 500), "offset": max(0, offset),
            })
            return
        if path == "/api/projects":
            self._send_json(HTTPStatus.OK, {"projects": self.server.app_state.store.projects()})
            return
        if path == "/v1/models" or (path.startswith("/v1/") and path.endswith("/models")):
            manager = self.server.app_state.manager
            servers = manager.status().get("servers", [])
            data = [{"id": server["target"], "object": "model", "owned_by": "dual-gpu-studio"} for server in servers]
            if manager.config.policy.auto_load_models:
                # Every catalog model is requestable by name; it loads on demand (GPU only).
                with manager.lock:
                    records = [record for record in manager.model_records if record.get("exists")]
                names = [record["name"] for record in records]
                loaded = {server["model"] for server in servers}
                data += [
                    {"id": record["name"] if names.count(record["name"]) == 1 else record["id"], "object": "model",
                     "owned_by": "dual-gpu-studio", "loaded": record["name"] in loaded}
                    for record in records
                ]
            self._send_json(HTTPStatus.OK, {"object": "list", "data": data})
            return
        if path.startswith("/api/runs/") and path.endswith("/hardware"):
            self._hardware_for_run(path.split("/")[3])
            return
        if path.startswith("/api/runs/"):
            run_id = path.rsplit("/", 1)[-1]
            run = self.server.app_state.store.get(run_id)
            if run is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Run not found"})
            else:
                self._send_json(HTTPStatus.OK, run)
            return
        if path == "/api/clients/schema":
            self._send_json(HTTPStatus.OK, CLIENT_METRICS_SCHEMA)
            return
        if path == "/api/lmstudio/gpus":
            try:
                self._send_json(HTTPStatus.OK, get_selection(
                    self.server.app_state.config, (query.get("backend") or ["rocm"])[0]))
            except GpuSelectionError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            return
        if path == "/api/leases":
            try:
                self._send_json(HTTPStatus.OK, self._leases_api("GET", path, query, {}))
            except AppError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            return
        if path.startswith("/api/hardware/"):
            self._hardware_get(path, query)
            return
        if path == "/api/hardware" or path.startswith("/api/training/"):
            try:
                result = self._training_get(path, query)
                if isinstance(result, FileResponse):
                    self._send_file(result)
                else:
                    self._send_json(HTTPStatus.OK, result)
            except TrainingError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid query parameter."})
            return
        if path.startswith("/api/clients/") and path.endswith("/runs"):
            client_id = path.split("/")[3]
            self._send_json(HTTPStatus.OK, {"runs": self.server.app_state.store.runs_for_client(client_id)})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in JSON_ONLY_CONTROL_POSTS:
                self._require_json_content_type()  # a web page cannot trigger these with a blind form POST
            payload = self._read_json()
            if path == "/v1/chat/completions" or (
                path.startswith("/v1/") and path.endswith("/chat/completions")
            ):
                self._proxy_openai_chat(path, payload)
                return
            if path.endswith("/infill") and (path == "/infill" or path.startswith("/v1/")):
                self._proxy_openai_completions(path, payload, infill=True)
                return
            if path.endswith("/completions") and path.startswith("/v1/"):
                # chat/completions handled above; this is the raw FIM completions endpoint
                self._proxy_openai_completions(path, payload)
                return
            if path == "/api/deploy":
                result = self.server.app_state.manager.deploy(payload)
            elif path == "/api/models/refresh":
                self.server.app_state.manager.refresh_models()
                result = {"catalog": self.server.app_state.manager.catalog()}
            elif path == "/api/lanes/auto-use":
                if "lane" not in payload or not isinstance(payload.get("enabled"), bool):
                    raise AppError("lane and enabled (true/false) are required.")
                result = self.server.app_state.manager.set_lane_auto_use(str(payload["lane"]), payload["enabled"])
            elif path == "/api/unload":
                lane = payload.get("lane")
                expected = payload.get("model")
                if expected and not lane:
                    raise AppError("`model` only applies together with `lane`: it makes the unload conditional "
                                   "on that model still serving the lane.")
                result = self.server.app_state.manager.unload(
                    lane=str(lane) if lane else None, orphans_only=payload.get("orphans_only") is True,
                    force=payload.get("force") is True, model=str(expected) if expected else None)
            elif path in ("/api/service/restart", "/api/service/stop"):
                self._require_json_content_type()
                action = path.rsplit("/", 1)[-1]
                result = self._service_control().request(action, reload_models=payload.get("reload_models") is True)
            elif path == "/api/settings":
                self._require_json_content_type()
                result = self._apply_settings(payload)
            elif path == "/api/chat":
                result = self.server.app_state.chat.chat(payload)
            elif path == "/api/clients/register":
                result = self.server.app_state.register_client(payload)
            elif path == "/api/projects":
                result = self.server.app_state.create_project(payload)
            elif path == "/api/agent-sessions":
                result = self.server.app_state.create_agent_session(payload)
            elif path == "/api/coding-tasks":
                result = self.server.app_state.create_coding_task(payload)
            elif path == "/api/tool-events":
                result = self.server.app_state.record_tool_event(payload)
            elif path == "/api/clients/deploy":
                context_window = payload.get("context_window")
                input_tokens = payload.get("input_tokens")
                max_output_tokens = payload.get("max_output_tokens")
                result = self.server.app_state.manager.deploy_for_client(
                    str(payload.get("model", "")),
                    str(payload["gpu"]) if payload.get("gpu") else None,
                    int(context_window) if context_window is not None else None,
                    int(input_tokens) if input_tokens is not None else None,
                    int(max_output_tokens) if max_output_tokens is not None else None,
                )
            elif path == "/api/clients/deploy_parallel":
                context_window = payload.get("context_window")
                input_tokens = payload.get("input_tokens")
                max_output_tokens = payload.get("max_output_tokens")
                result = self.server.app_state.manager.deploy_parallel_for_client(
                    [str(name) for name in (payload.get("models") or [])],
                    int(context_window) if context_window is not None else None,
                    int(input_tokens) if input_tokens is not None else None,
                    int(max_output_tokens) if max_output_tokens is not None else None,
                )
            elif path == "/api/clients/deploy_lane":
                context_window = payload.get("context_window")
                input_tokens = payload.get("input_tokens")
                max_output_tokens = payload.get("max_output_tokens")
                result = self.server.app_state.manager.deploy_lane_for_client(
                    str(payload.get("model", "")),
                    str(payload.get("lane", "")),
                    int(context_window) if context_window is not None else None,
                    int(input_tokens) if input_tokens is not None else None,
                    int(max_output_tokens) if max_output_tokens is not None else None,
                )
            elif path == "/api/clients/check_fit":
                context_window = payload.get("context_window")
                input_tokens = payload.get("input_tokens")
                max_output_tokens = payload.get("max_output_tokens")
                result = self.server.app_state.manager.check_fit(
                    str(payload.get("model", "")),
                    str(payload.get("lane", "")),
                    int(context_window) if context_window is not None else None,
                    int(input_tokens) if input_tokens is not None else None,
                    int(max_output_tokens) if max_output_tokens is not None else None,
                )
            elif path in ("/api/lmstudio/gpus", "/api/lmstudio/release"):
                # These edit LM Studio's config / unload its models. Requiring a JSON content type
                # forces a CORS preflight, so a web page cannot trigger them with a blind form POST.
                if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                    raise AppError("Content-Type must be application/json.", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                config = self.server.app_state.config
                try:
                    if path == "/api/lmstudio/gpus":
                        result = set_selection(config, payload.get("gpus"), str(payload.get("backend") or "rocm"))
                    else:
                        result = release(config, force=payload.get("force") is True,
                                         stop_server=payload.get("stop_server") is True)
                except GpuSelectionError as exc:
                    raise AppError(str(exc), exc.status) from exc
            elif path.startswith("/api/leases"):
                result = self._leases_api("POST", path, {}, payload)
            elif path.startswith("/api/training/"):
                result = self._training_post(path, payload)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            self._send_json(HTTPStatus.OK, result)
        except AppError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except TrainingError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body"})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if not parsed.path.startswith("/api/leases/"):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            self._send_json(HTTPStatus.OK, self._leases_api("DELETE", parsed.path, parse_qs(parsed.query), {}))
        except AppError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _leases_api(self, method: str, path: str, query: dict[str, list[str]], payload: dict[str, Any]) -> dict[str, Any]:
        """GET /api/leases | POST /api/leases | POST /api/leases/{id}/renew|preempt | DELETE /api/leases/{id}.
        Renew and release need the token returned at acquire (body `token`, `?token=`, or X-Lease-Token)."""
        state = self.server.app_state
        parts = [part for part in path.split("/") if part]
        token = payload.get("token") or (query.get("token") or [None])[0] or self.headers.get("X-Lease-Token")
        try:
            if method == "GET" and len(parts) == 2:
                return {"leases": state.leases.list_active() + state.inference_leases(),
                        "lane_holders": state.lane_holders()}
            if method == "POST" and len(parts) == 2:
                lanes = [str(lane) for lane in payload.get("lanes") or []]
                known = {lane.key for lane in state.config.lanes}
                if not set(lanes) <= known:
                    raise AppError(f"Unknown lane(s): {', '.join(sorted(set(lanes) - known))}.")
                busy = sorted(set(lanes) & state.manager.busy_lane_keys())
                if busy:
                    raise AppError(f"Inference is loaded on lane(s) {', '.join(busy)}.", HTTPStatus.CONFLICT)
                return state.leases.acquire(
                    str(payload.get("holder") or ""), lanes, kind=str(payload.get("kind") or "training"),
                    ttl_s=float(payload.get("ttl_s") or 120.0),
                    pid=int(payload["pid"]) if payload.get("pid") else None,
                )
            if len(parts) == 4 and method == "POST" and parts[3] == "renew":
                ttl = payload.get("ttl_s")
                return state.leases.renew(parts[2], token, float(ttl) if ttl else None)
            if len(parts) == 4 and method == "POST" and parts[3] == "preempt":
                return state.leases.terminate_holder(parts[2])
            if len(parts) == 3 and method == "DELETE":
                return state.leases.release(parts[2], token)
        except LeaseError as exc:
            raise AppError(str(exc), exc.status) from exc
        raise AppError("Not found.", HTTPStatus.NOT_FOUND)

    def _cached(self, key: str, ttl: float, produce: Any) -> Any:
        cache = getattr(self.server.app_state, "cached", None)
        return cache(key, ttl, produce) if cache else produce()

    @staticmethod
    def _epoch(value: Any) -> float | None:
        """Query values may be epoch seconds or an ISO timestamp."""
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()

    def _hardware_get(self, path: str, query: dict[str, list[str]]) -> None:
        """Saved hardware log: /api/hardware/{log,history,summary,alerts,events,export,archive}.
        Ranges are `start`/`end` (epoch seconds or ISO); default is the last hour."""
        state = self.server.app_state
        log = state.hardware_log

        def arg(name: str) -> str | None:
            return (query.get(name) or [None])[0]

        try:
            end = self._epoch(arg("end")) or time.time()
            start = self._epoch(arg("start"))
            start = end - 3600 if start is None else start
            host_os = normalize_os(arg("os"))
            name = path.removeprefix("/api/hardware/")
            if name == "log":
                self._send_json(HTTPStatus.OK, {**log.status(), "telemetry": state.telemetry_health(),
                                                "archive": state.archive.list_days()})
            elif name == "history":
                self._send_json(HTTPStatus.OK, log.history(start, end, int(arg("max_points") or 600), host_os))
            elif name == "summary":
                self._send_json(HTTPStatus.OK, log.summary(start, end, arg("lane"), host_os))
            elif name == "alerts":
                self._send_json(HTTPStatus.OK, {"alerts": log.alerts_between(
                    self._epoch(arg("start")), self._epoch(arg("end")), open_only=arg("open") == "1", host_os=host_os)})
            elif name == "events":
                self._send_json(HTTPStatus.OK, {"events": log.events_between(
                    self._epoch(arg("start")), self._epoch(arg("end")), host_os=host_os)})
            elif name == "range":
                self._send_json(HTTPStatus.OK, log.data_range(host_os))
            elif name == "archive":
                self._send_json(HTTPStatus.OK, {"files": state.archive.list_days(), "last_run": state.archive.last_run,
                                                "exchange": state.exchange.status() if state.exchange else None})
            elif name == "export":
                fmt = (arg("format") or "csv").lower()
                if fmt not in ("csv", "jsonl"):
                    raise AppError("format must be csv or jsonl.")
                content_type, body = log.export(start, end, fmt, arg("table") or "hardware_samples", state.archive, host_os)
                self._send_bytes(HTTPStatus.OK, body.encode("utf-8"), content_type)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except AppError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid query parameter: {exc}"})

    def _hardware_for_run(self, run_id: str) -> None:
        """Hardware summary and series over one inference run's own time window."""
        state = self.server.app_state
        run = state.store.get(run_id)
        if run is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Run not found"})
            return
        start = self._epoch(run["created_at"])
        end = self._epoch(run.get("finished_at")) or time.time()
        self._send_json(HTTPStatus.OK, {
            "run_id": run_id, "start": start, "end": end,
            "summary": state.hardware_log.summary(start, end),
            "history": state.hardware_log.history(start, end, 300),
        })

    def _training_get(self, path: str, query: dict[str, list[str]]) -> Any:
        state = self.server.app_state
        if path == "/api/hardware":
            return state.training_overview()
        return state.service.get(path, query)

    def _training_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.server.app_state.service.post(path, payload)

    def _send_file(self, response: FileResponse) -> None:
        size = response.path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{response.filename}"')
        self.send_header("X-Content-Type-Options", "nosniff")
        self._write_common_headers()
        self.end_headers()
        with open(response.path, "rb") as handle:
            while chunk := handle.read(1 << 20):
                self.wfile.write(chunk)

    def _resolve_lane_handle(self, path: str, payload: dict[str, Any]) -> tuple[str, "ServerHandle"]:
        """Pick the active lane for an OpenAI-compatible request. Base URLs: /v1 (model selects
        a lane), /v1/<lane>, /v1/<lane>/<role>. Shared by the chat and completions proxies so
        lane routing is identical for agent and autocomplete traffic.

        A `model` naming a catalog model (name or id) is served by that model: loaded on demand
        onto the GPUs when it is not already, evicting idle models (see ensure_model). `dgpu:<lane>`
        and unknown model names keep plain lane routing. The returned handle is claimed; the
        caller must enter `handle.serving(claimed=True)`."""
        state = self.server.app_state
        manager = state.manager
        path_parts = [part for part in path.split("/") if part]
        path_target = path_parts[1] if len(path_parts) >= 4 else None
        target_hint = self.headers.get("X-DGPU-Target") or path_target
        requested_model = str(payload.get("model") or "")
        if (manager.config.policy.auto_load_models and requested_model
                and not requested_model.startswith("dgpu:") and manager.knows_model(requested_model)):
            if target_hint and target_hint not in {lane.key for lane in manager.config.lanes}:
                raise AppError(f"Unknown GPU lane '{target_hint}'.")
            return manager.ensure_model(requested_model, lane_key=target_hint)
        with manager.lock:
            manager._mark_dead_lanes()
            if manager.state != "ready" or not manager.handles:
                raise AppError("Load a deployment before sending a message.", HTTPStatus.CONFLICT)
            handles = dict(manager.handles)
            if target_hint and target_hint not in handles:
                raise AppError(f"GPU lane '{target_hint}' is not loaded.", HTTPStatus.CONFLICT)
            target = target_hint if target_hint in handles else None
            if target is None:
                model_hint = requested_model.removeprefix("dgpu:")
                if model_hint in handles:
                    target = model_hint
                else:
                    matches = [key for key, handle in handles.items() if handle.model.name == requested_model]
                    if len(matches) == 1:
                        target = matches[0]
            if target is None and len(handles) == 1:
                target = next(iter(handles))
            if target is None:
                raise AppError("Select a GPU lane with model='dgpu:<lane>' or a /v1/<lane> base URL.")
            handles[target].claim()
            return target, handles[target]

    def _resolve_project(self) -> tuple[str, dict[str, Any] | None]:
        """Watertight, zero-config project attribution -- no API key, no pre-registration.

        Every request maps to a real project id, in this order:
          1. Bearer is a registered client_id -> that client's project.
          2. Otherwise a project name/id from the X-DGPU-Project header, else the Bearer key
             itself -> created on first use (so a tool with only an API-key field just sets a
             project name there and is tracked).
          3. Nothing identifying -> the configured default project.
        Returns (project_id, client_or_None). Never raises for an unknown key and never leaves
        a request unattributed.
        """
        state = self.server.app_state
        authorization = self.headers.get("Authorization", "")
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        client = state.store.get_client(token) if token and token != "local" else None
        if client is not None:
            project_id = client.get("project_id")
            if not project_id:
                project = state.store.upsert_project(client["project_name"], git_remote=client.get("github_repo"))
                project_id = project["id"]
                state.store.link_client_project(client["id"], project_id)
            return project_id, client
        name = (self.headers.get("X-DGPU-Project") or (token if token and token != "local" else "")).strip()
        if not name:
            name = state.config.project.default_project
        project = state.store.get_project(name) or state.store.upsert_project(name)
        return project["id"], None

    def _request_project_id(self) -> str | None:
        return self._resolve_project()[0]

    def _proxy_openai_completions(self, path: str, payload: dict[str, Any], infill: bool = False) -> None:
        """Proxy FIM autocomplete while asynchronously archiving per-model input/output telemetry.

        Autocomplete can fire on every typing pause, so it deliberately avoids synchronous run-row
        writes and GPU probes. The shared background sampler supplies resource telemetry and the
        full request/response is queued to ChatHistoryStore in O(1), off the inference path.
        """
        state = self.server.app_state
        target, handle = self._resolve_lane_handle(path, payload)
        connection_type = self.headers.get("X-DGPU-Connection-Type") or "autocomplete"
        project_id, client = self._resolve_project()
        suffix = "/infill" if infill else "/v1/completions"
        request_payload = dict(payload)
        request_payload["model"] = handle.model.name
        stream = bool(request_payload.get("stream"))
        history_on = (
            state.chat.history.enabled
            and payload.get("store_history") is not False
            and self.headers.get("X-DGPU-No-History", "").lower() not in {"1", "true", "yes"}
        )
        run_id = str(uuid.uuid4())
        baseline = state.manager.sampler.latest()
        started = time.monotonic()
        backend_payload: dict[str, Any] = {}
        output_parts: list[str] = []
        finish_reason: str | None = None
        status = "completed"
        error_text: str | None = None
        concurrent_at_start: int | None = None
        try:
            request = urlrequest.Request(
                f"{handle.process.base_url}{suffix}",
                data=json.dumps(handle.pin_serving_slot(request_payload)).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
                method="POST",
            )
            with handle.serving(claimed=True) as concurrent_at_start, urlrequest.urlopen(request, timeout=120) as response:
                if stream:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("X-DGPU-Run-ID", run_id)
                    self._write_common_headers()
                    self.end_headers()
                    self.close_connection = True
                    done = False
                    while True:
                        line = response.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
                        stripped = line.strip()
                        if done and not stripped:
                            break  # emit [DONE]'s terminating blank line, then stop
                        if stripped == b"data: [DONE]":
                            done = True
                            continue
                        if not stripped.startswith(b"data:"):
                            continue
                        try:
                            chunk = json.loads(stripped[5:].strip())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        backend_payload.update({
                            key: value for key, value in chunk.items()
                            if key != "choices" and value is not None
                        })
                        choices = chunk.get("choices") or []
                        if choices:
                            choice = choices[0]
                            delta = choice.get("delta") or {}
                            text = choice.get("text") or delta.get("content") or ""
                            if text:
                                output_parts.append(str(text))
                            if choice.get("finish_reason") is not None:
                                finish_reason = str(choice["finish_reason"])
                    if not done:
                        raise RuntimeError("Model stream ended before [DONE]; completion may be truncated.")
                else:
                    body = response.read()
                    try:
                        parsed = json.loads(body)
                    except json.JSONDecodeError:
                        parsed = {}
                    if isinstance(parsed, dict):
                        backend_payload = parsed
                        choice = (parsed.get("choices") or [{}])[0]
                        message = choice.get("message") or {}
                        output = choice.get("text") or message.get("content") or parsed.get("content") or ""
                        output_parts = [str(output)] if output else []
                        finish_reason = choice.get("finish_reason")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("X-DGPU-Run-ID", run_id)
                    self._write_common_headers()
                    self.end_headers()
                    self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as exc:
            status = "cancelled"
            error_text = str(exc)
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            status = "failed"
            error_text = detail
            raise AppError(f"Model server returned HTTP {exc.code}: {detail}", exc.code) from exc
        except Exception as exc:
            status = "failed"
            error_text = str(exc)
            raise
        finally:
            elapsed = time.monotonic() - started
            usage = ChatService._normalize_usage(backend_payload)
            if usage.get("input_tokens") is None:
                usage["input_tokens"] = backend_payload.get("tokens_evaluated")
            if usage.get("output_tokens") is None:
                usage["output_tokens"] = backend_payload.get("tokens_predicted")
            timing = ChatService._normalize_timing(backend_payload, usage, elapsed)
            timing["concurrent_requests_at_start"] = concurrent_at_start
            samples = state.chat._sample_window(started, baseline)
            resources = ChatService._resource_summary(samples, target)
            state.chat.usage.record(
                project_id, connection_type, usage.get("input_tokens"), usage.get("output_tokens")
            )
            if history_on:
                state.chat.history.enqueue({
                    "id": run_id,
                    "created_at": utc_now(),
                    "project_id": project_id,
                    "client_id": client["id"] if client else None,
                    "model_name": handle.model.name,
                    "target": target,
                    "workload_kind": connection_type,
                    "status": status,
                    "request": request_payload,
                    "usage": usage,
                    "timing": timing,
                    "resources": resources,
                    "finish_reason": finish_reason,
                    "output_text": "".join(output_parts) or None,
                    "error_text": error_text,
                })

    def _proxy_openai_chat(self, path: str, payload: dict[str, Any]) -> None:
        """Stream an OpenAI-compatible coding-agent request through an active GPU lane.

        Supported base URLs are /v1 (model selects a lane), /v1/<lane>, and
        /v1/<lane>/<role>. A registered client_id used as the Bearer key supplies project
        attribution, which works with clients such as Cline that expose an API-key field but
        do not expose arbitrary HTTP headers.
        """
        state = self.server.app_state
        path_parts = [part for part in path.split("/") if part]
        path_role = path_parts[2] if len(path_parts) >= 5 else None
        target, handle = self._resolve_lane_handle(path, payload)

        # Zero-config, watertight attribution: auto-provision the project from the caller's
        # Bearer key or X-DGPU-Project header (or the default), never 401 on an unknown key.
        project_id, client = self._resolve_project()
        agent_role = (
            self.headers.get("X-DGPU-Agent-Role")
            or path_role
            or ("cline" if "cline" in self.headers.get("User-Agent", "").lower() else "coding-agent")
        ).strip().lower()
        session_id = self.headers.get("X-DGPU-Session") or None
        coding_task_id = self.headers.get("X-DGPU-Task") or None
        task_label = self.headers.get("X-DGPU-Task-Label") or None
        connection_type = self.headers.get("X-DGPU-Connection-Type") or "coding_agent"

        request_payload = dict(payload)
        request_payload["model"] = handle.model.name
        request_payload, reasoning_settings = state.chat._coding_request(handle, request_payload)
        stream = bool(request_payload.get("stream"))
        if stream:
            # Request final token counts unless the caller explicitly opts out.
            stream_options = dict(request_payload.get("stream_options") or {})
            stream_options.setdefault("include_usage", True)
            request_payload["stream_options"] = stream_options
        capture_content = self.headers.get("X-DGPU-Capture-Content", "").lower() in {"1", "true", "yes"}
        # Full-conversation archive: on unless disabled in config or opted out per request. It
        # needs the streamed content, so accumulate output/reasoning when either it or the
        # metrics-DB content capture is active (kept out of the metrics DB regardless).
        history_on = state.chat.history.enabled and self.headers.get("X-DGPU-No-History", "").lower() not in {"1", "true", "yes"}
        keep_content = capture_content or history_on
        stored_request = dict(request_payload)
        if not capture_content and isinstance(stored_request.get("messages"), list):
            stored_request["messages"] = [
                {
                    "role": message.get("role"),
                    "content_redacted": True,
                    "content_characters": len(str(message.get("content") or "")),
                }
                for message in stored_request["messages"]
                if isinstance(message, dict)
            ]
        run_id = str(uuid.uuid4())
        # Reuse the path resolved at deploy time; re-globbing the models dir on
        # every agent tool turn only adds latency for a value that cannot change.
        model_path = getattr(handle.process, "model_path", None) or resolve_model_path(state.config, handle.model)
        state.store.begin_run(
            {
                "id": run_id,
                "comparison_id": None,
                "deployment_id": state.manager.deployment_id,
                "created_at": utc_now(),
                "mode": state.manager.mode,
                "target": target,
                "model_name": handle.model.name,
                "model_path": str(model_path),
                "lane_keys": [lane.key for lane in handle.lanes],
                "device": handle.process.device,
                "context_window": handle.process.ctx_size,
                "reasoning_budget": handle.model.reasoning_budget,
                "request": stored_request,
                "configuration": {
                    "app_version": APP_VERSION,
                    "gateway": "openai-compatible",
                    "endpoint": handle.process.base_url,
                    "device": handle.process.device,
                    "lanes": [asdict(lane) for lane in handle.lanes],
                    "runtime": handle.runtime(),
                    "reasoning": reasoning_settings,
                },
                "client_id": client["id"] if client else None,
                "task_label": task_label,
                "project_id": project_id,
                "agent_session_id": session_id,
                "coding_task_id": coding_task_id,
                "agent_role": agent_role,
                "workload_kind": "coding_agent",
            }
        )
        # GPU sampling launches PowerShell/Get-Counter on Windows. The background
        # sampler already owns that work; never put it on every agent tool turn.
        baseline = state.manager.sampler.latest()
        started = time.monotonic()
        backend_payload: dict[str, Any] = {}
        output_parts: list[str] = []
        reasoning_parts: list[str] = []
        first_token_at: float | None = None
        reasoning_started_at: float | None = None
        reasoning_last_at: float | None = None
        first_visible_at: float | None = None
        reasoning_characters = 0
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        response_started = False
        concurrent_at_start: int | None = None
        history_record = {
            "id": run_id,
            "project_id": project_id,
            "client_id": client["id"] if client else None,
            "agent_role": agent_role,
            "agent_session_id": session_id,
            "coding_task_id": coding_task_id,
            "model_name": handle.model.name,
            "target": target,
            "workload_kind": "coding_agent",
            "request": request_payload,
            "messages": payload.get("messages"),
            "tools": payload.get("tools"),
        }
        try:
            request = urlrequest.Request(
                f"{handle.process.base_url}/v1/chat/completions",
                data=json.dumps(handle.pin_serving_slot(dict(request_payload))).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
                method="POST",
            )
            # serving() marks the lane busy so a deployment change drains before stopping it.
            with handle.serving(claimed=True) as concurrent_at_start, urlrequest.urlopen(request, timeout=900) as response:
                if not stream:
                    backend_payload = json.loads(response.read().decode("utf-8", errors="replace"))
                else:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("X-DGPU-Run-ID", run_id)
                    self._write_common_headers()
                    self.end_headers()
                    self.close_connection = True
                    response_started = True
                    stream_complete = False
                    while True:
                        line = response.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
                        decoded = line.decode("utf-8", errors="replace").strip()
                        if stream_complete and not decoded:
                            break  # [DONE] ends SSE; do not wait for backend TCP EOF.
                        if decoded.startswith("data:") and decoded[5:].strip() == "[DONE]":
                            stream_complete = True
                            continue
                        if not decoded.startswith("data:"):
                            continue
                        try:
                            chunk = json.loads(decoded[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise RuntimeError(f"Model stream error: {chunk['error']}")
                        backend_payload.update({key: value for key, value in chunk.items()
                                                if key != "choices" and value is not None})
                        choices = chunk.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content") or ""
                            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                            now = time.monotonic()
                            if reasoning:
                                if reasoning_started_at is None:
                                    reasoning_started_at = now
                                reasoning_last_at = now
                                reasoning_characters += len(reasoning)
                                backend_payload["reasoning_observed"] = True
                            if (content or delta.get("tool_calls")) and first_visible_at is None:
                                first_visible_at = now
                            if (content or reasoning or delta.get("tool_calls")) and first_token_at is None:
                                first_token_at = time.monotonic()
                            for call in (delta.get("tool_calls") or []):
                                # Deltas stream a tool call's arguments in fragments keyed by
                                # index; collect fragments now (O(1) append), join once at end.
                                slot = tool_calls_acc.setdefault(call.get("index", 0), {"name": "", "arguments": []})
                                function = call.get("function") or {}
                                if function.get("name"):
                                    slot["name"] = function["name"]
                                if function.get("arguments"):
                                    slot["arguments"].append(function["arguments"])
                            if keep_content:
                                output_parts.append(content)
                                reasoning_parts.append(reasoning)
                            if choices[0].get("finish_reason") is not None:
                                backend_payload["finish_reason"] = choices[0]["finish_reason"]
                    if not stream_complete:
                        raise RuntimeError("Model stream ended before [DONE]; completion may be truncated.")

            elapsed = time.monotonic() - started
            if not stream:
                choice = (backend_payload.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                output_parts = [message.get("content") or choice.get("text") or ""]
                reasoning_parts = [message.get("reasoning_content") or message.get("reasoning") or ""]
                backend_payload["finish_reason"] = choice.get("finish_reason")
            if stream:
                # Streamed reasoning deltas are not retained on backend_payload, so
                # hand the observed character count to usage normalization. It yields
                # the exact backend reasoning-token count when present, else a labeled
                # estimate from these characters.
                backend_payload["reasoning_characters"] = reasoning_characters
            usage = ChatService._normalize_usage(backend_payload)
            tool_calls = (
                [{"function": {"name": slot["name"], "arguments": "".join(slot["arguments"])}}
                 for slot in tool_calls_acc.values()]
                if stream
                else ((backend_payload.get("choices") or [{}])[0].get("message") or {}).get("tool_calls")
            )
            usage.update(ChatService._tool_call_stats(tool_calls))
            timing = ChatService._normalize_timing(backend_payload, usage, elapsed)
            if first_token_at is not None:
                timing["time_to_first_token_ms"] = round((first_token_at - started) * 1000, 2)
            timing["reasoning_duration_ms"] = (
                round((reasoning_last_at - reasoning_started_at) * 1000, 2)
                if reasoning_started_at is not None and reasoning_last_at is not None else None
            )
            timing["reasoning_duration_source"] = "observed_stream_span" if reasoning_started_at is not None else None
            timing["time_to_first_visible_token_ms"] = (
                round((first_visible_at - started) * 1000, 2) if first_visible_at is not None else None
            )
            timing["concurrent_requests_at_start"] = concurrent_at_start
            samples = state.chat._sample_window(started, baseline)
            resources = ChatService._resource_summary(samples, target)
            stored_backend = backend_payload if capture_content else {
                key: value
                for key, value in backend_payload.items()
                if key not in {"choices"}
            }
            final = {
                "status": "completed",
                "output_text": "".join(output_parts) if capture_content else None,
                "reasoning_text": ("".join(reasoning_parts) or None) if capture_content else None,
                "finish_reason": backend_payload.get("finish_reason"),
                "usage": usage,
                "timing": timing,
                "resources": resources,
                "backend_response": stored_backend,
                "reasoning": reasoning_settings,
            }
            state.store.finish_run(run_id, final)
            state.store.save_samples(run_id, samples, started)
            if history_on:
                # Full content lives only here, in the separate archive -- never in the
                # metrics DB, whose stored_request/backend_response stay redacted.
                state.chat.history.enqueue({
                    **history_record, "created_at": utc_now(), "status": "completed", "usage": usage,
                    "timing": timing, "finish_reason": backend_payload.get("finish_reason"),
                    "resources": resources,
                    "output_text": "".join(output_parts) or None,
                    "reasoning_text": "".join(reasoning_parts) or None, "tool_calls": tool_calls,
                })
            state.chat.usage.record(project_id, connection_type,
                                    usage.get("input_tokens"), usage.get("output_tokens"))
            state.chat._deliver_to_client(client, run_id, target, handle, final)
            if not stream:
                backend_payload["dgpu_run_id"] = run_id
                self._send_json(HTTPStatus.OK, backend_payload)
        except (BrokenPipeError, ConnectionResetError) as exc:
            self._finish_gateway_failure(
                run_id, target, started, baseline, "cancelled", str(exc),
                history_record if history_on else None,
            )
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._finish_gateway_failure(
                run_id, target, started, baseline, "failed", detail,
                history_record if history_on else None,
            )
            if response_started:
                return
            raise AppError(f"Model server returned HTTP {exc.code}: {detail}", exc.code) from exc
        except Exception as exc:
            self._finish_gateway_failure(
                run_id, target, started, baseline, "failed", str(exc),
                history_record if history_on else None,
            )
            if response_started:
                return
            raise

    def _finish_gateway_failure(
        self,
        run_id: str,
        target: str,
        started: float,
        baseline: dict[str, Any] | None,
        status: str,
        error_text: str,
        history_record: dict[str, Any] | None = None,
    ) -> None:
        state = self.server.app_state
        try:
            samples = state.chat._sample_window(started, baseline)
            final = {
                "status": status,
                "usage": {},
                "timing": {"end_to_end_duration_ms": round((time.monotonic() - started) * 1000, 2)},
                "resources": ChatService._resource_summary(samples, target),
                "error_text": error_text,
            }
            state.store.finish_run(run_id, final)
            state.store.save_samples(run_id, samples, started)
            if history_record is not None and state.chat.history.enabled:
                state.chat.history.enqueue({
                    **history_record,
                    "created_at": utc_now(),
                    "status": status,
                    "usage": {},
                    "timing": final["timing"],
                    "resources": final["resources"],
                    "error_text": error_text,
                })
        except Exception:  # noqa: BLE001
            pass

    def _require_json_content_type(self) -> None:
        """State-changing control endpoints need a JSON content type, which forces a CORS preflight,
        so a web page cannot trigger them with a blind form POST."""
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            raise AppError("Content-Type must be application/json.", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)

    def _service_control(self) -> "ServiceControl":
        control = getattr(self.server.app_state, "control", None)
        if control is None:
            raise AppError("Service control is not available in this process.", HTTPStatus.NOT_IMPLEMENTED)
        return control

    def _settings_payload(self) -> dict[str, Any]:
        state = self.server.app_state
        return {"settings": service_settings.describe(state.config.policy), "config_path": str(state.config_path)}

    def _apply_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate, apply to the running config, and save into the config file's [policy] table."""
        state = self.server.app_state
        try:
            values = service_settings.validate(payload.get("values", payload))
        except ValueError as exc:
            raise AppError(str(exc)) from exc
        changed = {key: value for key, value in values.items() if getattr(state.config.policy, key) != value}
        if changed:
            try:
                service_settings.write_policy(state.config_path, changed)
            except (OSError, ValueError) as exc:
                raise AppError(f"Settings were not changed: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR) from exc
            for key, value in changed.items():
                setattr(state.config.policy, key, value)
        return {**self._settings_payload(), "changed": sorted(changed),
                "next_load": sorted(key for key in changed if service_settings.BY_KEY[key]["apply"] == "next_load")}

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
        self._write_common_headers()
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
        default=str(Path(__file__).with_name(default_config_name())),
        help="Path to the dual-GPU TOML configuration.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Web UI bind address. Defaults to localhost; use 0.0.0.0 explicitly for LAN access.",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Enable trusted-LAN access by binding the gateway to 0.0.0.0 (same as --host 0.0.0.0).",
    )
    parser.add_argument("--port", type=int, default=8090, help="Web UI port. Defaults to 8090.")
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help=(
            "Allow a browser app origin to call the API; repeat for multiple origins. "
            "Use '*' only on a trusted LAN."
        ),
    )
    parser.add_argument("--data-dir", default="./chat_runs", help="Directory for SQLite data and server logs.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the UI in the default browser.")
    parser.add_argument("--check", action="store_true", help="Validate configuration and storage, then exit.")
    parser.add_argument(
        "--replace-pid",
        type=int,
        default=None,
        help="Internal: wait for this process to exit before starting (used when the UI restarts the service).",
    )
    parser.add_argument(
        "--prune-older-than-days",
        type=int,
        default=None,
        help=(
            "One-shot maintenance that DELETES runs (and their resource samples) older than this many days. "
            "Refused unless --confirm-delete is also given; a verified backup is taken first. Nothing is "
            "deleted by default: the databases keep everything forever."
        ),
    )
    parser.add_argument("--confirm-delete", action="store_true",
                        help="Required for --prune-older-than-days, which permanently removes data.")
    parser.add_argument(
        "--skip-backup", action="store_true",
        help="Do not take the verified pre-start database backup (one per app/schema version, kept forever in <data-dir>/backups).",
    )
    parser.add_argument(
        "--exchange-dir", default=None, metavar="DIR",
        help=("Shared folder (for example on the NTFS partition both operating systems mount) where this install "
              "writes append-only run and telemetry files and imports the other OS's files. See docs/CROSS_OS_HISTORY.md."),
    )
    parser.add_argument(
        "--import-runs-from",
        action="append",
        default=[],
        metavar="CHATBOT_DB",
        help=("Before starting, import the inference run history from another install's chatbot.sqlite3 (read-only "
              "copy; runs already present are skipped; imported rows are tagged with the OS they came from)."),
    )
    parser.add_argument(
        "--import-from",
        action="append",
        default=[],
        metavar="TRAINING_DB",
        help=(
            "Before starting, merge the training telemetry (runs, steps, hardware samples, ...) from another "
            "install's training.sqlite3 into this data directory's, by run id. Repeatable. The other file is "
            "opened read-only and never modified; runs already present are left untouched, so it is safe to "
            "pass on every start. Use it to see Windows and Ubuntu runs in one dashboard."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.lan and args.host in LOOPBACK_HOSTS:
        args.host = "0.0.0.0"
    if args.replace_pid is not None and not wait_for_exit(args.replace_pid):
        print(f"WARNING: process {args.replace_pid} is still running; starting anyway.")
    config_path = Path(args.config).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (config_path.parent / data_dir).resolve()
    if args.prune_older_than_days is not None and not args.confirm_delete:
        print("Refusing --prune-older-than-days without --confirm-delete: it permanently deletes run history.")
        return 2
    if args.prune_older_than_days is not None:  # its own timestamped copy, whatever --skip-backup says
        safety = datasafe.backup_sqlite(data_dir / "chatbot.sqlite3", data_dir / "backups", "before-prune-" + time.strftime("%Y%m%dT%H%M%S"))
        if safety.error:
            print(f"ERROR: could not back up before pruning: {safety.error}")
            return 3
        print(f"Backed up before pruning -> {safety.path}", flush=True)
    backups: list[dict[str, Any]] = []
    if not args.skip_backup:
        from dual_gpu_setup.training.db import SCHEMA_VERSION

        label = f"pre-v{APP_VERSION}-s{SCHEMA_VERSION}"
        for name in ("chatbot.sqlite3", "chat_history.sqlite3", "project_usage.sqlite3", "training.sqlite3"):
            result = datasafe.backup_sqlite(data_dir / name, data_dir / "backups", label)
            backups.append(result.as_dict())
            if result.error:
                print(f"ERROR: backup of {name} failed: {result.error}\n"
                      "Not starting, so nothing is migrated without a safety copy. Free disk space or pass --skip-backup.")
                return 3
            if result.path and not result.skipped:
                print(f"Backed up {name} -> {result.path} (verified)", flush=True)
    if args.import_runs_from:
        from dual_gpu_setup.exchange import import_runs_db, import_usage_db

        before = datasafe.table_counts(data_dir / "chatbot.sqlite3")
        state_for_schema = RunStore(data_dir / "chatbot.sqlite3")  # create/upgrade the destination schema first
        for source in args.import_runs_from:
            result = import_runs_db(state_for_schema.path, Path(source).expanduser(), default_os="Windows")
            print(f"Imported runs from {source}: {json.dumps(result)}", flush=True)
            if not result["ok"]:
                print("WARNING: run history was not imported (see above); the service starts normally.")
            usage_source = Path(source).expanduser().with_name("project_usage.sqlite3")
            if usage_source.is_file() and (data_dir / "project_usage.sqlite3").is_file():
                print(f"Imported usage from {usage_source}: {json.dumps(import_usage_db(data_dir / 'project_usage.sqlite3', usage_source))}",
                      flush=True)
        problems = datasafe.verify_counts_not_lower(before, datasafe.table_counts(data_dir / "chatbot.sqlite3"))
        if problems:
            print(f"ERROR: row counts dropped during import: {problems}")
            return 4
    if args.import_from:
        from dual_gpu_setup.training.merge import merge_many

        for report in merge_many(data_dir / "training.sqlite3", args.import_from, all_telemetry=True,
                                 dest_os=os_info().get("system") or None):
            print(report.summary(), flush=True)
            if not report.ok:
                print("WARNING: some runs were not imported (see above); the service starts normally.")
    setup_service_logging(data_dir)
    state = ApplicationState(config_path, data_dir)
    state.backups = backups
    if args.exchange_dir:
        from dual_gpu_setup.exchange import ExchangeSync
        from dual_gpu_setup.training.merge import merge_training_db

        state.exchange = ExchangeSync(Path(args.exchange_dir).expanduser(), state.host_os, data_dir, state.store.path,
                                      data_dir / "training.sqlite3", merge_training=merge_training_db)
    if args.prune_older_than_days is not None:
        result = state.store.prune_older_than(args.prune_older_than_days)
        print(
            f"Pruned {result['deleted_runs']} run(s) and {result['deleted_deployments']} "
            f"deployment(s) older than {result['cutoff']}."
        )
    if args.check:
        print(json.dumps({"status": "ok", "config": str(config_path), "data_dir": str(data_dir)}, indent=2))
        state.manager.shutdown()
        state.history.close()
        state.usage.close()
        return 0

    if args.host not in LOOPBACK_HOSTS:
        print("WARNING: This app has no authentication. Binding outside localhost exposes prompts and model controls.")
    if args.cors_origin:
        print(f"  CORS origins: {', '.join(args.cors_origin)}")
    server = ChatbotHTTPServer((args.host, args.port), state, args.cors_origin)
    state.control = ServiceControl(state, args)
    state.control.server = server
    local_url = f"http://127.0.0.1:{args.port}"
    state.benchmarks.server_url = local_url
    is_lan_enabled = args.host not in LOOPBACK_HOSTS
    lan_urls = list(state.network["advertised_urls"]) if is_lan_enabled else []
    print("Dual GPU Studio is running")
    print(f"  Local: {local_url}")
    if lan_urls:
        for lan_url in lan_urls:
            print(f"  Local network: {lan_url}")
        print("  Open a Local network URL on another device.")
    elif is_lan_enabled:
        print(f"  Listening on: http://{args.host}:{args.port}")
        print("  No private IPv4 address was detected; check the host's network configuration.")
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
    state.start_telemetry()
    state.control.restore_after_restart(allowed=args.replace_pid is not None)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        state.stop_telemetry()
        state.manager.shutdown()
        state.history.close()  # flush any queued conversation records
        state.usage.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
