"""Training HTTP service: routing and orchestration over the stores, leases, evaluation queue and
benchmark runner. app.py only forwards requests here, so every endpoint is unit-testable without a
socket.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dual_gpu_setup.config import AppConfig
from dual_gpu_setup.host_info import os_info
from dual_gpu_setup.scheduling.leases import LeaseError, LeaseStore

from . import analysis, memory_forecast, selection
from .benchmarks import BenchmarkRunner
from .evaluation import EvalQueue
from .hardware import HardwareSampler, map_torch_devices
from .store import TrainingError, TrainingStore


@dataclass(slots=True)
class FileResponse:
    path: Path
    filename: str
    content_type: str = "application/octet-stream"


class TrainingService:
    def __init__(
        self,
        config: AppConfig,
        store: TrainingStore,
        leases: LeaseStore,
        eval_queue: EvalQueue,
        benchmarks: BenchmarkRunner,
        hardware: HardwareSampler,
        artifact_root: Path,
        busy_lanes: Callable[[], set[str]],
        unload_inference: Callable[[], Any],
    ):
        self.config = config
        self.store = store
        self.leases = leases
        self.eval_queue = eval_queue
        self.benchmarks = benchmarks
        self.hardware = hardware
        self.artifact_root = artifact_root
        self.busy_lanes = busy_lanes
        self.unload_inference = unload_inference
        self.settings = config.evaluation
        self._lanes = {lane.key: lane for lane in config.lanes}

    # ------------------------------------------------------------- run lifecycle

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a run and atomically lease its lanes. Inference loaded on those lanes blocks the
        run (409) unless the caller passes unload_inference=true."""
        lanes = [str(lane) for lane in payload.get("lanes") or []]
        unknown = [lane for lane in lanes if lane not in self._lanes]
        if unknown:
            raise TrainingError(f"Unknown lane(s): {', '.join(unknown)}. Configured: {', '.join(sorted(self._lanes))}.")
        run_id = str(payload.get("id") or uuid.uuid4().hex)
        lease = None
        existing = None
        try:
            existing = self.store.get_run(run_id)
        except TrainingError:
            pass
        if lanes and existing is None:
            busy = sorted(set(lanes) & self.busy_lanes())
            if busy:
                if not payload.get("unload_inference"):
                    raise TrainingError(
                        f"Inference is loaded on lane(s) {', '.join(busy)}. Unload it first or pass unload_inference=true.", 409)
                self.unload_inference()
            try:
                lease = self.leases.acquire(
                    f"training:{run_id}", lanes, kind="training", ttl_s=float(payload.get("lease_ttl_s") or 300.0),
                    pid=int(payload["pid"]) if payload.get("pid") else None)
            except LeaseError as exc:
                raise TrainingError(str(exc), exc.status) from exc
        env = dict(payload.get("env") or {})
        server_host_os = os_info()
        training_host_os = env.get("os") or payload.get("host_os") or server_host_os
        if not isinstance(training_host_os, dict):
            training_host_os = server_host_os
        env["server_host_os"] = server_host_os
        env["server_hardware"] = self.hardware.hardware_environment()
        torch_devices = (env.get("torch") or {}).get("devices") or []
        if torch_devices:
            env["gpu_map"] = map_torch_devices(torch_devices, self.hardware.gpus)
        try:
            run = self.store.create_run({**payload, "id": run_id, "env": env, "host_os": training_host_os})
        except Exception:
            if lease:
                self.leases.release_holder(f"training:{run_id}")
            raise
        return {**run, "lease": lease, "artifact_dir": str(self.artifact_root / run_id)}

    def record_metrics(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.store.record_metrics(run_id, payload)
        if self.leases.renew_holder(f"training:{run_id}") == "lost":
            self.store.add_event(run_id, {
                "eid": "lease_lost", "kind": "lease_lost", "severity": "warning",
                "payload": {"detail": "Lease expired and its lanes were taken by another holder."}})
        return result

    def add_checkpoint(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.store.add_checkpoint(run_id, payload)
        if result["valid"] and self.settings.enabled:
            valid = [c for c in self.store.checkpoints(run_id) if c["valid"]]
            every = self.settings.designate_every
            designated = bool(every) and len(valid) % every == 0
            if designated:
                self.store.set_checkpoint_flags(run_id, int(payload["step"]), designated=True)
            self.eval_queue.submit(run_id, int(payload["step"]), str(payload["path"]), designated=designated)
        return result

    def finish_run(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.store.finish_run(run_id, payload)
        self.leases.release_holder(f"training:{run_id}")
        lanes = result.get("lanes") or []
        gpu = next((g.pci_bus for g in self.hardware.gpus if g.lane in lanes), None) if lanes else None
        memory_forecast.record_run_profile(self.store, result, gpu or (lanes[0] if lanes else None))
        if result["status"] == "completed" and self.settings.enabled:
            valid = [c for c in self.store.checkpoints(run_id) if c["valid"]]
            if valid:
                newest = max(valid, key=lambda c: c["step"])
                self.eval_queue.submit(run_id, newest["step"], newest["path"], final=True)
        return result

    def cleanup_trial(self, run_id: str) -> None:
        """A benchmark trial's process has ended: close its run and free the lease if it never did."""
        try:
            if self.store.get_run(run_id)["recorded_status"] == "running":
                self.store.finish_run(run_id, {"status": "failed", "error": "process ended without tracker.finish()"})
        except TrainingError:
            pass
        self.leases.release_holder(f"training:{run_id}")

    # ------------------------------------------------------------------ routing

    def _alerts_context(self, run_id: str) -> list[dict[str, Any]]:
        return analysis.run_alerts(self.store, run_id)

    def _queue_alert(self) -> list[dict[str, Any]]:
        """The evaluation queue is shared by every run, so it is reported once, not once per run."""
        stats = self.eval_queue.stats()
        if stats["capacity"] and stats["depth"] >= 0.8 * stats["capacity"]:
            return [{"run_id": None, "severity": "warning", "kind": "eval_queue_saturated", "step": None,
                     "message": f"Evaluation queue at {stats['depth']}/{stats['capacity']}."}]
        return []

    def _vram_totals(self) -> dict[str, int]:
        return {g.pci_bus: g.vram_total_bytes for g in self.hardware.gpus if g.vram_total_bytes}

    def peak_tflops(self, run: dict[str, Any]) -> float | None:
        values = [self.settings.peak_tflops.get(lane) for lane in run.get("lanes") or [] if self.settings.peak_tflops.get(lane)]
        return min(values) if values else None

    def get(self, path: str, query: dict[str, list[str]]) -> dict[str, Any] | FileResponse:
        def q(name: str, default: str | None = None) -> str | None:
            return (query.get(name) or [default])[0]

        if path == "/api/training/runs":
            return self.store.list_runs(int(q("limit", "100")), int(q("offset", "0")))
        if path == "/api/training/compare":
            ids = [i for i in (q("ids", "") or "").split(",") if i]
            if len(ids) < 2:
                raise TrainingError("ids must list at least two run ids.")
            budget = q("budget")
            return analysis.compare_runs(self.store, ids, float(budget) if budget else None)
        if path == "/api/training/scaling":
            baselines = [i for i in (q("baselines", "") or "").split(",") if i]
            if not q("ddp") or not baselines:
                raise TrainingError("ddp and baselines are required.")
            return analysis.scaling_efficiency(self.store, q("ddp") or "", baselines)
        if path == "/api/training/alerts":
            runs = [r for r in self.store.list_runs(200)["runs"] if r["status"] in {"running", "stalled", "lost"}]
            return {"alerts": [a for r in runs for a in self._alerts_context(r["id"])] + self._queue_alert()}
        if path == "/api/training/eval-queue":
            latest = max([r["last_step"] for r in self.store.list_runs(50)["runs"]] or [0])
            return {"stats": self.eval_queue.stats(latest), "jobs": self.eval_queue.jobs(), "settings": {
                "enabled": self.settings.enabled, "lane": self.settings.lane, "capacity": self.settings.queue_capacity,
                "primary_metric": self.settings.primary_metric}}
        if path == "/api/training/benchmarks":
            return {"benchmarks": self.benchmarks.list()}
        if path == "/api/training/profiles":
            return {"profiles": self.store.profiles()}
        if path == "/api/training/notes":
            return {"notes": self.store.notes()}
        match = re.fullmatch(r"/api/training/benchmarks/([\w-]+)", path)
        if match:
            try:
                return self.benchmarks.get(match.group(1))
            except KeyError as exc:
                raise TrainingError("Benchmark not found.", 404) from exc
        match = re.fullmatch(r"/api/training/runs/([\w-]+)(?:/(\w[\w-]*))?(?:/(\d+))?", path)
        if not match:
            raise TrainingError("Not found.", 404)
        run_id, sub, number = match.group(1), match.group(2), match.group(3)
        store = self.store
        if sub is None:
            return {"run": store.get_run(run_id), "events": store.events(run_id), "checkpoints": store.checkpoints(run_id),
                    "hardware_summary": store.hardware_summary(run_id), "artifacts": store.artifacts(run_id),
                    "notes": store.notes(run_id), "alerts": self._alerts_context(run_id)}
        if sub == "series":
            max_points = int(q("max_points", "600"))
            return {"steps": store.series(run_id, max_points), "hardware": store.hardware_series(run_id, max_points)}
        if sub == "analysis":
            return analysis.analyse_run(store, run_id, self.peak_tflops(store.get_run(run_id)), self._vram_totals())
        if sub == "alerts":
            return {"alerts": self._alerts_context(run_id)}
        if sub == "evals":
            return {
                "selection": selection.select_best(store, run_id, self.settings, q("eval_set", "default") or "default"),
                "regressions": selection.detect_regressions(store, run_id, self.settings, q("eval_set", "default") or "default"),
            }
        if sub == "resume-report":
            return analysis.resume_report(store, run_id)
        if sub == "hard-examples":
            return {"examples": store.hard_examples(run_id)}
        if sub == "ranks":
            return analysis.straggler_analysis(store, run_id, self._vram_totals())
        if sub == "artifacts" and number is not None:
            artifact = store.get_artifact(run_id, int(number))
            return self._file(artifact)
        if sub == "artifacts":
            return {"artifacts": store.artifacts(run_id)}
        raise TrainingError("Not found.", 404)

    def _file(self, artifact: dict[str, Any]) -> FileResponse:
        path = Path(artifact["path"])
        if not artifact["downloadable"] or not self._under_root(path) or not path.is_file():
            raise TrainingError("This artifact is not downloadable (it lives outside the artifact folder).", 404)
        return FileResponse(path, path.name)

    def _under_root(self, path: Path) -> bool:
        try:
            return path.resolve().is_relative_to(self.artifact_root.resolve())
        except OSError:
            return False

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        store = self.store
        if path == "/api/training/runs":
            return self.create_run(payload)
        if path == "/api/training/eval-queue":
            run_id, step, ckpt = str(payload.get("run_id") or ""), payload.get("step"), str(payload.get("path") or "")
            if not run_id or step is None or not ckpt:
                raise TrainingError("run_id, step and path are required.")
            store.get_run(run_id)
            return self.eval_queue.submit(run_id, int(step), ckpt, final=bool(payload.get("final")),
                                          best=bool(payload.get("best")), designated=bool(payload.get("designated")))
        if path == "/api/training/forecast":
            lane = self._lanes.get(str(payload.get("lane") or ""))
            if lane is None:
                raise TrainingError("lane must be one of: " + ", ".join(sorted(self._lanes)))
            return memory_forecast.forecast(
                store, payload.get("config"), {"key": lane.key, "vram_gb": lane.vram_gb, "pci_bus": lane.pci_bus},
                safety_fraction=self.config.policy.vram_safety_fraction, flash_attention=bool(payload.get("flash_attention", True)))
        if path == "/api/training/benchmarks":
            try:
                benchmark_id = self.benchmarks.create(payload)
            except ValueError as exc:
                raise TrainingError(str(exc)) from exc
            self.benchmarks.start(benchmark_id)
            return {"id": benchmark_id}
        if path == "/api/training/notes":
            return store.add_note(None, payload)
        match = re.fullmatch(r"/api/training/benchmarks/([\w-]+)/cancel", path)
        if match:
            self.benchmarks.cancel(match.group(1))
            return {"ok": True}
        match = re.fullmatch(r"/api/training/runs/([\w-]+)/checkpoints/(\d+)/(designate|promote|demote)", path)
        if match:
            run_id, step, action = match.group(1), int(match.group(2)), match.group(3)
            if action == "promote":
                store.set_checkpoint_flags(run_id, step, best=True)
            elif action == "demote":
                store.set_checkpoint_flags(run_id, step, best=False)
            else:
                store.set_checkpoint_flags(run_id, step, designated=True)
            return {"ok": True}
        match = re.fullmatch(r"/api/training/runs/([\w-]+)/(\w+)", path)
        if not match:
            raise TrainingError("Not found.", 404)
        run_id, action = match.group(1), match.group(2)
        if action == "metrics":
            return self.record_metrics(run_id, payload)
        if action == "events":
            return store.add_event(run_id, payload)
        if action == "checkpoints":
            return self.add_checkpoint(run_id, payload)
        if action == "finish":
            return self.finish_run(run_id, payload)
        if action == "evals":
            return store.record_eval(run_id, payload)
        if action == "notes":
            return store.add_note(run_id, payload)
        if action == "examples":
            step = payload.get("step")
            if step is None or not isinstance(payload.get("items"), list):
                raise TrainingError("step and items are required.")
            return {"stored": store.add_example_stats(run_id, int(step), payload["items"])}
        if action == "select":
            return selection.select_best(store, run_id, self.settings, str(payload.get("eval_set") or "default"),
                                         apply=bool(payload.get("apply")))
        if action == "artifacts":
            path_value = str(payload.get("path") or "")
            if not path_value or not payload.get("kind"):
                raise TrainingError("kind and path are required.")
            candidate = Path(path_value)
            downloadable = self._under_root(candidate) and candidate.is_file()
            return store.add_artifact(run_id, str(payload["kind"]), path_value, size_bytes=payload.get("size_bytes"),
                                      downloadable=downloadable, meta=payload.get("meta"))
        raise TrainingError("Not found.", 404)
