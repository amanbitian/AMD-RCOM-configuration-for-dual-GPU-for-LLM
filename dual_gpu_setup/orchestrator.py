from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dual_gpu_setup.config import AppConfig, LaneConfig, ModelConfig
from dual_gpu_setup.lmstudio import model_size_gb, resolve_model_path
from dual_gpu_setup.server import LlamaServerProcess, warn_if_spilling
from dual_gpu_setup.tasks import TaskFailure, run_tasks_for_model


@dataclass(slots=True)
class PreparedModel:
    source: ModelConfig
    name: str
    path: Path
    size_gb: float


class SharedQueue:
    def __init__(self, models: list[PreparedModel]):
        self._models = list(sorted(models, key=lambda item: item.size_gb))
        self._lock = threading.Lock()

    def pull_largest(self, max_gb: float, lane_key: str) -> PreparedModel | None:
        with self._lock:
            for index in range(len(self._models) - 1, -1, -1):
                model = self._models[index]
                if model.source.pin_lane and model.source.pin_lane != lane_key:
                    continue
                if model.size_gb <= max_gb:
                    return self._models.pop(index)
        return None

    def pull_smallest(self, max_gb: float, lane_key: str) -> PreparedModel | None:
        with self._lock:
            for index, model in enumerate(self._models):
                if model.source.pin_lane and model.source.pin_lane != lane_key:
                    continue
                if model.size_gb <= max_gb:
                    return self._models.pop(index)
            return None

    @property
    def remaining(self) -> list[PreparedModel]:
        with self._lock:
            return list(self._models)


def lane_capacity_gb(config: AppConfig, lane: LaneConfig) -> float:
    return lane.vram_gb * config.policy.vram_safety_fraction


def ctx_for(config: AppConfig, model_size: float, lane_vram_gb: float) -> int:
    headroom = lane_vram_gb * config.policy.vram_safety_fraction - model_size
    if headroom >= config.policy.ctx_max_headroom_gb:
        return config.policy.ctx_max
    if headroom >= config.policy.ctx_mid_headroom_gb:
        return config.policy.ctx_mid
    return config.policy.ctx_min


def prepare_models(config: AppConfig) -> tuple[list[PreparedModel], list[PreparedModel]]:
    single_gpu: list[PreparedModel] = []
    multi_gpu: list[PreparedModel] = []
    for model in config.models:
        prepared = PreparedModel(
            source=model,
            name=model.name,
            path=resolve_model_path(config, model),
            size_gb=model_size_gb(config, model),
        )
        if model.multi_gpu:
            multi_gpu.append(prepared)
        else:
            single_gpu.append(prepared)
    return single_gpu, multi_gpu


def suite_id(config: AppConfig) -> str:
    return f"{config.project.suite_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


class DualGPUOrchestrator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.summary_lock = threading.Lock()

    def run(self, dry_run: bool = False) -> dict:
        single_gpu, multi_gpu = prepare_models(self.config)
        lanes = sorted(self.config.lanes, key=lambda item: item.vram_gb, reverse=True)
        big_lane, small_lane = lanes[0], lanes[1]
        queue = SharedQueue([model for model in single_gpu if self._fits_any_lane(model)])
        skipped = [model for model in single_gpu if not self._fits_any_lane(model)]
        summary = {
            "suite_id": suite_id(self.config),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "project": self.config.project.name,
            "dry_run": dry_run,
            "completed": [],
            "failed_loads": [],
            "task_failures": [],
            "skipped": [{"model": model.name, "reason": "does not fit on either single GPU lane"} for model in skipped],
            "multi_gpu": [model.name for model in multi_gpu],
        }
        run_dir = Path(self.config.base_dir / self.config.project.log_dir / summary["suite_id"]).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_summary(run_dir, summary)

        if dry_run:
            summary["planned_single_gpu_models"] = [model.name for model in queue.remaining]
            self._write_summary(run_dir, summary)
            return summary

        if self.config.project.cleanup_ports_on_start:
            for lane in lanes:
                LlamaServerProcess.kill_orphan_on_port(lane.port)

        workers = [
            threading.Thread(
                target=self._lane_worker,
                kwargs={
                    "lane": big_lane,
                    "queue": queue,
                    "pull_largest": True,
                    "run_dir": run_dir,
                    "summary": summary,
                },
                name=f"lane-{big_lane.key}",
            ),
            threading.Thread(
                target=self._lane_worker,
                kwargs={
                    "lane": small_lane,
                    "queue": queue,
                    "pull_largest": False,
                    "run_dir": run_dir,
                    "summary": summary,
                },
                name=f"lane-{small_lane.key}",
            ),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        for model in multi_gpu:
            self._run_multi_gpu_model(model, lanes, run_dir, summary)

        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_summary(run_dir, summary)
        return summary

    def _fits_any_lane(self, model: PreparedModel) -> bool:
        for lane in self.config.lanes:
            if model.source.pin_lane and model.source.pin_lane != lane.key:
                continue
            if model.size_gb <= lane_capacity_gb(self.config, lane):
                return True
        return False

    def _lane_worker(
        self,
        lane: LaneConfig,
        queue: SharedQueue,
        pull_largest: bool,
        run_dir: Path,
        summary: dict,
    ) -> None:
        max_gb = lane_capacity_gb(self.config, lane)
        while True:
            model = queue.pull_largest(max_gb, lane.key) if pull_largest else queue.pull_smallest(max_gb, lane.key)
            if model is None:
                return
            self._run_model(model, lane, run_dir, summary)

    def _run_model(self, model: PreparedModel, lane: LaneConfig, run_dir: Path, summary: dict) -> None:
        ctx = model.source.ctx_size or ctx_for(self.config, model.size_gb, lane.vram_gb)
        print(f"[{lane.key}] loading {model.name} ({model.size_gb:.2f} GB, ctx={ctx})")
        process = LlamaServerProcess.for_single_lane(
            config=self.config,
            lane=lane,
            model=model.source,
            model_path=model.path,
            ctx_size=ctx,
            run_dir=run_dir,
        )
        load_started = time.time()
        try:
            process.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[{lane.key}] failed to load {model.name}: {exc}")
            with self.summary_lock:
                summary["failed_loads"].append({"model": model.name, "lane": lane.key, "error": str(exc)})
                self._write_summary(run_dir, summary)
            return

        effective_ctx = process.actual_ctx()
        placement = warn_if_spilling(process.pid, model.name, lane.display)
        metadata = {
            "model": model.name,
            "lane": lane.key,
            "display": lane.display,
            "device": process.device,
            "ctx_requested": ctx,
            "ctx_actual": effective_ctx,
            "size_gb": model.size_gb,
            "load_seconds": round(time.time() - load_started, 1),
            "vram": {
                "dedicated_gb": round(placement[0], 2) if placement else None,
                "shared_gb": round(placement[1], 2) if placement else None,
            },
        }
        model_dir = run_dir / model.name.replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "load.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            run_tasks_for_model(
                config=self.config,
                lane=lane,
                model=model.source,
                model_size_gb=model.size_gb,
                base_url=process.base_url,
                suite_id=summary["suite_id"],
                run_dir=model_dir,
                actual_ctx=effective_ctx,
                device=process.device,
            )
            with self.summary_lock:
                summary["completed"].append({"model": model.name, "lane": lane.key, "mode": "single"})
                self._write_summary(run_dir, summary)
        except TaskFailure as exc:
            with self.summary_lock:
                summary["task_failures"].append({"model": model.name, "lane": lane.key, "error": str(exc)})
                self._write_summary(run_dir, summary)
        finally:
            process.stop()

    def _run_multi_gpu_model(self, model: PreparedModel, lanes: list[LaneConfig], run_dir: Path, summary: dict) -> None:
        ctx = model.source.ctx_size or ctx_for(self.config, model.size_gb, sum(lane.vram_gb for lane in lanes))
        print(f"[multi] loading {model.name} across {', '.join(lane.key for lane in lanes)} (ctx={ctx})")
        process = LlamaServerProcess.for_multi_gpu(
            config=self.config,
            lanes=lanes,
            model=model.source,
            model_path=model.path,
            ctx_size=ctx,
            run_dir=run_dir,
        )
        try:
            process.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[multi] failed to load {model.name}: {exc}")
            with self.summary_lock:
                summary["failed_loads"].append({"model": model.name, "lane": "multi", "error": str(exc)})
                self._write_summary(run_dir, summary)
            return

        effective_ctx = process.actual_ctx()
        placement = warn_if_spilling(process.pid, model.name, "multi")
        model_dir = run_dir / model.name.replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model": model.name,
            "lane": "multi",
            "device": process.device,
            "ctx_requested": ctx,
            "ctx_actual": effective_ctx,
            "size_gb": model.size_gb,
            "vram": {
                "dedicated_gb": round(placement[0], 2) if placement else None,
                "shared_gb": round(placement[1], 2) if placement else None,
            },
        }
        (model_dir / "load.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            run_tasks_for_model(
                config=self.config,
                lane=lanes[0],
                model=model.source,
                model_size_gb=model.size_gb,
                base_url=process.base_url,
                suite_id=summary["suite_id"],
                run_dir=model_dir,
                actual_ctx=effective_ctx,
                device=process.device,
            )
            with self.summary_lock:
                summary["completed"].append({"model": model.name, "lane": "multi", "mode": "multi"})
                self._write_summary(run_dir, summary)
        except TaskFailure as exc:
            with self.summary_lock:
                summary["task_failures"].append({"model": model.name, "lane": "multi", "error": str(exc)})
                self._write_summary(run_dir, summary)
        finally:
            process.stop()

    def _write_summary(self, run_dir: Path, summary: dict) -> None:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
