from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dual_gpu_setup.config import AppConfig, LaneConfig, ModelConfig


class TaskFailure(RuntimeError):
    """Raised when a configured eval task fails."""


def _context(
    config: AppConfig,
    lane: LaneConfig,
    model: ModelConfig,
    model_size_gb: float,
    base_url: str,
    suite_id: str,
    run_dir: Path,
    actual_ctx: int,
    device: str,
) -> dict[str, str]:
    return {
        "project_name": config.project.name,
        "suite_id": suite_id,
        "base_url": base_url,
        "host": lane.host or config.project.host,
        "port": str(lane.port),
        "lane_key": lane.key,
        "lane_display": lane.display,
        "backend": lane.backend,
        "device": device,
        "model_name": model.name,
        "model_path": model.path,
        "model_size_gb": f"{model_size_gb:.2f}",
        "actual_ctx": str(actual_ctx),
        "run_dir": str(run_dir),
        "config_dir": str(config.base_dir),
    }


def _render(value: str, context: dict[str, str]) -> str:
    return value.format_map(context)


def run_tasks_for_model(
    config: AppConfig,
    lane: LaneConfig,
    model: ModelConfig,
    model_size_gb: float,
    base_url: str,
    suite_id: str,
    run_dir: Path,
    actual_ctx: int,
    device: str,
) -> None:
    context = _context(config, lane, model, model_size_gb, base_url, suite_id, run_dir, actual_ctx, device)
    for task in config.tasks:
        log_path = run_dir / f"task-{task.name}.log"
        task_cwd = Path(_render(task.cwd, context)).resolve() if task.cwd else config.base_dir
        env = os.environ.copy()
        env.update({key: _render(value, context) for key, value in task.env.items()})
        env.update(
            {
                "DGPU_SUITE_ID": suite_id,
                "DGPU_BASE_URL": base_url,
                "DGPU_MODEL_NAME": model.name,
                "DGPU_LANE_KEY": lane.key,
                "DGPU_LANE_DISPLAY": lane.display,
                "DGPU_BACKEND": lane.backend,
                "DGPU_DEVICE": device,
                "DGPU_ACTUAL_CTX": str(actual_ctx),
                "DGPU_RUN_DIR": str(run_dir),
            }
        )

        if isinstance(task.command, str):
            command = _render(task.command, context)
        else:
            command = [_render(part, context) for part in task.command]
            if task.shell:
                command = subprocess.list2cmdline(command)

        with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"$ {command}\n\n")
            log_file.flush()
            result = subprocess.run(
                command,
                cwd=str(task_cwd),
                env=env,
                shell=task.shell,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            message = f"Task '{task.name}' failed for model '{model.name}' with exit code {result.returncode}"
            if task.continue_on_error:
                print(f"WARNING: {message}")
                continue
            raise TaskFailure(message)
