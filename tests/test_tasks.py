from __future__ import annotations

import subprocess
from pathlib import Path

from dual_gpu_setup.config import load_config
from dual_gpu_setup.tasks import run_tasks_for_model


def test_relative_task_cwd_is_resolved_from_the_config(tmp_path: Path, monkeypatch):
    work = tmp_path / "relative-work"
    work.mkdir()
    config_path = tmp_path / "tasks.toml"
    config_path.write_text(
        """
[[lanes]]
key = "a"
display = "A"
vram_gb = 16
port = 19001

[[lanes]]
key = "b"
display = "B"
vram_gb = 16
port = 19002

[[models]]
name = "model"
path = "model.gguf"

[[tasks]]
name = "smoke"
cwd = "./relative-work"
command = ["python3", "smoke.py"]
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("dual_gpu_setup.tasks.os_info", lambda: {"system": "Windows"})
    monkeypatch.setattr("dual_gpu_setup.tasks.os_log_line", lambda: '{"system": "Windows"}')
    monkeypatch.setattr("dual_gpu_setup.tasks.subprocess.run", fake_run)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    run_tasks_for_model(
        config, config.lanes[0], config.models[0], 1.0, "http://127.0.0.1:8081",
        "suite", run_dir, 4096, "ROCm0",
    )

    assert calls[0][1]["cwd"] == str(work.resolve())
    assert calls[0][0] == ["python3", "smoke.py"]
