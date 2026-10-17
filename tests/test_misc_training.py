from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_gpu_setup.training.hardware import HardwareSampler  # noqa: E402
from dual_gpu_setup.training.profiling import find_traces, rocprofv3_command  # noqa: E402
from dual_gpu_setup.training.tracker import TrainingTracker  # noqa: E402
from test_training import build_sysfs  # noqa: E402


def test_rocprofv3_command_wraps_the_launch_and_finds_traces(tmp_path):
    command = rocprofv3_command(["python", "train.py", "--max-steps", "20"], tmp_path / "prof", executable="/opt/rocm/bin/rocprofv3")
    assert command[0] == "/opt/rocm/bin/rocprofv3" and "--kernel-trace" in command
    assert command[command.index("-d") + 1] == str(tmp_path / "prof")
    assert command[command.index("--") + 1:] == ["python", "train.py", "--max-steps", "20"]  # tool flags end at `--`
    (tmp_path / "prof" / "node").mkdir(parents=True)
    old = tmp_path / "prof" / "node" / "old.pftrace"
    old.write_bytes(b"1")
    time.sleep(0.02)
    new = tmp_path / "prof" / "node" / "new.pftrace"
    new.write_bytes(b"2")
    assert find_traces(tmp_path / "prof") == [new, old]


def test_sampler_thread_survives_an_unexpected_read_error(tmp_path, monkeypatch):
    sampler = HardwareSampler(sysfs_root=build_sysfs(tmp_path), platform_name="linux", interval_s=0.02)
    original, calls = sampler.sample_once, {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("sensor vanished")
        return original()

    monkeypatch.setattr(sampler, "sample_once", flaky)
    sampler.start()
    deadline = time.time() + 5
    while time.time() < deadline and calls["n"] < 6:
        time.sleep(0.02)
    sampler.stop()
    snapshot = sampler.snapshot()["sampler"]
    assert calls["n"] >= 6 and snapshot["read_errors"] == 1 and "sensor vanished" in snapshot["last_error"]


def test_tracker_hot_path_is_cheap_and_never_touches_the_network(tmp_path):
    """The per-microbatch and per-step calls run inside the training loop; they must be near-free."""
    tracker = TrainingTracker("http://127.0.0.1:9", "bench", log_every=50, enabled=True, capture_environment=False,
                              spool_dir=tmp_path, warmup_windows=0)  # never started: no thread, no HTTP
    n = 20000
    started = time.perf_counter()
    for step in range(1, n + 1):
        tracker.microbatch_counts(4096, 3000, 1200)
        if step % 4 == 0:
            tracker.step_completed(step // 4)
    per_call_us = (time.perf_counter() - started) / (n + n // 4) * 1e6
    assert per_call_us < 100, f"{per_call_us:.1f} us per call"  # generous; measured well under 20 us
    assert tracker.enqueue_ms_total / max(1, tracker._windows_sent) < 5  # window conversion cost per flush
