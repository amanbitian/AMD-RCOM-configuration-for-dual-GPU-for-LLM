from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_gpu_setup.training import analysis, ddp  # noqa: E402
from dual_gpu_setup.training.amd_smi import AmdSmiReader  # noqa: E402
from dual_gpu_setup.training.hardware import (  # noqa: E402
    GpuDevice, HardwareSampler, map_torch_devices, sample_gpu, torch_pci_address,
)
from dual_gpu_setup.training.environment import rocm_installation  # noqa: E402
from dual_gpu_setup.training.tracker import TrainingTracker, resolve_run_id  # noqa: E402
from test_training import build_sysfs  # noqa: E402


def make_tracker(base, run_id, rank, world, tmp_path, **kw):
    return TrainingTracker(base, "ddp-run", run_id=run_id, rank=rank, world_size=world, log_every=2, warmup_windows=0,
                           capture_environment=False, flush_s=0.05, spool_dir=tmp_path / "spool", pci_bus=f"0000:0{rank + 3}:00.0", **kw)


def steps(tracker, n=4, tokens=(300, 120, 100)):
    for step in range(1, n + 1):
        tracker.microbatch_counts(*tokens)
        time.sleep(0.01 * (1 + tracker.rank))  # rank 1 is the slower card
        tracker.step_completed(step)


# --------------------------------------------------------------------- DDP


def test_only_rank0_owns_the_run_but_every_rank_reports_its_own_rows(live_app, tmp_path):
    live = live_app()
    r0 = make_tracker(live.base, "ddprun", 0, 2, tmp_path).start()
    r1 = make_tracker(live.base, "ddprun", 1, 2, tmp_path).start()
    time.sleep(0.3)
    steps(r0)
    steps(r1)
    r1.finish("completed")  # a non-primary rank must NOT finish the run
    time.sleep(0.3)
    assert live.state.training.get_run("ddprun")["recorded_status"] == "running"
    r0.finish("completed")
    run = live.state.training.get_run("ddprun")
    assert run["recorded_status"] == "completed" and run["world_size"] == 2
    assert run["tokens"]["nonpad"] == 4 * 120  # single-process view (no torch.distributed here): rank 0's counts
    rows = live.state.training.rank_steps("ddprun")
    assert {(r["rank"], r["pci_bus"]) for r in rows} == {(0, "0000:03:00.0"), (1, "0000:04:00.0")}
    by_rank = {r["rank"]: r["step_time_ms"] for r in rows if r["step"] == 2}
    assert by_rank[1] > by_rank[0]
    assert analysis.straggler_analysis(live.state.training, "ddprun")["ranks"][1]["slowest_windows"] >= 1


def test_global_token_counts_are_all_reduced_once_per_window(monkeypatch, live_app, tmp_path):
    """A fake torch.distributed that sums two ranks proves global counts come from one all_reduce per window."""
    calls = []

    class Tensor(list):
        def tolist(self):
            return list(self)

    fake = types.ModuleType("torch")
    fake.int64 = "int64"
    fake.device = lambda *a: "cpu"
    fake.tensor = lambda data, dtype=None, device=None: Tensor(data)
    fake.cuda = SimpleNamespace(is_available=lambda: False, current_device=lambda: 0)

    def all_reduce(tensor):
        calls.append(list(tensor))
        tensor[:] = [v * 2 for v in tensor]  # the other rank contributed identical counts

    fake.distributed = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, get_backend=lambda: "gloo",
                                       all_reduce=all_reduce, get_world_size=lambda: 2)
    monkeypatch.setitem(sys.modules, "torch", fake)
    live = live_app()
    tracker = make_tracker(live.base, "g1", 0, 2, tmp_path).start()
    time.sleep(0.2)
    steps(tracker, n=4)
    tracker.finish("completed")
    assert len(calls) == 2  # 4 steps / log_every 2 = 2 windows -> exactly 2 collectives
    run = live.state.training.get_run("g1")
    assert run["tokens"]["nonpad"] == 4 * 120 * 2 and run["tokens"]["supervised"] == 4 * 100 * 2  # global = both ranks
    local = live.state.training.rank_steps("g1")
    assert sum(r["nonpad_tokens"] for r in local) == 4 * 120  # per-rank rows stay local


def test_failed_run_skips_the_collective_so_a_crash_cannot_hang_the_peers(monkeypatch, tmp_path):
    called = []
    fake = types.ModuleType("torch")
    fake.distributed = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, get_backend=lambda: "gloo",
                                       all_reduce=lambda t: called.append(t), get_world_size=lambda: 2)
    fake.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    tracker = TrainingTracker("http://127.0.0.1:9", "x", rank=0, world_size=2, run_id="f", enabled=True, spool_dir=tmp_path)
    tracker._failed = True
    assert tracker._global_counts([1, 2, 3, 0]) == [1, 2, 3, 0] and called == []


def test_run_id_is_shared_via_env_and_explicit_value(monkeypatch):
    monkeypatch.delenv("DGPU_RUN_ID", raising=False)
    assert resolve_run_id("abc") == "abc"
    monkeypatch.setenv("DGPU_RUN_ID", "from-env")
    assert resolve_run_id() == "from-env"
    monkeypatch.delenv("DGPU_RUN_ID")
    assert len(resolve_run_id()) == 32  # no torch.distributed: a fresh id


def test_launch_env_makes_collectives_fail_instead_of_hang():
    env = ddp.launch_env(timeout_s=120, run_id="r1")
    assert env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1" and env["DGPU_RUN_ID"] == "r1"
    assert ddp.process_group_timeout(120).total_seconds() == 120
    assert ddp.install_comm_timing(object(), None) is False  # not a DDP model / no torch


def test_ddp_numerical_equivalence_on_cpu():
    pytest.importorskip("torch")
    token = ddp.run_equivalence_check(world_size=2, mode="token_normalised")
    assert token["equivalent"], token
    naive = ddp.run_equivalence_check(world_size=2, mode="naive_mean", port=29512)
    assert not naive["equivalent"]  # per-rank mean loss is NOT equivalent with ragged batches: the check has teeth


# ----------------------------------------------------------- hardware extras


def test_torch_devices_map_to_physical_gpus_even_when_indices_are_renumbered(tmp_path):
    from dual_gpu_setup.training.hardware import discover_gpus

    gpus, _ = discover_gpus(build_sysfs(tmp_path), platform_name="linux")
    for gpu, lane in zip(gpus, ("r9700", "9070xt")):
        gpu.lane = lane
    devices = [  # HIP_VISIBLE_DEVICES swapped the order: torch index 0 is the 16 GB card
        {"index": 0, "name": "gfx1201", "pci_domain_id": 0, "pci_bus_id": 8, "pci_device_id": 0, "total_memory": 16 * 1024**3},
        {"index": 1, "name": "gfx1201", "pci_domain_id": 0, "pci_bus_id": 3, "pci_device_id": 0, "total_memory": 32 * 1024**3},
    ]
    mapping = {m["torch_index"]: m for m in map_torch_devices(devices, gpus)}
    assert mapping[0]["lane"] == "9070xt" and mapping[1]["lane"] == "r9700" and mapping[0]["matched_by"] == "pci_bus"
    no_pci = [{"index": 0, "name": "x", "total_memory": 32 * 1024**3}]
    fallback = map_torch_devices(no_pci, gpus)[0]
    assert fallback["lane"] == "r9700" and fallback["matched_by"] == "vram_size"
    assert torch_pci_address({"pci_bus_id": "0000:0A:00.0"}) == "0000:0a:00.0" and torch_pci_address({"name": "x"}) is None


def test_hardware_environment_records_link_state_and_visibility(tmp_path, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,0")
    sampler = HardwareSampler(sysfs_root=build_sysfs(tmp_path), platform_name="linux")
    environment = sampler.hardware_environment()
    assert [g["pci_bus"] for g in environment["gpus"]] == ["0000:03:00.0", "0000:08:00.0"]
    assert environment["gpus"][0]["pcie_width"] == "8" and environment["visibility"] == {"HIP_VISIBLE_DEVICES": "1,0"}
    assert environment["os"]["system"] and "rocm" in environment


def test_rocm_installation_records_all_available_component_versions(tmp_path):
    info = tmp_path / ".info"
    info.mkdir()
    (info / "version").write_text("7.0.0\n", encoding="utf-8")
    (info / "version-hip-runtime").write_text("7.0.12345\n", encoding="utf-8")
    result = rocm_installation(tmp_path)
    assert result == {
        "path": str(tmp_path), "available": True, "version": "7.0.0",
        "components": {"version": "7.0.0", "version-hip-runtime": "7.0.12345"},
    }


def test_sampler_collects_system_samples_and_amd_smi_fills_only_gaps(tmp_path):
    systems = []
    smi = AmdSmiReader(module=FakeAmdSmi())
    sampler = HardwareSampler(sysfs_root=build_sysfs(tmp_path), platform_name="linux", smi=smi, on_system=systems.append)
    samples = sampler.sample_once()
    first, second = samples
    assert first["temp_memory_c"] == 82.0  # sysfs value kept, not overwritten by AMD SMI's 99
    assert second["temp_memory_c"] == 99.0 and second["amd_smi_fields"] == ["temp_memory_c", "throttle_status"] or "temp_memory_c" in second["amd_smi_fields"]
    assert "temp_memory_c" not in second["unavailable"] and second["throttle_status"] == 4
    assert systems and "ram_used_pct" in systems[0]
    assert sampler.snapshot()["amd_smi"]["available"] is True


class FakeAmdSmi:
    class AmdSmiTemperatureType:
        EDGE, JUNCTION, VRAM = "edge", "junction", "vram"

    class AmdSmiTemperatureMetric:
        CURRENT = "current"

    class AmdSmiClkType:
        GFX, MEM = "gfx", "mem"

    @staticmethod
    def amdsmi_init():
        return None

    @staticmethod
    def amdsmi_get_processor_handles():
        return ["h3", "h8"]

    @staticmethod
    def amdsmi_get_gpu_device_bdf(handle):
        return {"h3": "0000:03:00.0", "h8": "0000:08:00.0"}[handle]

    @staticmethod
    def amdsmi_get_temp_metric(handle, kind, metric):
        if kind == "vram":
            return 99
        raise RuntimeError("N/A on this card")  # unsupported sensors raise; the reader must tolerate it

    @staticmethod
    def amdsmi_get_gpu_metrics_info(handle):
        return {"throttle_status": 4 if handle == "h8" else "N/A"}


def test_amd_smi_unavailable_is_reported_not_raised():
    reader = AmdSmiReader()  # no amdsmi package on this machine
    assert reader.available is False and reader.reason
    assert reader.merge({"pci_bus": "0000:03:00.0", "temp_memory_c": None, "unavailable": {}}) == {
        "pci_bus": "0000:03:00.0", "temp_memory_c": None, "unavailable": {}}
