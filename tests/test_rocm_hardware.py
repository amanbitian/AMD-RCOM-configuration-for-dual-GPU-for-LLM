"""Real-hardware checks. Deselected by default; run ONLY on the AMD workstation:

    python -m pytest -m rocm_hardware -q

They read the live /sys, so they validate what the unit tests can only fake: that gfx1201 boards really expose the
sensors the dashboard relies on, and that unavailable ones come back as None (never 0).
"""
from __future__ import annotations

import sys

import pytest

from dual_gpu_setup.training.hardware import assign_lanes, discover_gpus, sample_gpu

pytestmark = [pytest.mark.rocm_hardware,
              pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs native Linux sysfs")]


@pytest.fixture(scope="module")
def gpus():
    found, reason = discover_gpus("/sys", platform_name="linux")
    if len(found) < 2:
        pytest.skip(f"fewer than two AMD GPUs visible ({reason or len(found)})")
    return found


def test_both_gpus_have_a_pci_address_and_vram(gpus):
    assert len({g.pci_bus for g in gpus}) == len(gpus)
    discrete = [g for g in gpus if (g.vram_total_bytes or 0) > 6 * 1024**3]
    assert len(discrete) >= 2, [(g.pci_bus, g.vram_total_bytes) for g in gpus]


def test_samples_report_core_sensors_and_never_fake_zero(gpus):
    for g in gpus:
        if (g.vram_total_bytes or 0) < 6 * 1024**3:
            continue
        s = sample_gpu(g)
        for field in ("gpu_busy_pct", "vram_used_bytes", "vram_total_bytes", "power_w", "temp_edge_c", "pcie_width"):
            assert s[field] is not None, f"{g.pci_bus}: {field} unavailable ({s['unavailable'].get(field)})"
        for field, reason in s["unavailable"].items():
            assert s[field] is None, f"{g.pci_bus}: {field} is {s[field]!r} but listed unavailable ({reason})"
        assert s["power_w"] > 0 and s["vram_total_bytes"] > 6 * 1024**3


def test_lane_assignment_by_pci_bus_on_this_machine(gpus):
    from types import SimpleNamespace
    big = sorted((g for g in gpus if (g.vram_total_bytes or 0) > 6 * 1024**3), key=lambda g: -g.vram_total_bytes)
    lanes = [SimpleNamespace(key="r9700", match="R9700", vram_gb=32, pci_bus=big[0].pci_bus),
             SimpleNamespace(key="9070xt", match="9070 XT", vram_gb=16, pci_bus=big[1].pci_bus)]
    assign_lanes(gpus, lanes)
    assert {g.pci_bus: g.lane for g in big} == {big[0].pci_bus: "r9700", big[1].pci_bus: "9070xt"}


def test_single_lane_isolation_hides_the_other_gpu_from_the_server(gpus):
    """With HIP_VISIBLE_DEVICES=<n> a llama-server must enumerate exactly one ROCm device and the other card must
    stay at idle. Run this (and read the result) before turning on policy.isolate_single_lane."""
    import os
    import shutil
    import subprocess
    import time

    from dual_gpu_setup.training.hardware import sample_gpu

    binary = shutil.which("llama-server") or os.environ.get("LLAMA_SERVER")
    if not binary:
        pytest.skip("llama-server not on PATH (set LLAMA_SERVER)")
    discrete = [g for g in gpus if (g.vram_total_bytes or 0) > 6 * 1024**3]
    before = {g.pci_bus: sample_gpu(g) for g in discrete}
    env = {**os.environ, "HIP_VISIBLE_DEVICES": "0"}
    out = subprocess.run([binary, "--list-devices"], env=env, capture_output=True, text=True, timeout=120)
    listed = [line for line in (out.stdout + out.stderr).splitlines() if line.strip().startswith("ROCm")]
    assert len(listed) == 1, listed
    time.sleep(2)
    after = {g.pci_bus: sample_gpu(g) for g in discrete}
    # HIP ordinal 0 is the first discrete card by PCI order; every other card must have been left untouched
    for gpu in discrete[1:]:
        assert abs((after[gpu.pci_bus]["vram_used_bytes"] or 0) - (before[gpu.pci_bus]["vram_used_bytes"] or 0)) < 256 * 1024**2
        assert (after[gpu.pci_bus]["gpu_busy_pct"] or 0) < 10
