"""CPU-only checks for the native-Ubuntu deployment files. No GPU, no Linux, no service needed."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from dual_gpu_setup.config import load_config

ROOT = Path(__file__).resolve().parents[1]
UBUNTU = ROOT / "example.dual_gpu.ubuntu.toml"
# the SLM trainer (SLM_training: dgpu.rank_lanes / DEFAULT_LANES) addresses GPUs by exactly these keys, rank order
TRAINER_LANE_KEYS = ["r9700", "9070xt"]
DRIVE = re.compile(r"(^|[\s\"'=])[A-Za-z]:[\\/]")


def _walk(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)
    elif isinstance(node, str):
        yield node


def _load_detect():
    spec = importlib.util.spec_from_file_location("detect_linux_lanes", ROOT / "scripts" / "detect_linux_lanes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["detect_linux_lanes"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ubuntu_example_loads_and_lane_keys_match_the_trainer():
    cfg = load_config(UBUNTU)
    assert [lane.key for lane in cfg.lanes] == TRAINER_LANE_KEYS
    assert [lane.vram_gb for lane in cfg.lanes] == [32.0, 16.0]
    assert cfg.project.host == "127.0.0.1"
    assert all(lane.backend == "rocm" for lane in cfg.lanes)


def test_ubuntu_example_has_no_drive_letters_or_gpu0_gpu1():
    raw = tomllib.loads(UBUNTU.read_text(encoding="utf-8"))
    assert not [v for v in _walk(raw) if DRIVE.search(v)]
    text = UBUNTU.read_text(encoding="utf-8")
    assert not re.search(r'key = "gpu[01]"', text), "lane keys must be r9700 / 9070xt, not gpu0 / gpu1"
    assert "scripts/detect_linux_lanes.py" in text and "PRIMARY identity" in text


def test_systemd_unit_is_loopback_restarting_graceful_and_scoped():
    unit = (ROOT / "deploy" / "dual-gpu.service").read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in unit and "0.0.0.0" not in unit.replace("do NOT add --host 0.0.0.0", "").replace(
        "Do NOT add --host 0.0.0.0", "") or "Do NOT add" in unit
    exec_line = next(l for l in unit.splitlines() if l.startswith("ExecStart="))
    assert "0.0.0.0" not in exec_line and "--lan" not in exec_line and "--no-browser" in exec_line
    for needle in ("Restart=on-failure", "KillSignal=SIGTERM", "TimeoutStopSec=", "ReadWritePaths=", "WantedBy=default.target"):
        assert needle in unit, needle
    assert b"\r\n" not in (ROOT / "deploy" / "dual-gpu.service").read_bytes()


def _fake_sysfs(tmp_path: Path, cards) -> Path:
    root = tmp_path / "sys"
    for card, bus, gb, product in cards:
        dev = root / "class" / "drm" / card / "device"
        dev.mkdir(parents=True)
        (dev / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME={bus}\n")
        (dev / "vendor").write_text("0x1002\n")
        (dev / "mem_info_vram_total").write_text(str(int(gb * 1024**3)) + "\n")
        if product:
            (dev / "product_name").write_text(product + "\n")
    return root


def test_detect_lanes_prints_real_pci_buses_and_ignores_the_igpu(tmp_path, capsys):
    detect = _load_detect()
    root = _fake_sysfs(tmp_path, [("card0", "0000:0e:00.0", 0.5, "AMD Radeon Graphics"),
                                  ("card1", "0000:0b:00.0", 16, None),
                                  ("card2", "0000:03:00.0", 32, "AMD Radeon AI PRO R9700")])
    assert detect.main(["--sysfs", str(root)]) == 0
    out = capsys.readouterr()
    lanes = tomllib.loads(out.out)["lanes"]
    assert {l["key"]: l["pci_bus"] for l in lanes} == {"r9700": "0000:03:00.0", "9070xt": "0000:0b:00.0"}
    assert "integrated" in out.err and "0000:0e:00.0" not in out.out


def test_detect_output_round_trips_through_the_config_loader(tmp_path, capsys):
    detect = _load_detect()
    root = _fake_sysfs(tmp_path, [("card0", "0000:03:00.0", 32, "AMD Radeon AI PRO R9700"),
                                  ("card1", "0000:0b:00.0", 16, None)])
    assert detect.main(["--sysfs", str(root)]) == 0
    base = UBUNTU.read_text(encoding="utf-8")
    head = base[: base.index("# Lane keys MUST")]
    tail = base[base.index("[[models]]"):]
    toml = tmp_path / "dual_gpu.toml"
    toml.write_text(head + capsys.readouterr().out + "\n" + tail)
    cfg = load_config(toml)
    assert [(l.key, l.pci_bus) for l in cfg.lanes] == [("r9700", "0000:03:00.0"), ("9070xt", "0000:0b:00.0")]


def test_detect_reports_failure_when_a_lane_cannot_be_assigned(tmp_path, capsys):
    detect = _load_detect()
    root = _fake_sysfs(tmp_path, [("card0", "0000:03:00.0", 32, "AMD Radeon AI PRO R9700")])
    assert detect.main(["--sysfs", str(root)]) == 1
    assert "<not detected>" in capsys.readouterr().out


def test_detect_without_amd_gpus_fails_cleanly(tmp_path):
    detect = _load_detect()
    (tmp_path / "sys" / "class" / "drm").mkdir(parents=True)
    assert detect.main(["--sysfs", str(tmp_path / "sys")]) == 1


def test_linux_service_doc_covers_required_validation_commands():
    doc = (ROOT / "docs" / "LINUX_SERVICE.md").read_text(encoding="utf-8")
    for needle in ("--check", "launch_service.py --start", "/api/health", "/api/hardware", "/api/leases", "llama-server",
                   "rocm_hardware", "127.0.0.1"):
        assert needle in doc, needle
