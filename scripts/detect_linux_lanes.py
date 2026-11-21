#!/usr/bin/env python
"""Print `[[lanes]]` stanzas with the REAL Linux PCI bus of the R9700 and RX 9070 XT.

    python scripts/detect_linux_lanes.py                  # reads /sys/class/drm (amdgpu), prints TOML
    python scripts/detect_linux_lanes.py --sysfs /path    # fake tree (tests)

Uses the same discovery as the dashboard's telemetry (dual_gpu_setup.training.hardware), so the addresses it prints are
exactly the ones the sampler, the lane assignment and the trainer's PCI-first device mapping will use.  Integrated GPUs
(small VRAM carve-out) are listed but never assigned a lane.  Exit code 1 when the two lanes cannot be assigned
unambiguously (then fix by hand with `lspci -nn | grep -i 'AMD/ATI'`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dual_gpu_setup.training.hardware import discover_gpus  # noqa: E402

LANES = (
    {"key": "r9700", "display": "AMD Radeon AI PRO R9700", "match": "R9700", "vram_gb": 32, "port": 8081},
    {"key": "9070xt", "display": "AMD Radeon RX 9070 XT", "match": "9070 XT", "vram_gb": 16, "port": 8082},
)
MIN_DISCRETE_GB = 6.0


def assign(gpus) -> tuple[dict[str, str], list[str]]:
    """lane key -> pci_bus, plus human notes. Name match first (product_name), then nearest VRAM; discrete cards only."""
    notes: list[str] = []
    discrete = []
    for g in gpus:
        gb = (g.vram_total_bytes or 0) / 1024**3
        if gb and gb < MIN_DISCRETE_GB:
            notes.append(f"{g.pci_bus} ({g.name}, {gb:.1f} GB) looks integrated: ignored")
        else:
            discrete.append((g, gb))
    out: dict[str, str] = {}
    used: set[str] = set()
    for lane in LANES:
        needle = lane["match"].lower()
        hit = next((g for g, _ in discrete if g.pci_bus not in used and needle in (g.name or "").lower()), None)
        if hit is None:
            cands = [(g, gb) for g, gb in discrete if g.pci_bus not in used and gb]
            cands = [(g, gb) for g, gb in cands if abs(gb - lane["vram_gb"]) <= max(1.5, 0.12 * lane["vram_gb"])]
            hit = cands[0][0] if len(cands) == 1 else None
        if hit is None:
            notes.append(f"lane {lane['key']}: no unambiguous GPU (name '{lane['match']}' / ~{lane['vram_gb']} GB)")
        else:
            out[lane["key"]] = hit.pci_bus
            used.add(hit.pci_bus)
    return out, notes


def render(found: dict[str, str]) -> str:
    chunks = []
    for lane in LANES:
        bus = found.get(lane["key"])
        chunks.append("\n".join([
            "[[lanes]]", f'key = "{lane["key"]}"', f'display = "{lane["display"]}"', f'match = "{lane["match"]}"',
            f'vram_gb = {lane["vram_gb"]}', f'port = {lane["port"]}', 'backend = "rocm"',
            f'pci_bus = "{bus}"' if bus else '# pci_bus = "<not detected>"']))
    return "\n\n".join(chunks) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sysfs", default="/sys")
    args = ap.parse_args(argv)
    gpus, reason = discover_gpus(args.sysfs, platform_name="linux")
    if reason:
        print(f"# cannot detect GPUs: {reason}", file=sys.stderr)
        return 1
    found, notes = assign(gpus)
    for g in gpus:
        print(f"# {g.pci_bus}  {g.card}  {g.name}  {(g.vram_total_bytes or 0) / 1024**3:.1f} GB", file=sys.stderr)
    for n in notes:
        print(f"# note: {n}", file=sys.stderr)
    print(render(found))
    return 0 if len(found) == len(LANES) else 1


if __name__ == "__main__":
    sys.exit(main())
