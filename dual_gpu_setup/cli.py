from __future__ import annotations

import argparse
import json
from collections import defaultdict

from dual_gpu_setup.config import load_config
from dual_gpu_setup.lmstudio import list_devices, resolve_device, runtime_dir
from dual_gpu_setup.orchestrator import DualGPUOrchestrator, prepare_models


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal dual-GPU llama-server orchestrator")
    parser.add_argument("--config", required=True, help="Path to the dual GPU TOML config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect", help="Show runtime discovery and lane-to-device resolution")
    subparsers.add_parser("plan", help="Print model eligibility and queue classification")
    run_parser = subparsers.add_parser("run", help="Run the configured dual-GPU workload")
    run_parser.add_argument("--dry-run", action="store_true", help="Create a run folder and summary without loading models")
    return parser


def _inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    grouped: dict[str, list[str]] = defaultdict(list)
    for lane in config.lanes:
        grouped[lane.backend].append(lane.key)
    print(f"Project: {config.project.name}")
    for backend, lane_keys in grouped.items():
        print(f"\nBackend: {backend}")
        print(f"  Runtime: {runtime_dir(config, backend)}")
        devices = list_devices(config, backend)
        for device_id, description in devices:
            print(f"  Device: {device_id} -> {description}")
        for lane in [candidate for candidate in config.lanes if candidate.backend == backend]:
            print(f"  Lane {lane.key}: {lane.display} -> {resolve_device(config, lane)} (port {lane.port})")
    return 0


def _plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    single_gpu, multi_gpu = prepare_models(config)
    lanes = sorted(config.lanes, key=lambda item: item.vram_gb, reverse=True)
    big_lane, small_lane = lanes[0], lanes[1]
    print(f"Project: {config.project.name}")
    print(f"Lanes: {big_lane.key}={big_lane.vram_gb:.1f} GB, {small_lane.key}={small_lane.vram_gb:.1f} GB")
    print("\nSingle GPU models:")
    for model in sorted(single_gpu, key=lambda item: item.size_gb):
        eligible = []
        for lane in lanes:
            if model.source.pin_lane and model.source.pin_lane != lane.key:
                continue
            if model.size_gb <= lane.vram_gb * config.policy.vram_safety_fraction:
                eligible.append(lane.key)
        status = ", ".join(eligible) if eligible else "SKIP"
        pin = f" pin={model.source.pin_lane}" if model.source.pin_lane else ""
        print(f"  {model.name:<35} {model.size_gb:>6.2f} GB  -> {status}{pin}")
    if multi_gpu:
        print("\nMulti GPU models:")
        for model in sorted(multi_gpu, key=lambda item: item.size_gb, reverse=True):
            extra = f" tensor_split={model.source.tensor_split}" if model.source.tensor_split else ""
            print(f"  {model.name:<35} {model.size_gb:>6.2f} GB{extra}")
    return 0


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if not config.tasks:
        raise SystemExit("Config must define at least one task before running.")
    summary = DualGPUOrchestrator(config).run(dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "plan":
        return _plan(args)
    if args.command == "run":
        return _run(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
