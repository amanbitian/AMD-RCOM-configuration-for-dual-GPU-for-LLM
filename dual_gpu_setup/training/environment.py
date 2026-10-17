"""Reproducibility snapshot collected inside the training process (so library versions are the
ones actually used by the run). Everything is best effort; missing items are simply absent."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ENV_VARS = (
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "TORCH_BLAS_PREFER_HIPBLASLT", "NCCL_DEBUG",
    "RCCL_DEBUG", "OMP_NUM_THREADS",
)
PACKAGES = ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes", "flash_attn")


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _git(args: list[str], cwd: str | None) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_environment(cwd: str | None = None) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "kernel": platform.release(),
        "packages": versions,
        "env": {name: os.environ[name] for name in ENV_VARS if name in os.environ},
        "env_prefixed": {k: v for k, v in os.environ.items() if k.startswith("MIOPEN_")},
        "rocm_version": _read("/opt/rocm/.info/version"),
        "amdgpu_driver": _read("/sys/module/amdgpu/version"),
    }
    try:  # imported lazily: the tracker must work without torch installed
        import torch

        snapshot["torch"] = {
            "version": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "cuda_available": torch.cuda.is_available(),
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                    "pci_bus_id": getattr(torch.cuda.get_device_properties(i), "pci_bus_id", None),
                    "pci_domain_id": getattr(torch.cuda.get_device_properties(i), "pci_domain_id", None),
                    "pci_device_id": getattr(torch.cuda.get_device_properties(i), "pci_device_id", None),
                }
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
    except Exception:  # noqa: BLE001
        pass
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit:
        snapshot["git"] = {
            "commit": commit,
            "dirty": bool(_git(["status", "--porcelain"], cwd)),
        }
    return snapshot
