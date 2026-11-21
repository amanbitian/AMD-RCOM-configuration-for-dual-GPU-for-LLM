"""Checkpoint manifests. The manifest is written LAST: a checkpoint directory without a valid
manifest is incomplete and must never be used for resume or evaluation."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
_STEP_RE = re.compile(r"checkpoint-(\d+)$")


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME and not p.name.endswith(".tmp")
    )


def write_manifest(
    directory: str | Path,
    *,
    step: int,
    kind: str = "resume",
    tokens_seen: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")
    started = time.perf_counter()
    paths = _files(directory)
    # hashlib releases the GIL, so large optimizer/adapter files hash concurrently.
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(paths)))) as pool:
        digests = list(pool.map(sha256_file, paths))
    files = [
        {"path": p.relative_to(directory).as_posix(), "size": p.stat().st_size, "sha256": digest}
        for p, digest in zip(paths, digests)
    ]
    if not files:
        raise ValueError(f"Checkpoint directory is empty: {directory}")
    manifest = {
        "version": MANIFEST_VERSION,
        "step": int(step),
        "kind": kind,
        "tokens_seen": tokens_seen,
        "created_at": time.time(),
        "files": files,
        "total_bytes": sum(f["size"] for f in files),
        "hash_ms": (time.perf_counter() - started) * 1000.0,
        **(extra or {}),
    }
    temporary = directory / (MANIFEST_NAME + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, directory / MANIFEST_NAME)
    return manifest


def read_manifest(directory: str | Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads((Path(directory) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


# What Hugging Face Trainer writes so that resume_from_checkpoint restores optimizer, schedule and RNG.
_RESUME_FILES = {
    "optimizer": ("optimizer.pt", "optimizer.bin"),
    "scheduler": ("scheduler.pt",),
    "trainer_state": ("trainer_state.json",),
}


def inspect_resume_state(directory: str | Path) -> dict[str, Any]:
    """Which exact-resume ingredients exist in a checkpoint directory (adapter-only checkpoints
    resume weights but not the optimizer/schedule/RNG, which silently changes the loss curve)."""
    directory = Path(directory)
    present = {name: any((directory / f).is_file() for f in files) for name, files in _RESUME_FILES.items()}
    present["rng"] = any(directory.glob("rng_state*.pth"))
    global_step = None
    try:
        global_step = int(json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))["global_step"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    missing = [name for name, ok in present.items() if not ok]
    return {"resume_state": present, "resume_state_complete": not missing, "resume_state_missing": missing,
            "trainer_global_step": global_step}


def validate_manifest(directory: str | Path, verify_hashes: bool = False) -> tuple[bool, str | None]:
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return False, "no manifest.json (checkpoint is incomplete)"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return False, "manifest.json is unreadable"
    for entry in files:
        target = directory / entry["path"]
        if not target.is_file():
            return False, f"missing file {entry['path']}"
        if target.stat().st_size != entry["size"]:
            return False, f"size mismatch for {entry['path']}"
        if verify_hashes and sha256_file(target) != entry["sha256"]:
            return False, f"hash mismatch for {entry['path']}"
    return True, None


_DISCOVER_RE = re.compile(r"^(?:checkpoint-|step_)0*(\d+)$")
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "adapter_model.bin")


def discover_checkpoints(directory: str | Path) -> list[dict[str, Any]]:
    """Checkpoint directories a trainer left on disk, whether or not it reported them to the service:
    `checkpoint-<step>` (Hugging Face) and `step_<step>` (custom loops). In-progress `.tmp_*` directories
    are ignored. No manifest is needed: `valid` means a weights file is present and non-empty, and the
    recorded manifest is marked `discovered` (sizes only, no hashes) so it is never mistaken for a
    trainer-written one. Nothing is written into the checkpoint directory."""
    directory = Path(directory)
    found = []
    if not directory.is_dir():
        return found
    for child in directory.iterdir():
        match = _DISCOVER_RE.match(child.name)
        if not match or not child.is_dir():
            continue
        files = [p for p in child.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
        weights = [p for p in files if p.name in _WEIGHT_FILES and p.stat().st_size > 0]
        sizes = [{"path": p.relative_to(child).as_posix(), "size": p.stat().st_size} for p in sorted(files)]
        total = sum(f["size"] for f in sizes)
        manifest = {"version": MANIFEST_VERSION, "step": int(match.group(1)), "kind": "discovered", "files": sizes,
                    "total_bytes": total, "hashed": False, "discovered_at": time.time(),
                    **inspect_resume_state(child)}
        stamps = [p.stat().st_mtime for p in files] or [child.stat().st_mtime]
        found.append({"step": int(match.group(1)), "path": str(child), "kind": "resume", "valid": bool(weights),
                      "mtime": max(stamps),
                      "size_bytes": total, "manifest": manifest,
                      "error": None if weights else "no weights file found in checkpoint directory"})
    return sorted(found, key=lambda c: c["step"])


def find_latest_valid(output_dir: str | Path, verify_hashes: bool = False) -> Path | None:
    """Newest `checkpoint-<step>` directory whose manifest validates (for Trainer resume)."""
    output_dir = Path(output_dir)
    candidates = []
    for child in output_dir.glob("checkpoint-*"):
        match = _STEP_RE.search(child.name)
        if child.is_dir() and match:
            candidates.append((int(match.group(1)), child))
    for _, child in sorted(candidates, reverse=True):
        ok, _reason = validate_manifest(child, verify_hashes)
        if ok:
            return child
    return None
