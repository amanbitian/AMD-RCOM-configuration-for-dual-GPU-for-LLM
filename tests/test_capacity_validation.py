"""End-to-end coverage for DeploymentManager._validate_capacity across all three deploy
modes -- this is the fix for the gap where a deploy's feasibility verdict only ever
checked model file size against lane VRAM, never the KV-cache cost of the context window
actually being requested (including one bumped up by input_tokens/max_output_tokens), and
never validated single_large_model deploys at all."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

_DENSE_ARCH = {
    "general.architecture": "llama",
    "llama.block_count": 32,
    "llama.embedding_length": 4096,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
}

_SSM_HYBRID_ARCH = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 32,
    "qwen35.embedding_length": 4096,
    "qwen35.attention.head_count": 32,
    "qwen35.attention.head_count_kv": 8,
    "qwen35.ssm.conv_kernel": 4,
}


def _write_config(tmp_path: Path, model_path: Path, model_size_gb: float, lane_vram_gb: float = 25.0) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[policy]
vram_safety_fraction = 0.90
ctx_min = 4096
ctx_mid = 16384
ctx_max = 65536
ctx_mid_headroom_gb = 5
ctx_max_headroom_gb = 10
safety_tokens = 256

[[lanes]]
key = "testlane"
display = "Test Lane"
vram_gb = {lane_vram_gb}
port = 19101

[[lanes]]
key = "otherlane"
display = "Other Lane"
vram_gb = {lane_vram_gb}
port = 19102

[[models]]
name = "test-model"
path = "{model_path.as_posix()}"
size_gb = {model_size_gb}
""",
        encoding="utf-8",
    )
    return config_path


def make_manager(config_path: Path, tmp_path: Path) -> "app.DeploymentManager":
    config = app.load_config(config_path)
    store = app.RunStore(tmp_path / "runs.sqlite3")
    return app.DeploymentManager(config, tmp_path / "data", store)


def test_single_gpu_rejects_when_kv_cache_pushes_past_capacity(tmp_path: Path, gguf_writer):
    model_path = gguf_writer("dense.gguf", _DENSE_ARCH)
    # lane capacity = 25 * 0.9 = 22.5GB. File alone (19GB) fits. head_dim=128, 8 kv_heads,
    # 32 layers: KV at ctx=32768 = 2*8*256*32768*32 bytes = 4GB -> total 23GB > 22.5GB capacity.
    config_path = _write_config(tmp_path, model_path, model_size_gb=19.0)
    manager = make_manager(config_path, tmp_path)
    try:
        try:
            manager._validate_profile("single_gpu", {"model": {"model": "test-model", "lane": "testlane", "context_window": 32768}})
            assert False, "expected AppError for oversized context"
        except app.AppError as exc:
            assert "KV cache" in str(exc)
            assert "22.50 GB" in str(exc)
    finally:
        manager.shutdown()


def test_single_gpu_passes_with_small_context(tmp_path: Path, gguf_writer):
    model_path = gguf_writer("dense.gguf", _DENSE_ARCH)
    config_path = _write_config(tmp_path, model_path, model_size_gb=19.0)
    manager = make_manager(config_path, tmp_path)
    try:
        # KV at ctx=4096 = 4GB * (4096/32768) = 0.5GB -> total 19.5GB < 22.5GB capacity.
        manager._validate_profile("single_gpu", {"model": {"model": "test-model", "lane": "testlane", "context_window": 4096}})
    finally:
        manager.shutdown()


def test_single_gpu_rejects_when_input_output_tokens_bump_context_past_capacity(tmp_path: Path, gguf_writer):
    model_path = gguf_writer("dense.gguf", _DENSE_ARCH)
    config_path = _write_config(tmp_path, model_path, model_size_gb=19.0)
    manager = make_manager(config_path, tmp_path)
    try:
        try:
            manager._validate_profile("single_gpu", {
                "model": {"model": "test-model", "lane": "testlane", "context_window": 4096,
                          "input_tokens": 30000, "max_output_tokens": 2000},
            })
            assert False, "expected AppError"
        except app.AppError as exc:
            assert "KV cache" in str(exc)
    finally:
        manager.shutdown()


def test_unknown_architecture_falls_back_to_file_size_only_check(tmp_path: Path, gguf_writer):
    """A model this estimator can't confidently read (SSM-hybrid here) must never be
    blocked by the new check -- preserves the exact pre-existing behavior for it."""
    model_path = gguf_writer("hybrid.gguf", _SSM_HYBRID_ARCH)
    config_path = _write_config(tmp_path, model_path, model_size_gb=19.0)
    manager = make_manager(config_path, tmp_path)
    try:
        # Even a huge context must pass: file alone (19GB) fits the 22.5GB budget, and KV
        # can't be estimated for this architecture, so the check must not reject it.
        manager._validate_profile("single_gpu", {"model": {"model": "test-model", "lane": "testlane", "context_window": 131072}})
    finally:
        manager.shutdown()


def test_single_large_model_mode_is_now_validated(tmp_path: Path, gguf_writer):
    """single_large_model deploys never called any capacity check before this fix."""
    model_path = gguf_writer("dense.gguf", _DENSE_ARCH)
    config_path = _write_config(tmp_path, model_path, model_size_gb=19.0)
    manager = make_manager(config_path, tmp_path)
    try:
        try:
            # combined capacity = 2 * 22.5 = 45GB. KV at ctx=262144 = 4GB * 8 = 32GB -> total 51GB > 45GB.
            manager._validate_profile("single_large_model", {"model": {"model": "test-model", "context_window": 262144}})
            assert False, "expected AppError"
        except app.AppError as exc:
            assert "KV cache" in str(exc)

        # A small context on the same combined budget must pass.
        manager._validate_profile("single_large_model", {"model": {"model": "test-model", "context_window": 4096}})
    finally:
        manager.shutdown()


def test_pin_lane_still_enforced_alongside_new_check(tmp_path: Path, gguf_writer):
    model_path = gguf_writer("dense.gguf", _DENSE_ARCH)
    config_path = _write_config(tmp_path, model_path, model_size_gb=19.0)
    config = app.load_config(config_path)
    config.models[0].pin_lane = "otherlane"
    store = app.RunStore(tmp_path / "runs.sqlite3")
    manager = app.DeploymentManager(config, tmp_path / "data", store)
    try:
        try:
            manager._validate_profile("single_gpu", {"model": {"model": "test-model", "lane": "testlane", "context_window": 4096}})
            assert False, "expected AppError for wrong pinned lane"
        except app.AppError as exc:
            assert "pinned" in str(exc)
    finally:
        manager.shutdown()
