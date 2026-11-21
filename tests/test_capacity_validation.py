"""End-to-end coverage for DeploymentManager._validate_capacity across all three deploy
modes -- this is the fix for the gap where a deploy's feasibility verdict only ever
checked model file size against lane VRAM, never the KV-cache cost of the context window
actually being requested (including one bumped up by input_tokens/max_output_tokens), and
never validated single_large_model deploys at all."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_capacity_check_counts_draft_model_weights(tmp_path: Path, gguf_writer, monkeypatch):
    """Speculative decoding loads a draft model on the same lane; the fit check must count it,
    or a deploy that OOMs would be approved (the auto-sizer already reserves it)."""
    target = gguf_writer("dense.gguf", _DENSE_ARCH)
    draft = gguf_writer("draft.gguf", _DENSE_ARCH)
    config_path = _write_config(tmp_path, target, model_size_gb=0.0, lane_vram_gb=25.0)  # capacity 22.5 GB
    manager = make_manager(config_path, tmp_path)
    # Target 20 GB fits alone (< 22.5); +4 GB draft -> 24 GB weights, which must be rejected.
    monkeypatch.setattr(app, "model_size_gb", lambda _c, m: 4.0 if "draft" in str(m.path) else 20.0)
    model = app.ModelConfig("test-model", str(target), ctx_size=4096, draft_model=str(draft))
    try:
        with pytest.raises(app.AppError, match="draft"):
            manager._validate_capacity(model, [manager.config.lanes[0]])
        # Without the draft, the same 20 GB target passes.
        manager._validate_capacity(app.ModelConfig("test-model", str(target), ctx_size=4096),
                                   [manager.config.lanes[0]])
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


# --- CPU-offloaded experts and KV-cache precision (capacity check used to ignore both) ---------------

_MOE_ARCH = {
    "general.architecture": "qwen3moe",
    "qwen3moe.block_count": 48,
    "qwen3moe.embedding_length": 4096,
    "qwen3moe.attention.head_count": 32,
    "qwen3moe.attention.head_count_kv": 4,
    "qwen3moe.expert_count": 128,
}


def _write_config_with_args(tmp_path: Path, model_path: Path, size_gb: float, model_args: str, lane_vram_gb: float = 25.0) -> Path:
    config_path = _write_config(tmp_path, model_path, size_gb, lane_vram_gb)
    config_path.write_text(config_path.read_text(encoding="utf-8") + f"extra_args = {model_args}\n", encoding="utf-8")
    return config_path


def test_cache_type_arguments_are_read_and_unknown_types_fall_back_to_f16():
    from dual_gpu_setup.orchestrator import cache_types_from_args

    assert cache_types_from_args(["--cache-type-k", "q8_0", "-ctv", "Q4_0"]) == ("q8_0", "q4_0")
    assert cache_types_from_args(["-ctk", "q8_0"], ["--cache-type-k", "f32"]) == ("f32", "f16")   # lane (last) wins
    assert cache_types_from_args(["--cache-type-k", "nonsense"], None) == ("f16", "f16")


def test_quantised_kv_cache_is_counted_at_its_real_size(gguf_writer):
    from dual_gpu_setup.orchestrator import estimate_kv_cache_gb

    path = gguf_writer("dense.gguf", _DENSE_ARCH)
    f16, _ = estimate_kv_cache_gb(path, 32768)
    q8, _ = estimate_kv_cache_gb(path, 32768, "q8_0", "q8_0")
    mixed, _ = estimate_kv_cache_gb(path, 32768, "q8_0", "f16")
    assert f16 == pytest.approx(4.0) and q8 == pytest.approx(4.0 * (34 / 32) / 2) and f16 > mixed > q8


def test_experts_held_on_the_cpu_do_not_count_against_vram(tmp_path: Path, gguf_writer):
    from dual_gpu_setup.orchestrator import offloaded_weights_gb

    path = gguf_writer("moe.gguf", _MOE_ARCH)
    assert offloaded_weights_gb(path, 59.0, ["--n-cpu-moe", "24"]) == pytest.approx(59 * 0.9 * 24 / 48)
    assert offloaded_weights_gb(path, 59.0, ["--cpu-moe"]) == pytest.approx(59 * 0.9)
    assert offloaded_weights_gb(path, 59.0, []) == 0.0
    assert offloaded_weights_gb(path, 59.0, [], gpu_layers=24) == pytest.approx(59 * 0.5)          # -ngl below the layer count
    assert offloaded_weights_gb(gguf_writer("dense.gguf", _DENSE_ARCH), 19.0, ["--n-cpu-moe", "9"]) == 0.0  # not an MoE model
    assert offloaded_weights_gb(tmp_path / "missing.gguf", 59.0, ["--cpu-moe"]) == 0.0              # unknown stays strict


def test_a_59gb_moe_model_with_experts_on_the_cpu_is_accepted_on_a_32gb_card(tmp_path: Path, gguf_writer):
    model_path = gguf_writer("moe.gguf", _MOE_ARCH)
    # 59 GB file, 90% experts, 40 of 48 layers' experts in RAM -> ~16 GB of weights in VRAM; budget 32 * 0.9 = 28.8 GB
    accepted = make_manager(_write_config_with_args(tmp_path, model_path, 59.0, '["--n-cpu-moe", "40"]', 32.0), tmp_path)
    try:
        accepted._validate_profile("single_gpu", {"model": {"model": "test-model", "lane": "testlane", "context_window": 4096}})
    finally:
        accepted.shutdown()
    other = tmp_path / "plain"
    other.mkdir()
    refused = make_manager(_write_config(other, model_path, 59.0, 32.0), other)
    try:
        with pytest.raises(app.AppError, match="needs 59.00 GB"):
            refused._validate_profile("single_gpu", {"model": {"model": "test-model", "lane": "testlane", "context_window": 4096}})
    finally:
        refused.shutdown()


def test_a_quantised_kv_cache_lets_a_context_fit_that_f16_would_refuse(tmp_path: Path, gguf_writer):
    model_path = gguf_writer("dense.gguf", _DENSE_ARCH)
    request = {"model": {"model": "test-model", "lane": "testlane", "context_window": 32768}}
    f16 = make_manager(_write_config(tmp_path, model_path, 19.0), tmp_path)
    try:
        with pytest.raises(app.AppError, match="KV cache"):
            f16._validate_profile("single_gpu", request)                   # 19 + 4.0 GB > 22.5 GB (the existing case)
    finally:
        f16.shutdown()
    other = tmp_path / "q8"
    other.mkdir()
    q8 = make_manager(_write_config_with_args(other, model_path, 19.0, '["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]'), other)
    try:
        q8._validate_profile("single_gpu", request)                        # 19 + 2.1 GB fits
    finally:
        q8.shutdown()
