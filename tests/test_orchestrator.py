from __future__ import annotations

from pathlib import Path

import pytest

from dual_gpu_setup.config import AppConfig, DiscoveryConfig, PolicyConfig, ProjectConfig
from dual_gpu_setup.orchestrator import ctx_for, estimate_kv_cache_gb, resolve_context_window


def make_config(**policy_overrides) -> AppConfig:
    policy = PolicyConfig(**policy_overrides)
    return AppConfig(
        project=ProjectConfig(),
        discovery=DiscoveryConfig(),
        policy=policy,
        lanes=[],
        models=[],
        tasks=[],
        config_path=Path("test.toml"),
    )


# --- ctx_for ---------------------------------------------------------------

def test_ctx_for_picks_max_when_headroom_is_generous():
    config = make_config(vram_safety_fraction=0.9, ctx_max_headroom_gb=10, ctx_mid_headroom_gb=5, ctx_max=49152, ctx_mid=32768, ctx_min=16384)
    # lane=32GB * 0.9 = 28.8GB capacity; model=10GB -> headroom=18.8GB >= 10 -> ctx_max
    assert ctx_for(config, model_size=10.0, lane_vram_gb=32.0) == 49152


def test_ctx_for_picks_mid_for_moderate_headroom():
    config = make_config(vram_safety_fraction=0.9, ctx_max_headroom_gb=10, ctx_mid_headroom_gb=5, ctx_max=49152, ctx_mid=32768, ctx_min=16384)
    # lane=16GB * 0.9 = 14.4GB capacity; model=8GB -> headroom=6.4GB -> between mid(5) and max(10) -> ctx_mid
    assert ctx_for(config, model_size=8.0, lane_vram_gb=16.0) == 32768


def test_ctx_for_picks_min_for_tight_headroom():
    config = make_config(vram_safety_fraction=0.9, ctx_max_headroom_gb=10, ctx_mid_headroom_gb=5, ctx_max=49152, ctx_mid=32768, ctx_min=16384)
    # lane=16GB * 0.9 = 14.4GB capacity; model=13GB -> headroom=1.4GB < mid(5) -> ctx_min
    assert ctx_for(config, model_size=13.0, lane_vram_gb=16.0) == 16384


# --- resolve_context_window --------------------------------------------------

def test_resolve_context_window_unchanged_without_token_budget():
    config = make_config()
    assert resolve_context_window(config, baseline_context=8192) == 8192


def test_resolve_context_window_grows_to_fit_real_token_budget():
    config = make_config(safety_tokens=256, ctx_max=49152)
    result = resolve_context_window(config, baseline_context=8192, input_tokens=18272, max_output_tokens=12000)
    assert result == 18272 + 12000 + 256  # bumped above the 8192 baseline


def test_resolve_context_window_keeps_baseline_if_already_big_enough():
    config = make_config(safety_tokens=256, ctx_max=49152)
    result = resolve_context_window(config, baseline_context=40000, input_tokens=1000, max_output_tokens=500)
    assert result == 40000


def test_resolve_context_window_rejects_when_exceeding_ctx_max():
    config = make_config(safety_tokens=256, ctx_max=16384)
    with pytest.raises(ValueError, match="exceeds policy ctx_max"):
        resolve_context_window(config, baseline_context=8192, input_tokens=18272, max_output_tokens=12000)


def test_resolve_context_window_rejects_negative_tokens():
    config = make_config()
    with pytest.raises(ValueError):
        resolve_context_window(config, baseline_context=8192, input_tokens=-1, max_output_tokens=100)


# --- estimate_kv_cache_gb -----------------------------------------------------

_DENSE_LLAMA = {
    "general.architecture": "llama",
    "llama.block_count": 32,
    "llama.embedding_length": 4096,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
}


def test_kv_cache_dense_model_exact_math(gguf_writer):
    path = gguf_writer("dense.gguf", _DENSE_LLAMA)
    gb, note = estimate_kv_cache_gb(path, context_size=8192)
    assert note == "ok"
    # head_dim = 4096/32 = 128; 2 bytes * 8 kv_heads * (128+128) * 8192 ctx * 32 layers
    expected_bytes = 2 * 8 * (128 + 128) * 8192 * 32
    assert gb == pytest.approx(expected_bytes / 1024**3, rel=1e-9)


def test_kv_cache_prefers_explicit_key_value_length(gguf_writer):
    """Regression test for the head_dim bug found while building this: some architectures
    (e.g. real Qwen3.5/3.8 GGUF files) expose attention.key_length/value_length that does
    NOT equal embedding_length/head_count, and that must win over the derived value."""
    meta = dict(_DENSE_LLAMA)
    meta["llama.attention.head_count"] = 24  # 4096/24 is not an integer
    meta["llama.attention.key_length"] = 256
    meta["llama.attention.value_length"] = 256
    path = gguf_writer("explicit_head_dim.gguf", meta)
    gb, note = estimate_kv_cache_gb(path, context_size=8192)
    assert note == "ok"
    expected_bytes = 2 * 8 * (256 + 256) * 8192 * 32
    assert gb == pytest.approx(expected_bytes / 1024**3, rel=1e-9)


def test_kv_cache_ambiguous_head_dim_without_key_value_length_is_skipped(gguf_writer):
    meta = dict(_DENSE_LLAMA)
    meta["llama.attention.head_count"] = 24  # 4096/24 not integer, and no key_length given
    path = gguf_writer("ambiguous.gguf", meta)
    gb, note = estimate_kv_cache_gb(path, context_size=8192)
    assert gb is None
    assert "ambiguous" in note


def test_kv_cache_skips_state_space_hybrid_architectures(gguf_writer):
    meta = dict(_DENSE_LLAMA)
    meta["general.architecture"] = "qwen35"
    meta = {k.replace("llama.", "qwen35.") if k != "general.architecture" else k: v for k, v in meta.items()}
    meta["qwen35.ssm.conv_kernel"] = 4
    path = gguf_writer("hybrid.gguf", meta)
    gb, note = estimate_kv_cache_gb(path, context_size=8192)
    assert gb is None
    assert "state-space" in note


def test_kv_cache_applies_sliding_window_pattern_when_present(gguf_writer):
    """Mirrors real gemma4 GGUF metadata: 5 local (capped) layers per 1 global (full) layer."""
    n_layers = 6
    meta = {
        "general.architecture": "gemma4",
        "gemma4.block_count": n_layers,
        "gemma4.embedding_length": 2816,
        "gemma4.attention.head_count": 16,
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 2],
        "gemma4.attention.sliding_window": 1024,
        "gemma4.attention.sliding_window_pattern": [True, True, True, True, True, False],
    }
    path = gguf_writer("sliding.gguf", meta)
    gb, note = estimate_kv_cache_gb(path, context_size=51200)
    assert note == "ok"

    # Without any sliding-window awareness, every layer would use the full context.
    full_context_gb, _ = estimate_kv_cache_gb(path, context_size=51200)
    meta_no_pattern = dict(meta)
    del meta_no_pattern["gemma4.attention.sliding_window_pattern"]
    path_no_pattern = gguf_writer("sliding_no_pattern.gguf", meta_no_pattern)
    naive_gb, naive_note = estimate_kv_cache_gb(path_no_pattern, context_size=51200)
    assert naive_note == "ok"
    assert gb < naive_gb  # the pattern-aware estimate must be smaller than the naive one


def test_kv_cache_missing_required_fields_is_skipped(gguf_writer):
    path = gguf_writer("incomplete.gguf", {"general.architecture": "llama", "llama.block_count": 32})
    gb, note = estimate_kv_cache_gb(path, context_size=8192)
    assert gb is None
    assert "missing" in note


def test_kv_cache_head_count_kv_array_length_mismatch_is_skipped(gguf_writer):
    meta = dict(_DENSE_LLAMA)
    meta["llama.attention.head_count_kv"] = [8, 8, 8]  # only 3 entries, block_count is 32
    path = gguf_writer("mismatch.gguf", meta)
    gb, note = estimate_kv_cache_gb(path, context_size=8192)
    assert gb is None
    assert "does not match" in note
