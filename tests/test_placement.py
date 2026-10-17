from dual_gpu_setup.server import PLACEMENT_MIN_VRAM_FRACTION, placement_error

GIB = 1024**3


def test_model_resident_in_vram_passes():
    # Qwen 9B-class file fully on GPU: dedicated VRAM ≈ file size plus KV cache.
    assert placement_error(dedicated_gb=5.8, model_bytes=int(5.3 * GIB), device="ROCm1") is None


def test_model_on_cpu_fails_with_device_and_sizes_in_message():
    # The Gemma 31B incident: 17.9 GB weights, only 0.3 GB reported in VRAM.
    message = placement_error(dedicated_gb=0.32, model_bytes=int(17.9 * GIB), device="ROCm0")
    assert message is not None
    assert "ROCm0" in message
    assert "0.32 GB" in message
    assert "CPU" in message


def test_threshold_is_inclusive_at_the_minimum_fraction():
    model_bytes = 10 * GIB
    minimum_gb = 10 * PLACEMENT_MIN_VRAM_FRACTION
    assert placement_error(minimum_gb, model_bytes, "ROCm0") is None
    assert placement_error(minimum_gb - 0.01, model_bytes, "ROCm0") is not None
