"""Regression coverage for load_config, including the dataclass(slots=True) class-vs-
instance default bug: PolicyConfig.some_field (class access) returns a slot descriptor,
not the real default, so load_config must never use ClassName.attr as a .get() fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from dual_gpu_setup.config import load_config

_MINIMAL_TOML = """
[discovery]
lmstudio_home = "{lmstudio_home}"
models_dir = "{models_dir}"

[[lanes]]
key = "a"
display = "A"
vram_gb = 16
port = 19001

[[lanes]]
key = "b"
display = "B"
vram_gb = 16
port = 19002

[[models]]
name = "test-model"
path = "test.gguf"
"""


def test_loads_with_every_optional_field_omitted(tmp_path: Path):
    """[project] and [policy] are entirely absent, and only a few [discovery]/[[lanes]]/
    [[models]] fields are set -- every default must come from the dataclass, not crash."""
    config_path = tmp_path / "minimal.toml"
    config_path.write_text(
        _MINIMAL_TOML.format(lmstudio_home=tmp_path.as_posix(), models_dir=tmp_path.as_posix()),
        encoding="utf-8",
    )
    config = load_config(config_path)

    assert config.project.name == "dual-gpu"
    assert config.project.host == "127.0.0.1"
    assert config.policy.reasoning_budget == 0
    assert config.policy.ctx_max == 49152
    assert config.policy.health_poll_seconds == 1.0
    assert config.policy.settle_seconds_after_restart == 2.0
    assert len(config.lanes) == 2
    assert config.models[0].name == "test-model"


def test_wrong_lane_count_rejected(tmp_path: Path):
    config_path = tmp_path / "one_lane.toml"
    config_path.write_text(
        """
[[lanes]]
key = "a"
display = "A"
vram_gb = 16
port = 19001

[[models]]
name = "test-model"
path = "test.gguf"
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        load_config(config_path)


def test_pinned_model_to_unknown_lane_rejected(tmp_path: Path):
    config_path = tmp_path / "bad_pin.toml"
    config_path.write_text(
        """
[[lanes]]
key = "a"
display = "A"
vram_gb = 16
port = 19001

[[lanes]]
key = "b"
display = "B"
vram_gb = 16
port = 19002

[[models]]
name = "test-model"
path = "test.gguf"
pin_lane = "nonexistent"
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        load_config(config_path)
