from __future__ import annotations

from pathlib import Path

import pytest

from dual_gpu_setup import gguf


def test_round_trips_scalars_and_strings(gguf_writer):
    path = gguf_writer("model.gguf", {
        "general.architecture": "llama",
        "llama.block_count": 32,
        "llama.embedding_length": 4096,
    })
    meta = gguf.read_metadata(path)
    assert meta["general.architecture"] == "llama"
    assert meta["llama.block_count"] == 32
    assert meta["llama.embedding_length"] == 4096


def test_round_trips_int_array(gguf_writer):
    path = gguf_writer("model.gguf", {
        "general.architecture": "gemma4",
        "gemma4.attention.head_count_kv": [8, 8, 2, 8, 8, 2],
    })
    meta = gguf.read_metadata(path)
    assert meta["gemma4.attention.head_count_kv"] == [8, 8, 2, 8, 8, 2]


def test_round_trips_bool_array(gguf_writer):
    path = gguf_writer("model.gguf", {
        "general.architecture": "gemma4",
        "gemma4.attention.sliding_window_pattern": [True, True, False, True, True, False],
    })
    meta = gguf.read_metadata(path)
    assert meta["gemma4.attention.sliding_window_pattern"] == [True, True, False, True, True, False]


def test_wanted_keys_skips_everything_else(gguf_writer):
    """The whole point of the skip path: unwanted keys must not corrupt subsequent reads,
    and the wanted key's value must come out identical to a full parse."""
    path = gguf_writer("model.gguf", {
        "general.architecture": "llama",
        "llama.block_count": 32,
        "llama.some_huge_array": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "llama.some_string_array": ["alpha", "beta", "gamma"],
        "llama.embedding_length": 4096,
    })
    full = gguf.read_metadata(path)
    targeted = gguf.read_metadata(path, wanted_keys={"general.architecture", "llama.embedding_length"})
    assert targeted == {
        "general.architecture": full["general.architecture"],
        "llama.embedding_length": full["llama.embedding_length"],
    }


def test_bad_magic_raises(tmp_path: Path):
    path = tmp_path / "not_gguf.bin"
    path.write_bytes(b"NOPE" + b"\x00" * 20)
    with pytest.raises(gguf.GGUFParseError):
        gguf.read_metadata(path)


def test_truncated_file_raises(gguf_writer):
    path = gguf_writer("model.gguf", {"general.architecture": "llama", "llama.block_count": 32})
    truncated = path.with_name("truncated.gguf")
    truncated.write_bytes(path.read_bytes()[:10])
    with pytest.raises(gguf.GGUFParseError):
        gguf.read_metadata(truncated)
