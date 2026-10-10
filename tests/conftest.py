"""Shared test fixtures.

`write_gguf` builds tiny synthetic GGUF files with just the metadata keys a test needs,
so the suite never depends on any real model file being present on disk (a real GGUF
carries gigabytes of tensor data we'd never touch anyway -- the reader only looks at the
metadata header). This mirrors the on-disk format closely enough that
dual_gpu_setup.gguf.read_metadata reads these fixtures exactly like a real file.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

_TYPE_UINT32 = 4
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10


def _encode_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _encode_value(value: Any) -> tuple[int, bytes]:
    """Returns (gguf_type, encoded_bytes) for a Python value used in test fixtures."""
    if isinstance(value, str):
        return _TYPE_STRING, _encode_string(value)
    if isinstance(value, bool):
        return _TYPE_BOOL, struct.pack("<?", value)
    if isinstance(value, int):
        return _TYPE_UINT32, struct.pack("<I", value)
    if isinstance(value, float):
        return _TYPE_FLOAT32, struct.pack("<f", value)
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty arrays aren't supported by this test fixture writer.")
        element_type, _ = _encode_value(value[0])
        body = b"".join(_encode_value(item)[1] for item in value)
        return _TYPE_ARRAY, struct.pack("<IQ", element_type, len(value)) + body
    raise TypeError(f"Unsupported fixture value type: {type(value)!r}")


def write_gguf(path: Path, metadata: dict[str, Any]) -> Path:
    """Write a minimal valid GGUF file (version 3, zero tensors) with the given metadata."""
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(metadata))
    for key, value in metadata.items():
        value_type, encoded = _encode_value(value)
        body += _encode_string(key) + struct.pack("<I", value_type) + encoded
    path.write_bytes(body)
    return path


@pytest.fixture
def gguf_writer(tmp_path: Path):
    """Returns a function(name, metadata) -> Path that writes a fixture GGUF under tmp_path."""

    def _write(name: str, metadata: dict[str, Any]) -> Path:
        return write_gguf(tmp_path / name, metadata)

    return _write
