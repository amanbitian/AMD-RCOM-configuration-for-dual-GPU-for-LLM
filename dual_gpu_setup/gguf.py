"""Minimal, dependency-free GGUF header reader.

Only reads the metadata key/value section of a GGUF file -- never the tensor data --
so it stays fast regardless of model file size. This exists to estimate KV-cache VRAM
cost (see orchestrator.estimate_kv_cache_gb), which needs a few architecture fields
(layer count, embedding size, head counts) that aren't available anywhere else in this
codebase; ModelConfig only ever stores a file size, never architecture details.

GGUF format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

_MAGIC = b"GGUF"

# GGUF value type -> (struct format char, byte size)
_SCALAR_TYPES: dict[int, tuple[str, int]] = {
    0: ("B", 1),   # UINT8
    1: ("b", 1),   # INT8
    2: ("H", 2),   # UINT16
    3: ("h", 2),   # INT16
    4: ("I", 4),   # UINT32
    5: ("i", 4),   # INT32
    6: ("f", 4),   # FLOAT32
    7: ("?", 1),   # BOOL
    10: ("Q", 8),  # UINT64
    11: ("q", 8),  # INT64
    12: ("d", 8),  # FLOAT64
}
_STRING_TYPE = 8
_ARRAY_TYPE = 9


class GGUFParseError(RuntimeError):
    pass


def _read_u32(f: BinaryIO) -> int:
    raw = f.read(4)
    if len(raw) != 4:
        raise GGUFParseError("Unexpected end of file while reading a uint32.")
    return struct.unpack("<I", raw)[0]


def _read_u64(f: BinaryIO) -> int:
    raw = f.read(8)
    if len(raw) != 8:
        raise GGUFParseError("Unexpected end of file while reading a uint64.")
    return struct.unpack("<Q", raw)[0]


def _read_value(f: BinaryIO, value_type: int) -> Any:
    """Fully decode a value -- used only for keys we actually care about."""
    if value_type == _STRING_TYPE:
        length = _read_u64(f)
        return f.read(length).decode("utf-8", errors="replace")
    if value_type == _ARRAY_TYPE:
        element_type = _read_u32(f)
        count = _read_u64(f)
        return [_read_value(f, element_type) for _ in range(count)]
    scalar = _SCALAR_TYPES.get(value_type)
    if scalar is None:
        raise GGUFParseError(f"Unsupported GGUF value type: {value_type}")
    fmt, size = scalar
    raw = f.read(size)
    if len(raw) != size:
        raise GGUFParseError("Unexpected end of file while reading a scalar value.")
    return struct.unpack("<" + fmt, raw)[0]


def _skip_value(f: BinaryIO, value_type: int) -> None:
    """Advance past a value without materializing it (tokenizer vocab arrays can hold
    100k+ strings; we only ever need a handful of scalar architecture fields)."""
    if value_type == _STRING_TYPE:
        length = _read_u64(f)
        f.seek(length, 1)
        return
    if value_type == _ARRAY_TYPE:
        element_type = _read_u32(f)
        count = _read_u64(f)
        if element_type in _SCALAR_TYPES:
            f.seek(_SCALAR_TYPES[element_type][1] * count, 1)
        else:
            for _ in range(count):
                _skip_value(f, element_type)
        return
    scalar = _SCALAR_TYPES.get(value_type)
    if scalar is None:
        raise GGUFParseError(f"Unsupported GGUF value type: {value_type}")
    f.seek(scalar[1], 1)


def read_metadata(path: Path, wanted_keys: set[str] | None = None) -> dict[str, Any]:
    """Read the GGUF metadata key/value section.

    When `wanted_keys` is given, only those keys are decoded into the result; every
    other key is skipped (still correctly, just without allocating its value) so this
    stays fast even on models with huge tokenizer vocabularies. Stops right after the
    metadata section -- tensor info and tensor data are never read.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise GGUFParseError(f"Not a GGUF file (bad magic {magic!r}): {path}")
        version = _read_u32(f)
        if version < 2:
            raise GGUFParseError(f"Unsupported GGUF version {version}: {path}")
        _tensor_count = _read_u64(f)
        kv_count = _read_u64(f)
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_value(f, _STRING_TYPE)
            value_type = _read_u32(f)
            if wanted_keys is None or key in wanted_keys:
                metadata[key] = _read_value(f, value_type)
            else:
                _skip_value(f, value_type)
        return metadata
