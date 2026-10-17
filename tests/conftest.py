"""Shared test fixtures.

`write_gguf` builds tiny synthetic GGUF files with just the metadata keys a test needs,
so the suite never depends on any real model file being present on disk (a real GGUF
carries gigabytes of tensor data we'd never touch anyway -- the reader only looks at the
metadata header). This mirrors the on-disk format closely enough that
dual_gpu_setup.gguf.read_metadata reads these fixtures exactly like a real file.
"""

from __future__ import annotations

import gc
import struct
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _release_sqlite_handles():
    # On Windows an unclosed sqlite3 connection (the `with connect() as c` pattern commits
    # but does not close) keeps the db file locked, which breaks pytest's tmp-dir cleanup for
    # later tests. Forcing a collection after each test finalizes those connections in time.
    yield
    gc.collect()


class _GuardedControl:
    """Process control as the app sees it while testing. The example config's lane ports are the
    real ones (8081/8082), so a test that unloads or loads would otherwise find -- and stop -- a
    llama-server the developer is running there. No test may see or stop a real server."""

    def __init__(self, real):
        self._real = real

    def pid_listening_on(self, port):
        return None

    def terminate_tree(self, pid, grace_s=10.0):
        raise AssertionError(f"a test tried to terminate real process {pid}; use a fake control")

    def __getattr__(self, name):
        return getattr(self._real, name)


class _GuardedHost:
    def __init__(self, real):
        self._real = real
        self.control = _GuardedControl(real.control)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture(autouse=True)
def _never_touch_real_servers(monkeypatch):
    """Tests that exercise a fake control override this with their own monkeypatch."""
    import app
    import dual_gpu_setup.server as server_module

    guarded = _GuardedHost(app.get_host())
    monkeypatch.setattr(app, "get_host", lambda: guarded)
    monkeypatch.setattr(server_module, "get_host", lambda: guarded)


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


import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from urllib import error as urlerror, request as urlrequest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_EXAMPLE = Path(__file__).resolve().parent.parent / "example.dual_gpu.toml"


class LiveApp:
    """A real ApplicationState + HTTP server on an ephemeral port, with JSON helpers."""

    def __init__(self, state, server, thread):
        self.state, self.server, self.thread = state, server, thread
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
        body = json.dumps(payload).encode() if payload is not None else None
        req = urlrequest.Request(self.base + path, data=body, method=method,
                                 headers={"Content-Type": "application/json"} if body else {})
        try:
            with urlrequest.urlopen(req, timeout=10) as response:
                raw = response.read()
                return response.status, (json.loads(raw) if response.headers.get_content_type() == "application/json" else {"raw": raw})
        except urlerror.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def get(self, path: str):
        return self.call("GET", path)

    def post(self, path: str, payload: dict | None = None):
        return self.call("POST", path, payload or {})


@pytest.fixture
def live_app(tmp_path: Path):
    """Factory: live_app(extra_toml="") -> LiveApp. Extra TOML is appended to the example config."""
    created: list[LiveApp] = []

    def make(extra_toml: str = "") -> LiveApp:
        import app

        config = tmp_path / f"config{len(created)}.toml"
        config.write_text(_EXAMPLE.read_text(encoding="utf-8") + "\n" + extra_toml, encoding="utf-8")
        state = app.ApplicationState(config, tmp_path / f"data{len(created)}")
        server = app.ChatbotHTTPServer(("127.0.0.1", 0), state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        live = LiveApp(state, server, thread)
        state.benchmarks.server_url = live.base
        created.append(live)
        return live

    yield make
    for live in created:
        live.server.shutdown()
        live.server.server_close()
        if live.state.eval_worker is not None:
            live.state.eval_worker.stop()
        live.state.manager.shutdown()
        live.state.close()  # stop usage/history writer threads + flush
        live.thread.join(timeout=5)
