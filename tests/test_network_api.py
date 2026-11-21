from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib import request

import app
import launch_service
from launch_service import build_parser


def _request(base: str, path: str, *, method: str = "GET", origin: str | None = None):
    headers = {"Origin": origin} if origin else {}
    req = request.Request(base + path, method=method, headers=headers)
    with request.urlopen(req, timeout=5) as response:
        return response.status, response.headers, json.loads(response.read() or b"{}")


def test_health_and_bootstrap_advertise_lan_contract(tmp_path: Path):
    config_path = Path(__file__).resolve().parent.parent / "example.dual_gpu.toml"
    state = app.ApplicationState(config_path, tmp_path / "data")
    server = app.ChatbotHTTPServer(
        ("127.0.0.1", 0), state, cors_origins=["http://client.example:3000"]
    )
    state.network = app.network_metadata("0.0.0.0", 8090)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, headers, health = _request(base, "/api/health", origin="http://client.example:3000")
        assert status == 200
        assert health["api_version"] == app.API_VERSION
        assert health["ready"] is False
        assert health["network"]["lan_enabled"] is True
        assert health["network"]["openai_base_url"].endswith("/v1")
        assert health["network"]["advertised_openai_base_urls"] == [
            f"{url}/v1" for url in health["network"]["advertised_urls"]
        ]
        assert headers["X-DGPU-API-Version"] == app.API_VERSION
        assert headers["Access-Control-Allow-Origin"] == "http://client.example:3000"

        _, _, bootstrap = _request(base, "/api/bootstrap")
        assert bootstrap["api_version"] == app.API_VERSION
        assert bootstrap["network"] == health["network"]

        _, _, status = _request(base, "/api/status")
        assert status["api_version"] == app.API_VERSION
        assert status["network"] == health["network"]
    finally:
        server.shutdown()
        server.server_close()
        state.manager.shutdown()
        state.close()
        thread.join(timeout=5)


def test_cors_preflight_and_unconfigured_origin():
    state = SimpleNamespace(health=lambda: {"status": "ok", "api_version": app.API_VERSION})
    server = app.ChatbotHTTPServer(("127.0.0.1", 0), state, cors_origins=["http://allowed.example"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        req = request.Request(
            base + "/api/health",
            method="OPTIONS",
            headers={
                "Origin": "http://allowed.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type, authorization",
            },
        )
        with request.urlopen(req, timeout=5) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == "http://allowed.example"
            assert "POST" in response.headers["Access-Control-Allow-Methods"]
            assert "X-Lease-Token" in response.headers["Access-Control-Allow-Headers"]
            assert response.headers["X-DGPU-API-Version"] == app.API_VERSION
            assert response.headers["Access-Control-Max-Age"] == "600"

        # The endpoint still works, but an origin that was not explicitly configured receives no
        # CORS permission and therefore cannot be used by a browser application.
        _, headers, _ = _request(base, "/api/health", origin="http://blocked.example")
        assert headers.get("Access-Control-Allow-Origin") is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_launch_parser_supports_lan_and_repeated_cors_origins():
    args = build_parser().parse_args(
        ["--start", "--lan", "--cors-origin", "http://a", "--cors-origin", "http://b"]
    )
    assert args.lan is True
    assert args.cors_origin == ["http://a", "http://b"]


def test_default_route_address_is_advertised_first(monkeypatch):
    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def connect(self, _address):
            return None

        def getsockname(self):
            return ("192.168.50.20", 12345)

    monkeypatch.setattr(app.socket, "socket", lambda *_args, **_kwargs: Probe())
    monkeypatch.setattr(
        app.socket,
        "gethostbyname_ex",
        lambda _host: ("gpu-host", [], ["10.0.0.8", "192.168.50.20"]),
    )

    info = app.network_metadata("0.0.0.0", 8090)

    assert info["advertised_urls"][0] == "http://192.168.50.20:8090"
    assert info["openai_base_url"] == "http://192.168.50.20:8090/v1"


def test_launcher_probe_requires_service_identity(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(self.payload).encode()

    monkeypatch.setattr(
        launch_service.urlrequest,
        "urlopen",
        lambda *_args, **_kwargs: Response({"service": "something-else", "state": "ready"}),
    )
    assert launch_service._probe("http://127.0.0.1:8090") is None

    monkeypatch.setattr(
        launch_service.urlrequest,
        "urlopen",
        lambda *_args, **_kwargs: Response({"service": "dual-gpu-studio", "state": "ready"}),
    )
    assert launch_service._probe("http://127.0.0.1:8090")["state"] == "ready"
