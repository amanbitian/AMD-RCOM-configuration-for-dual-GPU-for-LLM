from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402


def _json(url: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    with request.urlopen(req, timeout=5) as response:
        return json.loads(response.read())


def test_project_session_tool_and_dashboard_endpoints(tmp_path: Path):
    config_path = Path(__file__).resolve().parent.parent / "example.dual_gpu.toml"
    state = app.ApplicationState(config_path, tmp_path / "data")
    server = app.ChatbotHTTPServer(("127.0.0.1", 0), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        bootstrap = _json(f"{base_url}/api/bootstrap")
        assert bootstrap["coding_presets"]["workload_kind"] == "coding_agent"

        project = _json(
            f"{base_url}/api/projects",
            {"name": "HTTP Smoke", "repo_path": "F:/Projects/Smoke"},
        )
        session = _json(
            f"{base_url}/api/agent-sessions",
            {"project_id": project["id"], "agent_role": "developer", "runtime": "cline"},
        )
        _json(
            f"{base_url}/api/tool-events",
            {
                "project_id": project["id"],
                "session_id": session["id"],
                "tool_type": "test",
                "status": "completed",
                "exit_code": 0,
            },
        )
        dashboard = _json(f"{base_url}/api/dashboard?project_id={project['id']}")
        models = _json(f"{base_url}/v1/r9700/developer/models")

        assert dashboard["tool_summary"]["events"] == 1
        assert models == {"object": "list", "data": []}
    finally:
        server.shutdown()
        server.server_close()
        state.manager.shutdown()
        thread.join(timeout=5)
