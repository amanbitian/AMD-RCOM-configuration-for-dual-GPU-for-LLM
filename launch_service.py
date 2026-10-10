"""One-shot, on-demand launcher for the dual-GPU web service (app.py).

This is NOT an always-on daemon and does not itself accept HTTP requests -- a client
codebase runs it as a command (e.g. `subprocess.run([sys.executable, "launch_service.py", ...])`
from any language) to make sure the service is up before it starts calling the Client
Integration API described in CLIENT_API.md. There is no way to trigger a start over HTTP,
because if the service isn't running there's nothing listening to receive that request --
this script exists specifically to close that gap from outside the HTTP layer.

Behavior:
  - If the service is already running and healthy at --host:--port, this exits immediately
    (status 0) without touching it -- safe to call before every session.
  - Otherwise, it starts app.py as a detached background process (it keeps running after
    this launcher exits), waits until it reports healthy, then exits. The launcher itself
    does not stay resident.

Exit code 0 with a JSON summary on stdout means "the service is up, here's its base_url."
Exit code 1 with a JSON error on stdout means it could not confirm the service came up --
check the log file path included in that error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest


def _probe(base_url: str, timeout: float = 2.0) -> dict | None:
    """Returns the /api/status payload if base_url is our service and healthy, else None."""
    try:
        with urlrequest.urlopen(f"{base_url}/api/status", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, OSError, ValueError):
        return None
    if not isinstance(payload, dict) or "state" not in payload:
        return None
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the dual-GPU web service on demand if it isn't already running."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("example.dual_gpu.toml")),
        help="Path to the dual-GPU TOML configuration, forwarded to app.py.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host app.py should bind to if started.")
    parser.add_argument("--port", type=int, default=8090, help="Port app.py should bind to if started.")
    parser.add_argument("--data-dir", default="./chat_runs", help="Data directory forwarded to app.py.")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open a browser tab once started. Off by default since this is meant for programmatic starts.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the service to report healthy after spawning it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    probe_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    base_url = f"http://{probe_host}:{args.port}"

    status = _probe(base_url)
    if status is not None:
        print(json.dumps({"already_running": True, "base_url": base_url, "status": status}, indent=2))
        return 0

    app_path = Path(__file__).with_name("app.py")
    command = [
        sys.executable,
        str(app_path),
        "--config", args.config,
        "--host", args.host,
        "--port", str(args.port),
        "--data-dir", args.data_dir,
    ]
    if not args.open_browser:
        command.append("--no-browser")

    log_dir = Path(args.data_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = (Path(args.config).expanduser().resolve().parent / log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "launcher.log"

    creationflags = 0
    if sys.platform == "win32":
        # DETACHED_PROCESS: no console, so it survives this launcher exiting.
        # CREATE_NEW_PROCESS_GROUP: isolates it from signals sent to this launcher's group.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} :: launch_service.py :: {' '.join(command)} ===\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )

    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(json.dumps({
                "started": False,
                "error": f"app.py exited during startup with code {process.returncode}; see {log_path}",
            }))
            return 1
        status = _probe(base_url)
        if status is not None:
            print(json.dumps({
                "started": True,
                "pid": process.pid,
                "base_url": base_url,
                "status": status,
                "log_path": str(log_path),
            }, indent=2))
            return 0
        time.sleep(1.0)

    print(json.dumps({
        "started": False,
        "error": f"Service did not become healthy within {args.startup_timeout}s; see {log_path}",
    }))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
