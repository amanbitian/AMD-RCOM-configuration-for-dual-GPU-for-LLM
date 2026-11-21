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
  - If the service is not running, this is status-only unless the caller passes --start.
    This prevents login-time clients and other background integrations from silently
    starting the service and loading GPU models after a reboot.
  - With --start, it starts app.py as a detached background process (it keeps running
    after this launcher exits), waits until it reports healthy, then exits. The launcher
    itself does not stay resident.
  - Run from a terminal it opens the UI in your browser (also when the service was already
    running). Run by another program (no terminal attached) it stays silent. `--open-browser`
    and `--no-browser` override that.
  - Once the UI is open, restarting or stopping the service, unloading models and editing
    the policy settings are all done from its Service page.

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
import webbrowser
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from dual_gpu_setup.host_info import default_config_name, detached_process_kwargs, os_info, os_log_line


def _probe(base_url: str, timeout: float = 2.0) -> dict | None:
    """Return this service's health payload, with a legacy status fallback."""
    try:
        with urlrequest.urlopen(f"{base_url}/api/health", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if (
            isinstance(payload, dict)
            and payload.get("service") == "dual-gpu-studio"
            and "state" in payload
        ):
            return payload
        # A successful but differently identified health endpoint belongs to another service.
        return None
    except (urlerror.URLError, TimeoutError, OSError, ValueError):
        pass
    # Older Dual GPU Studio releases predate /api/health. Keep status probing for a rolling update.
    try:
        with urlrequest.urlopen(f"{base_url}/api/status", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and "state" in payload else None


def interactive() -> bool:
    """A person is at a terminal (as opposed to a program launching this with pipes)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def wants_browser(args: argparse.Namespace, is_interactive: bool) -> bool:
    if args.open_browser:
        return True
    if args.no_browser:
        return False
    return is_interactive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the dual-GPU web service on demand if it isn't already running."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name(default_config_name())),
        help="Path to the dual-GPU TOML configuration, forwarded to app.py.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host app.py should bind to if started.")
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Start app.py with trusted-LAN access (equivalent to --host 0.0.0.0).",
    )
    parser.add_argument("--port", type=int, default=8090, help="Port app.py should bind to if started.")
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="Allow a browser app origin to call the API; repeat for multiple origins.",
    )
    parser.add_argument("--data-dir", default="./chat_runs", help="Data directory forwarded to app.py.")
    parser.add_argument(
        "--start",
        action="store_true",
        help="Explicitly permit starting the service when it is not already running.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Always open the UI in a browser tab. Default: open it when run from a terminal.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Never open a browser tab, even from a terminal.",
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
    if args.lan and args.host in {"127.0.0.1", "localhost", "::1"}:
        args.host = "0.0.0.0"
    probe_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    base_url = f"http://{probe_host}:{args.port}"

    open_browser = wants_browser(args, interactive())
    status = _probe(base_url)
    if status is not None:
        network = status.get("network") if isinstance(status, dict) else None
        if args.lan and isinstance(network, dict) and not network.get("lan_enabled"):
            print(json.dumps({
                "started": False,
                "already_running": True,
                "base_url": base_url,
                "error": "A localhost-only service is already using this port; restart it with --lan.",
            }, indent=2))
            return 1
        if open_browser:
            webbrowser.open(base_url)
        print(json.dumps({"already_running": True, "base_url": base_url, "status": status,
                          "host_os": os_info(), "browser_opened": open_browser}, indent=2))
        return 0

    if not args.start:
        print(json.dumps({
            "started": False,
            "start_blocked": True,
            "base_url": base_url,
            "error": "Service is not running. Pass --start to start it explicitly.",
        }, indent=2))
        return 2

    app_path = Path(__file__).with_name("app.py")
    command = [
        sys.executable,
        str(app_path),
        "--config", args.config,
        "--host", args.host,
        "--port", str(args.port),
        "--data-dir", args.data_dir,
    ]
    for origin in args.cors_origin:
        command.extend(["--cors-origin", origin])
    command.append("--no-browser")  # this launcher opens the tab itself, once the service is healthy

    log_dir = Path(args.data_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = (Path(args.config).expanduser().resolve().parent / log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "launcher.log"

    with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} :: launch_service.py :: {' '.join(command)} ===\n")
        log_handle.write(f"host_os={os_log_line()}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **detached_process_kwargs(),
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
            if open_browser:
                webbrowser.open(base_url)
            print(json.dumps({
                "started": True,
                "pid": process.pid,
                "base_url": base_url,
                "status": status,
                "log_path": str(log_path),
                "host_os": os_info(),
                "browser_opened": open_browser,
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
