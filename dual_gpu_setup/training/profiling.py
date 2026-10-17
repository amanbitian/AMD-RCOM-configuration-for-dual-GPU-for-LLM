"""rocprofv3 helpers: wrap a launch command and attach the resulting trace to a run.

Profiling is deliberately NOT continuous (it carries real overhead). Capture one short window, then
`tracker.attach_artifact(trace_dir, kind="rocprof_trace")` links it to the run for download.
"""

from __future__ import annotations

import shutil
from pathlib import Path

DEFAULT_TRACES = ("--kernel-trace", "--hip-trace", "--memory-copy-trace")


def rocprofv3_command(command: list[str], output_dir: str | Path, *, traces: tuple[str, ...] = DEFAULT_TRACES,
                      output_format: str = "pftrace", executable: str | None = None, extra: tuple[str, ...] = ()) -> list[str]:
    """`rocprofv3 --kernel-trace ... -d <dir> -f pftrace -- <command>` (a leading `--` separates tool flags)."""
    binary = executable or shutil.which("rocprofv3") or "rocprofv3"
    return [binary, *traces, "--output-format", output_format, "-d", str(output_dir), *extra, "--", *command]


def find_traces(output_dir: str | Path) -> list[Path]:
    """Trace files rocprofv3 wrote (Perfetto .pftrace, .json, .csv, .db), newest first."""
    root = Path(output_dir)
    found = [p for pattern in ("*.pftrace", "*.json", "*.csv", "*.db") for p in root.rglob(pattern) if p.is_file()]
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)
