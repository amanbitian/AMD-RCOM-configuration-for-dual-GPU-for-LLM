"""Service settings editable from the UI: what can be changed, validation, and persistence.

Values are applied to the running config by the caller. `write_policy` saves them into the
`[policy]` table of the TOML file *in place*: comments, key order and every other table stay as
written, and nothing is written unless the result still parses and holds the requested values.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import tomllib
from pathlib import Path
from typing import Any

# apply: "live" takes effect immediately; "next_load" changes the command line of the next model
# load, so a model that is already loaded keeps its old behaviour until it is reloaded.
SETTINGS: list[dict[str, Any]] = [
    {"key": "load_mode", "type": "choice", "choices": ["none", "mmap", "auto"], "apply": "next_load",
     "label": "Weight loading",
     "help": "none keeps model weights out of system RAM once they are on the GPU. mmap/auto leave a "
             "resident copy about the size of the model."},
    {"key": "cache_ram_mib", "type": "int", "min": 0, "max": 65536, "apply": "next_load",
     "label": "Prompt cache in RAM (MiB)",
     "help": "0 turns it off. llama.cpp's own default is 8192 per server and grows with every new prompt. "
             "A conversation still reuses its own context from the GPU."},
    {"key": "vram_admission", "type": "bool", "apply": "live",
     "label": "Require free VRAM before loading",
     "help": "Refuse to start a model that does not fit the VRAM actually free on its GPU, instead of "
             "letting Windows back it with system RAM."},
    {"key": "vram_headroom_gb", "type": "float", "min": 0.0, "max": 16.0, "apply": "live",
     "label": "VRAM headroom (GB)",
     "help": "Extra free VRAM required beyond the model and its KV cache."},
    {"key": "auto_load_models", "type": "bool", "apply": "live",
     "label": "Load models on demand",
     "help": "A gateway request that names a model loads it, replacing the least recently used idle model "
             "when the GPUs are full. A model that is answering a request is never stopped."},
    {"key": "residency_guard", "type": "bool", "apply": "next_load",
     "label": "Keep models in VRAM",
     "help": "Windows can move an idle model's VRAM into system RAM. The guard pages it back in with a "
             "1-token request on a reserved slot. Servers started with it run two slots."},
    {"key": "residency_check_seconds", "type": "float", "min": 5.0, "max": 600.0, "apply": "live",
     "label": "Residency check interval (s)", "help": "How often idle models are checked."},
    {"key": "drain_timeout_seconds", "type": "float", "min": 0.0, "max": 600.0, "apply": "live",
     "label": "Swap drain timeout (s)",
     "help": "How long replacing a model waits for its running requests to finish."},
    {"key": "kill_busy_ports", "type": "bool", "apply": "live",
     "label": "Stop stale servers on lane ports",
     "help": "When loading, stop a leftover llama-server that holds a lane's port."},
]
BY_KEY = {spec["key"]: spec for spec in SETTINGS}


def describe(policy: Any) -> list[dict[str, Any]]:
    """Every editable setting with its current value, for the UI."""
    return [{**spec, "value": getattr(policy, spec["key"])} for spec in SETTINGS]


def coerce(spec: dict[str, Any], value: Any) -> Any:
    key, kind = spec["key"], spec["type"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false.")
        return value
    if kind == "choice":
        if value not in spec["choices"]:
            raise ValueError(f"{key} must be one of: {', '.join(spec['choices'])}.")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a number.")
    if kind == "int":
        if value != int(value):
            raise ValueError(f"{key} must be a whole number.")
        value = int(value)
    else:
        value = float(value)
    if not spec["min"] <= value <= spec["max"]:
        raise ValueError(f"{key} must be between {spec['min']:g} and {spec['max']:g}.")
    return value


def validate(updates: Any) -> dict[str, Any]:
    if not isinstance(updates, dict) or not updates:
        raise ValueError("Send an object of setting names and values.")
    unknown = sorted(set(updates) - set(BY_KEY))
    if unknown:
        raise ValueError(f"Unknown setting(s): {', '.join(unknown)}.")
    return {key: coerce(BY_KEY[key], value) for key, value in updates.items()}


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


_TABLE_HEADER = re.compile(r"^\s*\[")
_POLICY_HEADER = re.compile(r"^\s*\[policy\]\s*(#.*)?$")


def edit_policy_text(text: str, values: dict[str, Any]) -> str:
    """`text` with each key set inside `[policy]`: an existing assignment keeps its trailing comment,
    a missing key is added at the end of the table, a missing table is appended."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if _POLICY_HEADER.match(line)), None)
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("[policy]")
        start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines)) if _TABLE_HEADER.match(lines[i])), len(lines))
    for key, value in values.items():
        literal = _literal(value)
        assignment = re.compile(rf'^(\s*{re.escape(key)}\s*=\s*)("[^"]*"|[^\s#]+)(.*)$')
        for i in range(start + 1, end):
            match = assignment.match(lines[i])
            if match:
                lines[i] = f"{match.group(1)}{literal}{match.group(3)}"
                break
        else:
            at = end
            while at > start + 1 and not lines[at - 1].strip():
                at -= 1
            lines.insert(at, f"{key} = {literal}")
            end += 1
    return newline.join(lines) + newline


def write_policy(path: Path, values: dict[str, Any]) -> Path | None:
    """Persist `values` into the TOML at `path`. Returns the one-time backup path if one was made.
    Raises ValueError, leaving the file untouched, if the edit would not parse or hold the values."""
    original = path.read_bytes().decode("utf-8")  # bytes: read_text would hide a CRLF file's line endings
    updated = edit_policy_text(original, values)
    try:
        saved = tomllib.loads(updated).get("policy", {})
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Refusing to write {path.name}: the edited file would not parse ({exc}).") from exc
    wrong = [key for key, value in values.items() if saved.get(key) != value]
    if wrong:
        raise ValueError(f"Refusing to write {path.name}: could not set {', '.join(wrong)}.")
    backup = path.with_name(path.name + ".bak")
    made = None
    if not backup.exists():
        shutil.copy2(path, backup)
        made = backup
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(updated, encoding="utf-8", newline="")
    os.replace(temp, path)
    return made
