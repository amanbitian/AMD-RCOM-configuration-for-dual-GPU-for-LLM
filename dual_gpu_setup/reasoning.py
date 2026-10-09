"""Inspect native effort choices without turning them into arbitrary token caps."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from dual_gpu_setup.gguf import GGUFParseError, read_metadata


def template_reasoning(template: str, architecture: str = "") -> dict[str, Any]:
    levels: list[str] = []
    default = None
    # Conservative recognition: only publish literal choices from a template's
    # effort allow-list. Unknown templates remain usable with their native default.
    match = re.search(r"\b\w*reasoning_effort\s+(?:not\s+)?in\s*[\[(]([^\])]+)[\])]", template)
    if match:
        levels = re.findall(r"['\"]([a-z]+)['\"]", match.group(1))
    elif architecture == "gpt-oss" and "reasoning_effort" in template:
        levels = ["low", "medium", "high"]
    match = re.search(r"reasoning_effort\s*\|\s*default\(['\"]([a-z]+)['\"]\)", template)
    if not match:
        match = re.search(r"set\s+reasoning_effort\s*=\s*['\"]([a-z]+)['\"]", template)
    if match:
        default = match.group(1)
    order = ["minimal", "low", "medium", "high", "xhigh", "max"]
    levels = sorted(set(levels), key=lambda level: order.index(level) if level in order else len(order))
    return {"efforts": ["default", *levels], "default_effort": default,
            "budget_tokens": None, "budget_policy": "model_controlled",
            "source": "gguf_template" if template else "unavailable"}


@lru_cache(maxsize=128)
def _file_reasoning(path: str, size: int, modified_ns: int) -> dict[str, Any]:
    metadata = read_metadata(Path(path), {"general.architecture", "tokenizer.chat_template"})
    return template_reasoning(str(metadata.get("tokenizer.chat_template", "")),
                              str(metadata.get("general.architecture", "")))


def model_reasoning(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        result = _file_reasoning(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        return {**result, "efforts": list(result["efforts"])}
    except (OSError, GGUFParseError, ValueError):
        return template_reasoning("")


def coding_request(payload: dict[str, Any], *, capabilities: dict[str, Any],
                   default_effort: str, server_budget: int, server_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    # llama-server treats request budget=-1 as "inherit server budget", so a
    # capped deployment cannot be made unlimited with a request override.
    if server_budget != -1 or server_mode == "off":
        raise ValueError("Reload this model in Coding agent mode (reasoning auto, budget -1) to use uncapped native reasoning.")
    result = dict(payload)
    for key in ("reasoning_budget", "reasoning_budget_tokens", "thinking_budget_tokens"):
        if key in result and result[key] != -1:
            raise ValueError("Coding mode uses model-controlled reasoning; remove the numeric thinking budget.")
        result.pop(key, None)
    kwargs = dict(result.get("chat_template_kwargs") or {})
    effort = result.get("reasoning_effort", kwargs.get("reasoning_effort", default_effort)) or "default"
    if effort not in capabilities["efforts"] and effort != "none":
        raise ValueError(f"Unsupported reasoning effort '{effort}'. Available: {', '.join(capabilities['efforts'])}.")
    kwargs.pop("reasoning_effort", None)
    if effort == "default":
        result.pop("reasoning_effort", None)
    else:
        result["reasoning_effort"] = effort
    if "chat_template_kwargs" in result:
        result["chat_template_kwargs"] = kwargs
    disabled = effort == "none" or kwargs.get("enable_thinking") is False
    metadata = {"effort": effort, "effective_effort": (
                    "none" if disabled else capabilities["default_effort"] if effort == "default" else effort),
                "budget_tokens": None, "budget_policy": "model_controlled",
                "server_budget_tokens": server_budget,
                "output_limit_tokens": result.get("max_completion_tokens", result.get("max_tokens")),
                "capability_source": capabilities["source"]}
    return result, metadata
