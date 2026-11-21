"""Correctness canary for a model split across both GPUs.

Splitting a model buys capacity, not speed, and on some llama.cpp builds the split output is silently corrupt
(the b11699 ROCm build produced garbage). A server that loaded and answers /health proves nothing about that, so
after a split load a fixed prompt is sent twice at temperature 0:

  * greedy decoding must be deterministic (two different answers = unstable compute across the cards);
  * the answer must not be degenerate (empty, mostly non-text bytes, or one repeated token).

Whether the answer is the *expected* one is reported but not required, since a base or reasoning model may word
it differently. Policy `split_canary` = "off" | "warn" (default: record the failure) | "refuse" (stop the server).
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib import request as urlrequest

CANARY_PROMPT = "What is 12 plus 15? Reply with just the number."
EXPECTED = "27"


def _ask(base_url: str, model: str, timeout: float) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": CANARY_PROMPT}], "temperature": 0,
                       "top_k": 1, "seed": 1, "max_tokens": 24, "stream": False}).encode()
    req = urlrequest.Request(f"{base_url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - loopback llama-server
        message = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]
    return str(message.get("content") or message.get("reasoning_content") or "")


def degenerate(text: str) -> str | None:
    """Why this answer looks like corrupt output, or None."""
    if not text.strip():
        return "empty answer"
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    if printable / len(text) < 0.9:
        return "mostly non-text characters"
    tokens = re.findall(r"\S+", text)
    if len(tokens) >= 8 and len(set(tokens)) <= max(1, len(tokens) // 8):
        return "one token repeated"
    if len(text) >= 8 and len(set(text)) == 1:
        return "one character repeated"
    return None


def run_split_canary(base_url: str, model: str, timeout: float = 60.0) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "expected_answer_seen": False, "detail": ""}
    try:
        first, second = _ask(base_url, model, timeout), _ask(base_url, model, timeout)
    except Exception as exc:  # noqa: BLE001 - an unreachable or erroring server fails the canary
        result["detail"] = f"canary request failed: {type(exc).__name__}: {exc}"
        return result
    result["sample"] = first[:120]
    result["expected_answer_seen"] = EXPECTED in first
    problem = degenerate(first) or degenerate(second)
    if problem:
        result["detail"] = f"split output looks corrupt ({problem}): {first[:80]!r}"
    elif first != second:
        result["detail"] = f"greedy output is not deterministic across two identical requests: {first[:60]!r} vs {second[:60]!r}"
    else:
        result["ok"] = True
        result["detail"] = "deterministic, coherent output"
    return result
