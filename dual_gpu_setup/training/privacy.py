"""Keep personal data (resumes, candidate text, evaluation examples) out of general telemetry.

Structured metrics pass through untouched. Free-form payloads (events, eval details, per-example
extras) have content-bearing keys dropped, long strings truncated, and e-mail/phone-like strings
redacted, so a careless `tracker.event("bad_example", text=resume)` cannot leak into the database.
"""

from __future__ import annotations

import re
from typing import Any

CONTENT_KEYS = frozenset({
    "prompt", "completion", "response", "text", "resume", "resume_text", "cv", "document", "documents",
    "example", "examples", "raw", "messages", "content", "input", "output", "transcript", "candidate",
    "job_description", "jd", "body", "email_body", "generated", "generation", "label_text", "answer",
})
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_LONG_ID = re.compile(r"(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)")
MAX_ITEMS = 200


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")


def _phone(match: re.Match[str]) -> str:
    text = match.group()
    return text if len(re.sub(r"\D", "", text)) < 9 else "[phone]"  # short counters/versions are not phone numbers


def redact_string(value: str, max_length: int | None) -> str:
    value = _EMAIL.sub("[email]", value)
    value = _LONG_ID.sub("[id]", value)
    protected: list[str] = []  # dates and timestamps contain enough digits to look like phone numbers

    def stash(match: re.Match[str]) -> str:
        protected.append(match.group())
        return f"\x00{len(protected) - 1}\x00"

    value = _DATE.sub(stash, value)
    value = _PHONE.sub(_phone, value)
    for index, original in enumerate(protected):
        value = value.replace(f"\x00{index}\x00", original)
    if max_length is not None and len(value) > max_length:
        value = value[:max_length] + "…[truncated]"
    return value


def sanitize(value: Any, *, drop_content_keys: bool = True, max_string: int | None = 500, _depth: int = 0) -> Any:
    """Recursively sanitize a JSON-like value. Non-JSON scalars are stringified."""
    if _depth > 6:
        return "[depth limit]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_string(value, max_string)
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in list(value.items())[:MAX_ITEMS]:
            key = str(key)
            if drop_content_keys and key.lower() in CONTENT_KEYS:
                clean[key] = "[redacted]"
            else:
                clean[key] = sanitize(item, drop_content_keys=drop_content_keys, max_string=max_string, _depth=_depth + 1)
        return clean
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item, drop_content_keys=drop_content_keys, max_string=max_string, _depth=_depth + 1)
                for item in list(value)[:MAX_ITEMS]]
    return redact_string(str(value), max_string)
