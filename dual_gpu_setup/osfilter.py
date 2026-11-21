"""The dashboard's OS toggle: All / Windows / Ubuntu.

Rows are tagged with `platform.system()` of the install that recorded them ("Windows" or "Linux"); Ubuntu is
the Linux install, so "ubuntu" is accepted as another name for it.
"""

from __future__ import annotations

OS_NAMES = {"windows": "Windows", "win": "Windows", "linux": "Linux", "ubuntu": "Linux"}


def normalize_os(value: str | None) -> str | None:
    """None for all operating systems, otherwise the stored tag ("Windows" / "Linux").

    Raises ValueError for a name it does not know, so a typo is reported rather than silently showing nothing."""
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in ("", "all", "any"):
        return None
    if key in OS_NAMES:
        return OS_NAMES[key]
    raise ValueError(f"os must be all, windows or ubuntu/linux, not {value!r}")
