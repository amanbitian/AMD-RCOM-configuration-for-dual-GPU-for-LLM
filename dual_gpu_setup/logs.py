"""Service logging that never deletes a log line: files rotate by size into timestamped, gzip-compressed
archives that are kept forever, and routine successful GET polls (the dashboard refreshing every 2 s, which
once filled 30 MB with 347,000 near-identical lines) are not logged at all. Errors and every non-GET
request still are."""

from __future__ import annotations

import gzip
import logging
import shutil
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "dual_gpu"


class KeepEverythingRotatingHandler(RotatingFileHandler):
    """Like RotatingFileHandler, but a full file becomes `<name>.<timestamp>.gz` instead of being shifted
    through a fixed number of numbered backups (which silently deletes the oldest)."""

    def __init__(self, filename: str | Path, max_bytes: int = 20 * 1024 * 1024):
        super().__init__(filename, maxBytes=max_bytes, backupCount=1, encoding="utf-8")

    def doRollover(self) -> None:  # noqa: N802 - stdlib name
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]
        source = Path(self.baseFilename)
        target = source.with_name(f"{source.name}.{time.strftime('%Y%m%dT%H%M%S')}.gz")
        counter = 1
        while target.exists():  # two rotations in one second must not overwrite each other
            target = source.with_name(f"{source.name}.{time.strftime('%Y%m%dT%H%M%S')}-{counter}.gz")
            counter += 1
        try:
            with open(source, "rb") as raw, gzip.open(target, "wb") as packed:
                shutil.copyfileobj(raw, packed)
            source.write_bytes(b"")  # emptied only after the compressed copy is complete
        except OSError:
            pass  # keep writing to the same file rather than lose lines
        if not self.delay:
            self.stream = self._open()


def setup_service_logging(data_dir: Path | str, max_bytes: int = 20 * 1024 * 1024) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if any(isinstance(h, KeepEverythingRotatingHandler) for h in logger.handlers):
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = KeepEverythingRotatingHandler(log_dir / "service.log", max_bytes)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream)
    return logger


def is_routine_poll(request_line: str, status: object) -> bool:
    """A successful GET is a poll or page load, not an event worth a log line."""
    return request_line.startswith("GET ") and str(status).startswith(("2", "3"))
