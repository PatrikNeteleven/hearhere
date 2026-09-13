"""Logging setup for HearHere.

Two sinks: a console handler for interactive use, and (optionally) a per-meeting
``hearhere.log`` file inside the meeting folder so each meeting keeps its own
processing record.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "hearhere"
LOG_FILENAME = "hearhere.log"

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(module)s:%(lineno)d]: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the HearHere logger (or a child of it)."""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Configure console logging once. Idempotent."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
    return logger


def add_meeting_file_handler(
    meeting_dir: str | Path, level: int | str = logging.DEBUG
) -> logging.FileHandler:
    """Attach a file handler writing ``hearhere.log`` inside ``meeting_dir``.

    Returns the handler so the caller can :meth:`remove_meeting_file_handler`
    it when processing for that meeting finishes.
    """
    logger = setup_logging()
    log_path = Path(meeting_dir) / LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATEFMT))
    handler.set_name(f"meeting:{log_path}")
    logger.addHandler(handler)
    return handler


def remove_meeting_file_handler(handler: logging.FileHandler) -> None:
    """Detach and close a per-meeting file handler."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.removeHandler(handler)
    handler.close()
