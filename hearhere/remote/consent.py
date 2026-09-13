"""Persistent record of the user's consent to upload audio to a remote worker.

The remote backend is the one path where recordings leave the machine, so the
CLI asks for confirmation *before the first upload* and remembers the answer in
a small marker file. This module only stores/reads that marker; the interactive
prompt and the always-on warning live in the CLI and client respectively.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..logging_setup import get_logger

log = get_logger("remote.consent")

CONSENT_FILENAME = ".remote_upload_consent"


def consent_marker() -> Path:
    """Location of the consent marker (under the user config dir)."""
    return Path.home() / ".config" / "hearhere" / CONSENT_FILENAME


def has_consent() -> bool:
    """Whether the user has already consented to remote uploads."""
    return consent_marker().is_file()


def record_consent(url: str = "") -> Path:
    """Record consent to upload; returns the marker path."""
    marker = consent_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        f"consented={datetime.now().isoformat()}\nurl={url}\n", encoding="utf-8"
    )
    log.debug("Recorded remote-upload consent at %s", marker)
    return marker
