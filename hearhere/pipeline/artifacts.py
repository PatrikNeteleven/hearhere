"""Meeting-folder layout and create/resolve/read/write helpers.

A meeting is a self-contained folder under ``storage_dir``::

    2026-09-13_weekly-sync/
    ├── meeting.json          # canonical structured record
    ├── audio/
    │   ├── self.wav
    │   └── others.wav
    ├── transcript.md / .txt / .srt / ...
    ├── summary.md
    └── hearhere.log

Every export is derived from ``meeting.json``, so re-exporting never re-runs the
models.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from hearhere.models import Meeting

MEETING_JSON = "meeting.json"
AUDIO_DIR = "audio"
SELF_WAV = "self.wav"
OTHERS_WAV = "others.wav"
LOG_FILENAME = "hearhere.log"

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase ASCII slug: ``"Weekly Sync!"`` -> ``"weekly-sync"``."""
    normalized = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = _SLUG_STRIP.sub("-", normalized.lower()).strip("-")
    return slug or "meeting"


def meeting_id(title: str, when: date | datetime | None = None) -> str:
    """Build a folder id like ``2026-09-13_weekly-sync``."""
    when = when or datetime.now()
    day = when.date() if isinstance(when, datetime) else when
    return f"{day.isoformat()}_{slugify(title)}"


@dataclass(frozen=True)
class MeetingPaths:
    """Resolved paths for a single meeting folder."""

    root: Path

    @property
    def meeting_json(self) -> Path:
        return self.root / MEETING_JSON

    @property
    def audio_dir(self) -> Path:
        return self.root / AUDIO_DIR

    @property
    def self_wav(self) -> Path:
        return self.audio_dir / SELF_WAV

    @property
    def others_wav(self) -> Path:
        return self.audio_dir / OTHERS_WAV

    @property
    def log(self) -> Path:
        return self.root / LOG_FILENAME

    def export(self, name: str) -> Path:
        """Path for an export artifact, e.g. ``export("transcript.md")``."""
        return self.root / name


def resolve_meeting(path: str | Path) -> MeetingPaths:
    """Return the :class:`MeetingPaths` for an existing meeting folder."""
    root = Path(path).expanduser()
    return MeetingPaths(root=root)


def create_meeting_dir(
    storage_dir: str | Path,
    title: str,
    when: date | datetime | None = None,
    *,
    exist_ok: bool = False,
) -> MeetingPaths:
    """Create ``<storage_dir>/<id>/audio/`` and return its paths.

    Raises :class:`FileExistsError` if the folder exists and ``exist_ok`` is
    False.
    """
    root = Path(storage_dir).expanduser() / meeting_id(title, when)
    if root.exists() and not exist_ok:
        raise FileExistsError(f"Meeting folder already exists: {root}")
    (root / AUDIO_DIR).mkdir(parents=True, exist_ok=exist_ok)
    return MeetingPaths(root=root)


def write_meeting(paths: MeetingPaths, meeting: Meeting) -> Path:
    """Serialize ``meeting`` to ``meeting.json`` (pretty, UTF-8)."""
    paths.root.mkdir(parents=True, exist_ok=True)
    payload = meeting.model_dump(mode="json")
    paths.meeting_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths.meeting_json


def read_meeting(paths: MeetingPaths) -> Meeting:
    """Load and validate ``meeting.json`` into a :class:`Meeting`."""
    if not paths.meeting_json.is_file():
        raise FileNotFoundError(f"No {MEETING_JSON} in {paths.root}")
    data = json.loads(paths.meeting_json.read_text(encoding="utf-8"))
    return Meeting.model_validate(data)


def list_meetings(storage_dir: str | Path) -> list[MeetingPaths]:
    """List meeting folders (those containing ``meeting.json``), newest first."""
    base = Path(storage_dir).expanduser()
    if not base.is_dir():
        return []
    found = [
        MeetingPaths(root=child)
        for child in base.iterdir()
        if child.is_dir() and (child / MEETING_JSON).is_file()
    ]
    return sorted(found, key=lambda p: p.root.name, reverse=True)
