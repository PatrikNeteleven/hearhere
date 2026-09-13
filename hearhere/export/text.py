"""Plain-text export: ``transcript.txt``."""

from __future__ import annotations

from ..models import Meeting
from .timefmt import clock

FILENAME = "transcript.txt"


def render(meeting: Meeting) -> str:
    """Render ``meeting`` as a plain-text transcript."""
    lines: list[str] = []
    for seg in meeting.transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        speaker = seg.speaker or "Unknown"
        lines.append(f"[{clock(seg.start)}] {speaker}: {text}")
    return "\n".join(lines) + ("\n" if lines else "")
