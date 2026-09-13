"""Markdown export: a human-readable, speaker-attributed ``transcript.md``."""

from __future__ import annotations

from ..models import Meeting
from .timefmt import clock, human_duration

FILENAME = "transcript.md"


def render(meeting: Meeting) -> str:
    """Render ``meeting`` as Markdown (see the README example)."""
    date = meeting.created_at.date().isoformat()
    speakers = meeting.speakers or meeting.transcript.speakers()
    language = meeting.language or meeting.transcript.language or "unknown"
    duration = human_duration(meeting.duration or meeting.transcript.duration)

    lines = [
        f"# {meeting.title} — {date}",
        "",
        f"**Duration:** {duration} · **Language:** {language} · "
        f"**Speakers:** {', '.join(speakers) if speakers else '—'}",
        "",
        "---",
        "",
    ]
    for seg in meeting.transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        speaker = seg.speaker or "Unknown"
        lines.append(f"**[{clock(seg.start)}] {speaker}:** {text}")
    return "\n".join(lines) + "\n"
