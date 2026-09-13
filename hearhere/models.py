"""Core data models — the ``meeting.json`` shape and its building blocks.

``meeting.json`` is the canonical, source-of-truth record for a meeting; every
export format is derived from it, so re-exporting never re-runs the models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# The label used for the local user's own (mic) channel.
SELF_SPEAKER = "Me"

SCHEMA_VERSION = 1


class Word(BaseModel):
    """A single word with timing, when word/char timestamps are requested."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class Segment(BaseModel):
    """A time-stamped chunk of transcribed speech.

    ``speaker`` is ``"Me"`` for the self channel, a diarized ``"Speaker N"`` (or
    renamed) label for the others channel, or ``None`` before attribution.
    """

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker: Optional[str] = None
    channel: Optional[str] = None  # "self" | "others"
    words: list[Word] = Field(default_factory=list)


class SpeakerTurn(BaseModel):
    """A diarized speaker turn on a single channel (no text)."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str


class Transcript(BaseModel):
    """The merged, speaker-attributed transcript for a meeting."""

    segments: list[Segment] = Field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None

    def speakers(self) -> list[str]:
        """Distinct speaker labels in first-appearance order."""
        seen: list[str] = []
        for seg in self.segments:
            if seg.speaker and seg.speaker not in seen:
                seen.append(seg.speaker)
        return seen


class Summary(BaseModel):
    """LLM-produced meeting summary and structured takeaways."""

    summary: str = ""
    decisions: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


class Meeting(BaseModel):
    """Canonical structured record persisted as ``meeting.json``."""

    schema_version: int = SCHEMA_VERSION
    id: str  # folder slug, e.g. "2026-09-13_weekly-sync"
    title: str
    created_at: datetime = Field(default_factory=datetime.now)
    language: Optional[str] = None
    duration: Optional[float] = None
    speakers: list[str] = Field(default_factory=list)
    transcript: Transcript = Field(default_factory=Transcript)
    summary: Optional[Summary] = None
    # Paths are stored relative to the meeting folder.
    audio: dict[str, str] = Field(
        default_factory=lambda: {
            "self": "audio/self.wav",
            "others": "audio/others.wav",
        }
    )
    metadata: dict[str, object] = Field(default_factory=dict)
