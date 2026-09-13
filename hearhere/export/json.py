"""JSON export: a derived ``transcript.json`` (the transcript view).

This is distinct from the canonical ``meeting.json`` (which the pipeline always
writes): ``transcript.json`` is a slimmer, transcript-focused artifact for
downstream tooling. Like every export, it is derived from ``meeting.json``.
"""

from __future__ import annotations

import json

from ..models import Meeting

FILENAME = "transcript.json"


def render(meeting: Meeting) -> str:
    """Render the transcript portion of ``meeting`` as pretty JSON."""
    payload = {
        "id": meeting.id,
        "title": meeting.title,
        "language": meeting.language or meeting.transcript.language,
        "duration": meeting.duration or meeting.transcript.duration,
        "speakers": meeting.speakers or meeting.transcript.speakers(),
        "segments": [seg.model_dump(mode="json") for seg in meeting.transcript.segments],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
