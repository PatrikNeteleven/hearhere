"""Export formats derived from ``meeting.json`` (md/txt/json/srt/vtt).

Every export is a pure function of a :class:`~hearhere.models.Meeting`, so
re-exporting never re-runs the models. :func:`export_meeting` writes the
requested formats into a meeting folder.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..logging_setup import get_logger
from ..models import Meeting
from ..pipeline.artifacts import MeetingPaths
from . import json as json_export
from . import markdown as markdown_export
from . import subtitles as subtitles_export
from . import summary as summary_export
from . import text as text_export

log = get_logger("export")

Renderer = Callable[[Meeting], str]

# Config format name -> (output filename, renderer).
FORMATS: dict[str, tuple[str, Renderer]] = {
    "markdown": (markdown_export.FILENAME, markdown_export.render),
    "text": (text_export.FILENAME, text_export.render),
    "json": (json_export.FILENAME, json_export.render),
    "srt": (subtitles_export.SRT_FILENAME, subtitles_export.render_srt),
    "vtt": (subtitles_export.VTT_FILENAME, subtitles_export.render_vtt),
}

# Friendly aliases accepted from the CLI (`--format md,srt`).
_ALIASES = {"md": "markdown", "txt": "text"}


def normalize_format(name: str) -> str:
    """Map a CLI/alias format name to a canonical key."""
    key = name.strip().lower()
    return _ALIASES.get(key, key)


def render(meeting: Meeting, fmt: str) -> str:
    """Render ``meeting`` in a single format, returning the file contents."""
    key = normalize_format(fmt)
    if key not in FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}; known: {sorted(FORMATS)}.")
    return FORMATS[key][1](meeting)


def export_meeting(
    paths: MeetingPaths, meeting: Meeting, formats: list[str]
) -> list[Path]:
    """Write each requested format into the meeting folder; return written paths.

    Unknown formats are logged and skipped rather than aborting the batch.
    ``summary.md`` is written additionally whenever the meeting has a summary,
    regardless of the requested transcript ``formats``.
    """
    written: list[Path] = []
    for fmt in formats:
        key = normalize_format(fmt)
        entry = FORMATS.get(key)
        if entry is None:
            log.warning("Skipping unknown export format %r", fmt)
            continue
        filename, renderer = entry
        out = paths.export(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(renderer(meeting), encoding="utf-8")
        written.append(out)
        log.debug("exported %s", out)

    if summary_export.has_summary(meeting):
        out = paths.export(summary_export.FILENAME)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary_export.render(meeting), encoding="utf-8")
        written.append(out)
        log.debug("exported %s", out)

    return written
