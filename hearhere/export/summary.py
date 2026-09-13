"""Summary export: a human-readable ``summary.md`` from ``meeting.summary``.

Unlike the transcript formats, ``summary.md`` is written whenever a meeting has
a summary (it is not one of the selectable ``export.formats``). :func:`has_summary`
lets the caller decide whether there is anything to write.
"""

from __future__ import annotations

from ..models import Meeting, Summary

FILENAME = "summary.md"


def has_summary(meeting: Meeting) -> bool:
    """True if ``meeting`` carries any summary content worth writing."""
    s = meeting.summary
    return bool(s and (s.summary.strip() or s.decisions or s.action_items))


def render(meeting: Meeting) -> str:
    """Render ``meeting.summary`` as Markdown."""
    date = meeting.created_at.date().isoformat()
    summary: Summary = meeting.summary or Summary()

    lines = [f"# Summary — {meeting.title} — {date}", ""]

    if summary.summary.strip():
        lines.append(summary.summary.strip())
        lines.append("")

    if summary.decisions:
        lines.append("## Decisions")
        lines.append("")
        lines.extend(f"- {item}" for item in summary.decisions)
        lines.append("")

    if summary.action_items:
        lines.append("## Action Items")
        lines.append("")
        lines.extend(f"- {item}" for item in summary.action_items)
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"
