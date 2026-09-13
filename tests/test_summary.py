"""Tests for summarization prompts, bullet parsing, and summary.md export."""

from __future__ import annotations

from datetime import datetime

from hearhere.engines.llm.base import transcript_to_text
from hearhere.engines.llm.ollama import _bullets
from hearhere.export import summary as summary_export
from hearhere.llm.prompts import build_prompt
from hearhere.models import Meeting, Segment, Summary, Transcript


def _meeting_with_summary() -> Meeting:
    return Meeting(
        id="2026-09-13_weekly-sync",
        title="Weekly Sync",
        created_at=datetime(2026, 9, 13, 10, 0, 0),
        summary=Summary(
            summary="The team reviewed staging and agreed to ship.",
            decisions=["Ship on Friday."],
            action_items=["Anna to update the changelog."],
        ),
    )


def test_build_prompt_language_clause():
    _, user_de = build_prompt("summary", "Hallo", language="de")
    assert "in German" in user_de
    _, user_auto = build_prompt("summary", "Hi", language=None)
    assert "same language as the transcript" in user_auto
    assert "--- TRANSCRIPT ---" in user_auto


def test_build_prompt_unknown_task_raises():
    import pytest

    with pytest.raises(ValueError):
        build_prompt("mood", "text")


def test_bullets_parses_markers_and_enumerators():
    assert _bullets("- one\n* two\n3. three\n\n  \n- ") == ["one", "two", "three"]


def test_transcript_to_text_skips_blanks():
    t = Transcript(
        segments=[
            Segment(start=0, end=1, text="Hi", speaker="Me"),
            Segment(start=1, end=2, text="  ", speaker="Anna"),
            Segment(start=2, end=3, text="Hello", speaker="Anna"),
        ]
    )
    assert transcript_to_text(t) == "Me: Hi\nAnna: Hello"


def test_summary_export_render_and_has_summary():
    meeting = _meeting_with_summary()
    assert summary_export.has_summary(meeting)
    md = summary_export.render(meeting)
    assert md.startswith("# Summary — Weekly Sync — 2026-09-13")
    assert "The team reviewed staging" in md
    assert "## Decisions" in md
    assert "- Ship on Friday." in md
    assert "## Action Items" in md
    assert "- Anna to update the changelog." in md


def test_has_summary_false_when_empty():
    assert not summary_export.has_summary(Meeting(id="x", title="X"))
    assert not summary_export.has_summary(
        Meeting(id="x", title="X", summary=Summary())
    )
