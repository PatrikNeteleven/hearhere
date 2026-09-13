"""Tests for the export formats (all derived from a Meeting)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from hearhere.export import export_meeting, normalize_format, render
from hearhere.export.timefmt import clock, human_duration, srt_timestamp, vtt_timestamp
from hearhere.models import Meeting, Segment, Transcript
from hearhere.pipeline import artifacts


def _meeting() -> Meeting:
    return Meeting(
        id="2026-09-13_weekly-sync",
        title="Weekly Sync",
        created_at=datetime(2026, 9, 13, 10, 0, 0),
        language="de",
        duration=1934.0,
        speakers=["Me", "Anna"],
        transcript=Transcript(
            language="de",
            duration=1934.0,
            segments=[
                Segment(start=4.0, end=10.5, text="Kurzes Update.", speaker="Me", channel="self"),
                Segment(start=11.0, end=20.0, text="Staging is green.", speaker="Anna", channel="others"),
                Segment(start=21.0, end=21.0, text="", speaker="Anna", channel="others"),  # blank, skipped
            ],
        ),
    )


def test_time_helpers():
    assert clock(4) == "00:00:04"
    assert clock(3661) == "01:01:01"
    assert human_duration(1934) == "32m 14s"
    assert human_duration(3669) == "1h 01m 09s"
    assert human_duration(0) == "0s"
    assert srt_timestamp(1.5) == "00:00:01,500"
    assert vtt_timestamp(1.5) == "00:00:01.500"


def test_markdown_render():
    md = render(_meeting(), "markdown")
    assert md.startswith("# Weekly Sync — 2026-09-13")
    assert "**Speakers:** Me, Anna" in md
    assert "**[00:00:04] Me:** Kurzes Update." in md
    assert "**[00:00:11] Anna:** Staging is green." in md
    assert "Kurzes Update." in md  # blank segment produced no stray line


def test_text_render():
    txt = render(_meeting(), "text")
    assert txt.splitlines() == [
        "[00:00:04] Me: Kurzes Update.",
        "[00:00:11] Anna: Staging is green.",
    ]


def test_json_render_is_valid_and_skips_nothing():
    data = json.loads(render(_meeting(), "json"))
    assert data["id"] == "2026-09-13_weekly-sync"
    assert data["speakers"] == ["Me", "Anna"]
    assert len(data["segments"]) == 3  # json keeps the raw transcript


def test_srt_render():
    srt = render(_meeting(), "srt")
    assert srt.startswith("1\n00:00:04,000 --> 00:00:10,500\nMe: Kurzes Update.\n")
    assert "2\n00:00:11,000 --> 00:00:20,000\nAnna: Staging is green.\n" in srt


def test_vtt_render():
    vtt = render(_meeting(), "vtt")
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:04.000 --> 00:00:10.500\nMe: Kurzes Update.\n" in vtt


def test_normalize_format_aliases():
    assert normalize_format("md") == "markdown"
    assert normalize_format("TXT") == "text"
    assert normalize_format("srt") == "srt"


def test_render_unknown_format_raises():
    with pytest.raises(ValueError):
        render(_meeting(), "pdf")


def test_export_meeting_writes_files_and_skips_unknown(tmp_path):
    paths = artifacts.create_meeting_dir(tmp_path, "Weekly Sync")
    written = export_meeting(paths, _meeting(), ["markdown", "srt", "bogus"])
    names = {p.name for p in written}
    assert names == {"transcript.md", "transcript.srt"}
    assert (paths.root / "transcript.md").read_text(encoding="utf-8").startswith("# Weekly Sync")
