"""Tests for the meeting-folder layout and artifact helpers."""

from __future__ import annotations

from datetime import date

import pytest

from hearhere.models import Meeting, Segment, Summary, Transcript
from hearhere.pipeline import artifacts


def test_slugify():
    assert artifacts.slugify("Weekly Sync") == "weekly-sync"
    assert artifacts.slugify("Weekly Sync!!!") == "weekly-sync"
    assert artifacts.slugify("  Über Café  ") == "uber-cafe"
    assert artifacts.slugify("") == "meeting"


def test_meeting_id():
    assert artifacts.meeting_id("Weekly Sync", date(2026, 9, 13)) == "2026-09-13_weekly-sync"


def test_create_meeting_dir_layout(tmp_path):
    paths = artifacts.create_meeting_dir(tmp_path, "Weekly Sync", date(2026, 9, 13))
    assert paths.root == tmp_path / "2026-09-13_weekly-sync"
    assert paths.audio_dir.is_dir()
    assert paths.self_wav == paths.audio_dir / "self.wav"
    assert paths.others_wav == paths.audio_dir / "others.wav"
    assert paths.meeting_json == paths.root / "meeting.json"
    assert paths.log == paths.root / "hearhere.log"
    assert paths.export("transcript.md") == paths.root / "transcript.md"


def test_create_meeting_dir_conflict(tmp_path):
    artifacts.create_meeting_dir(tmp_path, "Sync", date(2026, 9, 13))
    with pytest.raises(FileExistsError):
        artifacts.create_meeting_dir(tmp_path, "Sync", date(2026, 9, 13))
    # exist_ok tolerates it.
    artifacts.create_meeting_dir(tmp_path, "Sync", date(2026, 9, 13), exist_ok=True)


def _sample_meeting() -> Meeting:
    return Meeting(
        id="2026-09-13_weekly-sync",
        title="Weekly Sync",
        language="de",
        duration=1934.0,
        speakers=["Me", "Anna"],
        transcript=Transcript(
            language="de",
            duration=1934.0,
            segments=[
                Segment(start=4.0, end=10.5, text="Kurzes Update.", speaker="Me", channel="self"),
                Segment(start=11.0, end=20.0, text="Staging is green.", speaker="Anna", channel="others"),
            ],
        ),
        summary=Summary(summary="A short sync.", decisions=["Ship Friday"], action_items=["Ben deploys"]),
    )


def test_write_then_read_roundtrip(tmp_path):
    paths = artifacts.create_meeting_dir(tmp_path, "Weekly Sync", date(2026, 9, 13))
    meeting = _sample_meeting()
    artifacts.write_meeting(paths, meeting)
    assert paths.meeting_json.is_file()

    loaded = artifacts.read_meeting(paths)
    assert loaded == meeting
    assert loaded.transcript.speakers() == ["Me", "Anna"]


def test_read_missing_meeting_json(tmp_path):
    paths = artifacts.resolve_meeting(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        artifacts.read_meeting(paths)


def test_list_meetings_newest_first(tmp_path):
    for title, d in [("Old", date(2026, 1, 1)), ("New", date(2026, 12, 31)), ("Mid", date(2026, 6, 15))]:
        p = artifacts.create_meeting_dir(tmp_path, title, d)
        artifacts.write_meeting(p, Meeting(id=p.root.name, title=title))
    # A stray non-meeting folder is ignored.
    (tmp_path / "not-a-meeting").mkdir()

    names = [m.root.name for m in artifacts.list_meetings(tmp_path)]
    assert names == [
        "2026-12-31_new",
        "2026-06-15_mid",
        "2026-01-01_old",
    ]


def test_list_meetings_missing_dir(tmp_path):
    assert artifacts.list_meetings(tmp_path / "does-not-exist") == []
