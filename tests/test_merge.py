"""Tests for merging self/others segments onto one timeline."""

from __future__ import annotations

from hearhere.models import SELF_SPEAKER, Segment
from hearhere.pipeline.merge import merge_segments


def test_self_segments_labeled_me_and_channel_set():
    transcript = merge_segments(
        [Segment(start=0.0, end=2.0, text="Hi")],
        [],
    )
    seg = transcript.segments[0]
    assert seg.speaker == SELF_SPEAKER
    assert seg.channel == "self"


def test_others_keep_speaker_none_and_channel_others():
    transcript = merge_segments(
        [],
        [Segment(start=0.0, end=2.0, text="Hello")],
    )
    seg = transcript.segments[0]
    assert seg.speaker is None
    assert seg.channel == "others"


def test_merged_order_is_by_time():
    transcript = merge_segments(
        [Segment(start=5.0, end=7.0, text="me-late"), Segment(start=1.0, end=2.0, text="me-early")],
        [Segment(start=3.0, end=4.0, text="other-mid")],
    )
    assert [s.text for s in transcript.segments] == ["me-early", "other-mid", "me-late"]


def test_tie_break_others_before_self():
    transcript = merge_segments(
        [Segment(start=1.0, end=2.0, text="me")],
        [Segment(start=1.0, end=2.0, text="other")],
    )
    assert [s.text for s in transcript.segments] == ["other", "me"]


def test_duration_is_max_end_and_language_passthrough():
    transcript = merge_segments(
        [Segment(start=0.0, end=9.0, text="a")],
        [Segment(start=0.0, end=4.0, text="b")],
        language="de",
    )
    assert transcript.duration == 9.0
    assert transcript.language == "de"


def test_inputs_not_mutated():
    original = Segment(start=0.0, end=1.0, text="x")
    merge_segments([original], [])
    assert original.speaker is None
    assert original.channel is None


def test_empty_inputs():
    transcript = merge_segments([], [])
    assert transcript.segments == []
    assert transcript.duration == 0.0
