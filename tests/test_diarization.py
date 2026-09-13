"""Tests for overlap-based speaker assignment (no heavy deps)."""

from __future__ import annotations

from hearhere.models import Segment, SpeakerTurn
from hearhere.pipeline.merge import assign_speakers


def test_assigns_by_max_overlap():
    segs = [
        Segment(start=0.0, end=2.0, text="a"),
        Segment(start=3.0, end=5.0, text="b"),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=2.5, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.5, end=6.0, speaker="SPEAKER_01"),
    ]
    out = assign_speakers(segs, turns)
    assert [s.speaker for s in out] == ["Speaker 1", "Speaker 2"]


def test_stable_numbering_by_first_appearance():
    # The later raw label appears first in time -> it becomes "Speaker 1".
    segs = [
        Segment(start=0.0, end=2.0, text="early"),
        Segment(start=5.0, end=7.0, text="late"),
    ]
    turns = [
        SpeakerTurn(start=4.0, end=8.0, speaker="SPEAKER_09"),
        SpeakerTurn(start=0.0, end=3.0, speaker="SPEAKER_02"),
    ]
    out = assign_speakers(segs, turns)
    # SPEAKER_02 (overlaps the early segment) is numbered first.
    assert out[0].speaker == "Speaker 1"
    assert out[1].speaker == "Speaker 2"


def test_same_raw_label_reused_across_segments():
    segs = [
        Segment(start=0.0, end=2.0, text="a"),
        Segment(start=10.0, end=12.0, text="b"),
    ]
    turns = [SpeakerTurn(start=0.0, end=15.0, speaker="SPEAKER_00")]
    out = assign_speakers(segs, turns)
    assert [s.speaker for s in out] == ["Speaker 1", "Speaker 1"]


def test_no_overlap_leaves_speaker_none():
    segs = [Segment(start=0.0, end=2.0, text="a")]
    turns = [SpeakerTurn(start=10.0, end=12.0, speaker="SPEAKER_00")]
    assert assign_speakers(segs, turns)[0].speaker is None


def test_no_turns_returns_copies_unlabeled():
    original = Segment(start=0.0, end=2.0, text="a")
    out = assign_speakers([original], [])
    assert out[0].speaker is None
    assert out[0] is not original  # copy, not the input


def test_inputs_not_mutated():
    original = Segment(start=0.0, end=2.0, text="a")
    turns = [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]
    assign_speakers([original], turns)
    assert original.speaker is None
