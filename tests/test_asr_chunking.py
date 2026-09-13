"""Tests for long-audio chunk planning + segment stitching (no torch needed).

Parakeet truncated a single long transcribe call (~32 s of a 46 s clip on real
hardware, 2026-09-13); long audio is now transcribed in overlapping windows and
stitched. The tiling logic is pure and tested here.
"""

from __future__ import annotations

from hearhere.engines.asr.parakeet_nemo import (
    _CHUNK_OVERLAP_SECONDS,
    _CHUNK_SECONDS,
    _chunk_windows,
    _offset_segment,
)
from hearhere.models import Segment, Word


def test_chunk_windows_single_when_short():
    # <= chunk length is handled by the single-shot path, but the planner is safe.
    windows = _chunk_windows(20.0)
    assert windows == [(0.0, 20.0, 0.0, 20.0)]


def test_chunk_windows_tile_without_gaps_or_overlaps():
    windows = _chunk_windows(46.0)  # the real-hardware case
    assert len(windows) == 2
    # keep-regions must tile [0, total) exactly: each hi == next lo, ends at total.
    keeps = [(lo, hi) for _, _, lo, hi in windows]
    assert keeps[0][0] == 0.0
    assert keeps[-1][1] == 46.0
    assert all(a[1] == b[0] for a, b in zip(keeps, keeps[1:]))


def test_chunk_windows_three_chunks_cover_everything():
    total = 70.0
    windows = _chunk_windows(total)
    step = _CHUNK_SECONDS - _CHUNK_OVERLAP_SECONDS
    assert len(windows) == 3
    # Every window's capture span stays within the audio and is <= chunk length.
    for start, end, lo, hi in windows:
        assert 0.0 <= start < end <= total
        assert end - start <= _CHUNK_SECONDS + 1e-9
        assert lo < hi
    # Contiguous keep regions covering [0, total].
    keeps = [(lo, hi) for _, _, lo, hi in windows]
    assert keeps[0][0] == 0.0 and keeps[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(keeps, keeps[1:]))
    assert windows[1][0] == step  # second window starts one step in


def test_offset_segment_shifts_segment_and_words():
    seg = Segment(
        start=1.0, end=2.0, text="hi",
        words=[Word(start=1.0, end=1.5, text="hi")],
    )
    moved = _offset_segment(seg, 25.0)
    assert (moved.start, moved.end) == (26.0, 27.0)
    assert (moved.words[0].start, moved.words[0].end) == (26.0, 26.5)
    # Original untouched.
    assert seg.start == 1.0 and seg.words[0].start == 1.0
