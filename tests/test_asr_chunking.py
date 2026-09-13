"""Tests for long-audio chunk planning + segment stitching (no torch needed).

Parakeet truncated a single long transcribe call (~32 s of a 46 s clip on real
hardware 2026-09-13), and a first hard-seam chunking scheme then dropped ~8 s at
the boundary. Long audio is now transcribed in overlapping windows, keeping every
segment and de-duplicating the overlaps. That logic is pure and tested here.
"""

from __future__ import annotations

from hearhere.engines.asr.parakeet_nemo import (
    _CHUNK_OVERLAP_SECONDS,
    _CHUNK_SECONDS,
    ParakeetNeMoEngine,
    _chunk_windows,
    _dedup_segments,
    _offset_segment,
)
from hearhere.models import Segment, Word


def _seg(start, end, text="x"):
    return Segment(start=start, end=end, text=text)


# --- window planning -------------------------------------------------------


def test_chunk_windows_single_when_short():
    assert _chunk_windows(20.0) == [(0.0, 20.0)]


def test_chunk_windows_overlap_and_cover_whole_clip():
    windows = _chunk_windows(46.0)  # the real-hardware case
    step = _CHUNK_SECONDS - _CHUNK_OVERLAP_SECONDS
    assert windows[0][0] == 0.0
    assert windows[-1][1] == 46.0  # reaches the end
    for start, end in windows:
        assert 0.0 <= start < end <= 46.0
        assert end - start <= _CHUNK_SECONDS + 1e-9  # never exceeds the safe length
    # Consecutive windows overlap (no region sits only at an edge).
    for (s0, e0), (s1, e1) in zip(windows, windows[1:]):
        assert s1 == s0 + step
        assert s1 < e0  # overlap


def test_chunk_windows_cover_a_long_clip_contiguously():
    total = 100.0
    windows = _chunk_windows(total)
    # Union of the windows covers [0, total] with no gap.
    covered_to = 0.0
    for start, end in windows:
        assert start <= covered_to + 1e-9  # next window starts before the last ended
        covered_to = max(covered_to, end)
    assert covered_to == total


# --- de-duplication of overlapping chunks ----------------------------------


def test_dedup_keeps_distinct_segments_in_order():
    segs = [_seg(10, 12, "b"), _seg(0, 2, "a"), _seg(20, 22, "c")]
    out = _dedup_segments(segs)
    assert [s.text for s in out] == ["a", "b", "c"]


def test_dedup_drops_overlapping_duplicate_keeps_longer():
    # Same utterance from two chunks: a partial copy and a fuller copy.
    partial = _seg(24.0, 25.0, "partial")
    full = _seg(23.5, 27.0, "full")
    out = _dedup_segments([partial, full])
    assert len(out) == 1
    assert out[0].text == "full"


def test_dedup_keeps_abutting_segments():
    # Back-to-back speech must not be treated as duplicates.
    out = _dedup_segments([_seg(0, 5), _seg(5, 10), _seg(10, 15)])
    assert len(out) == 3


def test_no_gap_across_a_seam():
    # Chunk A truncates its tail (loses 26-30); chunk B (starts 18) covers it.
    chunk_a = [_offset_segment(_seg(2, 6, "a1"), 0.0)]           # 2-6
    chunk_b = [                                                   # window start 18
        _offset_segment(_seg(6.0, 10.0, "b1"), 18.0),            # 24-28 (the seam)
        _offset_segment(_seg(12.0, 16.0, "b2"), 18.0),           # 30-34
    ]
    out = _dedup_segments(chunk_a + chunk_b)
    # The 24-28 region survives — no hole at the boundary.
    assert [round(s.start, 1) for s in out] == [2.0, 24.0, 30.0]


# --- timestamp offsetting --------------------------------------------------


# --- language hint selection (parakeet auto-detect can flip mid-clip) ------


class _FakeModel:
    """Stand-in NeMo model whose transcribe signature we control."""

    def __init__(self, kwarg_name=None):
        if kwarg_name is None:
            def transcribe(paths, timestamps=True, verbose=False):
                return []
        else:
            def transcribe(paths, timestamps=True, verbose=False, **_):
                return []
            transcribe.__signature__ = __import__("inspect").Signature(
                parameters=[
                    __import__("inspect").Parameter(
                        p, __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
                    )
                    for p in ("paths", "timestamps", "verbose", kwarg_name)
                ]
            )
        self.transcribe = transcribe


def _engine_with(kwarg_name):
    eng = ParakeetNeMoEngine()
    eng._model = _FakeModel(kwarg_name)
    return eng


def test_language_kwargs_empty_for_auto():
    eng = _engine_with("source_lang")
    assert eng._language_kwargs(None) == {}
    assert eng._language_kwargs("auto") == {}


def test_language_kwargs_uses_supported_name():
    assert _engine_with("source_lang")._language_kwargs("de") == {"source_lang": "de"}
    assert _engine_with("language")._language_kwargs("de") == {"language": "de"}


def test_language_kwargs_falls_back_when_unsupported():
    # A transcribe that accepts no language kwarg -> auto-detection (empty dict).
    assert _engine_with(None)._language_kwargs("de") == {}


def test_offset_segment_shifts_segment_and_words():
    seg = Segment(
        start=1.0, end=2.0, text="hi",
        words=[Word(start=1.0, end=1.5, text="hi")],
    )
    moved = _offset_segment(seg, 25.0)
    assert (moved.start, moved.end) == (26.0, 27.0)
    assert (moved.words[0].start, moved.words[0].end) == (26.0, 26.5)
    assert seg.start == 1.0 and seg.words[0].start == 1.0  # original untouched
