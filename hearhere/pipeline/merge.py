"""Merge and align the ``self`` and ``others`` segments onto one timeline.

The mic channel is trivially attributed to the local user (``"Me"``); the
output channel's ASR segments are labeled from diarization turns by
:func:`assign_speakers` before merging. Both sets share the same clock — both
WAVs start at recording start — so merging is a stable sort by start time.
"""

from __future__ import annotations

from ..models import SELF_SPEAKER, Segment, SpeakerTurn, Transcript

DEFAULT_SPEAKER_PREFIX = "Speaker"


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the time overlap between ``[a_start, a_end]`` and ``[b_start, b_end]``."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _best_turn_label(seg: Segment, turns: list[SpeakerTurn]) -> str | None:
    """Raw label of the turn that overlaps ``seg`` most (``None`` if none does)."""
    best_label: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = _overlap(seg.start, seg.end, turn.start, turn.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = turn.speaker
    return best_label


def assign_speakers(
    segments: list[Segment],
    turns: list[SpeakerTurn],
    *,
    prefix: str = DEFAULT_SPEAKER_PREFIX,
) -> list[Segment]:
    """Attach diarized speaker labels to ``segments`` by maximal time overlap.

    Each segment is tagged with the speaker of the turn it overlaps most; raw
    diarization labels (``"SPEAKER_00"``…) are remapped to stable, human
    ``"Speaker N"`` ids numbered by first appearance in time. Segments that
    overlap no turn keep ``speaker=None``. Inputs are not mutated.
    """
    if not turns:
        return [seg.model_copy() for seg in segments]

    # Number raw labels by first appearance along the timeline for stability.
    label_map: dict[str, str] = {}
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        raw = _best_turn_label(seg, turns)
        if raw is not None and raw not in label_map:
            label_map[raw] = f"{prefix} {len(label_map) + 1}"

    out: list[Segment] = []
    for seg in segments:
        raw = _best_turn_label(seg, turns)
        speaker = label_map.get(raw) if raw is not None else None
        out.append(seg.model_copy(update={"speaker": speaker}))
    return out


def _prepare(segments: list[Segment], channel: str, speaker: str | None) -> list[Segment]:
    """Copy ``segments`` tagging their channel and (optionally) speaker."""
    out: list[Segment] = []
    for seg in segments:
        out.append(
            seg.model_copy(
                update={
                    "channel": channel,
                    "speaker": speaker if speaker is not None else seg.speaker,
                }
            )
        )
    return out


def merge_segments(
    self_segments: list[Segment],
    others_segments: list[Segment],
    *,
    language: str | None = None,
) -> Transcript:
    """Combine self/others segments into a time-ordered :class:`Transcript`.

    ``self`` segments are labeled ``"Me"``; ``others`` segments keep whatever
    speaker they already carry (``None`` until diarization). Ties on start time
    are broken by end time then channel (``others`` before ``self``) for a
    stable, deterministic order.
    """
    prepared_self = _prepare(self_segments, channel="self", speaker=SELF_SPEAKER)
    prepared_others = _prepare(others_segments, channel="others", speaker=None)

    merged = prepared_self + prepared_others
    merged.sort(key=lambda s: (s.start, s.end, 0 if s.channel == "others" else 1))

    duration = max((s.end for s in merged), default=0.0)
    return Transcript(segments=merged, language=language, duration=duration)
