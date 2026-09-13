"""Timestamp formatting shared across export formats."""

from __future__ import annotations


def clock(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS`` (used in Markdown/text transcripts)."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def human_duration(seconds: float | None) -> str:
    """Seconds -> a compact human duration like ``32m 14s`` / ``1h 05m 09s``."""
    if not seconds or seconds <= 0:
        return "0s"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _sub_timestamp(seconds: float, sep: str) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def srt_timestamp(seconds: float) -> str:
    """Seconds -> SRT timestamp ``HH:MM:SS,mmm``."""
    return _sub_timestamp(seconds, ",")


def vtt_timestamp(seconds: float) -> str:
    """Seconds -> WebVTT timestamp ``HH:MM:SS.mmm``."""
    return _sub_timestamp(seconds, ".")
