"""Parakeet ASR via NVIDIA NeMo.

Loads ``nvidia/parakeet-tdt-0.6b-v3`` (multilingual, CC-BY-4.0) through NeMo and
transcribes a 16 kHz mono WAV into :class:`Segment`s with segment-level (and
optionally word-level) timestamps.

NeMo/torch are heavy and imported lazily: constructing the engine is cheap and
never imports them; the model is loaded on first :meth:`transcribe` (or via
:meth:`load`). The weights download automatically on first use. Install with
``pip install "hearhere[asr]"``.
"""

from __future__ import annotations

import contextlib
import tempfile
import wave
from pathlib import Path
from typing import Any

from ...logging_setup import get_logger
from ...models import Segment, Word

log = get_logger("asr.parakeet")

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

# Below this, the clip is effectively empty; NeMo's preprocessor crashes on
# zero-length input, so we skip it and return no segments instead.
_MIN_DURATION_S = 0.1

# Long audio is transcribed in overlapping chunks: some NeMo builds silently
# truncate a single long ``transcribe`` call (observed cutting off ~32 s of a
# 46 s clip). Chunks stay comfortably under that so no chunk truncates its own
# tail; the generous overlap means any segment straddling a cut is transcribed
# whole inside at least one chunk, and duplicates from the overlap are removed by
# :func:`_dedup_segments`. This tolerates a chunk dropping its edges, which a
# hard-seam scheme did not (it left an ~8 s gap at the boundary).
_CHUNK_SECONDS = 24.0
_CHUNK_OVERLAP_SECONDS = 6.0
# Two segments overlapping more than this fraction of the shorter one are treated
# as the same utterance seen in two chunks.
_DUP_OVERLAP_RATIO = 0.5


def _offset_segment(seg: Segment, dt: float) -> Segment:
    """Shift a segment (and its words) by ``dt`` seconds onto the global timeline."""
    return seg.model_copy(
        update={
            "start": seg.start + dt,
            "end": seg.end + dt,
            "words": [
                w.model_copy(update={"start": w.start + dt, "end": w.end + dt})
                for w in seg.words
            ],
        }
    )


def _chunk_windows(
    total_s: float, chunk_s: float = _CHUNK_SECONDS, overlap_s: float = _CHUNK_OVERLAP_SECONDS
) -> list[tuple[float, float]]:
    """Plan overlapping ``(start, end)`` capture windows covering ``[0, total_s]``.

    Consecutive windows overlap by ``overlap_s`` so no region sits only at a
    window edge; the caller keeps every segment and de-duplicates afterwards.
    """
    step = max(chunk_s - overlap_s, 1.0)
    windows: list[tuple[float, float]] = []
    start = 0.0
    while True:
        end = min(start + chunk_s, total_s)
        windows.append((start, end))
        if end >= total_s - 1e-9:
            break
        start += step
    return windows


def _dedup_segments(segments: list[Segment]) -> list[Segment]:
    """Sort by time and drop near-duplicates from overlapping chunks.

    Two segments that overlap by more than ``_DUP_OVERLAP_RATIO`` of the shorter
    one are the same utterance seen in two chunks; keep the longer (more complete)
    one. Segments that merely abut or briefly touch are both kept, so no audio is
    dropped at a seam.
    """
    kept: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        if kept:
            prev = kept[-1]
            overlap = min(prev.end, seg.end) - max(prev.start, seg.start)
            shorter = min(prev.end - prev.start, seg.end - seg.start) or 1e-9
            if overlap / shorter > _DUP_OVERLAP_RATIO:
                if (seg.end - seg.start) > (prev.end - prev.start):
                    kept[-1] = seg  # replace the partial copy with the fuller one
                continue
        kept.append(seg)
    return kept


def _wav_duration_seconds(wav_path: str) -> float | None:
    """Duration of a WAV via the stdlib (no torch/soundfile); ``None`` if unreadable."""
    path = Path(wav_path)
    if not path.is_file():
        return 0.0
    with contextlib.suppress(Exception):
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / rate if rate else 0.0
    return None  # unknown format — let the model try


class ParakeetNeMoEngine:
    """ASR engine backed by NeMo's Parakeet TDT model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cpu",
        timestamps: str = "segment",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.timestamps = timestamps
        self._model: Any = None

    def load(self) -> None:
        """Load (and cache) the NeMo model onto the configured device."""
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr  # noqa: PLC0415

        log.info("Loading ASR model %s onto %s", self.model_name, self.device)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        try:
            model = model.to(self.device)
        except Exception:  # pragma: no cover - device fallback
            log.warning("Could not move model to %s; using cpu", self.device)
            model = model.to("cpu")
        model.eval()
        self._model = model

    def transcribe(self, wav_path: str, language: str | None = None) -> list[Segment]:
        """Transcribe ``wav_path`` into time-stamped segments."""
        duration = _wav_duration_seconds(wav_path)
        if duration is not None and duration < _MIN_DURATION_S:
            log.warning(
                "Skipping %s: no usable audio (%.2fs). Was this channel captured?",
                wav_path,
                duration,
            )
            return []
        self.load()
        want_words = self.timestamps in ("word", "char")
        if duration is not None and duration > _CHUNK_SECONDS:
            segments = self._transcribe_chunked(str(wav_path), duration, want_words=want_words)
        else:
            segments = self._transcribe_file(str(wav_path), want_words=want_words)

        # Coverage check: if transcription ends well before the audio does, the
        # tail was dropped — surfaces any remaining truncation in the log.
        last_end = segments[-1].end if segments else 0.0
        if duration and last_end and duration - last_end > 3.0:
            log.warning(
                "Transcription of %s covers only %.1fs of %.1fs of audio "
                "(%.0f%%) — the tail may have been truncated.",
                wav_path, last_end, duration, 100.0 * last_end / duration,
            )
        else:
            log.info(
                "Transcribed %s: %.1fs audio, %d segment(s)",
                wav_path, duration or 0.0, len(segments),
            )
        return segments

    # -- transcription backends ------------------------------------------

    def _transcribe_file(self, wav_path: str, *, want_words: bool) -> list[Segment]:
        """Single-shot transcription of a whole WAV file."""
        results = self._model.transcribe([wav_path], timestamps=True, verbose=False)
        if not results:
            return []
        return self._to_segments(results[0], want_words=want_words)

    def _transcribe_chunked(
        self, wav_path: str, duration: float, *, want_words: bool
    ) -> list[Segment]:
        """Transcribe long audio in overlapping windows and stitch the timeline."""
        import soundfile as sf  # noqa: PLC0415

        audio, sr = sf.read(wav_path, dtype="float32")
        if getattr(audio, "ndim", 1) > 1:  # defensive: downmix if not already mono
            audio = audio.mean(axis=1)

        windows = _chunk_windows(duration)
        log.info(
            "Transcribing %s in %d chunk(s) of ~%.0fs (%.1fs total)",
            wav_path, len(windows), _CHUNK_SECONDS, duration,
        )
        collected: list[Segment] = []
        for start, end in windows:
            samples = audio[int(start * sr):int(end * sr)]
            for seg in self._transcribe_samples(samples, sr, want_words=want_words):
                collected.append(_offset_segment(seg, start))
        return _dedup_segments(collected)

    def _transcribe_samples(
        self, samples: Any, sr: int, *, want_words: bool
    ) -> list[Segment]:
        """Transcribe an in-memory chunk via a temp WAV (path input is best-tested)."""
        import soundfile as sf  # noqa: PLC0415

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            tmp = fh.name
        try:
            sf.write(tmp, samples, sr, subtype="PCM_16")
            return self._transcribe_file(tmp, want_words=want_words)
        finally:
            Path(tmp).unlink(missing_ok=True)

    # -- conversion ------------------------------------------------------

    @staticmethod
    def _to_segments(result: Any, *, want_words: bool) -> list[Segment]:
        """Convert one NeMo hypothesis into :class:`Segment`s.

        NeMo exposes ``result.timestamp`` as a dict with ``"segment"`` and
        ``"word"`` lists of ``{"start", "end", "segment"/"word"}`` entries.
        """
        stamp = getattr(result, "timestamp", None) or {}
        seg_stamps = stamp.get("segment") or []

        # Fall back to a single segment when no segment timestamps are present.
        if not seg_stamps:
            text = getattr(result, "text", "") or ""
            if not text:
                return []
            return [Segment(start=0.0, end=0.0, text=text)]

        word_stamps = stamp.get("word") or [] if want_words else []
        segments: list[Segment] = []
        for s in seg_stamps:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start))
            text = (s.get("segment") or s.get("text") or "").strip()
            words = [
                Word(
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    text=(w.get("word") or w.get("text") or "").strip(),
                )
                for w in word_stamps
                if start <= float(w.get("start", 0.0)) < end
            ]
            segments.append(Segment(start=start, end=end, text=text, words=words))
        return segments
