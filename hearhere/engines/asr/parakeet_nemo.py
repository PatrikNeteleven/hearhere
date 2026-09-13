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
        results = self._model.transcribe(
            [str(wav_path)], timestamps=True, verbose=False
        )
        if not results:
            return []
        return self._to_segments(results[0], want_words=want_words)

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
