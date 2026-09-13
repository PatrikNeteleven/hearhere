"""The ``DiarizationEngine`` interface and engine factory.

A diarization engine turns a mono WAV into a list of :class:`SpeakerTurn`s
(``start``/``end``/raw label). Only the ``others`` channel is diarized — the
``self`` channel is the local user and is labeled ``"Me"`` without diarization.
The protocol is small so backends can be swapped via config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...models import SpeakerTurn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import Config


@runtime_checkable
class DiarizationEngine(Protocol):
    """Splits a mono WAV into speaker turns with raw labels."""

    def diarize(
        self,
        wav_path: str,
        *,
        min_speakers: int = 0,
        max_speakers: int = 0,
    ) -> list[SpeakerTurn]:
        """Return time-ordered speaker turns for ``wav_path``.

        ``min_speakers``/``max_speakers`` bound the speaker count (``0`` = let
        the model decide). The ``speaker`` field carries the engine's raw label
        (e.g. ``"SPEAKER_00"``); the pipeline remaps these to stable
        ``"Speaker N"`` ids when assigning them to ASR segments.
        """
        ...


def create_diarization_engine(
    config: "Config", device: str | None = None
) -> DiarizationEngine:
    """Build the diarization engine named in ``config.diarization.engine``.

    ``device`` overrides the resolved compute device (see
    :func:`hearhere.compute.select_device`).
    """
    from ...compute import select_device  # noqa: PLC0415

    resolved = device or select_device(config.compute.device)
    name = config.diarization.engine

    if name == "pyannote":
        from .pyannote import PyannoteDiarizer  # noqa: PLC0415

        return PyannoteDiarizer(
            hf_token=config.diarization.hf_token,
            device=resolved,
        )
    raise ValueError(f"Unknown diarization engine {name!r}.")
