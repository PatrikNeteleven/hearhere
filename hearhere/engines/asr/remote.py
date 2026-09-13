"""ASR engine that delegates transcription to a remote worker.

:class:`RemoteASREngine` satisfies the :class:`~hearhere.engines.asr.base.ASREngine`
protocol but does no local compute: it uploads a single WAV to the worker's
``/transcribe`` endpoint and returns the segments. This is the lighter,
per-channel remote path; the full remote pipeline (ASR + diarization + summary
in one job) is driven by :class:`hearhere.remote.client.RemoteClient` from the
orchestrator. Both share the same worker.
"""

from __future__ import annotations

from ...logging_setup import get_logger
from ...models import Segment

log = get_logger("asr.remote")


class RemoteASREngine:
    """ASR engine backed by a remote HearHere worker."""

    def __init__(self, url: str, token: str | None = None) -> None:
        from ...remote.client import RemoteClient  # noqa: PLC0415

        self._client = RemoteClient(url, token=token)

    def transcribe(self, wav_path: str, language: str | None = None) -> list[Segment]:
        """Upload ``wav_path`` to the worker and return its segments."""
        log.info("Delegating ASR of %s to remote worker", wav_path)
        return self._client.transcribe(wav_path, language)
