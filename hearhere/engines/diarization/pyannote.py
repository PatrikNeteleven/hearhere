"""Speaker diarization via pyannote.audio.

Runs ``pyannote/speaker-diarization-3.1`` on the ``others`` channel to produce
:class:`SpeakerTurn`s. The pipeline pins those turns onto the ASR segments.

**First run** downloads the pretrained pipeline from the Hugging Face Hub, which
requires a (free) HF access token *and* accepting the model's user conditions:

1. Create a token at https://huggingface.co/settings/tokens
2. Accept the conditions on both
   https://huggingface.co/pyannote/speaker-diarization-3.1 and
   https://huggingface.co/pyannote/segmentation-3.0
3. Put the token in ``config.toml`` (``[diarization] hf_token``) or set the
   ``HF_TOKEN`` environment variable.

After the first download the model is cached locally and runs fully offline.
pyannote/torch are heavy and imported lazily. Install with
``pip install "hearhere[diarization]"``.
"""

from __future__ import annotations

from typing import Any

from ...logging_setup import get_logger
from ...models import SpeakerTurn

log = get_logger("diarization.pyannote")

DEFAULT_PIPELINE = "pyannote/speaker-diarization-3.1"


class PyannoteDiarizer:
    """Diarization engine backed by pyannote.audio."""

    def __init__(
        self,
        *,
        pipeline_name: str = DEFAULT_PIPELINE,
        hf_token: str = "",
        device: str = "cpu",
    ) -> None:
        self.pipeline_name = pipeline_name
        self.hf_token = hf_token
        self.device = device
        self._pipeline: Any = None

    def load(self) -> None:
        """Load (and cache) the pyannote pipeline onto the configured device."""
        if self._pipeline is not None:
            return
        import inspect  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from pyannote.audio import Pipeline  # noqa: PLC0415

        log.info("Loading diarization pipeline %s", self.pipeline_name)
        # pyannote.audio renamed the auth kwarg from ``use_auth_token`` to
        # ``token`` (aligning with huggingface_hub); pick whichever the
        # installed version accepts.
        params = inspect.signature(Pipeline.from_pretrained).parameters
        token_kwarg = "token" if "token" in params else "use_auth_token"
        pipeline = Pipeline.from_pretrained(
            self.pipeline_name,
            **{token_kwarg: self.hf_token or None},
        )
        if pipeline is None:
            raise RuntimeError(
                "pyannote returned no pipeline — is your HF token set and have "
                "you accepted the model conditions? See the module docstring."
            )
        try:
            pipeline.to(torch.device(self.device))
        except Exception:  # pragma: no cover - device fallback
            log.warning("Could not move pipeline to %s; using cpu", self.device)
            pipeline.to(torch.device("cpu"))
        self._pipeline = pipeline

    def diarize(
        self,
        wav_path: str,
        *,
        min_speakers: int = 0,
        max_speakers: int = 0,
    ) -> list[SpeakerTurn]:
        """Return speaker turns for ``wav_path`` (``0`` bounds mean auto)."""
        self.load()
        kwargs: dict[str, int] = {}
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

        annotation = self._pipeline(self._load_waveform(wav_path), **kwargs)
        return self._to_turns(annotation)

    @staticmethod
    def _load_waveform(wav_path: str) -> dict[str, Any]:
        """Decode ``wav_path`` in-memory for pyannote.

        Passing a pre-loaded waveform (rather than a path) keeps pyannote 4.x
        from routing decoding through ``torchcodec``/FFmpeg, which is awkward to
        install on Windows. ``soundfile`` (libsndfile) reads WAV natively and is
        already a dependency.
        """
        import soundfile as sf  # noqa: PLC0415
        import torch  # noqa: PLC0415

        data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        # soundfile yields (frames, channels); pyannote wants (channels, frames).
        waveform = torch.from_numpy(data).transpose(0, 1).contiguous()
        return {"waveform": waveform, "sample_rate": int(sample_rate)}

    # -- conversion ------------------------------------------------------

    @staticmethod
    def _to_turns(annotation: Any) -> list[SpeakerTurn]:
        """Convert a pyannote ``Annotation`` into time-ordered turns."""
        turns = [
            SpeakerTurn(start=float(segment.start), end=float(segment.end), speaker=str(label))
            for segment, _track, label in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: (t.start, t.end))
        return turns
