"""Windows dual-channel capture via WASAPI (priority-1 platform).

- **Mic** (``self``): the default (or configured) WASAPI input device.
- **System output** (``others``): WASAPI **loopback** of the default render
  device — recording exactly what you hear, with no virtual cable needed.

Uses the `soundcard <https://github.com/bastibe/SoundCard>`_ library, which
exposes loopback microphones on Windows. Both channels are recorded on
background threads into memory, then written as 16 kHz mono WAVs on stop.

``soundcard``/``numpy`` are imported lazily, so importing HearHere on any
platform (including this WSL2 dev box, which has no audio) never needs them.
Install with ``pip install "hearhere[capture]"`` on Windows.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from ..logging_setup import get_logger
from .base import AudioCapture, CaptureError, CaptureResult, write_wav_16k_mono

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

log = get_logger("capture.windows")

# Frames pulled per read; a small block keeps stop() responsive.
_BLOCKSIZE = 1024


class WindowsCapture(AudioCapture):
    """Records mic + WASAPI loopback into ``self.wav`` / ``others.wav``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._mic_frames: list["np.ndarray"] = []
        self._out_frames: list["np.ndarray"] = []
        # Per-channel capture errors raised on the worker threads, surfaced on stop().
        self._errors: dict[str, BaseException] = {}
        self._t0: float = 0.0
        self._duration: float = 0.0

    # -- soundcard device resolution -------------------------------------

    def _resolve_mic(self) -> Any:
        import soundcard as sc  # noqa: PLC0415

        if self.mic_device == "default":
            return sc.default_microphone()
        return sc.get_microphone(self.mic_device, include_loopback=True)

    def _resolve_loopback(self) -> Any:
        """The default speaker as a loopback 'microphone'."""
        import soundcard as sc  # noqa: PLC0415

        if self.output_device == "default":
            speaker = sc.default_speaker()
        else:
            speaker = sc.get_speaker(self.output_device)
        # Loopback capture of a render device is exposed as a microphone.
        return sc.get_microphone(speaker.name, include_loopback=True)

    # -- recording loops -------------------------------------------------

    def _record(self, channel: str, device: Any, sink: list["np.ndarray"]) -> None:
        try:
            with device.recorder(
                samplerate=self.sample_rate, blocksize=_BLOCKSIZE
            ) as rec:
                while not self._stop.is_set():
                    sink.append(rec.record(numframes=_BLOCKSIZE))
        except BaseException as exc:  # noqa: BLE001 - reported from stop()
            # AssertionError here is soundcard rejecting the device's shared-mode
            # WASAPI format (common with pro/USB interfaces); record it and stop
            # so we fail loudly instead of writing an empty WAV.
            self._errors[channel] = exc
            self._stop.set()
            log.error("Capture failed on %s channel (%s): %s", channel, device.name, exc)

    def start(self) -> None:
        mic = self._resolve_mic()
        loopback = self._resolve_loopback()
        log.info("Recording mic=%s loopback=%s @ %d Hz", mic.name, loopback.name, self.sample_rate)
        self._stop.clear()
        self._errors.clear()
        self._t0 = time.monotonic()
        self._threads = [
            threading.Thread(target=self._record, args=("self", mic, self._mic_frames), daemon=True),
            threading.Thread(target=self._record, args=("others", loopback, self._out_frames), daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> CaptureResult:
        self._stop.set()
        for t in self._threads:
            t.join()
        self._duration = time.monotonic() - self._t0
        self._threads = []

        import numpy as np  # noqa: PLC0415

        mic = (
            np.concatenate(self._mic_frames)
            if self._mic_frames
            else np.zeros((0, 1), dtype=np.float32)
        )
        out = (
            np.concatenate(self._out_frames)
            if self._out_frames
            else np.zeros((0, 1), dtype=np.float32)
        )
        self._mic_frames.clear()
        self._out_frames.clear()

        # Fail loudly if a channel's recorder crashed — never pass off an empty
        # WAV as a successful (silent) recording.
        if self._errors:
            raise self._capture_error()

        write_wav_16k_mono(self.self_wav, mic, self.sample_rate)
        write_wav_16k_mono(self.others_wav, out, self.sample_rate)
        for label, arr in (("self", mic), ("others", out)):
            if arr.shape[0] == 0:
                log.warning(
                    "The %s channel captured no audio — its transcript will be empty.",
                    label,
                )
        log.info("Captured %.1fs -> %s, %s", self._duration, self.self_wav, self.others_wav)
        return CaptureResult(
            self_wav=self.self_wav, others_wav=self.others_wav, duration=self._duration
        )

    def _capture_error(self) -> CaptureError:
        """Build an actionable error from the failed channel(s)."""
        channel = "self" if "self" in self._errors else "others"
        device = "microphone" if channel == "self" else "system-output (loopback)"
        exc = self._errors[channel]
        hint = (
            "This device is not compatible with soundcard's WASAPI shared-mode "
            "capture (common with pro/USB interfaces like Focusrite Scarlett). "
            "Try a different input via [capture].mic_device / output_device in "
            "config.toml, or change the Windows default device, then re-record."
            if isinstance(exc, AssertionError)
            else "Check that the device is connected and not exclusively held by "
            "another app."
        )
        return CaptureError(
            f"Recording the {device} channel failed ({type(exc).__name__}: {exc}). {hint}"
        )
