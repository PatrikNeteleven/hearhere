"""Linux dual-channel capture (priority-3 platform).

Both channels go through `soundcard <https://github.com/bastibe/SoundCard>`_,
which talks to **PulseAudio / PipeWire** (pipewire-pulse) directly:

- **Mic** (``self``): the default (or configured) input source.
- **System output** (``others``): the ``.monitor`` source of the default (or
  configured) sink — native loopback of exactly what you hear, no virtual cable
  and no extra software. PipeWire/PulseAudio exposes every sink's monitor as a
  loopback "microphone", which soundcard surfaces via ``include_loopback=True``.

Unlike Windows, both channels use the same backend: PulseAudio's shared-mode
capture doesn't have WASAPI's format-negotiation pitfalls, so there's no need to
split the mic onto ``sounddevice`` (which on Linux binds to raw ALSA ``hw:``
devices, bypassing PipeWire and its defaults). Recording symmetrically through
soundcard keeps us on the PipeWire graph and respects the user's default sink.

Each channel is captured at 16 kHz mono (Parakeet's required input); soundcard
resamples from the device rate for us, and :func:`write_wav_16k_mono` downmixes
to mono on write.

``soundcard``/``numpy`` are imported lazily, so importing HearHere never needs
them. Install with ``pip install "hearhere[capture]"`` (plus the system
``libportaudio2`` package and a running PipeWire/PulseAudio session).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from ..extras import require
from ..logging_setup import get_logger
from .base import (
    AudioCapture,
    CaptureError,
    CaptureResult,
    write_wav_16k_mono,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

log = get_logger("capture.linux")

# Frames pulled per read; a small block keeps stop() responsive.
_BLOCKSIZE = 1024


class LinuxCapture(AudioCapture):
    """Records mic + default-sink monitor into ``self.wav`` / ``others.wav``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._mic_frames: list["np.ndarray"] = []
        self._out_frames: list["np.ndarray"] = []
        # Both channels are recorded at the target rate (soundcard resamples).
        self._mic_rate: int = self.sample_rate
        self._out_rate: int = self.sample_rate
        # Per-channel capture errors, surfaced on stop().
        self._errors: dict[str, BaseException] = {}
        self._t0: float = 0.0
        self._duration: float = 0.0

    # -- device resolution -----------------------------------------------

    def _resolve_mic(self) -> Any:
        """The default (or configured) input source as a soundcard microphone."""
        with require("capture", "Recording"):
            import soundcard as sc  # noqa: PLC0415

        if self.mic_device == "default":
            return sc.default_microphone()
        return sc.get_microphone(self.mic_device, include_loopback=False)

    def _resolve_monitor(self) -> Any:
        """The default (or configured) sink's monitor as a loopback microphone."""
        with require("capture", "Recording"):
            import soundcard as sc  # noqa: PLC0415

        if self.output_device == "default":
            speaker = sc.default_speaker()
        else:
            speaker = sc.get_speaker(self.output_device)
        # A sink's monitor loopback is matched by the speaker's name.
        return sc.get_microphone(speaker.name, include_loopback=True)

    # -- recording -------------------------------------------------------

    def _record(self, channel: str, device: Any, sink: list["np.ndarray"]) -> None:
        """soundcard capture loop for one channel, run on a background thread."""
        try:
            with device.recorder(
                samplerate=self.sample_rate, blocksize=_BLOCKSIZE
            ) as rec:
                while not self._stop.is_set():
                    sink.append(rec.record(numframes=_BLOCKSIZE))
        except BaseException as exc:  # noqa: BLE001 - reported from stop()
            # Record the error and stop the other channel too, so we fail loudly
            # rather than writing an empty (silently truncated) WAV.
            self._errors[channel] = exc
            self._stop.set()
            log.error(
                "Capture failed on %s channel (%s): %s", channel, device.name, exc
            )

    def start(self) -> None:
        self._stop.clear()
        self._errors.clear()
        self._mic_frames.clear()
        self._out_frames.clear()
        self._t0 = time.monotonic()

        # Resolve both devices up front so a bad device name fails fast, before
        # the user thinks they're recording.
        mic = self._resolve_mic()
        monitor = self._resolve_monitor()
        log.info("Recording mic=%s @ %d Hz via soundcard", mic.name, self.sample_rate)
        log.info(
            "Recording monitor=%s @ %d Hz via soundcard", monitor.name, self.sample_rate
        )

        self._threads = [
            threading.Thread(
                target=self._record, args=("self", mic, self._mic_frames), daemon=True
            ),
            threading.Thread(
                target=self._record,
                args=("others", monitor, self._out_frames),
                daemon=True,
            ),
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

        write_wav_16k_mono(self.self_wav, mic, self._mic_rate)
        write_wav_16k_mono(self.others_wav, out, self._out_rate)
        for label, arr in (("self", mic), ("others", out)):
            if arr.shape[0] == 0:
                log.warning(
                    "The %s channel captured no audio — its transcript will be empty.",
                    label,
                )
        log.info(
            "Captured %.1fs -> %s, %s", self._duration, self.self_wav, self.others_wav
        )
        return CaptureResult(
            self_wav=self.self_wav, others_wav=self.others_wav, duration=self._duration
        )

    def _capture_error(self) -> CaptureError:
        """Build an actionable error from the failed channel(s)."""
        channel = "self" if "self" in self._errors else "others"
        exc = self._errors[channel]
        if channel == "self":
            device = "microphone"
            hint = (
                "Pick a different input via [capture].mic_device in config.toml, "
                "or change the default source in your sound settings, then "
                "re-record. Run `hearhere devices` to list names."
            )
        else:
            device = "system-output (sink monitor)"
            hint = (
                "Check that PipeWire/PulseAudio is running and the sink has a "
                "monitor source. Try a different sink via [capture].output_device "
                "in config.toml, or change the default output device, then "
                "re-record. Run `hearhere devices` to list monitor names."
            )
        return CaptureError(
            f"Recording the {device} channel failed ({type(exc).__name__}: {exc}). "
            f"{hint}"
        )
