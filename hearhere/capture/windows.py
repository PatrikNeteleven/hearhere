"""Windows dual-channel capture (priority-1 platform).

- **Mic** (``self``): the default (or configured) input device, captured with
  `sounddevice <https://python-sounddevice.readthedocs.io>`_ (PortAudio). This is
  more tolerant of pro/USB audio interfaces (e.g. Focusrite Scarlett) than
  soundcard's WASAPI shared-mode path, which asserts on some devices' mix format.
- **System output** (``others``): WASAPI **loopback** of the default render
  device via `soundcard <https://github.com/bastibe/SoundCard>`_ — recording
  exactly what you hear, with no virtual cable needed. soundcard exposes loopback
  as a "microphone", which PortAudio cannot do, so the two backends are split.

Each channel is captured at its device's native sample rate and downmixed +
resampled to 16 kHz mono on stop (Parakeet's required input).

``sounddevice``/``soundcard``/``numpy`` are imported lazily, so importing
HearHere on any platform (including this WSL2 dev box, which has no audio) never
needs them. Install with ``pip install "hearhere[capture]"`` on Windows.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from ..logging_setup import get_logger
from .base import (
    TARGET_SAMPLE_RATE,
    AudioCapture,
    CaptureError,
    CaptureResult,
    write_wav_16k_mono,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

log = get_logger("capture.windows")

# Frames pulled per read; a small block keeps stop() responsive.
_BLOCKSIZE = 1024
# Cap captured mic channels — multi-input interfaces (e.g. a 2i2) are downmixed
# to mono anyway, and we don't need to grab an 18-input console's every channel.
_MAX_MIC_CHANNELS = 2


def _mic_stream_params(info: dict) -> tuple[int, int]:
    """Pick (samplerate, channels) for the mic from a sounddevice device info dict.

    Recording at the device's native rate avoids WASAPI format negotiation; we
    resample to 16 kHz ourselves on write.
    """
    rate = int(round(info.get("default_samplerate") or 0)) or TARGET_SAMPLE_RATE
    max_ch = int(info.get("max_input_channels") or 1)
    channels = max(1, min(max_ch, _MAX_MIC_CHANNELS))
    return rate, channels


class WindowsCapture(AudioCapture):
    """Records mic + WASAPI loopback into ``self.wav`` / ``others.wav``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._mic_frames: list["np.ndarray"] = []
        self._out_frames: list["np.ndarray"] = []
        self._mic_stream: Any = None  # sounddevice InputStream
        # Native capture rates per channel (mic set when its stream opens).
        self._mic_rate: int = self.sample_rate
        self._out_rate: int = self.sample_rate
        # Per-channel capture errors, surfaced on stop().
        self._errors: dict[str, BaseException] = {}
        self._t0: float = 0.0
        self._duration: float = 0.0

    # -- device resolution -----------------------------------------------

    def _resolve_loopback(self) -> Any:
        """The default speaker as a loopback 'microphone'."""
        import soundcard as sc  # noqa: PLC0415

        if self.output_device == "default":
            speaker = sc.default_speaker()
        else:
            speaker = sc.get_speaker(self.output_device)
        # Loopback capture of a render device is exposed as a microphone.
        return sc.get_microphone(speaker.name, include_loopback=True)

    # -- recording -------------------------------------------------------

    def _start_mic_stream(self) -> Any:
        """Open and start the mic InputStream via sounddevice (raises on failure)."""
        import sounddevice as sd  # noqa: PLC0415

        device = None if self.mic_device == "default" else self.mic_device
        try:
            info = sd.query_devices(device, "input")
            rate, channels = _mic_stream_params(dict(info))
            self._mic_rate = rate

            def _callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
                if status:  # pragma: no cover - hardware-dependent
                    log.debug("mic stream status: %s", status)
                self._mic_frames.append(indata.copy())

            stream = sd.InputStream(
                device=device,
                samplerate=rate,
                channels=channels,
                dtype="float32",
                blocksize=_BLOCKSIZE,
                callback=_callback,
            )
            stream.start()
            log.info(
                "Recording mic=%s @ %d Hz (%d ch) via sounddevice",
                info.get("name", device or "default"), rate, channels,
            )
            return stream
        except Exception as exc:  # noqa: BLE001 - translated to CaptureError
            self._errors["self"] = exc
            log.error("Could not open microphone (%s): %s", device or "default", exc)
            raise self._capture_error()

    def _record(self, channel: str, device: Any, sink: list["np.ndarray"]) -> None:
        """Loopback capture loop (soundcard), run on a background thread."""
        try:
            with device.recorder(
                samplerate=self.sample_rate, blocksize=_BLOCKSIZE
            ) as rec:
                while not self._stop.is_set():
                    sink.append(rec.record(numframes=_BLOCKSIZE))
        except BaseException as exc:  # noqa: BLE001 - reported from stop()
            # AssertionError here is soundcard rejecting the device's shared-mode
            # WASAPI format; record it and stop so we fail loudly rather than
            # writing an empty WAV.
            self._errors[channel] = exc
            self._stop.set()
            log.error("Capture failed on %s channel (%s): %s", channel, device.name, exc)

    def start(self) -> None:
        self._stop.clear()
        self._errors.clear()
        self._mic_frames.clear()
        self._out_frames.clear()
        self._t0 = time.monotonic()

        # Mic first: fail fast (before the user "records") if it can't open.
        self._mic_stream = self._start_mic_stream()

        loopback = self._resolve_loopback()
        self._out_rate = self.sample_rate
        log.info("Recording loopback=%s @ %d Hz via soundcard", loopback.name, self.sample_rate)
        self._threads = [
            threading.Thread(
                target=self._record, args=("others", loopback, self._out_frames), daemon=True
            ),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> CaptureResult:
        self._stop.set()
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                log.debug("Error closing mic stream: %s", exc)
            self._mic_stream = None
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
        log.info("Captured %.1fs -> %s, %s", self._duration, self.self_wav, self.others_wav)
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
                "or change the Windows default microphone (Settings -> System -> "
                "Sound), then re-record. Run `hearhere devices` to list names."
            )
        else:
            device = "system-output (loopback)"
            base = (
                "This output device is not compatible with soundcard's WASAPI "
                "shared-mode loopback."
                if isinstance(exc, AssertionError)
                else "Check the output device is available and not exclusively held."
            )
            hint = (
                f"{base} Try a different device via [capture].output_device in "
                "config.toml, or change the Windows default playback device, then "
                "re-record."
            )
        return CaptureError(
            f"Recording the {device} channel failed ({type(exc).__name__}: {exc}). {hint}"
        )
