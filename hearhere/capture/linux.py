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
``libpulse0`` package — soundcard talks to PulseAudio directly, it does not use
PortAudio here — and a running PipeWire/PulseAudio session).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..extras import MissingExtraError, require
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
# How long stop() waits for a capture thread to drain before giving up on it. A
# yanked USB device can wedge soundcard's read loop without ever failing, and we
# must not block the user's Enter/Ctrl-C — and lose the recording — forever.
_STOP_JOIN_TIMEOUT = 5.0


def _device_label(device: Any) -> str:
    """A log-safe device name that never round-trips to a (possibly gone) source.

    ``_Microphone.name`` re-queries PulseAudio, which raises once the device has
    disappeared — exactly when we're logging the failure that caused it — so fall
    back to the cached id, which is a plain stored string.
    """
    try:
        return device.name
    except Exception:  # noqa: BLE001 - name lookup is best-effort in error paths
        return getattr(device, "id", None) or repr(device)


def _verify_requested(requested: str, device: Any, kind: str) -> None:
    """Reject soundcard's loose fuzzy fallback for an explicitly named device.

    ``_match_soundcard`` falls back to a ``'.*'.join(id)`` regex that matches
    almost anything, so a typo'd name silently resolves to some arbitrary
    device. Require the requested string to actually appear in the resolved
    name or id; otherwise it was a fuzzy guess and we fail fast instead.
    """
    needle = requested.lower()
    haystacks: list[str] = []
    for attr in ("name", "id"):
        try:
            value = getattr(device, attr)
        except Exception:  # noqa: BLE001 - best effort
            continue
        if value:
            haystacks.append(str(value).lower())
    if not any(needle in h for h in haystacks):
        raise CaptureError(
            f"No {kind} matches [capture] device {requested!r}; the closest was "
            f"{_device_label(device)!r}. Run `hearhere devices` and copy an exact "
            "name."
        )


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
        mic = sc.get_microphone(self.mic_device, include_loopback=False)
        _verify_requested(self.mic_device, mic, "microphone")
        return mic

    def _resolve_monitor(self) -> Any:
        """The default (or configured) sink's monitor as a loopback microphone."""
        with require("capture", "Recording"):
            import soundcard as sc  # noqa: PLC0415

        if self.output_device == "default":
            speaker = sc.default_speaker()
        else:
            speaker = sc.get_speaker(self.output_device)
            _verify_requested(self.output_device, speaker, "output device")
        # PulseAudio/PipeWire names each sink's loopback source "<sink id>.monitor".
        # Match it by that exact id: matching by the sink's *description* (which is
        # what speaker.name returns) would fall through to soundcard's substring
        # search, which on common hardware (USB headsets, built-ins) latches onto
        # the real mic source that shares the description verbatim — silently
        # recording the microphone into others.wav. The exact-id lookup can't.
        mic = sc.get_microphone(speaker.id + ".monitor", include_loopback=True)
        if not mic.isloopback:
            raise CaptureError(
                f"Resolved {_device_label(mic)!r} as the system-output monitor, "
                "but it is not a loopback source — refusing to record the "
                "microphone into others.wav. Check the sink has a monitor source, "
                "or set [capture].output_device (run `hearhere devices` for sink "
                "names)."
            )
        return mic

    def _resolve(self, channel: str, resolve: Callable[[], Any]) -> Any:
        """Resolve a device, translating any failure into a CaptureError.

        ``require()`` only maps the missing-extra case; but merely ``import
        soundcard`` can raise (OSError when libpulse is absent, AssertionError on
        a headless box), and ``get_microphone`` raises IndexError on an unknown
        device. Left unwrapped these escape ``cli.record`` as raw tracebacks, so
        catch them here — as WindowsCapture already does for its mic path.
        """
        try:
            return resolve()
        except (MissingExtraError, CaptureError):
            raise  # already actionable
        except BaseException as exc:  # noqa: BLE001 - translated to CaptureError
            self._errors[channel] = exc
            log.error("Could not resolve %s device: %s", channel, exc)
            raise self._capture_error() from exc

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
            # Record the error for stop() to surface, but do NOT stop the other
            # channel: a mic failure must not silently kill an otherwise-healthy
            # system-output recording while the user still thinks both are going.
            self._errors[channel] = exc
            log.error(
                "Capture failed on %s channel (%s): %s",
                channel, _device_label(device), exc,
            )

    def start(self) -> None:
        self._stop.clear()
        self._errors.clear()
        self._mic_frames.clear()
        self._out_frames.clear()
        self._t0 = time.monotonic()

        # Resolve both devices up front so a bad device name (or missing
        # PulseAudio) fails fast, before the user thinks they're recording — and
        # as a CaptureError, not a raw traceback.
        mic = self._resolve("self", self._resolve_mic)
        monitor = self._resolve("others", self._resolve_monitor)
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
            t.join(timeout=_STOP_JOIN_TIMEOUT)
            if t.is_alive():
                # soundcard's read loop can wedge on a yanked device without ever
                # raising; don't block on it forever. It's a daemon thread and
                # won't append once its backing device is gone, so proceed with
                # what we've already captured.
                log.warning(
                    "Capture thread %s did not stop within %.0fs; "
                    "proceeding with the audio captured so far.",
                    t.name, _STOP_JOIN_TIMEOUT,
                )
        # Only count elapsed time if we actually started; otherwise _t0 is 0.0
        # and subtracting it would report the machine's uptime as the duration.
        self._duration = time.monotonic() - self._t0 if self._t0 else 0.0
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

        # Persist whatever each channel captured BEFORE surfacing any error, so a
        # failure on one channel never discards a healthy channel's audio — the
        # user can still `hearhere process` the recovered folder.
        write_wav_16k_mono(self.self_wav, mic, self._mic_rate)
        write_wav_16k_mono(self.others_wav, out, self._out_rate)
        for label, arr in (("self", mic), ("others", out)):
            if arr.shape[0] == 0:
                log.warning(
                    "The %s channel captured no audio — its transcript will be empty.",
                    label,
                )

        # Fail loudly if a channel's recorder crashed — but only now, with the
        # good channel already written to disk, never mistaking a failed capture
        # for a silent meeting.
        if self._errors:
            raise self._capture_error()

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
                "re-record. Run `hearhere devices` to list sink names."
            )
        return CaptureError(
            f"Recording the {device} channel failed ({type(exc).__name__}: {exc}). "
            f"{hint}"
        )
