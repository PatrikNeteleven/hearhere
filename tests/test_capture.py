"""Regression tests for real-hardware failure modes found on Windows.

Both fixes are for silent failures: a capture thread that dies leaving an empty
WAV (soundcard rejecting a device's WASAPI format), and ASR then crashing on that
empty WAV. Neither test needs numpy/soundcard/torch — the guards run before any
heavy import.
"""

from __future__ import annotations

import wave

from hearhere.capture.base import CaptureError
from hearhere.capture.linux import LinuxCapture
from hearhere.capture.windows import WindowsCapture, _mic_stream_params
from hearhere.engines.asr.parakeet_nemo import (
    ParakeetNeMoEngine,
    _wav_duration_seconds,
)


def _write_wav(path, *, frames: bytes = b"", rate: int = 16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return path


# --- ASR empty-audio guard -------------------------------------------------


def test_asr_skips_empty_wav_without_loading_model(tmp_path):
    wav = _write_wav(tmp_path / "self.wav")  # 0 frames
    engine = ParakeetNeMoEngine()
    # Returns [] and never touches NeMo (self._model stays None).
    assert engine.transcribe(str(wav)) == []
    assert engine._model is None


def test_asr_skips_missing_wav(tmp_path):
    engine = ParakeetNeMoEngine()
    assert engine.transcribe(str(tmp_path / "nope.wav")) == []


def test_wav_duration_helper(tmp_path):
    assert _wav_duration_seconds(str(tmp_path / "missing.wav")) == 0.0
    wav = _write_wav(tmp_path / "empty.wav")
    assert _wav_duration_seconds(str(wav)) == 0.0
    # 16000 frames @ 16 kHz == 1 second.
    full = _write_wav(tmp_path / "one_sec.wav", frames=b"\x00\x00" * 16000)
    assert abs(_wav_duration_seconds(str(full)) - 1.0) < 1e-6


# --- capture fails loudly instead of writing an empty WAV ------------------


def test_capture_raises_actionable_error_on_device_format_failure(tmp_path):
    cap = WindowsCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    # Simulate the soundcard WASAPI assertion on the mic thread.
    cap._errors["self"] = AssertionError()
    err = cap._capture_error()
    assert isinstance(err, CaptureError)
    msg = str(err)
    assert "microphone" in msg
    assert "mic_device" in msg  # points the user at the fix


def test_capture_error_generic_for_other_failures(tmp_path):
    cap = WindowsCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    cap._errors["others"] = RuntimeError("device busy")
    msg = str(cap._capture_error())
    assert "system-output" in msg
    assert "device busy" in msg


# --- Linux capture fails loudly instead of writing an empty WAV ------------


def test_linux_capture_raises_actionable_error_on_mic_failure(tmp_path):
    cap = LinuxCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    cap._errors["self"] = RuntimeError("no default source")
    err = cap._capture_error()
    assert isinstance(err, CaptureError)
    msg = str(err)
    assert "microphone" in msg
    assert "mic_device" in msg  # points the user at the fix


def test_linux_capture_error_names_monitor_for_output_failure(tmp_path):
    cap = LinuxCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    cap._errors["others"] = RuntimeError("no monitor source")
    msg = str(cap._capture_error())
    assert "sink monitor" in msg
    assert "output_device" in msg
    assert "no monitor source" in msg


def test_linux_capture_stop_without_start_writes_empty_wavs(tmp_path):
    # No frames captured and no errors -> two empty (0-frame) 16 kHz mono WAVs,
    # rather than a crash. Mirrors a recording that captured pure silence.
    # stop() resamples + writes, so it needs the [capture] extra (numpy/soundfile);
    # skip on the light CI install that has neither.
    import pytest  # noqa: PLC0415

    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    self_wav = tmp_path / "self.wav"
    others_wav = tmp_path / "others.wav"
    cap = LinuxCapture(self_wav, others_wav)
    result = cap.stop()
    assert result.self_wav == self_wav
    assert result.others_wav == others_wav
    for path in (self_wav, others_wav):
        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 0


# --- mic stream parameter selection (sounddevice) --------------------------


def test_mic_stream_params_uses_native_rate_and_channels():
    rate, channels = _mic_stream_params(
        {"default_samplerate": 48000.0, "max_input_channels": 2}
    )
    assert rate == 48000
    assert channels == 2


def test_mic_stream_params_caps_channels_and_falls_back_on_rate():
    # 18-input console -> capped to 2; missing/zero rate -> 16 kHz fallback.
    rate, channels = _mic_stream_params(
        {"default_samplerate": 0, "max_input_channels": 18}
    )
    assert rate == 16000
    assert channels == 2


def test_mic_stream_params_defaults_for_empty_info():
    rate, channels = _mic_stream_params({})
    assert rate == 16000
    assert channels == 1
