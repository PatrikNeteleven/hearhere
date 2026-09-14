"""Missing optional-dependency extras must fail with an install hint, not a traceback.

The capture backends and the heavy engines import lazily, so a venv installed
without the extras (a bare ``pip install -e .``) only breaks mid-run. These tests
pin the translation into :class:`MissingExtraError` at each lazy-import site; they
run on any box precisely because none of the extras are installed here.
"""

from __future__ import annotations

import pytest

from hearhere.capture.windows import WindowsCapture
from hearhere.engines.asr.parakeet_nemo import ParakeetNeMoEngine
from hearhere.engines.diarization.pyannote import PyannoteDiarizer
from hearhere.engines.llm.ollama import OllamaSummarizer
from hearhere.extras import MissingExtraError, require


def test_require_names_the_extra_and_the_install_command():
    with pytest.raises(MissingExtraError) as excinfo:
        with require("capture", "Recording"):
            import definitely_not_installed  # noqa: F401
    message = str(excinfo.value)
    assert "Recording needs the '[capture]' extra" in message
    assert 'pip install -e ".[capture]"' in message


def test_require_only_wraps_module_not_found():
    """Other failures pass through untouched — not every error is a missing extra."""
    with pytest.raises(ValueError):
        with require("asr", "Transcription"):
            raise ValueError("boom")


@pytest.mark.parametrize(
    ("call", "extra"),
    [
        (lambda: ParakeetNeMoEngine().load(), "asr"),
        (lambda: PyannoteDiarizer().load(), "diarization"),
        (lambda: OllamaSummarizer()._client(), "llm"),
    ],
)
def test_engines_report_their_extra(call, extra):
    with pytest.raises(MissingExtraError, match=rf"\[{extra}\]"):
        call()


@pytest.mark.parametrize("method", ["_start_mic_stream", "_resolve_loopback"])
def test_capture_reports_its_extra(tmp_path, method):
    capture = WindowsCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    with pytest.raises(MissingExtraError, match=r"\[capture\]"):
        getattr(capture, method)()
