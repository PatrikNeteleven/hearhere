"""End-to-end Batch 2 test: diarization, summary, and speaker renaming.

Covers the "done when": a multi-speaker meeting yields distinct speakers plus a
``summary.md``, and renaming a speaker propagates to all exports. Uses fake
engines so no heavy deps (pyannote/torch/ollama) are needed.
"""

from __future__ import annotations

from hearhere.config import Config
from hearhere.models import Segment, SpeakerTurn, Summary
from hearhere.pipeline import artifacts
from hearhere.pipeline.orchestrator import process_meeting, rename_speakers


class FakeASR:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, wav_path: str, language: str | None):
        self.calls += 1
        if wav_path.endswith("self.wav"):
            return [Segment(start=0.0, end=3.0, text="Let's start.")]
        return [
            Segment(start=4.0, end=8.0, text="Staging is green."),
            Segment(start=9.0, end=13.0, text="Ship on Friday then."),
        ]


class FakeDiarizer:
    """Two distinct speakers on the others channel."""

    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, wav_path, *, min_speakers=0, max_speakers=0):
        self.calls += 1
        return [
            SpeakerTurn(start=4.0, end=8.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=9.0, end=13.0, speaker="SPEAKER_01"),
        ]


class FakeSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, transcript_text, *, language=None, tasks=None):
        self.calls += 1
        assert "Speaker 1" in transcript_text  # diarized labels reached the LLM
        return Summary(
            summary="Speaker 1 confirmed staging; the team will ship on Friday.",
            decisions=["Ship on Friday."],
            action_items=["Speaker 2 to prepare the release."],
        )


def _recorded(tmp_path):
    paths = artifacts.create_meeting_dir(tmp_path, "Weekly Sync")
    paths.self_wav.write_bytes(b"")
    paths.others_wav.write_bytes(b"")
    return paths


def _config() -> Config:
    return Config.model_validate({"export": {"formats": ["markdown", "json"]}})


def test_pipeline_yields_distinct_speakers_and_summary(tmp_path):
    paths = _recorded(tmp_path)
    diar = FakeDiarizer()
    summ = FakeSummarizer()

    meeting = process_meeting(
        paths.root, _config(), asr=FakeASR(), diarizer=diar, summarizer=summ,
        title="Weekly Sync",
    )

    assert diar.calls == 1
    assert summ.calls == 1
    assert meeting.speakers == ["Me", "Speaker 1", "Speaker 2"]
    assert meeting.summary is not None
    assert paths.export("summary.md").is_file()
    summary_md = paths.export("summary.md").read_text(encoding="utf-8")
    assert "Ship on Friday." in summary_md
    md = paths.export("transcript.md").read_text(encoding="utf-8")
    assert "Speaker 1:" in md and "Speaker 2:" in md


def test_diarization_disabled_leaves_others_unlabeled(tmp_path):
    paths = _recorded(tmp_path)
    cfg = Config.model_validate(
        {"diarization": {"enabled": False}, "llm": {"enabled": False},
         "export": {"formats": ["json"]}}
    )
    meeting = process_meeting(paths.root, cfg, asr=FakeASR())
    others = [s for s in meeting.transcript.segments if s.channel == "others"]
    assert others and all(s.speaker is None for s in others)
    assert meeting.summary is None
    assert not paths.export("summary.md").exists()


def test_rename_propagates_to_all_exports(tmp_path):
    paths = _recorded(tmp_path)
    cfg = _config()
    process_meeting(
        paths.root, cfg, asr=FakeASR(), diarizer=FakeDiarizer(),
        summarizer=FakeSummarizer(), title="Weekly Sync",
    )

    updated = rename_speakers(paths.root, cfg, {"Speaker 1": "Anna"})

    assert updated.speakers == ["Me", "Anna", "Speaker 2"]
    # meeting.json rewritten.
    reloaded = artifacts.read_meeting(paths)
    assert reloaded.speakers == ["Me", "Anna", "Speaker 2"]
    assert any(s.speaker == "Anna" for s in reloaded.transcript.segments)
    # Exports regenerated with the new name (transcript + summary).
    assert "Anna:" in paths.export("transcript.md").read_text(encoding="utf-8")
    assert "Anna confirmed" in paths.export("summary.md").read_text(encoding="utf-8")


def test_rename_unknown_speaker_is_noop(tmp_path):
    paths = _recorded(tmp_path)
    cfg = _config()
    process_meeting(paths.root, cfg, asr=FakeASR(), diarizer=FakeDiarizer(),
                    summarizer=FakeSummarizer(), title="Weekly Sync")
    updated = rename_speakers(paths.root, cfg, {"Nonexistent": "X"})
    assert updated.speakers == ["Me", "Speaker 1", "Speaker 2"]
