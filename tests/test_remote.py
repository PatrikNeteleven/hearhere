"""Batch 5 tests: remote worker + client, ASR-over-HTTP, consent, orchestrator.

The worker's HTTP layer is exercised in-process with ``httpx.ASGITransport`` and
a fake pipeline (no torch/pyannote/ollama). Covers the "done when": a full
meeting is processed remotely (distinct speakers + summary), and the
leaving-your-machine warning is surfaced before upload.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from hearhere.config import Config
from hearhere.engines.asr.remote import RemoteASREngine
from hearhere.models import Meeting, Segment, SpeakerTurn, Summary, Transcript
from hearhere.pipeline import artifacts
from hearhere.pipeline.orchestrator import process_meeting
from hearhere.remote import consent, protocol
from hearhere.remote.client import RemoteClient, RemoteError
from hearhere.remote.worker import JobManager, create_app


# --- fake engines (no heavy deps) -----------------------------------------


class FakeASR:
    def transcribe(self, wav_path: str, language: str | None):
        if wav_path.endswith("self.wav"):
            return [Segment(start=0.0, end=3.0, text="Let's start.")]
        return [
            Segment(start=4.0, end=8.0, text="Staging is green."),
            Segment(start=9.0, end=13.0, text="Ship on Friday."),
        ]


class FakeDiarizer:
    def diarize(self, wav_path, *, min_speakers=0, max_speakers=0):
        return [
            SpeakerTurn(start=4.0, end=8.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=9.0, end=13.0, speaker="SPEAKER_01"),
        ]


class FakeSummarizer:
    def summarize(self, transcript_text, *, language=None, tasks=None):
        return Summary(
            summary="The team confirmed staging and will ship Friday.",
            decisions=["Ship on Friday."],
            action_items=["Speaker 2 to prepare the release."],
        )


def _full_pipeline(root, cfg, *, asr=None, title=None):
    """Worker-side pipeline entry point wired with fakes."""
    return process_meeting(
        root, cfg, asr=FakeASR(), diarizer=FakeDiarizer(),
        summarizer=FakeSummarizer(), title=title,
    )


def _worker_config() -> Config:
    return Config.model_validate(
        {"diarization": {"enabled": False}, "llm": {"enabled": False},
         "export": {"formats": ["json"]}}
    )


def _asgi_client(app, token: str = "") -> RemoteClient:
    http = TestClient(app)  # sync httpx.Client that drives the ASGI app in a thread
    return RemoteClient("http://testserver", token=token or None, http_client=http, poll_interval=0.01)


def _dummy_wavs(tmp_path):
    s = tmp_path / "self.wav"
    o = tmp_path / "others.wav"
    s.write_bytes(b"RIFFself")
    o.write_bytes(b"RIFFothers")
    return s, o


# --- protocol --------------------------------------------------------------


def test_protocol_paths_and_bearer():
    assert protocol.job_path("abc") == "/jobs/abc"
    assert protocol.result_path("abc") == "/jobs/abc/result"
    assert protocol.bearer("t0k") == "Bearer t0k"
    assert "leaves this machine" in protocol.upload_warning("http://host").lower() \
        or "leaves this" in protocol.upload_warning("http://host")
    assert "http://host" in protocol.upload_warning("http://host")


# --- JobManager (no HTTP) --------------------------------------------------


def test_jobmanager_runs_full_pipeline(tmp_path):
    mgr = JobManager(_worker_config(), process_fn=_full_pipeline, workdir=tmp_path)
    job = mgr.submit(b"a", b"b", title="Weekly Sync")
    _wait_terminal(mgr, job.id)
    result = mgr.result(job.id)
    meeting = Meeting.model_validate(result)
    assert meeting.speakers == ["Me", "Speaker 1", "Speaker 2"]
    assert meeting.summary is not None


def test_jobmanager_forces_local_backend():
    cfg = Config.model_validate({"compute": {"backend": "remote"}})
    mgr = JobManager(cfg)
    assert mgr.config.compute.backend == "local"


def test_jobmanager_error_is_reported(tmp_path):
    def boom(root, cfg, *, asr=None, title=None):
        raise RuntimeError("kaboom")

    mgr = JobManager(_worker_config(), process_fn=boom, workdir=tmp_path)
    job = mgr.submit(b"a", b"b")
    _wait_terminal(mgr, job.id)
    assert mgr.status(job.id).state == "error"
    with pytest.raises(Exception):
        mgr.result(job.id)


def test_jobmanager_transcribe_uses_asr(tmp_path):
    mgr = JobManager(_worker_config(), asr=FakeASR(), workdir=tmp_path)
    segs = mgr.transcribe(b"RIFF....", language="en")
    assert len(segs) == 2  # temp name != self.wav -> "others" segments


def _wait_terminal(mgr, job_id, tries=500):
    import time

    for _ in range(tries):
        if mgr.status(job_id).state in protocol.TERMINAL_STATES:
            return
        time.sleep(0.005)
    raise AssertionError("job did not finish")


# --- HTTP: worker + client end to end -------------------------------------


def test_remote_process_end_to_end(tmp_path):
    mgr = JobManager(_worker_config(), process_fn=_full_pipeline, workdir=tmp_path)
    app = create_app(_worker_config(), manager=mgr, token="secret")
    client = _asgi_client(app, token="secret")

    assert client.health().status == "ok"

    states: list[str] = []
    s, o = _dummy_wavs(tmp_path)
    meeting = client.process(
        s, o, title="Weekly Sync", on_status=lambda st: states.append(st.state),
        max_wait=10,
    )
    assert meeting.speakers == ["Me", "Speaker 1", "Speaker 2"]
    assert meeting.summary and meeting.summary.decisions == ["Ship on Friday."]
    assert "done" in states


def test_remote_auth_required(tmp_path):
    mgr = JobManager(_worker_config(), process_fn=_full_pipeline, workdir=tmp_path)
    app = create_app(_worker_config(), manager=mgr, token="secret")
    client = _asgi_client(app, token="")  # no token
    s, o = _dummy_wavs(tmp_path)
    with pytest.raises(RemoteError) as exc:
        client.submit(s, o)
    assert "401" in str(exc.value)


def test_remote_unknown_job_404(tmp_path):
    mgr = JobManager(_worker_config(), process_fn=_full_pipeline, workdir=tmp_path)
    app = create_app(_worker_config(), manager=mgr, token="")
    client = _asgi_client(app)
    with pytest.raises(RemoteError) as exc:
        client.status("nope")
    assert "404" in str(exc.value)


def test_remote_result_not_ready_409(tmp_path):
    # A pipeline that blocks until released, so the job stays "running".
    import threading

    gate = threading.Event()

    def slow(root, cfg, *, asr=None, title=None):
        gate.wait(5)
        return _full_pipeline(root, cfg, title=title)

    mgr = JobManager(_worker_config(), process_fn=slow, workdir=tmp_path)
    app = create_app(_worker_config(), manager=mgr, token="")
    client = _asgi_client(app)
    s, o = _dummy_wavs(tmp_path)
    job_id = client.submit(s, o)
    with pytest.raises(RemoteError) as exc:
        client.result(job_id)
    assert "409" in str(exc.value)
    gate.set()


# --- remote ASR engine over HTTP ------------------------------------------


def test_remote_asr_engine_transcribes_over_http(tmp_path):
    mgr = JobManager(_worker_config(), asr=FakeASR(), workdir=tmp_path)
    app = create_app(_worker_config(), manager=mgr, token="")
    engine = RemoteASREngine("http://test")
    engine._client = _asgi_client(app)  # swap in the in-process transport

    wav = tmp_path / "chunk.wav"
    wav.write_bytes(b"RIFF....")
    segs = engine.transcribe(str(wav), "en")
    assert len(segs) == 2


# --- orchestrator remote branch -------------------------------------------


def test_orchestrator_remote_branch_reanchors(tmp_path, monkeypatch):
    paths = artifacts.create_meeting_dir(tmp_path, "Weekly Sync")
    paths.self_wav.write_bytes(b"")
    paths.others_wav.write_bytes(b"")
    cfg = Config.model_validate(
        {"compute": {"backend": "remote", "remote": {"url": "http://gpu:8808"}},
         "export": {"formats": ["json"]}}
    )

    captured: dict = {}

    class FakeRemoteClient:
        def __init__(self, url, token=None, **kw):
            captured["url"] = url
            captured["token"] = token

        def process(self, s, o, *, title=None, language=None, on_status=None, max_wait=None):
            captured["files"] = (str(s), str(o))
            return Meeting(
                id="worker-temp-id", title="Remote",
                speakers=["Me", "Speaker 1"],
                transcript=Transcript(
                    segments=[Segment(start=0, end=1, text="Hi", speaker="Me")]
                ),
                summary=Summary(summary="done"),
            )

    from hearhere.remote import client as client_mod

    monkeypatch.setattr(client_mod, "RemoteClient", FakeRemoteClient)

    meeting = process_meeting(paths.root, cfg, title="Weekly Sync")

    assert captured["url"] == "http://gpu:8808"
    assert meeting.id == paths.root.name  # re-anchored to the local folder
    assert meeting.title == "Weekly Sync"
    assert paths.meeting_json.is_file()
    assert paths.export("summary.md").is_file()  # exports happened locally


# --- consent ---------------------------------------------------------------


def test_consent_record_and_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert not consent.has_consent()
    marker = consent.record_consent("http://gpu:8808")
    assert marker.is_file()
    assert consent.has_consent()
    assert "http://gpu:8808" in marker.read_text(encoding="utf-8")


def test_remote_token_from_env(monkeypatch):
    monkeypatch.setenv("HEARHERE_REMOTE_TOKEN", "envtok")
    cfg = Config()
    assert cfg.compute.remote.token == "envtok"
