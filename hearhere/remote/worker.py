"""Remote worker: run the HearHere pipeline on a GPU host (e.g. RunPod).

Two layers live here, kept apart on purpose:

* :class:`JobManager` — the pure orchestration core. It owns a temp workspace,
  writes uploaded WAVs into meeting folders, runs the pipeline in a background
  thread, and tracks job state. It imports no web framework, so it is unit
  testable without the ``[remote]`` extra.
* :func:`create_app` / :func:`run_worker` — a thin FastAPI wrapper around a
  :class:`JobManager`. FastAPI/uvicorn are imported lazily; install them with
  ``pip install "hearhere[remote]"``.

The worker always runs the pipeline with ``backend = "local"`` (the heavy models
live here), regardless of the client's config.

.. note::
   This module intentionally does **not** use ``from __future__ import
   annotations``: FastAPI resolves endpoint parameter annotations at route
   registration, and the web-stack types (``UploadFile`` etc.) are imported
   lazily inside :func:`create_app`, so stringized annotations could not be
   resolved. Runtime ``X | None`` unions are fine on the supported Python 3.10+.
"""

import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from .. import __version__
from ..config import Config
from ..logging_setup import get_logger
from ..models import Meeting
from ..pipeline import artifacts
from . import protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

log = get_logger("remote.worker")

# Signature of the pipeline entry point (``orchestrator.process_meeting``),
# injectable so the worker is testable with a fake pipeline.
ProcessFn = Callable[..., Meeting]


@dataclass
class Job:
    """One remote processing job and its result."""

    id: str
    root: Path
    state: protocol.JobState = "pending"
    error: Optional[str] = None
    meeting: Optional[dict[str, Any]] = None

    def status(self) -> protocol.JobStatus:
        return protocol.JobStatus(id=self.id, state=self.state, error=self.error)


class JobError(Exception):
    """Raised by the manager for client-visible job problems."""

    def __init__(self, message: str, *, code: int = 400) -> None:
        super().__init__(message)
        self.code = code


class JobManager:
    """Accepts WAV uploads, runs the pipeline, and tracks results in memory."""

    def __init__(
        self,
        config: Config,
        *,
        process_fn: ProcessFn | None = None,
        asr: object | None = None,
        workdir: str | Path | None = None,
    ) -> None:
        # The worker is the compute host: force local processing here.
        self.config = config.model_copy(deep=True)
        self.config.compute.backend = "local"
        self._process_fn = process_fn
        self._asr = asr
        self._workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="hearhere-worker-"))
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # -- full pipeline (async job) --------------------------------------

    def submit(
        self,
        self_wav: bytes,
        others_wav: bytes,
        *,
        title: str | None = None,
        language: str | None = None,
    ) -> Job:
        """Store the uploaded audio and start processing in the background."""
        job_id = uuid.uuid4().hex[:12]
        # One folder per job id — collision-free even for same-titled meetings.
        paths = artifacts.resolve_meeting(self._workdir / job_id)
        paths.audio_dir.mkdir(parents=True, exist_ok=True)
        paths.self_wav.write_bytes(self_wav)
        paths.others_wav.write_bytes(others_wav)

        job = Job(id=job_id, root=paths.root)
        with self._lock:
            self._jobs[job_id] = job

        log.info("Accepted job %s (%d + %d bytes)", job_id, len(self_wav), len(others_wav))
        thread = threading.Thread(
            target=self._run, args=(job, title, language), daemon=True
        )
        thread.start()
        return job

    def _run(self, job: Job, title: str | None, language: str | None) -> None:
        with self._lock:
            job.state = "running"
        try:
            cfg = self.config
            if language:
                cfg = cfg.model_copy(deep=True)
                cfg.general.language = language
            meeting = self._pipeline()(job.root, cfg, asr=self._asr, title=title)
            with self._lock:
                job.meeting = meeting.model_dump(mode="json")
                job.state = "done"
            log.info("Job %s done", job.id)
        except Exception as exc:  # pragma: no cover - exercised via fake failures
            with self._lock:
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
            log.exception("Job %s failed", job.id)

    def _pipeline(self) -> ProcessFn:
        if self._process_fn is not None:
            return self._process_fn
        from ..pipeline.orchestrator import process_meeting  # noqa: PLC0415

        return process_meeting

    def status(self, job_id: str) -> protocol.JobStatus:
        return self._job(job_id).status()

    def result(self, job_id: str) -> dict[str, Any]:
        """Return the finished meeting dict, or raise :class:`JobError`."""
        job = self._job(job_id)
        if job.state == "error":
            raise JobError(job.error or "job failed", code=500)
        if job.state != "done" or job.meeting is None:
            raise JobError("job not finished", code=409)
        return job.meeting

    def _job(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobError(f"unknown job {job_id!r}", code=404)
        return job

    # -- ASR only (synchronous) -----------------------------------------

    def transcribe(self, wav: bytes, language: str | None = None):
        """Run ASR on a single uploaded WAV and return its segments."""
        engine = self._asr
        if engine is None:
            from ..engines.asr.base import create_asr_engine  # noqa: PLC0415

            engine = create_asr_engine(self.config)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=self._workdir, delete=False
        ) as fh:
            fh.write(wav)
            wav_path = fh.name
        try:
            return engine.transcribe(wav_path, language)  # type: ignore[attr-defined]
        finally:
            Path(wav_path).unlink(missing_ok=True)


# --- FastAPI wrapper -------------------------------------------------------


def create_app(
    config: Config,
    *,
    manager: JobManager | None = None,
    token: str | None = None,
) -> "FastAPI":
    """Build the FastAPI worker app around a :class:`JobManager`.

    ``token`` (falling back to ``config.compute.remote.token``) gates every
    endpoint except ``/health``. An empty token means the worker is open — a
    warning is logged, since it will process any audio sent to it.
    """
    from fastapi import (  # noqa: PLC0415
        Depends,
        FastAPI,
        File,
        Form,
        Header,
        HTTPException,
        Response,
        UploadFile,
    )

    mgr = manager or JobManager(config)
    auth_token = token if token is not None else config.compute.remote.token
    if not auth_token:
        log.warning("Worker running WITHOUT an auth token — anyone can submit audio.")

    app = FastAPI(title="HearHere worker", version=__version__)

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not auth_token:
            return
        expected = protocol.bearer(auth_token)
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid or missing token")

    @app.get(protocol.HEALTH, response_model=protocol.HealthResponse)
    def health() -> protocol.HealthResponse:
        return protocol.HealthResponse(version=__version__, backend="local")

    @app.post(protocol.JOBS, status_code=202, response_model=protocol.JobStatus)
    async def submit_job(
        self_wav: UploadFile = File(...),
        others_wav: UploadFile = File(...),
        title: str | None = Form(default=None),
        language: str | None = Form(default=None),
        _: None = Depends(require_auth),
    ) -> protocol.JobStatus:
        job = mgr.submit(
            await self_wav.read(),
            await others_wav.read(),
            title=title,
            language=language,
        )
        return job.status()

    @app.get(protocol.JOBS + "/{job_id}", response_model=protocol.JobStatus)
    def job_status(job_id: str, _: None = Depends(require_auth)) -> protocol.JobStatus:
        try:
            return mgr.status(job_id)
        except JobError as exc:
            raise HTTPException(status_code=exc.code, detail=str(exc))

    @app.get(protocol.JOBS + "/{job_id}/result")
    def job_result(job_id: str, _: None = Depends(require_auth)) -> Response:
        try:
            meeting = mgr.result(job_id)
        except JobError as exc:
            raise HTTPException(status_code=exc.code, detail=str(exc))
        return Response(
            content=Meeting.model_validate(meeting).model_dump_json(),
            media_type="application/json",
        )

    @app.post(protocol.TRANSCRIBE, response_model=protocol.TranscriptionResult)
    async def transcribe(
        wav: UploadFile = File(...),
        language: str | None = Form(default=None),
        _: None = Depends(require_auth),
    ) -> protocol.TranscriptionResult:
        segments = mgr.transcribe(await wav.read(), language)
        return protocol.TranscriptionResult(segments=list(segments), language=language)

    return app


def run_worker(
    config: Config,
    *,
    host: str = "0.0.0.0",
    port: int = 8808,
    token: str | None = None,
) -> None:  # pragma: no cover - starts a server
    """Serve the worker with uvicorn (blocking)."""
    import uvicorn  # noqa: PLC0415

    app = create_app(config, token=token)
    log.info("Starting HearHere worker on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
