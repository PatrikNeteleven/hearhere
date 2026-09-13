"""Shared HTTP contract between the remote worker and its clients.

Keeping the endpoint paths, header names, and payload shapes in one module means
:mod:`hearhere.remote.worker` (server) and :mod:`hearhere.remote.client` (the
laptop) can never drift out of sync. Only stdlib + pydantic are imported here so
the contract is available without the optional ``[remote]`` web stack installed.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..models import Meeting, Segment

# --- endpoints -------------------------------------------------------------

HEALTH = "/health"
JOBS = "/jobs"
TRANSCRIBE = "/transcribe"


def job_path(job_id: str) -> str:
    """Path for a single job's status."""
    return f"{JOBS}/{job_id}"


def result_path(job_id: str) -> str:
    """Path for a finished job's ``meeting.json`` result."""
    return f"{JOBS}/{job_id}/result"


# --- auth ------------------------------------------------------------------

AUTH_HEADER = "Authorization"
BEARER_PREFIX = "Bearer "


def bearer(token: str) -> str:
    """Format a token as an ``Authorization`` header value."""
    return f"{BEARER_PREFIX}{token}"


# --- job lifecycle ---------------------------------------------------------

JobState = Literal["pending", "running", "done", "error"]

#: States from which no further transition happens.
TERMINAL_STATES: frozenset[str] = frozenset({"done", "error"})


class JobStatus(BaseModel):
    """A job's current state, as returned by ``GET /jobs/{id}``."""

    id: str
    state: JobState
    error: Optional[str] = None


class TranscriptionResult(BaseModel):
    """The payload returned by ``POST /transcribe`` (ASR-only)."""

    segments: list[Segment] = Field(default_factory=list)
    language: Optional[str] = None


class HealthResponse(BaseModel):
    """The payload returned by ``GET /health``."""

    status: str = "ok"
    version: str
    backend: str = "local"


# Re-exported so clients can validate a downloaded result without importing
# from deep in the models package.
MeetingResult = Meeting


# --- the privacy warning ---------------------------------------------------

#: Shown before any audio is uploaded. HearHere is local-first; the remote
#: backend is the one place audio leaves the machine, so the warning is loud.
UPLOAD_WARNING = (
    "⚠  Remote backend: your microphone and system-audio recordings will be "
    "UPLOADED to the remote worker at {url} for processing. Audio leaves this "
    "machine. Only use a worker you trust and control."
)


def upload_warning(url: str) -> str:
    """Render :data:`UPLOAD_WARNING` for a specific worker URL."""
    return UPLOAD_WARNING.format(url=url)
