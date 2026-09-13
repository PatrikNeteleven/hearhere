"""Client for the remote HearHere worker (the laptop side).

:class:`RemoteClient` uploads ``self.wav``/``others.wav`` to a worker, polls the
job to completion, and downloads the finished :class:`Meeting`. ``httpx`` is
imported lazily (``pip install "hearhere[remote]"``); an ``http_client`` can be
injected for testing against an in-process app.

Uploading audio is the one place HearHere's data leaves the machine, so every
upload logs :func:`hearhere.remote.protocol.upload_warning` at WARNING level.
The interactive *confirmation* before the first upload lives in the CLI
(see :mod:`hearhere.remote.consent`); the client always at least warns.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from ..logging_setup import get_logger
from ..models import Meeting, Segment
from . import protocol

log = get_logger("remote.client")

StatusCallback = Callable[[protocol.JobStatus], None]


class RemoteError(RuntimeError):
    """A remote job failed or the worker returned an error."""


class RemoteClient:
    """Talks to a remote worker over HTTP."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        *,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
        http_client: Any | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token or ""
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._external = http_client

    # -- transport ------------------------------------------------------

    @contextmanager
    def _session(self) -> Iterator[Any]:
        if self._external is not None:
            yield self._external
            return
        import httpx  # noqa: PLC0415

        client = httpx.Client(base_url=self.url, timeout=self.timeout)
        try:
            yield client
        finally:
            client.close()

    def _headers(self) -> dict[str, str]:
        return {protocol.AUTH_HEADER: protocol.bearer(self.token)} if self.token else {}

    # -- endpoints ------------------------------------------------------

    def health(self) -> protocol.HealthResponse:
        """Check the worker is reachable and responding."""
        with self._session() as c:
            resp = c.get(protocol.HEALTH, headers=self._headers())
            resp.raise_for_status()
            return protocol.HealthResponse.model_validate(resp.json())

    def submit(
        self,
        self_wav: str | Path,
        others_wav: str | Path,
        *,
        title: str | None = None,
        language: str | None = None,
    ) -> str:
        """Upload both channels and return the created job id."""
        log.warning(protocol.upload_warning(self.url))
        data = {k: v for k, v in {"title": title, "language": language}.items() if v}
        self_p, others_p = Path(self_wav), Path(others_wav)
        with self._session() as c, self_p.open("rb") as sf, others_p.open("rb") as of:
            files = {
                "self_wav": (self_p.name, sf, "audio/wav"),
                "others_wav": (others_p.name, of, "audio/wav"),
            }
            resp = c.post(
                protocol.JOBS, files=files, data=data, headers=self._headers()
            )
        _raise_for_status(resp)
        return protocol.JobStatus.model_validate(resp.json()).id

    def status(self, job_id: str) -> protocol.JobStatus:
        """Fetch a job's current state."""
        with self._session() as c:
            resp = c.get(protocol.job_path(job_id), headers=self._headers())
        _raise_for_status(resp)
        return protocol.JobStatus.model_validate(resp.json())

    def result(self, job_id: str) -> Meeting:
        """Download a finished job's meeting record."""
        with self._session() as c:
            resp = c.get(protocol.result_path(job_id), headers=self._headers())
        _raise_for_status(resp)
        return Meeting.model_validate(resp.json())

    def wait(
        self,
        job_id: str,
        *,
        on_status: StatusCallback | None = None,
        max_wait: float | None = None,
    ) -> Meeting:
        """Poll ``job_id`` until it finishes, then return the meeting."""
        waited = 0.0
        last: Optional[str] = None
        while True:
            status = self.status(job_id)
            if on_status is not None and status.state != last:
                on_status(status)
            last = status.state
            if status.state == "done":
                return self.result(job_id)
            if status.state == "error":
                raise RemoteError(status.error or "remote job failed")
            if max_wait is not None and waited >= max_wait:
                raise RemoteError(f"job {job_id} not finished after {max_wait:.0f}s")
            time.sleep(self.poll_interval)
            waited += self.poll_interval

    def process(
        self,
        self_wav: str | Path,
        others_wav: str | Path,
        *,
        title: str | None = None,
        language: str | None = None,
        on_status: StatusCallback | None = None,
        max_wait: float | None = None,
    ) -> Meeting:
        """Submit both channels and block until the meeting is processed."""
        job_id = self.submit(
            self_wav, others_wav, title=title, language=language
        )
        log.info("Submitted remote job %s; waiting for result", job_id)
        return self.wait(job_id, on_status=on_status, max_wait=max_wait)

    def transcribe(
        self, wav: str | Path, language: str | None = None
    ) -> list[Segment]:
        """ASR-only: upload one WAV and return its segments."""
        log.warning(protocol.upload_warning(self.url))
        data = {"language": language} if language else {}
        wav_p = Path(wav)
        with self._session() as c, wav_p.open("rb") as fh:
            resp = c.post(
                protocol.TRANSCRIBE,
                files={"wav": (wav_p.name, fh, "audio/wav")},
                data=data,
                headers=self._headers(),
            )
        _raise_for_status(resp)
        return protocol.TranscriptionResult.model_validate(resp.json()).segments


def _raise_for_status(resp: Any) -> None:
    """Turn an HTTP error response into a :class:`RemoteError` with detail."""
    if resp.status_code < 400:
        return
    detail = ""
    try:
        detail = resp.json().get("detail", "")
    except Exception:  # pragma: no cover - non-JSON error body
        detail = resp.text
    raise RemoteError(f"worker returned {resp.status_code}: {detail}")
