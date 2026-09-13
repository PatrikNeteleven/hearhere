"""Pure service core for the local web UI — no web framework.

Everything the UI does maps onto the existing helpers: :mod:`pipeline.artifacts`
for the folder layout, :func:`export.export_meeting` for re-exports, and
:func:`pipeline.orchestrator.rename_speakers` for renaming. Keeping this layer
framework-free means it can be unit-tested without installing ``[webui]``.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..export import FORMATS, normalize_format
from ..logging_setup import get_logger
from ..models import Meeting
from ..pipeline import artifacts
from ..pipeline.artifacts import MeetingPaths

log = get_logger("webui")

# Files the UI knows how to surface as downloadable exports, newest layout.
_EXPORT_FILENAMES = tuple(dict.fromkeys(fn for fn, _ in FORMATS.values())) + (
    "summary.md",
)

_MEDIA_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".srt": "text/plain; charset=utf-8",
    ".vtt": "text/vtt; charset=utf-8",
}


class WebUIError(Exception):
    """Raised for client-visible problems (bad id, unknown meeting, …)."""

    def __init__(self, message: str, *, code: int = 400) -> None:
        super().__init__(message)
        self.code = code


class WebUIService:
    """Browse, re-export, and rename speakers for meetings under ``storage_dir``."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.storage_dir = Path(config.general.storage_dir).expanduser()

    # -- listing / reading ----------------------------------------------

    def list_meetings(self) -> list[dict]:
        """Return lightweight summaries of every meeting, newest first."""
        summaries: list[dict] = []
        for paths in artifacts.list_meetings(self.storage_dir):
            try:
                meeting = artifacts.read_meeting(paths)
            except Exception as exc:  # pragma: no cover - corrupt folder
                log.warning("Skipping unreadable meeting %s (%s)", paths.root.name, exc)
                continue
            summaries.append(self._summary(meeting))
        return summaries

    def get_meeting(self, meeting_id: str) -> dict:
        """Return the full meeting plus the list of available export files."""
        paths = self._paths(meeting_id)
        meeting = self._read(paths)
        payload = meeting.model_dump(mode="json")
        payload["exports"] = self._available_exports(paths)
        return payload

    # -- mutations -------------------------------------------------------

    def export(self, meeting_id: str, formats: list[str] | None = None) -> list[str]:
        """Re-export a meeting; return the filenames written (from meeting.json)."""
        from ..export import export_meeting  # noqa: PLC0415

        paths = self._paths(meeting_id)
        meeting = self._read(paths)
        formats = self._validated_formats(formats)
        written = export_meeting(paths, meeting, formats)
        return [p.name for p in written]

    def rename_speakers(self, meeting_id: str, mapping: dict[str, str]) -> dict:
        """Rename speakers (rewrites meeting.json + re-exports); return the meeting."""
        from ..pipeline.orchestrator import rename_speakers  # noqa: PLC0415

        paths = self._paths(meeting_id)
        self._read(paths)  # validate it exists before mutating
        if not mapping:
            raise WebUIError("no renames given", code=400)
        rename_speakers(paths.root, self.config, mapping)
        return self.get_meeting(meeting_id)

    # -- export downloads ------------------------------------------------

    def read_export(self, meeting_id: str, filename: str) -> tuple[str, str]:
        """Return ``(text, media_type)`` for one export file of a meeting."""
        paths = self._paths(meeting_id)
        if filename not in _EXPORT_FILENAMES:
            raise WebUIError(f"unknown export {filename!r}", code=404)
        target = paths.export(filename)
        if not target.is_file():
            raise WebUIError(f"{filename} has not been exported yet", code=404)
        media = _MEDIA_TYPES.get(target.suffix, "text/plain; charset=utf-8")
        return target.read_text(encoding="utf-8"), media

    # -- helpers ---------------------------------------------------------

    def _summary(self, meeting: Meeting) -> dict:
        return {
            "id": meeting.id,
            "title": meeting.title,
            "created_at": meeting.created_at.isoformat(),
            "language": meeting.language,
            "duration": meeting.duration,
            "speakers": meeting.speakers,
            "segments": len(meeting.transcript.segments),
            "has_summary": meeting.summary is not None,
        }

    def _paths(self, meeting_id: str) -> MeetingPaths:
        """Resolve a meeting folder, refusing anything outside ``storage_dir``."""
        if not meeting_id or meeting_id in {".", ".."} or "/" in meeting_id or "\\" in meeting_id:
            raise WebUIError(f"invalid meeting id {meeting_id!r}", code=400)
        root = (self.storage_dir / meeting_id).resolve()
        # Guard against traversal: the folder must sit directly under storage_dir.
        if root.parent != self.storage_dir.resolve():
            raise WebUIError(f"invalid meeting id {meeting_id!r}", code=400)
        return artifacts.resolve_meeting(root)

    def _read(self, paths: MeetingPaths) -> Meeting:
        try:
            return artifacts.read_meeting(paths)
        except FileNotFoundError as exc:
            raise WebUIError(str(exc), code=404) from exc

    def _available_exports(self, paths: MeetingPaths) -> list[str]:
        return [fn for fn in _EXPORT_FILENAMES if paths.export(fn).is_file()]

    def _validated_formats(self, formats: list[str] | None) -> list[str]:
        formats = formats if formats else list(self.config.export.formats)
        for fmt in formats:
            if normalize_format(fmt) not in FORMATS:
                raise WebUIError(f"unknown export format {fmt!r}", code=400)
        return formats
