"""The ``SummarizerEngine`` interface, transcript rendering, and engine factory.

A summarizer turns a speaker-attributed transcript into a :class:`Summary`
(prose summary + decisions + action items). The protocol is small so backends
(Ollama, llama.cpp) can be swapped via config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...models import Summary, Transcript

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import Config


@runtime_checkable
class SummarizerEngine(Protocol):
    """Produces a :class:`Summary` from a transcript."""

    def summarize(
        self,
        transcript_text: str,
        *,
        language: str | None = None,
        tasks: list[str] | None = None,
    ) -> Summary:
        """Summarize ``transcript_text`` for the requested ``tasks``.

        ``tasks`` is a subset of ``"summary"``/``"decisions"``/``"action_items"``
        (``None`` = the engine's configured default). ``language`` is the
        meeting language (ISO code or ``None``/``"auto"``).
        """
        ...


def transcript_to_text(transcript: Transcript) -> str:
    """Render a transcript as ``Speaker: text`` lines for LLM consumption."""
    lines: list[str] = []
    for seg in transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        speaker = seg.speaker or "Unknown"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def create_summarizer_engine(config: "Config") -> SummarizerEngine:
    """Build the summarizer named in ``config.llm.engine``."""
    name = config.llm.engine

    if name == "ollama":
        from .ollama import OllamaSummarizer  # noqa: PLC0415

        return OllamaSummarizer(model=config.llm.model, tasks=list(config.llm.tasks))
    if name == "llamacpp":
        raise NotImplementedError("The llama.cpp summarizer is not implemented yet.")
    raise ValueError(f"Unknown LLM engine {name!r}.")
