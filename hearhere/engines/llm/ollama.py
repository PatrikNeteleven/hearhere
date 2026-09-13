"""LLM summarization via a local Ollama server.

Runs the configured Ollama model (e.g. ``llama3.1``) for each requested task and
assembles a :class:`Summary`. Ollama runs fully locally, so transcripts never
leave the machine. The ``ollama`` client is imported lazily; install with
``pip install "hearhere[llm]"`` and have an Ollama server running
(``ollama serve`` + ``ollama pull <model>``).
"""

from __future__ import annotations

from typing import Any

from ...llm.prompts import build_prompt
from ...logging_setup import get_logger
from ...models import Summary

log = get_logger("llm.ollama")

DEFAULT_TASKS = ("summary", "action_items", "decisions")


def _bullets(text: str) -> list[str]:
    """Parse ``- item`` / ``* item`` / ``1. item`` lines into a clean list."""
    items: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0] in "-*•":  # bullet marker, with or without a trailing space
            line = line[1:].strip()
        else:
            # Strip a leading "1." / "2)" enumerator if present.
            head, sep, rest = line.partition(" ")
            if sep and head[:-1].isdigit() and head[-1] in ".)":
                line = rest.strip()
        if line:
            items.append(line)
    return items


class OllamaSummarizer:
    """Summarizer engine backed by a local Ollama server."""

    def __init__(
        self,
        model: str = "llama3.1",
        *,
        host: str | None = None,
        tasks: list[str] | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self.tasks = list(tasks) if tasks else list(DEFAULT_TASKS)

    def _client(self) -> Any:
        import ollama  # noqa: PLC0415

        return ollama.Client(host=self.host) if self.host else ollama

    def _chat(self, system: str, user: str) -> str:
        response = self._client().chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        # The client returns a mapping-like object; support both shapes.
        message = response["message"] if isinstance(response, dict) else response.message
        content = message["content"] if isinstance(message, dict) else message.content
        return (content or "").strip()

    def summarize(
        self,
        transcript_text: str,
        *,
        language: str | None = None,
        tasks: list[str] | None = None,
    ) -> Summary:
        """Run each requested task through Ollama and assemble a Summary."""
        wanted = tasks if tasks is not None else self.tasks
        result = Summary()
        if not transcript_text.strip():
            return result

        for task in wanted:
            system, user = build_prompt(task, transcript_text, language=language)
            log.info("Summarizing (%s) with %s", task, self.model)
            answer = self._chat(system, user)
            if task == "summary":
                result.summary = answer
            elif task == "decisions":
                result.decisions = _bullets(answer)
            elif task == "action_items":
                result.action_items = _bullets(answer)
            else:  # pragma: no cover - guarded by config Literal
                log.warning("Ignoring unknown summarization task %r", task)
        return result
