"""Prompt templates for LLM summarization (language-aware, EN + DE).

Each task has a system prompt (role + rules) and a user prompt built around the
speaker-attributed transcript. The model is told to answer in the meeting's own
language so a German meeting yields a German summary. ``build_prompt`` returns a
``(system, user)`` pair for one task.
"""

from __future__ import annotations

# ISO code -> language name used in the "answer in X" instruction.
_LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
}


def _language_clause(language: str | None) -> str:
    if not language or language == "auto":
        return "Write your answer in the same language as the transcript."
    name = _LANGUAGE_NAMES.get(language, language)
    return f"Write your answer in {name}."


_SYSTEM = {
    "summary": (
        "You are a meeting assistant. You write concise, faithful summaries of "
        "meeting transcripts. Never invent facts that are not in the transcript."
    ),
    "decisions": (
        "You are a meeting assistant. You extract the concrete decisions that "
        "were made in a meeting transcript. Never invent decisions."
    ),
    "action_items": (
        "You are a meeting assistant. You extract actionable follow-up tasks "
        "from a meeting transcript, including the owner when stated."
    ),
}

_INSTRUCTION = {
    "summary": (
        "Summarize the meeting below in a few short paragraphs covering the main "
        "topics and outcomes. {lang}\n\n"
        "Output only the summary text, with no preamble or headings."
    ),
    "decisions": (
        "List the concrete decisions made in the meeting below. {lang}\n\n"
        "Output one decision per line, each starting with '- '. If no decisions "
        "were made, output nothing."
    ),
    "action_items": (
        "List the action items / follow-up tasks from the meeting below, naming "
        "the owner when it is stated. {lang}\n\n"
        "Output one task per line, each starting with '- '. If there are none, "
        "output nothing."
    ),
}


def build_prompt(
    task: str, transcript_text: str, *, language: str | None = None
) -> tuple[str, str]:
    """Return a ``(system, user)`` prompt pair for ``task``.

    ``task`` is one of ``"summary"``, ``"decisions"``, ``"action_items"``.
    """
    if task not in _SYSTEM:
        raise ValueError(f"Unknown summarization task {task!r}.")
    system = _SYSTEM[task]
    instruction = _INSTRUCTION[task].format(lang=_language_clause(language))
    user = f"{instruction}\n\n--- TRANSCRIPT ---\n{transcript_text}\n--- END ---"
    return system, user
