"""Actionable errors for HearHere's optional-dependency extras.

The capture backends and the heavy engines (ASR, diarization, LLM) are imported
lazily, so the core installs with three dependencies and `import hearhere` works
anywhere. The cost is that a missing extra only shows up mid-run, as a bare
``ModuleNotFoundError`` from deep inside an engine. :func:`require` turns that
into the same one-line "install this extra" message the CLI already prints for
``worker`` and ``ui``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class MissingExtraError(RuntimeError):
    """An optional-dependency extra the requested feature needs is not installed.

    Carries the install command, so callers can print it as-is.
    """


@contextmanager
def require(extra: str, feature: str) -> Iterator[None]:
    """Translate a missing optional dependency into :class:`MissingExtraError`.

    Wrap only the lazy ``import`` itself::

        with require("capture", "Recording"):
            import sounddevice as sd

    so an unrelated import error inside the feature's own code isn't reported as
    a missing extra.
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        raise MissingExtraError(
            f"{feature} needs the '[{extra}]' extra: {exc}. "
            f'Install with: pip install -e ".[{extra}]" '
            "(see INSTRUCTIONS.md for the full set)."
        ) from exc
