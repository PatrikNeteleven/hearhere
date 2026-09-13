"""Compute-device selection.

The pipeline runs the same code regardless of where the models execute; this
module resolves the configured ``device`` (``"auto" | "cpu" | "cuda" | "mps"``)
to a concrete device string, preferring a local NVIDIA GPU when present.

Selection logic (per the README): ``auto`` -> ``cuda`` if a compatible NVIDIA
GPU is available, else ``cpu``. ``mps`` (Apple Silicon) is experimental and only
chosen when requested explicitly, never by ``auto``. An explicitly requested
accelerator that is unavailable falls back to ``cpu`` with a warning.
"""

from __future__ import annotations

from .logging_setup import get_logger

log = get_logger("compute")

VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def cuda_available() -> bool:
    """True if PyTorch reports a usable CUDA device."""
    try:
        import torch  # noqa: PLC0415 - optional heavy dep, imported lazily
    except Exception:  # pragma: no cover - torch not installed
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def mps_available() -> bool:
    """True if PyTorch reports a usable Apple-Silicon (MPS) device."""
    try:
        import torch  # noqa: PLC0415
    except Exception:  # pragma: no cover - torch not installed
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def select_device(requested: str = "auto") -> str:
    """Resolve ``requested`` to a concrete device string (never ``"auto"``).

    Raises :class:`ValueError` for an unknown device name.
    """
    req = (requested or "auto").lower()
    if req not in VALID_DEVICES:
        raise ValueError(
            f"Unknown device {requested!r}; expected one of {VALID_DEVICES}."
        )

    if req == "auto":
        if cuda_available():
            log.debug("device=auto resolved to cuda")
            return "cuda"
        log.debug("device=auto resolved to cpu (no CUDA GPU found)")
        return "cpu"

    if req == "cuda" and not cuda_available():
        log.warning("device=cuda requested but no CUDA GPU is available; using cpu")
        return "cpu"

    if req == "mps" and not mps_available():
        log.warning("device=mps requested but MPS is unavailable; using cpu")
        return "cpu"

    return req
