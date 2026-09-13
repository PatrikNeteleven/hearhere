"""Tests for compute-device selection (torch mocked via module patching)."""

from __future__ import annotations

import pytest

from hearhere import compute


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(compute, "cuda_available", lambda: True)
    monkeypatch.setattr(compute, "mps_available", lambda: False)
    assert compute.select_device("auto") == "cuda"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(compute, "cuda_available", lambda: False)
    monkeypatch.setattr(compute, "mps_available", lambda: True)  # never auto-picked
    assert compute.select_device("auto") == "cpu"


def test_explicit_cpu(monkeypatch):
    monkeypatch.setattr(compute, "cuda_available", lambda: True)
    assert compute.select_device("cpu") == "cpu"


def test_explicit_cuda_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(compute, "cuda_available", lambda: False)
    assert compute.select_device("cuda") == "cpu"


def test_explicit_cuda_available(monkeypatch):
    monkeypatch.setattr(compute, "cuda_available", lambda: True)
    assert compute.select_device("cuda") == "cuda"


def test_explicit_mps_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(compute, "mps_available", lambda: False)
    assert compute.select_device("mps") == "cpu"


def test_explicit_mps_available(monkeypatch):
    monkeypatch.setattr(compute, "mps_available", lambda: True)
    assert compute.select_device("mps") == "mps"


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        compute.select_device("quantum")
