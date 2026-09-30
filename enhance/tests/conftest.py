import pytest


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    """Default for every test: the bicubic stub, never a real model download.
    Tests that exercise the real load path override this (see test_model_cache.py)."""
    monkeypatch.setenv("SATENHANCE_STUB_MODEL", "1")
