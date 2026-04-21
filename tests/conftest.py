"""Shared pytest fixtures for all tests."""

from __future__ import annotations

import pytest


# --------------- quota stub (avoids ImportError on CI for calc_cost) --------

@pytest.fixture(autouse=True)
def _no_quota(monkeypatch):
    """Disable quota.record_usage so tests never hit the real billing path."""
    import quota
    monkeypatch.setattr(quota, "record_usage", lambda *a, **kw: None)
