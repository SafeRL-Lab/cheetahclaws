"""Shared pytest fixtures for all tests."""

from __future__ import annotations

import pytest

from tool_registry import ToolDef, register_tool, _registry


# --------------- quota stub (avoids ImportError on CI for calc_cost) --------

@pytest.fixture(autouse=True)
def _no_quota(monkeypatch):
    """Disable quota.record_usage so tests never hit the real billing path."""
    import quota
    monkeypatch.setattr(quota, "record_usage", lambda *a, **kw: None)


# --------------- receiver tool fixture -------------------------------------

@pytest.fixture
def receiver_tool():
    """Register a tool that captures whatever params it receives."""
    received = {}
    had_before = "receiver" in _registry
    register_tool(ToolDef(
        name="receiver",
        schema={
            "name": "receiver",
            "description": "records params for assertions",
            "input_schema": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        },
        func=lambda params, _cfg: received.setdefault("seen", dict(params)) and "ok",
        read_only=True, concurrent_safe=True,
    ))
    yield received
    if not had_before:
        _registry.pop("receiver", None)
