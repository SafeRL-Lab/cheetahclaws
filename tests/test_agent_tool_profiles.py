"""Agent-level contract: the advertised and executable tool sets match."""
from __future__ import annotations

from cheetahclaws import agent
from cheetahclaws.agent import AgentState, run
from cheetahclaws.providers import AssistantTurn


def _turn(text="", tool_calls=None):
    value = AssistantTurn.__new__(AssistantTurn)
    value.text = text
    value.tool_calls = tool_calls or []
    value.in_tokens = 1
    value.out_tokens = 1
    value.cache_read_tokens = 0
    value.cache_write_tokens = 0
    return value


def _config(**extra):
    return {
        "model": "test",
        "permission_mode": "accept-all",
        "_session_id": "tool-profile-test",
        **extra,
    }


def test_standard_profile_sends_only_the_compact_surface(monkeypatch):
    seen_schemas = []

    def fake_stream(**kwargs):
        seen_schemas.append(kwargs["tool_schemas"])
        yield _turn("done")

    monkeypatch.setattr(agent, "stream", fake_stream)

    list(run("hello", AgentState(), _config(tool_profile="standard"), "system"))

    names = {schema["name"] for schema in seen_schemas[0]}
    assert "Read" in names
    assert "MemorySearch" in names
    assert "WebFetch" not in names
    assert "Agent" not in names
    assert "ReadPDF" not in names


def test_research_profile_exposes_web_and_document_tools(monkeypatch):
    seen_schemas = []

    def fake_stream(**kwargs):
        seen_schemas.append(kwargs["tool_schemas"])
        yield _turn("done")

    monkeypatch.setattr(agent, "stream", fake_stream)

    list(run("hello", AgentState(), _config(tool_profile="research"), "system"))

    names = {schema["name"] for schema in seen_schemas[0]}
    assert {"Read", "WebFetch", "WebSearch", "ReadPDF"} <= names
    assert "Agent" not in names


def test_tool_outside_profile_is_rejected_without_permission_prompt(monkeypatch):
    replies = iter([
        _turn(tool_calls=[{
            "id": "stale-web", "name": "WebFetch",
            "input": {"url": "https://example.test"},
        }]),
        _turn("done"),
    ])

    def fake_stream(**_kwargs):
        yield next(replies)

    monkeypatch.setattr(agent, "stream", fake_stream)
    state = AgentState()
    list(run("hello", state, _config(tool_profile="standard"), "system"))

    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_results) == 1
    assert "not enabled" in tool_results[0]["content"]
    assert "standard" in tool_results[0]["content"]
