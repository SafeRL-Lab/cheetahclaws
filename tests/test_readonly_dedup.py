"""Tests for read-only tool-call dedup in agent.run().

Weak models (qwen2.5 + vLLM is the canonical case) sometimes fire the
same Read on the same file in two consecutive turns of the same run().
The user reported seeing `⚙ Read(<path>)` printed twice and the same
4 KB master plan echoed back as text twice.

The dedup:
  - Tracks (name, args) signatures of Read/Glob/Grep/WebFetch/WebSearch
    calls within a single run().
  - On the 2nd identical signature: short-circuit execute_tool, replace
    the result with a `[deduped] You already called X ...` reminder,
    suppress ToolStart/ToolEnd UI yields, still append a tool_result
    to history so OpenAI/Anthropic tool_calls ↔ tool_response pairing
    stays valid.
  - The reminder text is the tool_result the model sees, so it nudges
    the model to stop re-calling on subsequent turns too.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws import agent
from cheetahclaws.agent import AgentState, run, ToolStart, ToolEnd, TextChunk
from cheetahclaws.providers import AssistantTurn


def _fake_turn(text="", tool_calls=None):
    t = AssistantTurn.__new__(AssistantTurn)
    t.text = text
    t.tool_calls = tool_calls or []
    t.in_tokens = 1
    t.out_tokens = 1
    t.cache_read_tokens = 0
    t.cache_write_tokens = 0
    return t


def _baseline_config():
    return {
        "model":          "claude-opus-4-7",
        "permission_mode": "auto",
        "no_tools":       False,
        "_session_id":    "test-dedup",
    }


def _install_fake_stream(monkeypatch, replies):
    it = iter(replies)

    def fake_stream(**_):
        try:
            turn = next(it)
        except StopIteration:
            pytest.fail("fake_stream called more times than scripted")
        if turn.text:
            yield TextChunk(turn.text)
        yield turn

    monkeypatch.setattr(agent, "stream", fake_stream)


def _install_fake_execute(monkeypatch, registry: dict):
    """Stub execute_tool so we don't need real files. registry: name → result."""
    def fake_execute(name, inputs, permission_mode="auto", config=None):
        return registry.get(name, "ok")
    monkeypatch.setattr(agent, "execute_tool", fake_execute)


def test_second_identical_read_is_deduped(monkeypatch):
    """Same Read called twice in same run() → 2nd is short-circuited:
    no ToolStart/ToolEnd events, but tool_result still in state."""
    _install_fake_execute(monkeypatch, {"Read": "file contents A"})

    inputs = {"file_path": "/abs/path/foo.md"}
    replies = [
        # Turn 1: assistant calls Read
        _fake_turn(text="reading...", tool_calls=[
            {"id": "t1", "name": "Read", "input": inputs},
        ]),
        # Turn 2: assistant calls SAME Read again
        _fake_turn(text="re-reading...", tool_calls=[
            {"id": "t2", "name": "Read", "input": inputs},
        ]),
        # Turn 3: assistant produces final text only → loop ends
        _fake_turn(text="done"),
    ]
    _install_fake_stream(monkeypatch, replies)

    state = AgentState()
    # User message intentionally has NO absolute path — otherwise the
    # auto-nudge for path-bearing user messages would trigger an extra
    # stream call after the final text-only reply.
    events = list(run("look at the file", state, _baseline_config(), system_prompt="sys"))

    tool_starts = [e for e in events if isinstance(e, ToolStart)]
    tool_ends = [e for e in events if isinstance(e, ToolEnd)]
    # First Read yields ToolStart + ToolEnd. Second Read is deduped, so
    # NO additional ToolStart / ToolEnd events.
    assert len(tool_starts) == 1, (
        f"expected 1 ToolStart (dedup hides 2nd), got {len(tool_starts)}"
    )
    assert len(tool_ends) == 1

    # State must still have BOTH tool_result messages so the API contract
    # (one tool_response per tool_call) is preserved.
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2, (
        f"expected 2 tool_result history entries (API pairing), got {len(tool_msgs)}"
    )
    # The 2nd tool_result is the synthetic dedup reminder.
    assert "[deduped]" in tool_msgs[1]["content"]
    assert "Read" in tool_msgs[1]["content"]


def test_dedup_emits_brief_text_marker(monkeypatch):
    """The user still sees SOMETHING happened — a one-line `[deduped ...]`
    text chunk — but not the full ⚙ Read(<long path>) line."""
    _install_fake_execute(monkeypatch, {"Read": "X"})
    inputs = {"file_path": "/abs/foo"}
    replies = [
        _fake_turn(tool_calls=[{"id": "t1", "name": "Read", "input": inputs}]),
        _fake_turn(tool_calls=[{"id": "t2", "name": "Read", "input": inputs}]),
        _fake_turn(text="done"),
    ]
    _install_fake_stream(monkeypatch, replies)

    state = AgentState()
    events = list(run("look at the file", state, _baseline_config(), system_prompt="sys"))

    text_chunks = "".join(e.text for e in events if isinstance(e, TextChunk))
    assert "[deduped Read" in text_chunks


def test_dedup_only_for_readonly_tools(monkeypatch):
    """Write fired twice with same args must NOT be deduped — writes can
    be intentional (e.g. truncating then rewriting). Only Read/Glob/Grep/
    WebFetch/WebSearch dedup."""
    _install_fake_execute(monkeypatch, {"Write": "wrote"})
    inputs = {"file_path": "/abs/foo", "content": "x"}
    replies = [
        _fake_turn(tool_calls=[{"id": "w1", "name": "Write", "input": inputs}]),
        _fake_turn(tool_calls=[{"id": "w2", "name": "Write", "input": inputs}]),
        _fake_turn(text="done"),
    ]
    _install_fake_stream(monkeypatch, replies)

    state = AgentState()
    events = list(run("write the file twice", state, _baseline_config(), system_prompt="sys"))
    tool_starts = [e for e in events if isinstance(e, ToolStart)]
    # Both Writes ran — no dedup. (Permission gating in "auto" mode would
    # normally prompt for Write, but the fake execute_tool stub bypasses
    # that path; PermissionRequest events are simply unhandled in the
    # event stream and that's fine for this assertion.)
    assert len(tool_starts) == 2


def test_different_args_not_deduped(monkeypatch):
    """Read on file A then Read on file B → both run, no dedup."""
    _install_fake_execute(monkeypatch, {"Read": "X"})
    replies = [
        _fake_turn(tool_calls=[{"id": "r1", "name": "Read", "input": {"file_path": "/a"}}]),
        _fake_turn(tool_calls=[{"id": "r2", "name": "Read", "input": {"file_path": "/b"}}]),
        _fake_turn(text="done"),
    ]
    _install_fake_stream(monkeypatch, replies)

    state = AgentState()
    events = list(run("look at two files", state, _baseline_config(), system_prompt="sys"))
    tool_starts = [e for e in events if isinstance(e, ToolStart)]
    assert len(tool_starts) == 2
