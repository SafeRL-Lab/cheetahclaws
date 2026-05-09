"""Tests for the auto-nudge in :func:`agent.run`.

The nudge fires when (a) the user message hands the agent a concrete
absolute path and (b) the model's first response is text-only with zero
tool calls.  It is bounded to one shot per ``run()`` invocation so it
can never cause a loop — the second text-only reply always falls through
to the normal break.

These tests cover both the standalone heuristic and the loop integration
(via a fake provider stream)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import agent
from agent import _looks_like_investigation, AgentState, run
from providers import AssistantTurn, TextChunk


# ── Heuristic ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    # Positive — absolute paths
    ("analyze /home/user/project",                                    True),
    ("look at /etc/nginx/conf",                                       True),
    ("帮我分析一下 /home/foo/bar/baz",                                 True),
    ("see (/usr/local/bin)",                                          True),
    # Negative — no path
    ("hello",                                                         False),
    ("hi there",                                                      False),
    ("what does foo() do?",                                           False),
    # Negative — URL only (must not match)
    ("check https://example.com/foo/bar",                             False),
    # Negative — relative path only
    ("see ./relative/path.py",                                        False),
    # Negative — empty
    ("",                                                              False),
])
def test_looks_like_investigation(text: str, expected: bool):
    assert _looks_like_investigation(text) is expected


# ── Loop integration via a fake provider stream ──────────────────────────


def _fake_turn(text: str = "", tool_calls: list | None = None) -> AssistantTurn:
    """Build a minimal AssistantTurn that satisfies the loop's reads."""
    t = AssistantTurn.__new__(AssistantTurn)
    t.text             = text
    t.tool_calls       = tool_calls or []
    t.in_tokens        = 1
    t.out_tokens       = 1
    t.cache_read_tokens  = 0
    t.cache_write_tokens = 0
    return t


def _drain(gen):
    """Exhaust a generator; collect events for inspection."""
    return list(gen)


def _install_fake_stream(monkeypatch, replies: list[AssistantTurn]):
    """Replace providers.stream so each call returns the next scripted reply.

    The agent.py loop imported the symbol at module-import time
    (``from providers import stream``), so we must patch it on
    ``agent``, not on ``providers``."""
    it = iter(replies)

    def fake_stream(**_kwargs):
        try:
            turn = next(it)
        except StopIteration:
            pytest.fail("fake_stream called more times than scripted")
        if turn.text:
            yield TextChunk(turn.text)
        yield turn

    monkeypatch.setattr(agent, "stream", fake_stream)


def _baseline_config() -> dict:
    """Minimal config the loop needs to run without external services."""
    return {
        "model":          "custom/qwen2.5-72b",
        "permission_mode": "auto",
        "no_tools":       False,
        "_session_id":    "test-nudge",
    }


def test_nudge_fires_when_user_gave_path_and_model_replies_text_only(monkeypatch):
    """User gives a path → model says 'please specify' → loop nudges → model
    responds again. We verify the loop made TWO API calls (so the nudge took
    effect) and that the nudge message landed in state.messages."""
    replies = [
        _fake_turn(text="Please tell me which file to look at."),
        _fake_turn(text="OK I see the files now."),
    ]
    _install_fake_stream(monkeypatch, replies)

    state = AgentState()
    user_msg = "帮我分析 /home/shangdinggu/mycode/foo/bar"
    _drain(run(user_msg, state, _baseline_config(), system_prompt="sys"))

    # Three messages: original user, assistant#1, nudge user, assistant#2
    roles = [m["role"] for m in state.messages]
    assert roles == ["user", "assistant", "user", "assistant"], roles
    assert "[system reminder]" in state.messages[2]["content"]


def test_nudge_does_not_fire_without_path(monkeypatch):
    """Bare 'hi' → model text-only reply → no nudge, single API call."""
    replies = [_fake_turn(text="Hello!")]
    _install_fake_stream(monkeypatch, replies)

    state = AgentState()
    _drain(run("hi", state, _baseline_config(), system_prompt="sys"))

    roles = [m["role"] for m in state.messages]
    assert roles == ["user", "assistant"], (
        f"Expected exactly one round-trip, got: {roles}"
    )


def test_nudge_fires_at_most_once(monkeypatch):
    """Even if the model still refuses to use tools after the nudge, the loop
    must NOT keep nudging — that would be an infinite loop. The second
    text-only reply must terminate the turn."""
    replies = [
        _fake_turn(text="Tell me the file."),     # 1st: text only → nudge
        _fake_turn(text="I still need a file."),  # 2nd: text only → break
    ]
    _install_fake_stream(monkeypatch, replies)

    state = AgentState()
    _drain(run(
        "看一下 /srv/home/shangdinggu/mycode/foo",
        state, _baseline_config(), system_prompt="sys",
    ))

    roles = [m["role"] for m in state.messages]
    # user, asst#1, nudge user, asst#2 — and we stop here, no third nudge.
    assert roles == ["user", "assistant", "user", "assistant"], roles
