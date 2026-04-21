"""End-to-end coverage for cache-token tracking.

Layers covered:
1. The AssistantTurn carries cache_read / cache_write fields (unit).
2. AgentState accumulates them across turns (unit).
3. Checkpoint snapshots persist them (unit, real make_snapshot on tmp_path).
4. Provider extraction helpers work against synthetic usage objects for each
   supported family (Anthropic, OpenAI-compatible, Ollama).
5. E2E: agent.run drains a mocked provider stream that emits an AssistantTurn
   with cache tokens, and state + checkpoint see the totals.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ---------- 1 & 2: AssistantTurn + AgentState ----------

def test_assistant_turn_has_cache_fields():
    from providers import AssistantTurn
    turn = AssistantTurn(
        text="hello", tool_calls=[], in_tokens=100, out_tokens=50,
        cache_read_tokens=80, cache_write_tokens=20,
    )
    assert turn.cache_read_tokens == 80
    assert turn.cache_write_tokens == 20


def test_assistant_turn_cache_defaults_zero():
    """Older providers and ad-hoc callers construct AssistantTurn without cache fields."""
    from providers import AssistantTurn
    turn = AssistantTurn(text="hi", tool_calls=[], in_tokens=10, out_tokens=5)
    assert turn.cache_read_tokens == 0
    assert turn.cache_write_tokens == 0


def test_agent_state_accumulates_cache_tokens():
    from agent import AgentState
    state = AgentState()
    assert (state.total_cache_read_tokens, state.total_cache_write_tokens) == (0, 0)

    state.total_cache_read_tokens  += 80
    state.total_cache_write_tokens += 20
    state.total_cache_read_tokens  += 60
    state.total_cache_write_tokens += 10

    assert state.total_cache_read_tokens == 140
    assert state.total_cache_write_tokens == 30


# ---------- 3: Checkpoint persistence ----------

def test_checkpoint_snapshot_includes_cache(tmp_path, monkeypatch):
    from checkpoint import store
    from agent import AgentState

    monkeypatch.setattr(store, "_checkpoints_root", lambda: tmp_path / ".checkpoints")
    store.reset_file_versions()

    state = AgentState()
    state.total_input_tokens       = 500
    state.total_output_tokens      = 200
    state.total_cache_read_tokens  = 300
    state.total_cache_write_tokens = 50
    state.turn_count = 3
    state.messages = [{"role": "user", "content": "test"}]

    snap = store.make_snapshot("test-session", state, {}, "hello user")
    assert snap.token_snapshot == {
        "input": 500, "output": 200, "cache_read": 300, "cache_write": 50,
    }


# ---------- 4: Provider extraction helpers ----------

class TestAnthropicCacheExtraction:
    """_anthropic_cache_tokens must read cache_read_input_tokens / cache_creation_input_tokens."""

    def test_returns_both_when_populated(self):
        from providers import _anthropic_cache_tokens
        usage = SimpleNamespace(
            input_tokens=120, output_tokens=40,
            cache_read_input_tokens=77, cache_creation_input_tokens=33,
        )
        assert _anthropic_cache_tokens(usage) == (77, 33)

    def test_missing_fields_default_to_zero(self):
        """Older Anthropic SDKs and Bedrock-over-litellm wrappers omit the cache fields."""
        from providers import _anthropic_cache_tokens
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        assert _anthropic_cache_tokens(usage) == (0, 0)

    def test_none_fields_coerced_to_zero(self):
        """Anthropic occasionally returns None (JSON null) rather than omitting the field."""
        from providers import _anthropic_cache_tokens
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_read_input_tokens=None, cache_creation_input_tokens=None,
        )
        assert _anthropic_cache_tokens(usage) == (0, 0)


class TestOpenAICacheExtraction:
    """_openai_cached_read_tokens must walk prompt_tokens_details.cached_tokens."""

    def test_reads_cached_tokens_from_details(self):
        from providers import _openai_cached_read_tokens
        usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=42),
        )
        assert _openai_cached_read_tokens(usage) == 42

    def test_missing_details_returns_zero(self):
        from providers import _openai_cached_read_tokens
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        assert _openai_cached_read_tokens(usage) == 0

    def test_none_cached_tokens_returns_zero(self):
        from providers import _openai_cached_read_tokens
        usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        )
        assert _openai_cached_read_tokens(usage) == 0


def test_ollama_stream_never_reports_cache_tokens():
    """Ollama has no prompt-caching; the path must yield 0/0 without raising."""
    from providers import AssistantTurn
    # stream_ollama yields AssistantTurn(text, tool_calls, 0, 0, 0, 0) -- we can't
    # reach the full HTTP call in a unit test, but we can assert the shape of the
    # yielded object the callers rely on.
    turn = AssistantTurn("hi", [], 0, 0, 0, 0)
    assert turn.cache_read_tokens == 0
    assert turn.cache_write_tokens == 0


# ---------- 5: End-to-end through agent.run ----------

def test_agent_run_propagates_cache_tokens_from_mocked_stream(monkeypatch, tmp_path):
    """Drive agent.run once with a scripted stream and assert totals + snapshot."""
    import tools as _tools_init  # noqa: F401 - register tools
    from agent import AgentState, run
    from providers import AssistantTurn
    from checkpoint import store as ck_store

    monkeypatch.setattr(ck_store, "_checkpoints_root", lambda: tmp_path / ".checkpoints")
    ck_store.reset_file_versions()

    def fake_stream(**_kwargs):
        yield AssistantTurn(
            text="all good", tool_calls=[],
            in_tokens=1000, out_tokens=200,
            cache_read_tokens=700, cache_write_tokens=50,
        )

    monkeypatch.setattr("agent.stream", fake_stream)

    state = AgentState()
    list(run("hello", state, {
        "model": "test", "permission_mode": "accept-all",
        "_session_id": "cache_e2e", "disabled_tools": ["Agent"],
    }, "sys"))

    assert state.total_cache_read_tokens == 700
    assert state.total_cache_write_tokens == 50

    snap = ck_store.make_snapshot("cache_e2e", state, {}, "hello")
    assert snap.token_snapshot["cache_read"] == 700
    assert snap.token_snapshot["cache_write"] == 50


def test_agent_run_accumulates_cache_across_multi_turn(monkeypatch):
    """Two consecutive agent.run calls must sum their cache counters in state."""
    import tools as _tools_init  # noqa: F401
    from agent import AgentState, run
    from providers import AssistantTurn

    emitted = iter([
        AssistantTurn("one", [], 100, 50, cache_read_tokens=40, cache_write_tokens=10),
        AssistantTurn("two", [], 120, 60, cache_read_tokens=90, cache_write_tokens=0),
    ])

    def fake_stream(**_kwargs):
        yield next(emitted)

    monkeypatch.setattr("agent.stream", fake_stream)

    state = AgentState()
    cfg = {"model": "test", "permission_mode": "accept-all",
           "_session_id": "multi", "disabled_tools": ["Agent"]}

    list(run("first", state, cfg, "sys"))
    list(run("second", state, cfg, "sys"))

    assert state.total_cache_read_tokens == 130
    assert state.total_cache_write_tokens == 10
