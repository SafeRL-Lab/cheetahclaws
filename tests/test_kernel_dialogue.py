"""Tests for kernel.orchestrator.dialogue (RFC 0020) +
extensions to LlmRequest.messages and RunnerExitInfo.text/metadata."""
from __future__ import annotations

import json
import os
import sys

import pytest

from cheetahclaws.kernel import (
    AgentState,
    DialogueOrchestrator,
    DialogueQuotaBreached,
    DialogueTurnFailed,
    DialogueTurnTimeout,
    Kernel,
    SandboxPolicy,
    UnknownPid,
)
from cheetahclaws.kernel.runner.llm import (
    LlmRequest,
    LlmResponse,
    MockProvider,
    ProviderInvalidRequest,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="dialogue orchestrator spawns POSIX subprocesses",
)


# ── LlmRequest.messages extension ──────────────────────────────────────


def test_llm_request_messages_only_works():
    """user can be empty if messages is provided."""
    r = LlmRequest(
        model="m",
        messages=(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ),
    )
    assert r.has_messages
    assert r.user == ""
    assert len(r.messages) == 2


def test_llm_request_user_only_still_works():
    """Single-turn callers (the MVP shape) keep working."""
    r = LlmRequest(model="m", user="hi")
    assert r.user == "hi"
    assert r.messages == ()
    assert not r.has_messages


def test_llm_request_neither_user_nor_messages_rejected():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest(model="m")


def test_llm_request_invalid_role_rejected():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest(
            model="m",
            messages=(
                {"role": "stranger", "content": "hi"},
            ),
        )


def test_llm_request_message_without_content_rejected():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest(
            model="m",
            messages=({"role": "user"},),
        )


def test_llm_request_to_from_dict_with_messages():
    r = LlmRequest(
        model="m",
        messages=({"role": "user", "content": "hi"},),
        system="be brief",
    )
    d = r.to_dict()
    assert d["messages"] == [{"role": "user", "content": "hi"}]
    r2 = LlmRequest.from_dict(d)
    assert r2.messages == r.messages


def test_llm_request_from_dict_only_messages():
    """from_dict accepts payload with messages but no 'user' field."""
    r = LlmRequest.from_dict({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.has_messages
    assert r.user == ""


# ── MockProvider with messages ─────────────────────────────────────────


def test_mock_provider_accepts_messages():
    """MockProvider returns its fixed response regardless of
    request shape."""
    fixed = LlmResponse(text="ok", tokens_input=1, tokens_output=1,
                         cost_micro=10, model="m")
    p = MockProvider(fixed)
    r = LlmRequest(
        model="m",
        messages=(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again?"},
        ),
    )
    out = p(r)
    assert out is fixed
    # Provider records the call.
    assert len(p.calls) == 1
    assert p.calls[0].has_messages


# ── RunnerExitInfo.text / metadata extensions ──────────────────────────


def _spawn_llm(kernel: Kernel, response: dict, *,
               user_msg: str = "x") -> tuple[int, "RunnerExitInfo"]:
    """Helper: spawn the LLM runner with a fixed mock response,
    return (pid, info)."""
    a = kernel.create_agent(name="x", template="t")
    sup = kernel.make_supervisor()
    sup.spawn(
        pid=a.pid,
        argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
        policy=SandboxPolicy(wall_seconds=15),
        init_payload={"model": "m", "user": user_msg},
        env={**os.environ, "CC_LLM_PROVIDER": "mock",
             "CC_LLM_MOCK_RESPONSE_JSON": json.dumps(response)},
    )
    return a.pid, sup.wait(a.pid, timeout=20)


def test_exit_info_carries_text(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        _, info = _spawn_llm(k, {
            "text": "the full response",
            "tokens_input": 5, "tokens_output": 3,
            "cost_micro": 100, "model": "m",
        })
        assert info.exit_kind == "completed"
        assert info.text == "the full response"


def test_exit_info_carries_metadata(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        _, info = _spawn_llm(k, {
            "text": "x",
            "tokens_input": 7, "tokens_output": 3,
            "cost_micro": 130, "model": "m",
            "finish_reason": "stop",
        })
        assert info.metadata["finish_reason"] == "stop"
        assert info.metadata["tokens_input"]   == 7
        assert info.metadata["tokens_output"]  == 3
        assert info.metadata["tokens_total"]   == 10
        assert info.metadata["cost_micro"]     == 130


def test_existing_runner_text_defaults_empty(tmp_path):
    """Echo runner doesn't emit text — RunnerExitInfo.text=''."""
    with Kernel.open(tmp_path / "kernel.db") as k:
        a = k.create_agent(name="x", template="t")
        sup = k.make_supervisor()
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"],
            policy=SandboxPolicy(wall_seconds=10),
        )
        info = sup.wait(a.pid, timeout=15)
        assert info.exit_kind == "completed"
        assert info.text == ""           # default
        assert info.metadata == {}       # default


# ── DialogueOrchestrator: basic turn ───────────────────────────────────


@pytest.fixture
def kernel(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        yield k


def _mock_response(text: str = "echo response", *,
                    tokens_in: int = 10, tokens_out: int = 5,
                    cost: int = 200) -> dict:
    return {
        "text":          text,
        "tokens_input":  tokens_in,
        "tokens_output": tokens_out,
        "cost_micro":    cost,
        "model":         "claude-x",
        "finish_reason": "stop",
    }


def _orchestrator(kernel: Kernel, owner_pid: int, *,
                   response: dict, **overrides) -> DialogueOrchestrator:
    env = {**os.environ, "CC_LLM_PROVIDER": "mock",
           "CC_LLM_MOCK_RESPONSE_JSON": json.dumps(response)}
    return DialogueOrchestrator(
        kernel, agent_pid=owner_pid,
        model="claude-x",
        system="You are helpful.",
        runner_env=env,
        runner_policy=SandboxPolicy(wall_seconds=15),
        wait_timeout_s=20,
        **overrides,
    )


def test_turn_basic_round_trip(kernel):
    owner = kernel.create_agent(name="chat-1", template="dialogue")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response("hello"))
    out = orch.turn("hi")
    assert out == "hello"
    history = orch.history()
    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_turn_owner_pid_unchanged(kernel):
    """The owner pid stays in its original state across turns —
    only children are spawned."""
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    orch.turn("first")
    orch.turn("second")
    # Owner is still READY (never spawned into).
    assert kernel.process.get(owner.pid).state == AgentState.READY


def test_turn_creates_child_per_turn(kernel):
    """Each turn = a fresh child agent."""
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    initial_count = kernel.observability.summary()["agents"]["total"]
    orch.turn("a")
    orch.turn("b")
    orch.turn("c")
    final_count = kernel.observability.summary()["agents"]["total"]
    assert final_count == initial_count + 3


def test_turn_children_are_dead(kernel):
    """Children transition to DEAD after each turn."""
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    orch.turn("hi")
    # Find children of owner.
    agents, _ = kernel.process.list(parent_pid=owner.pid)
    assert len(agents) >= 1
    for c in agents:
        assert c.state == AgentState.DEAD


def test_turn_history_persists_across_orchestrators(kernel):
    """A new orchestrator instance picks up the same conversation
    via AgentFS."""
    owner = kernel.create_agent(name="x", template="t")
    orch1 = _orchestrator(kernel, owner.pid, response=_mock_response("first"))
    orch1.turn("hi")
    # Drop orch1, build orch2 with same owner pid.
    orch2 = _orchestrator(kernel, owner.pid, response=_mock_response("second"))
    history = orch2.history()
    assert len(history) == 2
    assert history[0]["content"] == "hi"
    assert history[1]["content"] == "first"


def test_turn_messages_grow_each_turn(kernel):
    """The LLM runner sees a growing messages list as the
    conversation advances."""
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    for i in range(3):
        orch.turn(f"q{i}")
    assert len(orch.history()) == 6   # 3 user + 3 assistant


# ── DialogueOrchestrator: stats ────────────────────────────────────────


def test_stats_initial(kernel):
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    stats = orch.stats()
    assert stats["turns"] == 0
    assert stats["total_tokens"] == 0
    assert stats["total_cost_micro"] == 0
    assert stats["last_turn_at"] is None


def test_stats_accumulate(kernel):
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid,
                          response=_mock_response(tokens_in=12,
                                                    tokens_out=3,
                                                    cost=180))
    orch.turn("a")
    orch.turn("b")
    orch.turn("c")
    stats = orch.stats()
    assert stats["turns"] == 3
    assert stats["total_tokens"] == 3 * 15
    assert stats["total_cost_micro"] == 3 * 180
    assert stats["last_turn_at"] is not None


# ── reset ──────────────────────────────────────────────────────────────


def test_reset_clears_history(kernel):
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    orch.turn("a")
    orch.turn("b")
    assert len(orch.history()) == 4
    orch.reset()
    assert orch.history() == []
    assert orch.stats()["turns"] == 0


def test_reset_keeps_system_by_default(kernel):
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    orch.turn("hi")
    orch.reset()
    state = orch._load_state()  # private — testing internal shape
    assert state["system"] == "You are helpful."


def test_reset_drops_system_when_requested(kernel):
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    orch.turn("hi")
    orch.reset(keep_system=False)
    state = orch._load_state()
    assert state["system"] == ""


# ── Validation ─────────────────────────────────────────────────────────


def test_constructor_rejects_unknown_pid(kernel):
    with pytest.raises(UnknownPid):
        DialogueOrchestrator(kernel, agent_pid=9999,
                              runner_env={**os.environ,
                                           "CC_LLM_PROVIDER": "mock",
                                           "CC_LLM_MOCK_RESPONSE_JSON": "{}"})


def test_constructor_rejects_non_int_pid(kernel):
    with pytest.raises(ValueError):
        DialogueOrchestrator(kernel, agent_pid="not-int")  # type: ignore[arg-type]


def test_turn_rejects_empty_message(kernel):
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    with pytest.raises(ValueError):
        orch.turn("")


def test_turn_rejects_non_string(kernel):
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(kernel, owner.pid, response=_mock_response())
    with pytest.raises(ValueError):
        orch.turn(123)  # type: ignore[arg-type]


# ── Failure paths ──────────────────────────────────────────────────────


def test_turn_failed_when_runner_crashes(kernel):
    """If the runner can't get a response (CC_LLM_PROVIDER unset),
    the runner exits failed and the orchestrator raises
    DialogueTurnFailed."""
    owner = kernel.create_agent(name="x", template="t")
    # No CC_LLM_PROVIDER set in env.
    env = {**os.environ}
    env.pop("CC_LLM_PROVIDER", None)
    orch = DialogueOrchestrator(
        kernel, agent_pid=owner.pid, model="m",
        runner_env=env,
        runner_policy=SandboxPolicy(wall_seconds=15),
        wait_timeout_s=20,
    )
    with pytest.raises(DialogueTurnFailed):
        orch.turn("hi")
    # History was NOT updated.
    assert orch.history() == []


# ── Per-turn ledger budgets via child_grants ──────────────────────────


def test_child_grants_create_per_turn_ledger(kernel):
    """When child_grants is set, each turn's child gets its own
    ledger row, not the owner."""
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(
        kernel, owner.pid, response=_mock_response(tokens_in=5, tokens_out=2),
        child_grants={"tokens": 1_000_000, "cost_micro": 1_000_000},
    )
    orch.turn("hi")
    # Owner has NO ledger (we didn't create one).
    led_owner = kernel.ledger.get(owner.pid)
    assert led_owner.entries == ()
    # Child has ledger reflecting the charge.
    children, _ = kernel.process.list(parent_pid=owner.pid)
    assert children
    led_child = kernel.ledger.get(children[0].pid)
    by_dim = {e.dim: e.used for e in led_child.entries}
    assert by_dim["tokens"] == 7
    assert "cost_micro" in by_dim


# ── Custom history path ────────────────────────────────────────────────


def test_custom_history_path(kernel):
    owner = kernel.create_agent(name="x", template="t")
    orch = _orchestrator(
        kernel, owner.pid, response=_mock_response(),
        history_path="/conversations/custom/path.json",
    )
    orch.turn("hi")
    # File exists at the custom path.
    assert kernel.fs.exists("/conversations/custom/path.json")
    # Default path doesn't exist.
    default_path = f"/conversations/{owner.pid}/history.json"
    assert not kernel.fs.exists(default_path)
