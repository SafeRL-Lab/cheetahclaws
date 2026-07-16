"""Tests for kernel.process — the state machine alone.

Pure unit tests; no SQLite, no daemon. The state machine is the single
most important invariant in RFC 0003, so we exhaustively cover both
legal and illegal transitions.
"""
from __future__ import annotations

from cheetahclaws.kernel.process import (
    ALLOWED_TRANSITIONS,
    AgentProcess,
    AgentState,
    is_transition_allowed,
)


# ── Allowed transitions per RFC 0003 §1 "State machine" ────────────────────


_LEGAL = {
    (AgentState.READY,     AgentState.RUNNING),
    (AgentState.READY,     AgentState.DEAD),
    (AgentState.RUNNING,   AgentState.WAITING),
    (AgentState.RUNNING,   AgentState.SUSPENDED),
    (AgentState.RUNNING,   AgentState.DEAD),
    (AgentState.WAITING,   AgentState.RUNNING),
    (AgentState.WAITING,   AgentState.SUSPENDED),
    (AgentState.WAITING,   AgentState.DEAD),
    (AgentState.SUSPENDED, AgentState.READY),
    (AgentState.SUSPENDED, AgentState.DEAD),
}


def test_every_legal_pair_is_allowed():
    for prev, target in _LEGAL:
        assert is_transition_allowed(prev, target), \
            f"{prev} -> {target} should be allowed"


def test_dead_is_terminal():
    for target in AgentState.ALL:
        assert not is_transition_allowed(AgentState.DEAD, target), \
            f"DEAD -> {target} must NOT be allowed"


def test_no_self_loops():
    for s in AgentState.ALL:
        assert not is_transition_allowed(s, s), \
            f"{s} -> {s} self-loop must NOT be allowed"


def test_table_completeness_exhaustive():
    """Every (prev, target) pair across all 5 states must be either in
    _LEGAL or rejected; nothing slips through ambiguous."""
    for prev in AgentState.ALL:
        for target in AgentState.ALL:
            allowed = is_transition_allowed(prev, target)
            expected = (prev, target) in _LEGAL
            assert allowed == expected, \
                f"({prev} -> {target}): allowed={allowed} expected={expected}"


def test_unknown_state_returns_false():
    assert not is_transition_allowed("BOGUS", AgentState.RUNNING)
    assert not is_transition_allowed(AgentState.READY, "BOGUS")


def test_allowed_transitions_table_well_formed():
    # Every state appears as a key, and DEAD has empty set.
    assert set(ALLOWED_TRANSITIONS.keys()) == set(AgentState.ALL)
    assert ALLOWED_TRANSITIONS[AgentState.DEAD] == frozenset()
    # Targets are all valid states.
    for prev, targets in ALLOWED_TRANSITIONS.items():
        for t in targets:
            assert t in AgentState.ALL, f"{prev} targets unknown state {t}"


# ── AgentProcess.to_dict round-trip-ish ────────────────────────────────────


def test_agent_process_to_dict_keys():
    p = AgentProcess(
        pid=1, parent_pid=None, name="alice", template="research/surveyor",
        state=AgentState.READY, state_reason=None,
        created_at=1000.0, updated_at=1000.0,
        started_at=None, ended_at=None,
        exit_kind=None, exit_detail=None,
        metadata={"x": 1}, last_event_id=0,
    )
    d = p.to_dict()
    expected_keys = {
        "pid", "parent_pid", "name", "template", "state", "state_reason",
        "created_at", "updated_at", "started_at", "ended_at",
        "exit_kind", "exit_detail", "metadata", "last_event_id",
    }
    assert set(d) == expected_keys
    assert d["state"] == AgentState.READY
    assert d["metadata"] == {"x": 1}


def test_agent_can_transition_to_uses_table():
    p = AgentProcess(
        pid=1, parent_pid=None, name="x", template="t",
        state=AgentState.RUNNING, state_reason=None,
        created_at=0, updated_at=0,
        started_at=None, ended_at=None,
        exit_kind=None, exit_detail=None,
        metadata={}, last_event_id=0,
    )
    assert p.can_transition_to(AgentState.WAITING)
    assert p.can_transition_to(AgentState.SUSPENDED)
    assert p.can_transition_to(AgentState.DEAD)
    assert not p.can_transition_to(AgentState.READY)
    assert not p.can_transition_to(AgentState.RUNNING)
