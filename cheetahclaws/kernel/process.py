"""process.py — AgentProcess dataclass + state machine.

Pure data + transition rules. No I/O, no SQLite, no event emission.
``KernelStore`` (in store.py) is the only module that mutates persistent
state; this module exists so the state machine is testable in isolation
and so other layers (RFC 0007 scheduler, RFC F-4 supervisor) can import
the transition table without depending on the storage layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── State strings ──────────────────────────────────────────────────────────
#
# We use plain strings rather than ``enum.Enum`` so SQLite rows round-trip
# without conversion and so JSON-RPC payloads carry the same value the
# DB stores. The ``AgentState`` namespace below gives clients a typo-proof
# constant set without forcing them to import an enum.

class AgentState:
    READY     = "READY"
    RUNNING   = "RUNNING"
    WAITING   = "WAITING"
    SUSPENDED = "SUSPENDED"
    DEAD      = "DEAD"

    ALL = ("READY", "RUNNING", "WAITING", "SUSPENDED", "DEAD")
    LIVE = ("READY", "RUNNING", "WAITING", "SUSPENDED")  # everything except DEAD
    AT_RESTART_STALE = ("RUNNING", "WAITING")            # recovery candidates


# ── Transition table ───────────────────────────────────────────────────────
#
# The single source of truth for the state machine. ``KernelStore`` reads
# this table to decide whether a requested transition is legal; tests
# read it to enumerate all (legal, illegal) pairs without hand-coding
# them; downstream RFCs reference it by name.

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    AgentState.READY:     frozenset({AgentState.RUNNING, AgentState.DEAD}),
    AgentState.RUNNING:   frozenset({AgentState.WAITING, AgentState.SUSPENDED, AgentState.DEAD}),
    AgentState.WAITING:   frozenset({AgentState.RUNNING, AgentState.SUSPENDED, AgentState.DEAD}),
    AgentState.SUSPENDED: frozenset({AgentState.READY, AgentState.DEAD}),
    AgentState.DEAD:      frozenset(),  # terminal
}


def is_transition_allowed(prev: str, target: str) -> bool:
    """Return True iff ``prev`` may transition to ``target`` per the
    state-machine table. Unknown states return False (defensively)."""
    return target in ALLOWED_TRANSITIONS.get(prev, frozenset())


# ── AgentProcess dataclass ────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentProcess:
    pid:           int
    parent_pid:    Optional[int]
    name:          str
    template:      str
    state:         str
    state_reason:  Optional[str]
    created_at:    float
    updated_at:    float
    started_at:    Optional[float]
    ended_at:      Optional[float]
    exit_kind:     Optional[str]
    exit_detail:   Optional[dict]
    metadata:      dict = field(default_factory=dict)
    last_event_id: int = 0

    # ── helpers ────────────────────────────────────────────────────────

    def can_transition_to(self, target: str) -> bool:
        return is_transition_allowed(self.state, target)

    def to_dict(self) -> dict:
        """JSON-serialisable representation for RPC responses."""
        return {
            "pid":           self.pid,
            "parent_pid":    self.parent_pid,
            "name":          self.name,
            "template":      self.template,
            "state":         self.state,
            "state_reason":  self.state_reason,
            "created_at":    self.created_at,
            "updated_at":    self.updated_at,
            "started_at":    self.started_at,
            "ended_at":      self.ended_at,
            "exit_kind":     self.exit_kind,
            "exit_detail":   self.exit_detail,
            "metadata":      self.metadata,
            "last_event_id": self.last_event_id,
        }
