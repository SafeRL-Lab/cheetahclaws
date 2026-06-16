"""Tests for kernel.store.recover() — startup recovery semantics.

RFC 0003 §2 "Recovery semantics":
  Stale RUNNING/WAITING rows on daemon startup are coerced to either
  SUSPENDED (default) or DEAD (--kernel-recovery=mark-dead) and an
  EV_PROCESS_RECOVERED event is emitted for each.
"""
from __future__ import annotations

import pytest

from cheetahclaws.kernel import (
    AgentState,
    KernelStore,
)
from cheetahclaws.kernel.errors import InvalidPayload
from cheetahclaws.kernel.store import (
    EV_PROCESS_RECOVERED,
    RECOVERY_MARK_DEAD,
    RECOVERY_SUSPEND,
)


def _seed_stale(db_path):
    """Open the DB, leave one RUNNING and one WAITING agent behind, close."""
    s = KernelStore.open(db_path)
    try:
        a1 = s.create(name="r1", template="t")
        s.transition(a1.pid, AgentState.RUNNING)        # left RUNNING

        a2 = s.create(name="r2", template="t")
        s.transition(a2.pid, AgentState.RUNNING)
        s.transition(a2.pid, AgentState.WAITING)        # left WAITING

        a3 = s.create(name="r3", template="t")          # READY (untouched)

        a4 = s.create(name="r4", template="t")
        s.transition(a4.pid, AgentState.RUNNING)
        s.terminate(a4.pid, exit_kind="completed")      # DEAD (untouched)
        return a1.pid, a2.pid, a3.pid, a4.pid
    finally:
        s.close()


def test_suspend_default_marks_running_and_waiting(tmp_path):
    db = tmp_path / "kernel.db"
    p1, p2, p3, p4 = _seed_stale(db)

    s2 = KernelStore.open(db)
    try:
        n = s2.recover()  # default: suspend
        assert n == 2

        assert s2.get(p1).state == AgentState.SUSPENDED
        assert s2.get(p2).state == AgentState.SUSPENDED
        # READY and DEAD should NOT have been touched.
        assert s2.get(p3).state == AgentState.READY
        assert s2.get(p4).state == AgentState.DEAD

        # state_reason recorded
        assert s2.get(p1).state_reason == "daemon_restart"
        assert s2.get(p2).state_reason == "daemon_restart"
    finally:
        s2.close()


def test_mark_dead_policy_terminates(tmp_path):
    db = tmp_path / "kernel.db"
    p1, p2, p3, p4 = _seed_stale(db)

    s2 = KernelStore.open(db)
    try:
        n = s2.recover(policy=RECOVERY_MARK_DEAD)
        assert n == 2

        assert s2.get(p1).state == AgentState.DEAD
        assert s2.get(p1).exit_kind == "crashed"
        assert s2.get(p2).state == AgentState.DEAD
        assert s2.get(p2).exit_kind == "crashed"
        assert s2.get(p3).state == AgentState.READY
        assert s2.get(p4).state == AgentState.DEAD
    finally:
        s2.close()


def test_recover_emits_recovered_event_per_row(tmp_path):
    db = tmp_path / "kernel.db"
    p1, p2, _p3, _p4 = _seed_stale(db)

    s2 = KernelStore.open(db)
    try:
        s2.recover()
        events = s2.events_tail(kind=EV_PROCESS_RECOVERED, limit=100)
        pids = {e.pid for e in events}
        assert pids == {p1, p2}
        for e in events:
            assert e.payload["policy"] == RECOVERY_SUSPEND
            assert e.payload["reason"] == "daemon_restart"
            assert e.payload["new_state"] == AgentState.SUSPENDED
            assert e.payload["prev_state"] in (AgentState.RUNNING, AgentState.WAITING)
    finally:
        s2.close()


def test_recover_idempotent(tmp_path):
    db = tmp_path / "kernel.db"
    _seed_stale(db)
    s = KernelStore.open(db)
    try:
        n1 = s.recover()
        n2 = s.recover()
        n3 = s.recover()
        assert n1 == 2
        assert n2 == 0  # nothing left in RUNNING/WAITING
        assert n3 == 0
    finally:
        s.close()


def test_recover_unknown_policy_raises(tmp_path):
    s = KernelStore.open(tmp_path / "kernel.db")
    try:
        with pytest.raises(InvalidPayload):
            s.recover(policy="bogus")
    finally:
        s.close()


def test_recover_with_no_stale_rows_is_noop(tmp_path):
    s = KernelStore.open(tmp_path / "kernel.db")
    try:
        s.create(name="x", template="t")  # READY
        n = s.recover()
        assert n == 0
        # No EV_PROCESS_RECOVERED event emitted.
        events = s.events_tail(kind=EV_PROCESS_RECOVERED, limit=100)
        assert events == []
    finally:
        s.close()
