"""Tests for id_uniquify — prevent GC auto-stubbing of fresh tool_results."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from id_uniquify import uniquify_tool_call_ids, _collect_used_ids, _pick_fresh_id


# ── Helpers ──────────────────────────────────────────────────────────────

class _FakeGCState:
    def __init__(self, trashed_ids=None, snippets=None):
        self.trashed_ids = trashed_ids or set()
        self.snippets = snippets or {}


class _FakeState:
    def __init__(self, messages=None, gc_state=None, turn_count=1):
        self.messages = messages or []
        self.gc_state = gc_state or _FakeGCState()
        self.turn_count = turn_count


# ── collect_used_ids ─────────────────────────────────────────────────────

class TestCollectUsedIds:
    def test_empty_state(self):
        state = _FakeState()
        assert _collect_used_ids(state) == set()

    def test_collects_from_messages(self):
        state = _FakeState(messages=[
            {"role": "assistant", "tool_calls": [{"id": "r1"}]},
            {"role": "tool", "tool_call_id": "r1"},
        ])
        assert "r1" in _collect_used_ids(state)

    def test_collects_from_trashed_ids(self):
        gc = _FakeGCState(trashed_ids={"old1", "old2"})
        state = _FakeState(gc_state=gc)
        used = _collect_used_ids(state)
        assert "old1" in used
        assert "old2" in used

    def test_collects_from_snippets(self):
        gc = _FakeGCState(snippets={"s1": {"keep_after": "foo"}})
        state = _FakeState(gc_state=gc)
        assert "s1" in _collect_used_ids(state)

    def test_no_gc_state(self):
        """Gracefully handles state without gc_state."""
        state = _FakeState()
        state.gc_state = None
        assert _collect_used_ids(state) == set()


# ── pick_fresh_id ────────────────────────────────────────────────────────

class TestPickFreshId:
    def test_basic(self):
        assert _pick_fresh_id("r1", 2, set()) == "t2_r1"

    def test_conflict_adds_suffix(self):
        used = {"t2_r1"}
        assert _pick_fresh_id("r1", 2, used) == "t2_r1_2"

    def test_multiple_conflicts(self):
        used = {"t3_x", "t3_x_2", "t3_x_3"}
        assert _pick_fresh_id("x", 3, used) == "t3_x_4"


# ── uniquify_tool_call_ids ───────────────────────────────────────────────

class TestUniquifyToolCallIds:
    def test_no_collision_no_remap(self):
        tcs = [{"id": "r1", "name": "Read", "input": {}}]
        state = _FakeState()
        remap = uniquify_tool_call_ids(tcs, state)
        assert remap == {}
        assert tcs[0]["id"] == "r1"

    def test_collision_with_trashed_id(self):
        gc = _FakeGCState(trashed_ids={"r1"})
        state = _FakeState(gc_state=gc, turn_count=2)
        tcs = [{"id": "r1", "name": "Read", "input": {}}]
        remap = uniquify_tool_call_ids(tcs, state)
        assert remap == {"r1": "t2_r1"}
        assert tcs[0]["id"] == "t2_r1"

    def test_collision_with_existing_message(self):
        state = _FakeState(
            messages=[
                {"role": "assistant", "tool_calls": [{"id": "r1"}]},
                {"role": "tool", "tool_call_id": "r1"},
            ],
            turn_count=3,
        )
        tcs = [{"id": "r1", "name": "Read", "input": {}}]
        remap = uniquify_tool_call_ids(tcs, state)
        assert remap == {"r1": "t3_r1"}
        assert tcs[0]["id"] == "t3_r1"

    def test_depends_on_rewritten(self):
        gc = _FakeGCState(trashed_ids={"w1"})
        state = _FakeState(gc_state=gc, turn_count=2)
        tcs = [
            {"id": "w1", "name": "Write", "input": {}},
            {"id": "b1", "name": "Bash", "input": {"depends_on": ["w1"]}},
        ]
        remap = uniquify_tool_call_ids(tcs, state)
        assert tcs[0]["id"] == "t2_w1"
        assert tcs[1]["input"]["depends_on"] == ["t2_w1"]

    def test_multiple_collisions(self):
        gc = _FakeGCState(trashed_ids={"r1", "r2"})
        state = _FakeState(gc_state=gc, turn_count=4)
        tcs = [
            {"id": "r1", "name": "Read", "input": {}},
            {"id": "r2", "name": "Read", "input": {}},
            {"id": "r3", "name": "Read", "input": {}},
        ]
        remap = uniquify_tool_call_ids(tcs, state)
        assert "r1" in remap
        assert "r2" in remap
        assert "r3" not in remap
        assert tcs[2]["id"] == "r3"

    def test_empty_tool_calls(self):
        state = _FakeState()
        assert uniquify_tool_call_ids([], state) == {}

    def test_no_gc_state_graceful(self):
        state = _FakeState()
        state.gc_state = None
        tcs = [{"id": "r1", "name": "Read", "input": {}}]
        remap = uniquify_tool_call_ids(tcs, state)
        assert remap == {}
