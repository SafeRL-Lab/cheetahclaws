"""Tests that DAG level execution enforces concurrent_safe flags for parallel vs sequential tools."""
from __future__ import annotations

import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from dag import _execute_level, _build_dag_levels


# ---------------------------------------------------------------------------
# Helpers — track execution timing per tool call to prove parallelism/sequencing
# ---------------------------------------------------------------------------

_execution_log: list[dict] = []
_log_lock = threading.Lock()


def _make_fake_tool(name: str, duration: float = 0.05):
    """Return a function that logs start/end times and sleeps for `duration`."""
    def _fake(tool_name, params, permission_mode=None, _extra=None, config=None):
        start = time.monotonic()
        tid = threading.current_thread().ident
        time.sleep(duration)
        end = time.monotonic()
        with _log_lock:
            _execution_log.append({
                "id": params.get("_tc_id", tool_name),
                "name": tool_name,
                "thread": tid,
                "start": start,
                "end": end,
            })
        return f"ok:{tool_name}"
    return _fake


def _clear_log():
    with _log_lock:
        _execution_log.clear()


def _overlaps(a: dict, b: dict) -> bool:
    """True if two execution intervals overlap in time."""
    return a["start"] < b["end"] and b["start"] < a["end"]


# ---------------------------------------------------------------------------
# Fixtures — monkeypatch execute_tool and is_concurrent_safe
# ---------------------------------------------------------------------------

_CONCURRENT_SAFE_TOOLS = {"Read", "Grep", "Glob", "GetDiagnostics", "ContextGC"}


@pytest.fixture(autouse=True)
def _patch_tools(monkeypatch):
    """Replace execute_tool with a fake that logs timing, and patch is_concurrent_safe."""
    fake = _make_fake_tool("generic", duration=0.05)
    monkeypatch.setattr("dag.execute_tool", fake)

    def _fake_is_concurrent_safe(name: str) -> bool:
        return name in _CONCURRENT_SAFE_TOOLS

    monkeypatch.setattr("tool_registry.is_concurrent_safe", _fake_is_concurrent_safe)
    _clear_log()
    yield
    _clear_log()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_tool_runs_directly():
    """Single tool in a level runs without ThreadPoolExecutor."""
    level = [{"id": "t1", "name": "Read", "input": {"_tc_id": "t1"}}]
    results, durations = {}, {}
    _execute_level(level, results, durations, config={})

    assert results["t1"] == "ok:Read"
    assert durations["t1"] > 0
    assert len(_execution_log) == 1


def test_all_concurrent_safe_run_in_parallel():
    """Multiple concurrent_safe tools run in parallel (overlapping times)."""
    level = [
        {"id": f"t{i}", "name": "Read", "input": {"_tc_id": f"t{i}"}}
        for i in range(4)
    ]
    results, durations = {}, {}
    _execute_level(level, results, durations, config={})

    assert len(results) == 4
    assert all(v.startswith("ok:") for v in results.values())
    # With 4 parallel tools each sleeping 50ms, total should be ~50-100ms not ~200ms
    logs = sorted(_execution_log, key=lambda x: x["start"])
    assert any(_overlaps(logs[i], logs[j])
               for i in range(len(logs)) for j in range(i + 1, len(logs))), \
        "Expected overlapping execution for concurrent_safe tools"


def test_all_sequential_run_one_at_a_time():
    """Multiple non-concurrent_safe tools run sequentially (no overlap)."""
    level = [
        {"id": f"t{i}", "name": "Bash", "input": {"_tc_id": f"t{i}"}}
        for i in range(3)
    ]
    results, durations = {}, {}
    _execute_level(level, results, durations, config={})

    assert len(results) == 3
    logs = sorted(_execution_log, key=lambda x: x["start"])
    for i in range(len(logs) - 1):
        assert not _overlaps(logs[i], logs[i + 1]), \
            f"Sequential tools {logs[i]['id']} and {logs[i+1]['id']} should not overlap"


def test_mixed_level_parallel_then_sequential():
    """Mixed level: concurrent_safe tools run in parallel first, then sequential tools one at a time."""
    level = [
        {"id": "r1", "name": "Read", "input": {"_tc_id": "r1"}},
        {"id": "r2", "name": "Grep", "input": {"_tc_id": "r2"}},
        {"id": "r3", "name": "Glob", "input": {"_tc_id": "r3"}},
        {"id": "s1", "name": "Bash", "input": {"_tc_id": "s1"}},
        {"id": "s2", "name": "Skill", "input": {"_tc_id": "s2"}},
    ]
    results, durations = {}, {}
    _execute_level(level, results, durations, config={})

    assert len(results) == 5

    logs_by_id = {e["id"]: e for e in _execution_log}
    parallel_logs = [logs_by_id[k] for k in ("r1", "r2", "r3")]
    seq_logs = sorted([logs_by_id[k] for k in ("s1", "s2")], key=lambda x: x["start"])

    # Parallel tools should all finish before sequential tools start
    parallel_end = max(e["end"] for e in parallel_logs)
    seq_start = min(e["start"] for e in seq_logs)
    assert parallel_end <= seq_start + 0.01, \
        "Sequential tools should start after parallel tools finish"

    # Sequential tools should not overlap each other
    assert not _overlaps(seq_logs[0], seq_logs[1]), \
        "Sequential tools should not overlap"


def test_only_sequential_tools_no_parallel_batch():
    """When all tools are non-concurrent_safe, no parallel batch runs."""
    level = [
        {"id": "w1", "name": "Write", "input": {"_tc_id": "w1"}},
        {"id": "w2", "name": "Edit", "input": {"_tc_id": "w2"}},
    ]
    results, durations = {}, {}
    _execute_level(level, results, durations, config={})

    assert len(results) == 2
    logs = sorted(_execution_log, key=lambda x: x["start"])
    assert not _overlaps(logs[0], logs[1])


def test_build_dag_levels_all_independent():
    """All independent tools land in a single level."""
    tcs = [
        {"id": "t1", "name": "Read", "input": {}},
        {"id": "t2", "name": "Grep", "input": {}},
        {"id": "t3", "name": "Skill", "input": {}},
    ]
    levels, deps = _build_dag_levels(tcs)
    assert len(levels) == 1
    assert len(levels[0]) == 3


def test_build_dag_levels_with_deps():
    """Tools with depends_on are ordered into separate levels."""
    tcs = [
        {"id": "t1", "name": "Write", "input": {"tool_call_alias": "w1"}},
        {"id": "t2", "name": "Bash", "input": {"depends_on": ["w1"]}},
    ]
    levels, deps = _build_dag_levels(tcs)
    assert len(levels) == 2
    assert levels[0][0]["id"] == "t1"
    assert levels[1][0]["id"] == "t2"
