"""End-to-end: LLM drives the plan-mode workflow via agent.run + mocked stream.

The plan file is written using the regular `Write` tool, whose permission
check only allows writes to the current plan_file while in plan mode -- so
this test also exercises the agent._check_permission plan-mode branch.

Only `providers.stream` is mocked. Plan tools, registry dispatch, Write tool
and the per-session RuntimeContext all run for real against tmp_path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import tools as _tools_init  # noqa: F401 - register built-ins + plan_mode
import runtime
from agent import AgentState, run
from providers import AssistantTurn


def _scripted_stream(turns):
    cursor = iter(turns)

    def fake_stream(**_kwargs):
        spec = next(cursor)
        yield AssistantTurn(
            text=spec.get("text", ""),
            tool_calls=spec.get("tool_calls") or [],
            in_tokens=1, out_tokens=1,
        )

    return fake_stream


@pytest.fixture(autouse=True)
def _reset_plan_ctx():
    yield
    for sid in ("default", "plan_e2e", "plan_rogue"):
        ctx = runtime.get_session_ctx(sid)
        ctx.plan_file = None
        ctx.prev_permission_mode = None


def test_full_plan_mode_flow_through_agent_loop(monkeypatch, tmp_path):
    """EnterPlanMode → Write(plan_file) → ExitPlanMode, all via the real agent loop."""
    plan_file = str(tmp_path / ".nano_claude" / "plans" / "plan_e2e.md")
    plan_body = "# Plan: Refactor X\n\n## Steps\n1. explore\n2. implement\n"
    turns = [
        {"tool_calls": [{
            "id": "t1", "name": "EnterPlanMode",
            "input": {"task_description": "Refactor X"},
        }]},
        {"tool_calls": [{
            "id": "t2", "name": "Write",
            "input": {"file_path": plan_file, "content": plan_body},
        }]},
        {"tool_calls": [{
            "id": "t3", "name": "ExitPlanMode", "input": {},
        }]},
        {"text": "all done"},
    ]
    monkeypatch.setattr("agent.stream", _scripted_stream(turns))

    state = AgentState()
    config = {
        "model": "test",
        "permission_mode": "auto",  # plan mode will flip it to "plan"
        "_session_id": "plan_e2e",
        "_worktree_cwd": str(tmp_path),
    }
    list(run("plan a refactor", state, config, "sys"))

    # Plan file ended up on disk with the Write-tool content.
    assert Path(plan_file).read_text(encoding="utf-8") == plan_body

    # ExitPlanMode restored the previous permission mode.
    assert config["permission_mode"] == "auto"


def test_write_outside_plan_file_is_rejected_in_plan_mode(monkeypatch, tmp_path):
    """The permission-mode 'plan' branch must deny Writes to any file != plan_file."""
    plan_file = str(tmp_path / ".nano_claude" / "plans" / "plan_rogue.md")
    unrelated = str(tmp_path / "src" / "config.py")
    turns = [
        {"tool_calls": [{
            "id": "t1", "name": "EnterPlanMode",
            "input": {"task_description": "secure"},
        }]},
        {"tool_calls": [{
            "id": "t2", "name": "Write",
            "input": {"file_path": unrelated, "content": "print('pwned')"},
        }]},
        {"text": "stopped"},
    ]
    monkeypatch.setattr("agent.stream", _scripted_stream(turns))

    state = AgentState()
    config = {
        "model": "test",
        "permission_mode": "auto",
        "_session_id": "plan_rogue",
        "_worktree_cwd": str(tmp_path),
    }
    list(run("try a rogue write", state, config, "sys"))

    # The unrelated file was NEVER created.
    assert not Path(unrelated).exists()

    # The Write tool_result for t2 carries the rejection message.
    t2_result = next(m for m in state.messages
                     if m.get("role") == "tool" and m.get("tool_call_id") == "t2")
    assert "Denied" in t2_result["content"] or "plan" in t2_result["content"].lower()
