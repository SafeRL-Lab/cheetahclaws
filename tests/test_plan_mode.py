"""Unit tests for the plan-mode tools.

Exercise `tools.plan_mode._enter_plan_mode` and `_exit_plan_mode` in
isolation: the permission-mode transitions, plan-file lifecycle and the
"empty plan" guard. E2E coverage (through agent.run + a mocked LLM stream
+ the Write tool) lives in test_plan_mode_e2e.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import runtime
import tools as _tools_init  # noqa: F401 — register tools including plan_mode
from tools.plan_mode import _enter_plan_mode, _exit_plan_mode


@pytest.fixture(autouse=True)
def _isolated_ctx():
    """Ensure plan-mode state is not leaked between tests (same session_id)."""
    yield
    for sid in ("default", "unit_sess"):
        ctx = runtime.get_session_ctx(sid)
        ctx.plan_file = None
        ctx.prev_permission_mode = None


def _mk_config(cwd):
    return {
        "_session_id": "unit_sess",
        "_worktree_cwd": str(cwd),
        "permission_mode": "auto",
    }


class TestEnterPlanMode:
    def test_creates_plan_file_with_header(self, tmp_path):
        config = _mk_config(tmp_path)
        result = _enter_plan_mode({"task_description": "Refactor X"}, config)

        plan_path = tmp_path / ".nano_claude" / "plans" / "unit_sess.md"
        assert plan_path.exists()
        assert plan_path.read_text(encoding="utf-8").startswith("# Plan: Refactor X")
        assert "Plan mode activated" in result

    def test_flips_permission_mode_to_plan(self, tmp_path):
        config = _mk_config(tmp_path)
        _enter_plan_mode({}, config)
        assert config["permission_mode"] == "plan"

    def test_is_idempotent_if_already_in_plan_mode(self, tmp_path):
        config = _mk_config(tmp_path)
        _enter_plan_mode({}, config)
        second = _enter_plan_mode({}, config)
        assert "Already in plan mode" in second


class TestExitPlanMode:
    def test_rejects_empty_plan(self, tmp_path):
        config = _mk_config(tmp_path)
        _enter_plan_mode({}, config)  # writes only the "# Plan" header
        result = _exit_plan_mode({}, config)
        assert "empty" in result.lower()
        # Still in plan mode, since exit was refused.
        assert config["permission_mode"] == "plan"

    def test_accepts_plan_with_real_content_and_restores_permission(self, tmp_path):
        config = _mk_config(tmp_path)
        _enter_plan_mode({}, config)
        plan_path = tmp_path / ".nano_claude" / "plans" / "unit_sess.md"
        plan_path.write_text("# Plan\n\n## Steps\n1. read\n2. write\n", encoding="utf-8")

        result = _exit_plan_mode({}, config)
        assert "Plan mode exited" in result
        assert "## Steps" in result
        assert config["permission_mode"] == "auto"

    def test_noop_when_not_in_plan_mode(self, tmp_path):
        config = _mk_config(tmp_path)  # permission_mode = "auto"
        result = _exit_plan_mode({}, config)
        assert "Not in plan mode" in result
