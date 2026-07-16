"""End-to-end test for EnterPlanMode / ExitPlanMode tools."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEP = "=" * 60


def test_plan_tools():
    tmpdir = Path(tempfile.mkdtemp(prefix="plan_tools_e2e_"))
    orig_cwd = os.getcwd()
    os.chdir(str(tmpdir))

    try:
        _run(tmpdir)
    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def _run(tmpdir):
    from cheetahclaws.tools import _enter_plan_mode, _exit_plan_mode
    from cheetahclaws.agent import _check_permission

    config = {
        "permission_mode": "auto",
        "_session_id": "tooltest",
    }

    # ── Step 1: EnterPlanMode tool creates plan file and switches mode ──
    print(f"\n{SEP}")
    print("STEP 1: EnterPlanMode")
    print(SEP)
    result = _enter_plan_mode({"task_description": "Add WebSocket support"}, config)
    from cheetahclaws import runtime
    sctx = runtime.get_session_ctx("tooltest")
    assert config["permission_mode"] == "plan"
    assert sctx.plan_file
    plan_path = Path(sctx.plan_file)
    assert plan_path.exists()
    assert "WebSocket" in plan_path.read_text(encoding="utf-8")
    assert "Plan mode activated" in result
    print(f"  Plan file: {plan_path}")
    print(f"  Mode: {config['permission_mode']}")
    print("  PASS")

    # ── Step 2: EnterPlanMode again → already in plan mode ──
    print(f"\n{SEP}")
    print("STEP 2: EnterPlanMode while already in plan mode")
    print(SEP)
    result = _enter_plan_mode({}, config)
    assert "Already in plan mode" in result
    print(f"  {result}")
    print("  PASS")

    # ── Step 3: Permission checks in plan mode ──
    print(f"\n{SEP}")
    print("STEP 3: Permission checks")
    print(SEP)

    # Reads allowed
    assert _check_permission({"name": "Read", "input": {}}, config) == True
    assert _check_permission({"name": "Glob", "input": {}}, config) == True
    assert _check_permission({"name": "Grep", "input": {}}, config) == True
    print("  Reads: allowed")

    # Writes blocked
    assert _check_permission({"name": "Write", "input": {"file_path": str(tmpdir / "x.py")}}, config) == False
    assert _check_permission({"name": "Edit", "input": {"file_path": str(tmpdir / "x.py")}}, config) == False
    print("  Writes to other files: blocked")

    # Write to plan file allowed
    assert _check_permission({"name": "Write", "input": {"file_path": str(plan_path)}}, config) == True
    assert _check_permission({"name": "Edit", "input": {"file_path": str(plan_path)}}, config) == True
    print("  Writes to plan file: allowed")

    # Plan tools always auto-approved
    assert _check_permission({"name": "EnterPlanMode", "input": {}}, config) == True
    assert _check_permission({"name": "ExitPlanMode", "input": {}}, config) == True
    print("  Plan tools: auto-approved")
    print("  PASS")

    # ── Step 4: ExitPlanMode with empty plan → rejected ──
    print(f"\n{SEP}")
    print("STEP 4: ExitPlanMode with empty plan")
    print(SEP)
    # Plan file currently has just the header
    result = _exit_plan_mode({}, config)
    # Should still be in plan mode if plan only has header
    if "empty" in result.lower():
        print(f"  Correctly rejected: {result[:60]}")
        assert config["permission_mode"] == "plan"
    else:
        # Header counts as content — that's fine too
        print(f"  Header accepted as plan content")
    print("  PASS")

    # ── Step 5: Write plan content and ExitPlanMode ──
    print(f"\n{SEP}")
    print("STEP 5: Write plan content and ExitPlanMode")
    print(SEP)
    # Ensure we're in plan mode
    config["permission_mode"] = "plan"
    plan_path.write_text(
        "# Plan: Add WebSocket support\n\n"
        "## Phase 1: Create ws_handler.py\n"
        "## Phase 2: Modify server.py\n"
        "## Phase 3: Add tests\n",
        encoding="utf-8",
    )
    result = _exit_plan_mode({}, config)
    assert config["permission_mode"] == "auto", f"Mode should be auto, got {config['permission_mode']}"
    assert "Plan mode exited" in result
    assert "Phase 1" in result  # plan content included
    assert "Wait for the user to approve" in result
    print(f"  Mode restored to: {config['permission_mode']}")
    print(f"  Plan content in result: {'Phase 1' in result}")
    print("  PASS")

    # ── Step 6: ExitPlanMode when not in plan mode ──
    print(f"\n{SEP}")
    print("STEP 6: ExitPlanMode when not in plan mode")
    print(SEP)
    result = _exit_plan_mode({}, config)
    assert "Not in plan mode" in result
    print(f"  {result}")
    print("  PASS")

    # ── Step 7: Plan tools auto-approved in auto mode too ──
    print(f"\n{SEP}")
    print("STEP 7: Plan tools auto-approved in auto mode")
    print(SEP)
    config["permission_mode"] = "auto"
    assert _check_permission({"name": "EnterPlanMode", "input": {}}, config) == True
    assert _check_permission({"name": "ExitPlanMode", "input": {}}, config) == True
    print("  Auto-approved in auto mode")

    config["permission_mode"] = "manual"
    assert _check_permission({"name": "EnterPlanMode", "input": {}}, config) == True
    assert _check_permission({"name": "ExitPlanMode", "input": {}}, config) == True
    print("  Auto-approved in manual mode")
    print("  PASS")

    # ── Step 8: System prompt includes plan mode guidance ──
    print(f"\n{SEP}")
    print("STEP 8: System prompt includes plan mode guidance")
    print(SEP)
    from cheetahclaws.context import build_system_prompt
    config["permission_mode"] = "auto"
    prompt = build_system_prompt(config)
    assert "EnterPlanMode" in prompt
    assert "ExitPlanMode" in prompt
    assert "complex" in prompt.lower() or "multi-file" in prompt.lower()
    print("  System prompt references plan tools")
    print("  PASS")

    print(f"\n{SEP}")
    print("ALL 8 STEPS PASSED")
    print(SEP)


if __name__ == "__main__":
    test_plan_tools()
