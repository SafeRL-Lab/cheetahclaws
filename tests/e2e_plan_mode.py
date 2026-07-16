"""End-to-end test for plan mode."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEP = "=" * 60


@dataclass
class FakeState:
    messages: list = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turn_count: int = 0


def test_plan_mode():
    tmpdir = Path(tempfile.mkdtemp(prefix="plan_e2e_"))
    print(f"Workspace: {tmpdir}")

    # ── Step 1: Setup config ──
    print(f"\n{SEP}")
    print("STEP 1: Setup")
    print(SEP)

    config = {
        "permission_mode": "auto",
        "_session_id": "plantest",
    }
    state = FakeState()

    # Import _check_permission from agent
    from cheetahclaws.agent import _check_permission

    print("  PASS")

    # ── Step 2: Before plan mode, writes need permission ──
    print(f"\n{SEP}")
    print("STEP 2: Auto mode — writes need permission (return False)")
    print(SEP)

    write_tc = {"name": "Write", "input": {"file_path": str(tmpdir / "test.py"), "content": "x"}}
    edit_tc = {"name": "Edit", "input": {"file_path": str(tmpdir / "test.py"), "old_string": "a", "new_string": "b"}}
    read_tc = {"name": "Read", "input": {"file_path": str(tmpdir / "test.py")}}
    bash_tc = {"name": "Bash", "input": {"command": "ls"}}
    bash_write_tc = {"name": "Bash", "input": {"command": "rm -rf /"}}

    assert _check_permission(read_tc, config) == True, "Read should be auto-approved"
    assert _check_permission(write_tc, config) == False, "Write should need permission in auto mode"
    assert _check_permission(bash_tc, config) == True, "Safe bash should be auto-approved"
    print("  PASS")

    # ── Step 3: Enter plan mode ──
    print(f"\n{SEP}")
    print("STEP 3: Enter plan mode")
    print(SEP)

    plans_dir = tmpdir / ".nano_claude" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / "plantest.md"
    plan_path.write_text("# Plan: Add WebSocket support\n\n", encoding="utf-8")

    from cheetahclaws import runtime
    sctx = runtime.get_session_ctx("test_plan")
    config["_session_id"] = "test_plan"
    sctx.prev_permission_mode = config["permission_mode"]
    config["permission_mode"] = "plan"
    sctx.plan_file = str(plan_path)

    print(f"  Plan file: {plan_path}")
    assert config["permission_mode"] == "plan"
    print("  PASS")

    # ── Step 4: In plan mode — reads are allowed ──
    print(f"\n{SEP}")
    print("STEP 4: Plan mode — reads allowed")
    print(SEP)

    assert _check_permission(read_tc, config) == True, "Read should be allowed in plan mode"
    assert _check_permission({"name": "Glob", "input": {"pattern": "*.py"}}, config) == True
    assert _check_permission({"name": "Grep", "input": {"pattern": "def"}}, config) == True
    assert _check_permission({"name": "WebSearch", "input": {"query": "test"}}, config) == True
    assert _check_permission(bash_tc, config) == True, "Safe bash allowed"
    print("  PASS")

    # ── Step 5: In plan mode — writes to NON-plan files are BLOCKED ──
    print(f"\n{SEP}")
    print("STEP 5: Plan mode — writes to non-plan files blocked")
    print(SEP)

    assert _check_permission(write_tc, config) == False, "Write to arbitrary file should be blocked"
    assert _check_permission(edit_tc, config) == False, "Edit to arbitrary file should be blocked"
    nb_tc = {"name": "NotebookEdit", "input": {"notebook_path": str(tmpdir / "nb.ipynb"), "new_source": "x"}}
    assert _check_permission(nb_tc, config) == False, "NotebookEdit should be blocked"
    assert _check_permission(bash_write_tc, config) == False, "Dangerous bash should be blocked"
    print("  PASS")

    # ── Step 6: In plan mode — writes to plan file ARE allowed ──
    print(f"\n{SEP}")
    print("STEP 6: Plan mode — writes to plan file allowed")
    print(SEP)

    plan_write_tc = {"name": "Write", "input": {"file_path": str(plan_path), "content": "# Updated plan"}}
    plan_edit_tc = {"name": "Edit", "input": {"file_path": str(plan_path), "old_string": "# Plan", "new_string": "# Revised Plan"}}

    assert _check_permission(plan_write_tc, config) == True, "Write to plan file should be allowed"
    assert _check_permission(plan_edit_tc, config) == True, "Edit to plan file should be allowed"
    print("  PASS")

    # ── Step 7: Plan file write with normalized path ──
    print(f"\n{SEP}")
    print("STEP 7: Plan file detection with path normalization")
    print(SEP)

    # Test with slightly different path (e.g., trailing slash, double slash)
    alt_path = str(plan_path).replace("\\", "/")
    alt_write_tc = {"name": "Write", "input": {"file_path": alt_path, "content": "x"}}
    assert _check_permission(alt_write_tc, config) == True, "Normalized path should match plan file"
    print("  PASS")

    # ── Step 8: Exit plan mode ──
    print(f"\n{SEP}")
    print("STEP 8: Exit plan mode — permissions restored")
    print(SEP)

    prev = sctx.prev_permission_mode or "auto"
    sctx.prev_permission_mode = None
    config["permission_mode"] = prev

    assert config["permission_mode"] == "auto"
    # Now writes go back to needing permission (return False in auto mode)
    assert _check_permission(write_tc, config) == False, "Should be back to auto mode"
    assert _check_permission(read_tc, config) == True, "Reads still auto-approved"
    print("  PASS")

    # ── Step 9: Plan file persists on disk ──
    print(f"\n{SEP}")
    print("STEP 9: Plan file persists on disk after exit")
    print(SEP)

    assert plan_path.exists(), "Plan file should still exist"
    content = plan_path.read_text(encoding="utf-8")
    assert "# Plan: Add WebSocket support" in content
    print(f"  Plan file content: {content.strip()!r}")
    print("  PASS")

    # ── Step 10: System prompt includes plan mode instructions ──
    print(f"\n{SEP}")
    print("STEP 10: System prompt injection")
    print(SEP)

    # Re-enter plan mode for this test
    config["permission_mode"] = "plan"
    sctx.plan_file = str(plan_path)

    from cheetahclaws.context import build_system_prompt
    prompt = build_system_prompt(config)
    assert "Plan Mode (ACTIVE)" in prompt, "System prompt should include plan mode section"
    assert str(plan_path) in prompt, "System prompt should reference plan file path"
    assert "ONLY write to the plan file" in prompt

    # Without plan mode
    config["permission_mode"] = "auto"
    prompt_normal = build_system_prompt(config)
    assert "Plan Mode" not in prompt_normal, "Normal mode should NOT have plan mode instructions"

    print("  PASS")

    # ── Cleanup ──
    import shutil
    shutil.rmtree(str(tmpdir), ignore_errors=True)

    print(f"\n{SEP}")
    print("ALL 10 STEPS PASSED")
    print(SEP)


if __name__ == "__main__":
    test_plan_mode()
