"""Plan mode tools — EnterPlanMode / ExitPlanMode.

Extracted from tools/__init__.py so plan-mode logic lives in a single focused
module rather than scattered inline at the bottom of the tools package.

Model flow
----------
1. `EnterPlanMode` is called; a per-session plan file is (re-)created under
   `<cwd>/.nano_claude/plans/<session_id>.md` with a Markdown header, and
   `config["permission_mode"]` flips to "plan". In that mode, `Write` is only
   allowed against the plan file (see agent._check_permission).
2. The model writes the plan by calling the regular `Write` tool with
   `file_path=<plan_file>`. No dedicated WritePlan tool — `Write` already
   exists and the permission gate takes care of scoping.
3. `ExitPlanMode` reads the plan file, refuses to exit if it is empty /
   only-header, and restores the previous permission mode. The plan content
   is embedded in the tool_result so it is visible to the user on approval.
"""
from __future__ import annotations

from pathlib import Path

import runtime
from tool_registry import register_tool, ToolDef


def _plan_file_for(config: dict) -> Path:
    session_id = config.get("_session_id", "default")
    cwd = Path(config.get("_worktree_cwd") or Path.cwd())
    plans_dir = cwd / ".nano_claude" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    return plans_dir / f"{session_id}.md"


def _enter_plan_mode(params: dict, config: dict) -> str:
    """Enter plan mode: create plan file, flip permission_mode, remember previous."""
    if config.get("permission_mode") == "plan":
        return (
            "Already in plan mode. Write your plan to the plan file, "
            "then call ExitPlanMode."
        )

    plan_path = _plan_file_for(config)
    if not plan_path.exists() or plan_path.stat().st_size == 0:
        task_desc = params.get("task_description", "")
        header = f"# Plan: {task_desc}\n\n" if task_desc else "# Plan\n\n"
        plan_path.write_text(header, encoding="utf-8")

    sctx = runtime.get_ctx(config)
    sctx.prev_permission_mode = config.get("permission_mode", "auto")
    config["permission_mode"] = "plan"
    sctx.plan_file = str(plan_path)

    return (
        f"Plan mode activated. Plan file: {plan_path}\n"
        "Write your step-by-step plan to the plan file, then call ExitPlanMode "
        "when ready to implement."
    )


def _exit_plan_mode(_params: dict, config: dict) -> str:
    """Exit plan mode: read plan file, reject if empty, restore permissions."""
    if config.get("permission_mode") != "plan":
        return "Not in plan mode."

    sctx = runtime.get_ctx(config)
    plan_file = sctx.plan_file or ""
    plan_content = _read_plan_content(plan_file)

    if not _plan_has_substance(plan_content):
        return (
            "Plan is empty -- please write your step-by-step plan to the plan "
            f"file ({plan_file}) before exiting plan mode."
        )

    config["permission_mode"] = sctx.prev_permission_mode or "auto"
    sctx.prev_permission_mode = None
    sctx.plan_file = None

    return (
        "Plan mode exited. Resuming normal permissions.\n\n"
        f"Plan content:\n{plan_content}\n\n"
        "Wait for the user to approve the plan before executing any steps."
    )


def _read_plan_content(plan_file: str) -> str:
    if not plan_file:
        return ""
    path = Path(plan_file)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _plan_has_substance(content: str) -> bool:
    """Accept the plan only if it has real content beyond a single top-level title.

    A lone `# Title` line counts as empty so the model is forced to actually
    write steps; `## Section` and below count as real content.
    """
    if not content:
        return False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_top_level_title = stripped.startswith("# ") and not stripped.startswith("## ")
        if not is_top_level_title:
            return True
    return False


_ENTER_SCHEMA = {
    "name": "EnterPlanMode",
    "description": (
        "Switch to plan mode: read-only except for writing the plan file. "
        "Use this to analyze a task and write a step-by-step plan before executing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "Brief description of what you plan to do",
            },
        },
        "required": [],
    },
}

_EXIT_SCHEMA = {
    "name": "ExitPlanMode",
    "description": (
        "Exit plan mode and return to normal permissions to begin executing the plan."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


register_tool(ToolDef(
    name="EnterPlanMode", schema=_ENTER_SCHEMA, func=_enter_plan_mode,
    read_only=True, concurrent_safe=False,
))
register_tool(ToolDef(
    name="ExitPlanMode", schema=_EXIT_SCHEMA, func=_exit_plan_mode,
    read_only=False, concurrent_safe=False,
))
