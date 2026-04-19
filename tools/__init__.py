"""Tool definitions and implementations for CheetahClaws.

Implementations live in focused sub-modules:
  tools.security     _check_path_allowed, _is_safe_bash
  tools.fs           Read / Write / Edit / Glob + diff helpers
  tools.shell        Bash / Grep
  tools.web          WebFetch / WebSearch
  tools.notebook     NotebookEdit
  tools.diagnostics  GetDiagnostics
  tools.interaction  AskUserQuestion / SleepTimer / bridge routing

This module re-exports every public symbol for backward compatibility,
holds the TOOL_SCHEMAS list, the execute_tool dispatcher, and calls
_register_builtins() which wires all built-ins into the tool registry.
"""
from __future__ import annotations

from typing import Callable, Optional

# ── Re-exports (backward compat) ──────────────────────────────────────────

from tools.security import _check_path_allowed, _is_safe_bash  # noqa: F401

from tools.fs import (  # noqa: F401
    _read, _write, _edit, _glob,
    generate_unified_diff, maybe_truncate_diff,
)

from tools.shell import _bash, _grep, _kill_proc_tree, _has_rg  # noqa: F401

from tools.web import _webfetch, _websearch  # noqa: F401

from tools.notebook import _notebook_edit, _parse_cell_id  # noqa: F401

from tools.diagnostics import (  # noqa: F401
    _get_diagnostics, _detect_language, _run_quietly,
)

from tools.interaction import (  # noqa: F401
    _tg_thread_local, _wx_thread_local, _slack_thread_local,
    _is_in_tg_turn, _is_in_wx_turn, _is_in_slack_turn, _is_in_web_turn,
    _ask_user_question, ask_input_interactive, drain_pending_questions,
    _sleeptimer, _pending_questions, _ask_lock, _INPUT_WAIT_TIMEOUT,
)

from tool_registry import ToolDef, register_tool
from tool_registry import execute_tool as _registry_execute


# ── Tool JSON schemas (sent to the LLM API) ───────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "Read",
        "description": (
            "Read a file's contents. Returns content with line numbers "
            "(format: 'N\\tline'). Use limit/offset to read large files in chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path"},
                "limit":     {"type": "integer", "description": "Max lines to read"},
                "offset":    {"type": "integer", "description": "Start line (0-indexed)"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content":   {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": (
            "Replace exact text in a file. old_string must match exactly (including whitespace). "
            "If old_string appears multiple times, use replace_all=true or add more context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path":   {"type": "string"},
                "old_string":  {"type": "string", "description": "Exact text to replace"},
                "new_string":  {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Bash",
        "description": "Execute a shell command. Returns stdout+stderr. Stateless (no cd persistence).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds before timeout (default 30). Use 120-300 for package installs (npm, pip, npx), builds, and long-running commands."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files matching a glob pattern. Returns sorted list of matching paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern e.g. **/*.py"},
                "path":    {"type": "string", "description": "Base directory (default: cwd)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex using ripgrep (falls back to grep).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern":          {"type": "string", "description": "Regex pattern"},
                "path":             {"type": "string", "description": "File or directory to search"},
                "glob":             {"type": "string", "description": "File filter e.g. *.py"},
                "output_mode":      {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "content=matching lines, files_with_matches=file paths, count=match counts",
                },
                "case_insensitive": {"type": "boolean"},
                "context":          {"type": "integer", "description": "Lines of context around matches"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "WebFetch",
        "description": "Fetch a URL and return its text content (HTML stripped).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url":    {"type": "string"},
                "prompt": {"type": "string", "description": "Hint for what to extract"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "WebSearch",
        "description": "Search the web via DuckDuckGo and return top results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "TaskCreate",
        "description": (
            "Create a new task in the task list. "
            "Use this to track work items, to-dos, and multi-step plans."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject":     {"type": "string", "description": "Brief title"},
                "description": {"type": "string", "description": "What needs to be done"},
                "active_form": {"type": "string", "description": "Present-continuous label while in_progress"},
                "metadata":    {"type": "object", "description": "Arbitrary metadata"},
            },
            "required": ["subject", "description"],
        },
    },
    {
        "name": "TaskUpdate",
        "description": (
            "Update a task: change status, subject, description, owner, "
            "dependency edges, or metadata. "
            "Set status='deleted' to remove. "
            "Statuses: pending, in_progress, completed, cancelled, deleted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":        {"type": "string"},
                "subject":        {"type": "string"},
                "description":    {"type": "string"},
                "status":         {"type": "string", "enum": ["pending","in_progress","completed","cancelled","deleted"]},
                "active_form":    {"type": "string"},
                "owner":          {"type": "string"},
                "add_blocks":     {"type": "array", "items": {"type": "string"}},
                "add_blocked_by": {"type": "array", "items": {"type": "string"}},
                "metadata":       {"type": "object"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "TaskGet",
        "description": "Retrieve full details of a single task by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to retrieve"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "TaskList",
        "description": "List all tasks with their status, owner, and pending blockers.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "NotebookEdit",
        "description": (
            "Edit a Jupyter notebook (.ipynb) cell. "
            "Supports replace (modify existing cell), insert (add new cell after cell_id), "
            "and delete (remove cell) operations. "
            "Read the notebook with the Read tool first to see cell IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notebook_path": {
                    "type": "string",
                    "description": "Absolute path to the .ipynb notebook file",
                },
                "new_source": {
                    "type": "string",
                    "description": "New source code/text for the cell",
                },
                "cell_id": {
                    "type": "string",
                    "description": (
                        "ID of the cell to edit. For insert, the new cell is inserted after this cell "
                        "(or at the beginning if omitted). Use 'cell-N' (0-indexed) if no IDs are set."
                    ),
                },
                "cell_type": {
                    "type": "string",
                    "enum": ["code", "markdown"],
                    "description": "Cell type. Required for insert; defaults to current type for replace.",
                },
                "edit_mode": {
                    "type": "string",
                    "enum": ["replace", "insert", "delete"],
                    "description": "replace (default) / insert / delete",
                },
            },
            "required": ["notebook_path", "new_source"],
        },
    },
    {
        "name": "GetDiagnostics",
        "description": (
            "Get LSP-style diagnostics (errors, warnings, hints) for a source file. "
            "Uses pyright/mypy/flake8 for Python, tsc for TypeScript/JavaScript, "
            "and shellcheck for shell scripts. Returns structured diagnostic output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to diagnose",
                },
                "language": {
                    "type": "string",
                    "description": (
                        "Override auto-detected language: python, javascript, typescript, "
                        "shellscript. Omit to auto-detect from file extension."
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "AskUserQuestion",
        "description": (
            "Pause execution and ask the user a clarifying question. "
            "Use this when you need a decision from the user before proceeding. "
            "Returns the user's answer as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "description": "Optional list of choices. Each item: {label, description}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label":       {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["label"],
                    },
                },
                "allow_freetext": {
                    "type": "boolean",
                    "description": "If true (default), user may type a free-text answer instead of selecting an option.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "SleepTimer",
        "description": (
            "Schedule a silent background timer. When the timer finishes, it injects an automated "
            "prompt: '(System Automated Event): The timer has finished...' so you can wake up and "
            "execute deferred monitoring tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "Number of seconds to sleep before waking up."},
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "ContextGC",
        "description": (
            "Garbage-collect your context to free space. Trash tool results you no longer "
            "need, keep only relevant snippets from large results, and save key information "
            "in notes that persist across turns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trash":         {"type": "array", "items": {"type": "string"}, "description": "tool_call_ids to fully discard"},
                "keep_snippets": {"type": "array", "description": "Partial keeps with text anchors"},
                "notes":         {"type": "array", "description": "Named scratchpad entries: {name, content}"},
                "trash_notes":   {"type": "array", "items": {"type": "string"}, "description": "Note names to discard"},
            },
            "required": [],
        },
    },
]


# ── Dispatcher (backward-compatible wrapper) ──────────────────────────────

def execute_tool(
    name: str,
    inputs: dict,
    permission_mode: str = "auto",
    ask_permission: Optional[Callable[[str], bool]] = None,
    config: dict = None,
) -> str:
    """Dispatch tool execution; ask permission for write/destructive ops."""
    cfg = config or {}

    def _check(desc: str) -> bool:
        if permission_mode == "accept-all":
            return True
        if ask_permission:
            return ask_permission(desc)
        return False  # deny by default when no permission handler is set

    if name == "Write":
        if not _check(f"Write to {inputs['file_path']}"):
            return "Denied: user rejected write operation"
    elif name == "Edit":
        if not _check(f"Edit {inputs['file_path']}"):
            return "Denied: user rejected edit operation"
    elif name == "Bash":
        cmd = inputs["command"]
        if permission_mode != "accept-all" and not _is_safe_bash(cmd):
            if not _check(f"Bash: {cmd}"):
                return "Denied: user rejected bash command"
    elif name == "NotebookEdit":
        if not _check(f"Edit notebook {inputs['notebook_path']}"):
            return "Denied: user rejected notebook edit operation"

    return _registry_execute(name, inputs, cfg)


# ── Register built-in tools with the plugin registry ─────────────────────

def _register_builtins() -> None:
    """Register all built-in tools into the central registry."""
    _schemas = {s["name"]: s for s in TOOL_SCHEMAS}

    _tool_defs = [
        ToolDef(
            name="Read",
            schema=_schemas["Read"],
            func=lambda p, c: (
                "Error: missing required parameter 'file_path'" if not p.get("file_path")
                else _check_path_allowed(p["file_path"], c) or _read(**p)
            ),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="Write",
            schema=_schemas["Write"],
            func=lambda p, c: (
                "Error: missing required parameter 'file_path'" if not p.get("file_path")
                else _check_path_allowed(p["file_path"], c) or _write(**p)
            ),
            read_only=False, concurrent_safe=False,
        ),
        ToolDef(
            name="Edit",
            schema=_schemas["Edit"],
            func=lambda p, c: (
                "Error: missing required parameter 'file_path'" if not p.get("file_path")
                else _check_path_allowed(p["file_path"], c) or _edit(**p)
            ),
            read_only=False, concurrent_safe=False,
        ),
        ToolDef(
            name="Bash",
            schema=_schemas["Bash"],
            func=lambda p, c: _bash(
                p["command"], p.get("timeout", 30),
                c.get("_worktree_cwd"),
                c.get("shell_policy", "allow"),
                c.get("_session_id", "default"),
            ),
            read_only=False, concurrent_safe=False,
        ),
        ToolDef(
            name="Glob",
            schema=_schemas["Glob"],
            func=lambda p, c: _glob(p["pattern"], p.get("path"), c.get("_worktree_cwd")),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="Grep",
            schema=_schemas["Grep"],
            func=lambda p, c: _grep(
                p["pattern"], p.get("path"), p.get("glob"),
                p.get("output_mode", "files_with_matches"),
                p.get("case_insensitive", False),
                p.get("context", 0),
                c.get("_worktree_cwd"),
            ),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="WebFetch",
            schema=_schemas["WebFetch"],
            func=lambda p, c: _webfetch(p["url"], p.get("prompt")),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="WebSearch",
            schema=_schemas["WebSearch"],
            func=lambda p, c: _websearch(p["query"]),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="NotebookEdit",
            schema=_schemas["NotebookEdit"],
            func=lambda p, c: _notebook_edit(
                p["notebook_path"], p["new_source"],
                p.get("cell_id"), p.get("cell_type"),
                p.get("edit_mode", "replace"),
            ),
            read_only=False, concurrent_safe=False,
        ),
        ToolDef(
            name="GetDiagnostics",
            schema=_schemas["GetDiagnostics"],
            func=lambda p, c: _get_diagnostics(p["file_path"], p.get("language")),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="AskUserQuestion",
            schema=_schemas["AskUserQuestion"],
            func=lambda p, c: _ask_user_question(
                p["question"], p.get("options"), p.get("allow_freetext", True),
            ),
            read_only=True, concurrent_safe=False,
        ),
        ToolDef(
            name="SleepTimer",
            schema=_schemas["SleepTimer"],
            func=lambda p, c: _sleeptimer(p["seconds"], c),
            read_only=False, concurrent_safe=True,
        ),
    ]

    # ContextGC tool
    from context_gc import process_gc_call
    _tool_defs.append(ToolDef(
        name="ContextGC",
        schema=_schemas["ContextGC"],
        func=lambda p, c: process_gc_call(p, c),
        read_only=False, concurrent_safe=False,
    ))

    for td in _tool_defs:
        register_tool(td)


_register_builtins()


# ── Extension tools (auto-discovery) ─────────────────────────────────────
# Each module self-registers its tools on import. Failures are best-effort.

_EXTENSION_MODULES = [
    "memory.tools",
    "multi_agent.tools",
    "skill.tools",
    "cc_mcp.tools",
    "task.tools",
]

for _mod_name in _EXTENSION_MODULES:
    try:
        __import__(_mod_name)
    except Exception:
        pass  # Extension loading is best-effort; never crash startup

from multi_agent.tools import get_agent_manager as _get_agent_manager  # noqa: F401

try:
    from plugin.loader import register_plugin_tools as _reg_plugin_tools
    _reg_plugin_tools()
except Exception:
    pass   # Plugin loading is best-effort; never crash startup

try:
    from checkpoint.hooks import install_hooks as _install_checkpoint_hooks
    _install_checkpoint_hooks()
except Exception:
    pass

# Sub-modules within tools/ package (self-registering on import)
import importlib as _il
for _sub in ("browser", "email", "files"):
    try:
        _il.import_module(f"tools.{_sub}")
    except Exception:
        pass

# ── Plan mode tools (EnterPlanMode / ExitPlanMode) ────────────────────────

from pathlib import Path as _Path


def _enter_plan_mode(params: dict, config: dict) -> str:
    if config.get("permission_mode") == "plan":
        return "Already in plan mode. Write your plan to the plan file, then call ExitPlanMode."

    session_id = config.get("_session_id", "default")
    plans_dir  = _Path(config.get("_worktree_cwd") or _Path.cwd()) / ".nano_claude" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path  = plans_dir / f"{session_id}.md"

    task_desc = params.get("task_description", "")
    if not plan_path.exists() or plan_path.stat().st_size == 0:
        header = f"# Plan: {task_desc}\n\n" if task_desc else "# Plan\n\n"
        plan_path.write_text(header, encoding="utf-8")

    import runtime
    sctx = runtime.get_ctx(config)
    sctx.prev_permission_mode = config.get("permission_mode", "auto")
    config["permission_mode"]  = "plan"
    sctx.plan_file             = str(plan_path)
    return (
        f"Plan mode activated. Plan file: {plan_path}\n"
        "Write your step-by-step plan to the plan file, then call ExitPlanMode when ready to implement."
    )


def _exit_plan_mode(params: dict, config: dict) -> str:
    if config.get("permission_mode") != "plan":
        return "Not in plan mode."
    import runtime
    sctx = runtime.get_ctx(config)
    plan_file = sctx.plan_file or ""
    plan_content = ""
    if plan_file:
        try:
            plan_content = _Path(plan_file).read_text(encoding="utf-8").strip()
        except Exception:
            plan_content = ""

    # Reject if plan file is effectively empty (only whitespace / top-level title)
    # A top-level title is exactly "# ..." (single #).  ## sections count as content.
    non_trivial_lines = [
        l for l in plan_content.splitlines()
        if l.strip() and not (l.strip().startswith("# ") and not l.strip().startswith("## "))
    ]
    if not non_trivial_lines:
        return (
            "Plan is empty — please write your step-by-step plan to the plan file "
            f"({plan_file}) before exiting plan mode."
        )

    config["permission_mode"] = sctx.prev_permission_mode or "auto"
    sctx.prev_permission_mode = None
    sctx.plan_file = None
    return (
        f"Plan mode exited. Resuming normal permissions.\n\n"
        f"Plan content:\n{plan_content}\n\n"
        "Wait for the user to approve the plan before executing any steps."
    )


_plan_schema_enter = {
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
_plan_schema_exit = {
    "name": "ExitPlanMode",
    "description": "Exit plan mode and return to normal permissions to begin executing the plan.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

register_tool(ToolDef("EnterPlanMode", _plan_schema_enter, _enter_plan_mode,
                       read_only=True, concurrent_safe=False))
register_tool(ToolDef("ExitPlanMode",  _plan_schema_exit,  _exit_plan_mode,
                       read_only=False, concurrent_safe=False))
