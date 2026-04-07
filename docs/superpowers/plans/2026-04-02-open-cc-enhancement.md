# Open-CC Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve cheetahclaws into a capable AI coding CLI with context management, pluggable tools, sub-agents, memory, skills, and diff view.

**Architecture:** Layered enhancement on existing 6-file structure. New modules (tool_registry, compaction, memory, subagent, skills) added as flat files alongside existing code. Modules communicate via function params and dataclasses, no globals. Agent.py gains depth/cancel_check params for sub-agent support.

**Tech Stack:** Python 3.8-3.10, threading/concurrent.futures, difflib, tiktoken (optional), existing deps (anthropic, openai, httpx, rich)

**Spec:** `docs/superpowers/specs/2026-04-02-open-cc-design.md`

---

## File Structure Overview

**New files:**
- `tool_registry.py` — Tool plugin registry (ToolDef dataclass, register/get/execute)
- `compaction.py` — Context window management (autoCompact + snip)
- `memory.py` — File-based memory system (MEMORY.md index + per-entry markdown files)
- `subagent.py` — Sub-agent lifecycle (ThreadPoolExecutor, depth limit, cancel)
- `skills.py` — Skill loading and execution (markdown frontmatter, prompt injection)
- `tests/test_tool_registry.py` — Tests for tool registry
- `tests/test_compaction.py` — Tests for compaction
- `tests/test_memory.py` — Tests for memory
- `tests/test_subagent.py` — Tests for sub-agent
- `tests/test_skills.py` — Tests for skills
- `tests/test_diff_view.py` — Tests for diff generation

**Modified files:**
- `tools.py` — Refactor to use registry, add diff generation in Edit/Write
- `agent.py` — Add compaction call, depth/cancel_check params, sub-agent tool dispatch
- `context.py` — Inject memory context into system prompt
- `config.py` — Add new config keys (max_tool_output, max_depth, etc.)
- `cheetahclaws.py` — Add /memory, /skill, /agents slash commands, diff rendering

---

## Task 1: Tool Registry (`tool_registry.py`)

**Files:**
- Create: `tool_registry.py`
- Create: `tests/test_tool_registry.py`

- [ ] **Step 1: Write failing tests for tool registry**

```python
# tests/test_tool_registry.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tool_registry import ToolDef, register_tool, get_tool, get_all_tools, get_tool_schemas, execute_tool, clear_registry


@pytest.fixture(autouse=True)
def clean_registry():
    clear_registry()
    yield
    clear_registry()


def _echo_tool(params, config):
    return f"echo: {params['text']}"


ECHO_SCHEMA = {
    "name": "Echo",
    "description": "Echo input text",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


def test_register_and_get():
    td = ToolDef(name="Echo", schema=ECHO_SCHEMA, func=_echo_tool,
                 read_only=True, concurrent_safe=True)
    register_tool(td)
    assert get_tool("Echo") is td
    assert get_tool("NonExistent") is None


def test_get_all_tools():
    td = ToolDef(name="Echo", schema=ECHO_SCHEMA, func=_echo_tool,
                 read_only=True, concurrent_safe=True)
    register_tool(td)
    all_tools = get_all_tools()
    assert len(all_tools) == 1
    assert all_tools[0].name == "Echo"


def test_get_tool_schemas():
    td = ToolDef(name="Echo", schema=ECHO_SCHEMA, func=_echo_tool,
                 read_only=True, concurrent_safe=True)
    register_tool(td)
    schemas = get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "Echo"


def test_execute_tool():
    td = ToolDef(name="Echo", schema=ECHO_SCHEMA, func=_echo_tool,
                 read_only=True, concurrent_safe=True)
    register_tool(td)
    result = execute_tool("Echo", {"text": "hello"}, {})
    assert result == "echo: hello"


def test_execute_unknown_tool():
    result = execute_tool("Unknown", {}, {})
    assert "Unknown tool" in result


def test_output_truncation():
    def big_output(params, config):
        return "x" * 100_000

    td = ToolDef(name="Big", schema=ECHO_SCHEMA, func=big_output,
                 read_only=True, concurrent_safe=True)
    register_tool(td)
    result = execute_tool("Big", {}, {}, max_output=1000)
    assert len(result) < 2000
    assert "truncated" in result


def test_duplicate_register_overwrites():
    td1 = ToolDef(name="Echo", schema=ECHO_SCHEMA, func=_echo_tool,
                  read_only=True, concurrent_safe=True)
    td2 = ToolDef(name="Echo", schema=ECHO_SCHEMA, func=lambda p, c: "v2",
                  read_only=False, concurrent_safe=False)
    register_tool(td1)
    register_tool(td2)
    assert get_tool("Echo").read_only is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_tool_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tool_registry'`

- [ ] **Step 3: Implement tool_registry.py**

```python
# tool_registry.py
"""
Tool plugin registry for cheetahclaws.

Provides a central registry for tool definitions. Tools register themselves
at import time. The agent loop queries the registry for schemas and dispatches
execution through it.

Public API:
    register_tool(tool_def)  — add a tool to the registry
    get_tool(name)           — look up by name
    get_all_tools()          — list all registered tools
    get_tool_schemas()       — schemas for API calls
    execute_tool(name, ...)  — dispatch execution with output truncation
    clear_registry()         — reset (for testing)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


DEFAULT_MAX_OUTPUT = 32_000  # ~8K tokens


@dataclass
class ToolDef:
    """Definition of a registered tool."""
    name: str
    schema: dict
    func: Callable[[dict, dict], str]
    read_only: bool = False
    concurrent_safe: bool = False


_TOOLS: Dict[str, ToolDef] = {}


def register_tool(tool_def: ToolDef) -> None:
    """Register a tool. Overwrites if name already exists."""
    _TOOLS[tool_def.name] = tool_def


def get_tool(name: str) -> Optional[ToolDef]:
    """Look up a tool by name. Returns None if not found."""
    return _TOOLS.get(name)


def get_all_tools() -> List[ToolDef]:
    """Return all registered tools in registration order."""
    return list(_TOOLS.values())


def get_tool_schemas() -> List[dict]:
    """Return JSON schemas for all tools (for API calls)."""
    return [t.schema for t in _TOOLS.values()]


def execute_tool(name: str, params: dict, config: dict,
                 max_output: int = DEFAULT_MAX_OUTPUT) -> str:
    """
    Execute a tool by name with output truncation.

    Args:
        name: tool name, str
        params: tool input parameters, dict
        config: global config, dict
        max_output: max chars before truncation, int
    Returns:
        tool result string, possibly truncated
    """
    tool = _TOOLS.get(name)
    if tool is None:
        return f"Error: Unknown tool '{name}'"

    try:
        result = tool.func(params, config)
    except Exception as e:
        return f"Error executing {name}: {e}"

    # Hard truncation at source
    if len(result) > max_output:
        head_size = max_output // 2
        tail_size = max_output // 4
        snipped = len(result) - head_size - tail_size
        result = (
            f"{result[:head_size]}\n\n"
            f"[... {snipped} chars truncated ...]\n\n"
            f"{result[-tail_size:]}"
        )

    return result


def clear_registry() -> None:
    """Clear all registered tools. For testing only."""
    _TOOLS.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_tool_registry.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd D:/git/open-cc/cheetahclaws
git add tool_registry.py tests/test_tool_registry.py
git commit -m "feat: add tool plugin registry with output truncation"
```

---

## Task 2: Migrate Built-in Tools to Registry (`tools.py` refactor)

**Files:**
- Modify: `tools.py` (lines 1-360)
- Modify: `agent.py` (lines 5-6 imports, lines 52-130 run function)

- [ ] **Step 1: Refactor tools.py to register via tool_registry**

The key changes to `tools.py`:
1. Import `tool_registry` and call `register_tool()` for each built-in tool
2. Keep `TOOL_SCHEMAS` and `execute_tool` as thin wrappers for backward compat
3. Keep all `_read`, `_write`, etc. implementations unchanged

Add at the top of `tools.py` after existing imports:

```python
from tool_registry import ToolDef, register_tool
from tool_registry import execute_tool as _registry_execute
from tool_registry import get_tool_schemas as _registry_schemas
```

At the bottom of `tools.py` (after all function definitions, replacing the existing `execute_tool` function), add the registration block and compatibility layer:

```python
# --- Tool Registration ---------------------------------------------------

def _register_builtin_tools():
    """Register all 8 built-in tools with the registry."""
    _tool_defs = [
        ToolDef(name="Read",      schema=TOOL_SCHEMAS[0], func=lambda p, c: _read(**p),
                read_only=True,  concurrent_safe=True),
        ToolDef(name="Write",     schema=TOOL_SCHEMAS[1], func=lambda p, c: _write(**p),
                read_only=False, concurrent_safe=False),
        ToolDef(name="Edit",      schema=TOOL_SCHEMAS[2], func=lambda p, c: _edit(**p),
                read_only=False, concurrent_safe=False),
        ToolDef(name="Bash",      schema=TOOL_SCHEMAS[3], func=lambda p, c: _bash(**p),
                read_only=False, concurrent_safe=False),
        ToolDef(name="Glob",      schema=TOOL_SCHEMAS[4], func=lambda p, c: _glob(**p),
                read_only=True,  concurrent_safe=True),
        ToolDef(name="Grep",      schema=TOOL_SCHEMAS[5], func=lambda p, c: _grep(**p),
                read_only=True,  concurrent_safe=True),
        ToolDef(name="WebFetch",  schema=TOOL_SCHEMAS[6], func=lambda p, c: _webfetch(**p),
                read_only=True,  concurrent_safe=True),
        ToolDef(name="WebSearch", schema=TOOL_SCHEMAS[7], func=lambda p, c: _websearch(**p),
                read_only=True,  concurrent_safe=True),
    ]
    for td in _tool_defs:
        register_tool(td)


_register_builtin_tools()


# --- Backward Compatibility -----------------------------------------------

def execute_tool(name, inputs, permission_mode="auto", ask_permission=None):
    """
    Backward-compatible wrapper. Delegates to tool_registry.execute_tool
    but preserves the permission_mode/ask_permission interface used by agent.py.
    """
    # Permission check (same logic as before)
    from tool_registry import get_tool
    tool = get_tool(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"

    needs_permission = False
    if permission_mode == "manual":
        needs_permission = True
    elif permission_mode == "auto":
        if not tool.read_only:
            if name == "Bash" and _is_safe_bash(inputs.get("command", "")):
                needs_permission = False
            else:
                needs_permission = True

    if needs_permission and ask_permission:
        desc = _permission_desc_for(name, inputs)
        if not ask_permission(desc):
            return "[Tool call denied by user]"

    return _registry_execute(name, inputs, {})


def _permission_desc_for(name, inputs):
    """Build a human-readable permission description."""
    if name == "Bash":
        return f"Run: {inputs.get('command', '')}"
    elif name == "Write":
        return f"Write to: {inputs.get('file_path', '')}"
    elif name == "Edit":
        return f"Edit: {inputs.get('file_path', '')}"
    return f"{name}: {inputs}"
```

Remove the old `execute_tool` function (lines 304-360 in original tools.py) and replace with the above.

- [ ] **Step 2: Update agent.py imports**

Change `agent.py` lines 5-6 from:

```python
from tools import TOOL_SCHEMAS, execute_tool
```

to:

```python
from tool_registry import get_tool_schemas, get_tool
from tools import execute_tool  # backward-compat wrapper with permissions
import tools as _tools_init  # ensure built-in tools are registered
```

Update the `run()` function where it references `TOOL_SCHEMAS` (around line 65) to use `get_tool_schemas()` instead.

- [ ] **Step 3: Run the existing code to verify nothing is broken**

Run: `cd D:/git/open-cc/cheetahclaws && python -c "from tools import execute_tool; from tool_registry import get_all_tools; print(f'{len(get_all_tools())} tools registered'); print(execute_tool('Read', {'file_path': 'config.py'}, 'auto'))"`
Expected: `8 tools registered` + contents of config.py

- [ ] **Step 4: Commit**

```bash
cd D:/git/open-cc/cheetahclaws
git add tools.py agent.py
git commit -m "refactor: migrate built-in tools to plugin registry"
```

---

## Task 3: Diff View (`tools.py` + `cheetahclaws.py`)

**Files:**
- Modify: `tools.py` (the `_edit` and `_write` functions)
- Modify: `cheetahclaws.py` (the `print_tool_end` function)
- Create: `tests/test_diff_view.py`

- [ ] **Step 1: Write failing tests for diff generation**

```python
# tests/test_diff_view.py
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_generate_unified_diff():
    from tools import generate_unified_diff
    old = "line1\nline2\nline3\n"
    new = "line1\nline2_modified\nline3\n"
    diff = generate_unified_diff(old, new, "test.py")
    assert "--- a/test.py" in diff
    assert "+++ b/test.py" in diff
    assert "-line2" in diff
    assert "+line2_modified" in diff


def test_generate_unified_diff_empty_old():
    from tools import generate_unified_diff
    diff = generate_unified_diff("", "new content\n", "test.py")
    assert "+new content" in diff


def test_edit_returns_diff(tmp_path):
    from tools import _edit
    f = tmp_path / "test.txt"
    f.write_text("hello world\n")
    result = _edit(str(f), "hello", "goodbye")
    assert "-hello world" in result
    assert "+goodbye world" in result


def test_write_existing_returns_diff(tmp_path):
    from tools import _write
    f = tmp_path / "test.txt"
    f.write_text("old content\n")
    result = _write(str(f), "new content\n")
    assert "-old content" in result
    assert "+new content" in result


def test_write_new_file_no_diff(tmp_path):
    from tools import _write
    f = tmp_path / "new.txt"
    result = _write(str(f), "content\n")
    assert "Created" in result
    assert "---" not in result  # no diff for new files


def test_diff_truncation():
    from tools import generate_unified_diff, maybe_truncate_diff
    old = "\n".join(f"line{i}" for i in range(200))
    new = "\n".join(f"CHANGED{i}" for i in range(200))
    diff = generate_unified_diff(old, new, "big.py")
    truncated = maybe_truncate_diff(diff, max_lines=50)
    assert "more lines" in truncated
    assert truncated.count("\n") < 60
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_diff_view.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_unified_diff'`

- [ ] **Step 3: Add diff generation functions to tools.py**

Add near the top of `tools.py`, after the existing imports:

```python
import difflib


def generate_unified_diff(old, new, filename, context_lines=3):
    """
    Generate a unified diff between two strings.

    Args:
        old: original content, str
        new: modified content, str
        filename: display filename, str
        context_lines: lines of surrounding context, int
    Returns:
        unified diff string, str
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=context_lines,
    )
    return "".join(diff)


def maybe_truncate_diff(diff_text, max_lines=80):
    """
    Truncate diff for display if it exceeds max_lines.

    Args:
        diff_text: unified diff string, str
        max_lines: max lines to show, int
    Returns:
        possibly truncated diff string, str
    """
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    shown = lines[:max_lines]
    remaining = len(lines) - max_lines
    return "\n".join(shown) + f"\n\n[... {remaining} more lines ...]"
```

- [ ] **Step 4: Modify _edit to return diff**

Replace the `_edit` function (originally lines 175-192) with:

```python
def _edit(file_path, old_string, new_string, replace_all=False):
    p = Path(file_path)
    if not p.exists():
        return f"Error: {file_path} not found"
    content = p.read_text(encoding="utf-8")
    if old_string not in content:
        return f"Error: old_string not found in {file_path}"
    old_content = content
    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        count = content.count(old_string)
        if count > 1:
            return f"Error: old_string appears {count} times; use replace_all=true or provide more context"
        new_content = content.replace(old_string, new_string, 1)
    p.write_text(new_content, encoding="utf-8")
    filename = p.name
    diff = generate_unified_diff(old_content, new_content, filename)
    return f"Changes applied to {filename}:\n\n{diff}" if diff else f"No changes in {filename}"
```

- [ ] **Step 5: Modify _write to return diff for existing files**

Replace the `_write` function (originally lines 164-172) with:

```python
def _write(file_path, content):
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old_content = p.read_text(encoding="utf-8")
        p.write_text(content, encoding="utf-8")
        filename = p.name
        diff = generate_unified_diff(old_content, content, filename)
        truncated = maybe_truncate_diff(diff)
        return f"File updated — {filename}:\n\n{truncated}" if diff else f"No changes in {filename}"
    else:
        p.write_text(content, encoding="utf-8")
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"Created {file_path} ({line_count} lines)"
```

- [ ] **Step 6: Add diff rendering to cheetahclaws.py**

Add a new function after the `flush_response` function (around line 105):

```python
def render_diff(text):
    """Render unified diff with ANSI colors."""
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            print(clr(line, "bold"))
        elif line.startswith("+"):
            print(clr(line, "green"))
        elif line.startswith("-"):
            print(clr(line, "red"))
        elif line.startswith("@@"):
            print(clr(line, "cyan"))
        else:
            print(line)


def _has_diff(text):
    """Check if text contains a unified diff."""
    return ("--- a/" in text and "+++ b/" in text) or text.startswith("@@")
```

Modify the `print_tool_end` function (originally line 115) to detect and render diffs:

```python
def print_tool_end(name, result, verbose):
    if name in ("Edit", "Write") and _has_diff(result):
        # Extract the diff portion and render with colors
        parts = result.split("\n\n", 1)
        if len(parts) == 2:
            info(parts[0])           # "Changes applied to file.py:"
            render_diff(parts[1])    # colored diff
        else:
            render_diff(result)
    elif verbose:
        info(f"  {C.get('dim','')}Result: {result[:200]}{'...' if len(result)>200 else ''}{C.get('reset','')}")
```

Also ensure the color dict `C` (line 60-70 of cheetahclaws.py) includes the needed keys. Check and add if missing:

```python
# In the C dict, ensure these entries exist:
"bold": "\033[1m",
"cyan": "\033[36m",
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_diff_view.py -v`
Expected: All 6 tests PASS

- [ ] **Step 8: Manual smoke test**

Run: `cd D:/git/open-cc/cheetahclaws && python -c "
from tools import _write, _edit
import tempfile, os
f = os.path.join(tempfile.mkdtemp(), 'test.py')
print(_write(f, 'def hello():\n    return 42\n'))
print('---')
print(_edit(f, 'return 42', 'return 99'))
"`
Expected: First call shows "Created", second call shows colored diff with red `-return 42` and green `+return 99`

- [ ] **Step 9: Commit**

```bash
cd D:/git/open-cc/cheetahclaws
git add tools.py cheetahclaws.py tests/test_diff_view.py
git commit -m "feat: add git-style diff view for Edit and Write tools"
```

---

## Task 4: Context Window Management (`compaction.py`)

**Files:**
- Create: `compaction.py`
- Create: `tests/test_compaction.py`
- Modify: `agent.py` (add compaction call in run loop)
- Modify: `providers.py` (add context_limit to PROVIDERS entries)

- [ ] **Step 1: Write failing tests for compaction**

```python
# tests/test_compaction.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from compaction import (
    estimate_tokens,
    get_context_limit,
    snip_old_tool_results,
    find_split_point,
)


def test_estimate_tokens_simple():
    msgs = [{"role": "user", "content": "hello world"}]
    tokens = estimate_tokens(msgs)
    assert 2 < tokens < 10  # "hello world" ~ 2-3 tokens


def test_estimate_tokens_empty():
    assert estimate_tokens([]) == 0


def test_estimate_tokens_with_tool_results():
    msgs = [
        {"role": "user", "content": "read file"},
        {"role": "tool", "content": "x" * 10000, "name": "Read"},
    ]
    tokens = estimate_tokens(msgs)
    assert tokens > 2000  # 10000 / ~3.5


def test_get_context_limit_known_model():
    limit = get_context_limit("gpt-4o")
    assert limit >= 100_000


def test_get_context_limit_unknown_model():
    limit = get_context_limit("unknown-model-xyz")
    assert limit == 128_000  # default fallback


def test_snip_old_tool_results():
    msgs = [
        {"role": "user", "content": "read file"},
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1", "name": "Read", "input": {}}]},
        {"role": "tool", "content": "x" * 5000, "tool_call_id": "1", "name": "Read"},
        {"role": "assistant", "content": "got it"},
        {"role": "user", "content": "now do something else"},
        {"role": "assistant", "content": "sure"},
    ]
    snipped = snip_old_tool_results(msgs, max_chars=100, preserve_last_n_turns=2)
    tool_msg = snipped[2]
    assert len(tool_msg["content"]) < 300
    assert "snipped" in tool_msg["content"].lower()


def test_snip_preserves_recent():
    msgs = [
        {"role": "user", "content": "recent"},
        {"role": "tool", "content": "x" * 5000, "tool_call_id": "1", "name": "Read"},
    ]
    snipped = snip_old_tool_results(msgs, max_chars=100, preserve_last_n_turns=4)
    # Everything is recent, nothing should be snipped
    assert snipped[1]["content"] == "x" * 5000


def test_find_split_point():
    msgs = [{"role": "user", "content": "x" * 1000}] * 10
    split = find_split_point(msgs, keep_ratio=0.3)
    assert 5 <= split <= 8  # keep ~30% means split at ~70%
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_compaction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'compaction'`

- [ ] **Step 3: Add context_limit to providers.py**

In `providers.py`, add a `context_limit` field to each entry in the `PROVIDERS` dict. Add after the `"models"` field in each provider entry:

```python
# Add to each provider in PROVIDERS dict:
# anthropic:
"context_limit": 200_000,
# openai:
"context_limit": 128_000,
# gemini:
"context_limit": 1_000_000,
# kimi:
"context_limit": 128_000,
# qwen:
"context_limit": 1_000_000,
# zhipu:
"context_limit": 128_000,
# deepseek:
"context_limit": 64_000,
# ollama:
"context_limit": 128_000,
# lmstudio:
"context_limit": 128_000,
# custom:
"context_limit": 128_000,
```

- [ ] **Step 4: Implement compaction.py**

```python
# compaction.py
"""
Context window management for cheetahclaws.

Two-layer compression strategy:
  Layer 1 (snip): Truncate old tool results — rule-based, no model call
  Layer 2 (autoCompact): Summarize old messages — requires model call

Public API:
    maybe_compact(state, config)    — run both layers if needed
    estimate_tokens(messages)       — fast token estimation
    get_context_limit(model)        — context window size for model
    snip_old_tool_results(...)      — truncate old tool outputs
    find_split_point(...)           — find where to split old/recent
"""
from __future__ import annotations

from typing import List, Optional

from providers import detect_provider, PROVIDERS, stream


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(messages: List[dict]) -> int:
    """
    Estimate token count for a message list.

    Uses chars/3.5 as a rough heuristic. tiktoken can be used for GPT
    models if installed, but is not required.

    Args:
        messages: list of message dicts with "content" field
    Returns:
        estimated token count, int
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", ""))
                elif isinstance(block, str):
                    total_chars += len(block)
    return int(total_chars / 3.5)


def get_context_limit(model: str) -> int:
    """
    Return context window size for a given model.

    Args:
        model: model name string
    Returns:
        context limit in tokens, int
    """
    provider_name = detect_provider(model)
    provider = PROVIDERS.get(provider_name, {})
    return provider.get("context_limit", 128_000)


# ---------------------------------------------------------------------------
# Layer 1: Tool-result snipping (rule-based, no model call)
# ---------------------------------------------------------------------------

def snip_old_tool_results(
    messages: List[dict],
    max_chars: int = 2000,
    preserve_last_n_turns: int = 6,
) -> List[dict]:
    """
    Truncate old tool result contents to max_chars.

    Preserves the most recent preserve_last_n_turns messages untouched.
    For older tool results, keeps first and last lines with a snip marker.

    Args:
        messages: message list (mutated in place and returned)
        max_chars: max chars per tool result, int
        preserve_last_n_turns: how many recent messages to leave alone, int
    Returns:
        the same list, with old tool results truncated
    """
    cutoff = max(0, len(messages) - preserve_last_n_turns)

    for i in range(cutoff):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if len(content) <= max_chars:
            continue

        # Keep first half + last quarter of allowed chars
        head = content[: max_chars // 2]
        tail = content[-(max_chars // 4) :]
        snipped = len(content) - len(head) - len(tail)
        messages[i] = {
            **msg,
            "content": f"{head}\n\n[... {snipped} chars snipped ...]\n\n{tail}",
        }

    return messages


# ---------------------------------------------------------------------------
# Layer 2: Auto-compact (model-driven summary)
# ---------------------------------------------------------------------------

def find_split_point(messages: List[dict], keep_ratio: float = 0.3) -> int:
    """
    Find the index that splits messages into [old | recent],
    keeping approximately keep_ratio of total tokens in 'recent'.

    Args:
        messages: message list
        keep_ratio: fraction of tokens to keep as recent, float
    Returns:
        split index, int
    """
    total = estimate_tokens(messages)
    target_recent = int(total * keep_ratio)

    cumulative = 0
    for i in range(len(messages) - 1, -1, -1):
        content = messages[i].get("content", "")
        if isinstance(content, str):
            cumulative += int(len(content) / 3.5)
        if cumulative >= target_recent:
            return i
    return 0


_SUMMARY_PROMPT = """Summarize the conversation so far. Preserve:
- Key decisions and their reasoning
- File paths that were read or modified
- Code changes made and their purpose
- Unfinished tasks or pending questions
- Important error messages or debugging findings

Be concise but complete. Use bullet points."""


def compact_messages(messages: List[dict], config: dict) -> List[dict]:
    """
    Summarize old messages via model call, keep recent messages verbatim.

    Args:
        messages: full message list
        config: global config dict (needs "model" key)
    Returns:
        new message list: [summary_msg, ack_msg, *recent]
    """
    split = find_split_point(messages, keep_ratio=0.3)
    if split <= 1:
        return messages  # nothing worth compacting

    old = messages[:split]
    recent = messages[split:]

    # Build a summary request
    summary_messages = [
        {"role": "user", "content": _SUMMARY_PROMPT},
    ]
    # Prepend old messages as context
    for msg in old:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "tool":
            summary_messages.append({
                "role": "user",
                "content": f"[Tool {msg.get('name', '?')} result]: {content[:1000]}",
            })
        else:
            summary_messages.append({"role": role, "content": str(content)[:2000]})
    summary_messages.append({"role": "user", "content": _SUMMARY_PROMPT})

    # Call model for summary (non-streaming, collect full response)
    model = config.get("model", "gpt-4o")
    summary_text = ""
    for event in stream(model, "You are a helpful summarizer.", summary_messages, [], config):
        if hasattr(event, "text") and not hasattr(event, "tool_calls"):
            summary_text += event.text

    return [
        {"role": "user", "content": f"[Conversation summary]\n{summary_text}"},
        {"role": "assistant", "content": "Understood, I have the context from the summary."},
        *recent,
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def maybe_compact(state, config: dict) -> bool:
    """
    Run compaction if needed. Called before each API call in agent loop.

    Layer 1 (snip) runs first — cheap and fast.
    Layer 2 (autoCompact) runs if still over threshold.

    Args:
        state: AgentState with .messages
        config: global config dict
    Returns:
        True if compaction occurred, bool
    """
    model = config.get("model", "gpt-4o")
    limit = get_context_limit(model)
    threshold = int(limit * 0.7)

    current = estimate_tokens(state.messages)
    if current <= threshold:
        return False

    # Layer 1: snip old tool results
    snip_old_tool_results(state.messages)
    current = estimate_tokens(state.messages)
    if current <= threshold:
        return True

    # Layer 2: auto-compact via model summary
    state.messages = compact_messages(state.messages, config)
    return True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_compaction.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Integrate compaction into agent.py**

In `agent.py`, add import at the top:

```python
from compaction import maybe_compact
```

In the `run()` function, add compaction check before the API call. Inside the `while True` loop, before the `for event in stream(...)` line, add:

```python
        # --- Context compaction ---
        maybe_compact(state, config)
```

- [ ] **Step 7: Commit**

```bash
cd D:/git/open-cc/cheetahclaws
git add compaction.py tests/test_compaction.py agent.py providers.py
git commit -m "feat: add context window management with snip + auto-compact"
```

---

## Task 5: Memory System (`memory.py`)

**Files:**
- Create: `memory.py`
- Create: `tests/test_memory.py`
- Modify: `context.py` (inject memory into system prompt)
- Modify: `tools.py` (register MemorySave + MemoryDelete tools)

- [ ] **Step 1: Write failing tests for memory**

```python
# tests/test_memory.py
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from memory import MemoryEntry, save_memory, load_index, delete_memory, search_memory, get_memory_context


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    """Use a temp directory for memory storage."""
    import memory
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory, "INDEX_FILE", tmp_path / "MEMORY.md")
    return tmp_path


def test_save_and_load(memory_dir):
    entry = MemoryEntry(
        name="test entry",
        description="a test memory",
        type="user",
        content="User prefers Python.",
    )
    save_memory(entry)

    index = load_index()
    assert len(index) == 1
    assert index[0].name == "test entry"
    assert index[0].content == "User prefers Python."


def test_save_creates_file(memory_dir):
    entry = MemoryEntry(name="file check", description="check file", type="feedback",
                        content="Always use type hints.")
    save_memory(entry)
    files = list(memory_dir.glob("*.md"))
    assert len(files) == 2  # MEMORY.md + the entry file


def test_delete_memory(memory_dir):
    entry = MemoryEntry(name="to delete", description="temp", type="project",
                        content="Temporary.")
    save_memory(entry)
    assert len(load_index()) == 1
    delete_memory("to delete")
    assert len(load_index()) == 0


def test_delete_nonexistent(memory_dir):
    # Should not raise
    delete_memory("nonexistent")


def test_search_memory(memory_dir):
    save_memory(MemoryEntry(name="python pref", description="user likes python",
                            type="user", content="Python is preferred."))
    save_memory(MemoryEntry(name="go pref", description="user knows go",
                            type="user", content="User knows Go."))
    results = search_memory("python")
    assert len(results) == 1
    assert results[0].name == "python pref"


def test_get_memory_context(memory_dir):
    save_memory(MemoryEntry(name="ctx test", description="context entry",
                            type="feedback", content="Be concise."))
    ctx = get_memory_context()
    assert "ctx test" in ctx
    assert "context entry" in ctx


def test_get_memory_context_empty(memory_dir):
    ctx = get_memory_context()
    assert ctx == ""


def test_update_existing(memory_dir):
    save_memory(MemoryEntry(name="evolving", description="v1", type="user", content="Version 1."))
    save_memory(MemoryEntry(name="evolving", description="v2", type="user", content="Version 2."))
    index = load_index()
    assert len(index) == 1
    assert index[0].description == "v2"
    assert index[0].content == "Version 2."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory'`

- [ ] **Step 3: Implement memory.py**

```python
# memory.py
"""
File-based memory system for cheetahclaws.

Stores memories as individual markdown files with YAML frontmatter.
MEMORY.md serves as an index. Memories persist across conversations.

Public API:
    save_memory(entry)       — write/update a memory file + index
    load_index()             — parse all memory files
    delete_memory(name)      — remove a memory file + index entry
    search_memory(query)     — keyword search across memories
    get_memory_context()     — return index text for system prompt
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List


MEMORY_DIR = Path.home() / ".cheetahclaws" / "memory"
INDEX_FILE = MEMORY_DIR / "MEMORY.md"


@dataclass
class MemoryEntry:
    """A single memory record."""
    name: str
    description: str
    type: str               # "user" | "feedback" | "project" | "reference"
    content: str
    file_path: str = ""
    created: str = ""


def _slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s]+", "_", slug).strip("_")
    return slug[:60] or "memory"


def _parse_frontmatter(text: str) -> tuple:
    """
    Parse YAML-like frontmatter from markdown text.

    Returns:
        (metadata_dict, body_text)
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip()
    return meta, parts[2].strip()


def _write_entry_file(entry: MemoryEntry) -> Path:
    """Write a memory entry to its markdown file."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    slug = _slugify(entry.name)
    file_path = MEMORY_DIR / f"{slug}.md"

    created = entry.created or datetime.now().strftime("%Y-%m-%d")
    content = (
        f"---\n"
        f"name: {entry.name}\n"
        f"description: {entry.description}\n"
        f"type: {entry.type}\n"
        f"created: {created}\n"
        f"---\n\n"
        f"{entry.content}\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


def _update_index(entries: List[MemoryEntry]) -> None:
    """Rewrite MEMORY.md index from a list of entries."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for e in entries:
        slug = _slugify(e.name)
        lines.append(f"- [{e.name}]({slug}.md) — {e.description}")
    INDEX_FILE.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def save_memory(entry: MemoryEntry) -> None:
    """
    Save or update a memory entry.

    If a memory with the same name exists, it is overwritten.
    Updates both the entry file and the MEMORY.md index.
    """
    # Load existing entries, filter out same-name
    existing = [e for e in load_index() if e.name != entry.name]
    existing.append(entry)

    _write_entry_file(entry)
    _update_index(existing)


def load_index() -> List[MemoryEntry]:
    """
    Load all memory entries by scanning markdown files in MEMORY_DIR.

    Returns:
        list of MemoryEntry, sorted by name
    """
    if not MEMORY_DIR.exists():
        return []

    entries = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        try:
            text = f.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            entries.append(MemoryEntry(
                name=meta.get("name", f.stem),
                description=meta.get("description", ""),
                type=meta.get("type", "user"),
                content=body,
                file_path=str(f),
                created=meta.get("created", ""),
            ))
        except Exception:
            continue
    return entries


def delete_memory(name: str) -> None:
    """
    Delete a memory by name.

    Removes the entry file and updates the index.
    """
    slug = _slugify(name)
    file_path = MEMORY_DIR / f"{slug}.md"
    if file_path.exists():
        file_path.unlink()

    remaining = [e for e in load_index() if e.name != name]
    _update_index(remaining)


def search_memory(query: str) -> List[MemoryEntry]:
    """
    Search memories by keyword match against name, description, and content.

    Args:
        query: search string (case-insensitive)
    Returns:
        list of matching MemoryEntry
    """
    query_lower = query.lower()
    results = []
    for entry in load_index():
        searchable = f"{entry.name} {entry.description} {entry.content}".lower()
        if query_lower in searchable:
            results.append(entry)
    return results


def get_memory_context() -> str:
    """
    Return memory index content for system prompt injection.

    Returns:
        MEMORY.md content string, or empty string if no memories
    """
    if not INDEX_FILE.exists():
        return ""
    text = INDEX_FILE.read_text(encoding="utf-8").strip()
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_memory.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Register MemorySave and MemoryDelete tools**

Add to the bottom of `tools.py`, after the `_register_builtin_tools()` call:

```python
# --- Memory tools ---------------------------------------------------------

from memory import save_memory, delete_memory, MemoryEntry
from datetime import datetime as _dt


def _memory_save(params, config):
    entry = MemoryEntry(
        name=params["name"],
        description=params["description"],
        type=params["type"],
        content=params["content"],
        created=_dt.now().strftime("%Y-%m-%d"),
    )
    save_memory(entry)
    return f"Memory saved: {entry.name}"


def _memory_delete(params, config):
    delete_memory(params["name"])
    return f"Memory deleted: {params['name']}"


_MEMORY_SAVE_SCHEMA = {
    "name": "MemorySave",
    "description": "Save a persistent memory that survives across conversations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short name for the memory"},
            "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"],
                     "description": "Memory category"},
            "description": {"type": "string", "description": "One-line description"},
            "content": {"type": "string", "description": "Full memory content"},
        },
        "required": ["name", "type", "description", "content"],
    },
}

_MEMORY_DELETE_SCHEMA = {
    "name": "MemoryDelete",
    "description": "Delete a persistent memory by name.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of memory to delete"},
        },
        "required": ["name"],
    },
}

register_tool(ToolDef(name="MemorySave", schema=_MEMORY_SAVE_SCHEMA,
                       func=_memory_save, read_only=False, concurrent_safe=True))
register_tool(ToolDef(name="MemoryDelete", schema=_MEMORY_DELETE_SCHEMA,
                       func=_memory_delete, read_only=False, concurrent_safe=True))
```

- [ ] **Step 6: Inject memory into context.py**

Modify `context.py`'s `build_system_prompt()` function (line 92). Add import at top:

```python
from memory import get_memory_context
```

At the end of `build_system_prompt()`, before the return statement, add:

```python
    memory_ctx = get_memory_context()
    if memory_ctx:
        prompt += f"\n\n# Memory\nYour persistent memories:\n{memory_ctx}\n"
```

- [ ] **Step 7: Commit**

```bash
cd D:/git/open-cc/cheetahclaws
git add memory.py tests/test_memory.py tools.py context.py
git commit -m "feat: add file-based memory system with MemorySave/MemoryDelete tools"
```

---

## Task 6: Sub-Agent System (`subagent.py`)

**Files:**
- Create: `subagent.py`
- Create: `tests/test_subagent.py`
- Modify: `agent.py` (add depth, cancel_check params)
- Modify: `tools.py` (register Agent, CheckAgentResult, ListAgentTasks)

- [ ] **Step 1: Write failing tests for sub-agent**

```python
# tests/test_subagent.py
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from subagent import SubAgentTask, SubAgentManager


def _mock_agent_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
    """Mock agent.run that yields a simple response."""
    from agent import TurnDone
    # Simulate work
    for i in range(3):
        if cancel_check and cancel_check():
            return
        time.sleep(0.05)
    state.messages.append({"role": "assistant", "content": f"Done: {prompt}"})
    yield TurnDone(input_tokens=10, output_tokens=20)


@pytest.fixture
def manager(monkeypatch):
    import subagent
    monkeypatch.setattr(subagent, "_agent_run", _mock_agent_run)
    mgr = SubAgentManager(max_concurrent=2, max_depth=3)
    yield mgr
    mgr.shutdown()


def test_spawn_and_wait(manager):
    task = manager.spawn("say hello", {}, "system prompt")
    result = manager.wait(task.id, timeout=5)
    assert result.status == "completed"
    assert "Done: say hello" in result.result


def test_spawn_returns_immediately(manager):
    task = manager.spawn("do work", {}, "system prompt")
    assert task.status in ("pending", "running")
    assert task.id is not None


def test_list_tasks(manager):
    manager.spawn("task 1", {}, "sys")
    manager.spawn("task 2", {}, "sys")
    tasks = manager.list_tasks()
    assert len(tasks) == 2


def test_cancel(manager):
    # Use a slow mock
    import subagent
    original = subagent._agent_run

    def slow_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
        from agent import TurnDone
        for i in range(100):
            if cancel_check and cancel_check():
                return
            time.sleep(0.05)
        state.messages.append({"role": "assistant", "content": "should not reach"})
        yield TurnDone(input_tokens=0, output_tokens=0)

    subagent._agent_run = slow_run
    task = manager.spawn("slow task", {}, "sys")
    time.sleep(0.1)
    assert manager.cancel(task.id) is True
    manager.wait(task.id, timeout=5)
    assert task.status == "cancelled"
    subagent._agent_run = original


def test_depth_limit(manager):
    manager.max_depth = 2
    task = manager.spawn("deep task", {}, "sys", depth=2)
    assert task.status == "failed"
    assert "depth" in task.result.lower()


def test_get_result(manager):
    task = manager.spawn("hello", {}, "sys")
    manager.wait(task.id, timeout=5)
    result = manager.get_result(task.id)
    assert "Done: hello" in result


def test_get_result_unknown():
    mgr = SubAgentManager()
    assert mgr.get_result("nonexistent") is None
    mgr.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_subagent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'subagent'`

- [ ] **Step 3: Update agent.py to accept depth and cancel_check**

Modify the `run()` function signature in `agent.py` (line 52) from:

```python
def run(user_message: str, state: AgentState, config: dict, system_prompt: str) -> Generator:
```

to:

```python
def run(user_message: str, state: AgentState, config: dict, system_prompt: str,
        depth: int = 0, cancel_check=None) -> Generator:
```

Add a cancellation check inside the `while True` loop, at the very top of the loop body:

```python
        # --- Cancellation check ---
        if cancel_check and cancel_check():
            return
```

- [ ] **Step 4: Implement subagent.py**

```python
# subagent.py
"""
Sub-agent lifecycle management for cheetahclaws.

Sub-agents run in background threads via ThreadPoolExecutor.
Each has independent message history and fresh context.
Depth limiting prevents infinite recursion. Cancellation is cooperative.

Public API:
    SubAgentManager  — spawn, wait, cancel, list, get_result, shutdown
"""
from __future__ import annotations

import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


@dataclass
class SubAgentTask:
    """A single sub-agent task."""
    id: str
    prompt: str
    status: str = "pending"       # pending | running | completed | failed | cancelled
    result: Optional[str] = None
    depth: int = 0
    _cancel_flag: bool = False
    _future: Optional[Future] = field(default=None, repr=False)


def _agent_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
    """
    Reference to agent.run(). Replaced in __init__ or monkeypatched in tests.
    Importing agent at module level would create circular import,
    so we do a lazy import.
    """
    from agent import run
    return run(prompt, state, config, system_prompt,
               depth=depth, cancel_check=cancel_check)


def _extract_final_text(messages: list) -> str:
    """Extract the last assistant message text from a message list."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return content
    return "(no response)"


class SubAgentManager:
    """
    Manages sub-agent lifecycle.

    Args:
        max_concurrent: max parallel sub-agents, int
        max_depth: max recursion depth for sub-agent chains, int
    """

    def __init__(self, max_concurrent: int = 3, max_depth: int = 3):
        self.tasks: Dict[str, SubAgentTask] = {}
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent)

    def spawn(self, prompt: str, config: dict, system_prompt: str,
              depth: int = 0) -> SubAgentTask:
        """
        Spawn a new sub-agent in a background thread.

        Args:
            prompt: task description for the sub-agent
            config: global config dict
            system_prompt: system prompt for sub-agent
            depth: current recursion depth
        Returns:
            SubAgentTask (status may be "failed" if depth exceeded)
        """
        task_id = uuid.uuid4().hex[:8]

        if depth >= self.max_depth:
            task = SubAgentTask(
                id=task_id, prompt=prompt, status="failed", depth=depth,
                result="Error: max sub-agent depth reached. Complete this task directly.",
            )
            self.tasks[task_id] = task
            return task

        task = SubAgentTask(id=task_id, prompt=prompt, status="running", depth=depth)
        self.tasks[task_id] = task

        def _run():
            from agent import AgentState, TurnDone

            sub_state = AgentState()
            sub_config = {**config, "max_tokens": config.get("max_tokens", 8192)}

            sub_sys = (
                f"You are a sub-agent. Complete the following task:\n\n"
                f"{prompt}\n\n"
                f"---\n{system_prompt}"
            )

            try:
                for event in _agent_run(
                    prompt, sub_state, sub_config, sub_sys,
                    depth=depth + 1,
                    cancel_check=lambda: task._cancel_flag,
                ):
                    pass  # consume events

                if task._cancel_flag:
                    task.status = "cancelled"
                    task.result = "Task was cancelled."
                else:
                    task.result = _extract_final_text(sub_state.messages)
                    task.status = "completed"
            except Exception as e:
                task.result = f"Error: {e}"
                task.status = "failed"

        task._future = self._pool.submit(_run)
        return task

    def wait(self, task_id: str, timeout: float = None) -> Optional[SubAgentTask]:
        """
        Block until a task completes or timeout.

        Args:
            task_id: task identifier
            timeout: max wait seconds (None = wait forever)
        Returns:
            SubAgentTask or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if task._future is not None:
            try:
                task._future.result(timeout=timeout)
            except Exception:
                pass
        return task

    def get_result(self, task_id: str) -> Optional[str]:
        """Get the result string for a task. None if not found."""
        task = self.tasks.get(task_id)
        return task.result if task else None

    def list_tasks(self) -> List[SubAgentTask]:
        """Return all tasks."""
        return list(self.tasks.values())

    def cancel(self, task_id: str) -> bool:
        """
        Request cancellation of a running task.

        Args:
            task_id: task identifier
        Returns:
            True if cancellation was requested, False if task not found or not running
        """
        task = self.tasks.get(task_id)
        if task is None or task.status != "running":
            return False
        task._cancel_flag = True
        return True

    def shutdown(self) -> None:
        """Shutdown the thread pool. Cancel all running tasks first."""
        for task in self.tasks.values():
            if task.status == "running":
                task._cancel_flag = True
        self._pool.shutdown(wait=False)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_subagent.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Register Agent, CheckAgentResult, ListAgentTasks tools**

Add to the bottom of `tools.py`:

```python
# --- Sub-agent tools ------------------------------------------------------

from subagent import SubAgentManager

# Singleton manager (created lazily)
_agent_manager = None


def _get_agent_manager():
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = SubAgentManager()
    return _agent_manager


def _agent_tool(params, config):
    from context import build_system_prompt
    mgr = _get_agent_manager()
    prompt = params["prompt"]
    model = params.get("model")
    wait = params.get("wait", True)

    sub_config = {**config}
    if model:
        sub_config["model"] = model

    task = mgr.spawn(prompt, sub_config, build_system_prompt())

    if task.status == "failed":
        return task.result

    if wait:
        mgr.wait(task.id, timeout=300)
        return f"[Agent {task.id}] {task.status}: {task.result}"
    else:
        return f"[Agent {task.id}] Spawned in background. Use CheckAgentResult with task_id='{task.id}' to get the result."


def _check_agent_result(params, config):
    mgr = _get_agent_manager()
    task_id = params["task_id"]
    task = mgr.tasks.get(task_id)
    if task is None:
        return f"Error: no task with id '{task_id}'"
    if task.status == "running":
        return f"[Agent {task_id}] Still running..."
    return f"[Agent {task_id}] {task.status}: {task.result}"


def _list_agent_tasks(params, config):
    mgr = _get_agent_manager()
    tasks = mgr.list_tasks()
    if not tasks:
        return "No sub-agent tasks."
    lines = ["ID       | Status    | Prompt"]
    lines.append("-------- | --------- | ------")
    for t in tasks:
        preview = t.prompt[:50] + ("..." if len(t.prompt) > 50 else "")
        lines.append(f"{t.id} | {t.status:9s} | {preview}")
    return "\n".join(lines)


_AGENT_SCHEMA = {
    "name": "Agent",
    "description": "Launch a sub-agent to handle a task independently. "
                   "Use for complex subtasks that can be delegated.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Task description for the sub-agent"},
            "model": {"type": "string", "description": "Optional model override"},
            "wait": {"type": "boolean", "description": "True (default) = wait for result. False = return task_id for later polling."},
        },
        "required": ["prompt"],
    },
}

_CHECK_AGENT_RESULT_SCHEMA = {
    "name": "CheckAgentResult",
    "description": "Check the result of a background sub-agent task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task ID returned by Agent tool"},
        },
        "required": ["task_id"],
    },
}

_LIST_AGENT_TASKS_SCHEMA = {
    "name": "ListAgentTasks",
    "description": "List all sub-agent tasks and their status.",
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}

register_tool(ToolDef(name="Agent", schema=_AGENT_SCHEMA,
                       func=_agent_tool, read_only=False, concurrent_safe=False))
register_tool(ToolDef(name="CheckAgentResult", schema=_CHECK_AGENT_RESULT_SCHEMA,
                       func=_check_agent_result, read_only=True, concurrent_safe=True))
register_tool(ToolDef(name="ListAgentTasks", schema=_LIST_AGENT_TASKS_SCHEMA,
                       func=_list_agent_tasks, read_only=True, concurrent_safe=True))
```

- [ ] **Step 7: Commit**

```bash
cd D:/git/open-cc/cheetahclaws
git add subagent.py tests/test_subagent.py agent.py tools.py
git commit -m "feat: add threaded sub-agent system with Agent/CheckAgentResult/ListAgentTasks tools"
```

---

## Task 7: Skills System (`skills.py`)

**Files:**
- Create: `skills.py`
- Create: `tests/test_skills.py`
- Modify: `cheetahclaws.py` (skill dispatch in REPL)

- [ ] **Step 1: Write failing tests for skills**

```python
# tests/test_skills.py
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from skills import SkillDef, load_skills, find_skill, _parse_skill_file


@pytest.fixture
def skill_dir(tmp_path):
    """Create a temp skill directory with sample skills."""
    skill_file = tmp_path / "commit.md"
    skill_file.write_text(
        "---\n"
        "name: commit\n"
        "description: Create a git commit\n"
        "triggers: [\"/commit\", \"commit changes\"]\n"
        "tools: [Bash, Read]\n"
        "---\n\n"
        "# Commit Skill\n\n"
        "Analyze staged changes and create a commit message.\n"
    )

    review_file = tmp_path / "review.md"
    review_file.write_text(
        "---\n"
        "name: review\n"
        "description: Review a PR\n"
        "triggers: [\"/review\", \"/review-pr\"]\n"
        "tools: [Bash, Read, Grep]\n"
        "---\n\n"
        "# Review Skill\n\n"
        "Review the current PR for issues.\n"
    )
    return tmp_path


def test_parse_skill_file(skill_dir):
    skill = _parse_skill_file(skill_dir / "commit.md")
    assert skill.name == "commit"
    assert skill.description == "Create a git commit"
    assert "/commit" in skill.triggers
    assert "Bash" in skill.tools
    assert "Analyze staged" in skill.prompt


def test_load_skills(skill_dir, monkeypatch):
    import skills
    monkeypatch.setattr(skills, "SKILL_PATHS", [skill_dir])
    loaded = load_skills()
    assert len(loaded) == 2
    names = {s.name for s in loaded}
    assert "commit" in names
    assert "review" in names


def test_find_skill_by_slash_command(skill_dir, monkeypatch):
    import skills
    monkeypatch.setattr(skills, "SKILL_PATHS", [skill_dir])
    skill = find_skill("/commit")
    assert skill is not None
    assert skill.name == "commit"


def test_find_skill_with_args(skill_dir, monkeypatch):
    import skills
    monkeypatch.setattr(skills, "SKILL_PATHS", [skill_dir])
    skill = find_skill("/review")
    assert skill is not None
    assert skill.name == "review"


def test_find_skill_not_found(skill_dir, monkeypatch):
    import skills
    monkeypatch.setattr(skills, "SKILL_PATHS", [skill_dir])
    skill = find_skill("/nonexistent")
    assert skill is None


def test_load_skills_empty_dir(tmp_path, monkeypatch):
    import skills
    monkeypatch.setattr(skills, "SKILL_PATHS", [tmp_path])
    loaded = load_skills()
    assert loaded == []


def test_load_skills_nonexistent_dir(monkeypatch):
    import skills
    monkeypatch.setattr(skills, "SKILL_PATHS", ["/nonexistent/path"])
    loaded = load_skills()
    assert loaded == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_skills.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'skills'`

- [ ] **Step 3: Implement skills.py**

```python
# skills.py
"""
Skill loading and execution for cheetahclaws.

Skills are markdown files with YAML frontmatter that define reusable
prompt templates. They are injected into the agent loop as user messages.

Public API:
    load_skills()                          — scan skill directories
    find_skill(query)                      — match by /command or keyword
    execute_skill(skill, args, state, ...) — run skill via agent loop
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional


SKILL_PATHS = [
    Path.cwd() / ".cheetahclaws" / "skills",
    Path.home() / ".cheetahclaws" / "skills",
]


@dataclass
class SkillDef:
    """Definition of a loaded skill."""
    name: str
    description: str
    triggers: List[str]
    tools: List[str]
    prompt: str
    file_path: str


def _parse_list_field(value: str) -> List[str]:
    """Parse a YAML-like list field: [a, b, c] or \"a, b, c\"."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    # Handle quoted items
    items = []
    for item in value.split(","):
        item = item.strip().strip('"').strip("'")
        if item:
            items.append(item)
    return items


def _parse_skill_file(path: Path) -> Optional[SkillDef]:
    """
    Parse a single skill markdown file.

    Args:
        path: path to .md skill file
    Returns:
        SkillDef or None if parsing fails
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    if not text.startswith("---"):
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None

    # Parse frontmatter
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip()

    name = meta.get("name", "")
    if not name:
        return None

    return SkillDef(
        name=name,
        description=meta.get("description", ""),
        triggers=_parse_list_field(meta.get("triggers", "")),
        tools=_parse_list_field(meta.get("tools", "")),
        prompt=parts[2].strip(),
        file_path=str(path),
    )


def load_skills() -> List[SkillDef]:
    """
    Scan all skill directories and parse skill files.

    Project-level skills override user-level skills with the same name.

    Returns:
        list of SkillDef
    """
    skills_by_name = {}
    # Reverse order so project-level (first in list) overrides user-level
    for skill_dir in reversed(SKILL_PATHS):
        if not Path(skill_dir).exists():
            continue
        for f in sorted(Path(skill_dir).glob("*.md")):
            skill = _parse_skill_file(f)
            if skill:
                skills_by_name[skill.name] = skill
    return list(skills_by_name.values())


def find_skill(query: str) -> Optional[SkillDef]:
    """
    Find a skill matching a query string.

    Matches against trigger strings (e.g., "/commit").
    The query is matched with and without leading arguments.

    Args:
        query: user input string (e.g., "/commit -m fix bug")
    Returns:
        matching SkillDef or None
    """
    # Extract the command part (first word)
    cmd = query.strip().split()[0] if query.strip() else ""

    for skill in load_skills():
        for trigger in skill.triggers:
            if cmd == trigger or query.strip() == trigger:
                return skill
    return None


def execute_skill(skill: SkillDef, args: str, state, config: dict,
                  system_prompt: str) -> Generator:
    """
    Execute a skill by injecting its prompt into the agent loop.

    Args:
        skill: the skill to execute
        args: additional arguments from the user
        state: AgentState
        config: global config dict
        system_prompt: current system prompt
    Yields:
        agent events (TextChunk, ToolStart, etc.)
    """
    from agent import run

    prompt = f"[Skill: {skill.name}]\n\n{skill.prompt}"
    if args:
        prompt += f"\n\nUser context: {args}"

    for event in run(prompt, state, config, system_prompt):
        yield event
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/test_skills.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Integrate skills into cheetahclaws.py REPL**

In `cheetahclaws.py`, add import near the top:

```python
from skills import find_skill, execute_skill, load_skills
```

Modify the `handle_slash` function (line 373) to fall through to skill lookup. At the end of the function, after checking all built-in commands, before the "unknown command" error:

```python
    # Try skills
    skill = find_skill(line)
    if skill:
        # Extract args: everything after the trigger
        cmd_parts = line.strip().split(maxsplit=1)
        args = cmd_parts[1] if len(cmd_parts) > 1 else ""
        return skill, args  # return skill for REPL to execute
```

This requires adjusting `handle_slash` return type. Currently it returns `bool`. Change it to return `tuple | bool`:
- `True` / `False` for built-in commands
- `(SkillDef, args_str)` for skill matches

In the REPL loop (around line 440), where `handle_slash` is called, add skill execution:

```python
            result = handle_slash(user_input, state, config)
            if isinstance(result, tuple):
                # It's a skill
                skill, args = result
                system_prompt = build_system_prompt()
                info(f"Running skill: {skill.name}")
                for event in execute_skill(skill, args, state, config, system_prompt):
                    # Reuse existing event handling (same as query events)
                    if isinstance(event, TextChunk):
                        stream_text(event.text)
                    elif isinstance(event, ThinkingChunk):
                        stream_thinking(event.text, config.get("verbose"))
                    elif isinstance(event, ToolStart):
                        print_tool_start(event.name, event.inputs, config.get("verbose"))
                    elif isinstance(event, ToolEnd):
                        print_tool_end(event.name, event.result, config.get("verbose"))
                    elif isinstance(event, TurnDone):
                        flush_response()
                continue
            elif result:
                continue
```

Also add a `/skills` slash command to list available skills:

```python
def cmd_skills(_args, _state, _config):
    skills = load_skills()
    if not skills:
        info("No skills found. Place .md files in ~/.cheetahclaws/skills/")
        return True
    for s in skills:
        triggers = ", ".join(s.triggers)
        info(f"  {s.name:15s} {triggers:25s} {s.description}")
    return True
```

Add `"skills": cmd_skills` to the `COMMANDS` dict.

- [ ] **Step 6: Commit**

```bash
cd D:/git/open-cc/cheetahclaws
git add skills.py tests/test_skills.py cheetahclaws.py
git commit -m "feat: add skills system with markdown definitions and /skills command"
```

---

## Task 8: Config Updates + Remaining Slash Commands

**Files:**
- Modify: `config.py` (new config keys)
- Modify: `cheetahclaws.py` (add /memory, /agents commands)

- [ ] **Step 1: Add new config defaults**

In `config.py`, add to the `DEFAULTS` dict (line 11):

```python
    "max_tool_output": 32000,
    "max_agent_depth": 3,
    "max_concurrent_agents": 3,
```

- [ ] **Step 2: Add /memory slash command**

In `cheetahclaws.py`, add:

```python
def cmd_memory(args, _state, _config):
    from memory import load_index, search_memory
    if args:
        results = search_memory(args)
        if not results:
            info(f"No memories matching '{args}'")
            return True
        for m in results:
            info(f"  [{m.type}] {m.name}: {m.description}")
            info(f"    {m.content[:100]}{'...' if len(m.content) > 100 else ''}")
        return True
    entries = load_index()
    if not entries:
        info("No memories stored. The model can save memories via MemorySave tool.")
        return True
    info(f"  {len(entries)} memories:")
    for m in entries:
        info(f"  [{m.type:9s}] {m.name}: {m.description}")
    return True
```

Add `"memory": cmd_memory` to the `COMMANDS` dict.

- [ ] **Step 3: Add /agents slash command**

In `cheetahclaws.py`, add:

```python
def cmd_agents(_args, _state, _config):
    try:
        from tools import _get_agent_manager
        mgr = _get_agent_manager()
        tasks = mgr.list_tasks()
        if not tasks:
            info("No sub-agent tasks.")
            return True
        info(f"  {len(tasks)} sub-agent tasks:")
        for t in tasks:
            preview = t.prompt[:40] + ("..." if len(t.prompt) > 40 else "")
            info(f"  {t.id} [{t.status:9s}] {preview}")
    except Exception:
        info("Sub-agent system not initialized.")
    return True
```

Add `"agents": cmd_agents` to the `COMMANDS` dict.

- [ ] **Step 4: Update /help to include new commands**

In `cmd_help`, add the new commands to the help text:

```python
    info("  /memory [query]   Show/search persistent memories")
    info("  /skills           List available skills")
    info("  /agents           Show sub-agent tasks")
```

- [ ] **Step 5: Verify all tests pass**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/ -v`
Expected: All tests across all test files PASS

- [ ] **Step 6: Commit**

```bash
cd D:/git/open-cc/cheetahclaws
git add config.py cheetahclaws.py
git commit -m "feat: add /memory, /skills, /agents commands and new config defaults"
```

---

## Task 9: Integration Smoke Test

**Files:** None (testing only)

- [ ] **Step 1: Verify tool registry loads all tools**

Run: `cd D:/git/open-cc/cheetahclaws && python -c "
from tool_registry import get_all_tools
tools = get_all_tools()
print(f'{len(tools)} tools registered:')
for t in tools:
    print(f'  {t.name:20s} read_only={t.read_only}  concurrent_safe={t.concurrent_safe}')
"`
Expected: 13 tools (8 built-in + MemorySave + MemoryDelete + Agent + CheckAgentResult + ListAgentTasks)

- [ ] **Step 2: Verify compaction module loads**

Run: `cd D:/git/open-cc/cheetahclaws && python -c "
from compaction import estimate_tokens, get_context_limit
msgs = [{'role': 'user', 'content': 'hello ' * 1000}]
print(f'Tokens: {estimate_tokens(msgs)}')
print(f'GPT-4o limit: {get_context_limit(\"gpt-4o\")}')
print(f'Gemini limit: {get_context_limit(\"gemini-2.0-flash\")}')
"`
Expected: Token count ~1700, GPT-4o 128000, Gemini 1000000

- [ ] **Step 3: Verify memory roundtrip**

Run: `cd D:/git/open-cc/cheetahclaws && python -c "
import tempfile
from pathlib import Path
import memory
memory.MEMORY_DIR = Path(tempfile.mkdtemp())
memory.INDEX_FILE = memory.MEMORY_DIR / 'MEMORY.md'
from memory import MemoryEntry, save_memory, load_index, get_memory_context
save_memory(MemoryEntry(name='test', description='integration test', type='user', content='works!'))
entries = load_index()
print(f'Entries: {len(entries)}, name={entries[0].name}')
print(f'Context: {get_memory_context()}')
"`
Expected: 1 entry, context contains "test"

- [ ] **Step 4: Verify skills loading**

Run: `cd D:/git/open-cc/cheetahclaws && python -c "
from skills import load_skills
skills = load_skills()
print(f'{len(skills)} skills loaded')
for s in skills:
    print(f'  {s.name}: {s.triggers}')
"`
Expected: 0 skills (none created yet), no errors

- [ ] **Step 5: Run full test suite**

Run: `cd D:/git/open-cc/cheetahclaws && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 6: Final commit with all tests passing**

```bash
cd D:/git/open-cc/cheetahclaws
git add -A
git status
# Only commit if there are uncommitted changes
git commit -m "chore: integration verification complete"
```
