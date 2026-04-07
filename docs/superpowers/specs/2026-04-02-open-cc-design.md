# Open-CC: CheetahClaws Enhancement Design

**Date:** 2026-04-02
**Status:** Approved
**Target:** GPT-5.4, Gemini 3/3.1 Pro (Claude not in scope)
**Code budget:** ~10K lines total (currently ~2.2K)
**Constraint:** PR-friendly, mergeable back to cheetahclaws upstream

---

## 1. Overview

Evolve cheetahclaws from a minimal ~2.2K-line reference implementation into a capable AI coding CLI, approaching Claude Code's core functionality while staying lean. Five enhancement areas:

1. **Context Window Management** (`compaction.py`)
2. **Tool System Enhancement** (`tool_registry.py` + `tools.py` refactor)
3. **Sub-Agent** (`subagent.py`)
4. **Memory System** (`memory.py`)
5. **Skills System** (`skills.py`)

### Strategy

**Approach A: Layered Enhancement** -- add new modules alongside existing files, minimize changes to existing code. When agent.py grows too complex, refactor into Approach B (package structure under `ncc/`).

### Design Principles

- Modules communicate via function parameters / dataclasses, no globals
- Each new module exposes 2-3 public functions, internals self-contained
- New logic in agent.py grouped by clear `# --- section ---` comments
- All code in English (comments, docstrings, commit messages)

---

## 2. File Structure

```
cheetahclaws/
├── cheetahclaws.py      # REPL -- add /memory, /skill slash commands
├── agent.py            # Agent loop -- add compaction call + sub-agent dispatch
├── providers.py        # No changes (already solid)
├── tools.py            # Refactor: register built-in tools via registry
├── context.py          # Extend: inject memory context
├── config.py           # Add new config keys
│
├── compaction.py       # NEW: Context window management
├── subagent.py         # NEW: Sub-agent lifecycle
├── memory.py           # NEW: File-based memory system
├── skills.py           # NEW: Skill loading and execution
└── tool_registry.py    # NEW: Tool plugin registry
```

### Module Dependency Graph (unidirectional)

```
cheetahclaws.py
    ├-> agent.py
    │    ├-> providers.py
    │    ├-> tool_registry.py -> tools.py (built-in implementations)
    │    ├-> compaction.py -> providers.py (for summary model call)
    │    └-> subagent.py (calls agent.py:run recursively)
    ├-> context.py -> memory.py
    ├-> skills.py -> tool_registry.py
    └-> config.py
```

---

## 3. Context Window Management (`compaction.py`)

Two-layer compression, inspired by Claude Code's three-layer strategy (Layer 3 contextCollapse is experimental, deferred).

### 3.1 Layer 1: Auto-Compact (model-driven summary)

Triggered when estimated token count exceeds 70% of model's context limit.

```python
def compact_messages(messages: list[dict], config: dict) -> list[dict]:
    """
    Split messages into [old | recent].
    Summarize old via model call.
    Return [summary_msg, ack_msg, *recent].
    """
    split_point = find_split_point(messages, keep_ratio=0.3)
    old = messages[:split_point]
    recent = messages[split_point:]
    summary = call_model_for_summary(old, config)
    return [
        {"role": "user", "content": f"[Conversation summary]\n{summary}"},
        {"role": "assistant", "content": "Understood, I have the context."},
        *recent
    ]
```

### 3.2 Layer 2: Tool-Result Snipping (rule-based)

Truncate old tool outputs without model call. Fast and cheap.

```python
def snip_old_tool_results(messages: list[dict], max_chars: int = 2000) -> list[dict]:
    """
    For tool results older than N turns, truncate to max_chars.
    Preserve first/last lines, add [snipped N chars] marker.
    """
```

### 3.3 Token Estimation

```python
def estimate_tokens(messages: list[dict]) -> int:
    """Use tiktoken for GPT models, chars/3.5 fallback."""

def get_context_limit(model: str) -> int:
    """Return context window size from provider registry."""
```

### 3.4 Integration Point

```python
# In agent.py run() loop, before each API call:
def _maybe_compact(state: AgentState, config: dict) -> bool:
    token_count = estimate_tokens(state.messages)
    threshold = get_context_limit(config["model"]) * 0.7
    if token_count > threshold:
        state.messages = compact_messages(state.messages, config)
        return True
    return False
```

### 3.5 Public API

```python
maybe_compact(state: AgentState, config: dict) -> bool
estimate_tokens(messages: list[dict]) -> int
get_context_limit(model: str) -> int
```

---

## 4. Tool System Enhancement (`tool_registry.py` + `tools.py`)

### 4.1 Tool Registry

```python
@dataclass
class ToolDef:
    name: str
    schema: dict            # JSON schema for parameters
    func: Callable          # (params: dict, config: dict) -> str
    read_only: bool         # True = auto-approve in 'auto' mode
    concurrent_safe: bool   # True = safe for parallel sub-agent use

_TOOLS: dict[str, ToolDef] = {}

def register_tool(tool_def: ToolDef) -> None
def get_tool(name: str) -> ToolDef | None
def get_all_tools() -> list[ToolDef]
def get_tool_schemas() -> list[dict]
def execute_tool(name: str, params: dict, config: dict) -> str
```

### 4.2 Tool Output Truncation

Prevent oversized tool outputs (e.g., `cat` large file, `ls -R`) from blowing up context
before compaction even gets a chance to run. Applied at the `execute_tool` boundary:

```python
MAX_TOOL_OUTPUT = 32_000  # ~8K tokens, configurable per tool

def execute_tool(name, params, config):
    tool = get_tool(name)
    result = tool.func(params, config)

    # Immediate truncation at source
    if len(result) > MAX_TOOL_OUTPUT:
        head = result[:MAX_TOOL_OUTPUT // 2]
        tail = result[-MAX_TOOL_OUTPUT // 4:]
        snipped = len(result) - len(head) - len(tail)
        result = f"{head}\n\n[... {snipped} chars truncated ...]\n\n{tail}"

    return result
```

Additionally, `Bash` tool caps `subprocess` stdout reads to prevent unbounded
output (e.g., `cat /dev/urandom`).

This creates a two-layer defense:
- **Layer 0 (here):** hard truncation at tool execution time — prevents oversized messages
- **Layer 2 (compaction.py snip):** soft truncation of old tool results — reclaims context space

### 4.3 Built-in Tools Refactor

Existing tools.py implementations unchanged. Wrap each with `register_tool()` at module load:

```python
register_tool(ToolDef(
    name="Read", schema=READ_SCHEMA, func=_read_file,
    read_only=True, concurrent_safe=True
))
```

### 4.3 Permission Logic (unified)

```python
# agent.py
def _check_permission(tool_name, params, config):
    tool = get_tool(tool_name)
    if config["permission_mode"] == "accept-all":
        return True
    if tool.read_only:
        return True
    if tool_name == "Bash" and _is_safe_command(params["command"]):
        return True
    return None  # ask user
```

---

## 5. Sub-Agent (`subagent.py`)

### 5.1 Data Model

```python
@dataclass
class SubAgentTask:
    id: str
    prompt: str
    status: str              # "pending" | "running" | "completed" | "failed" | "cancelled"
    messages: list[dict]     # independent message history
    result: str | None
    model: str | None        # optional model override
    depth: int = 0           # recursion depth counter
    _cancel_flag: bool = False
    _future: Future | None = None

@dataclass
class SubAgentManager:
    tasks: dict[str, SubAgentTask] = field(default_factory=dict)
    max_concurrent: int = 3
    max_depth: int = 3
    _pool: ThreadPoolExecutor = field(default_factory=
        lambda: ThreadPoolExecutor(max_workers=3))

    def spawn(self, prompt, config, system_prompt, depth=0) -> SubAgentTask
    def get_result(self, task_id) -> str | None
    def list_tasks(self) -> list[SubAgentTask]
    def cancel(self, task_id) -> bool
    def wait(self, task_id, timeout=None) -> SubAgentTask
```

### 5.2 Execution Model — Threading from Day 1

Sub-agents run in background threads via `ThreadPoolExecutor`. This enables:
- Non-blocking spawn (main agent continues or waits by choice)
- Cancellation via cooperative flag
- Concurrent sub-agents (up to `max_concurrent`)

```python
def spawn(self, prompt, config, system_prompt, depth=0):
    if depth >= self.max_depth:
        return SubAgentTask(status="failed",
            result="Error: max sub-agent depth reached.")

    task = SubAgentTask(id=uuid4().hex[:8], prompt=prompt,
                        status="running", depth=depth, ...)

    def _run():
        sub_state = AgentState()
        try:
            for event in agent.run(
                prompt, sub_state, config, system_prompt,
                depth=depth + 1,
                cancel_check=lambda: task._cancel_flag
            ):
                if isinstance(event, TurnDone):
                    task.result = extract_final_text(sub_state.messages)
            task.status = "completed"
        except Exception as e:
            task.result = f"Error: {e}"
            task.status = "failed"

    task._future = self._pool.submit(_run)
    self.tasks[task.id] = task
    return task
```

### 5.3 Cooperative Cancellation

Python threads cannot be killed safely. Instead, `agent.run()` checks a
`cancel_check` callable each loop iteration:

```python
# agent.py run() — new parameter
def run(user_message, state, config, system_prompt,
        depth=0, cancel_check=None):
    ...
    while True:
        if cancel_check and cancel_check():
            return  # clean exit
        for event in stream(...):
            yield event
        ...
```

### 5.4 Depth Limiting (No Tool Removal)

Sub-agents CAN call Agent tool (enabling A -> B -> C chains). Depth is
passed through, and the Agent tool returns an error at `max_depth`:

```python
def _agent_tool_func(params, config, depth=0):
    if depth >= manager.max_depth:
        return ("Error: max sub-agent depth reached. "
                "Complete this task directly without spawning sub-agents.")
    return manager.spawn(params["prompt"], config, system_prompt, depth)
```

The model sees the error and adapts — no silent capability removal.

### 5.5 Context Strategy

Sub-agent gets **fresh context** (no parent message history):

```python
sub_system_prompt = f"""You are a sub-agent. Your task:
{prompt}

Working directory: {cwd}
{memory_context}
"""
```

### 5.6 Tool Registration — 3 Tools

The sub-agent system registers three tools:

**Agent** — spawn a sub-agent:

```python
AGENT_SCHEMA = {
    "name": "Agent",
    "description": "Launch a sub-agent to handle a task independently.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Task description"},
            "model": {"type": "string", "description": "Optional model override"},
            "wait": {"type": "boolean", "default": True,
                     "description": "True = block until done (default). "
                                    "False = return task_id immediately."}
        },
        "required": ["prompt"]
    }
}
```

- `wait=True` (default): spawn + block + return result. Feels synchronous to model.
- `wait=False`: spawn + return task_id immediately. Model must use CheckAgentResult later.

**CheckAgentResult** — poll a background sub-agent:

```python
CHECK_AGENT_RESULT_SCHEMA = {
    "name": "CheckAgentResult",
    "description": "Check the result of a background sub-agent task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task ID from Agent tool"}
        },
        "required": ["task_id"]
    }
}
```

Returns: status + result (if completed), or status + "still running".

**ListAgentTasks** — overview of all sub-agents:

```python
LIST_AGENT_TASKS_SCHEMA = {
    "name": "ListAgentTasks",
    "description": "List all sub-agent tasks and their status.",
    "input_schema": {"type": "object", "properties": {}}
}
```

Returns a table of `[id, status, prompt_preview]` for all tasks.

---

## 6. Memory System (`memory.py`)

### 6.1 Storage

```
~/.cheetahclaws/memory/
├── MEMORY.md              # Index file (max 200 lines)
├── user_role.md           # Individual memory files
├── feedback_testing.md
└── ...
```

Memory file format:

```markdown
---
name: user role
description: user is a data scientist focused on logging
type: user
created: 2026-04-02
---

User is a data scientist, currently investigating observability/logging.
```

### 6.2 Public API

```python
@dataclass
class MemoryEntry:
    name: str
    description: str
    type: str              # "user" | "feedback" | "project" | "reference"
    content: str
    file_path: str
    created: str

def load_index() -> list[MemoryEntry]
def save_memory(entry: MemoryEntry) -> None
def delete_memory(name: str) -> None
def search_memory(query: str) -> list[MemoryEntry]
def get_memory_context() -> str   # for system prompt injection
```

### 6.3 Tool Registration

Two tools for model-driven memory management:

- **MemorySave**: `{name, type, description, content}` -> write file + update index
- **MemoryDelete**: `{name}` -> remove file + update index

### 6.4 Context Integration

`context.py:build_system_prompt()` appends `memory.get_memory_context()` (the MEMORY.md index). Model uses Read tool to access full memory file content when needed.

---

## 7. Skills System (`skills.py`)

### 7.1 Skill Definition

Markdown files with frontmatter:

```
~/.cheetahclaws/skills/commit.md
```

```markdown
---
name: commit
description: Create a git commit with conventional format
triggers: ["/commit", "commit changes"]
tools: [Bash, Read]
---

# Commit Skill

Analyze staged changes and create a well-formatted commit message.
...
```

### 7.2 Search Path

```python
SKILL_PATHS = [
    Path.cwd() / ".cheetahclaws" / "skills",    # project-level (priority)
    Path.home() / ".cheetahclaws" / "skills",    # user-level
]
```

### 7.3 Public API

```python
@dataclass
class SkillDef:
    name: str
    description: str
    triggers: list[str]
    tools: list[str]
    prompt: str
    file_path: str

def load_skills() -> list[SkillDef]
def find_skill(query: str) -> SkillDef | None
def execute_skill(skill, args, state, config) -> Generator
```

### 7.4 Execution Model

Skills are just prompts injected into the normal agent loop:

```python
def execute_skill(skill, args, state, config):
    prompt = f"[Skill: {skill.name}]\n\n{skill.prompt}"
    if args:
        prompt += f"\n\nUser context: {args}"
    system_prompt = build_system_prompt(config)
    for event in agent.run(prompt, state, config, system_prompt):
        yield event
```

### 7.5 REPL Integration

In `cheetahclaws.py`, unmatched `/` commands fall through to skill lookup:

```python
if user_input.startswith("/"):
    # Try built-in slash commands first
    # If no match -> find_skill(user_input)
    # If skill found -> execute_skill(...)
```

---

## 8. Diff View for File Modifications

Core UX improvement: show git-style red/green diff when Edit or Write modifies an existing file.

### 8.1 Diff Generation (in tools.py)

Edit and Write tool implementations capture before/after content and generate unified diff:

```python
import difflib

def generate_unified_diff(old, new, filename, context_lines=3):
    """
    Args:
        old: original file content, str
        new: modified file content, str
        filename: display name, str
        context_lines: lines of context around changes, int
    Returns:
        unified diff string
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        n=context_lines
    )
    return "".join(diff)
```

Tool return values change:
- **Edit**: `"Changes applied to {filename}:\n\n{diff}"`
- **Write** (existing file): `"File updated:\n\n{diff}"`
- **Write** (new file): `"New file created: {filename} ({n} lines)"` (no diff)

### 8.2 REPL Rendering (in cheetahclaws.py)

Detect diff blocks in tool output and render with ANSI colors:

```python
def render_diff(diff_text):
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            print(f"\033[1m{line}\033[0m")        # bold
        elif line.startswith("+"):
            print(f"\033[32m{line}\033[0m")        # green
        elif line.startswith("-"):
            print(f"\033[31m{line}\033[0m")        # red
        elif line.startswith("@@"):
            print(f"\033[36m{line}\033[0m")        # cyan
        else:
            print(line)
```

### 8.3 Diff Truncation

For large diffs (e.g., Write replaces entire file), cap the diff display:

```python
MAX_DIFF_LINES = 80

def maybe_truncate_diff(diff_text):
    lines = diff_text.splitlines()
    if len(lines) > MAX_DIFF_LINES:
        shown = lines[:MAX_DIFF_LINES]
        remaining = len(lines) - MAX_DIFF_LINES
        return "\n".join(shown) + f"\n\n[... {remaining} more lines ...]"
    return diff_text
```

Note: truncation applies to the **display** in REPL only. The full diff is still
returned to the model so it can verify the change.

---

## 9. Implementation Order

Each step is an independent PR:

| Phase | Module | Depends On | Estimated Lines |
|-------|--------|-----------|-----------------|
| 1 | `tool_registry.py` + `tools.py` refactor | None | ~600 |
| 2 | Diff view in `tools.py` + `cheetahclaws.py` | Phase 1 | ~100 |
| 3 | `compaction.py` + agent.py integration | Phase 1 | ~300 |
| 4 | `memory.py` + context.py integration | Phase 1 | ~200 |
| 5 | `subagent.py` + agent.py integration (threading) | Phase 1 | ~350 |
| 6 | `skills.py` + cheetahclaws.py integration | Phase 1, 4 | ~200 |
| 7 | Slash commands + config updates | All above | ~300 |

**Total new code: ~2050 lines. Grand total: ~4.2K lines.**

---

## 10. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Compression layers | 2 (autoCompact + snip) | Layer 3 is experimental in Claude Code |
| Tool output truncation | Hard cap at execute_tool boundary | Prevents oversized outputs before compaction runs |
| Sub-agent execution | Threading from day 1 | Sync blocks main agent, can't cancel, can't parallelize |
| Sub-agent depth | Depth counter (max 3), no tool removal | Model sees error and adapts; sub-sub-agents allowed |
| Sub-agent tools | Agent + CheckAgentResult + ListAgentTasks | Model needs feedback loop for async tasks |
| Diff view | difflib unified diff + ANSI colors | Core UX, zero dependencies |
| Memory search | Keyword match, no embeddings | Keep simple, model judges relevance |
| Skills format | Markdown + frontmatter | Human-readable, git-friendly, no Python needed |
| Tool registry | Global dict + register function | Simple, extensible, easy to migrate to package |
| Target models | GPT-5.4, Gemini 3/3.1 Pro | User's primary use case |
| No Claude support | Intentional | Official Claude Code exists |

---

## 11. Future Considerations (Not in Scope)

- MCP protocol support
- Remote skill marketplace
- Voice mode
- Bridge to desktop apps
- contextCollapse (Layer 3 compression)
