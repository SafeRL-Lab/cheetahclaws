# CheetahClaws — Update Notes

This document describes three major feature additions to cheetahclaws:
**Multi-Agent**, **Memory**, and **Skill**. Each feature is organized as a
self-contained Python package, follows the same architectural pattern, and
includes a backward-compatibility shim so existing code continues to work.

---

## Architecture Overview

All three packages follow the same pattern:

```
feature/
  __init__.py   — public re-exports
  <core>.py     — data model, loading, business logic
  tools.py      — registers tools into the central tool_registry
  ...
feature.py      — backward-compat shim (re-exports from feature/)
```

The **tool registry** (`tool_registry.py`) is the central hub. Each feature's
`tools.py` calls `register_tool(ToolDef(...))` at import time. The top-level
`tools.py` imports all three feature tool modules, triggering auto-registration.

The **agent loop** (`agent.py`) injects `_depth` and `_system_prompt` into the
`config` dict on every call, so tool functions can read them via `config.get(...)`.

---

## 1. Multi-Agent (`multi_agent/`)

### What it does

Allows Claude to spawn sub-agents — nested agent loops that run concurrently
in background threads. Sub-agents can share the parent's context or run in an
isolated git worktree. The user can send follow-up messages to named background
agents and retrieve their results.

### Package structure

```
multi_agent/
  __init__.py       — re-exports AgentDefinition, SubAgentTask, SubAgentManager, etc.
  subagent.py       — core: AgentDefinition, SubAgentTask, SubAgentManager, worktree helpers
  tools.py          — registers: Agent, SendMessage, CheckAgentResult, ListAgentTasks, ListAgentTypes
subagent.py         — backward-compat shim
```

### Key classes and functions

**`AgentDefinition`** (`multi_agent/subagent.py`)
```python
@dataclass
class AgentDefinition:
    name: str
    description: str
    system_prompt: str   # prepended to base prompt for this agent type
    model: str           # "" = inherit from parent
    tools: list          # [] = all tools
    source: str          # "built-in" | "user" | "project"
```

**Built-in agent types**: `general-purpose`, `coder`, `reviewer`, `researcher`, `tester`

**Custom agent definitions** — place a `.md` file with YAML frontmatter in:
- `~/.cheetahclaws/agents/<name>.md` (user-level)
- `.cheetahclaws/agents/<name>.md` (project-level, takes priority)

Frontmatter format:
```markdown
---
name: my-agent
description: What this agent does
model: claude-opus-4-6
tools: [Read, Glob, Grep]
---
Extra system prompt instructions for this agent.
```

**`SubAgentManager`** (`multi_agent/subagent.py`)
- `spawn(prompt, config, agent_def, isolation, name, wait)` — runs agent in thread pool
- `send_message(task_id_or_name, message)` — enqueues message to a running background agent
- `get_result(task_id)` — returns final text or status
- `list_tasks()` — returns all SubAgentTask objects

**Git worktree isolation**:
When `isolation="worktree"` is passed to `Agent`, a temporary git worktree is
created on a fresh branch. The sub-agent works in isolation; if it makes no
changes the worktree is cleaned up automatically.

### Tools registered

| Tool | Description |
|------|-------------|
| `Agent` | Spawn a sub-agent (sync or background with `wait=false`) |
| `SendMessage` | Send a follow-up message to a named background agent |
| `CheckAgentResult` | Poll status / retrieve result of a background agent |
| `ListAgentTasks` | List all active and finished sub-agent tasks |
| `ListAgentTypes` | List all available agent type definitions |

### Agent tool parameters

```python
Agent(
    prompt="...",           # required — task description
    subagent_type="coder",  # optional — use a specialized agent
    isolation="worktree",   # optional — isolated git branch
    name="my-agent",        # optional — name for SendMessage later
    wait=False,             # optional — run in background
    model="...",            # optional — model override
)
```

### How it was wired in

1. `multi_agent/subagent.py` uses **absolute imports** (`import agent as _agent_mod`)
   because the project root is in `sys.path` when running from that directory.
2. `agent.py` was updated to inject `_system_prompt` into `config`:
   ```python
   config = {**config, "_depth": depth, "_system_prompt": system_prompt}
   ```
3. `tools.py` (top-level) was updated to pass `config` through to the registry:
   ```python
   return _registry_execute(name, inputs, cfg)
   ```
   and at the bottom:
   ```python
   import multi_agent.tools as _multiagent_tools
   ```
4. `context.py` system prompt template lists Agent, SendMessage, etc. under
   `## Multi-Agent`.
5. `cheetahclaws.py` `/agents` command calls `get_agent_manager().list_tasks()`
   and prints status/worktree info. A `_print_background_notifications()` function
   checks for newly completed background agents before each user prompt.

### Files changed

| File | Change |
|------|--------|
| `multi_agent/__init__.py` | Created (re-exports) |
| `multi_agent/subagent.py` | Created (moved + enhanced from `subagent.py`) |
| `multi_agent/tools.py` | Created (tool registrations) |
| `subagent.py` | Converted to backward-compat shim |
| `agent.py` | Inject `_system_prompt` into config |
| `tools.py` | Pass config to registry; import `multi_agent.tools` |
| `context.py` | Add Multi-Agent section to system prompt |
| `cheetahclaws.py` | `/agents` command; background notification; `_tool_desc()` |
| `tests/test_subagent.py` | Update imports to `multi_agent.subagent` |

---

## 2. Memory (`memory/`)

### What it does

Provides persistent, file-based memory across sessions. Memories are stored as
markdown files with YAML frontmatter. There are two scopes — **user** (global,
`~/.cheetahclaws/memory/`) and **project** (per-repo, `.cheetahclaws/memory/`).
A `MEMORY.md` index is auto-rebuilt after every save/delete and injected into
the system prompt so Claude knows what memories exist.

### Package structure

```
memory/
  __init__.py   — re-exports all public symbols
  types.py      — MEMORY_TYPES, type descriptions, format guidance
  store.py      — MemoryEntry, save/load/delete/search, index rebuilding
  scan.py       — MemoryHeader, scan_memory_dir, age/freshness helpers
  context.py    — get_memory_context(), find_relevant_memories(), truncation
  tools.py      — registers: MemorySave, MemoryDelete, MemorySearch, MemoryList
memory.py       — backward-compat shim
```

### Memory types

Defined in `memory/types.py`, mirrors the four types from Claude Code:

| Type | Purpose |
|------|---------|
| `user` | User's role, goals, preferences |
| `feedback` | Corrections and confirmed approaches |
| `project` | Ongoing work, decisions, deadlines |
| `reference` | Pointers to external resources |

### Storage layout

```
~/.cheetahclaws/memory/
  MEMORY.md          ← auto-generated index (<=200 lines, <=25 KB)
  my_note.md
  feedback_testing.md
  ...

.cheetahclaws/memory/   ← project-local (relative to cwd)
  MEMORY.md
  ...
```

Each memory file format:
```markdown
---
name: My Note
description: one-line description for relevance decisions
type: user
created: 2026-04-02
---

Memory content goes here.
**Why:** ...
**How to apply:** ...
```

### Key API

**`memory/store.py`**
```python
save_memory(entry: MemoryEntry, scope="user")   # save or update (same name = update)
delete_memory(name: str, scope="user")           # remove entry + rebuild index
load_entries(scope="user") -> list[MemoryEntry]  # load all entries for scope
load_index(scope="all") -> list[MemoryEntry]     # "all" merges user + project
search_memory(query: str, scope="all") -> list   # keyword search across content+name
get_index_content(scope="all") -> str            # raw MEMORY.md text
```

**`memory/scan.py`**
```python
scan_memory_dir(mem_dir, scope) -> list[MemoryHeader]  # newest-first, capped at 200
scan_all_memories() -> list[MemoryHeader]              # user + project merged
memory_age_str(mtime_s) -> str          # "today" | "yesterday" | "N days ago"
memory_freshness_text(mtime_s) -> str   # staleness warning for memories >1 day old
format_memory_manifest(headers) -> str  # formatted list for display
```

**`memory/context.py`**
```python
get_memory_context() -> str             # injected into system prompt
truncate_index_content(raw) -> str      # enforces <=200 lines / <=25 KB
find_relevant_memories(query, max_results=5, use_ai=False, config=None)
```

`find_relevant_memories` supports optional AI ranking: when `use_ai=True` it
makes a small API call to rank candidates by relevance to the query.

### Tools registered

| Tool | Parameters | Description |
|------|-----------|-------------|
| `MemorySave` | `name, description, type, content, scope` | Save or update a memory |
| `MemoryDelete` | `name, scope` | Delete a memory by name |
| `MemorySearch` | `query, scope, use_ai, max_results` | Search by keyword (or AI) |
| `MemoryList` | `scope` | List all memories with age and metadata |

### Index truncation

The `MEMORY.md` index is truncated before being injected into the system prompt:
- Hard limit: **200 lines** (mirrors Claude Code's limit)
- Byte limit: **25 000 bytes** (mirrors Claude Code's limit)
- A `WARNING:` line is appended when either limit is hit

### How it was wired in

1. `memory/store.py` exports `USER_MEMORY_DIR` and `get_project_memory_dir` as
   module-level names so tests can monkeypatch them cleanly.
2. `context.py` (system prompt builder) calls `get_memory_context()` at the end
   of `build_system_prompt()` and appends the result.
3. `tools.py` (top-level) adds:
   ```python
   import memory.tools as _memory_tools
   ```
4. `memory.py` (top-level) is now a shim:
   ```python
   from memory.store import MemoryEntry, save_memory, ...
   from memory.context import get_memory_context
   ```
5. `cheetahclaws.py` `/memory` command uses `scan_all_memories()` to display a
   mtime-sorted list with freshness warnings.

### Files changed

| File | Change |
|------|--------|
| `memory/__init__.py` | Created (re-exports) |
| `memory/types.py` | Created (MEMORY_TYPES, descriptions, format guidance) |
| `memory/store.py` | Created (replaced top-level `memory.py` logic) |
| `memory/scan.py` | Created (MemoryHeader, age/freshness, manifest) |
| `memory/context.py` | Created (context injection, truncation, AI search) |
| `memory/tools.py` | Created (MemorySave, MemoryDelete, MemorySearch, MemoryList) |
| `memory.py` | Converted to backward-compat shim |
| `tools.py` | Import `memory.tools` |
| `context.py` | Call `get_memory_context()` in `build_system_prompt()` |
| `cheetahclaws.py` | `/memory` command uses `scan_all_memories()` |
| `tests/test_memory.py` | Completely rewritten (101 tests total) |

---

## 3. Skill (`skill/`)

### What it does

Skills are reusable prompt templates stored as markdown files. A user types
`/commit` or `/review pr-123` in the REPL and the skill's prompt (with
arguments substituted) is injected into the conversation. Skills can run
**inline** (current conversation context) or **forked** (isolated sub-agent).
Two built-in skills (`/commit`, `/review`) are registered programmatically.

### Package structure

```
skill/
  __init__.py   — re-exports all public symbols; imports builtin to register them
  loader.py     — SkillDef dataclass, file parsing, load_skills, find_skill, substitute_arguments
  builtin.py    — built-in skills: /commit, /review
  executor.py   — execute_skill() (inline or forked)
  tools.py      — registers: Skill, SkillList
skills.py       — backward-compat shim
```

### Skill file format

Place `.md` files in:
- `~/.cheetahclaws/skills/<name>.md` (user-level)
- `.cheetahclaws/skills/<name>.md` (project-level, takes priority)

```markdown
---
name: deploy
description: Deploy to an environment
triggers: [/deploy]
allowed-tools: [Bash, Read]
when_to_use: Use when the user wants to deploy. Examples: '/deploy staging v1.2'
argument-hint: [env] [version]
arguments: [env, version]
context: inline
---

Deploy $VERSION to $ENV.

Full args provided: $ARGUMENTS
```

### Frontmatter fields

| Field | Default | Description |
|-------|---------|-------------|
| `name` | required | Skill identifier |
| `description` | `""` | One-line description shown in `/skills` |
| `triggers` | `[/<name>]` | Slash commands or phrases that activate this skill |
| `allowed-tools` / `tools` | `[]` | Tools the skill is allowed to use |
| `when_to_use` | `""` | Guidance for when Claude should auto-invoke |
| `argument-hint` | `""` | Hint shown in `/skills` list, e.g. `[branch] [desc]` |
| `arguments` | `[]` | Named argument list for `$ARG_NAME` substitution |
| `model` | `""` | Model override (fork context only) |
| `user-invocable` | `true` | Show in `/skills` list |
| `context` | `inline` | `inline` = current conversation, `fork` = isolated sub-agent |

### Argument substitution

`substitute_arguments(prompt, args, arg_names)` in `skill/loader.py`:

- `$ARGUMENTS` → the full raw args string
- `$ARG_NAME` → positional substitution (first word → first arg name, etc.)
- Missing args become empty strings

```
prompt:    "Deploy $VERSION to $ENV. Full: $ARGUMENTS"
args:      "1.0 staging"
arg_names: ["env", "version"]

result:    "Deploy staging to 1.0. Full: 1.0 staging"
```

### Execution modes

**Inline** (`context: inline`, default):
- Skill prompt is injected into the current `AgentState`
- History is shared — the user can see and continue the conversation

**Fork** (`context: fork`):
- A fresh `AgentState` is created (no shared history)
- Optional `model` and `allowed-tools` overrides are applied
- Good for self-contained tasks that don't need mid-process user input

### Built-in skills

Defined in `skill/builtin.py` and registered via `register_builtin_skill()`:

| Trigger | Name | Description |
|---------|------|-------------|
| `/commit` | commit | Review staged changes and create a well-structured git commit |
| `/review`, `/review-pr` | review | Review code or PR diff with structured feedback |

Project-level skill files with the same name override built-ins.

### Tools registered

| Tool | Parameters | Description |
|------|-----------|-------------|
| `Skill` | `name, args` | Invoke a skill by name from inside a conversation |
| `SkillList` | — | List all available skills with triggers and metadata |

### Priority order

When multiple skill sources define the same name, the highest priority wins:

```
builtin  <  user (~/.cheetahclaws/skills/)  <  project (.cheetahclaws/skills/)
```

### REPL usage

```
/commit                          # run built-in commit skill
/review 123                      # review PR #123 (args = "123")
/deploy staging 2.1.0            # custom skill with named args
/skills                          # list all skills
```

The `/skills` command output includes source label, triggers, argument hint,
and the first 80 chars of `when_to_use` per skill.

### How it was wired in

1. `skill/__init__.py` imports `skill.builtin` which calls `register_builtin_skill()`
   for each built-in — just importing the package registers them.
2. `tools.py` (top-level) adds:
   ```python
   import skill.tools as _skill_tools
   ```
3. `skills.py` (top-level) becomes a shim re-exporting from `skill/`.
4. `context.py` adds a `## Skills` section listing `Skill` and `SkillList`.
5. `cheetahclaws.py`:
   - `cmd_skills` imports from `skill`, shows `when_to_use` and source label
   - `handle_slash` imports `find_skill` from `skill`; returns `(skill, args)` tuple
   - REPL loop calls `substitute_arguments` before building the injected message

### Files changed

| File | Change |
|------|--------|
| `skill/__init__.py` | Created (re-exports; imports builtin) |
| `skill/loader.py` | Created (SkillDef, parse, load, find, substitute) |
| `skill/builtin.py` | Created (/commit, /review built-ins) |
| `skill/executor.py` | Created (inline + fork execution) |
| `skill/tools.py` | Created (Skill, SkillList tool registration) |
| `skills.py` | Converted to backward-compat shim |
| `tools.py` | Import `skill.tools` |
| `context.py` | Add Skills section to system prompt |
| `cheetahclaws.py` | `cmd_skills`, `handle_slash`, REPL loop updated |
| `tests/test_skills.py` | Rewritten (22 tests; patches `skill.loader`) |

---

## How to add custom agents, memories, and skills

### Custom agent type

Create `~/.cheetahclaws/agents/myagent.md`:
```markdown
---
name: myagent
description: Does specialized work
model: claude-haiku-4-5-20251001
tools: [Read, Grep, Bash]
---
You are specialized in X. Focus on Y. Never do Z.
```

Then use: `Agent(prompt="...", subagent_type="myagent")`

### Custom memory

Use the REPL `MemorySave` tool or write a file directly to
`~/.cheetahclaws/memory/my_note.md` with frontmatter:
```markdown
---
name: my note
description: short description
type: feedback
created: 2026-04-02
---
Memory content here.
```

### Custom skill

Create `~/.cheetahclaws/skills/myskill.md` (user-level) or
`.cheetahclaws/skills/myskill.md` (project-level):
```markdown
---
name: myskill
description: Does something useful
triggers: [/myskill]
arguments: [target]
argument-hint: [target]
when_to_use: Use when the user wants to do X with a target.
---

Do something useful with $TARGET.

Full context: $ARGUMENTS
```

Then invoke with `/myskill some-target`.

---

## Running tests

```bash
cd cheetahclaws

# All tests
python -m pytest tests/ -v

# Per-feature
python -m pytest tests/test_subagent.py -v   # multi-agent
python -m pytest tests/test_memory.py   -v   # memory
python -m pytest tests/test_skills.py   -v   # skills
```

Total: **101 tests**, all passing. Each feature's tests use `monkeypatch` to
redirect file system paths to `tmp_path` so no real `~/.cheetahclaws/`
directories are touched during testing.
