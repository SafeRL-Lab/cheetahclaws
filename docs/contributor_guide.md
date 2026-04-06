# Contributor Guide: Where to Change What in nano-claude-code

This guide is for contributors implementing new features or updating existing behavior.
It focuses on **which files matter**, **how data flows**, and **how to make safe changes quickly**.

---

## 1) Fast mental model

If you remember only one thing, remember this flow:

1. `nano_claude.py` handles CLI + REPL + slash commands.
2. `context.py` rebuilds the system prompt each turn.
3. `agent.py` runs the core loop (stream model output, execute tools, append tool results, continue).
4. `providers.py` adapts model APIs (Anthropic vs OpenAI-compatible providers).
5. `tool_registry.py` is the single source of truth for all callable tools.
6. Feature packages (`memory/`, `multi_agent/`, `skill/`, `mcp/`, `plugin/`, `task/`, `checkpoint/`, `voice/`) plug into that loop.

---

## 2) Core files you should read first

### Runtime + UX shell
- `nano_claude.py`
  - Entry point (`main()`), REPL loop (`repl()`), command dispatch (`COMMANDS`, `handle_slash()`), permission prompt UI, diff rendering, voice command handling.
  - Add or change slash commands here.

### Agent execution loop
- `agent.py`
  - `run(...)` generator is the heart of the app.
  - Event model: `TextChunk`, `ThinkingChunk`, `ToolStart`, `ToolEnd`, `PermissionRequest`, `TurnDone`.
  - Permission gate logic (`_check_permission`) and per-turn context compaction trigger.

### Tool system
- `tool_registry.py`
  - `ToolDef`, `register_tool`, `get_tool_schemas`, and centralized `execute_tool` dispatch/truncation.
  - Every tool (built-in, package, MCP, plugin) ends up here.

- `tools.py`
  - Core built-in tool schemas and implementations (`Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `WebFetch`, `WebSearch`, `NotebookEdit`, `GetDiagnostics`, `AskUserQuestion`).
  - `_register_builtins()` registers core tools, then imports package tool modules to auto-register additional tools.

### Model providers + prompt context + compaction
- `providers.py` — provider detection, model metadata, API key lookup, stream adapters, neutral message format conversion.
- `context.py` — system prompt assembly (git info, `CLAUDE.md`, memory injection).
- `compaction.py` — context window management (`snip_old_tool_results` + `compact_messages`).
- `config.py` — defaults + persistent config file handling.

---

## 3) Feature packages: exact entrypoints

## Memory (`memory/`)
- Start at `memory/tools.py` (tool behavior and schemas).
- Persistence/index rules are in `memory/store.py`.
- Memory retrieval/ranking context is in `memory/context.py`.
- Metadata scanning and freshness helpers are in `memory/scan.py`.

Use this package when adding memory types, changing indexing, staleness behavior, or search behavior.

## Multi-agent (`multi_agent/`)
- `multi_agent/tools.py` registers `Agent`, `SendMessage`, `CheckAgentResult`, `ListAgentTasks`, `ListAgentTypes`.
- `multi_agent/subagent.py` manages thread pool lifecycle, isolation (`git worktree`), depth control, and messaging.

Use this package for new agent types, changes to worktree behavior, depth/concurrency limits, or background task lifecycle.

## Skills (`skill/`)
- `skill/loader.py` parses markdown frontmatter and resolves project/user/builtin precedence.
- `skill/executor.py` runs inline vs forked skill execution.
- `skill/tools.py` exposes `Skill` and `SkillList` tool APIs.

Use this package when adding skill metadata fields, argument substitution behavior, or skill execution modes.

## MCP (`mcp/`)
- `mcp/config.py` loads/merges project `.mcp.json` and user config.
- `mcp/client.py` handles stdio/SSE/HTTP transport and JSON-RPC.
- `mcp/tools.py` connects servers and registers discovered tools as `mcp__<server>__<tool>`.

Use this package for transport support, tool discovery behavior, reconnect logic, or MCP config precedence changes.

## Plugins (`plugin/`)
- `plugin/store.py` install/uninstall/enable/disable/update and config persistence.
- `plugin/loader.py` dynamic import and registration of plugin tools/skills/MCP config.
- `plugin/recommend.py` recommendation logic.

Use this package for plugin manifest semantics, install lifecycle, or recommendation strategy updates.

## Tasks (`task/`)
- `task/types.py` task model + status enum.
- `task/store.py` thread-safe CRUD and dependency edge maintenance.
- `task/tools.py` `TaskCreate/Update/Get/List` schemas + formatting.

Use this package for status transitions, dependency graph behavior, metadata semantics, and storage format updates.

## Checkpoints (`checkpoint/`)
- `checkpoint/types.py` `FileBackup` + `Snapshot` data models.
- `checkpoint/store.py` file-level backup, snapshot persistence, rewind, cleanup.
- `checkpoint/hooks.py` Write/Edit/NotebookEdit interception (backup before modify).
- REPL command wiring lives in `nano_claude.py` (`cmd_checkpoint`, `cmd_rewind`).

Use this package for snapshot policies, backup strategies, file restore behavior, or storage format updates.

## Voice (`voice/`)
- `voice/recorder.py` capture backends (`sounddevice`, `arecord`, `sox`) + silence detection.
- `voice/stt.py` backend fallback chain (`faster-whisper`, `openai-whisper`, OpenAI API).
- `voice/keyterms.py` keyterm extraction from repo/branch/files.
- REPL command wiring lives in `nano_claude.py` (`cmd_voice`).

Use this package for STT backend changes, audio capture behavior, and prompt-boosting vocabulary logic.

---

## 4) “I need to implement X” → where to edit

### Add a new built-in tool
1. Add schema + implementation in `tools.py`.
2. Register in `_register_builtins()` as a `ToolDef`.
3. Decide `read_only` and `concurrent_safe` correctly.
4. If it mutates files/system, ensure permission behavior is correct in `agent.py` / `tools.execute_tool` wrapper.
5. Add tests in `tests/test_tool_registry.py` and/or feature-specific tests.

### Add a new slash command
1. Add `cmd_<name>` function in `nano_claude.py`.
2. Add command mapping in `COMMANDS`.
3. If command needs tool behavior, prefer a tool module and call that logic.
4. Add tests in relevant test module (or create a focused one).

### Add a new model provider or provider behavior
1. Update `providers.py` (`PROVIDERS`, auto-detection prefixes, stream adapter behavior).
2. Add/verify key lookup path in `get_api_key`.
3. Confirm message conversion in `messages_to_openai`/`messages_to_anthropic` if provider-specific quirks exist.
4. Update docs and provider list references.

### Change prompt/context injection
1. Modify `context.py` (`build_system_prompt`, `get_git_info`, `get_claude_md`).
2. If memory behavior changes, update `memory/context.py` and `memory/store.py` as needed.
3. Validate prompt size impact via `compaction.py` behavior.

### Change compaction behavior
1. Edit thresholds/splitting in `compaction.py`.
2. Ensure both cheap-snipping and model-summarization layers still compose safely.
3. Add or update tests in `tests/test_compaction.py`.

### Add a new feature package
1. Create package module(s) with clear API + `ToolDef` registrations.
2. Ensure package is imported from `tools.py` so registrations execute at startup.
3. Add slash command wiring in `nano_claude.py` only if user-facing command is needed.
4. Add focused tests under `tests/test_<feature>.py`.

---

## 5) Tests: what to run and where to add coverage

Current tests are organized by subsystem:

- `tests/test_tool_registry.py`
- `tests/test_compaction.py`
- `tests/test_memory.py`
- `tests/test_subagent.py`
- `tests/test_skills.py`
- `tests/test_mcp.py`
- `tests/test_plugin.py`
- `tests/test_task.py`
- `tests/test_voice.py`
- `tests/test_diff_view.py`
- `tests/test_checkpoint.py`
- `tests/e2e_checkpoint.py`
- `tests/e2e_plan_mode.py`

Recommended contributor workflow:

1. Run only impacted test module(s) first.
2. Then run the full suite before opening a PR.
3. If adding a new capability, add at least one success-path and one failure/edge test.

---

## 6) Important conventions and gotchas

- **Registry-first architecture:** if functionality should be callable by the model, it should be a registered tool.
- **Import side effects matter:** package tool modules are often imported for registration side effects.
- **Permission model is split:** `agent.py` does high-level checks; `tools.execute_tool` includes backward-compatible gating too.
- **Context pressure is real:** large tool outputs are truncated in `tool_registry.execute_tool`, then old results may be snipped/compacted.
- **Neutral message format is the internal contract:** provider adapters must preserve tool call IDs and arguments correctly.
- **Task and memory persistence are cwd/home dependent:** behavior can vary if tests or runtime change working directory.
- **Path naming note:** most runtime dirs use `.nano_claude` (underscore).

---

## 7) Suggested order for onboarding contributors

If you are new and want to ship your first feature quickly, read in this order:

1. `README.md` (user surface)
2. `nano_claude.py` (runtime shell)
3. `agent.py` (core loop)
4. `tool_registry.py` + `tools.py` (extension spine)
5. Your target package (`memory/`, `mcp/`, `task/`, etc.)
6. Matching `tests/test_*.py`

This sequence minimizes time-to-productivity and reduces accidental architecture drift.

---

## 8) PR checklist (practical)

- [ ] Feature is implemented in the correct layer (tool vs slash command vs provider vs package).
- [ ] Tool schema and implementation are both updated where needed.
- [ ] Permission behavior is intentional and safe.
- [ ] Tests updated/added for changed behavior.
- [ ] README/docs updated if user-facing behavior changed.
- [ ] No unrelated refactors mixed into the same PR.
