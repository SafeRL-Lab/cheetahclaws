# Reference — CLI, Commands, Tools, and Configuration

## CLI Reference

```
cheetahclaws [OPTIONS] [PROMPT]
# or: python cheetahclaws.py [OPTIONS] [PROMPT]

Options:
  -p, --print          Non-interactive: run prompt and exit
  -m, --model MODEL    Override model (e.g. gpt-4o, ollama/llama3.3)
  --accept-all         Auto-approve all operations (no permission prompts)
  --verbose            Show thinking blocks and per-turn token counts
  --show-tools         Show each tool call instead of a per-turn summary
                       (alias: --no-quiet; default is the compact summary)
  --thinking           Enable Extended Thinking (Claude only)
  --budget AMOUNT      Session budget cap: --budget $5 (cost) or --budget 200k
                       (tokens). Auto-saves and prompts to resume / raise on hit.
  --version            Print version and exit
  -h, --help           Show help
```

**Examples:**

```bash
# Interactive REPL with default model
cheetahclaws

# Switch model at startup
cheetahclaws --model gpt-4o
cheetahclaws -m ollama/deepseek-r1:32b

# Non-interactive / scripting
cheetahclaws --print "Write a Python fibonacci function"
cheetahclaws -p "Explain the Rust borrow checker in 3 sentences" -m gemini/gemini-2.0-flash

# CI / automation (no permission prompts)
cheetahclaws --accept-all --print "Initialize a Python project with pyproject.toml"

# Debug mode (see tokens + thinking)
cheetahclaws --thinking --verbose
```

---

## Slash Commands (REPL)

Type `/` and press **Tab** to see all commands with descriptions. Continue typing to filter, then Tab again to auto-complete. After a command name, press **Tab** again to see its subcommands (e.g. `/plugin ` → `install`, `uninstall`, `enable`, …).

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/model` | Show current model + list all available models |
| `/model <name>` | Switch model (takes effect immediately) |
| `/config` | Show all current config values |
| `/config key=value` | Set a config value (persisted to disk). v3.05.78+ parses JSON values: `["a","b"]`, `{"k":"v"}`, signed numbers, quoted strings — list/dict configs no longer get silently saved as literal strings. |
| `/config context_window=<N>` | Override the context window (tokens) for the session. `0` = use the model's default. Drives the prompt `%` indicator, `/context`, the compaction trigger, **and** the per-call output-token cap — all consistently. Distinct from `max_tokens` (which is the **output** cap, not the window). Bidirectional: a smaller value forces earlier compaction; a larger value corrects a stale default. Read live, so it takes effect on the next prompt (no restart). Warns if set above the model's real window (that would disable compaction and the API may reject oversized prompts). |
| `/config stream_mode=<mode>` | Force the Markdown streaming tier: `live` (full in-place Rich redraw), `commit` (append-only progressive Markdown — safe over SSH / Apple Terminal / pipes), or `plain` (raw tokens). Unset = auto-detected per device (`ui.render.auto_stream_mode`). Legacy `/config rich_live=true\|false` still works (`true`→`live`, `false`→`commit`). |
| `/save` | Save session (auto-named by timestamp) |
| `/save <filename>` | Save session to named file |
| `/load` | Interactive list grouped by date; enter number, `1,2,3` to merge, or `H` for full history |
| `/load <filename>` | Load a saved session by filename |
| `/resume` | Restore the last auto-saved session (`mr_sessions/session_latest.json`) |
| `/resume <filename>` | Load a specific file from `mr_sessions/` (or absolute path) |
| `/history` | Print full conversation history |
| `/context` | Visualize context-window usage as a Claude-Code-style cell grid, broken down by category (system prompt, system tools, memory files, skills, messages, free space) with per-category token counts and percentages. Honors a `context_window` override; falls back to `#`/`.` when the terminal isn't UTF-8. |
| `/cost` | Show token usage and estimated USD cost |
| `/budget` | View or set token/cost budgets. No args = show usage vs each budget (bars + %). `/budget $5` = session cost cap (USD); `/budget 200k` = session token cap (supports `200k`/`1.5m`); `/budget daily $20` / `/budget daily 2m` = daily caps; `/budget clear` = remove all. **One budget per scope** — a new cap *replaces* the other unit for that scope (so `/budget $5` after `/budget 200k` switches the session cap to cost, it doesn't stack). Enforced before each model call (projects the next request's input + clamps its output, so overshoot stays ≈ 0); warns at ≥80%/95%; on hit, auto-saves the session and prints how to `/resume` or raise the **same** cap (the hint matches the breached unit) and continue. Backed by the `session_token_budget` / `session_cost_budget` / `daily_token_budget` / `daily_cost_budget` config keys. |
| `/verbose` | Toggle verbose mode (tokens + thinking) |
| `/quiet` | Toggle compact tool display — hide per-tool execution lines and show one summary line per turn (on by default; `/verbose` overrides it) |
| `/thinking` | Toggle Extended Thinking (Claude only) |
| `/permissions` | Show current permission mode |
| `/permissions <mode>` | Set permission mode: `auto` / `accept-all` / `manual` |
| `/cwd` | Show current working directory |
| `/cwd <path>` | Change working directory |
| `/memory` | List all persistent memories |
| `/memory <query>` | Search memories by keyword (ranked by confidence × recency) |
| `/memory consolidate` | AI-extract up to 3 long-term insights from the current session |
| `/skills` | List available skills |
| `/agents` | Show sub-agent task status |
| `/mcp` | List configured MCP servers and their tools |
| `/mcp reload` | Reconnect all MCP servers and refresh tools |
| `/mcp reload <name>` | Reconnect a single MCP server |
| `/mcp add <name> <cmd> [args]` | Add a stdio MCP server to user config |
| `/mcp remove <name>` | Remove a server from user config |
| `/voice` | Record voice, transcribe with Whisper, auto-submit as prompt |
| `/voice status` | Show recording and STT backend availability |
| `/voice lang <code>` | Set STT language (e.g. `zh`, `en`, `ja`; `auto` to detect) |
| `/voice device` | List available input microphones and select one interactively |
| `/image [prompt]` | Capture clipboard image and send to vision model with optional prompt |
| `/img [prompt]` | Alias for `/image` |
| `/proactive` | Show current proactive polling status (ON/OFF and interval) |
| `/proactive <duration>` | Enable background sentinel polling (e.g. `5m`, `30s`, `1h`) |
| `/proactive off` | Disable background polling |
| `/cloudsave setup <token>` | Configure GitHub Personal Access Token for Gist sync |
| `/cloudsave` | Upload current session to a private GitHub Gist |
| `/cloudsave push [desc]` | Upload with an optional description |
| `/cloudsave auto on\|off` | Toggle auto-upload on `/exit` |
| `/cloudsave list` | List your cheetahclaws Gists |
| `/cloudsave load <gist_id>` | Download and restore a session from Gist |
| `/brainstorm` | Run a multi-persona AI brainstorm; prompts for agent count (2–100, default 5) |
| `/brainstorm <topic>` | Focus the brainstorm on a specific topic; prompts for agent count |
| `/ssj` | Open SSJ Developer Mode — interactive power menu with 14 workflow shortcuts |
| `/monitor` | Interactive wizard: add subscriptions, run now, start/stop scheduler, configure notifications |
| `/monitor run [topic]` | Run all (or one) subscription(s) immediately and print the AI report |
| `/monitor start` | Start the background scheduler daemon |
| `/monitor stop` | Stop the background scheduler daemon |
| `/monitor status` | Show scheduler status, subscriptions, and configured delivery channels |
| `/monitor set telegram <token> <chat_id>` | Configure Telegram delivery for monitor reports |
| `/monitor set slack <token> <channel_id>` | Configure Slack delivery for monitor reports |
| `/monitor topics` | List all built-in topics |
| `/subscribe <topic> [schedule] [--telegram] [--slack]` | Subscribe to a monitoring topic (e.g. `ai_research`, `stock_TSLA`, `crypto_BTC`, `world_news`, `custom:quantum computing`) |
| `/subscriptions` | List active subscriptions with schedule and last-run time |
| `/subs` | Alias for `/subscriptions` |
| `/unsubscribe <topic>` | Remove a subscription |
| `/agent` | Interactive wizard: choose template, answer questions, start background agent loop |
| `/agent start <template> [args]` | Direct launch — e.g. `/agent start research_assistant ~/papers/` |
| `/agent stop <name\|all>` | Stop a running agent (or all agents) |
| `/agent list` | Show all currently running agents and their status |
| `/agent status <name>` | Show recent iteration log for a named agent |
| `/agent templates` | List available templates (built-in + user-defined) |
| `/worker` | Auto-implement all pending tasks from `brainstorm_outputs/todo_list.txt` |
| `/worker <n,m,…>` | Implement specific pending tasks by number (e.g. `/worker 1,4,6`) |
| `/worker --path <file>` | Use a custom todo file path instead of the default |
| `/worker --workers <n>` | Limit the batch to N tasks per run (e.g. `/worker --workers 3`) |
| `/telegram <token> <chat_id>` | Configure and start the Telegram bot bridge |
| `/telegram` | Start the bridge using previously saved token + chat_id |
| `/telegram stop` | Stop the Telegram bridge |
| `/telegram status` | Show whether the bridge is running and the configured chat_id |
| `/wechat login` | Scan QR code with WeChat to authenticate, then start the bridge |
| `/wechat` | Start with saved credentials; triggers QR login if none saved |
| `/wechat stop` | Stop the WeChat bridge |
| `/wechat status` | Show running state and account info |
| `/wechat logout` | Clear saved credentials and stop the bridge |
| `/slack <token> <channel_id>` | Configure and start the Slack bridge |
| `/slack` | Start with saved credentials |
| `/slack stop` | Stop the Slack bridge |
| `/slack status` | Show running state and channel |
| `/slack logout` | Clear saved credentials and stop the bridge |
| `/video [topic]` | AI video factory: story → voice → images → subtitles → `.mp4` |
| `/video status` | Show video pipeline dependency availability |
| `/video niches` | List all 10 viral content niches |
| `/video --niche <id> [topic]` | Use a specific content niche |
| `/video --short [topic]` | Generate vertical short format (9:16) |
| `/tts [topic]` | TTS Content Factory: AI script → any voice style → MP3 audio file |
| `/tts status` | Show TTS dependency availability (ffmpeg, edge-tts, API keys) |
| `/checkpoint` | List all checkpoints (snapshots) for the current session |
| `/checkpoint <id>` | Rewind to checkpoint — restore files and conversation to that snapshot |
| `/checkpoint clear` | Delete all checkpoints for the current session |
| `/rewind` | Alias for `/checkpoint` |
| `/plan <description>` | Enter plan mode: read-only analysis, writes only to the plan file |
| `/plan` | Show current plan file contents |
| `/plan done` | Exit plan mode and restore original permissions |
| `/plan status` | Show whether plan mode is active |
| `/compact` | Manually compact the conversation (same as auto-compact but user-triggered) |
| `/compact <focus>` | Compact with focus instructions (e.g. `/compact keep the auth refactor context`) |
| `/init` | Create a `CLAUDE.md` template in the current working directory |
| `/export` | Export the conversation as a Markdown file to `.nano_claude/exports/` |
| `/export <filename>` | Export as Markdown or JSON (detected by `.json` extension) |
| `/copy` | Copy the last assistant response to the clipboard |
| `/status` | Show version, model, provider, permissions, session ID, token usage, and context % |
| `/doctor` | Diagnose installation health: Python, git, API key, optional deps, CLAUDE.md, checkpoint disk usage |
| `/theme` | List 15 console color presets with a live `info / ok / warn / err` swatch per row (current marked `*`) |
| `/theme <name>` | Apply and persist a theme (e.g. `dracula`, `nord`, `gruvbox`, `none`); also drives Rich Markdown code-block style |
| `/exit` / `/quit` | Exit |

**Switching models inside a session:**

<div align=center>
<img src="../media/demos/multimodel_demo.gif" width="850"/>
</div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Multi-Model Switching: Claude → GPT-4o → Ollama → back, full history preserved</center>
</div>

```
[myproject] ❯ /model
  Current model: claude-opus-4-6  (provider: anthropic)

  Available models by provider:
    anthropic     claude-opus-4-6, claude-sonnet-4-6, ...
    openai        gpt-4o, gpt-4o-mini, o3-mini, ...
    ollama        llama3.3, llama3.2, phi4, mistral, ...
    ...

[myproject] ❯ /model gpt-4o
  Model set to gpt-4o  (provider: openai)

[myproject] ❯ /model ollama/qwen2.5-coder
  Model set to ollama/qwen2.5-coder  (provider: ollama)
```

---

## Console Themes

`/theme` switches the entire CLI palette in-place — every existing `info / ok / warn / err` call site picks up the new colors without any code change. The choice persists to `~/.cheetahclaws/config.json` under `"theme"` and is re-applied on the next launch before any output renders.

### Available themes

| Theme         | Notes                                                  |
|---------------|--------------------------------------------------------|
| `default`     | Cyan accent, amber warn — the original CheetahClaws look |
| `dracula`     | The Dracula palette (purple accent, soft green ok)     |
| `nord`        | Frost blue accent, aurora green ok                     |
| `gruvbox`     | Gruvbox Dark hard-contrast yellows / reds              |
| `solarized`   | Solarized Dark blue accent, olive ok                   |
| `tokyo-night` | Tokyo Night blues + soft pinks                         |
| `catppuccin`  | Catppuccin Mocha pastels                               |
| `matrix`      | Pure-green hacker aesthetic                            |
| `synthwave`   | Magenta accent, neon green ok — vaporwave              |
| `midnight`    | Cyan/lime/red high-contrast dark                       |
| `ocean`       | Sky-blue + emerald, easy on the eyes                   |
| `monokai`     | Monokai cyan / green / yellow / pink semantics         |
| `cheetah`     | Amber accent — matches the CheetahClaws logo           |
| `mono`        | Truly grayscale (no chromatic colors)                  |
| `none`        | Strips every ANSI escape — output is plain text        |

### Color roles

Each palette declares four semantic colors plus a code style:

| Role     | Used for                                          |
|----------|---------------------------------------------------|
| `accent` | `info()`, primary chrome, `clr(text, "cyan"\|"blue")` |
| `ok`     | `ok()`, diff additions (`+`), `clr(text, "green")`  |
| `warn`   | `warn()`, `clr(text, "yellow"\|"magenta")`          |
| `err`    | `err()`, diff removals (`-`), `clr(text, "red")`    |
| `code`   | `rich.markdown.Markdown(code_theme=...)` for code blocks |

`info` and `ok` are intentionally distinct hexes per theme so success (`ok()`) and informational (`info()`) messages stay visually separable in every theme. `render_diff` follows the same split — additions are always the `ok` color, removals are always the `err` color.

### Defining a custom theme

Add an entry to `THEMES` in `ui/render.py`:

```python
"my-theme": {
    "accent": "#00D7FF",
    "ok":     "#00FF87",
    "warn":   "#FFAF00",
    "err":    "#FF5F5F",
    "code":   "monokai",   # any Pygments style name
},
```

It immediately appears in `/theme` with no further wiring. To ship a "no color at all" theme, use `{"disable_color": True, "code": "default"}` instead.

### Examples

```
[myproject] ❯ /theme
Available themes:
  * default          info  ok  warn  err   (monokai)
    dracula          info  ok  warn  err   (dracula)
    nord             info  ok  warn  err   (nord)
    ...
    none           (no color)              (default)

  Usage: /theme <name>

[myproject] ❯ /theme dracula
  Theme set to dracula.
```

---

## Configuring API Keys

### Method 1: Environment Variables (recommended)

```bash
# Add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export MOONSHOT_API_KEY=sk-...       # Kimi
export DASHSCOPE_API_KEY=sk-...      # Qwen
export ZHIPU_API_KEY=...             # Zhipu GLM
export DEEPSEEK_API_KEY=sk-...       # DeepSeek
export MINIMAX_API_KEY=...           # MiniMax
```

#### `.env` file (loaded automatically)

CheetahClaws loads a `.env` file from the project directory at startup, before any other module reads `os.environ`. Existing shell variables take priority over `.env` values, so you can override locally with `export VAR=...`.

```ini
# .env in your project root
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
ANTHROPIC_ENDPOINT=https://api.anthropic.com   # see below
```

Both `KEY=value` and `KEY="quoted value"` are supported. Lines starting with `#` are comments.

#### `ANTHROPIC_ENDPOINT` (corporate proxy override)

Set `ANTHROPIC_ENDPOINT` to point Claude API traffic at a corporate proxy or compatible relay instead of `https://api.anthropic.com`:

```bash
export ANTHROPIC_ENDPOINT=https://anthropic-proxy.corp.example.com
```

The env var always wins over any persisted value in `~/.cheetahclaws/config.json`, so `.env` changes take effect on the next launch without editing the JSON file. The endpoint is used by both the streaming client (`providers.py`) and the connectivity probes in `/doctor` and the setup wizard.

### Method 2: Set Inside the REPL (persisted)

```
/config anthropic_api_key=sk-ant-...
/config openai_api_key=sk-...
/config gemini_api_key=AIza...
/config kimi_api_key=sk-...
/config qwen_api_key=sk-...
/config zhipu_api_key=...
/config deepseek_api_key=sk-...
/config minimax_api_key=...
```

Keys are saved to `~/.cheetahclaws/config.json` and loaded automatically on next launch.

### Method 3: Edit the Config File Directly

```json
// ~/.cheetahclaws/config.json
{
  "model": "qwen/qwen-max",
  "max_tokens": 8192,
  "context_window": 0,
  "permission_mode": "auto",
  "verbose": false,
  "quiet": true,
  "thinking": false,
  "stream_mode": null,
  "session_token_budget": null,
  "session_cost_budget": null,
  "daily_token_budget": null,
  "daily_cost_budget": null,
  "qwen_api_key": "sk-...",
  "kimi_api_key": "sk-...",
  "deepseek_api_key": "sk-...",
  "minimax_api_key": "..."
}
```

---

## Permission System

| Mode | Behavior |
|---|---|
| `auto` (default) | Read-only operations always allowed. Prompts before Bash commands and file writes. |
| `accept-all` | Never prompts. All operations proceed automatically. |
| `manual` | Prompts before every single operation, including reads. |
| `plan` | Read-only analysis mode. Only the plan file (`.nano_claude/plans/`) is writable. Entered via `/plan <desc>` or the `EnterPlanMode` tool. |

**When prompted:**

```
  Allow: Run: git commit -am "fix bug"  [y/N/a(ccept-all)]
```

- `y` — approve this one action
- `n` or Enter — deny
- `a` — approve and switch to `accept-all` for the rest of the session

**Commands always auto-approved in `auto` mode:**
`ls`, `cat`, `head`, `tail`, `wc`, `pwd`, `echo`, `git status`, `git log`, `git diff`, `git show`, `find`, `grep`, `rg`, `python`, `node`, `pip show`, `npm list`, and other read-only shell commands.

---

## Built-in Tools

### Core Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `Read` | Read file with line numbers | `file_path`, `limit`, `offset` |
| `Write` | Create or overwrite file (shows diff) | `file_path`, `content` |
| `Edit` | Exact string replacement (shows diff) | `file_path`, `old_string`, `new_string`, `replace_all` |
| `Bash` | Execute shell command | `command`, `timeout` (default 30s) |
| `Glob` | Find files by glob pattern | `pattern` (e.g. `**/*.py`), `path` |
| `Grep` | Regex search in files (uses ripgrep if available) | `pattern`, `path`, `glob`, `output_mode` |
| `WebFetch` | Fetch and extract text from URL | `url`, `prompt` |
| `WebSearch` | Search the web via DuckDuckGo | `query` |

### Notebook & Diagnostics Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `NotebookEdit` | Edit a Jupyter notebook (`.ipynb`) cell | `notebook_path`, `new_source`, `cell_id`, `cell_type`, `edit_mode` (`replace`/`insert`/`delete`) |
| `GetDiagnostics` | Get LSP-style diagnostics for a source file (pyright/mypy/flake8 for Python; tsc/eslint for JS/TS; shellcheck for shell) | `file_path`, `language` (optional override) |

### Memory Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `MemorySave` | Save or update a persistent memory | `name`, `type`, `description`, `content`, `scope` |
| `MemoryDelete` | Delete a memory by name | `name`, `scope` |
| `MemorySearch` | Search memories by keyword (or AI ranking) | `query`, `scope`, `use_ai`, `max_results` |
| `MemoryList` | List all memories with age and metadata | `scope` |

### Sub-Agent Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `Agent` | Spawn a sub-agent for a task | `prompt`, `subagent_type`, `isolation`, `name`, `model`, `wait` |
| `SendMessage` | Send a message to a named background agent | `name`, `message` |
| `CheckAgentResult` | Check status/result of a background agent | `task_id` |
| `ListAgentTasks` | List all active and finished agent tasks | — |
| `ListAgentTypes` | List available agent type definitions | — |

### Background & Autonomy Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `SleepTimer` | Schedule a silent background timer; injects an automated wake-up prompt when it fires so the agent can resume monitoring or deferred tasks | `seconds` |

### Skill Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `Skill` | Invoke a skill by name from within the conversation | `name`, `args` |
| `SkillList` | List all available skills with triggers and metadata | — |

### MCP Tools

MCP tools are discovered automatically from configured servers and registered under the name `mcp__<server>__<tool>`. Claude can use them exactly like built-in tools.

| Example tool name | Where it comes from |
|---|---|
| `mcp__git__git_status` | `git` server, `git_status` tool |
| `mcp__filesystem__read_file` | `filesystem` server, `read_file` tool |
| `mcp__myserver__my_action` | custom server you configured |

> **Adding custom tools:** See [Architecture Guide](docs/architecture.md#tool-registry) for how to register your own tools.

---


## AskUserQuestion Tool

Claude can pause mid-task and interactively ask you a question before proceeding.

**Example invocation by Claude:**
```json
{
  "tool": "AskUserQuestion",
  "question": "Which database should I use?",
  "options": [
    {"label": "SQLite", "description": "Simple, file-based"},
    {"label": "PostgreSQL", "description": "Full-featured, requires server"}
  ],
  "allow_freetext": true
}
```

**What you see in the terminal:**
```
❓ Question from assistant:
   Which database should I use?

  [1] SQLite — Simple, file-based
  [2] PostgreSQL — Full-featured, requires server
  [0] Type a custom answer

Your choice (number or text):
```

- Select by number or type free text directly
- Claude receives your answer and continues the task
- 5-minute timeout (returns "(no answer — timeout)" if unanswered)

---

## Task Management

The `task/` package gives Claude (and you) a structured task list for tracking multi-step work within a session.

### Tools available to Claude

| Tool | Parameters | What it does |
|------|-----------|--------------|
| `TaskCreate` | `subject`, `description`, `active_form?`, `metadata?` | Create a task; returns `#id created: subject` |
| `TaskUpdate` | `task_id`, `subject?`, `description?`, `status?`, `owner?`, `add_blocks?`, `add_blocked_by?`, `metadata?` | Update any field; `status='deleted'` removes the task |
| `TaskGet` | `task_id` | Return full details of one task |
| `TaskList` | _(none)_ | List all tasks with status icons and pending blockers |

**Valid statuses:** `pending` → `in_progress` → `completed` / `cancelled` / `deleted`

### Dependency edges

```
TaskUpdate(task_id="3", add_blocked_by=["1","2"])
# Task 3 is now blocked by tasks 1 and 2.
# Reverse edges are set automatically: tasks 1 and 2 get task 3 in their "blocks" list.
```

Completed tasks are treated as resolved — `TaskList` hides their blocking effect on dependents.

### Persistence

Tasks are saved to `.cheetahclaws/tasks.json` in the current working directory after every mutation and reloaded on first access.

### REPL commands

```
/tasks                    list all tasks
/tasks create <subject>   quick-create a task
/tasks start <id>         mark in_progress
/tasks done <id>          mark completed
/tasks cancel <id>        mark cancelled
/tasks delete <id>        remove a task
/tasks get <id>           show full details
/tasks clear              delete all tasks
```

### Typical Claude workflow

```
User: implement the login feature

Claude:
  TaskCreate(subject="Design auth schema", description="JWT vs session")  → #1
  TaskCreate(subject="Write login endpoint", description="POST /auth/login") → #2
  TaskCreate(subject="Write tests", description="Unit + integration") → #3
  TaskUpdate(task_id="2", add_blocked_by=["1"])
  TaskUpdate(task_id="3", add_blocked_by=["2"])

  TaskUpdate(task_id="1", status="in_progress", active_form="Designing schema")
  ... (does the work) ...
  TaskUpdate(task_id="1", status="completed")
  TaskList()  → task 2 is now unblocked
  ...
```

---

## Session Search

Search across all past conversations using full-text search (powered by SQLite FTS5).

```
/search authentication bug
/search database migration
/search "React Server Components"
```

Output shows matching sessions with highlighted snippets:

```
  [a3f8c2e1] Auth refactor (gpt-4o)
    2026-04-14 15:30:22 · 12 turns
    How do I fix the >>>authentication<<< bug in login.py?

  [b7d4e9f0] DB migration plan (ollama/qwen)
    2026-04-13 09:15:00 · 8 turns
    ...>>>database<<< migration strategy for PostgreSQL...
```

Load any result with `/load <session_id>`.

Sessions are automatically indexed when saved. Legacy JSON sessions are auto-imported on first search.

---

## Auxiliary Model

Side tasks like context compression use a fast, cheap model instead of your primary model. This saves cost and speeds up compaction.

**Auto-detection order** (first available wins):
1. `config["auxiliary_model"]` (if explicitly set)
2. Gemini 2.0 Flash (`GEMINI_API_KEY`)
3. GPT-4o-mini (`OPENAI_API_KEY`)
4. DeepSeek Chat (`DEEPSEEK_API_KEY`)
5. Claude Haiku (`ANTHROPIC_API_KEY`)
6. Qwen Turbo (`DASHSCOPE_API_KEY`)
7. GLM-4 Flash (`ZHIPU_API_KEY`)
8. Your primary model (fallback)

**Manual override:**
```
/config auxiliary_model=gemini/gemini-2.0-flash
```

---

## Error Classification

API errors are automatically classified into categories with specific recovery actions:

| Category | Examples | Recovery |
|----------|----------|----------|
| `auth` | Invalid API key, 401/403 | Stop retrying, show hint |
| `billing` | Insufficient credits, 402 | Stop retrying, show hint |
| `rate_limit` | Too many requests, 429 | Retry with 3x backoff |
| `context_overflow` | Context too long | Auto-compact, then retry |
| `model_not_found` | Model does not exist, 404 | Stop retrying, suggest /model |
| `overloaded` | Server busy, 503 | Retry with 3x backoff |
| `connection` | Network error, refused | Retry with normal backoff |
| `timeout` | Request timed out | Retry with normal backoff |

Non-retryable errors (auth, billing, model_not_found) fail immediately with an actionable hint instead of wasting time on futile retries.

---

## Prompt Injection Detection

CLAUDE.md files are scanned for prompt injection patterns before being included in the system prompt. Detected threats are excluded with a warning.

**Patterns detected:**
- `ignore previous/all instructions`
- `system prompt override/replace`
- `you are now a ...` (identity hijack)
- `disregard all previous rules`
- `new instructions:` (injection)
- `curl ... $API_KEY` (credential exfiltration)
- `echo/export $SECRET` (env var leak)
- `base64 encode ... key/token` (obfuscation)

If a CLAUDE.md contains a threat, you'll see:
```
[SECURITY WARNING] Potential prompt injection detected in Project CLAUDE.md (/path/to/CLAUDE.md):
  Pattern: 'ignore all previous instructions'
  This content has been excluded from the system prompt.
```

---

## Tool Result Cache

Read-only tools (Read, Glob, Grep, WebSearch, etc.) automatically cache their results within a session. If the AI reads the same file with the same parameters twice, the cached result is returned instantly without re-executing.

- Cache key: `sha256(tool_name + params)`
- Max entries: 64 (LRU eviction)
- Write tools (Write, Edit, Bash, NotebookEdit) automatically invalidate the entire cache

This eliminates redundant file reads that waste tokens in multi-step agent loops.

---

## Parallel Tool Execution

When the AI issues multiple tool calls in a single response, tools marked as `concurrent_safe=True` run in parallel (up to 8 threads). This speeds up scenarios like reading multiple files simultaneously.

**Parallel-safe tools:** Read, Glob, Grep, WebSearch, WebFetch, MemorySearch, MemoryList, CheckAgentResult, ListAgentTasks, ListAgentTypes, SkillList, TaskGet, TaskList

**Always sequential:** Write, Edit, Bash, NotebookEdit, Agent (and any tool needing user interaction)

No configuration needed — parallelization is automatic when safe.

---

## Browser Tool (WebBrowse)

Renders JavaScript-heavy pages with a headless Chromium browser. Use instead of `WebFetch` for dynamic/SPA pages.

**Install:** `pip install cheetahclaws[browser]` then `playwright install chromium`

**Actions:**

| Action | Description |
|--------|-------------|
| `extract` (default) | Get page text content |
| `screenshot` | Capture page as PNG image |
| `click` | Click a CSS selector, then extract resulting content |

**Examples:**

```
# Extract text from a React SPA
WebBrowse(url="https://example.com/dashboard")

# Extract specific elements
WebBrowse(url="https://news.ycombinator.com", selector=".titleline > a")

# Click a button then read the result
WebBrowse(url="https://example.com", action="click", selector="#load-more")

# Wait longer for slow pages
WebBrowse(url="https://example.com/report", wait=10)
```

Falls back gracefully with install instructions if `playwright` is not installed.

---

## Email Tools (ReadEmail / SendEmail)

Read and send emails directly from the REPL. Uses Python stdlib (no external deps).

**Setup:**
```
/config email_address=you@gmail.com
/config email_password=your-app-password
/config email_imap_host=imap.gmail.com
/config email_smtp_host=smtp.gmail.com
```

> **Gmail users:** Use an [App Password](https://myaccount.google.com/apppasswords), not your regular password.

**ReadEmail examples:**

```
# Read latest 5 emails
ReadEmail()

# Read emails from a specific sender
ReadEmail(search="boss@company.com", limit=10)

# Search by subject
ReadEmail(search="quarterly report")

# Read from a different folder
ReadEmail(folder="Sent")
```

**SendEmail example:**

```
SendEmail(
  to="colleague@company.com",
  subject="Meeting notes",
  body="Here are the key takeaways from today's meeting..."
)
```

The AI will always ask for confirmation before sending.

---

## File Tools (ReadPDF / ReadImage / ReadSpreadsheet)

Enhanced file reading for common document formats.

### ReadPDF

**Install:** `pip install cheetahclaws[files]`

```
# Read entire PDF
ReadPDF(file_path="/path/to/document.pdf")

# Read specific pages
ReadPDF(file_path="/path/to/report.pdf", pages="1-5")

# Read specific pages by number
ReadPDF(file_path="/path/to/manual.pdf", pages="1,3,7-10")
```

### ReadImage (OCR)

**Install:** `pip install cheetahclaws[ocr]` + system Tesseract

```bash
# Install Tesseract OCR engine:
# macOS:  brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr
# For Chinese: sudo apt install tesseract-ocr-chi-sim
```

```
# English OCR
ReadImage(file_path="/path/to/screenshot.png")

# Chinese OCR
ReadImage(file_path="/path/to/document.jpg", language="chi_sim")

# Japanese
ReadImage(file_path="/path/to/scan.tiff", language="jpn")
```

### ReadSpreadsheet

**Install:** `pip install cheetahclaws[files]` (for Excel; CSV works without any install)

```
# Read CSV
ReadSpreadsheet(file_path="/path/to/data.csv")

# Read Excel with specific sheet
ReadSpreadsheet(file_path="/path/to/report.xlsx", sheet="Q4 Results")

# Limit rows
ReadSpreadsheet(file_path="/path/to/big_data.csv", max_rows=50)
```

Output is formatted as an aligned text table:

```
data.csv (showing 5 rows)

Name       | Age | City
-----------+-----+-----------
Alice      | 30  | New York
Bob        | 25  | London
Charlie    | 35  | Tokyo
```

---

