# Extensions — Memory, Skills, Sub-Agents, MCP, Plugins

## Memory

<div align=center>
<img src="../media/demos/memory_demo.gif" width="850"/>
</div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Memory: save preferences in session 1 → auto-recalled in session 2, no re-explanation needed</center>
</div>

The model can remember things across conversations using the built-in memory system.

### Storage

Memories are stored as individual markdown files in two scopes:

| Scope | Path | Visibility |
|---|---|---|
| **User** (default) | `~/.cheetahclaws/memory/` | Shared across all projects |
| **Project** | `.cheetahclaws/memory/` in cwd | Local to the current repo |

A `MEMORY.md` index (≤ 200 lines / 25 KB) is auto-rebuilt on every save or delete and injected into the system prompt so the model always has an overview of what's been remembered.

### Memory types

| Type | Use for |
|---|---|
| `user` | Your role, preferences, background |
| `feedback` | How you want the model to behave (corrections AND confirmations) |
| `project` | Ongoing work, deadlines, decisions not in git history |
| `reference` | Links to external systems (Linear, Grafana, Slack, etc.) |

### Memory file format

Each memory is a markdown file with YAML frontmatter:

```markdown
---
name: coding_style
description: Python formatting preferences
type: feedback
created: 2026-04-02
confidence: 0.95
source: user
last_used_at: 2026-04-05
conflict_group: coding_style
---
Prefer 4-space indentation and full type hints in all Python code.
**Why:** user explicitly stated this preference.
**How to apply:** apply to every Python file written or edited.
```

**Metadata fields** (new — auto-managed):

| Field | Default | Description |
|---|---|---|
| `confidence` | `1.0` | Reliability score 0–1. Explicit user statements = 1.0; inferred preferences ≈ 0.8; auto-consolidated ≈ 0.8 |
| `source` | `user` | Origin: `user` / `model` / `tool` / `consolidator` |
| `last_used_at` | — | Updated automatically each time this memory is returned by MemorySearch |
| `conflict_group` | — | Groups related memories (e.g. `writing_style`) for conflict tracking |

### Conflict detection

When `MemorySave` is called with a name that already exists but different content, the system reports the conflict before overwriting:

```
Memory saved: 'writing_style' [feedback/user]
⚠ Replaced conflicting memory (was user-sourced, 100% confidence, written 2026-04-01).
  Old content: Prefer formal, academic style...
```

### Ranked retrieval

`MemorySearch` ranks results by **confidence × recency** (30-day exponential decay) rather than plain keyword order. Memories that haven't been used recently fade in priority. Each search hit also updates `last_used_at` so frequently-accessed memories stay prominent.

```
You: /memory python
  [feedback/user] coding_style [conf:95% src:user]
    Python formatting preferences
    Prefer 4-space indentation and full type hints...
```

### `/memory consolidate` — auto-extract long-term insights

After a meaningful session, run:

```
[myproject] ❯ /memory consolidate
  Analyzing session for long-term memories…
  ✓ Consolidated 2 memory/memories: user_prefers_direct_answers, avoid_trailing_summaries
```

The command sends a condensed session transcript to the model and asks it to identify up to **3** insights worth keeping long-term (user preferences, feedback corrections, project decisions). Extracted memories are saved with `confidence: 0.80` and `source: consolidator` — they **never overwrite** an existing memory that already has higher confidence.

Good times to run `/memory consolidate`:
- After correcting the model's behavior several times in a row
- After a session where you shared project background or decisions
- After completing a task with clear planning choices

### Example interaction

```
You: Remember that I prefer 4-space indentation and type hints.
AI: [calls MemorySave] Memory saved: 'coding_style' [feedback/user]

You: /memory
  1 memory/memories:
  [feedback  |user   ] coding_style.md
    Python formatting preferences

You: /memory python
  Found 1 relevant memory for 'python':
  [feedback/user] coding_style
    Prefer 4-space indentation and full type hints in all Python code.

You: /memory consolidate
  ✓ Consolidated 1 memory: user_prefers_verbose_commit_messages
```

**Staleness warnings:** Memories older than 1 day show a `⚠ stale` caveat — claims about file:line citations or code state may be outdated; verify before acting.

**AI-ranked search:** `MemorySearch(query="...", use_ai=true)` uses the model to rank candidates by relevance before applying the confidence × recency re-ranking.

---

## Skills

Skills are reusable prompt templates that give the model specialized capabilities. Two built-in skills ship out of the box — no setup required.

**Built-in skills:**

| Trigger | Description |
|---|---|
| `/commit` | Review staged changes and create a well-structured git commit |
| `/review [PR]` | Review code or PR diff with structured feedback |

**Quick start — custom skill:**

```bash
mkdir -p ~/.cheetahclaws/skills
```

Create `~/.cheetahclaws/skills/deploy.md`:

```markdown
---
name: deploy
description: Deploy to an environment
triggers: [/deploy]
allowed-tools: [Bash, Read]
when_to_use: Use when the user wants to deploy a version to an environment.
argument-hint: [env] [version]
arguments: [env, version]
context: inline
---

Deploy $VERSION to the $ENV environment.
Full args: $ARGUMENTS
```

Now use it:

```
You: /deploy staging 2.1.0
AI: [deploys version 2.1.0 to staging]
```

**Argument substitution:**
- `$ARGUMENTS` — the full raw argument string
- `$ARG_NAME` — positional substitution by named argument (first word → first name)
- Missing args become empty strings

**Execution modes:**
- `context: inline` (default) — runs inside current conversation history
- `context: fork` — runs as an isolated sub-agent with fresh history; supports `model` override

**Priority** (highest wins): project-level > user-level > built-in

**List skills:** `/skills` — shows triggers, argument hint, source, and `when_to_use`

**Skill search paths:**

```
./.cheetahclaws/skills/     # project-level (overrides user-level)
~/.cheetahclaws/skills/     # user-level
```

---

## Sub-Agents

<div align=center>
<img src="../media/demos/subagent_demo.gif" width="850"/>
</div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Sub-Agents: spawn coder + security agents in parallel, merge results automatically</center>
</div>

The model can spawn independent sub-agents to handle tasks in parallel.

**Specialized agent types** — built-in:

| Type | Optimized for |
|---|---|
| `general-purpose` | Research, exploration, multi-step tasks |
| `coder` | Writing, reading, and modifying code |
| `reviewer` | Security, correctness, and code quality analysis |
| `researcher` | Web search and documentation lookup |
| `tester` | Writing and running tests |

**Basic usage:**
```
You: Search this codebase for all TODO comments and summarize them.
AI: [calls Agent(prompt="...", subagent_type="researcher")]
    Sub-agent reads files, greps for TODOs...
    Result: Found 12 TODOs across 5 files...
```

**Background mode** — spawn without waiting, collect result later:
```
AI: [calls Agent(prompt="run all tests", name="test-runner", wait=false)]
AI: [continues other work...]
AI: [calls CheckAgentResult / SendMessage to follow up]
```

**Git worktree isolation** — agents work on an isolated branch with no conflicts:
```
Agent(prompt="refactor auth module", isolation="worktree")
```
The worktree is auto-cleaned up if no changes were made; otherwise the branch name is reported.

**Custom agent types** — create `~/.cheetahclaws/agents/myagent.md`:
```markdown
---
name: myagent
description: Specialized for X
model: claude-haiku-4-5-20251001
tools: [Read, Grep, Bash]
---
Extra system prompt for this agent type.
```

**List running agents:** `/agents`

Sub-agents have independent conversation history, share the file system, and are limited to 3 levels of nesting.

---

## MCP (Model Context Protocol)

MCP lets you connect any external tool server — local subprocess or remote HTTP — and Claude can use its tools automatically. This is the same protocol Claude Code uses to extend its capabilities.

### Supported transports

| Transport | Config `type` | Description |
|---|---|---|
| **stdio** | `"stdio"` | Spawn a local subprocess (most common) |
| **SSE** | `"sse"` | HTTP Server-Sent Events stream |
| **HTTP** | `"http"` | Streamable HTTP POST (newer servers) |

### Configuration

Place a `.mcp.json` file in your project directory **or** edit `~/.cheetahclaws/mcp.json` for user-wide servers.

```json
{
  "mcpServers": {
    "git": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-git"]
    },
    "filesystem": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/tmp"]
    },
    "my-remote": {
      "type": "sse",
      "url": "http://localhost:8080/sse",
      "headers": {"Authorization": "Bearer my-token"}
    },
    "github-tools": {
      "type": "http",
      "url": "https://example.com/mcp"
    },
    "sap-jira": {
      "type": "http",
      "url": "https://jira.example.com/mcp"
    }
  }
}
```

Config priority: `.mcp.json` (project) overrides `~/.cheetahclaws/mcp.json` (user) by server name.

#### Environment variable expansion in headers

Header values support `$VAR` and `${VAR}` syntax — useful for keeping secrets out of `mcp.json`:

```json
"headers": {"Authorization": "Bearer $GITHUB_TOKEN"}
```

The variables are expanded once at config load time, after the `.env` loader runs (see [Reference: Environment Variables](reference.md#environment-variables)).

#### OAuth 2.0 (HTTP transport)

For HTTP MCP servers that require OAuth (e.g. enterprise SAP/Jira), CheetahClaws speaks the full MCP Authorization spec:

- Resource server metadata discovery (RFC 9728)
- Authorization server metadata discovery (RFC 8414)
- Dynamic client registration (RFC 7591) — used when no `client_id` is configured
- Authorization Code + PKCE (S256) flow with browser redirect
- Automatic refresh-token rotation
- Token persistence to `~/.cheetahclaws/mcp_oauth.json` (mode `0600`)

You don't have to do anything beyond declaring the server URL: on the first `401` the client opens your browser, you sign in, and the resulting access token is cached and refreshed transparently. To force a re-auth, delete the relevant entry in `~/.cheetahclaws/mcp_oauth.json`.

### Quick start

```bash
# Install a popular MCP server
pip install uv        # uv includes uvx
uvx mcp-server-git --help   # verify it works

# Add to user config via REPL
/mcp add git uvx mcp-server-git

# Or create .mcp.json in your project dir, then:
/mcp reload
```

### REPL commands

```
/mcp                                       # list servers + their tools + connection status
/mcp list                                  # alias for the above
/mcp reload                                # reconnect all servers, refresh tool list
/mcp reload git                            # reconnect a single server
/mcp add myserver uvx mcp-server-x         # add stdio server
/mcp add myserver --transport http <url>   # add HTTP/SSE server (OAuth runs on first 401)
/mcp add myserver --transport sse <url>
/mcp remove myserver                       # remove from user config
```

### How Claude uses MCP tools

Once connected, Claude can call MCP tools directly:

```
You: What files changed in the last git commit?
AI: [calls mcp__git__git_diff_staged()]
    → shows diff output from the git MCP server
```

Tool names follow the pattern `mcp__<server_name>__<tool_name>`. All characters
that are not alphanumeric or `_` are automatically replaced with `_`.

### Popular MCP servers

| Server | Install | Provides |
|---|---|---|
| `mcp-server-git` | `uvx mcp-server-git` | git operations (status, diff, log, commit) |
| `mcp-server-filesystem` | `uvx mcp-server-filesystem <path>` | file read/write/list |
| `mcp-server-fetch` | `uvx mcp-server-fetch` | HTTP fetch tool |
| `mcp-server-postgres` | `uvx mcp-server-postgres <conn-str>` | PostgreSQL queries |
| `mcp-server-sqlite` | `uvx mcp-server-sqlite --db-path x.db` | SQLite queries |
| `mcp-server-brave-search` | `uvx mcp-server-brave-search` | Brave web search |

> Browse the full registry at [modelcontextprotocol.io/servers](https://modelcontextprotocol.io/servers)

---

## Plugin System

The `plugin/` package lets you extend cheetahclaws with additional tools, skills, and MCP servers from git repositories or local directories.

### Install a plugin

```bash
/plugin install my-plugin@https://github.com/user/my-plugin
/plugin install local-plugin@/path/to/local/plugin
```

### Manage plugins

```bash
/plugin                   # list installed plugins
/plugin enable my-plugin  # enable a disabled plugin
/plugin disable my-plugin # disable without uninstalling
/plugin disable-all       # disable all plugins
/plugin update my-plugin  # pull latest from git
/plugin uninstall my-plugin
/plugin info my-plugin    # show manifest details
```

### Plugin recommendation engine

```bash
/plugin recommend                    # auto-detect from project files
/plugin recommend "docker database"  # recommend by keyword context
```

The engine matches your context against a curated marketplace (git-tools, python-linter, docker-tools, sql-tools, test-runner, diagram-tools, aws-tools, web-scraper) using tag and keyword scoring.

### Plugin manifest (plugin.json)

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "Does something useful",
  "author": "you",
  "tags": ["git", "python"],
  "tools": ["tools"],        // Python module(s) that export TOOL_DEFS
  "skills": ["skills/my.md"],
  "mcp_servers": {},
  "dependencies": ["httpx"]  // pip packages
}
```

Alternatively use YAML frontmatter in `PLUGIN.md`.

### Scopes

| Scope | Location | Config |
|-------|----------|--------|
| user (default) | `~/.cheetahclaws/plugins/` | `~/.cheetahclaws/plugins.json` |
| project | `.cheetahclaws/plugins/` | `.cheetahclaws/plugins.json` |

Use `--project` flag: `/plugin install name@url --project`

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


## Monitor — AI Subscriptions & Alerts

`/monitor` turns cheetahclaws into a 24/7 AI research assistant that watches stocks, crypto, arxiv, world news, or any custom topic on a schedule and pushes AI-written reports to Telegram, Slack, or your terminal.

### Quick start — interactive wizard

```
[myproject] ❯ /monitor

╭─ AI Monitor + Decision Assistant ─────────────────────
│  Monitor anything 24/7. AI summarizes & pushes to you.
│
│  What do you want to do?
│
│  1.  Add a new subscription
│  2.  Run all subscriptions now  (preview reports)
│  3.  Start background scheduler
│  5.  Configure push notifications  (Telegram / Slack)
│  0.  Exit
╰────────────────────────────────────────────────────────

  » 1
```

The wizard walks you through topic → schedule → delivery channel → run now → start scheduler. Zero prior knowledge required.

### Available topics

| Topic | Source | Example |
|---|---|---|
| `ai_research` | arxiv RSS (cs.AI/cs.LG/cs.CL) + weekend API fallback | `/subscribe ai_research` |
| `stock_<TICKER>` | Yahoo Finance JSON API (no key) | `/subscribe stock_TSLA daily` |
| `crypto_<SYMBOL>` | CoinGecko public API (no key) | `/subscribe crypto_BTC 6h` |
| `world_news` | Reuters · BBC · Guardian · AP RSS | `/subscribe world_news --telegram` |
| `custom:<query>` | DuckDuckGo Instant Answer | `/subscribe custom:quantum computing weekly` |

### Schedules

`15m` · `30m` · `1h` · `2h` · `6h` · `12h` · `daily` · `weekly`

### Commands

```
/monitor                          # interactive wizard
/monitor run [topic]              # run now and print report
/monitor start                    # start background scheduler
/monitor stop                     # stop background scheduler
/monitor status                   # show scheduler state + subscriptions
/monitor set telegram <t> <id>    # configure Telegram delivery
/monitor set slack <t> <ch>       # configure Slack delivery
/monitor topics                   # list all built-in topics

/subscribe ai_research            # quick-add subscription (daily, auto channel)
/subscribe stock_TSLA daily --telegram
/subscriptions                    # list all active subscriptions
/unsubscribe ai_research          # remove a subscription
```

### Sample report output

```
📊 AI Research Digest — 2026-04-12

3 new papers on Large Language Models:

• **RLVR-Pro** (Chen et al.) — New RL training method achieves +4.2% on MMLU
  vs PPO baseline. Key insight: reward shaping with KL penalty prevents mode
  collapse on reasoning tasks.

• **FlashKV** (Park et al.) — KV-cache compression reducing memory 3× with
  <0.5% perplexity loss. Applicable to any transformer at inference time.

• **AgentBench-2** (Liu et al.) — 200-task evaluation suite for coding agents.
  Claude-3.7 leads at 73.4%, GPT-5 at 71.1%.
```

---

## Autonomous Agents

`/agent` starts autonomous background agent loops driven by Markdown task templates (inspired by Karpathy's `program.md` pattern). Each agent gets an isolated `AgentState`, calls the real model with full tools, and runs until stopped or the task is complete.

### Quick start — interactive wizard

```
[myproject] ❯ /agent

╭────────────────────────────────────────────────────────╮
│  🤖  Auto Agent  —  What do you want to do?            │
╰────────────────────────────────────────────────────────╯

  1  📚  Research Assistant
        Read papers → summarize → build related work

  2  🐛  Auto Bug Fixer
        Run tests → find failures → fix & commit

  3  ✍️   Paper Writer
        Write paper sections from an outline

  4  💻  Auto Coder
        Implement tasks from a backlog → test → commit

  5  📄  Custom template…
        Use your own .md program file

  q  Quit

  Choice [1-5, q]: 1
  Paper directory or search topic [.]: ~/papers/
  Output notes file [research_notes.md]:
  Agent name [research]:
  Auto-approve file writes? [Y/n]: Y
  Seconds between iterations [2]:

  ─── Summary ───────────────────────────────────
  Template  : research_assistant
  Name      : research
  Args      : ~/papers/ --output ~/.cheetahclaws/agents/research/output/research_notes.md
  Interval  : 2.0s
  Auto-approve: True
  Output    : ~/.cheetahclaws/agents/research/output/research_notes.md

  Start? [Y/n]: Y
✓ Agent 'research' is running.
  Log    : ~/.cheetahclaws/agents/research/log.jsonl
  Output : ~/.cheetahclaws/agents/research/output/research_notes.md
  Progress → this terminal (iterations print here).
  Stop   : /agent stop research
```

> **Where do outputs land?** When you give a *relative* output filename
> (e.g. `research_notes.md`), the wizard rewrites it to an absolute path
> under `~/.cheetahclaws/agents/<name>/output/` so generated artifacts
> stay out of your current working directory and your repo. Pass an
> *absolute* path (e.g. `/tmp/notes.md` or `~/Desktop/notes.md`) to
> override and save anywhere you want.

### Built-in templates

| Template | What it does |
|---|---|
| `research_assistant` | Lists PDFs / searches topic → extracts key contributions → updates `research_notes.md` + `related_work.md` → repeats |
| `auto_bug_fixer` | Runs `<test_cmd>` → reads failing test + source → fixes root cause → commits → repeats until green |
| `paper_writer` | Reads outline → writes each section in academic style → appends to `paper_draft.md` → repeats section by section |
| `auto_coder` | Reads `tasks.md` backlog → implements one task → tests → commits → marks done → repeats |

### Direct launch (power user)

```bash
/agent start research_assistant ~/papers/
/agent start auto_bug_fixer --test-cmd "pytest tests/" --repo .
/agent start paper_writer outline.md --output paper_draft.md --style NeurIPS
/agent start auto_coder --task "add rate limiting to the API"
/agent start /path/to/my_template.md --name myagent
```

**Flags:** `--name <name>` · `--interval <seconds>` · `--no-auto-approve`

### Agent lifecycle commands

```
/agent list                   # show all running agents + current status
/agent status <name>          # show recent 3 iteration summaries
/agent stop <name>            # stop a specific agent
/agent stop all               # stop all running agents
/agent templates              # list built-in + user templates
```

### Custom templates

Drop any `.md` file into `~/.cheetahclaws/agent_templates/` following the program.md pattern:

```markdown
# My Custom Agent

You are an autonomous agent. Your goal: <describe the goal>.

## Setup (first iteration only)
1. ...

## Each iteration
1. ...
2. ...

## Rules
- Do not stop unless <condition>.
- NEVER STOP unless explicitly stopped.
```

Templates are plain Markdown — no special syntax. The agent runs them as a system prompt.

### Phone control

Agents push iteration summaries to the active bridge (Telegram/Slack/WeChat) automatically. From your phone:

```
!agent list           # see all running agents
!agent status research  # last 3 iterations of 'research'
!agent stop research  # stop the agent
```

Iteration log is also persisted to `~/.cheetahclaws/agents/<name>/log.jsonl` for offline review.

---
