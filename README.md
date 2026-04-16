English | [中文](https://github.com/SafeRL-Lab/clawspring/blob/main/docs/README.CN.MD) | [한국어](https://github.com/SafeRL-Lab/clawspring/blob/main/docs/README.KO.MD) | [日本語](https://github.com/SafeRL-Lab/clawspring/blob/main/docs/README.JP.MD) | [Français](https://github.com/SafeRL-Lab/clawspring/blob/main/docs/README.FR.MD) | [Deutsch](https://github.com/SafeRL-Lab/clawspring/blob/main/docs/README.DE.MD) | [Español](https://github.com/SafeRL-Lab/cheetahclaws/blob/main/docs/README.ES.MD) | [Português](https://github.com/SafeRL-Lab/cheetahclaws/blob/main/docs/README.PT.MD)

<br> 

<div align="center">
  <a href="[https://github.com/SafeRL-Lab/Robust-Gymnasium](https://github.com/SafeRL-Lab/clawspring)">
    <img src="docs/logo-5.png" alt="Logo" width="280"> 
  </a>

  
<h2 align="center" style="font-size: 30px;"><strong><em>CheetahClaws (Nano Claude Code) </em></strong>: A Fast, Easy-to-Use, Python-Native Personal AI Assistant for Any Model, Inspired by OpenClaw and Claude Code, Built to Work for You Autonomously 24/7</h2>
<p align="center">
    <a href="https://cheetahclaws.github.io/">Website</a>
    ·
    <a href="https://deepwiki.com/SafeRL-Lab/cheetahclaws">Brief Intro</a>
    ·
    <a href="https://github.com/SafeRL-Lab/clawspring/issues">Issue</a>
    ·
    <a href="https://github.com/chauncygu/collection-claude-code-source-code">The newest source of Claude Code</a>
    
  
  </p>
</div>


### Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/SafeRL-Lab/cheetahclaws/main/scripts/install.sh | bash
```

After installation:

```bash
source ~/.zshrc     # macOS
# or: source ~/.bashrc   # Linux
cheetahclaws        # start chatting!
```

Other install methods: [pip install](#alternative-install-with-pip) | [uv install](#alternative-install-with-uv) | [run from source](#alternative-run-directly-from-source-no-install) | [full details](#installation)

### Demos
 <div align=center>
 <img src="https://github.com/SafeRL-Lab/clawspring/blob/main/docs/demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Task Excution</center>
 </div>
 
 
---

  <div align=center>
 <img src="https://github.com/SafeRL-Lab/clawspring/blob/main/docs/brainstorm_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Brainstorm Mode: Multi-Agent Brainstorm</center>
 </div>



---

  <div align=center>
 <img src="https://github.com/SafeRL-Lab/clawspring/blob/main/docs/proactive_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Proactive Mode: Autonomous Agent</center>
 </div>

---

  <div align=center>
 <img src="https://github.com/SafeRL-Lab/clawspring/blob/main/docs/ssj_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">SSJ Mode (Simple and Smart Job Mode): Power Menu Workflow</center>
 </div>

---

  <div align=center>
 <img src="https://github.com/SafeRL-Lab/clawspring/blob/main/docs/telegram_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Telegram Bridge: Control cheetahclaws from Your Phone</center>
 </div>

---

  <div align=center>
 <img src="https://github.com/SafeRL-Lab/cheetahclaws/blob/main/docs/wechat_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">WeChat Bridge: Control cheetahclaws from WeChat (微信)</center>
 </div>

---

  <div align=center>
 <img src="https://github.com/SafeRL-Lab/cheetahclaws/blob/main/docs/slack_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Slack Bridge: Control cheetahclaws from Slack</center>
 </div>

---

 <div align=center>
 <img src="https://github.com/SafeRL-Lab/cheetahclaws/blob/main/docs/trading_demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Autonomous Trading Agent</center>
 </div>

---



 
## 🔥🔥🔥 News (Pacific Time)

 
- Apr 16, 2026 (**v3.05.74**): **Web UI production hardening — persistence, multi-user auth, ops endpoints, JS module split, pytest suite**
  - **SQLite persistence** (`web/db.py`, `web/models.py`) — SQLAlchemy-backed store with 4 tables: `users`, `chat_sessions`, `messages`, `api_credentials`. Sessions + message history now survive server restarts (previously in-memory only, lost on restart). DB file at `~/.cheetahclaws/web.db` (0600). Config key `CHEETAHCLAWS_WEB_DB` overrides the path.
  - **Multi-user auth** (`web/auth.py`) — replaced single generated password with full accounts: bcrypt password hashing (passlib) + stateless JWT cookies (PyJWT, HS256, 7-day TTL). JWT signing secret persisted to `~/.cheetahclaws/web_secret` (0600) so logins survive restarts. New endpoints: `POST /api/auth/register` (first user becomes admin), `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/whoami`, `GET /api/auth/bootstrap` (first-run routing). Legacy `POST /api/auth` kept for the terminal password page.
  - **Session CRUD** — new `PATCH /api/sessions/{id}` to rename, `DELETE /api/sessions/{id}` to remove, `GET /api/sessions/{id}/export` to download conversation as Markdown. Auto-titling from first user message. Cross-user isolation enforced even on in-memory cache hits (one session hit patched after smoke test revealed the leak).
  - **Structured JSON logging** (`web/logging_setup.py`) — `logging` + custom JSON formatter emits one record per line to stderr, e.g. `{"ts":..., "level":"info", "logger":"web.server", "msg":"req", "method":"POST", "path":"/api/auth/login", "status":200, "dur_ms":259, "user_id":1}`. Every HTTP response auto-logs method/path/status/dur_ms/user_id/peer. Level controlled by `CHEETAHCLAWS_LOG_LEVEL` env (default INFO).
  - **Ops endpoints** — `GET /health` returns `{ok, db, uptime_s}` (503 if DB unreachable); `GET /metrics` returns Prometheus v0.0.4 text with `cheetahclaws_{uptime_seconds, requests_total, requests_4xx, requests_5xx, auth_logins_total, auth_logins_failed, auth_registrations_total, users_total, ws_connections_total}`. Unauthenticated so Prometheus/k8s probes can hit them.
  - **JS module split** (`web/static/js/`) — monolithic 1813-line `chat.html` → 552 lines of HTML + 9 vanilla JS modules (`chat.js` core class, `util.js`, `auth.js`, `sidebar.js`, `tools.js`, `approval.js`, `settings.js`, `welcome.js`, `init.js`) loaded via plain `<script src>` tags. Prototype-mixin pattern (`Object.assign(ChatApp.prototype, {...})`) keeps `app.foo()` call sites unchanged. No bundler, no build step.
  - **ETag + conditional caching** — JS/CSS/HTML served with `Cache-Control: no-cache, must-revalidate` + weak ETag (`mtime-size`). Browser gets 304 when unchanged, fresh content after any edit. Binary assets keep 24h cache. Path traversal blocked by resolved-path `is_relative_to` check.
  - **pytest suite** (`tests/test_web_api.py`) — 21 end-to-end HTTP tests using httpx: bootstrap/register/login/whoami/logout, sessions CRUD + export + markdown, cross-user isolation, persistence after cache clear, `/health`, `/metrics` counter deltas, CORS preflight, auth gating of every endpoint. Spins the real server in a thread on a random port, DB truncated between tests. Runs in ~5s. `pytest tests/test_web_api.py`.
  - **Sidebar UX** — chat sessions now show title + relative time ("just now", "12m ago", "3d ago") + message count + busy dot. Search box filters by title/id on the client. Right-click (or long-press) gives a context menu: Rename / Export Markdown / Delete. Footer shows current username + Sign out link.
  - **Register-or-login on first visit** — chat UI now calls `/api/auth/bootstrap` on load; if no user exists it shows a "Create your first account" form (first registration becomes admin), otherwise the "Sign in" form. Username + password instead of a single server-generated password.
  - **Theme: light default + system auto** — `:root` now carries the light palette; `@media (prefers-color-scheme: dark)` swaps in the dark palette when the user hasn't explicitly chosen a theme. Toggle button cycles **system → light → dark → system**, icon reflects the effective theme, title tooltip spells out the current mode. Inline pre-paint script in `<head>` sets `data-theme` before first paint to avoid FOUC.
  - **Auto port selection** — `cheetahclaws --web` (no `--port`) now tries 8080 first; on `EADDRINUSE` it binds `:0` and lets the kernel pick a free port, banner reports the real URL. Explicit `--port N` binds exactly N or fails loudly (user intent preserved). `--port` argparse default changed from `8080` → `None` as a sentinel.
  - **Favicon + MIME polish** — `web/static/favicon.{png,ico}` cropped from `docs/logo-5.png` (leaping cheetah, transparent background, multi-size ICO 16/32/48). Served from root as `/favicon.ico` for browser defaults. MIME table extended with `.ico` (`image/vnd.microsoft.icon`), `.svg`, `.jpg`, `.woff`, `.woff2`.
  - **Welcome dashboard rebalanced** — old 5-card "Bridges & Media" row (ragged in 2×2 grid) split into two 4-card sections: **Bridges** (Telegram · WeChat · Slack · Monitor) and **Multi-Modal Media** (Voice Input · Vision · Copy Output · Export). `/cwd` added to Development Tools. Tagline changed to **"Personal AI Assistant · Support Any Model · Autonomous 24/7"**.
  - **Bridges commands in Chat UI** — `/telegram`, `/wechat` (+`/weixin` alias), `/slack`, `/voice` now registered in `web/api.py`'s slash registry (previously only the terminal REPL had them), so clicking the dashboard cards actually runs the command.
  - **New extras** — `pip install 'cheetahclaws[web]'` installs `sqlalchemy>=2.0`, `passlib[bcrypt]>=1.7.4`, `PyJWT>=2.8.0`. CLI-only installs remain dependency-free. `[all]` extra updated.
  - **Version bumped to 3.05.74.**











 
For more news, see [here](https://github.com/SafeRL-Lab/cheetahclaws/blob/main/docs/news.md)


---

# CheetahClaws

CheetahClaws: **A Lightweight** and **Easy-to-Use** Python Reimplementation of Claude Code **Supporting Any Model**, such as Claude, GPT, Gemini, Kimi, Qwen, Zhipu, DeepSeek, MiniMax, and local open-source models via Ollama or any OpenAI-compatible endpoint.

---

## Content
  * [Why CheetahClaws](#why-cheetahclaws)
  * [CheetahClaws vs OpenClaw](#cheetahclaws-vs-openclaw)
  * [Features](#features)
  * [Supported Models](#supported-models)
  * [Installation](#installation)
  * [Usage: Closed-Source API Models](#usage-closed-source-api-models)
  * [Usage: Open-Source Models (Local)](#usage-open-source-models-local)
  * [Model Name Format](#model-name-format)
  * [Trading Agent](#trading-agent) (multi-agent analysis, backtesting, memory)
  * [Web UI](#web-ui) (chat interface, settings, API endpoints)
  * [Documentation](#documentation) (guides for all features)
  * [Contributing](#contributing)
  * [FAQ](#faq)
  * [Citation](#citation)




## Why CheetahClaws

Claude Code is a powerful, production-grade AI coding assistant — but its source code is a compiled, 12 MB TypeScript/Node.js bundle (~1,300 files, ~283K lines). It is tightly coupled to the Anthropic API, hard to modify, and impossible to run against a local or alternative model.

**CheetahClaws** reimplements the same core loop in ~40K lines of readable Python, keeping everything you need and dropping what you don't. See here for more detailed analysis (CheetahClaws v3.03), [English version](https://github.com/SafeRL-Lab/clawspring/blob/main/docs/comparison_claude_code_vs_nano_v3.03_en.md) and [Chinese version](https://github.com/SafeRL-Lab/clawspring/blob/main/docs/comparison_claude_code_vs_nano_v3.03_cn.md)

### At a glance

| Dimension | Claude Code (TypeScript) | CheetahClaws (Python) |
|-----------|--------------------------|---------------------------|
| Language | TypeScript + React/Ink | Python 3.8+ |
| Source files | ~1,332 TS/TSX files | ~85 Python files |
| Lines of code | ~283K | ~40K |
| Built-in tools | 44+ | 27 |
| Slash commands | 88 | 36 |
| Voice input | Proprietary Anthropic WebSocket (OAuth required) | Local Whisper / OpenAI API — works offline, no subscription |
| Model providers | Anthropic only | 8+ (Anthropic · OpenAI · Gemini · Kimi · Qwen · DeepSeek · MiniMax · Ollama · …) |
| Local models | No | Yes — Ollama, LM Studio, vLLM, any OpenAI-compatible endpoint |
| Build step required | Yes (Bun + esbuild) | No — run directly with `python cheetahclaws.py` (or install to use `cheetahclaws`) |
| Runtime extensibility | Closed (compile-time) | Open — `register_tool()` at runtime, Markdown skills, git plugins |
| Task dependency graph | No | Yes — `blocks` / `blocked_by` edges in `task/` package |

### Where Claude Code wins

- **UI quality** — React/Ink component tree with streaming rendering, fine-grained diff visualization, and dialog systems.
- **Tool breadth** — 44 tools including `RemoteTrigger`, `EnterWorktree`, and more UI-integrated tools.
- **Enterprise features** — MDM-managed config, team permission sync, OAuth, keychain storage, GrowthBook feature flags.
- **AI-driven memory extraction** — `extractMemories` service proactively extracts knowledge from conversations without explicit tool calls.
- **Production reliability** — single distributable `cli.js`, comprehensive test coverage, version-locked releases.

### Where CheetahClaws wins

- **Multi-provider** — switch between Claude, GPT-4o, Gemini 2.5 Pro, DeepSeek, Qwen, MiniMax, or a local Llama model with `--model` or `/model` — no recompile needed.
- **Local model support** — run entirely offline with Ollama, LM Studio, or any vLLM-hosted model.
- **Readable source** — the full agent loop is 174 lines (`agent.py`). Any Python developer can read, fork, and extend it in minutes.
- **Zero build** — `pip install -r requirements.txt` and you're running. Changes take effect immediately.
- **Dynamic extensibility** — register new tools at runtime with `register_tool(ToolDef(...))`, install skill packs from git URLs, or wire in any MCP server.
- **Task dependency graph** — `TaskCreate` / `TaskUpdate` support `blocks` / `blocked_by` edges for structured multi-step planning (not available in Claude Code).
- **Two-layer context compression** — rule-based snip + AI summarization, configurable via `preserve_last_n_turns`.
- **Notebook editing** — `NotebookEdit` directly manipulates `.ipynb` JSON (replace/insert/delete cells) with no kernel required.
- **Diagnostics without LSP server** — `GetDiagnostics` chains pyright → mypy → flake8 → py_compile for Python and tsc/shellcheck for other languages, with zero configuration.
- **Offline voice input** — `/voice` records via `sounddevice`/`arecord`/SoX, transcribes with local `faster-whisper` (no API key, no subscription), and auto-submits. Keyterms from your git branch and project files boost coding-term accuracy.
- **Cloud session sync** — `/cloudsave` backs up conversations to private GitHub Gists with zero extra dependencies; restore any past session on any machine with `/cloudsave load <id>`.
- **SSJ Developer Mode** — `/ssj` opens a persistent power menu with 10 workflow shortcuts: Brainstorm → TODO → Worker pipeline, expert debate, code review, README generation, commit helper, and more. Stays open between actions; supports `/command` passthrough.
- **Telegram Bot Bridge** — `/telegram <token> <chat_id>` turns cheetahclaws into a Telegram bot: receive user messages, run the model, and send back responses — all from your phone. Slash commands pass through, and a typing indicator keeps the chat feeling live.
- **WeChat Bridge** — `/wechat login` authenticates with WeChat via a QR code scan (the same iLink Bot API used by the official WeixinClawBot / `openclaw-weixin` plugin), then starts a long-poll bridge. Slash command passthrough, interactive menu routing, typing indicator, session auto-recovery, and per-peer `context_token` management all work out of the box.
- **Slack Bridge** — `/slack <xoxb-token> <channel_id>` connects cheetahclaws to a Slack channel using the Slack Web API (stdlib only — no `slack_sdk` required). Polls `conversations.history` every 2 seconds; replies update an in-place "Thinking…" placeholder. Slash command passthrough, interactive menu routing, and auto-start on launch.
- **Worker command** — `/worker` auto-implements pending tasks from `brainstorm_outputs/todo_list.txt`, marks each one done after completion, and supports task selection by number (e.g. `1,4,6`).
- **Force quit** — 3× Ctrl+C within 2 seconds triggers immediate `os._exit(1)`, unblocking any frozen I/O.
- **Proactive background monitoring** — `/proactive 5m` activates a sentinel daemon that wakes the agent automatically after a period of inactivity, enabling continuous monitoring loops, scheduled checks, or trading bots without user prompts.
- **Rich Live streaming rendering** — When `rich` is installed, responses stream as live-updating Markdown in place (no duplicate raw text), with clean tool-call interleaving.
- **Native Ollama reasoning** — Local reasoning models (deepseek-r1, qwen3, gemma4) stream their `<think>` tokens directly to the terminal via `ThinkingChunk` events; enable with `/verbose` and `/thinking`.
- **Native Ollama vision** — `/image [prompt]` captures the clipboard and sends it to local vision models (llava, gemma4, llama3.2-vision) via Ollama's native image API. No cloud required.
- **Built-in Web UI** — `--web` launches a production-ready browser interface: multi-user accounts (bcrypt + JWT), SQLite-backed session history that survives restarts, rich Chat UI at `/chat` with streaming messages, tool cards, permission approval, sidebar session CRUD + search + markdown export, light/dark/system theme, settings panel with per-provider API keys. Full xterm.js PTY terminal at `/` keeps 100% CLI parity. Ops endpoints (`/health`, `/metrics`) + structured JSON logs + 21 pytest end-to-end tests. Nine tiny vanilla-JS modules under `web/static/js/` — no Node.js, no React, no build step. `cheetahclaws --web` auto-picks a free port if 8080 is taken.
- **Reliable multi-line paste** — Bracketed Paste Mode (`ESC[?2004h`) collects any pasted text — code blocks, multi-paragraph prompts, long diffs — as a single turn with zero latency and no blank-line artifacts.
- **Rich Tab completion** — Tab after `/` shows all commands with one-line descriptions and subcommand hints; subcommand Tab-complete works for `/mcp`, `/plugin`, `/tasks`, `/cloudsave`, and more.
- **Checkpoint & rewind** — `/checkpoint` lists all auto-snapshots of conversation + file state; `/checkpoint <id>` rewinds both files and history to any earlier point in the session.
- **Plan mode** — `/plan <desc>` (or the `EnterPlanMode` tool) puts Claude into a structured read-only analysis phase; only the plan file is writable. Claude writes a detailed plan, then `/plan done` restores full write permissions for implementation.

---

## CheetahClaws vs OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is another popular open-source AI assistant built on TypeScript/Node.js. The two projects have **different primary goals** — here is how they compare.

### At a glance

| Dimension | OpenClaw (TypeScript) | CheetahClaws (Python) |
|-----------|----------------------|---------------------|
| Language | TypeScript + Node.js | Python 3.8+ |
| Source files | ~10,349 TS/JS files | ~85 Python files |
| Lines of code | ~245K | ~12K |
| Primary focus | Personal life assistant across messaging channels | AI **coding** assistant / developer tool |
| Architecture | Always-on Gateway daemon + companion apps | Zero-install terminal REPL |
| Messaging channels | 20+ (WhatsApp · Telegram · Slack · Discord · Signal · iMessage · Matrix · WeChat · …) | Terminal + Telegram bridge + WeChat bridge (iLink) + Slack bridge (Web API) |
| Model providers | Multiple (cloud-first) | 7+ including full local support (Ollama · vLLM · LM Studio · …) |
| Local / offline models | Limited | Full — Ollama, vLLM, any OpenAI-compatible endpoint |
| Voice | Wake word · PTT · Talk Mode (macOS/iOS/Android) | Offline Whisper STT (local, no API key) |
| Code editing tools | Browser control, Canvas workspace | Read · Write · Edit · Bash · Glob · Grep · NotebookEdit · GetDiagnostics |
| Build step required | Yes (`pnpm install` + daemon setup) | No — `pip install` and run |
| Mobile companion | macOS menu bar + iOS/Android apps | — |
| Live Canvas / UI | Yes (A2UI agent-driven visual workspace) | — |
| MCP support | — | Yes (stdio/SSE/HTTP) |
| Runtime extensibility | Skills platform (bundled/managed/workspace) | `register_tool()` at runtime, MCP, git plugins, Markdown skills |
| Hackability | Large codebase (245K lines), harder to modify | ~12K lines — full agent loop visible in one file |

### Where OpenClaw wins

- **Omni-channel inbox** — connects to 20+ messaging platforms (WhatsApp, Signal, iMessage, Discord, Teams, Matrix, WeChat…); users interact from wherever they already are.
- **Always-on daemon** — Gateway runs as a background service (launchd/systemd); no terminal required for day-to-day use.
- **Mobile-first** — macOS menu bar, iOS Voice Wake / Talk Mode, Android camera/screen recording — feels like a native app, not a CLI tool.
- **Live Canvas** — agent-driven visual workspace rendered in the browser; supports A2UI push/eval/snapshot.
- **Browser automation** — dedicated Chrome/Chromium profile with snapshot, actions, and upload tools.
- **Production reliability** — versioned npm releases, comprehensive CI, onboarding wizard, `openclaw doctor` diagnostics.

### Where CheetahClaws wins

- **Coding toolset** — Read/Write/Edit/Bash/Glob/Grep/NotebookEdit/GetDiagnostics are purpose-built for software development; CheetahClaws understands diffs, file trees, and code structure.
- **True local model support** — full Ollama/vLLM/LM Studio integration with streaming, tool-calling, and vision — no cloud required.
- **8+ model providers** — switch between Claude, GPT-4o, Gemini, DeepSeek, Qwen, MiniMax, and local models with a single `--model` flag.
- **Hackable in minutes** — 12K lines of readable Python; the entire agent loop is in `agent.py`; extend with `register_tool()` at runtime without rebuilding.
- **Zero setup** — `pip install cheetahclaws` and run `cheetahclaws`; no daemon, no pairing, no onboarding wizard.
- **MCP support** — connect any MCP server (stdio/SSE/HTTP); tools auto-registered.
- **SSJ Developer Mode** — `/ssj` power menu chains Brainstorm → TODO → Worker → Debate in a persistent interactive session; automates entire dev workflows.
- **Offline voice** — `/voice` transcribes locally with `faster-whisper`; no subscription, no OAuth, works without internet.
- **Session cloud sync** — `/cloudsave` backs up full conversations to private GitHub Gists with zero extra dependencies.

### When to choose which

| If you want… | Use |
|---|---|
| A personal assistant you can message on WhatsApp/Signal/Discord | **OpenClaw** |
| An AI coding assistant in your terminal | **CheetahClaws** |
| Full offline / local model support | **CheetahClaws** |
| A mobile-friendly always-on experience | **OpenClaw** |
| To read and modify the source in an afternoon | **CheetahClaws** |
| Browser automation and a visual Canvas | **OpenClaw** |
| Multi-provider LLM switching without rebuilding | **CheetahClaws** |

---

### Key design differences

**Agent loop** — CheetahClaws uses a Python generator that `yield`s typed events (`TextChunk`, `ToolStart`, `ToolEnd`, `TurnDone`). The entire loop is visible in one file, making it easy to add hooks, custom renderers, or logging.

**Tool registration** — every tool is a `ToolDef(name, schema, func, read_only, concurrent_safe)` dataclass. Any module can call `register_tool()` at import time; MCP servers, plugins, and skills all use the same mechanism.

**Context compression**

| | Claude Code | CheetahClaws |
|-|-------------|-----------------|
| Trigger | Exact token count | `len / 3.5` estimate, fires at 70 % |
| Layer 1 | — | Snip: truncate old tool outputs (no API cost) |
| Layer 2 | AI summarization | AI summarization of older turns |
| Control | System-managed | `preserve_last_n_turns` parameter |

**Memory** — Claude Code's `extractMemories` service has the model proactively surface facts. CheetahClaws's `memory/` package is tool-driven: the model calls `MemorySave` explicitly, which is more predictable and auditable. Each memory now carries `confidence`, `source`, `last_used_at`, and `conflict_group` metadata; search re-ranks by confidence × recency; and `/memory consolidate` offers a manual consolidation pass without silently modifying memories in the background.

### Who should use CheetahClaws

- Developers who want to **use a local or non-Anthropic model** as their coding assistant.
- Researchers studying **how agentic coding assistants work** — the entire system fits in one screen.
- Teams who need a **hackable baseline** to add proprietary tools, custom permission policies, or specialised agent types.
- Anyone who wants Claude Code-style productivity **without a Node.js build chain**.

---

## Features

| Feature | Details |
|---|---|
| Multi-provider | Anthropic · OpenAI · Gemini · Kimi · Qwen · Zhipu · DeepSeek · MiniMax · Ollama · LM Studio · Custom endpoint |
| Interactive REPL | readline history, Tab-complete slash commands with descriptions + subcommand hints; Bracketed Paste Mode for reliable multi-line paste |
| Agent loop | Streaming API + automatic tool-use loop |
| 27 built-in tools | Read · Write · Edit · Bash · Glob · Grep · WebFetch · WebSearch · **NotebookEdit** · **GetDiagnostics** · MemorySave · MemoryDelete · MemorySearch · MemoryList · Agent · SendMessage · CheckAgentResult · ListAgentTasks · ListAgentTypes · Skill · SkillList · AskUserQuestion · TaskCreate/Update/Get/List · **SleepTimer** · **EnterPlanMode** · **ExitPlanMode** · *(MCP + plugin tools auto-added at startup)* |
| MCP integration | Connect any MCP server (stdio/SSE/HTTP), tools auto-registered and callable by Claude |
| Plugin system | Install/uninstall/enable/disable/update plugins from git URLs or local paths; multi-scope (user/project); recommendation engine |
| AskUserQuestion | Claude can pause and ask the user a clarifying question mid-task, with optional numbered choices |
| Task management | TaskCreate/Update/Get/List tools; sequential IDs; dependency edges; metadata; persisted to `.cheetahclaws/tasks.json`; `/tasks` REPL command |
| Diff view | Git-style red/green diff display for Edit and Write |
| Context compression | Auto-compact long conversations to stay within model limits |
| Persistent memory | Dual-scope memory (user + project) with 4 types, confidence/source metadata, conflict detection, recency-weighted search, `last_used_at` tracking, and `/memory consolidate` for auto-extraction |
| Multi-agent | Spawn typed sub-agents (coder/reviewer/researcher/…), git worktree isolation, background mode |
| Skills | Built-in `/commit` · `/review` + custom markdown skills with argument substitution and fork/inline execution |
| Plugin tools | Register custom tools via `tool_registry.py` |
| Permission system | `auto` / `accept-all` / `manual` / `plan` modes |
| Checkpoints | Auto-snapshot conversation + file state after each turn; `/checkpoint` to list, `/checkpoint <id>` to rewind; `/rewind` alias; 100-snapshot sliding window |
| Plan mode | `/plan <desc>` enters read-only analysis mode; Claude writes only to the plan file; `EnterPlanMode` / `ExitPlanMode` agent tools for autonomous planning |
| 36 slash commands | `/model` · `/config` · `/save` · `/cost` · `/memory` · `/skills` · `/agents` · `/voice` · `/proactive` · `/checkpoint` · `/plan` · `/compact` · `/status` · `/doctor` · … |
| Voice input | Record → transcribe → auto-submit. Backends: `sounddevice` / `arecord` / SoX + `faster-whisper` / `openai-whisper` / OpenAI API. Works fully offline. |
| Brainstorm | `/brainstorm [topic]` generates N expert personas suited to the topic (2–100, default 5, chosen interactively), runs an iterative debate, saves results to `brainstorm_outputs/`, and synthesizes a Master Plan + auto-generates `brainstorm_outputs/todo_list.txt`. |
| SSJ Developer Mode | `/ssj` opens a persistent interactive power menu with **15 shortcuts**: Brainstorm, TODO viewer, Worker, Expert Debate, Propose, Review, Readme, Commit, Scan, Promote, Video factory, TTS factory, Monitor, **Trading**, Agent. Stays open between actions; `/command` passthrough supported. |
| Trading agent | `/trading analyze <SYMBOL>` runs a full multi-agent pipeline: data collection → Bull/Bear researcher debate → research judge → risk management panel (aggressive/conservative/neutral) → portfolio manager final decision (BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL). `/trading backtest` runs strategy backtests with 4 built-in strategies. BM25 memory system learns from past trades. Supports US/HK/A-share stocks and 20+ cryptos. |
| Monitor | `/monitor` (no args → wizard) subscribes to AI-monitored topics on a schedule and pushes reports to Telegram/Slack/console. Topics: `ai_research` (arxiv), `stock_<TICKER>`, `crypto_<SYMBOL>`, `world_news` (Reuters/BBC/AP), `custom:<query>`. Schedules: 15m to weekly. Background scheduler daemon with `/monitor start/stop/status`. |
| Autonomous Agents | `/agent` (no args → wizard) launches autonomous background agent loops driven by Markdown task templates. 4 built-in templates: `research_assistant`, `auto_bug_fixer`, `paper_writer`, `auto_coder`. Iteration summaries pushed via bridge. Custom templates: drop a `.md` file into `~/.cheetahclaws/agent_templates/`. |
| Remote Control job queue | All three bridges (Telegram/Slack/WeChat) maintain a per-bridge FIFO job queue when the AI is busy. `!jobs` / `!j` — dashboard; `!job <id>` — detail; `!retry <id>` — re-run a failed job; `!cancel [id]` — stop current job. Tool step tracking with `on_tool_start`/`on_tool_end` hooks. Persistent log at `~/.cheetahclaws/jobs.json`. |
| Worker | `/worker [task#s]` reads `brainstorm_outputs/todo_list.txt`, implements each pending task with a dedicated model prompt, and marks it done (`- [x]`). Supports task selection (`/worker 1,4,6`), custom path (`--path`), and worker count limit (`--workers`). Detects and redirects accidental brainstorm `.md` paths. |
| Telegram bridge | `/telegram <token> <chat_id>` starts a bot bridge: receive messages from Telegram, run the model, and reply — all from your phone. Typing indicator, slash command passthrough (including interactive menus), and auto-start on launch if configured. |
| WeChat bridge | `/wechat login` authenticates via QR code scan (same as WeixinClawBot / openclaw-weixin plugin), then starts the iLink long-poll bridge. `context_token` echoed per peer, typing indicator, slash command passthrough, session expiry auto-recovery. Credentials saved for auto-start on next launch. |
| Slack bridge | `/slack <xoxb-token> <channel_id>` connects to a Slack channel via the Web API (no external packages). Polls `conversations.history` every 2 s; replies update an in-place "Thinking…" placeholder. Slash command passthrough, interactive menu routing, auth validation on start, auto-start on next launch. |
| Video factory | `/video [topic]` runs the full AI video pipeline: story generation (active model) → TTS narration (Edge/Gemini/ElevenLabs) → AI images (Gemini Web free or placeholders) → subtitle burn (Whisper) → FFmpeg assembly → final `.mp4`. 10 viral content niches, landscape or short format, zero-cost path available. |
| TTS factory | `/tts` interactive wizard: AI writes script (or paste your own) → synthesize to MP3 in any voice style (narrator, newsreader, storyteller, ASMR, motivational, documentary, children, podcast, meditation, custom). Engine auto-selects: Gemini TTS → ElevenLabs → Edge TTS (always-free). CJK text auto-switches to a matching voice. |
| Vision input | `/image` (or `/img`) captures the clipboard image and sends it to any vision-capable model — Ollama (`llava`, `gemma4`, `llama3.2-vision`) via native format, or cloud models (GPT-4o, Gemini 2.0 Flash, …) via OpenAI `image_url` multipart format. Requires `pip install cheetahclaws[vision]`; Linux also needs `xclip`. |
| Tmux integration | 11 tmux tools for direct terminal control: create sessions/windows/panes, send commands, capture output. Auto-detected; zero impact if tmux is absent. Enables long-running tasks that outlive Bash tool timeouts. Cross-platform (tmux on Unix, psmux on Windows). |
| Shell escape | Type `!command` in the REPL to execute any shell command directly without AI involvement (`!git status`, `!ls`, `!python --version`). Output prints inline. |
| Proactive monitoring | `/proactive [duration]` starts a background sentinel daemon; agent wakes automatically after inactivity, enabling continuous monitoring loops without user prompts |
| Force quit | 3× Ctrl+C within 2 seconds triggers `os._exit(1)` — kills the process immediately regardless of blocking I/O |
| Rich Live streaming | When `rich` is installed, responses render as live-updating Markdown in place. Auto-disabled in SSH sessions to prevent repeated output; override with `/config rich_live=false`. |
| Context injection | Auto-loads `CLAUDE.md`, git status, cwd, persistent memory |
| Session persistence | Autosave on exit to `daily/YYYY-MM-DD/` (per-day limit) + `history.json` (master, all sessions) + `session_latest.json` (/resume); sessions include `session_id` and `saved_at` metadata; `/load` grouped by date |
| Cloud sync | `/cloudsave` syncs sessions to private GitHub Gists; auto-sync on exit; load from cloud by Gist ID. No new dependencies (stdlib `urllib`). |
| Extended Thinking | Toggle on/off for Claude models; native `<think>` block streaming for local Ollama reasoning models (deepseek-r1, qwen3, gemma4) |
| Cost tracking | Token usage + estimated USD cost |
| Non-interactive mode | `--print` flag for scripting / CI |
| **Web UI** | `--web` opens the browser. Multi-user accounts (bcrypt + JWT), SQLite-persisted history, session CRUD + markdown export, light/dark/system theme, `/health` + `/metrics`, auto-picks a free port if 8080 is busy. `pip install 'cheetahclaws[web]'`. |

---

## Supported Models

### Closed-Source (API)

| Provider | Model | Context | Strengths | API Key Env |
|---|---|---|---|---|
| **Anthropic** | `claude-opus-4-6` | 200k | Most capable, best for complex reasoning | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-sonnet-4-6` | 200k | Balanced speed & quality | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-haiku-4-5-20251001` | 200k | Fast, cost-efficient | `ANTHROPIC_API_KEY` |
| **OpenAI** | `gpt-4o` | 128k | Strong multimodal & coding | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | 128k | Fast, cheap | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-4.1` | 128k | Latest GPT-4 generation | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-4.1-mini` | 128k | Fast GPT-4.1 | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-5` | 128k | Next-gen flagship | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-5-nano` | 128k | Fastest GPT-5 variant | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-5-mini` | 128k | Balanced GPT-5 variant | `OPENAI_API_KEY` |
| **OpenAI** | `o4-mini` | 200k | Fast reasoning | `OPENAI_API_KEY` |
| **OpenAI** | `o3` | 200k | Strong reasoning | `OPENAI_API_KEY` |
| **OpenAI** | `o3-mini` | 200k | Compact reasoning | `OPENAI_API_KEY` |
| **OpenAI** | `o1` | 200k | Advanced reasoning | `OPENAI_API_KEY` |
| **Google** | `gemini-2.5-pro-preview-03-25` | 1M | Long context, multimodal | `GEMINI_API_KEY` |
| **Google** | `gemini-2.0-flash` | 1M | Fast, large context | `GEMINI_API_KEY` |
| **Google** | `gemini-1.5-pro` | 2M | Largest context window | `GEMINI_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-8k` | 8k | Chinese & English | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-32k` | 32k | Chinese & English | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-128k` | 128k | Long context | `MOONSHOT_API_KEY` |
| **Alibaba (Qwen)** | `qwen-max` | 32k | Best Qwen quality | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-plus` | 128k | Balanced | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-turbo` | 1M | Fast, cheap | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwq-32b` | 32k | Strong reasoning | `DASHSCOPE_API_KEY` |
| **Zhipu (GLM)** | `glm-4-plus` | 128k | Best GLM quality | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4` | 128k | General purpose | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4-flash` | 128k | Free tier available | `ZHIPU_API_KEY` |
| **DeepSeek** | `deepseek-chat` | 64k | Strong coding | `DEEPSEEK_API_KEY` |
| **DeepSeek** | `deepseek-reasoner` | 64k | Chain-of-thought reasoning | `DEEPSEEK_API_KEY` |
| **MiniMax** | `MiniMax-Text-01` | 1M | Long context, strong reasoning | `MINIMAX_API_KEY` |
| **MiniMax** | `MiniMax-VL-01` | 1M | Vision + language | `MINIMAX_API_KEY` |
| **MiniMax** | `abab6.5s-chat` | 256k | Fast, cost-efficient | `MINIMAX_API_KEY` |
| **MiniMax** | `abab6.5-chat` | 256k | Balanced quality | `MINIMAX_API_KEY` |

### Open-Source (Local via Ollama)

| Model | Size | Strengths | Pull Command |
|---|---|---|---|
| `llama3.3` | 70B | General purpose, strong reasoning | `ollama pull llama3.3` |
| `llama3.2` | 3B / 11B | Lightweight | `ollama pull llama3.2` |
| `qwen2.5-coder` | 7B / 32B | **Best for coding tasks** | `ollama pull qwen2.5-coder` |
| `qwen2.5` | 7B / 72B | Chinese & English | `ollama pull qwen2.5` |
| `deepseek-r1` | 7B–70B | Reasoning, math | `ollama pull deepseek-r1` |
| `deepseek-coder-v2` | 16B | Coding | `ollama pull deepseek-coder-v2` |
| `mistral` | 7B | Fast, efficient | `ollama pull mistral` |
| `mixtral` | 8x7B | Strong MoE model | `ollama pull mixtral` |
| `phi4` | 14B | Microsoft, strong reasoning | `ollama pull phi4` |
| `gemma3` | 4B / 12B / 27B | Google open model | `ollama pull gemma3` |
| `codellama` | 7B / 34B | Code generation | `ollama pull codellama` |
| `llava` | 7B / 13B | **Vision** — image understanding | `ollama pull llava` |
| `llama3.2-vision` | 11B | **Vision** — multimodal reasoning | `ollama pull llama3.2-vision` |

> **Note:** Tool calling requires a model that supports function calling. Recommended local models: `qwen2.5-coder`, `llama3.3`, `mistral`, `phi4`.

> **OpenAI newer models (gpt-5 / o3 / o4 family):** These models require `max_completion_tokens` instead of the legacy `max_tokens` parameter. CheetahClaws handles this automatically — no configuration needed.

> **Reasoning models:** `deepseek-r1`, `qwen3`, and `gemma4` stream native `<think>` blocks. Enable with `/verbose` and `/thinking` to see thoughts in the terminal. Note: models fed a large system prompt (like cheetahclaws's 25 tool schemas) may suppress their thinking phase to avoid breaking the expected JSON format — this is model behavior, not a bug.

---

## Installation

### Quick Install (one command)

```bash
curl -fsSL https://raw.githubusercontent.com/SafeRL-Lab/cheetahclaws/main/scripts/install.sh | bash
```

Works on **Linux, macOS, WSL2, and Android (Termux)**. The installer handles everything: checks Python 3.10+, clones the repo, installs via pip, and adds `cheetahclaws` to your PATH.

After installation:

```bash
source ~/.zshrc     # macOS (zsh)
# or: source ~/.bashrc   # Linux (bash)
cheetahclaws        # start chatting!
```

First run will guide you through setup (pick provider, set API key). Or run `cheetahclaws --setup` anytime.

> **Windows:** Native Windows is not supported. Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and run the command above inside WSL.
>
> **Android / Termux:** The installer auto-detects Termux and skips incompatible optional dependencies. Manual install: `pkg install python git && pip install cheetahclaws`.

---

### Alternative: install with `pip`

```bash
git clone https://github.com/SafeRL-Lab/cheetahclaws.git
cd cheetahclaws
pip install .
```

After that, `cheetahclaws` is available as a global command:

```bash
cheetahclaws                        # start REPL
cheetahclaws --model gpt-4o         # choose a model
cheetahclaws -p "explain this"      # non-interactive
cheetahclaws --setup                # re-run setup wizard
```

To update after pulling new code:

```bash
cd cheetahclaws
git pull
pip install .
```

#### Optional extras

```bash
pip install ".[voice]"              # voice input (sounddevice)
pip install ".[vision]"             # clipboard image capture (Pillow)
pip install ".[autosuggest]"        # typing-time slash command autosuggest (prompt_toolkit)
pip install ".[browser]"            # headless browser for JS-rendered pages (playwright)
pip install ".[files]"              # PDF + Excel reading (pymupdf, openpyxl)
pip install ".[ocr]"                # image OCR (pytesseract, Pillow)
pip install ".[trading]"            # trading agent (yfinance, rank-bm25)
pip install ".[all]"                # everything above
```

> **Note:** After installing `[browser]`, run `playwright install chromium` to download the browser binary.
---

### Alternative: install with `uv`

[uv](https://docs.astral.sh/uv/) installs `cheetahclaws` into an isolated environment and puts it on your PATH:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install with all optional dependencies (voice, vision, autosuggest, browser, files, OCR, trading etc.)
git clone https://github.com/SafeRL-Lab/cheetahclaws.git
cd cheetahclaws
uv tool install ".[all]"
```

Prefer a minimal install? Use `uv tool install .` (core only) and add extras later, e.g. `uv tool install ".[voice,vision,autosuggest]" --reinstall`.

To update: `uv tool install ".[all]" --reinstall`

To uninstall: `uv tool uninstall cheetahclaws`

---

### Alternative: run directly from source (no install)

```bash
git clone https://github.com/SafeRL-Lab/cheetahclaws.git
cd cheetahclaws
pip install -r requirements.txt
python cheetahclaws.py
```

This is useful for development — changes take effect immediately without reinstalling.

---

## Usage: Closed-Source API Models

### Anthropic Claude

Get your API key at [console.anthropic.com](https://console.anthropic.com).

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Default model (claude-opus-4-6)
cheetahclaws

# Choose a specific model
cheetahclaws --model claude-sonnet-4-6
cheetahclaws --model claude-haiku-4-5-20251001

# Enable Extended Thinking
cheetahclaws --model claude-opus-4-6 --thinking --verbose
```

### OpenAI GPT

Get your API key at [platform.openai.com](https://platform.openai.com).

```bash
export OPENAI_API_KEY=sk-...

cheetahclaws --model gpt-4o
cheetahclaws --model gpt-4o-mini
cheetahclaws --model gpt-4.1-mini
cheetahclaws --model o3-mini
```

### Google Gemini

Get your API key at [aistudio.google.com](https://aistudio.google.com).

```bash
export GEMINI_API_KEY=AIza...

cheetahclaws --model gemini/gemini-3-flash-preview
cheetahclaws --model gemini/gemini-3.1-pro-preview
```

### Kimi (Moonshot AI)

Get your API key at [platform.moonshot.cn](https://platform.moonshot.cn).

```bash
export MOONSHOT_API_KEY=sk-...

cheetahclaws --model kimi/moonshot-v1-32k
cheetahclaws --model kimi/moonshot-v1-128k
```

### Qwen (Alibaba DashScope)

Get your API key at [dashscope.aliyun.com](https://dashscope.aliyun.com).

```bash
export DASHSCOPE_API_KEY=sk-...

cheetahclaws --model qwen/Qwen3.5-Plus
cheetahclaws --model qwen/Qwen3-MAX
cheetahclaws --model qwen/Qwen3.5-Flash
```

### Zhipu GLM

Get your API key at [open.bigmodel.cn](https://open.bigmodel.cn).

```bash
export ZHIPU_API_KEY=...

cheetahclaws --model zhipu/glm-4-plus
cheetahclaws --model zhipu/glm-4-flash   # free tier
```

### DeepSeek

Get your API key at [platform.deepseek.com](https://platform.deepseek.com).

```bash
export DEEPSEEK_API_KEY=sk-...

cheetahclaws --model deepseek/deepseek-chat
cheetahclaws --model deepseek/deepseek-reasoner
```

### MiniMax

Get your API key at [platform.minimaxi.chat](https://platform.minimaxi.chat).

```bash
export MINIMAX_API_KEY=...

cheetahclaws --model minimax/MiniMax-Text-01
cheetahclaws --model minimax/MiniMax-VL-01
cheetahclaws --model minimax/abab6.5s-chat
```

---

## Usage: Open-Source Models (Local)

### Option A — Ollama (Recommended)

Ollama runs models locally with zero configuration. No API key required.

**Step 1: Install Ollama**

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

**Step 2: Pull a model**

```bash
# Best for coding (recommended)
ollama pull qwen2.5-coder          # 4.7 GB (7B)
ollama pull qwen2.5-coder:32b      # 19 GB (32B)

# General purpose
ollama pull llama3.3               # 42 GB (70B)
ollama pull llama3.2               # 2.0 GB (3B)

# Reasoning
ollama pull deepseek-r1            # 4.7 GB (7B)
ollama pull deepseek-r1:32b        # 19 GB (32B)

# Other
ollama pull phi4                   # 9.1 GB (14B)
ollama pull mistral                # 4.1 GB (7B)
```

**Step 3: Start Ollama server** (runs automatically on macOS; on Linux run manually)

```bash
ollama serve     # starts on http://localhost:11434
```

**Step 4: Run cheetahclaws**

```bash
cheetahclaws --model ollama/qwen2.5-coder
cheetahclaws --model ollama/llama3.3
cheetahclaws --model ollama/deepseek-r1
```

Or

```bash
python cheetahclaws.py --model ollama/qwen2.5-coder
python cheetahclaws.py --model ollama/llama3.3
python cheetahclaws.py --model ollama/deepseek-r1
python cheetahclaws.py --model ollama/qwen3.5:35b
```

**List your locally available models:**

```bash
ollama list
```

Then use any model from the list:

```bash
cheetahclaws --model ollama/<model-name>
```

---

### Option B — LM Studio

LM Studio provides a GUI to download and run models, with a built-in OpenAI-compatible server.

**Step 1:** Download [LM Studio](https://lmstudio.ai) and install it.

**Step 2:** Search and download a model inside LM Studio (GGUF format).

**Step 3:** Go to **Local Server** tab → click **Start Server** (default port: 1234).

**Step 4:**

```bash
cheetahclaws --model lmstudio/<model-name>
# e.g.:
cheetahclaws --model lmstudio/phi-4-GGUF
cheetahclaws --model lmstudio/qwen2.5-coder-7b
```

The model name should match what LM Studio shows in the server status bar.

---

### Option C — vLLM / Self-Hosted OpenAI-Compatible Server

For self-hosted inference servers (vLLM, TGI, llama.cpp server, etc.) that expose an OpenAI-compatible API:

Quick Start for option C:
Step 1: Start vllm:
 ```
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --host 0.0.0.0 \
      --port 8000 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes
```


 Step 2: Start cheetahclaws：
```
  export CUSTOM_BASE_URL=http://localhost:8000/v1
  export CUSTOM_API_KEY=none
  cheetahclaws --model custom/Qwen/Qwen2.5-Coder-7B-Instruct
```


```bash
# Example: vLLM serving Qwen2.5-Coder-32B
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Then run cheetahclaws pointing to your server:
cheetahclaws
```

Inside the REPL:

```
/config custom_base_url=http://localhost:8000/v1
/config custom_api_key=token-abc123    # skip if no auth
/model custom/Qwen2.5-Coder-32B-Instruct
```

Or set via environment:

```bash
export CUSTOM_BASE_URL=http://localhost:8000/v1
export CUSTOM_API_KEY=token-abc123

cheetahclaws --model custom/Qwen2.5-Coder-32B-Instruct
```

For a remote GPU server:

```bash
/config custom_base_url=http://192.168.1.100:8000/v1
/model custom/your-model-name
```

---

## Model Name Format

Three equivalent formats are supported:

```bash
# 1. Auto-detect by prefix (works for well-known models)
cheetahclaws --model gpt-4o
cheetahclaws --model gemini-2.0-flash
cheetahclaws --model deepseek-chat

# 2. Explicit provider prefix with slash
cheetahclaws --model ollama/qwen2.5-coder
cheetahclaws --model kimi/moonshot-v1-128k

# 3. Explicit provider prefix with colon (also works)
cheetahclaws --model kimi:moonshot-v1-32k
cheetahclaws --model qwen:qwen-max
```

**Auto-detection rules:**

| Model prefix | Detected provider |
|---|---|
| `claude-` | anthropic |
| `gpt-`, `o1`, `o3` | openai |
| `gemini-` | gemini |
| `moonshot-`, `kimi-` | kimi |
| `qwen`, `qwq-` | qwen |
| `glm-` | zhipu |
| `deepseek-` | deepseek |
| `MiniMax-`, `minimax-`, `abab` | minimax |
| `llama`, `mistral`, `phi`, `gemma`, `mixtral`, `codellama` | ollama |

---

## Trading Agent

CheetahClaws includes a built-in AI-powered trading analysis and backtesting module. Install trading dependencies:

```bash
pip install "cheetahclaws[trading]"
```

### Multi-agent analysis

```bash
/trading analyze NVDA
```

Runs a 5-phase pipeline: **data collection** (technical indicators, fundamentals, news) → **Bull/Bear researcher debate** → **research judge** recommendation → **risk management panel** (aggressive / conservative / neutral) → **portfolio manager** final decision with a 5-tier rating: `BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL`.

Each agent uses BM25 memory to recall similar past situations and learns from outcomes via post-trade reflection.

### Backtesting

```bash
/trading backtest AAPL dual_ma           # single strategy
/trading backtest TSLA                   # AI picks best strategy
```

4 built-in strategies: `dual_ma` (SMA crossover), `rsi_mean_reversion`, `bollinger_breakout`, `macd_crossover`. Engines for US/HK equities and crypto. Reports Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor.

### SSJ integration

`/ssj` → **14. 📈 Trading** opens a guided sub-menu:

| Option | Action |
|---|---|
| a. Quick Analyze | Full multi-agent analysis for any symbol |
| b. Backtest | Pick strategy or compare all 4 |
| c. Price Check | Current price + key metrics |
| d. Indicators | 11 technical indicators report |
| e. Trading Bot | Autonomous multi-symbol analysis |
| f. History | Past trading decisions |
| g. Memory | Trading memory status |

### Supported markets

US stocks (`AAPL`), HK stocks (`0700.HK`), A-shares (`000001.SZ`), crypto (`BTC`, `ETH`, + 18 more). Data sources with automatic fallback chains — no API keys required.

> **Full guide:** [docs/guides/trading.md](docs/guides/trading.md)

---

## Web UI

A production-ready browser interface with real user accounts, SQLite-backed session history, and ops endpoints — bundled Python stdlib HTTP server plus nine small vanilla-JS modules, no Node.js / React / build step.

### Install and start

```bash
pip install 'cheetahclaws[web]'              # pulls sqlalchemy + passlib + PyJWT

cheetahclaws --web                           # auto-picks a free port (tries 8080 first)
cheetahclaws --web --port 9000               # bind exactly :9000 (fails loudly if taken)
cheetahclaws --web --host 0.0.0.0            # open to the local network
cheetahclaws --web --no-auth                 # skip login (localhost dev only)
```

On first visit to `http://localhost:<port>/chat`, the UI routes you to a **registration form** — the first account becomes admin. Subsequent visits show **Sign in**. Credentials: bcrypt-hashed password + 7-day JWT cookie (`ccjwt`, HttpOnly, SameSite=Strict). The JWT signing key is persisted to `~/.cheetahclaws/web_secret` so logins survive restarts.

### Chat UI (`/chat`)

| Feature | Details |
|---------|---------|
| **Streaming chat** | WebSocket for live prompts + SSE for long-running slash commands |
| **Persistent history** | Every session + message lives in SQLite (`~/.cheetahclaws/web.db`). Server restart does not lose state. |
| **Sidebar session management** | Title auto-titled from first user message, relative time ("12m ago"), message count, busy dot, client-side search, right-click menu (Rename / Export Markdown / Delete) |
| **Cross-user isolation** | Each user only sees their own sessions — enforced at DB query and in-memory cache |
| **Tool cards** | Collapsible cards show tool name, inputs, outputs, status (running / done / denied) |
| **Permission approval** | Inline Allow / Deny buttons |
| **45+ slash commands** | `/status`, `/model`, `/brainstorm`, `/ssj`, `/plan`, `/telegram`, `/wechat`, `/slack`, `/voice`, `/image`, etc. |
| **Settings panel** | Model picker (11 providers), permission mode, thinking/verbose toggles, per-provider API key entry, quick-action buttons |
| **Theme** | Light default, `@media (prefers-color-scheme: dark)` follows the OS automatically. Toggle cycles **system → light → dark → system**; choice stored in localStorage, no flash-of-wrong-theme on first paint |
| **Feature dashboard** | Welcome screen with 4×6 clickable cards — Core, Agent Features, Session & Memory, Multi-Model, Development Tools, Bridges, Multi-Modal Media |
| **Export as Markdown** | `GET /api/sessions/{id}/export` downloads the conversation with all tool calls |
| **Favicon** | Leaping-cheetah icon served at `/favicon.ico` and `/static/favicon.png` |

### PTY Terminal (`/`)

Full xterm.js terminal — still there, still 100% CLI parity. Uses the same one-time generated password (printed on startup) — separate from the chat JWT flow.

### API shape

```
Browser ──→ /chat                ──→ 9 JS modules load from /static/js/*.js
        ──→ /api/auth/login      ──→ bcrypt + JWT cookie
        ──→ /api/prompt (POST)   ──→ persists to SQLite, fans events out
        ──→ /api/events (WS)     ──→ real-time text_chunk / tool_* / permission_*
        ──→ /api/sessions/*      ──→ list / get / rename / delete / export

        ──→ /                     ──→ xterm.js PTY (password-gated)
        ──→ /health               ──→ { ok, db, uptime_s }        (unauthenticated)
        ──→ /metrics              ──→ Prometheus text              (unauthenticated)
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/auth/bootstrap` | GET | Any users registered yet? |
| `/api/auth/register` | POST | Create user (first one is admin) |
| `/api/auth/login` | POST | Verify bcrypt + issue JWT cookie |
| `/api/auth/logout` | POST | Clear cookie |
| `/api/auth/whoami` | GET | Current user |
| `/api/prompt` | POST | Submit prompt / slash command (inline JSON or SSE for long commands) |
| `/api/events` | WS | Structured event stream for a session |
| `/api/approve` | POST | Respond to a permission request |
| `/api/sessions` | GET | List this user's sessions |
| `/api/sessions/{id}` | GET / PATCH / DELETE | Detail / rename / remove |
| `/api/sessions/{id}/export` | GET | Download conversation as Markdown |
| `/api/config` | GET / PATCH | Read or update session config |
| `/api/models` | GET | Providers + models + API-key status |
| `/health` | GET | Liveness + DB probe |
| `/metrics` | GET | Prometheus counters (`requests_total`, `auth_logins_failed`, `users_total`, ...) |

### Observability

- **Structured logs** — one JSON line per HTTP response on stderr, e.g.
  ```json
  {"ts":1776368300.054,"level":"info","logger":"web.server","msg":"req","method":"POST","path":"/api/prompt","status":200,"dur_ms":650,"user_id":1}
  ```
  Tune with `CHEETAHCLAWS_LOG_LEVEL=DEBUG|INFO|WARNING`.
- **Metrics** — point Prometheus at `/metrics`. Counters increment inside `_send_http` and the auth routes.
- **Tests** — `pytest tests/test_web_api.py` runs 21 end-to-end HTTP tests against a real server in ~5 seconds (no mocks, real SQLite, real bcrypt, real JWT).

> **Full guide:** [docs/guides/web-ui.md](docs/guides/web-ui.md)

---

## Documentation

Detailed guides have been moved to [`docs/guides/`](docs/guides/) to keep this README focused. Click any link below:

| Guide | What's Inside |
|-------|---------------|
| [**Web UI**](docs/guides/web-ui.md) | Chat UI, PTY terminal, API endpoints, settings panel, model switching, dark/light theme, SSE streaming, session management, authentication |
| [**Reference**](docs/guides/reference.md) | CLI, 36+ commands, 33 built-in tools (incl. WebBrowse, ReadEmail, SendEmail, ReadPDF, ReadImage, ReadSpreadsheet), session search, auxiliary model, error classification, prompt injection detection, tool cache, parallel tools |
| [**Extensions**](docs/guides/extensions.md) | Memory system, Skills, Sub-Agents, MCP servers, Plugin system, Monitor subscriptions, Autonomous Agents |
| [**Bridges**](docs/guides/bridges.md) | Telegram, WeChat, Slack setup and remote control from your phone |
| [**Voice & Video**](docs/guides/voice-and-video.md) | Voice input (offline Whisper), Video Content Factory, TTS Content Factory |
| [**Trading**](docs/guides/trading.md) | Multi-agent analysis (Bull/Bear debate, Risk panel, PM), backtesting (4 strategies, equity + crypto engines), BM25 memory, data fallback chains, SSJ integration |
| [**Advanced**](docs/guides/advanced.md) | Brainstorm, SSJ Developer Mode, Tmux, Proactive monitoring, Checkpoints, Plan mode, Session management, Cloud sync |
| [**Recipes**](docs/guides/recipes.md) | 12 step-by-step examples: code review, Telegram remote control, autonomous research, bug fix, brainstorm, session search, browse web pages, email, PDF/Excel analysis, and more |
| [**Plugin Authoring**](docs/guides/plugin-authoring.md) | Build your own plugin: tools, commands, skills, MCP servers, publishing checklist |
| [**Example Plugin**](examples/example-plugin/) | Copy-and-edit starter template with working tools, commands, and skills |
| [**Contributing**](CONTRIBUTING.md) | Project structure, architecture guide, PR checklist |

---

## Quick Reference

```
cheetahclaws [OPTIONS] [PROMPT]

Options:
  -p, --print          Non-interactive: run prompt and exit
  -m, --model MODEL    Override model (e.g. gpt-4o, ollama/llama3.3)
  --accept-all         Auto-approve all operations (no permission prompts)
  --verbose            Show thinking blocks and per-turn token counts
  --thinking           Enable Extended Thinking (Claude only)
  --web                Start web server (Chat UI + PTY terminal in browser)
  --port PORT          Web server port (default: 8080)
  --host HOST          Web server host (default: 127.0.0.1)
  --no-auth            Disable web password (local use only)
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

# Web UI (browser-based chat + terminal)
cheetahclaws --web
cheetahclaws --web --port 8008 --no-auth
```

See [Reference Guide](docs/guides/reference.md) for the full list of 36+ slash commands, tool descriptions, and configuration options.

---

## Contributing

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for project architecture, code conventions, and PR checklist.

Quick start for contributors:

```bash
git clone https://github.com/SafeRL-Lab/cheetahclaws.git
cd cheetahclaws
pip install -r requirements.txt
pip install pytest
python -m pytest tests/ -x -q       # 341+ tests should pass
python cheetahclaws.py               # run the REPL
```

Building a plugin? See the [Plugin Authoring Guide](docs/guides/plugin-authoring.md) and the [example plugin template](examples/example-plugin/).

---

## FAQ

**Q: How do I add an MCP server?**

Option 1 — via REPL (stdio server):
```
/mcp add git uvx mcp-server-git
```

Option 2 — create `.mcp.json` in your project:
```json
{
  "mcpServers": {
    "git": {"type": "stdio", "command": "uvx", "args": ["mcp-server-git"]}
  }
}
```

Then run `/mcp reload` or restart. Use `/mcp` to check connection status.

**Q: An MCP server is showing an error. How do I debug it?**

```
/mcp                    # shows error message per server
/mcp reload git         # try reconnecting
```

If the server uses stdio, make sure the command is in your `$PATH`:
```bash
which uvx               # should print a path
uvx mcp-server-git      # run manually to see errors
```

**Q: Can I use MCP servers that require authentication?**

For HTTP/SSE servers with a Bearer token:
```json
{
  "mcpServers": {
    "my-api": {
      "type": "sse",
      "url": "https://myserver.example.com/sse",
      "headers": {"Authorization": "Bearer sk-my-token"}
    }
  }
}
```

For stdio servers with env-based auth:
```json
{
  "mcpServers": {
    "brave": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-brave-search"],
      "env": {"BRAVE_API_KEY": "your-key"}
    }
  }
}
```

**Q: Tool calls don't work with my local Ollama model.**

Not all models support function calling. Use one of the recommended tool-calling models: `qwen2.5-coder`, `llama3.3`, `mistral`, or `phi4`.

```bash
ollama pull qwen2.5-coder
cheetahclaws --model ollama/qwen2.5-coder
```

**Q: How do I connect to a remote GPU server running vLLM?**

```
/config custom_base_url=http://your-server-ip:8000/v1
/config custom_api_key=your-token
/model custom/your-model-name
```

**Q: How do I check my API cost?**

```
/cost

  Input tokens:  3,421
  Output tokens:   892
  Est. cost:     $0.0648 USD
```

**Q: Can I use multiple API keys in the same session?**

Yes. Set all the keys you need upfront (via env vars or `/config`). Then switch models freely — each call uses the key for the active provider.

**Q: How do I make a model available across all projects?**

Add keys to `~/.bashrc` or `~/.zshrc`. Set the default model in `~/.cheetahclaws/config.json`:

```json
{ "model": "claude-sonnet-4-6" }
```

**Q: Qwen / Zhipu returns garbled text.**

Ensure your `DASHSCOPE_API_KEY` / `ZHIPU_API_KEY` is correct and the account has sufficient quota. Both providers use UTF-8 and handle Chinese well.

**Q: Can I pipe input to cheetahclaws?**

```bash
echo "Explain this file" | cheetahclaws --print --accept-all
cat error.log | cheetahclaws -p "What is causing this error?"
```

**Q: How do I run it as a CLI tool from anywhere?**

Use `uv tool install` — it creates an isolated environment and puts `cheetahclaws` on your PATH:

```bash
cd cheetahclaws
uv tool install ".[all]"
```

After that, just run `cheetahclaws` from any directory. To update after pulling changes, run `uv tool install ".[all]" --reinstall`. For a minimal install, use `uv tool install .` and add extras as needed.

**Q: How do I set up voice input?**

```bash
# Minimal setup (local, offline, no API key):
pip install sounddevice faster-whisper numpy

# Then in the REPL:
/voice status          # verify backends are detected
/voice                 # speak your prompt
```

On first use, `faster-whisper` downloads the `base` model (~150 MB) automatically.
Use a larger model for better accuracy: `export NANO_CLAUDE_WHISPER_MODEL=small`

**Q: Voice input transcribes my words wrong (misses coding terms).**

The keyterm booster already injects coding vocabulary from your git branch and project files.
For persistent domain terms, put them in a `.cheetahclaws/voice_keyterms.txt` file (one term per line) — this is checked automatically on each recording.

**Q: Can I use voice input in Chinese / Japanese / other languages?**

Yes. Set the language before recording:

```
/voice lang zh    # Mandarin Chinese
/voice lang ja    # Japanese
/voice lang auto  # reset to auto-detect (default)
```

Whisper supports 99 languages. `auto` detection works well but explicit codes improve accuracy for short utterances.



## Citation
If you find the repository useful, please cite the study
``` Bash
@article{cheetahclaws2026,
  title={CheetahClaws: An Extensible, Python-Native Agent System for Autonomous Multi-Model Workflows},
  author={CheetahClaws Team},
  journal={github},
  year={2026}
}
```
