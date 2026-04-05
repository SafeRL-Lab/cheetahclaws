English | [中文](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.CN.MD) | [Français](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.FR.MD) | [한국어](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.KO.MD) | [日本語](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.JP.MD) | [Deutsch](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.DE.MD) | [Português](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.ES.MD)

<div align="center">
  <a href="[https://github.com/SafeRL-Lab/Robust-Gymnasium](https://github.com/SafeRL-Lab/nano-claude-code)">
    <img src="https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/logo-v1.png" alt="Logo" width="280"> 
  </a>

  
<h1 align="center" style="font-size: 30px;"><strong><em>Nano Claude Code</em></strong>: A Fast, Easy-to-Use Python Reimplementation of Claude Code Supporting Any Model</h1>
<p align="center">
    <a href="https://github.com/chauncygu/collection-claude-code-source-code">The newest source of Claude Code</a>
    ·
    <a href="https://github.com/SafeRL-Lab/nano-claude-code/issues">Issue</a>
  ·
    <a href="https://deepwiki.com/SafeRL-Lab/nano-claude-code">Brief Intro</a>
  
  </p>
</div>

 <div align=center>
 <img src="https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> </center>
 </div>

---



## 🔥🔥🔥 News (Pacific Time)
- 09:34 AM, Apr 05, 2026: **v3.05.3** — Added GitHub Gist cloud sync: `/cloudsave setup <token>` to configure, `/cloudsave` to upload the current session to a private Gist, `/cloudsave auto on` to sync automatically on `/exit`, `/cloudsave list` to browse cloud sessions, and `/cloudsave load <id>` to restore from the cloud. Uses stdlib `urllib` — no new dependencies. Also added version number (e.g., `v3.05.2`) in the startup banner: The startup banner now displays the current version number (v3.05.2) in green, making it easy to identify which version is running at a glance.
- 08:58 AM, Apr 05, 2026: **v3.05.2** — Introduced `/proactive [duration]` command: a background daemon thread watches for user inactivity and automatically wakes the agent up after the specified interval (e.g. `/proactive 5m`), enabling continuous monitoring loops without user intervention. `/proactive` with no args now shows current status; `/proactive off` disables it explicitly. Proactive polling state is stored in `config` (no module-level globals). Watcher exceptions are logged via `traceback` instead of silently swallowed. Also fixed duplicated output in Rich-enabled terminals by buffering text during streaming and rendering Markdown once via `rich.live.Live` — updates happen in-place for a true streaming Markdown experience. 
- 10:51 PM, Apr 04, 2026: **v3.05_fix04** — Fixed a crash on `/model` and config save commands caused by the newly introduced `_run_query_callback` being serialized to JSON; also added `SleepTimer` usage    
  guidance to the system prompt so the agent knows when to invoke background timers proactively.
- 10:28 PM, Apr 04, 2026: **v3.05_fix03** — Added a native `SleepTimer` tool that lets the agent schedule background timers and autonomously wake itself up after a delay — no user prompt required. Paired with a `threading.Lock` to prevent output collisions when background and foreground calls overlap. Also includes cross-platform fixes: Windows ANSI color support, CRLF-aware Edit tool matching, an interactive numbered menu for `/load`, native Ollama streaming via `/api/chat`, and auto-capping `max_tokens` per provider to prevent API errors. 
- 08:31 PM, Apr 04, 2026: **v3.05_fix** — Autosave + `/resume`: session is automatically saved to `mr_sessions/session_latest.json` on `/exit`, `/quit`, `Ctrl+C`, and `Ctrl+D`. Run `/resume` to restore the last session instantly, or `/resume <file>` to load a specific file from `mr_sessions/`, and better support for api and local Ollama models (specifically gemma4), along with Windows compatibility enhancements, session management UX improvements, and cross-platform reliability fixes for the Edit tool.
- 00:41 AM, Apr 04, 2026: **v3.05** — Voice input (`voice/` package): `sounddevice` → `arecord` → SoX recording backends, `faster-whisper` → `openai-whisper` → OpenAI API STT backends. Smart keyterm extraction from git branch + project name + recent files passed as Whisper `initial_prompt` for coding-domain accuracy. `/voice`, `/voice status`, `/voice lang <code>` REPL commands. Works fully offline with no API key. 29 new tests (**~11.6K** lines of Python).
- 10:29 PM, Apr 03, 2026: **v3.04** — Expanded tool coverage: `NotebookEdit` (edit Jupyter `.ipynb` cells — replace/insert/delete with full JSON round-trip) and `GetDiagnostics` (LSP-style diagnostics via pyright/mypy/flake8/tsc/shellcheck). Also fixed a pre-existing schema-index bug in `_register_builtins` by switching to name-based lookup (**~10.5K** lines of Python).
- 06:00 PM, Apr 03, 2026: **v3.03** — Task management system (`task/` package): `TaskCreate` / `TaskUpdate` / `TaskGet` / `TaskList` tools with sequential IDs, dependency edges (blocks/blocked_by), metadata, persistence to `.nano_claude/tasks.json`, thread-safe store, `/tasks` REPL command, 37 new tests (**~9500** lines of Python).
- 02:50 PM, Apr 03, 2026: **v3.02** — Plugin system (`plugin/` package): install/uninstall/enable/disable/update via `/plugin` CLI, recommendation engine (keyword+tag matching), multi-scope (user/project), git-based marketplace. `AskUserQuestion` tool: interactive mid-task user prompts with numbered options and free-text input (**~8500** lines of Python).
- 10:00 AM, Apr 03, 2026: **v3.01** — MCP (Model Context Protocol) support: `mcp/` package, stdio + SSE + HTTP transports, auto tool discovery, `/mcp` command, 34 new tests (**~7000** lines of Python).
- 12:20 PM, Apr 02, 2026: **v3.0** — Multi-agent packages (`multi_agent/`), memory package (`memory/`), skill package (`skill/`) with built-in skills, argument substitution, fork/inline execution, AI memory search, git worktree isolation, agent type definitions (**~5000** lines of Python), see [update](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/update_readme_v3.0.md).
- 10:00 AM, Apr 02, 2026: **v2.0** — Context compression, memory, sub-agents, skills, diff view, tool plugin system (**~3400** lines of Python Code).
- 01:47 PM, Apr 01, 2026: Support VLLM inference (**~2000** lines of Python Code).
- 11:30 AM, Apr 01, 2026: Support more **closed-source** models and **open-source models**: Claude, GPT, Gemini, Kimi, Qwen, Zhipu, DeepSeek, and local open-source models via Ollama or any OpenAI-compatible endpoint. (**~1700** lines of Python Code).
- 09:50 AM, Apr 01, 2026: Support more **closed-source** models: Claude, GPT, Gemini. (**~1300** lines of Python Code).
- 08:23 AM, Apr 01, 2026: Release the initial version of Nano Claude Code (**~900 lines** of Python Code).

---

# Nano Claude Code

Nano Claude Code: **A Lightweight** and **Easy-to-Use** Python Reimplementation of Claude Code **Supporting Any Model**, such as Claude, GPT, Gemini, Kimi, Qwen, Zhipu, DeepSeek, and local open-source models via Ollama or any OpenAI-compatible endpoint.

---

## Content
  * [Why Nano Claude Code](#why-nano-claude-code)
  * [Features](#features)
  * [Supported Models](#supported-models)
  * [Installation](#installation)
  * [Usage: Closed-Source API Models](#usage--closed-source-api-models)
  * [Usage: Open-Source Models (Local)](#usage--open-source-models--local-)
  * [Model Name Format](#model-name-format)
  * [CLI Reference](#cli-reference)
  * [Slash Commands (REPL)](#slash-commands--repl-)
  * [Configuring API Keys](#configuring-api-keys)
  * [Permission System](#permission-system)
  * [Built-in Tools](#built-in-tools)
  * [Memory](#memory)
  * [Skills](#skills)
  * [Sub-Agents](#sub-agents)
  * [MCP (Model Context Protocol)](#mcp-model-context-protocol)
  * [Plugin System](#plugin-system)
  * [AskUserQuestion Tool](#askuserquestion-tool)
  * [Task Management](#task-management)
  * [Voice Input](#voice-input)
  * [Proactive Background Monitoring](#proactive-background-monitoring)
  * [Context Compression](#context-compression)
  * [Diff View](#diff-view)
  * [CLAUDE.md Support](#claudemd-support)
  * [Session Management](#session-management)
  * [Cloud Sync (GitHub Gist)](#cloud-sync-github-gist)
  * [Project Structure](#project-structure)
  * [FAQ](#faq)




## Why Nano Claude Code

Claude Code is a powerful, production-grade AI coding assistant — but its source code is a compiled, 12 MB TypeScript/Node.js bundle (~1,300 files, ~283K lines). It is tightly coupled to the Anthropic API, hard to modify, and impossible to run against a local or alternative model.

**Nano Claude Code** reimplements the same core loop in ~10K lines of readable Python, keeping everything you need and dropping what you don't. See here for more detailed analysis (Nano Claude code v3.03), [English version](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/comparison_claude_code_vs_nano_v3.03_en.md) and [Chinese version](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/comparison_claude_code_vs_nano_v3.03_cn.md)

### At a glance

| Dimension | Claude Code (TypeScript) | Nano Claude Code (Python) |
|-----------|--------------------------|---------------------------|
| Language | TypeScript + React/Ink | Python 3.8+ |
| Source files | ~1,332 TS/TSX files | 51 Python files |
| Lines of code | ~283K | ~11.6K |
| Built-in tools | 44+ | 25 |
| Slash commands | 88 | 20 |
| Voice input | Proprietary Anthropic WebSocket (OAuth required) | Local Whisper / OpenAI API — works offline, no subscription |
| Model providers | Anthropic only | 7+ (Anthropic · OpenAI · Gemini · Kimi · Qwen · DeepSeek · Ollama · …) |
| Local models | No | Yes — Ollama, LM Studio, vLLM, any OpenAI-compatible endpoint |
| Build step required | Yes (Bun + esbuild) | No — run directly with `python nano_claude.py` |
| Runtime extensibility | Closed (compile-time) | Open — `register_tool()` at runtime, Markdown skills, git plugins |
| Task dependency graph | No | Yes — `blocks` / `blocked_by` edges in `task/` package |

### Where Claude Code wins

- **UI quality** — React/Ink component tree with streaming rendering, fine-grained diff visualization, and dialog systems.
- **Tool breadth** — 44 tools including `RemoteTrigger`, `EnterWorktree`, and more UI-integrated tools.
- **Enterprise features** — MDM-managed config, team permission sync, OAuth, keychain storage, GrowthBook feature flags.
- **AI-driven memory extraction** — `extractMemories` service proactively extracts knowledge from conversations without explicit tool calls.
- **Production reliability** — single distributable `cli.js`, comprehensive test coverage, version-locked releases.

### Where Nano Claude Code wins

- **Multi-provider** — switch between Claude, GPT-4o, Gemini 2.5 Pro, DeepSeek, Qwen, or a local Llama model with `--model` or `/model` — no recompile needed.
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
- **Proactive background monitoring** — `/proactive 5m` activates a sentinel daemon that wakes the agent automatically after a period of inactivity, enabling continuous monitoring loops, scheduled checks, or trading bots without user prompts.
- **Rich Live streaming rendering** — When `rich` is installed, responses stream as live-updating Markdown in place (no duplicate raw text), with clean tool-call interleaving.

### Key design differences

**Agent loop** — Nano uses a Python generator that `yield`s typed events (`TextChunk`, `ToolStart`, `ToolEnd`, `TurnDone`). The entire loop is visible in one file, making it easy to add hooks, custom renderers, or logging.

**Tool registration** — every tool is a `ToolDef(name, schema, func, read_only, concurrent_safe)` dataclass. Any module can call `register_tool()` at import time; MCP servers, plugins, and skills all use the same mechanism.

**Context compression**

| | Claude Code | Nano Claude Code |
|-|-------------|-----------------|
| Trigger | Exact token count | `len / 3.5` estimate, fires at 70 % |
| Layer 1 | — | Snip: truncate old tool outputs (no API cost) |
| Layer 2 | AI summarization | AI summarization of older turns |
| Control | System-managed | `preserve_last_n_turns` parameter |

**Memory** — Claude Code's `extractMemories` service has the model proactively surface facts. Nano's `memory/` package is tool-driven: the model calls `MemorySave` explicitly, which is more predictable and auditable.

### Who should use Nano Claude Code

- Developers who want to **use a local or non-Anthropic model** as their coding assistant.
- Researchers studying **how agentic coding assistants work** — the entire system fits in one screen.
- Teams who need a **hackable baseline** to add proprietary tools, custom permission policies, or specialised agent types.
- Anyone who wants Claude Code-style productivity **without a Node.js build chain**.

---

## Features

| Feature | Details |
|---|---|
| Multi-provider | Anthropic · OpenAI · Gemini · Kimi · Qwen · Zhipu · DeepSeek · Ollama · LM Studio · Custom endpoint |
| Interactive REPL | readline history, Tab-complete slash commands |
| Agent loop | Streaming API + automatic tool-use loop |
| 25 built-in tools | Read · Write · Edit · Bash · Glob · Grep · WebFetch · WebSearch · **NotebookEdit** · **GetDiagnostics** · MemorySave · MemoryDelete · MemorySearch · MemoryList · Agent · SendMessage · CheckAgentResult · ListAgentTasks · ListAgentTypes · Skill · SkillList · AskUserQuestion · TaskCreate/Update/Get/List · **SleepTimer** · *(MCP + plugin tools auto-added at startup)* |
| MCP integration | Connect any MCP server (stdio/SSE/HTTP), tools auto-registered and callable by Claude |
| Plugin system | Install/uninstall/enable/disable/update plugins from git URLs or local paths; multi-scope (user/project); recommendation engine |
| AskUserQuestion | Claude can pause and ask the user a clarifying question mid-task, with optional numbered choices |
| Task management | TaskCreate/Update/Get/List tools; sequential IDs; dependency edges; metadata; persisted to `.nano_claude/tasks.json`; `/tasks` REPL command |
| Diff view | Git-style red/green diff display for Edit and Write |
| Context compression | Auto-compact long conversations to stay within model limits |
| Persistent memory | Dual-scope memory (user + project) with 4 types, AI search, staleness warnings |
| Multi-agent | Spawn typed sub-agents (coder/reviewer/researcher/…), git worktree isolation, background mode |
| Skills | Built-in `/commit` · `/review` + custom markdown skills with argument substitution and fork/inline execution |
| Plugin tools | Register custom tools via `tool_registry.py` |
| Permission system | `auto` / `accept-all` / `manual` modes |
| 19 slash commands | `/model` · `/config` · `/save` · `/cost` · `/memory` · `/skills` · `/agents` · `/voice` · `/proactive` · … |
| Voice input | Record → transcribe → auto-submit. Backends: `sounddevice` / `arecord` / SoX + `faster-whisper` / `openai-whisper` / OpenAI API. Works fully offline. |
| Proactive monitoring | `/proactive [duration]` starts a background sentinel daemon; agent wakes automatically after inactivity, enabling continuous monitoring loops without user prompts |
| Rich Live streaming | When `rich` is installed, responses render as live-updating Markdown in place — no duplicate raw text, clean tool-call interleaving |
| Context injection | Auto-loads `CLAUDE.md`, git status, cwd, persistent memory |
| Session persistence | Save / load conversations to `~/.nano_claude/sessions/`; **autosave on exit** + `/resume` to instantly restore last session |
| Cloud sync | `/cloudsave` syncs sessions to private GitHub Gists; auto-sync on exit; load from cloud by Gist ID. No new dependencies (stdlib `urllib`). |
| Extended Thinking | Toggle on/off (Claude models only) |
| Cost tracking | Token usage + estimated USD cost |
| Non-interactive mode | `--print` flag for scripting / CI |

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
| **OpenAI** | `o3-mini` | 200k | Strong reasoning | `OPENAI_API_KEY` |
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

> **Note:** Tool calling requires a model that supports function calling. Recommended local models: `qwen2.5-coder`, `llama3.3`, `mistral`, `phi4`.

---

## Installation

```bash
git clone <repo-url>
cd nano_claude_code

pip install -r requirements.txt
# or manually:
pip install anthropic openai httpx rich sounddevice
```

---

## Usage: Closed-Source API Models

### Anthropic Claude

Get your API key at [console.anthropic.com](https://console.anthropic.com).

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Default model (claude-opus-4-6)
python nano_claude.py

# Choose a specific model
python nano_claude.py --model claude-sonnet-4-6
python nano_claude.py --model claude-haiku-4-5-20251001

# Enable Extended Thinking
python nano_claude.py --model claude-opus-4-6 --thinking --verbose
```

### OpenAI GPT

Get your API key at [platform.openai.com](https://platform.openai.com).

```bash
export OPENAI_API_KEY=sk-...

python nano_claude.py --model gpt-4o
python nano_claude.py --model gpt-4o-mini
python nano_claude.py --model gpt-4.1-mini
python nano_claude.py --model o3-mini
```

### Google Gemini

Get your API key at [aistudio.google.com](https://aistudio.google.com).

```bash
export GEMINI_API_KEY=AIza...

python nano_claude.py --model gemini/gemini-2.0-flash
python nano_claude.py --model gemini/gemini-1.5-pro
python nano_claude.py --model gemini/gemini-2.5-pro-preview-03-25
```

### Kimi (Moonshot AI)

Get your API key at [platform.moonshot.cn](https://platform.moonshot.cn).

```bash
export MOONSHOT_API_KEY=sk-...

python nano_claude.py --model kimi/moonshot-v1-32k
python nano_claude.py --model kimi/moonshot-v1-128k
```

### Qwen (Alibaba DashScope)

Get your API key at [dashscope.aliyun.com](https://dashscope.aliyun.com).

```bash
export DASHSCOPE_API_KEY=sk-...

python nano_claude.py --model qwen/Qwen3.5-Plus
python nano_claude.py --model qwen/Qwen3-MAX
python nano_claude.py --model qwen/Qwen3.5-Flash
```

### Zhipu GLM

Get your API key at [open.bigmodel.cn](https://open.bigmodel.cn).

```bash
export ZHIPU_API_KEY=...

python nano_claude.py --model zhipu/glm-4-plus
python nano_claude.py --model zhipu/glm-4-flash   # free tier
```

### DeepSeek

Get your API key at [platform.deepseek.com](https://platform.deepseek.com).

```bash
export DEEPSEEK_API_KEY=sk-...

python nano_claude.py --model deepseek/deepseek-chat
python nano_claude.py --model deepseek/deepseek-reasoner
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

**Step 4: Run nano claude**

```bash
python nano_claude.py --model ollama/qwen2.5-coder
python nano_claude.py --model ollama/llama3.3
python nano_claude.py --model ollama/deepseek-r1
```

**List your locally available models:**

```bash
ollama list
```

Then use any model from the list:

```bash
python nano_claude.py --model ollama/<model-name>
```

---

### Option B — LM Studio

LM Studio provides a GUI to download and run models, with a built-in OpenAI-compatible server.

**Step 1:** Download [LM Studio](https://lmstudio.ai) and install it.

**Step 2:** Search and download a model inside LM Studio (GGUF format).

**Step 3:** Go to **Local Server** tab → click **Start Server** (default port: 1234).

**Step 4:**

```bash
python nano_claude.py --model lmstudio/<model-name>
# e.g.:
python nano_claude.py --model lmstudio/phi-4-GGUF
python nano_claude.py --model lmstudio/qwen2.5-coder-7b
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


 Step 2: Start nano claude：
```
  export CUSTOM_BASE_URL=http://localhost:8000/v1
  export CUSTOM_API_KEY=none
  python nano_claude.py --model custom/Qwen/Qwen2.5-Coder-7B-Instruct
```


```bash
# Example: vLLM serving Qwen2.5-Coder-32B
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000

# Then run nano claude pointing to your server:
python nano_claude.py
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

python nano_claude.py --model custom/Qwen2.5-Coder-32B-Instruct
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
python nano_claude.py --model gpt-4o
python nano_claude.py --model gemini-2.0-flash
python nano_claude.py --model deepseek-chat

# 2. Explicit provider prefix with slash
python nano_claude.py --model ollama/qwen2.5-coder
python nano_claude.py --model kimi/moonshot-v1-128k

# 3. Explicit provider prefix with colon (also works)
python nano_claude.py --model kimi:moonshot-v1-32k
python nano_claude.py --model qwen:qwen-max
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
| `llama`, `mistral`, `phi`, `gemma`, `mixtral`, `codellama` | ollama |

---

## CLI Reference

```
python nano_claude.py [OPTIONS] [PROMPT]

Options:
  -p, --print          Non-interactive: run prompt and exit
  -m, --model MODEL    Override model (e.g. gpt-4o, ollama/llama3.3)
  --accept-all         Auto-approve all operations (no permission prompts)
  --verbose            Show thinking blocks and per-turn token counts
  --thinking           Enable Extended Thinking (Claude only)
  --version            Print version and exit
  -h, --help           Show help
```

**Examples:**

```bash
# Interactive REPL with default model
python nano_claude.py

# Switch model at startup
python nano_claude.py --model gpt-4o
python nano_claude.py -m ollama/deepseek-r1:32b

# Non-interactive / scripting
python nano_claude.py --print "Write a Python fibonacci function"
python nano_claude.py -p "Explain the Rust borrow checker in 3 sentences" -m gemini/gemini-2.0-flash

# CI / automation (no permission prompts)
python nano_claude.py --accept-all --print "Initialize a Python project with pyproject.toml"

# Debug mode (see tokens + thinking)
python nano_claude.py --thinking --verbose
```

---

## Slash Commands (REPL)

Type `/` and press **Tab** to autocomplete.

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/model` | Show current model + list all available models |
| `/model <name>` | Switch model (takes effect immediately) |
| `/config` | Show all current config values |
| `/config key=value` | Set a config value (persisted to disk) |
| `/save` | Save session (auto-named by timestamp) |
| `/save <filename>` | Save session to named file |
| `/load` | List all saved sessions |
| `/load <filename>` | Load a saved session |
| `/resume` | Restore the last auto-saved session (`mr_sessions/session_latest.json`) |
| `/resume <filename>` | Load a specific file from `mr_sessions/` (or absolute path) |
| `/history` | Print full conversation history |
| `/context` | Show message count and token estimate |
| `/cost` | Show token usage and estimated USD cost |
| `/verbose` | Toggle verbose mode (tokens + thinking) |
| `/thinking` | Toggle Extended Thinking (Claude only) |
| `/permissions` | Show current permission mode |
| `/permissions <mode>` | Set permission mode: `auto` / `accept-all` / `manual` |
| `/cwd` | Show current working directory |
| `/cwd <path>` | Change working directory |
| `/memory` | List all persistent memories |
| `/memory <query>` | Search memories by keyword |
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
| `/proactive` | Show current proactive polling status (ON/OFF and interval) |
| `/proactive <duration>` | Enable background sentinel polling (e.g. `5m`, `30s`, `1h`) |
| `/proactive off` | Disable background polling |
| `/cloudsave setup <token>` | Configure GitHub Personal Access Token for Gist sync |
| `/cloudsave` | Upload current session to a private GitHub Gist |
| `/cloudsave push [desc]` | Upload with an optional description |
| `/cloudsave auto on\|off` | Toggle auto-upload on `/exit` |
| `/cloudsave list` | List your nano-claude-code Gists |
| `/cloudsave load <gist_id>` | Download and restore a session from Gist |
| `/exit` / `/quit` | Exit |

**Switching models inside a session:**

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
```

### Method 2: Set Inside the REPL (persisted)

```
/config anthropic_api_key=sk-ant-...
/config openai_api_key=sk-...
/config gemini_api_key=AIza...
/config kimi_api_key=sk-...
/config qwen_api_key=sk-...
/config zhipu_api_key=...
/config deepseek_api_key=sk-...
```

Keys are saved to `~/.nano_claude/config.json` and loaded automatically on next launch.

### Method 3: Edit the Config File Directly

```json
// ~/.nano_claude/config.json
{
  "model": "qwen/qwen-max",
  "max_tokens": 8192,
  "permission_mode": "auto",
  "verbose": false,
  "thinking": false,
  "qwen_api_key": "sk-...",
  "kimi_api_key": "sk-...",
  "deepseek_api_key": "sk-..."
}
```

---

## Permission System

| Mode | Behavior |
|---|---|
| `auto` (default) | Read-only operations always allowed. Prompts before Bash commands and file writes. |
| `accept-all` | Never prompts. All operations proceed automatically. |
| `manual` | Prompts before every single operation, including reads. |

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

## Memory

The model can remember things across conversations using the built-in memory system.

**How it works:** Memories are stored as markdown files. There are two scopes:
- **User scope** (`~/.nano_claude/memory/`) — follows you across all projects
- **Project scope** (`.nano_claude/memory/` in cwd) — specific to the current repo

A `MEMORY.md` index (≤ 200 lines / 25 KB) is auto-rebuilt on every save or delete and injected into the system prompt so Claude always has an overview.

**Memory types:**

| Type | Use for |
|---|---|
| `user` | Your role, preferences, background |
| `feedback` | How you want the model to behave |
| `project` | Ongoing work, deadlines, decisions |
| `reference` | Links to external resources |

**Memory file format** (`~/.nano_claude/memory/coding_style.md`):
```markdown
---
name: coding style
description: Python formatting preferences
type: feedback
created: 2026-04-02
---
Prefer 4-space indentation and full type hints in all Python code.
**Why:** user explicitly stated this preference.
**How to apply:** apply to every Python file written or edited.
```

**Example interaction:**

```
You: Remember that I prefer 4-space indentation and type hints in all Python code.
AI: [calls MemorySave] Memory saved: coding_style [feedback/user]

You: /memory
  [feedback/user] coding_style (today): Python formatting preferences

You: /memory python
  [feedback/user] coding_style: Prefers 4-space indent and type hints in Python
```

**Staleness warnings:** Memories older than 1 day get a freshness note in `/memory` output so you know when to review or update them.

**AI-ranked search:** `MemorySearch(query="...", use_ai=true)` uses the model to rank results by relevance rather than simple keyword matching.

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
mkdir -p ~/.nano_claude/skills
```

Create `~/.nano_claude/skills/deploy.md`:

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
./.nano_claude/skills/     # project-level (overrides user-level)
~/.nano_claude/skills/     # user-level
```

---

## Sub-Agents

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

**Custom agent types** — create `~/.nano_claude/agents/myagent.md`:
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

Place a `.mcp.json` file in your project directory **or** edit `~/.nano_claude/mcp.json` for user-wide servers.

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
    }
  }
}
```

Config priority: `.mcp.json` (project) overrides `~/.nano_claude/mcp.json` (user) by server name.

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
/mcp                          # list servers + their tools + connection status
/mcp reload                   # reconnect all servers, refresh tool list
/mcp reload git               # reconnect a single server
/mcp add myserver uvx mcp-server-x   # add stdio server
/mcp remove myserver          # remove from user config
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

The `plugin/` package lets you extend nano-claude-code with additional tools, skills, and MCP servers from git repositories or local directories.

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
| user (default) | `~/.nano_claude/plugins/` | `~/.nano_claude/plugins.json` |
| project | `.nano_claude/plugins/` | `.nano_claude/plugins.json` |

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

Tasks are saved to `.nano_claude/tasks.json` in the current working directory after every mutation and reloaded on first access.

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

## Voice Input

Nano Claude Code v3.05 adds a fully offline voice-to-prompt pipeline. Speak your request — it is transcribed and submitted as if you had typed it.

### Quick start

```bash
# 1. Install a recording backend (choose one)
pip install sounddevice        # recommended: cross-platform, no extra binary
# sudo apt install alsa-utils  # Linux arecord fallback
# sudo apt install sox         # SoX rec fallback

# 2. Install a local STT backend (recommended — works offline, no API key)
pip install faster-whisper numpy

# 3. Start Nano Claude Code and speak
python nano_claude.py
[myproject] ❯ /voice
  🎙  Listening… (speak now, auto-stops on silence, Ctrl+C to cancel)
  🎙  ████
✓  Transcribed: "fix the authentication bug in user.py"
[auto-submitting…]
```

### STT backends (tried in order)

| Backend | Install | Notes |
|---|---|---|
| `faster-whisper` | `pip install faster-whisper` | **Recommended** — local, offline, fastest, GPU optional |
| `openai-whisper` | `pip install openai-whisper` | Local, offline, original OpenAI model |
| OpenAI Whisper API | set `OPENAI_API_KEY` | Cloud, requires internet + API key |

Override the Whisper model size with `NANO_CLAUDE_WHISPER_MODEL` (default: `base`):

```bash
export NANO_CLAUDE_WHISPER_MODEL=small   # better accuracy, slower
export NANO_CLAUDE_WHISPER_MODEL=tiny    # fastest, lightest
```

### Recording backends (tried in order)

| Backend | Install | Notes |
|---|---|---|
| `sounddevice` | `pip install sounddevice` | **Recommended** — cross-platform, Python-native |
| `arecord` | `sudo apt install alsa-utils` | Linux ALSA, no pip needed |
| `sox rec` | `sudo apt install sox` / `brew install sox` | Built-in silence detection |

### Keyterm boosting

Before each recording, Nano extracts coding vocabulary from:
- **Git branch** (e.g. `feat/voice-input` → "feat", "voice", "input")
- **Project root name** (e.g. "nano-claude-code")
- **Recent source file stems** (e.g. `authentication_handler.py` → "authentication", "handler")
- **Global coding terms**: `MCP`, `grep`, `TypeScript`, `OAuth`, `regex`, `gRPC`, …

These are passed as Whisper's `initial_prompt` so the STT engine prefers correct spellings of coding terms.

### Commands

| Command | Description |
|---|---|
| `/voice` | Record voice and auto-submit the transcript as your next prompt |
| `/voice status` | Show which recording and STT backends are available |
| `/voice lang <code>` | Set transcription language (`en`, `zh`, `ja`, `de`, `fr`, … default: `auto`) |

### How it compares to Claude Code

| | Claude Code | Nano Claude Code v3.05 |
|---|---|---|
| STT service | Anthropic private WebSocket (`voice_stream`) | `faster-whisper` / `openai-whisper` / OpenAI API |
| Requires Anthropic OAuth | Yes | **No** |
| Works offline | No | **Yes** (with local Whisper) |
| Keyterm hints | Deepgram `keyterms` param | Whisper `initial_prompt` (git + files + vocab) |
| Language support | Server-allowlisted codes | Any language Whisper supports |

---

## Proactive Background Monitoring

Nano Claude Code v3.05.2 adds a **sentinel daemon** that automatically wakes the agent after a configurable period of inactivity — no user prompt required. This enables use cases like continuous log monitoring, market script polling, or scheduled code checks.

### Quick start

```
[myproject] ❯ /proactive 5m
Proactive background polling: ON  (triggering every 300s of inactivity)

[myproject] ❯ keep monitoring the build log and alert me if errors appear

╭─ Claude ● ─────────────────────────
│ Understood. I'll check the build log each time I wake up.

[Background Event Triggered]
╭─ Claude ● ─────────────────────────
│ ⚙ Bash(tail -50 build.log)
│ ✓ → Build failed: ImportError in auth.py line 42
│ **Action needed:** fix the import before the next CI run.
```

### Commands

| Command | Description |
|---|---|
| `/proactive` | Show current status (ON/OFF and interval) |
| `/proactive 5m` | Enable — trigger every 5 minutes of inactivity |
| `/proactive 30s` | Enable — trigger every 30 seconds |
| `/proactive 1h` | Enable — trigger every hour |
| `/proactive off` | Disable sentinel polling |

Duration suffix: `s` = seconds, `m` = minutes, `h` = hours. Plain integer = seconds.

### How it works

- A background daemon thread starts when the REPL launches (paused by default).
- The daemon checks elapsed time since the last user or agent interaction every second.
- When the inactivity threshold is reached, it calls the agent with a wake-up prompt.
- The `threading.Lock` used by the main agent loop ensures wake-ups never interrupt an active session — they queue and fire after the current turn completes.
- Watcher exceptions are logged via `traceback` so failures are visible and debuggable.

### Complements SleepTimer

| | `SleepTimer` | `/proactive` |
|---|---|---|
| Who initiates | The agent | The user |
| Trigger | After a fixed delay from now | After N seconds of inactivity |
| Use case | "Check back in 10 minutes" | "Keep watching until I stop typing" |

---

## Context Compression

Long conversations are automatically compressed to stay within the model's context window.

**Two layers:**

1. **Snip** — Old tool outputs (file reads, bash results) are truncated after a few turns. Fast, no API cost.
2. **Auto-compact** — When token usage exceeds 70% of the context limit, older messages are summarized by the model into a concise recap.

This happens transparently. You don't need to do anything.

---

## Diff View

When the model edits or overwrites a file, you see a git-style diff:

```diff
  Changes applied to config.py:

--- a/config.py
+++ b/config.py
@@ -12,7 +12,7 @@
     "model": "claude-opus-4-6",
-    "max_tokens": 8192,
+    "max_tokens": 16384,
     "permission_mode": "auto",
```

Green lines = added, red lines = removed. New file creations show a summary instead.

---

## CLAUDE.md Support

Place a `CLAUDE.md` file in your project to give the model persistent context about your codebase. Nano Claude automatically finds and injects it into the system prompt.

```
~/.claude/CLAUDE.md          # Global — applies to all projects
/your/project/CLAUDE.md      # Project-level — found by walking up from cwd
```

**Example `CLAUDE.md`:**

```markdown
# Project: FastAPI Backend

## Stack
- Python 3.12, FastAPI, PostgreSQL, SQLAlchemy 2.0, Alembic
- Tests: pytest, coverage target 90%

## Conventions
- Format with black, lint with ruff
- Full type annotations required
- New endpoints must have corresponding tests

## Important Notes
- Never hard-code credentials — use environment variables
- Do not modify existing Alembic migration files
- The `staging` branch deploys automatically to staging on push
```

---

## Session Management

### Manual save / load

```bash
# Inside REPL:
/save                          # auto-name: session_20260401_143022.json
/save debug_auth_bug           # named save

/load                          # list all saved sessions
/load debug_auth_bug           # resume a session
/load session_20260401_143022.json
```

Sessions are stored as JSON in `~/.nano_claude/sessions/`.

### Autosave + resume (v3.05_fix)

Every time you exit — via `/exit`, `/quit`, `Ctrl+C`, or `Ctrl+D` — the current session is automatically saved to:

```
~/.nano_claude/sessions/mr_sessions/session_latest.json
```

To continue where you left off, simply run `/resume` at the start of the next session:

```bash
python nano_claude.py
[myproject] ❯ /resume
✓  Session loaded from …/mr_sessions/session_latest.json (42 messages)
```

You can also resume a specific file:

```bash
/resume session_latest.json          # loads from mr_sessions/
/resume /absolute/path/to/file.json  # loads from absolute path
```

---

## Cloud Sync (GitHub Gist)

Nano Claude Code v3.05.3 adds optional cloud backup of conversation sessions via **GitHub Gist**. Sessions are stored as private Gists (JSON), browsable in the GitHub UI. No extra dependencies — uses Python's stdlib `urllib`.

### Setup (one-time)

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens) → **Generate new token (classic)**
2. Enable the **`gist`** scope
3. Copy the token and run:

```
[myproject] ❯ /cloudsave setup ghp_xxxxxxxxxxxxxxxxxxxx
✓ GitHub token saved (logged in as: Chauncygu). Cloud sync is ready.
```

### Upload a session

```
[myproject] ❯ /cloudsave
Uploading session to GitHub Gist…
✓ Session uploaded → https://gist.github.com/abc123def456
```

Add an optional description:

```
[myproject] ❯ /cloudsave push auth refactor debug session
```

### Auto-sync on exit

```
[myproject] ❯ /cloudsave auto on
✓ Auto cloud-sync ON — session will be uploaded to Gist on /exit.
```

From that point on, every `/exit` or `/quit` automatically uploads the session before closing.

### Browse and restore

```
[myproject] ❯ /cloudsave list
  Found 3 session(s):
  abc123de…  2026-04-05 11:02  auth refactor debug session
  7f9e12ab…  2026-04-04 22:18  proactive monitoring test
  3b4c5d6e…  2026-04-04 18:31

[myproject] ❯ /cloudsave load abc123de...full-gist-id...
✓ Session loaded from Gist (42 messages).
```

### Commands reference

| Command | Description |
|---|---|
| `/cloudsave setup <token>` | Save GitHub token (needs `gist` scope) |
| `/cloudsave` | Upload current session to a new or existing Gist |
| `/cloudsave push [desc]` | Upload with optional description |
| `/cloudsave auto on\|off` | Toggle auto-upload on exit |
| `/cloudsave list` | List all nano-claude-code Gists |
| `/cloudsave load <gist_id>` | Download and restore a session |

---

## Project Structure

```
nano_claude_code/
├── nano_claude.py        # Entry point: REPL + slash commands + diff rendering + Rich Live streaming + proactive sentinel daemon
├── agent.py              # Agent loop: streaming, tool dispatch, compaction
├── providers.py          # Multi-provider: Anthropic, OpenAI-compat streaming
├── tools.py              # Core tools (Read/Write/Edit/Bash/Glob/Grep/Web/NotebookEdit/GetDiagnostics) + registry wiring
├── tool_registry.py      # Tool plugin registry: register, lookup, execute
├── compaction.py         # Context compression: snip + auto-summarize
├── context.py            # System prompt builder: CLAUDE.md + git + memory
├── config.py             # Config load/save/defaults
├── cloudsave.py          # GitHub Gist cloud sync (upload/download/list sessions)
│
├── multi_agent/          # Multi-agent package
│   ├── __init__.py       # Re-exports
│   ├── subagent.py       # AgentDefinition, SubAgentManager, worktree helpers
│   └── tools.py          # Agent, SendMessage, CheckAgentResult, ListAgentTasks, ListAgentTypes
├── subagent.py           # Backward-compat shim → multi_agent/
│
├── memory/               # Memory package
│   ├── __init__.py       # Re-exports
│   ├── types.py          # MEMORY_TYPES and format guidance
│   ├── store.py          # save/load/delete/search, MEMORY.md index rebuilding
│   ├── scan.py           # MemoryHeader, age/freshness helpers
│   ├── context.py        # get_memory_context(), truncation, AI search
│   └── tools.py          # MemorySave, MemoryDelete, MemorySearch, MemoryList
├── memory.py             # Backward-compat shim → memory/
│
├── skill/                # Skill package
│   ├── __init__.py       # Re-exports; imports builtin to register built-ins
│   ├── loader.py         # SkillDef, parse, load_skills, find_skill, substitute_arguments
│   ├── builtin.py        # Built-in skills: /commit, /review
│   ├── executor.py       # execute_skill(): inline or forked sub-agent
│   └── tools.py          # Skill, SkillList
├── skills.py             # Backward-compat shim → skill/
│
├── mcp/                  # MCP (Model Context Protocol) package
│   ├── __init__.py       # Re-exports
│   ├── types.py          # MCPServerConfig, MCPTool, MCPServerState, JSON-RPC helpers
│   ├── client.py         # StdioTransport, HttpTransport, MCPClient, MCPManager
│   ├── config.py         # Load .mcp.json (project) + ~/.nano_claude/mcp.json (user)
│   └── tools.py          # Auto-discover + register MCP tools into tool_registry
│
├── voice/                # Voice input package (v3.05)
│   ├── __init__.py       # Public API: check_voice_deps, voice_input
│   ├── recorder.py       # Audio capture: sounddevice → arecord → sox rec
│   ├── stt.py            # STT: faster-whisper → openai-whisper → OpenAI API
│   └── keyterms.py       # Coding-domain vocab from git branch + project files
│
└── tests/                # 239+ unit tests
    ├── test_mcp.py
    ├── test_memory.py
    ├── test_skills.py
    ├── test_subagent.py
    ├── test_tool_registry.py
    ├── test_compaction.py
    ├── test_diff_view.py
    └── test_voice.py      # 29 voice tests (no hardware required)
```

> **For developers:** Each feature package (`multi_agent/`, `memory/`, `skill/`, `mcp/`, `voice/`) is self-contained. Add custom tools by calling `register_tool(ToolDef(...))` from any module imported by `tools.py`.

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
python nano_claude.py --model ollama/qwen2.5-coder
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

Add keys to `~/.bashrc` or `~/.zshrc`. Set the default model in `~/.nano_claude/config.json`:

```json
{ "model": "claude-sonnet-4-6" }
```

**Q: Qwen / Zhipu returns garbled text.**

Ensure your `DASHSCOPE_API_KEY` / `ZHIPU_API_KEY` is correct and the account has sufficient quota. Both providers use UTF-8 and handle Chinese well.

**Q: Can I pipe input to nano claude?**

```bash
echo "Explain this file" | python nano_claude.py --print --accept-all
cat error.log | python nano_claude.py -p "What is causing this error?"
```

**Q: How do I run it as a CLI tool from anywhere?**

```bash
# Add an alias to ~/.bashrc or ~/.zshrc
alias nc='python /path/to/nano_claude_code/nano_claude.py'

# Or install as a script
pip install -e .   # if setup.py exists
```

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
For persistent domain terms, put them in a `.nano_claude/voice_keyterms.txt` file (one term per line) — this is checked automatically on each recording.

**Q: Can I use voice input in Chinese / Japanese / other languages?**

Yes. Set the language before recording:

```
/voice lang zh    # Mandarin Chinese
/voice lang ja    # Japanese
/voice lang auto  # reset to auto-detect (default)
```

Whisper supports 99 languages. `auto` detection works well but explicit codes improve accuracy for short utterances.
