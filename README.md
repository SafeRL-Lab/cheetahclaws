English | [中文](docs/i18n/README.CN.MD) | [한국어](docs/i18n/README.KO.MD) | [日本語](docs/i18n/README.JP.MD) | [Français](docs/i18n/README.FR.MD) | [Deutsch](docs/i18n/README.DE.MD) | [Español](docs/i18n/README.ES.MD) | [Português](docs/i18n/README.PT.MD)

<br> 

<div align="center">
  <a href="[https://github.com/SafeRL-Lab/Robust-Gymnasium](https://github.com/SafeRL-Lab/cheetahclaws)">
    <img src="docs/media/logos/logo-5.png" alt="Logo" width="280"> 
  </a>

  
<h2 align="center" style="font-size: 30px;"><strong><em>CheetahClaws</em></strong>: A Fast and Easy-to-Use Agent Harness Infrastructure for Long-Horizon, Multi-Model, and Tool-Using AI Systems</h2>
<p align="center">
    <a href="https://cheetahclaws.github.io/">Website</a>
    ·
    <a href="https://arxiv.org/pdf/2605.26112">Scaling the Harness</a>
    ·
    <a href="https://github.com/SafeRL-Lab/cheetahclaws/issues">Issue</a>
    
  
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

## 🔥🔥🔥 News (Pacific Time)

- June 4, 2026 (latest, **v3.05.81**): **Claude-Code-style quiet output — hide tool execution, show one summary line per turn.** Quiet mode (on by default) suppresses the per-tool `⚙ Tool(...)` / `✓ → N lines` clutter; the spinner shows live activity and a single line (`Read 2 files, ran 3 shell commands`) is emitted just above the reply. The permission prompt also collapses multi-line commands to one line. Errors still surface. The spinner shows a live timer + running token estimate (`Thinking… (7s · ↓ 435 tokens)`) and each turn closes with a real-usage footer (`✻ Worked for 7.2s · ↑ 1.2k · ↓ 435`). `/verbose` overrides it; toggle with `/quiet` or `--show-tools`; the banner shows `Output: quiet/full`. Details: [docs/guides/features.md](docs/guides/features.md) · [docs/news.md](docs/news.md).
- June 4, 2026: **Context-window override — the prompt % and compaction follow a settable context length.** New `/config context_window=<N>` overrides the model's context window (`0` = default), distinct from `max_tokens` (the output cap). One value drives the prompt `%`, `/context`, the compaction trigger, and the per-call output cap consistently — read live, so switching model or window updates the `%` with no restart. Details: [docs/guides/reference.md](docs/guides/reference.md) · [docs/news.md](docs/news.md).
- June 4, 2026: **Rich Live streaming — long responses stay live via a bounded tail window.** Long responses that would overflow the terminal keep rendering live but show only the most recent screenful (a bounded tail window), committing the full output when done — fixing the duplicate/stale frames some terminals left behind. Builds on PR #133. Details: [docs/guides/features.md](docs/guides/features.md) · [docs/news.md](docs/news.md).
- May 31, 2026: **QQ bot bridge — `/qq` connects cheetahclaws to QQ groups + C2C private chats via the official `qq-botpy` SDK (PR #121).** Details: [docs/guides/bridges.md](docs/guides/bridges.md#qq-bridge) · [docs/news.md](docs/news.md).
- May 12, 2026: **Security hardening sweep — env-var bot tokens, web CSRF cookie, terminal session owner-binding, and plugin/MCP/filesystem sandboxing (two CRITICAL + HIGH rounds, 2347 tests green).** Details: [docs/guides/security.md](docs/guides/security.md) · [docs/news.md](docs/news.md).
- May 12, 2026: **Daemon foundation roadmap — all nine F-1…F-9 items landed: subprocess agent runners, on-crash restart policy, daemonized Telegram/Slack/WeChat bridges, and budget guardrails.** Details: [docs/news.md](docs/news.md).

For more news, see [here](docs/news.md).

---

# CheetahClaws

CheetahClaws: **A Lightweight** and **Easy-to-Use** Python native Agent Harness Infrastructure, **Supporting Any Model**, such as Claude, GPT, Gemini, Kimi, Qwen, Zhipu, DeepSeek, MiniMax, and local open-source models via Ollama or any OpenAI-compatible endpoint.

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
  * [Trading Agent](#trading-agent)
  * [Web UI](#web-ui)
  * [Documentation](#documentation) (guides for all features)
  * [Contributing](#contributing) · [FAQ](#faq) · [Citation](#citation)


### Demos

<div align=center>
<img src="docs/media/demos/demo.gif" width="850"/>
</div>
<div align=center><sub><i>Task execution in the terminal</i></sub></div>

<br/>

<div align=center>
<img src="docs/media/demos/web_demo.gif" width="850"/>
</div>
<div align=center><sub><i>Web UI: browser chat — sidebar, tool cards, approval prompts, Markdown streaming</i></sub></div>

<br/>

<div align=center>
<img src="docs/media/demos/trading_demo.gif" width="850"/>
</div>
<div align=center><sub><i>Autonomous trading agent</i></sub></div>

> More animated demos (code review, `/research`, `/brainstorm`, `/lab`, Telegram/WeChat/Slack bridges) live in [`docs/media/`](docs/media/).

---

## Why CheetahClaws

Claude Code is a powerful, production-grade AI coding assistant — but its source is a compiled ~12 MB TypeScript/Node bundle (~1,300 files, ~283K lines), tightly coupled to the Anthropic API, hard to modify, and impossible to run against a local or alternative model.

**CheetahClaws** reimplements the same core loop in ~90K lines of readable Python — keeping what you need, dropping what you don't, and adding multi-provider + local-model support. Full comparison: [docs/guides/comparison.md](docs/guides/comparison.md).

| Dimension | Claude Code (TypeScript) | CheetahClaws (Python) |
|---|---|---|
| Language | TypeScript + React/Ink | Python 3.8+ |
| Source files / LoC | ~1,332 files / ~283K | ~315 files / ~90K (core; ~127K with tests) |
| Built-in tools / commands | 44+ / 88 | 27 / 50+ |
| Model providers | Anthropic only | 8+ (Anthropic · OpenAI · Gemini · Kimi · Qwen · DeepSeek · MiniMax · …) |
| Local models | No | Yes — Ollama, LM Studio, vLLM, any OpenAI-compatible endpoint |
| Build step | Yes (Bun + esbuild) | No — `python cheetahclaws.py` |
| Extensibility | Closed (compile-time) | Open — `register_tool()` at runtime, Markdown skills, git plugins, MCP |
| Voice input | Proprietary WebSocket (OAuth) | Local Whisper / OpenAI — works offline |

**Where Claude Code wins:** richer React/Ink UI, more built-in tools, enterprise features (MDM, team permission sync, OAuth/keychain), AI-driven memory extraction, single-binary production reliability.

**Where CheetahClaws wins:** any-model switching (`--model`/`/model`, no recompile) incl. full local/offline support; a readable agent loop in one file (`agent.py`, ~740 lines); zero build; runtime tool registration + MCP + git plugins + Markdown skills; task dependency graph (`blocks`/`blocked_by`); two-layer context compression; offline voice; cloud session sync; bridges to Telegram/WeChat/Slack/QQ.

**Who it's for:** developers who want a local/non-Anthropic coding assistant, researchers studying how agentic assistants work, and teams who need a hackable baseline — without a Node.js build chain.

---

## CheetahClaws vs OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is another popular open-source assistant (TypeScript/Node). The two have **different primary goals** — OpenClaw is a personal life-assistant across messaging channels; CheetahClaws is a developer/coding tool.

| Dimension | OpenClaw (TypeScript) | CheetahClaws (Python) |
|---|---|---|
| Lines of code | ~245K (~10,349 files) | ~90K core (~315 files) |
| Primary focus | Personal assistant across channels | AI coding assistant / dev tool |
| Architecture | Always-on Gateway daemon + apps | Zero-install terminal REPL |
| Messaging channels | 20+ (WhatsApp · Signal · iMessage · Discord · Matrix · …) | Terminal + Telegram · WeChat · Slack · QQ bridges |
| Local / offline models | Limited | Full — Ollama · vLLM · LM Studio · any OpenAI-compatible |
| Code editing tools | Browser control, Canvas | Read · Write · Edit · Bash · Glob · Grep · NotebookEdit · GetDiagnostics |
| Mobile / Live Canvas | Yes (menu bar + iOS/Android, A2UI) | — |
| MCP support | — | Yes (stdio/SSE/HTTP) |
| Hackability | 245K lines, harder to modify | ~90K lines — agent loop in one file |

| If you want… | Use |
|---|---|
| A personal assistant on WhatsApp/Signal/Discord, mobile-first, browser automation + Canvas | **OpenClaw** |
| An AI coding assistant in your terminal, full offline/local models, multi-provider switching, source you can read in an afternoon | **CheetahClaws** |

> Full comparison — both sides' wins + key design differences (agent loop, tool registration, context compression, memory): [docs/guides/comparison.md](docs/guides/comparison.md#cheetahclaws-vs-openclaw).

---

## Features

| Feature | Details |
|---|---|
| Multi-provider | Anthropic · OpenAI · Gemini · Kimi · Qwen · Zhipu · DeepSeek · MiniMax · Ollama · LM Studio · Custom endpoint |
| Agent loop | Streaming API + automatic tool-use loop; the whole loop is in `agent.py` |
| 27 built-in tools | Read · Write · Edit · Bash · Glob · Grep · WebFetch · WebSearch · NotebookEdit · GetDiagnostics · Memory* · Agent/SendMessage · Skill · AskUserQuestion · Task* · SleepTimer · EnterPlanMode/ExitPlanMode · *(MCP + plugin tools auto-added)* |
| MCP integration | Connect any MCP server (stdio/SSE/HTTP); tools auto-registered — see [extensions guide](docs/guides/extensions.md) |
| Plugin system | Install/enable/update plugins from git URLs or local paths; multi-scope; recommendation engine |
| Task management | `TaskCreate/Update/Get/List`, sequential IDs, dependency edges, persisted to `.cheetahclaws/tasks.json` |
| Context compression | Four cooperating layers — dynamic `max_tokens` cap, per-model context-window registry, two-layer snip + AI summarize at 70%, and auto-fanout for oversized tool outputs. [Details](docs/guides/reference.md) |
| Persistent memory | Dual-scope (user + project), 4 types, confidence/source metadata, conflict detection, recency-weighted search, `/memory consolidate` |
| Multi-agent | Spawn typed sub-agents (coder/reviewer/researcher/…), git-worktree isolation, background mode |
| Permission system | `auto` / `accept-all` / `manual` / `plan` modes |
| Checkpoints & plan mode | Auto-snapshot conversation + files each turn (`/checkpoint`, `/rewind`); `/plan` read-only analysis mode |
| Slash commands & themes | 50+ slash commands with Tab-complete; `/theme` offers 15 curated palettes |
| Brainstorm → Worker | `/brainstorm` runs an N-persona debate → `todo_list.txt`; `/worker` auto-implements the pending tasks |
| SSJ Developer Mode | `/ssj` — persistent power menu chaining Brainstorm, Worker, Review, Trading, Agent, Video/TTS, Monitor, etc. |
| Trading agent | `/trading` multi-agent analysis, backtesting, paper-trade calibration, MV portfolios. [Guide](docs/guides/trading.md) |
| Monitor | `/monitor` subscribes to AI-monitored topics on a schedule (arxiv / stock / crypto / news / custom), pushes reports to bridges/console |
| Research (multi-source) | `/research` fans out to **20 sources** with attention heat table, entity extraction, trend sparkline, comparison mode. [Guide](docs/guides/research.md) |
| Autonomous agents | `/agent` background loops from Markdown templates; iteration summaries pushed via bridge; stagnation-stop guard |
| Bridges + remote control | Telegram · WeChat · Slack · QQ — chat round-trip, slash passthrough, per-bridge job queue (`!jobs`/`!retry`/`!cancel`). [Guide](docs/guides/bridges.md) |
| Voice / Vision / Video / TTS | Offline Whisper `/voice`; `/image` clipboard vision (local + cloud); `/video` + `/tts` content factories. [Guide](docs/guides/voice-and-video.md) |
| Web UI | `--web` — multi-user browser chat + PTY terminal. [Guide](docs/guides/web-ui.md) |
| More | Tmux integration · `!cmd` shell escape · proactive monitoring · 3×Ctrl+C force-quit · session persistence · `/cloudsave` GitHub-Gist sync · cost tracking · `--print` non-interactive mode |

> **Full feature reference** — every row above with complete detail (context-compression layers, auto-fanout, 15 themes, the full Trading/Research/Agents writeups, …): [docs/guides/features.md](docs/guides/features.md).

---

## Supported Models

### Closed-Source (API)

| Provider | Example models | Context | API Key Env |
|---|---|---|---|
| **Anthropic** | `claude-opus-4-6` · `claude-sonnet-4-6` · `claude-haiku-4-5-20251001` | 200k | `ANTHROPIC_API_KEY` |
| **OpenAI** | `gpt-4o` · `gpt-4.1` · `gpt-5` · `o3` · `o4-mini` | 128–200k | `OPENAI_API_KEY` |
| **Google** | `gemini-2.5-pro` · `gemini-2.0-flash` · `gemini-1.5-pro` | 1–2M | `GEMINI_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-8k` / `-32k` / `-128k` | 8–128k | `MOONSHOT_API_KEY` |
| **Alibaba (Qwen)** | `qwen-max` · `qwen-plus` · `qwen-turbo` · `qwq-32b` | 32k–1M | `DASHSCOPE_API_KEY` |
| **Zhipu (GLM)** | `glm-4-plus` · `glm-4` · `glm-4-flash` (free tier) | 128k | `ZHIPU_API_KEY` |
| **DeepSeek** | `deepseek-chat` · `deepseek-reasoner` | 64k | `DEEPSEEK_API_KEY` |
| **MiniMax** | `MiniMax-Text-01` · `MiniMax-VL-01` · `abab6.5s-chat` | 256k–1M | `MINIMAX_API_KEY` |
| **AWS Bedrock / Azure / Vertex** _(via litellm)_ | `litellm/<provider>/<model>` | varies | provider-specific |

> **`litellm/` adapter:** routes to 100+ providers behind one SDK — mainly for upstreams with awkward auth (Bedrock SigV4, Azure deployment routing, Vertex service-account JWTs). For plain OpenAI-shaped endpoints, prefer the zero-dependency `custom/` adapter. Install with `pip install ".[litellm]"`. See [recipes.md](docs/guides/recipes.md#alternative-cloud-providers-with-non-trivial-auth-via-the-litellm-provider).

### Open-Source (Local via Ollama)

| Model | Size | Strengths | Pull |
|---|---|---|---|
| `qwen2.5-coder` | 7B / 32B | **Best for coding** | `ollama pull qwen2.5-coder` |
| `llama3.3` / `llama3.2` | 70B / 3B–11B | General purpose | `ollama pull llama3.3` |
| `deepseek-r1` | 7B–70B | Reasoning, math | `ollama pull deepseek-r1` |
| `mistral` / `mixtral` | 7B / 8x7B | Fast / strong MoE | `ollama pull mistral` |
| `phi4` · `gemma3` · `codellama` | 14B · 4–27B · 7–34B | Reasoning / open / code | `ollama pull phi4` |
| `llava` · `llama3.2-vision` | 7–13B · 11B | **Vision** | `ollama pull llava` |

> **Tool calling** needs a function-calling model — recommended: `qwen2.5-coder`, `llama3.3`, `mistral`, `phi4`. Reasoning models (`deepseek-r1`, `qwen3`, `gemma4`) stream native `<think>` blocks; enable with `/verbose` + `/thinking`.

---

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/SafeRL-Lab/cheetahclaws/main/scripts/install.sh | bash
# or:
pip install cheetahclaws
```

Works on **Linux, macOS, WSL2, and Android (Termux)** (Python 3.10+). First run guides you through provider + API-key setup; re-run anytime with `cheetahclaws --setup`.

> **Windows:** native Windows is not supported — use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install). **Android/Termux:** `pkg install python git && pip install cheetahclaws`.

### Alternative: install with `pip`

```bash
git clone https://github.com/SafeRL-Lab/cheetahclaws.git
cd cheetahclaws
pip install .                       # then: cheetahclaws
git pull && pip install --force-reinstall .   # to update
```

#### Optional extras

```bash
pip install ".[voice]"      # voice input (sounddevice + faster-whisper)
pip install ".[vision]"     # clipboard image capture (Pillow)
pip install ".[autosuggest]"# typing-time slash autosuggest (prompt_toolkit)
pip install ".[browser]"    # headless browser (playwright); then: playwright install chromium
pip install ".[files]"      # PDF + Excel reading (pymupdf, openpyxl)
pip install ".[ocr]"        # image OCR (pytesseract)
pip install ".[trading]"    # trading agent (yfinance, rank-bm25)
pip install ".[qq]"         # QQ bot bridge (qq-botpy)
pip install ".[litellm]"    # AWS Bedrock / Azure / Vertex auth via litellm
pip install ".[all]"        # everything above
```

### Alternative: install with `uv`

```bash
git clone https://github.com/SafeRL-Lab/cheetahclaws.git && cd cheetahclaws
uv tool install ".[all]"            # minimal: uv tool install .
uv tool install ".[all]" --reinstall   # update   ·   uv tool uninstall cheetahclaws
```

### Alternative: run directly from source (no install)

```bash
git clone https://github.com/SafeRL-Lab/cheetahclaws.git && cd cheetahclaws
pip install -r requirements.txt
python cheetahclaws.py              # changes take effect immediately
```

---

## Usage: Closed-Source API Models

Every cloud provider follows the same pattern — export its API key (see the [Supported Models](#closed-source-api) table for the env-var name), then select a model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # or OPENAI_API_KEY / GEMINI_API_KEY / DEEPSEEK_API_KEY / …
cheetahclaws                            # default model
cheetahclaws --model gpt-4o             # pick any model
cheetahclaws --model deepseek-chat --thinking --verbose
```

Provider get-key pages: [Anthropic](https://console.anthropic.com) · [OpenAI](https://platform.openai.com) · [Gemini](https://aistudio.google.com) · [Kimi](https://platform.moonshot.cn) · [Qwen](https://dashscope.aliyun.com) · [Zhipu](https://open.bigmodel.cn) · [DeepSeek](https://platform.deepseek.com) · [MiniMax](https://platform.minimaxi.chat).

**AWS Bedrock / Azure / Vertex** use the `litellm/<provider>/<model>` form (`pip install ".[litellm]"`) — full env-var recipes in [recipes.md](docs/guides/recipes.md#alternative-cloud-providers-with-non-trivial-auth-via-the-litellm-provider).

> **Full per-provider guide** — every provider's get-key page + example model commands, plus Bedrock/Azure/Vertex env-var recipes: [docs/guides/usage.md](docs/guides/usage.md).

---

## Usage: Open-Source Models (Local)

### Ollama (recommended)

```bash
curl -fsSL https://ollama.com/install.sh | sh   # install
ollama pull qwen2.5-coder                        # pull a tool-calling model
ollama serve                                     # http://localhost:11434 (auto-starts on macOS)
cheetahclaws --model ollama/qwen2.5-coder        # run (use `ollama list` to see local models)
```

### LM Studio

Download [LM Studio](https://lmstudio.ai), grab a GGUF model, start its **Local Server** (port 1234), then:

```bash
cheetahclaws --model lmstudio/<model-name>
```

### vLLM / self-hosted OpenAI-compatible server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes

export CUSTOM_BASE_URL=http://localhost:8000/v1
export CUSTOM_API_KEY=token-abc123      # any non-empty string if the server has no auth
cheetahclaws --model custom/Qwen2.5-Coder-32B-Instruct
```

The name after `custom/` must match the server's `--served-model-name`. For the Web UI, `--web --model custom/<name>` persists the model before the server starts. Remote server? Point `CUSTOM_BASE_URL` at its IP.

> **Full local-model guide** — Ollama step-by-step, LM Studio, vLLM + Web UI: [docs/guides/usage.md](docs/guides/usage.md#usage-open-source-models-local).

### Atlas Cloud (hosted, OpenAI-compatible)

> 🎁 **[Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=cheetahclaws)** is a full-modal AI inference platform with an OpenAI-compatible API — DeepSeek, Qwen, GLM, Kimi, MiniMax and more behind one endpoint. It plugs into the zero-dependency `custom/` adapter:

```bash
export CUSTOM_BASE_URL=https://api.atlascloud.ai/v1
export CUSTOM_API_KEY=your_atlascloud_api_key
cheetahclaws --model custom/deepseek-ai/deepseek-v4-pro
```

`deepseek-ai/deepseek-v4-pro` is a reasoning model; any other Atlas chat model id works the same way.

<details>
<summary>All Atlas Cloud chat models (59)</summary>

- **Anthropic (Claude):** `anthropic/claude-haiku-4.5-20251001`, `anthropic/claude-opus-4.8`, `anthropic/claude-sonnet-4.6`
- **OpenAI (GPT):** `openai/gpt-5.4`, `openai/gpt-5.5`
- **Google (Gemini):** `google/gemini-3.1-flash-lite`, `google/gemini-3.1-pro-preview`, `google/gemini-3.5-flash`
- **Qwen:** `qwen/qwen2.5-7b-instruct`, `Qwen/Qwen3-235B-A22B-Instruct-2507`, `qwen/qwen3-235b-a22b-thinking-2507`, `qwen/qwen3-30b-a3b`, `Qwen/Qwen3-30B-A3B-Instruct-2507`, `qwen/qwen3-30b-a3b-thinking-2507`, `qwen/qwen3-32b`, `qwen/qwen3-8b`, `Qwen/Qwen3-Coder`, `qwen/qwen3-coder-next`, `qwen/qwen3-max-2026-01-23`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-Next-80B-A3B-Thinking`, `Qwen/Qwen3-VL-235B-A22B-Instruct`, `qwen/qwen3-vl-235b-a22b-thinking`, `qwen/qwen3-vl-30b-a3b-instruct`, `qwen/qwen3-vl-30b-a3b-thinking`, `qwen/qwen3-vl-8b-instruct`, `qwen/qwen3.5-122b-a10b`, `qwen/qwen3.5-27b`, `qwen/qwen3.5-35b-a3b`, `qwen/qwen3.5-397b-a17b`, `qwen/qwen3.6-35b-a3b`, `qwen/qwen3.6-plus`
- **DeepSeek:** `deepseek-ai/deepseek-ocr`, `deepseek-ai/deepseek-r1-0528`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-V3.1`, `deepseek-ai/DeepSeek-V3.1-Terminus`, `deepseek-ai/deepseek-v3.2`, `deepseek-ai/DeepSeek-V3.2-Exp`, `deepseek-ai/deepseek-v4-flash`, `deepseek-ai/deepseek-v4-pro`
- **Moonshot (Kimi):** `moonshotai/Kimi-K2-Instruct`, `moonshotai/Kimi-K2-Instruct-0905`, `moonshotai/Kimi-K2-Thinking`, `moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`
- **Zhipu (GLM):** `zai-org/GLM-4.6`, `zai-org/glm-4.7`, `zai-org/glm-5`, `zai-org/glm-5-turbo`, `zai-org/glm-5.1`, `zai-org/glm-5v-turbo`
- **MiniMax:** `MiniMaxAI/MiniMax-M2`, `minimaxai/minimax-m2.1`, `minimaxai/minimax-m2.5`, `minimaxai/minimax-m2.7`
- **xAI:** `xai/grok-4.3`
- **Kwaipilot:** `kwaipilot/kat-coder-pro-v2`
- **Other:** `owl`

</details>

---

## Model Name Format

Three equivalent forms are accepted:

```bash
cheetahclaws --model gpt-4o                  # 1. auto-detect by prefix
cheetahclaws --model ollama/qwen2.5-coder    # 2. provider/model
cheetahclaws --model kimi:moonshot-v1-32k    # 3. provider:model
```

**Auto-detection by prefix:** `claude-`→anthropic · `gpt-`/`o1`/`o3`→openai · `gemini-`→gemini · `moonshot-`/`kimi-`→kimi · `qwen`/`qwq-`→qwen · `glm-`→zhipu · `deepseek-`→deepseek · `MiniMax-`/`abab`→minimax · `llama`/`mistral`/`phi`/`gemma`/`mixtral`/`codellama`→ollama.

---

## Trading Agent

A built-in AI trading analysis + backtesting module (`pip install "cheetahclaws[trading]"`).

```bash
/trading analyze NVDA            # 5-phase pipeline: data → Bull/Bear debate → Judge → Risk panel → PM decision
/trading backtest AAPL dual_ma   # backtest a strategy (or let AI pick); Sharpe/Sortino/Calmar/drawdown/win-rate
```

4 strategies (`dual_ma`, `rsi_mean_reversion`, `bollinger_breakout`, `macd_crossover`), BM25 memory of past situations, US/HK/A-share + crypto markets with no-API-key data fallbacks. Guided sub-menu via `/ssj` → **Trading**.

> **Full guide:** [docs/guides/trading.md](docs/guides/trading.md)

---

## Web UI

A production-ready browser interface — real user accounts (bcrypt + JWT), SQLite-backed history, ops endpoints — served by Python stdlib + ten vanilla-JS modules (no Node.js / React / build step).

```bash
pip install 'cheetahclaws[web]'
cheetahclaws --web                  # auto-picks a free port (tries 8080)
cheetahclaws --web --port 9000 --host 0.0.0.0   # bind explicitly / open to LAN
cheetahclaws --web --no-auth        # skip login (localhost dev only)
```

Open `http://localhost:<port>/chat` — first account becomes admin. Includes streaming chat (WS) + SSE slash commands, persistent sessions with folders/search/Markdown export, tool cards, inline permission approval, settings panel, light/dark/system theme, and `/health` + `/metrics` endpoints. A full xterm.js PTY terminal lives at `/` (100% CLI parity).

> **Full guide:** [docs/guides/web-ui.md](docs/guides/web-ui.md) · **Docker / home server:** [docs/guides/docker.md](docs/guides/docker.md)

---

## Documentation

Detailed guides live in [`docs/guides/`](docs/guides/) to keep this README focused:

| Guide | What's inside |
|---|---|
| [**Features (full)**](docs/guides/features.md) | The complete feature table — every row with full detail (context compression, auto-fanout, themes, Trading/Research/Agents writeups) |
| [**Usage (all providers)**](docs/guides/usage.md) | Per-provider setup + example commands: Anthropic/OpenAI/Gemini/Kimi/Qwen/Zhipu/DeepSeek/MiniMax/litellm, and local Ollama/LM Studio/vLLM |
| [**Web UI**](docs/guides/web-ui.md) | Chat UI, PTY terminal, API endpoints, settings, auth, SSE streaming |
| [**Docker / Home Server**](docs/guides/docker.md) | Dockerfile + compose: web UI + bridges in one container, host Ollama, workspace mount |
| [**Reference**](docs/guides/reference.md) | CLI, 50+ commands, 33 built-in tools, session search, error classification, tool cache |
| [**Extensions**](docs/guides/extensions.md) | Memory, Skills, Sub-Agents, MCP servers, Plugins, Monitor, Autonomous Agents |
| [**Bridges**](docs/guides/bridges.md) | Telegram, WeChat, Slack, QQ setup + remote control from your phone |
| [**Security & env vars**](docs/guides/security.md) | Threat model, `CHEETAHCLAWS_*` vars, bot-token handling, Bash denylist, fs sandbox, CSRF |
| [**Voice & Video**](docs/guides/voice-and-video.md) | Offline Whisper voice input, Video factory, TTS factory |
| [**Trading**](docs/guides/trading.md) | Multi-agent analysis, backtesting, BM25 memory, data fallbacks, SSJ integration |
| [**Advanced**](docs/guides/advanced.md) | Brainstorm, SSJ, Tmux, proactive monitoring, checkpoints, plan mode, sessions, cloud sync |
| [**Comparison**](docs/guides/comparison.md) | Full positioning vs Claude Code and OpenClaw — at-a-glance tables, both sides' wins, key design differences |
| [**Recipes**](docs/guides/recipes.md) | 12 step-by-step examples: code review, remote control, research, bug fix, browse, email, PDF/Excel |
| [**FAQ**](docs/guides/faq.md) | The full FAQ (MCP, models/providers, CLI/scripting, voice) |
| [**Plugin Authoring**](docs/guides/plugin-authoring.md) · [Example](examples/example-plugin/) | Build a plugin: tools, commands, skills, MCP; starter template |
| [**Research Lab**](docs/guides/research-lab.md) | `/lab start <topic>` — autonomous multi-agent paper writing with sandboxed experiments |
| [**Agent OS**](docs/agent-os.md) · [RFC index](docs/RFC/) | The `cc_kernel/` layer + all design notes (RFC 0001-0032) |
| [**Contributing**](CONTRIBUTING.md) | Project structure, architecture guide, PR checklist |

---

## Quick Reference

```
cheetahclaws [OPTIONS] [PROMPT]

  -p, --print          Non-interactive: run prompt and exit
  -m, --model MODEL    Override model (e.g. gpt-4o, ollama/llama3.3)
  --accept-all         Auto-approve all operations (no permission prompts)
  --verbose            Show thinking blocks and per-turn token counts
  --show-tools         Show each tool call instead of a per-turn summary
                       (alias: --no-quiet; compact summary is the default)
  --thinking           Enable Extended Thinking (Claude only)
  --web                Start web server (Chat UI + PTY terminal in browser)
  --port / --host      Web server port / host (default 8080 / 127.0.0.1)
  --no-auth            Disable web password (local use only)
  --version / -h       Print version / show help
```

```bash
cheetahclaws                                          # interactive REPL, default model
cheetahclaws -m ollama/deepseek-r1:32b                # pick a model
cheetahclaws -p "Write a Python fibonacci function"   # non-interactive
cheetahclaws --accept-all -p "Init a pyproject.toml"  # CI / automation
cheetahclaws --web --port 8008 --no-auth              # browser chat + terminal
```

See the [Reference Guide](docs/guides/reference.md) for all 50+ slash commands, tools, and config options.

---

## Contributing

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for architecture, conventions, and the PR checklist.

```bash
git clone https://github.com/SafeRL-Lab/cheetahclaws.git && cd cheetahclaws
pip install -r requirements.txt && pip install pytest
python -m pytest tests/ -x -q       # 341+ tests should pass
python cheetahclaws.py              # run the REPL
```

Building a plugin? See the [Plugin Authoring Guide](docs/guides/plugin-authoring.md) and the [example template](examples/example-plugin/).

---

## FAQ

A few common questions — the **full FAQ** is in [docs/guides/faq.md](docs/guides/faq.md).

**Q: How do I add an MCP server?**
```
/mcp add git uvx mcp-server-git          # or create .mcp.json in your project, then /mcp reload
```

**Q: Tool calls don't work with my local Ollama model.**
Not all models support function calling — use `qwen2.5-coder`, `llama3.3`, `mistral`, or `phi4`.

**Q: How do I connect to a remote GPU server running vLLM?**
```
/config custom_base_url=http://your-server-ip:8000/v1
/config custom_api_key=your-token
/model custom/your-model-name
```

**Q: How do I check my API cost?** Run `/cost` (shows input/output tokens + estimated USD).

**Q: Can I use multiple API keys in one session?** Yes — set all keys upfront (env or `/config`), then switch models freely; each call uses the active provider's key.

**Q: How do I set a default model across projects?** Add keys to `~/.bashrc`/`~/.zshrc` and set `{ "model": "claude-sonnet-4-6" }` in `~/.cheetahclaws/config.json`.

**Q: Can I pipe input to cheetahclaws?**
```bash
cat error.log | cheetahclaws -p "What is causing this error?"
```

**Q: How do I set up voice input?** `pip install sounddevice faster-whisper numpy`, then `/voice` in the REPL (downloads a ~150 MB Whisper model on first use). See the [full FAQ](docs/guides/faq.md) for languages + keyterm tuning.

---

## Citation
If you find the repository useful, please cite the study
``` Bash
@article{gu2026model,
  title={From Model Scaling to System Scaling: Scaling the Harness in Agentic AI},
  author={Gu, Shangding},
  journal={arXiv preprint arXiv:2605.26112},
  year={2026}
}

@article{cheetahclaws2026,
  title={CheetahClaws: Agent Harness Infrastructure for Long-Horizon, Multi-Model, and Tool-Using AI Systems},
  author={CheetahClaws Team},
  journal={github},
  year={2026}
}
```


---


## Thanks to all contributors:

<!-- contributors:start -->

<a href="https://github.com/chauncygu"><img src="https://avatars.githubusercontent.com/u/27274029?v=4&s=48" width="48" height="48" alt="chauncygu"/></a>
<a href="https://github.com/KevRojo"><img src="https://avatars.githubusercontent.com/u/9065636?v=4&s=48" width="48" height="48" alt="KevRojo"/></a>
<a href="https://github.com/mxh1999"><img src="https://avatars.githubusercontent.com/u/30319236?v=4&s=48" width="48" height="48" alt="mxh1999"/></a>
<a href="https://github.com/seetvn"><img src="https://avatars.githubusercontent.com/u/100040536?v=4&s=48" width="48" height="48" alt="seetvn"/></a>
<a href="https://github.com/bmaltais"><img src="https://avatars.githubusercontent.com/u/7474674?v=4&s=48" width="48" height="48" alt="bmaltais"/></a>
<a href="https://github.com/RheagalFire"><img src="https://avatars.githubusercontent.com/u/60213893?v=4&s=48" width="48" height="48" alt="RheagalFire"/></a>
<a href="https://github.com/yamaceay"><img src="https://avatars.githubusercontent.com/u/46201716?v=4&s=48" width="48" height="48" alt="yamaceay"/></a>
<a href="https://github.com/tsint"><img src="https://avatars.githubusercontent.com/u/63944253?v=4&s=48" width="48" height="48" alt="tsint"/></a>
<a href="https://github.com/albertcheng"><img src="https://avatars.githubusercontent.com/u/2686135?v=4&s=48" width="48" height="48" alt="albertcheng"/></a>
<a href="https://github.com/LostAion"><img src="https://avatars.githubusercontent.com/u/84846068?v=4&s=48" width="48" height="48" alt="LostAion"/></a>
<a href="https://github.com/skint007"><img src="https://avatars.githubusercontent.com/u/37035851?v=4&s=48" width="48" height="48" alt="skint007"/></a>
<a href="https://github.com/thekbbohara"><img src="https://avatars.githubusercontent.com/u/133592644?v=4&s=48" width="48" height="48" alt="thekbbohara"/></a>

<!-- contributors:end -->
