#!/usr/bin/env python3
"""
Nano Claude Code — Minimal Python implementation of Claude Code.

Usage:
  python nano_claude.py [options] [prompt]

Options:
  -p, --print          Non-interactive: run prompt and exit (also --print-output)
  -m, --model MODEL    Override model
  --accept-all         Never ask permission (dangerous)
  --verbose            Show thinking + token counts
  --version            Print version and exit

Slash commands in REPL:
  /help       Show this help
  /clear      Clear conversation
  /model [m]  Show or set model
  /config     Show config / set key=value
  /save [f]   Save session to file
  /load [f]   Load session from file
  /history    Print conversation history
  /context    Show context window usage
  /cost       Show API cost this session
  /verbose    Toggle verbose mode
  /thinking   Toggle extended thinking
  /permissions [mode]  Set permission mode
  /cwd [path] Show or change working directory
  /memory [query]         Show/search persistent memories
  /memory consolidate     Extract long-term insights from current session via AI
  /skills           List available skills
  /agents           Show sub-agent tasks
  /mcp              List MCP servers and their tools
  /mcp reload       Reconnect all MCP servers
  /mcp add <n> <cmd> [args]  Add a stdio MCP server
  /mcp remove <n>   Remove an MCP server from config
  /plugin           List installed plugins
  /plugin install name@url   Install a plugin
  /plugin uninstall name     Uninstall a plugin
  /plugin enable/disable name  Toggle plugin
  /plugin update name        Update a plugin
  /plugin recommend [ctx]    Recommend plugins for context
  /tasks            List all tasks
  /tasks create <subject>    Quick-create a task
  /tasks start/done/cancel <id>  Update task status
  /tasks delete <id>         Delete a task
  /tasks get <id>            Show full task details
  /tasks clear               Delete all tasks
  /voice            Record voice input, transcribe, and submit
  /voice status     Show available recording and STT backends
  /voice lang <code>  Set STT language (e.g. zh, en, ja — default: auto)
  /proactive [dur]  Background sentinel polling (e.g. /proactive 5m)
  /proactive off    Disable proactive polling
  /cloudsave setup <token>   Configure GitHub token for cloud sync
  /cloudsave        Upload current session to GitHub Gist
  /cloudsave push [desc]     Upload with optional description
  /cloudsave auto on|off     Toggle auto-upload on exit
  /cloudsave list   List your nano-claude-code Gists
  /cloudsave load <gist_id>  Download and load a session from Gist
  /exit /quit Exit
"""
from __future__ import annotations

import os
import re
import sys
if sys.platform == "win32":
    os.system("")  # Enable ANSI escape codes on Windows CMD
import json
try:
    import readline
except ImportError:
    readline = None  # Windows compatibility
import atexit
import argparse
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import threading
# ── Optional rich for markdown rendering ──────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich import print as rprint
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None

VERSION = "3.05.5"

# ── ANSI helpers (used even with rich for non-markdown output) ─────────────
C = {
    "cyan":    "\033[36m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "red":     "\033[31m",
    "blue":    "\033[34m",
    "magenta": "\033[35m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "reset":   "\033[0m",
}

def clr(text: str, *keys: str) -> str:
    return "".join(C[k] for k in keys) + str(text) + C["reset"]

def info(msg: str):   print(clr(msg, "cyan"))
def ok(msg: str):     print(clr(msg, "green"))
def warn(msg: str):   print(clr(f"Warning: {msg}", "yellow"))
def err(msg: str):    print(clr(f"Error: {msg}", "red"), file=sys.stderr)


def render_diff(text: str):
    """Print diff text with ANSI colors: red for removals, green for additions."""
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            print(C["bold"] + line + C["reset"])
        elif line.startswith("+"):
            print(C["green"] + line + C["reset"])
        elif line.startswith("-"):
            print(C["red"] + line + C["reset"])
        elif line.startswith("@@"):
            print(C["cyan"] + line + C["reset"])
        else:
            print(line)

def _has_diff(text: str) -> bool:
    """Check if text contains a unified diff."""
    return "--- a/" in text and "+++ b/" in text


# ── Conversation rendering ─────────────────────────────────────────────────

_accumulated_text: list[str] = []   # buffer text during streaming
_current_live: "Live | None" = None  # active Rich Live instance (one at a time)
_RICH_LIVE = True  # set to False (via config rich_live=false) to disable in-place Live streaming

def _make_renderable(text: str):
    """Return a Rich renderable: Markdown if text contains markup, else plain."""
    if any(c in text for c in ("#", "*", "`", "_", "[")):
        return Markdown(text)
    return text

def _start_live() -> None:
    """Start a Rich Live block for in-place Markdown streaming (no-op if not Rich)."""
    global _current_live
    if _RICH and _RICH_LIVE and _current_live is None:
        _current_live = Live(console=console, auto_refresh=False,
                             vertical_overflow="visible")
        _current_live.start()

def stream_text(chunk: str) -> None:
    """Buffer chunk; update Live in-place when Rich available, else print directly."""
    global _current_live
    _accumulated_text.append(chunk)
    if _RICH and _RICH_LIVE:
        if _current_live is None:
            _start_live()
        _current_live.update(_make_renderable("".join(_accumulated_text)), refresh=True)
    else:
        print(chunk, end="", flush=True)

def stream_thinking(chunk: str, verbose: bool):
    if verbose:
        # Strip internal newlines when models stream token-by-token (like Qwen).
        clean_chunk = chunk.replace("\n", " ")
        if clean_chunk:
            # We explicitly do NOT use clr() wrapper here to avoid outputting \033[0m (reset)
            # after every single token. Repeated ANSI resets can cause formatting glitches and vertical cascades.
            print(f"{C['dim']}{clean_chunk}", end="", flush=True)

def flush_response() -> None:
    """Commit buffered text to screen: stop Live (freezes rendered Markdown in place)."""
    global _current_live
    full = "".join(_accumulated_text)
    _accumulated_text.clear()
    if _current_live is not None:
        _current_live.stop()
        _current_live = None
    elif _RICH and _RICH_LIVE and full.strip():
        # Fallback: no Live was running but Rich is available (e.g. after thinking)
        console.print(_make_renderable(full))
    else:
        print()  # ensure newline after plain-text stream

_TOOL_SPINNER_PHRASES = [
    "☕ Brewing some coffee...",
    "🚰 Drinking some water...",
    "🧠 Thinking really hard...",
    "🔧 Tightening some bolts...",
    "🎯 Locking on target...",
    "🔍 Investigating...",
    "🧩 Connecting the dots...",
    "⚡ Charging up...",
    "🎨 Painting the bits...",
    "🏗️ Building something cool...",
    "🌀 Spinning the wheels...",
    "🧪 Running experiments...",
    "📡 Scanning frequencies...",
    "🛠️ Tuning the engine...",
    "🐛 Chasing a bug...",
]

_tool_spinner_thread = None
_tool_spinner_stop = threading.Event()

_spinner_phrase = ""
_spinner_lock = threading.Lock()

def _run_tool_spinner():
    """Background spinner on a single line using carriage return."""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not _tool_spinner_stop.is_set():
        with _spinner_lock:
            phrase = _spinner_phrase
        frame = chars[i % len(chars)]
        sys.stdout.write(f"\r  {frame} {clr(phrase, 'dim')}   ")
        sys.stdout.flush()
        i += 1
        _tool_spinner_stop.wait(0.1)

def _start_tool_spinner():
    global _tool_spinner_thread
    if _tool_spinner_thread and _tool_spinner_thread.is_alive():
        return  # already running
    import random
    with _spinner_lock:
        global _spinner_phrase
        _spinner_phrase = random.choice(_TOOL_SPINNER_PHRASES)
    _tool_spinner_stop.clear()
    _tool_spinner_thread = threading.Thread(target=_run_tool_spinner, daemon=True)
    _tool_spinner_thread.start()

def _change_spinner_phrase():
    """Change the spinner phrase without stopping it."""
    import random
    with _spinner_lock:
        global _spinner_phrase
        _spinner_phrase = random.choice(_TOOL_SPINNER_PHRASES)

def _stop_tool_spinner():
    global _tool_spinner_thread
    if not _tool_spinner_thread:
        return
    _tool_spinner_stop.set()
    _tool_spinner_thread.join(timeout=1)
    _tool_spinner_thread = None
    # Clear the spinner on the same line
    sys.stdout.write(f"\r{' ' * 50}\r")
    sys.stdout.flush()

def print_tool_start(name: str, inputs: dict, verbose: bool):
    """Show tool invocation."""
    desc = _tool_desc(name, inputs)
    print(clr(f"  ⚙  {desc}", "dim", "cyan"), flush=True)
    if verbose:
        print(clr(f"     inputs: {json.dumps(inputs, ensure_ascii=False)[:200]}", "dim"))

def print_tool_end(name: str, result: str, verbose: bool):
    lines = result.count("\n") + 1
    size = len(result)
    summary = f"→ {lines} lines ({size} chars)"
    if not result.startswith("Error") and not result.startswith("Denied"):
        print(clr(f"  ✓ {summary}", "dim", "green"), flush=True)
        # Render diff for Edit/Write results
        if name in ("Edit", "Write") and _has_diff(result):
            parts = result.split("\n\n", 1)
            if len(parts) == 2:
                print(clr(f"  {parts[0]}", "dim"))
                render_diff(parts[1])
    else:
        print(clr(f"  ✗ {result[:120]}", "dim", "red"), flush=True)
    if verbose and not result.startswith("Denied"):
        preview = result[:500] + ("…" if len(result) > 500 else "")
        print(clr(f"     {preview.replace(chr(10), chr(10)+'     ')}", "dim"))

def _tool_desc(name: str, inputs: dict) -> str:
    if name == "Read":   return f"Read({inputs.get('file_path','')})"
    if name == "Write":  return f"Write({inputs.get('file_path','')})"
    if name == "Edit":   return f"Edit({inputs.get('file_path','')})"
    if name == "Bash":   return f"Bash({inputs.get('command','')[:80]})"
    if name == "Glob":   return f"Glob({inputs.get('pattern','')})"
    if name == "Grep":   return f"Grep({inputs.get('pattern','')})"
    if name == "WebFetch":    return f"WebFetch({inputs.get('url','')[:60]})"
    if name == "WebSearch":   return f"WebSearch({inputs.get('query','')})"
    if name == "Agent":
        atype = inputs.get("subagent_type", "")
        aname = inputs.get("name", "")
        iso   = inputs.get("isolation", "")
        bg    = not inputs.get("wait", True)
        parts = []
        if atype:  parts.append(atype)
        if aname:  parts.append(f"name={aname}")
        if iso:    parts.append(f"isolation={iso}")
        if bg:     parts.append("background")
        suffix = f"({', '.join(parts)})" if parts else ""
        prompt_short = inputs.get("prompt", "")[:60]
        return f"Agent{suffix}: {prompt_short}"
    if name == "SendMessage":
        return f"SendMessage(to={inputs.get('to','')}: {inputs.get('message','')[:50]})"
    if name == "CheckAgentResult": return f"CheckAgentResult({inputs.get('task_id','')})"
    if name == "ListAgentTasks":   return "ListAgentTasks()"
    if name == "ListAgentTypes":   return "ListAgentTypes()"
    return f"{name}({list(inputs.values())[:1]})"


# ── Permission prompt ──────────────────────────────────────────────────────

def ask_permission_interactive(desc: str, config: dict) -> bool:
    try:
        print()
        ans = input(clr(f"  Allow: {desc}  [y/N/a(ccept-all)] ", "yellow")).strip().lower()
        if ans == "a":
            config["permission_mode"] = "accept-all"
            ok("  Permission mode set to accept-all for this session.")
            return True
        return ans in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        print()
        return False


# ── Slash commands ─────────────────────────────────────────────────────────

import time
import traceback

def _proactive_watcher_loop(config):
    """Background daemon that fires a wake-up prompt after a period of inactivity."""
    while True:
        time.sleep(1)
        if not config.get("_proactive_enabled"):
            continue
        try:
            now = time.time()
            interval = config.get("_proactive_interval", 300)
            last = config.get("_last_interaction_time", now)
            if now - last >= interval:
                config["_last_interaction_time"] = now
                cb = config.get("_run_query_callback")
                if cb:
                    cb(f"(System Automated Event) You have been inactive for {interval} seconds. "
                       "Before doing anything else, review your previous messages in this conversation. "
                       "If you said you would implement, fix, or do something and didn't finish it, "
                       "continue and complete that work now. "
                       "Otherwise, check if you have any pending tasks to execute or simply say 'No pending tasks'.")
        except Exception as e:
            traceback.print_exc()
            print(f"\n[proactive watcher error]: {e}", flush=True)

def cmd_help(_args: str, _state, _config) -> bool:
    print(__doc__)
    return True

def cmd_model(args: str, _state, config) -> bool:
    from providers import PROVIDERS, detect_provider
    if not args:
        model = config["model"]
        pname = detect_provider(model)
        info(f"Current model:    {model}  (provider: {pname})")
        info("\nAvailable models by provider:")
        for pn, pdata in PROVIDERS.items():
            ms = pdata.get("models", [])
            if ms:
                info(f"  {pn:12s}  " + ", ".join(ms[:4]) + ("..." if len(ms) > 4 else ""))
        info("\nFormat: 'provider/model' or just model name (auto-detected)")
        info("  e.g. /model gpt-4o")
        info("  e.g. /model ollama/qwen2.5-coder")
        info("  e.g. /model kimi:moonshot-v1-32k")
    else:
        # Accept both "ollama/model" and "ollama:model" syntax
        # Only treat ':' as provider separator if left side is a known provider
        m = args.strip()
        if "/" not in m and ":" in m:
            left, right = m.split(":", 1)
            if left in PROVIDERS:
                m = f"{left}/{right}"
        config["model"] = m
        pname = detect_provider(m)
        ok(f"Model set to {m}  (provider: {pname})")
        from config import save_config
        save_config(config)
    return True

def _generate_personas(topic: str, curr_model: str, config: dict, count: int = 5) -> dict | None:
    """Ask the LLM to generate `count` topic-appropriate expert personas as a dict."""
    from providers import stream, TextChunk
    import json

    example_entries = "\n".join(
        f'  "p{i+1}": {{"icon": "emoji", "role": "Expert Title", "desc": "One sentence describing their analytical angle."}}'
        for i in range(count)
    )
    user_msg = f"""Generate {count} expert personas for a multi-perspective brainstorming debate on: "{topic}"

Return ONLY a valid JSON object — no markdown fences, no extra text — like this:
{{
{example_entries}
}}

Choose experts whose domains are most relevant to analyzing "{topic}" from different angles."""

    internal_config = config.copy()
    internal_config["no_tools"] = True
    chunks = []
    try:
        for event in stream(curr_model, "You are a debate facilitator. Return only valid JSON.", [{"role": "user", "content": user_msg}], [], internal_config):
            if isinstance(event, TextChunk):
                chunks.append(event.text)
    except Exception:
        return None

    raw = "".join(chunks).strip()
    # Strip markdown code fences if the model wraps in ```json ... ```
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    try:
        return json.loads(raw)
    except Exception:
        return None


_TECH_PERSONAS = {
    "architect":   {"icon": "🏗️", "role": "Principal Software Architect",       "desc": "Focus on modularity, clear boundaries, patterns, and long-term maintainability."},
    "innovator":   {"icon": "💡", "role": "Pragmatic Product Innovator",          "desc": "Focus on bold, technically feasible ideas that add high user value and differentiation."},
    "security":    {"icon": "🛡️", "role": "Security & Risk Engineer",            "desc": "Focus on vulnerabilities, data integrity, secrets handling, and project robustness."},
    "refactor":    {"icon": "🔧", "role": "Senior Code Quality Lead",             "desc": "Focus on code smells, complexity reduction, DRY principles, and readability."},
    "performance": {"icon": "⚡", "role": "Performance & Optimization Specialist","desc": "Focus on I/O bottlenecks, resource efficiency, latency, and scalability."},
}


def _interactive_ollama_picker(config: dict) -> bool:
    """Prompt the user to select from locally available Ollama models."""
    from providers import PROVIDERS, list_ollama_models
    prov = PROVIDERS.get("ollama", {})
    base_url = prov.get("base_url", "http://localhost:11434")
    
    models = list_ollama_models(base_url)
    if not models:
        err(f"No local Ollama models found at {base_url}.")
        return False
        
    print(clr("\n  ── Local Ollama Models ──", "dim"))
    for i, m in enumerate(models):
        print(clr(f"  [{i+1:2d}] ", "yellow") + m)
    print()
    
    try:
        ans = input(clr("  Select a model number or Enter to cancel > ", "cyan")).strip()
        if not ans: return False
        idx = int(ans) - 1
        if 0 <= idx < len(models):
            new_model = f"ollama/{models[idx]}"
            config["model"] = new_model
            from config import save_config
            save_config(config)
            ok(f"Model updated to {new_model}")
            return True
        else:
            err("Invalid selection.")
    except (ValueError, KeyboardInterrupt, EOFError):
        pass
    return False

def cmd_brainstorm(args: str, state, config) -> bool:
    """Run a multi-persona iterative brainstorming session on the project.
    
    Usage: /brainstorm [topic]
    """
    from providers import stream
    import time
    from pathlib import Path
    
    # ── Context Snapshot ──────────────────────────────────────────────────
    readme_path = Path("README.md")
    readme_content = ""
    if readme_path.exists():
        readme_content = readme_path.read_text("utf-8", errors="replace")
    
    claude_md = Path("CLAUDE.md")
    claude_content = ""
    if claude_md.exists():
        claude_content = claude_md.read_text("utf-8", errors="replace")
        
    project_files = "\n".join([f.name for f in Path(".").glob("*") if f.is_file() and not f.name.startswith(".")])
    
    user_topic = args.strip() or "general project improvement and architectural evolution"

    # ── Ask user for agent count interactively ────────────────────────────
    if config.get("_telegram_incoming"):
        agent_count = 5  # skip interactive input when called from Telegram
    else:
        try:
            ans = input(clr(f"  How many agents? (2-100, default 5) > ", "cyan")).strip()
            agent_count = int(ans) if ans else 5
            agent_count = max(2, min(agent_count, 100))
        except (ValueError, KeyboardInterrupt, EOFError):
            agent_count = 5
    
    snapshot = f"""PROJECT CONTEXT:
README:
{readme_content[:3000]}

CLAUDE.MD:
{claude_content[:1000]}

ROOT FILES:
{project_files}

USER FOCUS: {user_topic}
"""
    curr_model = config["model"]

    # ── Personas (dynamically generated per topic) ────────────────────────
    info(clr(f"Generating {agent_count} topic-appropriate expert personas...", "dim"))
    personas = _generate_personas(user_topic, curr_model, config, count=agent_count)
    if not personas:
        info(clr("(persona generation failed, using default tech personas)", "dim"))
        personas = dict(list(_TECH_PERSONAS.items())[:agent_count])
    
    # ── Identity Generator ────────────────────────────────────────────────
    def get_identity(letter):
        try:
            from faker import Faker
            fake = Faker()
            return f"{letter}", fake.name()
        except:
            first = ["Alex", "Sam", "Taylor", "Jordan", "Casey", "Riley", "Drew", "Avery"]
            last = ["Garcia", "Martinez", "Lopez", "Hernandez", "Gonzalez", "Sanchez", "Ramirez", "Torres"]
            import random
            return f"{letter}", f"{random.choice(first)} {random.choice(last)}"
            
    # ── Debate Loop ───────────────────────────────────────────────────────
    outputs_dir = Path("brainstorm_outputs")
    outputs_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = outputs_dir / f"brainstorm_{ts}.md"
    
    brainstorm_history = []
    
    ok(f"Starting {agent_count}-Agent Brainstorming Session on: {clr(user_topic, 'bold')}")
    info(clr("Generating diverse perspectives...", "dim"))

    # Helper function to call the model via the unified stream() function
    def call_persona(persona_name, p_data, history):
        letter, name = get_identity(persona_name[0].upper())
        # We wrap the persona instructions into a 'system' role
        system_prompt = f"""You are {name}, the {p_data['role']}. Identity: Agent {letter}.
{p_data['desc']}

TOPIC UNDER DISCUSSION: {user_topic}

PROJECT CONTEXT (if relevant to the topic):
{snapshot}

INSTRUCTIONS:
1. Provide 3-5 concrete, actionable insights or ideas from your expert perspective on the topic.
2. If there are prior ideas from other agents, briefly acknowledge them and build upon or challenge them.
3. Be specific, well-reasoned, and professional. Stay in character as your role.
4. Prefix each of your points with: [Agent {letter} — {name}]
5. Output your response in clean Markdown.
"""
        user_msg = f"TOPIC: {user_topic}\n\nPRIOR IDEAS FROM DEBATE:\n{history or 'No previous ideas yet. You are the first to speak.'}"
        
        full_response = []
        # Internal calls should not include tools (tool_schemas already passed as [])
        internal_config = config.copy()
        internal_config["no_tools"] = True
        
        try:
            from providers import TextChunk
            for event in stream(curr_model, system_prompt, [{"role": "user", "content": user_msg}], [], internal_config):
                if isinstance(event, TextChunk):
                    full_response.append(event.text)
        except Exception as e:
            return f"Error from Agent {letter}: {e}"
            
        return "".join(full_response).strip()

    full_log = [f"# Brainstorming Session: {user_topic}", f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}", f"**Model:** {curr_model}", "---"]
    
    for p_name, p_data in personas.items():
        icon = p_data.get("icon", "🤖")
        info(f"{icon} {clr(p_data['role'], 'yellow')} is thinking...")
        _start_tool_spinner()

        hist_text = "\n\n".join(brainstorm_history) if brainstorm_history else ""
        content = call_persona(p_name, p_data, hist_text)

        _stop_tool_spinner()
        if content:
            brainstorm_history.append(content)
            full_log.append(f"## {icon} {p_data['role']}\n{content}")
            print(clr("  └─ Perspective captured.", "dim"))
        else:
            err(f"  └─ Failed to capture {p_name} perspective.")

    # Save to file
    final_output = "\n\n".join(full_log)
    out_file.write_text(final_output, encoding="utf-8")
    
    ok(f"Brainstorming complete! Results saved to {clr(str(out_file), 'bold')}")
    
    # ── Synthetic Injection ──────────────────────────────────────────────
    info(clr("Injecting debate results into current session for final analysis...", "dim"))

    synthesis_prompt = f"""I have just completed a multi-agent brainstorming session regarding: '{user_topic}'.
The full debate results have been saved to the file: {out_file}

Please read that file, then analyze the diverse perspectives. Identify the strongest ideas, potential conflicts, and provide a synthesized 'Master Plan' with concrete phases. Be concise and actionable."""

    # Return sentinel to trigger synthesis via run_query in the main REPL loop
    # Pass out_file so the REPL can append the synthesis to the same file.
    return ("__brainstorm__", synthesis_prompt, str(out_file))

def _save_synthesis(state, out_file: str) -> None:
    """Append the last assistant response as the synthesis section of the brainstorm file."""
    from pathlib import Path
    for msg in reversed(state.messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            return
        text = text.strip()
        if not text:
            return
        try:
            with Path(out_file).open("a", encoding="utf-8") as f:
                f.write("\n\n---\n\n## 🧠 Synthesis — Master Plan\n\n")
                f.write(text)
                f.write("\n")
            ok(f"Synthesis appended to {clr(out_file, 'bold')}")
        except Exception as e:
            err(f"Failed to save synthesis: {e}")
        return


def cmd_clear(_args: str, state, _config) -> bool:
    state.messages.clear()
    state.turn_count = 0
    ok("Conversation cleared.")
    return True

def cmd_config(args: str, _state, config) -> bool:
    from config import save_config
    if not args:
        display = {k: v for k, v in config.items() if k != "api_key"}
        print(json.dumps(display, indent=2))
    elif "=" in args:
        key, _, val = args.partition("=")
        key, val = key.strip(), val.strip()
        # Type coercion
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.isdigit():
            val = int(val)
        config[key] = val
        save_config(config)
        ok(f"Set {key} = {val}")
    else:
        k = args.strip()
        v = config.get(k, "(not set)")
        info(f"{k} = {v}")
    return True

def cmd_save(args: str, state, _config) -> bool:
    from config import SESSIONS_DIR
    import uuid
    sid   = uuid.uuid4().hex[:8]
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = args.strip() or f"session_{ts}_{sid}.json"
    path  = Path(fname) if "/" in fname else SESSIONS_DIR / fname
    data  = _build_session_data(state, session_id=sid)
    path.write_text(json.dumps(data, indent=2, default=str))
    ok(f"Session saved → {path}  (id: {sid})"  )
    return True

def save_latest(args: str, state, config_or_none=None) -> bool:
    """Save session on exit: session_latest.json + daily/ copy + append to history.json."""
    from config import MR_SESSION_DIR, DAILY_DIR, SESSION_HIST_FILE
    if not state.messages:
        return True

    cfg = config_or_none or {}
    daily_limit   = cfg.get("session_daily_limit",   5)
    history_limit = cfg.get("session_history_limit", 100)

    import uuid
    now = datetime.now()
    sid = uuid.uuid4().hex[:8]
    ts  = now.strftime("%H%M%S")
    date_str = now.strftime("%Y-%m-%d")
    data = _build_session_data(state, session_id=sid)
    payload = json.dumps(data, indent=2, default=str)

    # 1. session_latest.json — always overwrite for quick /resume
    MR_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = MR_SESSION_DIR / "session_latest.json"
    latest_path.write_text(payload)

    # 2. daily/YYYY-MM-DD/session_HHMMSS_sid.json
    day_dir = DAILY_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    daily_path = day_dir / f"session_{ts}_{sid}.json"
    daily_path.write_text(payload)

    # Prune daily folder: keep only the latest `daily_limit` files
    daily_files = sorted(day_dir.glob("session_*.json"))
    for old in daily_files[:-daily_limit]:
        old.unlink(missing_ok=True)

    # 3. Append to history.json (master file)
    if SESSION_HIST_FILE.exists():
        try:
            hist = json.loads(SESSION_HIST_FILE.read_text())
        except Exception:
            hist = {"total_turns": 0, "sessions": []}
    else:
        hist = {"total_turns": 0, "sessions": []}

    hist["sessions"].append(data)
    hist["total_turns"] = sum(s.get("turn_count", 0) for s in hist["sessions"])

    # Prune history: keep only the latest `history_limit` sessions
    if len(hist["sessions"]) > history_limit:
        hist["sessions"] = hist["sessions"][-history_limit:]

    SESSION_HIST_FILE.write_text(json.dumps(hist, indent=2, default=str))

    ok(f"Session saved → {latest_path}")
    ok(f"             → {daily_path}  (id: {sid})")
    ok(f"             → {SESSION_HIST_FILE}  ({len(hist['sessions'])} sessions / {hist['total_turns']} total turns)")
    return True
def cmd_load(args: str, state, _config) -> bool:
    from config import SESSIONS_DIR, MR_SESSION_DIR, DAILY_DIR

    path = None
    if not args.strip():
        # Collect sessions from daily/ folders, newest first
        sessions: list[Path] = []
        if DAILY_DIR.exists():
            for day_dir in sorted(DAILY_DIR.iterdir(), reverse=True):
                if day_dir.is_dir():
                    sessions.extend(sorted(day_dir.glob("session_*.json"), reverse=True))
        # Fall back to legacy mr_sessions/ if daily/ is empty
        if not sessions and MR_SESSION_DIR.exists():
            sessions = [s for s in sorted(MR_SESSION_DIR.glob("*.json"), reverse=True)
                        if s.name != "session_latest.json"]
        # Also include manually /save'd sessions from SESSIONS_DIR root
        sessions.extend(sorted(SESSIONS_DIR.glob("session_*.json"), reverse=True))

        if not sessions:
            info("No saved sessions found.")
            return True

        print(clr("  Select a session to load:", "cyan", "bold"))
        prev_date = None
        for i, s in enumerate(sessions):
            # Group by date header
            date_label = s.parent.name if s.parent.name != "mr_sessions" else ""
            if date_label and date_label != prev_date:
                print(clr(f"\n  ── {date_label} ──", "dim"))
                prev_date = date_label

            label = s.name
            try:
                meta     = json.loads(s.read_text())
                saved_at = meta.get("saved_at", "")[-8:]   # HH:MM:SS
                sid      = meta.get("session_id", "")
                turns    = meta.get("turn_count", "?")
                label    = f"{saved_at}  id:{sid}  turns:{turns}  {s.name}"
            except Exception:
                pass
            print(clr(f"  [{i+1:2d}] ", "yellow") + label)

        # Show history.json option at the bottom if it exists
        from config import SESSION_HIST_FILE
        has_history = SESSION_HIST_FILE.exists()
        if has_history:
            try:
                hist_meta = json.loads(SESSION_HIST_FILE.read_text())
                n_sess  = len(hist_meta.get("sessions", []))
                n_turns = hist_meta.get("total_turns", 0)
                print(clr(f"\n  ── Complete History ──", "dim"))
                print(clr("  [ H] ", "yellow") +
                      f"Load ALL history  ({n_sess} sessions / {n_turns} total turns)  {SESSION_HIST_FILE}")
            except Exception:
                has_history = False

        print()
        ans = input(clr("  Enter number(s) (e.g. 1 or 1,2,3), H for full history, or Enter to cancel > ", "cyan")).strip().lower()

        if not ans:
            info("  Cancelled.")
            return True

        if ans == "h":
            if not has_history:
                err("history.json not found.")
                return True
            hist_data = json.loads(SESSION_HIST_FILE.read_text())
            all_sessions = hist_data.get("sessions", [])
            if not all_sessions:
                info("history.json is empty.")
                return True
            all_messages = []
            for s in all_sessions:
                all_messages.extend(s.get("messages", []))
            total_turns = sum(s.get("turn_count", 0) for s in all_sessions)
            est_tokens = sum(len(str(m.get("content", ""))) for m in all_messages) // 4
            print()
            print(clr(f"  {len(all_messages)} messages / ~{est_tokens:,} tokens estimated", "dim"))
            confirm = input(clr("  Load full history into current session? [y/N] > ", "yellow")).strip().lower()
            if confirm != "y":
                info("  Cancelled.")
                return True
            state.messages = all_messages
            state.turn_count = total_turns
            ok(f"Full history loaded from {SESSION_HIST_FILE} ({len(all_messages)} messages across {len(all_sessions)} sessions)")
            return True

        # Parse comma-separated numbers (e.g. "1", "1,2,3", "1, 3")
        raw_parts = [p.strip() for p in ans.split(",")]
        indices = []
        for p in raw_parts:
            if not p.isdigit():
                err(f"Invalid input '{p}'. Enter numbers separated by commas, or H.")
                return True
            idx = int(p) - 1
            if idx < 0 or idx >= len(sessions):
                err(f"Invalid selection: {p} (valid range: 1–{len(sessions)})")
                return True
            if idx not in indices:
                indices.append(idx)

        if len(indices) == 1:
            # Single session — load directly
            path = sessions[indices[0]]
        else:
            # Multiple sessions — merge in selected order
            all_messages = []
            total_turns  = 0
            loaded_names = []
            for idx in indices:
                s_path = sessions[idx]
                s_data = json.loads(s_path.read_text())
                all_messages.extend(s_data.get("messages", []))
                total_turns += s_data.get("turn_count", 0)
                loaded_names.append(s_path.name)
            est_tokens = sum(len(str(m.get("content", ""))) for m in all_messages) // 4
            print()
            print(clr(f"  {len(loaded_names)} sessions / {len(all_messages)} messages / ~{est_tokens:,} tokens estimated", "dim"))
            confirm = input(clr("  Merge and load? [y/N] > ", "yellow")).strip().lower()
            if confirm != "y":
                info("  Cancelled.")
                return True
            state.messages = all_messages
            state.turn_count = total_turns
            ok(f"Loaded {len(loaded_names)} sessions ({len(all_messages)} messages): {', '.join(loaded_names)}")
            return True

    if not path:
        fname = args.strip()
        path = Path(fname) if "/" in fname or "\\" in fname else SESSIONS_DIR / fname
        if not path.exists() and ("/" not in fname and "\\" not in fname):
            for alt in [MR_SESSION_DIR / fname,
                        *(d / fname for d in DAILY_DIR.iterdir()
                          if DAILY_DIR.exists() and d.is_dir())]:
                if alt.exists():
                    path = alt
                    break
        if not path.exists():
            err(f"File not found: {path}")
            return True
        
    data = json.loads(path.read_text())
    state.messages = data.get("messages", [])
    state.turn_count = data.get("turn_count", 0)
    state.total_input_tokens = data.get("total_input_tokens", 0)
    state.total_output_tokens = data.get("total_output_tokens", 0)
    ok(f"Session loaded from {path} ({len(state.messages)} messages)")
    return True

def cmd_resume(args: str, state, _config) -> bool:
    from config import MR_SESSION_DIR

    if not args.strip():
        path = MR_SESSION_DIR / "session_latest.json"
        if not path.exists():
            info("No auto-saved sessions found.")
            return True
    else:
        fname = args.strip()
        path = Path(fname) if "/" in fname else MR_SESSION_DIR / fname

    if not path.exists():
        err(f"File not found: {path}")
        return True

    data = json.loads(path.read_text())
    state.messages = data.get("messages", [])
    state.turn_count = data.get("turn_count", 0)
    state.total_input_tokens = data.get("total_input_tokens", 0)
    state.total_output_tokens = data.get("total_output_tokens", 0)
    ok(f"Session loaded from {path} ({len(state.messages)} messages)")
    return True

def cmd_history(_args: str, state, _config) -> bool:
    if not state.messages:
        info("(empty conversation)")
        return True
    for i, m in enumerate(state.messages):
        role = clr(m["role"].upper(), "bold",
                   "cyan" if m["role"] == "user" else "green")
        content = m["content"]
        if isinstance(content, str):
            print(f"[{i}] {role}: {content[:200]}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                else:
                    btype = getattr(block, "type", "")
                if btype == "text":
                    text = block.get("text", "") if isinstance(block, dict) else block.text
                    print(f"[{i}] {role}: {text[:200]}")
                elif btype == "tool_use":
                    name = block.get("name", "") if isinstance(block, dict) else block.name
                    print(f"[{i}] {role}: [tool_use: {name}]")
                elif btype == "tool_result":
                    cval = block.get("content", "") if isinstance(block, dict) else block.content
                    print(f"[{i}] {role}: [tool_result: {str(cval)[:100]}]")
    return True

def cmd_context(_args: str, state, config) -> bool:
    import anthropic
    # Rough token estimate: 4 chars ≈ 1 token
    msg_chars = sum(
        len(str(m.get("content", ""))) for m in state.messages
    )
    est_tokens = msg_chars // 4
    info(f"Messages:         {len(state.messages)}")
    info(f"Estimated tokens: ~{est_tokens:,}")
    info(f"Model:            {config['model']}")
    info(f"Max tokens:       {config['max_tokens']:,}")
    return True

def cmd_cost(_args: str, state, config) -> bool:
    from config import calc_cost
    cost = calc_cost(config["model"],
                     state.total_input_tokens,
                     state.total_output_tokens)
    info(f"Input tokens:  {state.total_input_tokens:,}")
    info(f"Output tokens: {state.total_output_tokens:,}")
    info(f"Est. cost:     ${cost:.4f} USD")
    return True

def cmd_verbose(_args: str, _state, config) -> bool:
    from config import save_config
    config["verbose"] = not config.get("verbose", False)
    state_str = "ON" if config["verbose"] else "OFF"
    ok(f"Verbose mode: {state_str}")
    save_config(config)
    return True

def cmd_thinking(_args: str, _state, config) -> bool:
    from config import save_config
    config["thinking"] = not config.get("thinking", False)
    state_str = "ON" if config["thinking"] else "OFF"
    ok(f"Extended thinking: {state_str}")
    save_config(config)
    return True

def cmd_permissions(args: str, _state, config) -> bool:
    from config import save_config
    modes = ["auto", "accept-all", "manual"]
    if not args.strip():
        info(f"Permission mode: {config.get('permission_mode','auto')}")
        info(f"Available modes: {', '.join(modes)}")
    else:
        m = args.strip()
        if m not in modes:
            err(f"Unknown mode: {m}. Choose: {', '.join(modes)}")
        else:
            config["permission_mode"] = m
            save_config(config)
            ok(f"Permission mode set to: {m}")
    return True

def cmd_cwd(args: str, _state, _config) -> bool:
    if not args.strip():
        info(f"Working directory: {os.getcwd()}")
    else:
        p = args.strip()
        try:
            os.chdir(p)
            ok(f"Changed directory to: {os.getcwd()}")
        except Exception as e:
            err(str(e))
    return True

def _build_session_data(state, session_id: str | None = None) -> dict:
    """Serialize current conversation state to a JSON-serializable dict."""
    import uuid
    return {
        "session_id": session_id or uuid.uuid4().hex[:8],
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [
            m if not isinstance(m.get("content"), list) else
            {**m, "content": [
                b if isinstance(b, dict) else b.model_dump()
                for b in m["content"]
            ]}
            for m in state.messages
        ],
        "turn_count": state.turn_count,
        "total_input_tokens": state.total_input_tokens,
        "total_output_tokens": state.total_output_tokens,
    }


def cmd_cloudsave(args: str, state, config) -> bool:
    """Sync sessions to GitHub Gist.

    /cloudsave setup <token>   — configure GitHub Personal Access Token
    /cloudsave                 — upload current session to Gist
    /cloudsave push [desc]     — same as above with optional description
    /cloudsave auto on|off     — toggle auto-upload on /exit
    /cloudsave list            — list your nano-claude-code Gists
    /cloudsave load <gist_id>  — download and load a session from Gist
    """
    from cloudsave import validate_token, upload_session, list_sessions, download_session
    from config import save_config

    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    token = config.get("gist_token", "")

    # ── setup ──────────────────────────────────────────────────────────────────
    if sub == "setup":
        if not rest:
            err("Usage: /cloudsave setup <GitHub_Personal_Access_Token>")
            return True
        new_token = rest.strip()
        info("Validating token…")
        valid, msg = validate_token(new_token)
        if not valid:
            err(msg)
            return True
        config["gist_token"] = new_token
        save_config(config)
        ok(f"GitHub token saved (logged in as: {msg}). Cloud sync is ready.")
        return True

    # ── auto on/off ────────────────────────────────────────────────────────────
    if sub == "auto":
        flag = rest.strip().lower()
        if flag == "on":
            config["cloudsave_auto"] = True
            save_config(config)
            ok("Auto cloud-sync ON — session will be uploaded to Gist on /exit.")
        elif flag == "off":
            config["cloudsave_auto"] = False
            save_config(config)
            ok("Auto cloud-sync OFF.")
        else:
            status = "ON" if config.get("cloudsave_auto") else "OFF"
            info(f"Auto cloud-sync is currently {status}. Use 'on' or 'off' to toggle.")
        return True

    # ── remaining subcommands require a token ─────────────────────────────────
    if not token:
        err("No GitHub token configured. Run: /cloudsave setup <token>")
        info("Get a token at https://github.com/settings/tokens (needs 'gist' scope)")
        return True

    # ── list ───────────────────────────────────────────────────────────────────
    if sub == "list":
        info("Fetching your nano-claude-code sessions from GitHub Gist…")
        sessions, err_msg = list_sessions(token)
        if err_msg:
            err(err_msg)
            return True
        if not sessions:
            info("No sessions found. Upload one with /cloudsave")
            return True
        info(f"Found {len(sessions)} session(s):")
        for s in sessions:
            ts = s["updated_at"][:16].replace("T", " ")
            desc = s["description"].replace("[nano-claude-code]", "").strip()
            print(f"  {clr(s['id'][:8], 'yellow')}…  {clr(ts, 'dim')}  {desc or s['files'][0]}")
        return True

    # ── load ───────────────────────────────────────────────────────────────────
    if sub == "load":
        gist_id = rest.strip()
        if not gist_id:
            err("Usage: /cloudsave load <gist_id>")
            return True
        info(f"Downloading session {gist_id[:8]}… from Gist…")
        data, err_msg = download_session(token, gist_id)
        if err_msg:
            err(err_msg)
            return True
        state.messages = data.get("messages", [])
        state.turn_count = data.get("turn_count", 0)
        state.total_input_tokens = data.get("total_input_tokens", 0)
        state.total_output_tokens = data.get("total_output_tokens", 0)
        ok(f"Session loaded from Gist ({len(state.messages)} messages).")
        return True

    # ── push (default when no subcommand or sub == "push") ────────────────────
    if sub in ("", "push"):
        description = rest.strip() if sub == "push" else ""
        if not state.messages:
            info("Nothing to save — conversation is empty.")
            return True
        info("Uploading session to GitHub Gist…")
        session_data = _build_session_data(state)
        existing_id = config.get("cloudsave_last_gist_id")
        gist_id, err_msg = upload_session(session_data, token, description, existing_id)
        if err_msg:
            err(f"Upload failed: {err_msg}")
            return True
        config["cloudsave_last_gist_id"] = gist_id
        save_config(config)
        ok(f"Session uploaded → https://gist.github.com/{gist_id}")
        return True

    err(f"Unknown subcommand '{sub}'. Run /help for usage.")
    return True


def cmd_exit(_args: str, _state, _config) -> bool:
    if sys.stdin.isatty() and sys.platform != "win32":
        sys.stdout.write("\x1b[?2004l")  # disable bracketed paste mode
        sys.stdout.flush()
    ok("Goodbye!")
    save_latest("", _state, _config)
    # Auto cloud-sync if enabled
    if _config.get("cloudsave_auto") and _config.get("gist_token") and _state.messages:
        info("Auto cloud-sync: uploading session to Gist…")
        from cloudsave import upload_session
        from config import save_config
        session_data = _build_session_data(_state)
        gist_id, err_msg = upload_session(
            session_data, _config["gist_token"],
            existing_gist_id=_config.get("cloudsave_last_gist_id"),
        )
        if err_msg:
            err(f"Cloud sync failed: {err_msg}")
        else:
            _config["cloudsave_last_gist_id"] = gist_id
            save_config(_config)
            ok(f"Session synced → https://gist.github.com/{gist_id}")
    sys.exit(0)

def cmd_memory(args: str, _state, _config) -> bool:
    from memory import search_memory, load_index
    from memory.scan import scan_all_memories, format_memory_manifest, memory_freshness_text

    stripped = args.strip()

    # /memory consolidate  — extract long-term memories from current session
    if stripped == "consolidate":
        from memory import consolidate_session
        msgs = _state.get("messages", [])
        info("  Analyzing session for long-term memories…")
        saved = consolidate_session(msgs, _config)
        if saved:
            info(f"  ✓ Consolidated {len(saved)} memory/memories: {', '.join(saved)}")
        else:
            info("  Nothing new worth saving (session too short, or nothing extractable).")
        return True

    if stripped:
        results = search_memory(stripped)
        if not results:
            info(f"No memories matching '{stripped}'")
            return True
        info(f"  {len(results)} result(s) for '{stripped}':")
        for m in results:
            conf_tag = f" conf:{m.confidence:.0%}" if m.confidence < 1.0 else ""
            src_tag = f" src:{m.source}" if m.source and m.source != "user" else ""
            info(f"  [{m.type:9s}|{m.scope:7s}] {m.name}{conf_tag}{src_tag}: {m.description}")
            info(f"    {m.content[:120]}{'...' if len(m.content) > 120 else ''}")
        return True

    # Show manifest with age/freshness
    headers = scan_all_memories()
    if not headers:
        info("No memories stored. The model saves memories via MemorySave.")
        return True
    info(f"  {len(headers)} memory/memories (newest first):")
    for h in headers:
        fresh_warn = "  ⚠ stale" if memory_freshness_text(h.mtime_s) else ""
        tag = f"[{h.type or '?':9s}|{h.scope:7s}]"
        info(f"  {tag} {h.filename}{fresh_warn}")
        if h.description:
            info(f"    {h.description}")
    return True

def cmd_agents(_args: str, _state, _config) -> bool:
    try:
        from multi_agent.tools import get_agent_manager
        mgr = get_agent_manager()
        tasks = mgr.list_tasks()
        if not tasks:
            info("No sub-agent tasks.")
            return True
        info(f"  {len(tasks)} sub-agent task(s):")
        for t in tasks:
            preview = t.prompt[:50] + ("..." if len(t.prompt) > 50 else "")
            wt_info = f"  branch:{t.worktree_branch}" if t.worktree_branch else ""
            info(f"  {t.id} [{t.status:9s}] name={t.name}{wt_info}  {preview}")
    except Exception:
        info("Sub-agent system not initialized.")
    return True


def _print_background_notifications():
    """Print notifications for newly completed background agent tasks.

    Called before each user prompt so the user sees results without polling.
    """
    try:
        from multi_agent.tools import get_agent_manager
        mgr = get_agent_manager()
    except Exception:
        return

    notified_key = "_notified"
    if not hasattr(_print_background_notifications, "_seen"):
        _print_background_notifications._seen = set()

    for task in mgr.list_tasks():
        if task.id in _print_background_notifications._seen:
            continue
        if task.status in ("completed", "failed", "cancelled"):
            _print_background_notifications._seen.add(task.id)
            icon = "✓" if task.status == "completed" else "✗"
            color = "green" if task.status == "completed" else "red"
            branch_info = f" [branch: {task.worktree_branch}]" if task.worktree_branch else ""
            print(clr(
                f"\n  {icon} Background agent '{task.name}' {task.status}{branch_info}",
                color, "bold"
            ))
            if task.result:
                preview = task.result[:200] + ("..." if len(task.result) > 200 else "")
                print(clr(f"    {preview}", "dim"))
            print()

def cmd_skills(_args: str, _state, _config) -> bool:
    from skill import load_skills
    skills = load_skills()
    if not skills:
        info("No skills found.")
        return True
    info(f"Available skills ({len(skills)}):")
    for s in skills:
        triggers = ", ".join(s.triggers)
        source_label = f"[{s.source}]" if s.source != "builtin" else ""
        hint = f"  args: {s.argument_hint}" if s.argument_hint else ""
        print(f"  {clr(s.name, 'cyan'):24s} {s.description}  {clr(triggers, 'dim')}{hint} {clr(source_label, 'yellow')}")
        if s.when_to_use:
            print(f"    {clr(s.when_to_use[:80], 'dim')}")
    return True

def cmd_mcp(args: str, _state, _config) -> bool:
    """Show MCP server status, or manage servers.

    /mcp               — list all configured servers and their tools
    /mcp reload        — reconnect all servers and refresh tools
    /mcp reload <name> — reconnect a single server
    /mcp add <name> <command> [args...] — add a stdio server to user config
    /mcp remove <name> — remove a server from user config
    """
    from mcp.client import get_mcp_manager
    from mcp.config import (load_mcp_configs, add_server_to_user_config,
                             remove_server_from_user_config, list_config_files)
    from mcp.tools import initialize_mcp, reload_mcp, refresh_server

    parts = args.split() if args.strip() else []
    subcmd = parts[0].lower() if parts else ""

    if subcmd == "reload":
        target = parts[1] if len(parts) > 1 else ""
        if target:
            err = refresh_server(target)
            if err:
                err(f"Failed to reload '{target}': {err}")
            else:
                ok(f"Reloaded MCP server: {target}")
        else:
            errors = reload_mcp()
            for name, e in errors.items():
                if e:
                    print(f"  {clr('✗', 'red')} {name}: {e}")
                else:
                    print(f"  {clr('✓', 'green')} {name}: connected")
        return True

    if subcmd == "add":
        if len(parts) < 3:
            err("Usage: /mcp add <name> <command> [arg1 arg2 ...]")
            return True
        name = parts[1]
        command = parts[2]
        cmd_args = parts[3:]
        raw = {"type": "stdio", "command": command}
        if cmd_args:
            raw["args"] = cmd_args
        add_server_to_user_config(name, raw)
        ok(f"Added MCP server '{name}' → restart or /mcp reload to connect")
        return True

    if subcmd == "remove":
        if len(parts) < 2:
            err("Usage: /mcp remove <name>")
            return True
        name = parts[1]
        removed = remove_server_from_user_config(name)
        if removed:
            ok(f"Removed MCP server '{name}' from user config")
        else:
            err(f"Server '{name}' not found in user config")
        return True

    # Default: list servers
    mgr = get_mcp_manager()
    servers = mgr.list_servers()

    config_files = list_config_files()
    if config_files:
        info(f"Config files: {', '.join(str(f) for f in config_files)}")

    if not servers:
        configs = load_mcp_configs()
        if not configs:
            info("No MCP servers configured.")
            info("Add servers in ~/.nano_claude/mcp.json or .mcp.json")
            info("Example: /mcp add my-git uvx mcp-server-git")
        else:
            info("MCP servers configured but not yet connected. Run /mcp reload")
        return True

    info(f"MCP servers ({len(servers)}):")
    total_tools = 0
    for client in servers:
        status_color = {
            "connected":    "green",
            "connecting":   "yellow",
            "disconnected": "dim",
            "error":        "red",
        }.get(client.state.value, "dim")
        print(f"  {clr(client.status_line(), status_color)}")
        for tool in client._tools:
            print(f"      {clr(tool.qualified_name, 'cyan')}  {tool.description[:60]}")
            total_tools += 1

    if total_tools:
        info(f"Total: {total_tools} MCP tool(s) available to Claude")
    return True


def cmd_plugin(args: str, _state, _config) -> bool:
    """Manage plugins.

    /plugin                      — list installed plugins
    /plugin install name@url     — install a plugin
    /plugin uninstall name       — uninstall a plugin
    /plugin enable name          — enable a plugin
    /plugin disable name         — disable a plugin
    /plugin disable-all          — disable all plugins
    /plugin update name          — update a plugin from its source
    /plugin recommend [context]  — recommend plugins for context
    /plugin info name            — show plugin details
    """
    from plugin import (
        install_plugin, uninstall_plugin, enable_plugin, disable_plugin,
        disable_all_plugins, update_plugin, list_plugins, get_plugin,
        PluginScope, recommend_plugins, format_recommendations,
    )

    parts = args.split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    rest   = parts[1].strip() if len(parts) > 1 else ""

    if not subcmd:
        # List all plugins
        plugins = list_plugins()
        if not plugins:
            info("No plugins installed.")
            info("Install: /plugin install name@git_url")
            info("Recommend: /plugin recommend")
            return True
        info(f"Installed plugins ({len(plugins)}):")
        for p in plugins:
            state_color = "green" if p.enabled else "dim"
            state_str   = "enabled" if p.enabled else "disabled"
            desc = p.manifest.description if p.manifest else ""
            print(f"  {clr(p.name, state_color)} [{p.scope.value}] {state_str}  {desc[:60]}")
        return True

    if subcmd == "install":
        if not rest:
            err("Usage: /plugin install name@git_url")
            return True
        scope_str = "user"
        if " --project" in rest:
            scope_str = "project"
            rest = rest.replace("--project", "").strip()
        scope = PluginScope(scope_str)
        success, msg = install_plugin(rest, scope=scope)
        (ok if success else err)(msg)
        return True

    if subcmd == "uninstall":
        if not rest:
            err("Usage: /plugin uninstall name")
            return True
        success, msg = uninstall_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "enable":
        if not rest:
            err("Usage: /plugin enable name")
            return True
        success, msg = enable_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "disable":
        if not rest:
            err("Usage: /plugin disable name")
            return True
        success, msg = disable_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "disable-all":
        success, msg = disable_all_plugins()
        (ok if success else err)(msg)
        return True

    if subcmd == "update":
        if not rest:
            err("Usage: /plugin update name")
            return True
        success, msg = update_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "recommend":
        from pathlib import Path as _Path
        context = rest
        if not context:
            # Auto-detect context from project files
            from plugin.recommend import recommend_from_files
            files = list(_Path.cwd().glob("**/*"))[:200]
            recs = recommend_from_files(files)
        else:
            recs = recommend_plugins(context)
        print(format_recommendations(recs))
        return True

    if subcmd == "info":
        if not rest:
            err("Usage: /plugin info name")
            return True
        entry = get_plugin(rest)
        if entry is None:
            err(f"Plugin '{rest}' not found.")
            return True
        m = entry.manifest
        print(f"Name:    {entry.name}")
        print(f"Scope:   {entry.scope.value}")
        print(f"Source:  {entry.source}")
        print(f"Dir:     {entry.install_dir}")
        print(f"Enabled: {entry.enabled}")
        if m:
            print(f"Version: {m.version}")
            print(f"Author:  {m.author}")
            print(f"Desc:    {m.description}")
            if m.tags:
                print(f"Tags:    {', '.join(m.tags)}")
            if m.tools:
                print(f"Tools:   {', '.join(m.tools)}")
            if m.skills:
                print(f"Skills:  {', '.join(m.skills)}")
        return True

    err(f"Unknown plugin subcommand: {subcmd}  (try /plugin or /help)")
    return True


def cmd_tasks(args: str, _state, _config) -> bool:
    """Show and manage tasks.

    /tasks                  — list all tasks
    /tasks create <subject> — quick-create a task
    /tasks done <id>        — mark task completed
    /tasks start <id>       — mark task in_progress
    /tasks cancel <id>      — mark task cancelled
    /tasks delete <id>      — delete a task
    /tasks get <id>         — show full task details
    /tasks clear            — delete all tasks
    """
    from task import list_tasks, get_task, create_task, update_task, delete_task, clear_all_tasks
    from task.types import TaskStatus

    parts = args.split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    rest   = parts[1].strip() if len(parts) > 1 else ""

    STATUS_MAP = {
        "done":   "completed",
        "start":  "in_progress",
        "cancel": "cancelled",
    }

    if not subcmd:
        tasks = list_tasks()
        if not tasks:
            info("No tasks. Use TaskCreate tool or /tasks create <subject>.")
            return True
        resolved = {t.id for t in tasks if t.status == TaskStatus.COMPLETED}
        total = len(tasks)
        done  = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        info(f"Tasks ({done}/{total} completed):")
        for t in tasks:
            pending_blockers = [b for b in t.blocked_by if b not in resolved]
            owner_str   = f" {clr(f'({t.owner})', 'dim')}" if t.owner else ""
            blocked_str = clr(f" [blocked by #{', #'.join(pending_blockers)}]", "yellow") if pending_blockers else ""
            status_color = {
                TaskStatus.PENDING:     "dim",
                TaskStatus.IN_PROGRESS: "cyan",
                TaskStatus.COMPLETED:   "green",
                TaskStatus.CANCELLED:   "red",
            }.get(t.status, "dim")
            icon = t.status_icon()
            print(f"  #{t.id} {clr(icon + ' ' + t.status.value, status_color)} {t.subject}{owner_str}{blocked_str}")
        return True

    if subcmd == "create":
        if not rest:
            err("Usage: /tasks create <subject>")
            return True
        t = create_task(rest, description="(created via REPL)")
        ok(f"Task #{t.id} created: {t.subject}")
        return True

    if subcmd in STATUS_MAP:
        new_status = STATUS_MAP[subcmd]
        if not rest:
            err(f"Usage: /tasks {subcmd} <task_id>")
            return True
        task, fields = update_task(rest, status=new_status)
        if task is None:
            err(f"Task #{rest} not found.")
        else:
            ok(f"Task #{task.id} → {new_status}: {task.subject}")
        return True

    if subcmd == "delete":
        if not rest:
            err("Usage: /tasks delete <task_id>")
            return True
        removed = delete_task(rest)
        if removed:
            ok(f"Task #{rest} deleted.")
        else:
            err(f"Task #{rest} not found.")
        return True

    if subcmd == "get":
        if not rest:
            err("Usage: /tasks get <task_id>")
            return True
        t = get_task(rest)
        if t is None:
            err(f"Task #{rest} not found.")
            return True
        print(f"  #{t.id} [{t.status.value}] {t.subject}")
        print(f"  Description: {t.description}")
        if t.owner:         print(f"  Owner:       {t.owner}")
        if t.active_form:   print(f"  Active form: {t.active_form}")
        if t.blocked_by:    print(f"  Blocked by:  #{', #'.join(t.blocked_by)}")
        if t.blocks:        print(f"  Blocks:      #{', #'.join(t.blocks)}")
        if t.metadata:      print(f"  Metadata:    {t.metadata}")
        print(f"  Created: {t.created_at[:19]}  Updated: {t.updated_at[:19]}")
        return True

    if subcmd == "clear":
        clear_all_tasks()
        ok("All tasks deleted.")
        return True

    err(f"Unknown tasks subcommand: {subcmd}  (try /tasks or /help)")
    return True


# ── SSJ Developer Mode ─────────────────────────────────────────────────────

def cmd_ssj(args: str, state, config) -> bool:
    """SSJ Developer Mode — Interactive power menu for project workflows.

    Usage: /ssj
    """
    _SSJ_MENU = (
        clr("\n╭─ SSJ Developer Mode ", "dim") + clr("⚡", "yellow") + clr(" ─────────────────────────", "dim")
        + "\n│"
        + "\n│  " + clr(" 1.", "bold") + " 💡  Brainstorm — Multi-persona AI debate"
        + "\n│  " + clr(" 2.", "bold") + " 📋  Show TODO — View todo_list.txt"
        + "\n│  " + clr(" 3.", "bold") + " 👷  Worker — Auto-implement pending tasks"
        + "\n│  " + clr(" 4.", "bold") + " 🧠  Debate — Expert debate on a file"
        + "\n│  " + clr(" 5.", "bold") + " ✨  Propose — AI improvement for a file"
        + "\n│  " + clr(" 6.", "bold") + " 🔎  Review — Quick file analysis"
        + "\n│  " + clr(" 7.", "bold") + " 📘  Readme — Auto-generate README.md"
        + "\n│  " + clr(" 8.", "bold") + " 💬  Commit — AI-suggested commit message"
        + "\n│  " + clr(" 9.", "bold") + " 🧪  Scan — Analyze git diff"
        + "\n│  " + clr("10.", "bold") + " 📝  Promote — Idea to tasks"
        + "\n│  " + clr(" 0.", "bold") + " 🚪  Exit SSJ Mode"
        + "\n│"
        + "\n" + clr("╰──────────────────────────────────────────────", "dim")
    )

    from pathlib import Path

    def _pick_file(prompt_text="  Select file #: ", exts=None):
        """Show numbered file list and let user pick one."""
        files = sorted([
            f for f in Path(".").iterdir()
            if f.is_file() and not f.name.startswith(".")
            and (exts is None or f.suffix in exts)
        ])
        if not files:
            err("No matching files found in current directory.")
            return None
        print(clr(f"\n  📂 Files in {Path.cwd().name}/", "cyan"))
        for i, f in enumerate(files, 1):
            print(f"  {i:3d}. {f.name}")
        sel = input(clr(prompt_text, "cyan")).strip()
        if sel.isdigit() and 1 <= int(sel) <= len(files):
            return str(files[int(sel) - 1])
        elif sel:  # typed a filename directly
            return sel
        err("Invalid selection.")
        return None

    print(_SSJ_MENU)

    while True:
        try:
            choice = input(clr("\n  ⚡ SSJ » ", "yellow", "bold")).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice.startswith("/"):
            # Pass slash commands through to nano — exit SSJ and let REPL handle it
            return ("__ssj_passthrough__", choice)

        if choice == "0" or choice.lower() in ("exit", "q"):
            ok("Exiting SSJ Mode.")
            break

        elif choice == "1":
            topic = input(clr("  Topic (Enter for general): ", "cyan")).strip()
            return ("__ssj_cmd__", "brainstorm", topic)

        elif choice == "2":
            todo_path = Path("todo_list.txt")
            if todo_path.exists():
                content = todo_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                task_lines = [(i, l) for i, l in enumerate(lines) if l.strip().startswith("- [")]
                pending = sum(1 for _, l in task_lines if l.strip().startswith("- [ ]"))
                done = sum(1 for _, l in task_lines if l.strip().startswith("- [x]"))
                print(clr(f"\n  📋 TODO List ({done} done / {pending} pending):", "cyan"))
                print(clr("  " + "─" * 46, "dim"))
                for num, (_, ln) in enumerate(task_lines, 1):
                    ln_s = ln.strip()
                    if ln_s.startswith("- [x]"):
                        label = ln_s[5:].strip()
                        print(clr(f"  {num:3d}. ✓ {label}", "green"))
                    elif ln_s.startswith("- [ ]"):
                        label = ln_s[5:].strip()
                        print(f"  {num:3d}. ○ {label}")
                print(clr("  " + "─" * 46, "dim"))
                print(clr("  Tip: use Worker (3) with task #s e.g. 1,4,6 to run specific tasks", "dim"))
            else:
                err("No todo_list.txt found. Run Brainstorm (1) first.")

        elif choice == "3":
            task_num = input(clr("  Task # (Enter for all, or e.g. 1,4,6): ", "cyan")).strip()
            return ("__ssj_cmd__", "worker", task_num)

        elif choice == "4":
            filepath = _pick_file("  File to debate #: ")
            if not filepath:
                continue
            return ("__ssj_query__", (
                f"Act as two expert developers with opposing views debating improvements for the file: {filepath}. "
                f"Read the file first. Expert A focuses on architecture and clean code. "
                f"Expert B focuses on performance and pragmatism. "
                f"Have them debate 3 rounds, then produce a final consensus with actionable changes."
            ))

        elif choice == "5":
            filepath = _pick_file("  File to improve #: ")
            if not filepath:
                continue
            return ("__ssj_query__", (
                f"Read {filepath} and propose specific, concrete improvements. "
                f"For each improvement: explain the problem, show the fix, and apply it with Edit if the user approves. "
                f"Focus on bugs, performance, readability, and security. Be concise."
            ))

        elif choice == "6":
            filepath = _pick_file("  File to review #: ")
            if not filepath:
                continue
            return ("__ssj_query__", (
                f"Read {filepath} and provide a thorough code review. "
                f"Rate it 1-10 on: readability, maintainability, performance, security. "
                f"List specific issues with line numbers. Do NOT modify the file, review only."
            ))

        elif choice == "7":
            filepath = _pick_file("  Generate README for file #: ", exts={".py", ".js", ".ts", ".go", ".rs"})
            if not filepath:
                continue
            return ("__ssj_query__", (
                f"Read ONLY the file {filepath}. Based on that single file, generate a professional README.md. "
                f"Include: project description, features, installation, usage with examples, "
                f"and contributing guidelines. Use the Write tool to create README.md. "
                f"Do NOT read other files unless the user explicitly asks."
            ))

        elif choice == "8":
            return ("__ssj_query__", (
                "Run 'git diff --cached' and 'git diff' using Bash, analyze ALL changes, "
                "and suggest a concise, descriptive commit message following conventional commits format. "
                "Show the suggested message and ask for confirmation before committing."
            ))

        elif choice == "9":
            return ("__ssj_query__", (
                "Run 'git status' and 'git diff' using Bash. Analyze the current state of the repository. "
                "Summarize: what files changed, what was added/removed, potential issues in the changes, "
                "and suggest next steps."
            ))

        elif choice == "10":
            brainstorm_dir = Path("brainstorm_outputs")
            if not brainstorm_dir.exists() or not list(brainstorm_dir.glob("*.md")):
                err("No brainstorm outputs found. Run Brainstorm (1) first.")
                continue
            latest = sorted(brainstorm_dir.glob("*.md"))[-1]
            return ("__ssj_query__", (
                f"Read the brainstorm file {latest} and extract all actionable ideas. "
                f"Convert each idea into a task with checkbox format (- [ ] task description). "
                f"Write them to todo_list.txt using the Write tool. Prioritize by impact."
            ))

        else:
            err("Invalid option. Pick 0-10.")

    return True


# ── Worker command ─────────────────────────────────────────────────────────

def cmd_worker(args: str, state, config) -> bool:
    """Auto-implement pending tasks from todo_list.txt one by one.

    Usage: /worker          — work through all pending tasks
           /worker <n>      — implement task number n only

    The worker reads todo_list.txt, picks pending tasks (- [ ]),
    implements them, and marks them as done (- [x]) in the file.
    """
    from pathlib import Path

    todo_path = Path("todo_list.txt")
    if not todo_path.exists():
        err("No todo_list.txt found. Run /brainstorm first to generate one.")
        return True

    content = todo_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    pending = [(i, ln) for i, ln in enumerate(lines) if ln.strip().startswith("- [ ]")]

    if not pending:
        ok("All tasks completed! No pending items in todo_list.txt.")
        return True

    # If specific task numbers are given (e.g. "3" or "1,4,6")
    target = args.strip()
    if target:
        try:
            nums = [int(x.strip()) for x in target.split(",") if x.strip()]
            selected = []
            for n in nums:
                if 1 <= n <= len(pending):
                    selected.append(pending[n - 1])
                else:
                    err(f"Task #{n} out of range (1-{len(pending)}).")
                    return True
            pending = selected
        except ValueError:
            err(f"Invalid task number(s). Use 1-{len(pending)} or e.g. 1,4,6")
            return True

    ok(f"Worker starting — {len(pending)} task(s) to implement")
    info("Pending tasks:")
    for n, (_, ln) in enumerate(pending, 1):
        print(f"  {n}. {ln.strip()}")

    worker_prompts = []
    for line_idx, task_line in pending:
        task_text = task_line.strip().replace("- [ ] ", "", 1)
        prompt = (
            f"You are the Worker. Your job is to implement this task:\n\n"
            f"  {task_text}\n\n"
            f"Instructions:\n"
            f"1. Read the relevant files, understand the codebase.\n"
            f"2. Implement the task — write code, edit files, run tests.\n"
            f"3. When DONE, use the Edit tool to mark this exact line in todo_list.txt:\n"
            f'   Change "- [ ] {task_text}" to "- [x] {task_text}"\n'
            f"4. If you CANNOT complete it, leave it as - [ ] and explain why.\n"
            f"5. Be concise. Act, don't explain."
        )
        worker_prompts.append((line_idx, task_text, prompt))

    return ("__worker__", worker_prompts)


# ── Telegram bot ───────────────────────────────────────────────────────────

_telegram_thread = None
_telegram_stop = threading.Event()

def _tg_api(token: str, method: str, params: dict = None):
    """Call Telegram Bot API. Returns parsed JSON or None on error."""
    import urllib.request, urllib.parse
    url = f"https://api.telegram.org/bot{token}/{method}"
    if params:
        data = json.dumps(params).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def _tg_send(token: str, chat_id: int, text: str):
    """Send a message to a Telegram chat, splitting if too long."""
    MAX = 4000  # Telegram limit is 4096, leave margin
    chunks = [text[i:i+MAX] for i in range(0, len(text), MAX)]
    for chunk in chunks:
        _tg_api(token, "sendMessage", {"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})

def _tg_typing_loop(token: str, chat_id: int, stop_event: threading.Event):
    """Send 'typing...' indicator every 4 seconds until stop_event is set."""
    while not stop_event.is_set():
        _tg_api(token, "sendChatAction", {"chat_id": chat_id, "action": "typing"})
        stop_event.wait(4)

def _tg_poll_loop(token: str, chat_id: int, config: dict):
    """Long-polling loop that reads Telegram messages and feeds them to run_query."""
    run_query_cb = config.get("_run_query_callback")
    # Flush old messages so we don't process stale commands on startup
    flush = _tg_api(token, "getUpdates", {"offset": -1, "timeout": 0})
    if flush and flush.get("ok") and flush.get("result"):
        offset = flush["result"][-1]["update_id"] + 1
    else:
        offset = 0
    # Notify user bot is online
    _tg_send(token, chat_id, "🟢 nano-claude is online.\nSend me a message and I'll process it.")

    while not _telegram_stop.is_set():
        try:
            result = _tg_api(token, "getUpdates", {
                "offset": offset,
                "timeout": 30,
                "allowed_updates": ["message"]
            })
            if not result or not result.get("ok"):
                _telegram_stop.wait(5)
                continue

            for update in result.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                msg_chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "")

                if msg_chat_id != chat_id:
                    _tg_api(token, "sendMessage", {
                        "chat_id": msg_chat_id,
                        "text": "⛔ Unauthorized."
                    })
                    continue

                if not text:
                    continue

                # Handle Telegram bot commands
                if text.strip().startswith("/"):
                    tg_cmd = text.strip().lower()
                    if tg_cmd in ("/stop", "/off"):
                        _tg_send(token, chat_id, "🔴 Telegram bridge stopped.")
                        _telegram_stop.set()
                        break
                    elif tg_cmd == "/start":
                        _tg_send(token, chat_id, "🟢 nano-claude bridge is active. Send me anything.")
                        continue
                    # Pass nano slash commands through handle_slash
                    slash_cb = config.get("_handle_slash_callback")
                    if slash_cb:
                        try:
                            config["_telegram_incoming"] = True
                            cmd_type = slash_cb(text)
                        except Exception as e:
                            _tg_send(token, chat_id, f"⚠ Error: {e}")
                            continue
                        # Simple commands (toggle, etc.) — just confirm
                        if cmd_type == "simple":
                            cmd_name = text.strip().split()[0]
                            _tg_send(token, chat_id, f"✅ {cmd_name} executed.")
                            continue
                        # Query commands — grab the model response
                        tg_state = config.get("_state")
                        if tg_state and tg_state.messages:
                            for m in reversed(tg_state.messages):
                                if m.get("role") == "assistant":
                                    content = m.get("content", "")
                                    if isinstance(content, list):
                                        parts = []
                                        for block in content:
                                            if isinstance(block, dict) and block.get("type") == "text":
                                                parts.append(block["text"])
                                            elif isinstance(block, str):
                                                parts.append(block)
                                        content = "\n".join(parts)
                                    if content:
                                        _tg_send(token, chat_id, content)
                                    break
                    continue

                # Show on local terminal
                print(clr(f"\n  📩 Telegram: {text}", "cyan"))

                # Run through nano's model
                _typing_stop = threading.Event()
                _typing_t = threading.Thread(target=_tg_typing_loop, args=(token, chat_id, _typing_stop), daemon=True)
                _typing_t.start()
                if run_query_cb:
                    try:
                        config["_telegram_incoming"] = True
                        run_query_cb(text)
                    except Exception as e:
                        _typing_stop.set()
                        _tg_send(token, chat_id, f"⚠ Error: {e}")
                        continue
                _typing_stop.set()

                # Grab the last assistant response from state
                state = config.get("_state")
                if state and state.messages:
                    for m in reversed(state.messages):
                        if m.get("role") == "assistant":
                            content = m.get("content", "")
                            if isinstance(content, list):
                                # Extract text blocks from content array
                                parts = []
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        parts.append(block["text"])
                                    elif isinstance(block, str):
                                        parts.append(block)
                                content = "\n".join(parts)
                            if content:
                                _tg_send(token, chat_id, content)
                            break
        except Exception:
            _telegram_stop.wait(5)

    global _telegram_thread
    _telegram_thread = None


def cmd_telegram(args: str, _state, config) -> bool:
    """Telegram bot bridge — receive and respond to messages via Telegram.

    Usage: /telegram <bot_token> <chat_id>   — start the bridge
           /telegram stop                    — stop the bridge
           /telegram status                  — show current status

    First time: create a bot via @BotFather, then send any message to your bot
    and check https://api.telegram.org/bot<TOKEN>/getUpdates to find your chat_id.
    Settings are saved so you only configure once.
    """
    global _telegram_thread, _telegram_stop
    from config import save_config

    parts = args.strip().split()

    # /telegram stop
    if parts and parts[0].lower() in ("stop", "off"):
        if _telegram_thread and _telegram_thread.is_alive():
            _telegram_stop.set()
            _telegram_thread.join(timeout=5)
            _telegram_thread = None
            ok("Telegram bridge stopped.")
        else:
            warn("Telegram bridge is not running.")
        return True

    # /telegram status
    if parts and parts[0].lower() == "status":
        running = _telegram_thread and _telegram_thread.is_alive()
        token = config.get("telegram_token", "")
        chat_id = config.get("telegram_chat_id", "")
        if running:
            ok(f"Telegram bridge is running. Chat ID: {chat_id}")
        elif token:
            info(f"Configured but not running. Use /telegram to start.")
        else:
            info("Not configured. Use /telegram <bot_token> <chat_id>")
        return True

    # /telegram <token> <chat_id> — configure and start
    if len(parts) >= 2:
        token = parts[0]
        try:
            chat_id = int(parts[1])
        except ValueError:
            err("Chat ID must be a number. Send a message to your bot, then check getUpdates.")
            return True
        config["telegram_token"] = token
        config["telegram_chat_id"] = chat_id
        save_config(config)
        ok("Telegram config saved.")
    else:
        # Try to use saved config
        token = config.get("telegram_token", "")
        chat_id = config.get("telegram_chat_id", 0)

    if not token or not chat_id:
        err("No config found. Usage: /telegram <bot_token> <chat_id>")
        return True

    # Already running?
    if _telegram_thread and _telegram_thread.is_alive():
        warn("Telegram bridge is already running. Use /telegram stop first.")
        return True

    # Verify token
    me = _tg_api(token, "getMe")
    if not me or not me.get("ok"):
        err("Invalid bot token. Check your token from @BotFather.")
        return True

    bot_name = me["result"].get("username", "unknown")
    ok(f"Connected to @{bot_name}. Starting bridge...")

    # Store state reference so the poll loop can read responses
    config["_state"] = _state

    _telegram_stop = threading.Event()
    _telegram_thread = threading.Thread(
        target=_tg_poll_loop, args=(token, chat_id, config), daemon=True
    )
    _telegram_thread.start()
    ok(f"Telegram bridge active. Chat ID: {chat_id}")
    info("Send messages to your bot — they'll be processed here.")
    info("Stop with /telegram stop or send /stop in Telegram.")
    return True


# ── Voice command ──────────────────────────────────────────────────────────

# Per-session voice language setting (BCP-47 code or "auto")
_voice_language: str = "auto"


def cmd_proactive(args: str, state, config) -> bool:
    """Manage proactive background polling.

    /proactive            — show current status
    /proactive 5m         — enable, trigger after 5 min of inactivity
    /proactive 30s / 1h   — enable with custom interval
    /proactive off        — disable
    """
    args = args.strip().lower()

    # Status query: no args → just print current state
    if not args:
        if config.get("_proactive_enabled"):
            interval = config.get("_proactive_interval", 300)
            info(f"Proactive background polling: ON  (triggering every {interval}s of inactivity)")
        else:
            info("Proactive background polling: OFF  (use /proactive 5m to enable)")
        return True

    # Explicit disable
    if args == "off":
        config["_proactive_enabled"] = False
        info("Proactive background polling: OFF")
        return True

    # Parse duration (e.g. "5m", "30s", "1h", or plain integer seconds)
    multiplier = 1
    val_str = args
    if args.endswith("m"):
        multiplier = 60
        val_str = args[:-1]
    elif args.endswith("h"):
        multiplier = 3600
        val_str = args[:-1]
    elif args.endswith("s"):
        val_str = args[:-1]

    try:
        val = int(val_str)
        config["_proactive_interval"] = val * multiplier
    except ValueError:
        err(f"Invalid duration: '{args}'. Use '5m', '30s', '1h', or 'off'.")
        return True

    config["_proactive_enabled"] = True
    config["_last_interaction_time"] = time.time()
    info(f"Proactive background polling: ON  (triggering every {config['_proactive_interval']}s of inactivity)")
    return True

def cmd_voice(args: str, state, config) -> bool:
    """Voice input: record → STT → auto-submit as user message.

    /voice            — record once, transcribe, submit
    /voice status     — show backend availability
    /voice lang <code> — set STT language (e.g. zh, en, ja; 'auto' to reset)
    """
    global _voice_language

    subcmd = args.strip().lower().split()[0] if args.strip() else ""
    rest = args.strip()[len(subcmd):].strip()

    # ── /voice lang <code> ──
    if subcmd == "lang":
        if not rest:
            info(f"Current STT language: {_voice_language}  (use '/voice lang auto' to reset)")
            return True
        _voice_language = rest.lower()
        ok(f"STT language set to '{_voice_language}'")
        return True

    # ── /voice status ──
    if subcmd == "status":
        try:
            from voice import check_voice_deps, check_recording_availability, check_stt_availability
            from voice.stt import get_stt_backend_name
        except ImportError as e:
            err(f"voice package not available: {e}")
            return True

        rec_ok, rec_reason = check_recording_availability()
        stt_ok, stt_reason = check_stt_availability()

        print(clr("  Voice status:", "cyan", "bold"))
        if rec_ok:
            ok("  Recording backend: available")
        else:
            err(f"  Recording: {rec_reason}")
        if stt_ok:
            ok(f"  STT backend:       {get_stt_backend_name()}")
        else:
            err(f"  STT: {stt_reason}")
        info(f"  Language: {_voice_language}")
        info("  Env override: NANO_CLAUDE_WHISPER_MODEL (default: base)")
        return True

    # ── /voice [start] — record once and submit ──
    try:
        from voice import check_voice_deps, voice_input as _voice_input
    except ImportError:
        err("voice/ package not found — this should not happen")
        return True

    available, reason = check_voice_deps()
    if not available:
        err(f"Voice input not available:\n{reason}")
        return True

    # Live energy bar (blocks are ▁▂▃▄▅▆▇█)
    _BARS = " ▁▂▃▄▅▆▇█"
    _last_bar: list[str] = [""]

    def on_energy(rms: float) -> None:
        level = min(int(rms * 8 / 0.08), 8)  # normalise ~0–0.08 to 0–8
        bar = _BARS[level]
        if bar != _last_bar[0]:
            _last_bar[0] = bar
            print(f"\r\033[K  🎙  {bar}  ", end="", flush=True)

    print(clr("  🎙  Listening… (speak now, auto-stops on silence, Ctrl+C to cancel)", "cyan"))

    try:
        text = _voice_input(language=_voice_language, on_energy=on_energy)
    except KeyboardInterrupt:
        print()
        info("  Voice input cancelled.")
        return True
    except Exception as e:
        print()
        err(f"Voice input error: {e}")
        return True

    print()  # newline after energy bar

    if not text:
        info("  (nothing transcribed — no speech detected)")
        return True

    ok(f'  Transcribed: \u201c{text}\u201d')
    print()

    # Submit the transcribed text as a user message (same path as typed input)
    # We call run_query via the closure captured in repl().
    # Since cmd_voice is called from handle_slash which is inside repl(),
    # we pass the text back via a sentinel return value that repl() recognises.
    return ("__voice__", text)


def cmd_image(args: str, state, config) -> Union[bool, tuple]:
    """Grab image from clipboard and send to vision model with optional prompt."""
    import sys as _sys
    try:
        from PIL import ImageGrab
        import io, base64
    except ImportError:
        err("Pillow is required for /image. Install with: pip install nano-claude-code[vision]")
        if _sys.platform == "linux":
            err("On Linux, clipboard support also requires xclip: sudo apt install xclip")
        return True

    img = ImageGrab.grabclipboard()
    if img is None:
        if _sys.platform == "linux":
            err("No image found in clipboard. On Linux, xclip is required (sudo apt install xclip). "
                "Copy an image with Flameshot, GNOME Screenshot, or: xclip -selection clipboard -t image/png -i file.png")
        elif _sys.platform == "darwin":
            err("No image found in clipboard. Copy an image first "
                "(Cmd+Ctrl+Shift+4 captures a screenshot region to clipboard).")
        else:
            err("No image found in clipboard. Copy an image first "
                "(Win+Shift+S captures a screenshot region to clipboard).")
        return True

    # Convert to base64 PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    size_kb = len(buf.getvalue()) / 1024

    info(f"📷 Clipboard image captured ({size_kb:.0f} KB, {img.size[0]}x{img.size[1]})")

    # Store in config for agent.py to pick up
    config["_pending_image"] = b64

    prompt = args.strip() if args.strip() else "What do you see in this image? Describe it in detail."
    return ("__image__", prompt)


COMMANDS = {
    "help":        cmd_help,
    "clear":       cmd_clear,
    "model":       cmd_model,
    "config":      cmd_config,
    "save":        cmd_save,
    "load":        cmd_load,
    "history":     cmd_history,
    "context":     cmd_context,
    "cost":        cmd_cost,
    "verbose":     cmd_verbose,
    "thinking":    cmd_thinking,
    "permissions": cmd_permissions,
    "cwd":         cmd_cwd,
    "skills":      cmd_skills,
    "memory":      cmd_memory,
    "agents":      cmd_agents,
    "mcp":         cmd_mcp,
    "plugin":      cmd_plugin,
    "tasks":       cmd_tasks,
    "task":        cmd_tasks,
    "proactive":   cmd_proactive,
    "cloudsave":   cmd_cloudsave,
    "voice":       cmd_voice,
    "image":       cmd_image,
    "brainstorm":  cmd_brainstorm,
    "worker":      cmd_worker,
    "ssj":         cmd_ssj,
    "telegram":    cmd_telegram,
    "exit":        cmd_exit,
    "quit":        cmd_exit,
    "resume":      cmd_resume
}


def handle_slash(line: str, state, config) -> Union[bool, tuple]:
    """Handle /command [args]. Returns True if handled, tuple (skill, args) for skill match."""
    if not line.startswith("/"):
        return False
    parts = line[1:].split(None, 1)
    if not parts:
        return False
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    handler = COMMANDS.get(cmd)
    if handler:
        result = handler(args, state, config)
        # cmd_voice/cmd_image/cmd_brainstorm return sentinels to ask the REPL to run_query
        if isinstance(result, tuple) and result[0] in ("__voice__", "__image__", "__brainstorm__", "__worker__", "__ssj_cmd__", "__ssj_query__", "__ssj_passthrough__"):
            return result
        return True

    # Fall through to skill lookup
    from skill import find_skill
    skill = find_skill(line)
    if skill:
        cmd_parts = line.strip().split(maxsplit=1)
        skill_args = cmd_parts[1] if len(cmd_parts) > 1 else ""
        return (skill, skill_args)

    err(f"Unknown command: /{cmd}  (type /help for commands)")
    return True


# ── Input history setup ────────────────────────────────────────────────────

# Descriptions and subcommands for each slash command (used by Tab completion)
_CMD_META: dict[str, tuple[str, list[str]]] = {
    "help":        ("Show help",                          []),
    "clear":       ("Clear conversation history",         []),
    "model":       ("Show / set model",                   []),
    "config":      ("Show / set config key=value",        []),
    "save":        ("Save session to file",               []),
    "load":        ("Load a saved session",               []),
    "history":     ("Show conversation history",          []),
    "context":     ("Show token-context usage",           []),
    "cost":        ("Show cost estimate",                 []),
    "verbose":     ("Toggle verbose output",              []),
    "thinking":    ("Toggle extended thinking",           []),
    "permissions": ("Set permission mode",                ["auto", "accept-all", "manual"]),
    "cwd":         ("Show / change working directory",    []),
    "skills":      ("List available skills",              []),
    "memory":      ("Search / list / consolidate memories", ["consolidate"]),
    "agents":      ("Show background agents",             []),
    "mcp":         ("Manage MCP servers",                 ["reload", "add", "remove"]),
    "plugin":      ("Manage plugins",                     ["install", "uninstall", "enable",
                                                           "disable", "disable-all", "update",
                                                           "recommend", "info"]),
    "tasks":       ("Manage tasks",                       ["create", "delete", "get", "clear",
                                                           "todo", "in-progress", "done", "blocked"]),
    "task":        ("Manage tasks (alias)",               ["create", "delete", "get", "clear",
                                                           "todo", "in-progress", "done", "blocked"]),
    "proactive":   ("Manage proactive background watcher", ["off"]),
    "cloudsave":   ("Cloud-sync sessions to GitHub Gist", ["setup", "auto", "list", "load", "push"]),
    "voice":       ("Voice input (record → STT)",         ["lang", "status"]),
    "image":       ("Send clipboard image to model",      []),
    "brainstorm":  ("Multi-persona AI debate + auto tasks", []),
    "worker":      ("Auto-implement pending tasks",       []),
    "ssj":         ("SSJ Developer Mode — power menu",    []),
    "telegram":    ("Telegram bot bridge",                ["stop", "status"]),
    "exit":        ("Exit nano-claude-code",              []),
    "quit":        ("Exit (alias for /exit)",             []),
    "resume":      ("Resume last session",                []),
}


def setup_readline(history_file: Path):
    if readline is None:
        return
    try:
        readline.read_history_file(str(history_file))
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, str(history_file))

    # Allow "/" to be part of a completion token so "/model" is one word
    delims = readline.get_completer_delims().replace("/", "")
    readline.set_completer_delims(delims)

    def completer(text: str, state: int):
        line = readline.get_line_buffer()

        # ── Completing a command name: line has "/" but no space yet ──────────
        if "/" in line and " " not in line:
            matches = sorted(f"/{c}" for c in _CMD_META if f"/{c}".startswith(text))
            return matches[state] if state < len(matches) else None

        # ── Completing a subcommand: "/cmd <partial>" ─────────────────────────
        if line.startswith("/") and " " in line:
            cmd = line.split()[0][1:]          # e.g. "mcp"
            if cmd in _CMD_META:
                subs = _CMD_META[cmd][1]
                matches = sorted(s for s in subs if s.startswith(text))
                return matches[state] if state < len(matches) else None

        return None

    def display_matches(substitution: str, matches: list, longest: int):
        """Custom display: show command descriptions alongside each match."""
        sys.stdout.write("\n")
        line = readline.get_line_buffer()
        is_cmd = "/" in line and " " not in line

        if is_cmd:
            col_w = max(len(m) for m in matches) + 2
            for m in sorted(matches):
                cmd = m[1:]
                desc = _CMD_META.get(cmd, ("", []))[0]
                subs = _CMD_META.get(cmd, ("", []))[1]
                sub_hint = ("  [" + ", ".join(subs[:4])
                            + ("…" if len(subs) > 4 else "") + "]") if subs else ""
                sys.stdout.write(f"  \033[36m{m:<{col_w}}\033[0m  {desc}{sub_hint}\n")
        else:
            for m in sorted(matches):
                sys.stdout.write(f"  {m}\n")
        sys.stdout.flush()

    readline.set_completion_display_matches_hook(display_matches)
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


# ── Main REPL ──────────────────────────────────────────────────────────────

def repl(config: dict, initial_prompt: str = None):
    from config import HISTORY_FILE
    from context import build_system_prompt
    from agent import AgentState, run, TextChunk, ThinkingChunk, ToolStart, ToolEnd, TurnDone, PermissionRequest

    setup_readline(HISTORY_FILE)
    state = AgentState()
    verbose = config.get("verbose", False)

    # Banner
    if not initial_prompt:
        from providers import detect_provider
        model    = config["model"]
        pname    = detect_provider(model)
        model_clr = clr(model, "cyan", "bold")
        prov_clr  = clr(f"({pname})", "dim")
        pmode     = clr(config.get("permission_mode", "auto"), "yellow")
        ver_clr   = clr(f"v{VERSION}", "green")
        _top_left  = "╭─ Nano Claude Code "
        _top_right = " ─────────────────────────╮"
        _box_w     = len(_top_left) + len(f"v{VERSION}") + len(_top_right)

        def _box_row(content: str) -> str:
            vis_len = len(re.sub(r'\x1b\[[0-9;]*m', '', content))
            pad     = _box_w - vis_len - 1
            return content + " " * max(0, pad) + clr("│", "dim")

        print(clr(_top_left, "dim") + ver_clr + clr(_top_right, "dim"))
        print(_box_row(clr("│  Model: ", "dim") + model_clr + " " + prov_clr))
        print(_box_row(clr("│  Permissions: ", "dim") + pmode))
        print(_box_row(clr("│  /model to switch provider · /help for commands", "dim")))
        print(clr("╰" + "─" * (_box_w - 2) + "╯", "dim"))

        # Show active non-default settings
        active_flags = []
        if config.get("verbose"):
            active_flags.append("verbose")
        if config.get("thinking"):
            active_flags.append("thinking")
        if config.get("_proactive_enabled"):
            active_flags.append("proactive")
        if config.get("telegram_token") and config.get("telegram_chat_id"):
            active_flags.append("telegram")
        if active_flags:
            flags_str = " · ".join(clr(f, "green") for f in active_flags)
            info(f"Active: {flags_str}")
        print()

    query_lock = threading.RLock()

    # Apply rich_live config: disable in-place Live streaming if terminal has issues.
    # Auto-detect SSH sessions and dumb terminals where ANSI cursor-up doesn't work.
    import os as _os
    _in_ssh = bool(_os.environ.get("SSH_CLIENT") or _os.environ.get("SSH_TTY"))
    _is_dumb = (console is not None and getattr(console, "is_dumb_terminal", False))
    _rich_live_default = not _in_ssh and not _is_dumb
    global _RICH_LIVE
    _RICH_LIVE = _RICH and config.get("rich_live", _rich_live_default)

    # Initialize proactive polling state in config (avoids module-level globals)
    config.setdefault("_proactive_enabled", False)
    config.setdefault("_proactive_interval", 300)
    config.setdefault("_last_interaction_time", time.time())
    if config.get("_proactive_thread") is None:
        t = threading.Thread(target=_proactive_watcher_loop, args=(config,), daemon=True)
        config["_proactive_thread"] = t
        t.start()
    
    def run_query(user_input: str, is_background: bool = False):
        nonlocal verbose
        
        with query_lock:
            verbose = config.get("verbose", False)
    
            # Rebuild system prompt each turn (picks up cwd changes, etc.)
            system_prompt = build_system_prompt()
            
            if is_background and not config.get("_telegram_incoming"):
                print(clr("\n\n[Background Event Triggered]", "yellow"))
            config.pop("_telegram_incoming", None)

            print(clr("\n╭─ Claude ", "dim") + clr("●", "green") + clr(" ─────────────────────────", "dim"))

            thinking_started = False
            spinner_shown = True
            _start_tool_spinner()
            _pre_tool_text = []   # text chunks before a tool call
            _post_tool = False    # true after a tool has executed
            _post_tool_buf = []   # text chunks after tool (to check for duplicates)
            _duplicate_suppressed = False

            try:
                for event in run(user_input, state, config, system_prompt):
                    # Stop spinner only when visible output arrives
                    if spinner_shown:
                        show_thinking = isinstance(event, ThinkingChunk) and verbose
                        if isinstance(event, TextChunk) or show_thinking or isinstance(event, ToolStart):
                            _stop_tool_spinner()
                            spinner_shown = False
                            # Restore │ prefix for first text chunk in plain-text (non-Rich) mode
                            if isinstance(event, TextChunk) and not _RICH and not _post_tool:
                                print(clr("│ ", "dim"), end="", flush=True)

                    if isinstance(event, TextChunk):
                        if thinking_started:
                            print("\033[0m\n")  # Reset dim ANSI + break line after thinking block
                            thinking_started = False

                        if _post_tool and not _duplicate_suppressed:
                            # Buffer post-tool text to check for duplicates
                            _post_tool_buf.append(event.text)
                            post_so_far = "".join(_post_tool_buf).strip()
                            pre_text = "".join(_pre_tool_text).strip()
                            # If post-tool text matches start of pre-tool text, suppress
                            if pre_text and pre_text.startswith(post_so_far):
                                if len(post_so_far) >= len(pre_text):
                                    # Full duplicate confirmed — suppress entirely
                                    _duplicate_suppressed = True
                                    _post_tool_buf.clear()
                                continue
                            elif post_so_far and not pre_text.startswith(post_so_far):
                                # Not a duplicate — flush buffered text
                                for chunk in _post_tool_buf:
                                    stream_text(chunk)
                                _post_tool_buf.clear()
                                _duplicate_suppressed = True  # stop checking
                                continue

                        # stream_text auto-starts Live on first chunk when Rich available
                        if not _post_tool:
                            _pre_tool_text.append(event.text)
                        stream_text(event.text)

                    elif isinstance(event, ThinkingChunk):
                        if verbose:
                            if not thinking_started:
                                flush_response()  # stop Live before printing static thinking
                                print(clr("  [thinking]", "dim"))
                                thinking_started = True
                            stream_thinking(event.text, verbose)

                    elif isinstance(event, ToolStart):
                        flush_response()
                        print_tool_start(event.name, event.inputs, verbose)

                    elif isinstance(event, PermissionRequest):
                        _stop_tool_spinner()
                        flush_response()
                        event.granted = ask_permission_interactive(event.description, config)
                        # Live will restart automatically on next TextChunk

                    elif isinstance(event, ToolEnd):
                        print_tool_end(event.name, event.result, verbose)
                        _post_tool = True
                        _post_tool_buf.clear()
                        _duplicate_suppressed = False
                        if not _RICH:
                            print(clr("│ ", "dim"), end="", flush=True)
                        # Restart spinner while waiting for model's next action
                        _change_spinner_phrase()
                        _start_tool_spinner()
                        spinner_shown = True

                    elif isinstance(event, TurnDone):
                        _stop_tool_spinner()
                        spinner_shown = False
                        if verbose:
                            flush_response()  # stop Live before printing token info
                            print(clr(
                                f"\n  [tokens: +{event.input_tokens} in / "
                                f"+{event.output_tokens} out]", "dim"
                            ))
            except KeyboardInterrupt:
                _stop_tool_spinner()
                flush_response()
                raise  # propagate to REPL handler which calls _track_ctrl_c
            except Exception as e:
                _stop_tool_spinner()
                import urllib.error
                # Catch 404 Not Found (Ollama model missing)
                if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                    from providers import detect_provider
                    if detect_provider(config["model"]) == "ollama":
                        flush_response()
                        err(f"Ollama model '{config['model']}' not found.")
                        if _interactive_ollama_picker(config):
                            # Remove the user message added by run() before retrying
                            if state.messages and state.messages[-1]["role"] == "user":
                                state.messages.pop()
                            return run_query(user_input, is_background)
                        # User cancelled picker — abort gracefully without crashing
                        return
                raise e

            _stop_tool_spinner()
            flush_response()  # stop Live, commit any remaining text
            print(clr("╰──────────────────────────────────────────────", "dim"))
            print()
            
            # If this was a background task, we redraw the prompt for the user
            if is_background:
                print(clr("\n[claude-code-local] » ", "yellow"), end="", flush=True)

        # Drain any AskUserQuestion prompts raised during this turn
        from tools import drain_pending_questions
        drain_pending_questions()
        
        config["_last_interaction_time"] = time.time()

    config["_run_query_callback"] = lambda msg: run_query(msg, is_background=True)

    def _handle_slash_from_telegram(line: str):
        """Process a /command from Telegram, handling sentinels inline.
        Returns 'simple' for toggle commands, 'query' if run_query was called."""
        result = handle_slash(line, state, config)
        if not isinstance(result, tuple):
            return "simple"
        # Process sentinels the same way the REPL does
        if result[0] == "__brainstorm__":
            _, brain_prompt, brain_out_file = result
            run_query(brain_prompt)
            _save_synthesis(state, brain_out_file)
            run_query(
                "Based on the Master Plan you just synthesized, generate a todo_list.txt file in the current directory. "
                "Format: one task per line, each starting with '- [ ] '. "
                "Order by priority. Include ALL actionable items from the plan. "
                "Use the Write tool to create the file. Do NOT explain, just write the file now."
            )
        elif result[0] == "__worker__":
            _, worker_tasks = result
            for i, (line_idx, task_text, prompt) in enumerate(worker_tasks):
                print(clr(f"\n  ── Worker ({i+1}/{len(worker_tasks)}): {task_text} ──", "yellow"))
                run_query(prompt)
        return "query"

    config["_handle_slash_callback"] = _handle_slash_from_telegram

    # ── Auto-start Telegram bridge if configured ──────────────────────
    if config.get("telegram_token") and config.get("telegram_chat_id"):
        global _telegram_thread, _telegram_stop
        if not (_telegram_thread and _telegram_thread.is_alive()):
            config["_state"] = state
            _telegram_stop = threading.Event()
            _telegram_thread = threading.Thread(
                target=_tg_poll_loop,
                args=(config["telegram_token"], config["telegram_chat_id"], config),
                daemon=True
            )
            _telegram_thread.start()

    # ── Rapid Ctrl+C force-quit ─────────────────────────────────────────
    # 3 Ctrl+C presses within 2 seconds → immediate hard exit
    # Uses the default SIGINT (raises KeyboardInterrupt) but wraps the
    # main loop to track timing of consecutive interrupts.
    _ctrl_c_times = []

    def _track_ctrl_c():
        """Call this on every KeyboardInterrupt. Returns True if force-quit triggered."""
        now = time.time()
        _ctrl_c_times.append(now)
        # Keep only presses within the last 2 seconds
        _ctrl_c_times[:] = [t for t in _ctrl_c_times if now - t <= 2.0]
        if len(_ctrl_c_times) >= 3:
            _stop_tool_spinner()
            print(clr("\n\n  Force quit (3x Ctrl+C).", "red", "bold"))
            os._exit(1)
        return False

    # ── Main loop ──
    if initial_prompt:
        try:
            run_query(initial_prompt)
        except KeyboardInterrupt:
            _track_ctrl_c()
            print()
        return

    # ── Bracketed paste mode ──────────────────────────────────────────────
    # Terminals that support bracketed paste wrap pasted content with
    #   ESC[200~  (start)  …content…  ESC[201~  (end)
    # This lets us collect the entire paste as one unit regardless of
    # how many newlines it contains, without any fragile timing tricks.
    _PASTE_START = "\x1b[200~"
    _PASTE_END   = "\x1b[201~"
    _bpm_active  = sys.stdin.isatty() and sys.platform != "win32"

    if _bpm_active:
        sys.stdout.write("\x1b[?2004h")   # enable bracketed paste mode
        sys.stdout.flush()

    def _read_input(prompt: str) -> str:
        """Read one user turn, collecting multi-line pastes as a single string.

        Strategy (in priority order):
        1. Bracketed paste mode (ESC[200~ … ESC[201~): reliable, zero latency,
           supported by virtually all modern terminal emulators on Linux/macOS.
        2. Timing fallback: for terminals without bracketed paste support, read
           any data buffered in stdin within a short window after the first line.
        3. Plain input(): for pipes / non-interactive use / Windows.
        """
        import select as _sel

        # ── Phase 1: get first line via readline (history, line-edit intact) ──
        first = input(prompt)

        # ── Phase 2: bracketed paste? ─────────────────────────────────────────
        if _PASTE_START in first:
            # Strip leading marker; first line may already contain paste end too
            body = first.replace(_PASTE_START, "")
            if _PASTE_END in body:
                # Single-line paste (no embedded newlines)
                return body.replace(_PASTE_END, "").strip()

            # Multi-line paste: keep reading until end marker arrives
            lines = [body]
            while True:
                ready = _sel.select([sys.stdin], [], [], 2.0)[0]
                if not ready:
                    break  # safety timeout — paste stalled
                raw = sys.stdin.readline()
                if not raw:
                    break
                raw = raw.rstrip("\n")
                if _PASTE_END in raw:
                    tail = raw.replace(_PASTE_END, "")
                    if tail:
                        lines.append(tail)
                    break
                lines.append(raw)

            result = "\n".join(lines).strip()
            n = result.count("\n") + 1
            info(f"  (pasted {n} line{'s' if n > 1 else ''})")
            return result

        # ── Phase 3: timing fallback ─────────────────────────────────────────
        if sys.stdin.isatty():
            lines = [first]
            import time as _time

            if sys.platform == "win32":
                # Windows: use msvcrt.kbhit() to detect buffered paste data
                import msvcrt
                deadline = 0.12   # wider window for Windows paste latency
                chunk_to = 0.03
                t0 = _time.monotonic()
                while (_time.monotonic() - t0) < deadline:
                    _time.sleep(chunk_to)
                    if not msvcrt.kbhit():
                        break
                    raw = sys.stdin.readline()
                    if not raw:
                        break
                    stripped = raw.rstrip("\n").rstrip("\r")
                    lines.append(stripped)
                    t0 = _time.monotonic()  # extend while data keeps coming
            else:
                # Unix: use select() for precise timing
                deadline = 0.06
                chunk_to = 0.025
                t0 = _time.monotonic()
                while (_time.monotonic() - t0) < deadline:
                    ready = _sel.select([sys.stdin], [], [], chunk_to)[0]
                    if not ready:
                        break
                    raw = sys.stdin.readline()
                    if not raw:
                        break
                    stripped = raw.rstrip("\n")
                    if _PASTE_END in stripped:
                        break
                    lines.append(stripped)
                    t0 = _time.monotonic()

            if len(lines) > 1:
                result = "\n".join(lines).strip()
                info(f"  (pasted {len(lines)} lines)")
                return result

        return first

    while True:
        # Show notifications for background agents that finished
        _print_background_notifications()
        try:
            cwd_short = Path.cwd().name
            prompt = clr(f"\n[{cwd_short}] ", "dim") + clr("» ", "cyan", "bold")
            user_input = _read_input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            try:
                save_latest("", state, config)
            except Exception as e:
                warn(f"Auto-save failed on exit: {e}")
            if _bpm_active:
                sys.stdout.write("\x1b[?2004l")  # disable bracketed paste mode
                sys.stdout.flush()
            ok("Goodbye!")
            sys.exit(0)

        if not user_input:
            continue

        result = handle_slash(user_input, state, config)
        # ── Sentinel processing loop ──
        # Processes sentinel tuples returned by commands. SSJ-originated
        # sentinels loop back to the SSJ menu after completion.
        while isinstance(result, tuple):
            # Voice sentinel: ("__voice__", transcribed_text)
            if result[0] == "__voice__":
                _, voice_text = result
                try:
                    run_query(voice_text)
                except KeyboardInterrupt:
                    _track_ctrl_c()
                    print(clr("\n  (interrupted)", "yellow"))
                break
            # Image sentinel: ("__image__", prompt_text)
            if result[0] == "__image__":
                _, image_prompt = result
                try:
                    run_query(image_prompt)
                except KeyboardInterrupt:
                    _track_ctrl_c()
                    print(clr("\n  (interrupted)", "yellow"))
                break

            # SSJ passthrough: user typed a /command inside SSJ menu
            if result[0] == "__ssj_passthrough__":
                _, slash_line = result
                inner = handle_slash(slash_line, state, config)
                if isinstance(inner, tuple):
                    result = inner
                    continue
                break

            # SSJ command sentinel: ("__ssj_cmd__", cmd_name, args)
            # Delegate to the real command and re-process its returned sentinel
            if result[0] == "__ssj_cmd__":
                _, cmd_name, cmd_args = result
                inner = handle_slash(f"/{cmd_name} {cmd_args}".strip(), state, config)
                if isinstance(inner, tuple):
                    # Tag so we know to loop back to SSJ after processing
                    result = ("__ssj_wrap__", inner)
                    continue
                # Command handled directly, loop back to SSJ
                result = handle_slash("/ssj", state, config)
                continue

            # Unwrap SSJ-wrapped sentinel and process the inner sentinel
            if result[0] == "__ssj_wrap__":
                result = result[1]
                _from_ssj_flag = True
            else:
                _from_ssj_flag = result[0] == "__ssj_query__"

            # Brainstorm sentinel: ("__brainstorm__", synthesis_prompt, out_file)
            if result[0] == "__brainstorm__":
                _, brain_prompt, brain_out_file = result
                print(clr("\n  ── Analysis from Main Agent ──", "dim"))
                try:
                    run_query(brain_prompt)
                    _save_synthesis(state, brain_out_file)
                    print(clr("\n  ── Generating TODO List from Master Plan ──", "dim"))
                    run_query(
                        "Based on the Master Plan you just synthesized, generate a todo_list.txt file in the current directory. "
                        "Format: one task per line, each starting with '- [ ] '. "
                        "Order by priority. Include ALL actionable items from the plan. "
                        "Use the Write tool to create the file. Do NOT explain, just write the file now."
                    )
                    info("TODO list saved to todo_list.txt. Edit it freely, then use /worker to start implementing.")
                except KeyboardInterrupt:
                    _track_ctrl_c()
                    print(clr("\n  (interrupted)", "yellow"))
                if _from_ssj_flag:
                    result = handle_slash("/ssj", state, config)
                    continue
                break
            # Worker sentinel: ("__worker__", [(line_idx, task_text, prompt), ...])
            if result[0] == "__worker__":
                _, worker_tasks = result
                for i, (line_idx, task_text, prompt) in enumerate(worker_tasks):
                    print(clr(f"\n  ── Worker ({i+1}/{len(worker_tasks)}): {task_text} ──", "yellow"))
                    try:
                        run_query(prompt)
                    except KeyboardInterrupt:
                        _track_ctrl_c()
                        print(clr("\n  (worker interrupted — remaining tasks skipped)", "yellow"))
                        break
                ok("Worker finished. Run /worker to check remaining tasks.")
                if _from_ssj_flag:
                    result = handle_slash("/ssj", state, config)
                    continue
                break
            # SSJ query sentinel: ("__ssj_query__", prompt)
            if result[0] == "__ssj_query__":
                _, ssj_prompt = result
                try:
                    run_query(ssj_prompt)
                except KeyboardInterrupt:
                    _track_ctrl_c()
                    print(clr("\n  (interrupted)", "yellow"))
                # Loop back to SSJ menu
                result = handle_slash("/ssj", state, config)
                continue
            # Skill match (fallback): (SkillDef, args_str)
            skill, skill_args = result
            info(f"Running skill: {skill.name}" + (f" [{skill.context}]" if skill.context == "fork" else ""))
            try:
                from skill import substitute_arguments
                rendered = substitute_arguments(skill.prompt, skill_args, skill.arguments)
                run_query(f"[Skill: {skill.name}]\n\n{rendered}")
            except KeyboardInterrupt:
                _track_ctrl_c()
                print(clr("\n  (interrupted)", "yellow"))
            break
        # Sentinel or command was handled — don't fall through to run_query
        if result:
            continue

        try:
            run_query(user_input)
        except KeyboardInterrupt:
            _track_ctrl_c()
            print(clr("\n  (interrupted)", "yellow"))
            # Keep conversation history up to the interruption


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="nano_claude",
        description="Nano Claude Code — minimal Python Claude Code implementation",
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*", help="Initial prompt (non-interactive)")
    parser.add_argument("-p", "--print", "--print-output",
                        dest="print_mode", action="store_true",
                        help="Non-interactive mode: run prompt and exit")
    parser.add_argument("-m", "--model", help="Override model")
    parser.add_argument("--accept-all", action="store_true",
                        help="Never ask permission (accept all operations)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show thinking + token counts")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable extended thinking")
    parser.add_argument("--version", action="store_true", help="Print version")
    parser.add_argument("-h", "--help", action="store_true", help="Show help")

    args = parser.parse_args()

    if args.version:
        print(f"nano claude code v{VERSION}")
        sys.exit(0)

    if args.help:
        print(__doc__)
        sys.exit(0)

    from config import load_config, save_config, has_api_key
    from providers import detect_provider, PROVIDERS

    config = load_config()

    # Apply CLI overrides first (so key check uses the right provider)
    if args.model:
        m = args.model
        # Convert "provider:model" → "provider/model" only when left side is a known provider
        # (e.g. "ollama:llama3.3" → "ollama/llama3.3"), but leave version tags intact
        # (e.g. "ollama/qwen3.5:35b" must NOT become "ollama/qwen3.5/35b")
        if "/" not in m and ":" in m:
            from providers import PROVIDERS
            left, _ = m.split(":", 1)
            if left in PROVIDERS:
                m = m.replace(":", "/", 1)
        config["model"] = m
    if args.accept_all:
        config["permission_mode"] = "accept-all"
    if args.verbose:
        config["verbose"] = True
    if args.thinking:
        config["thinking"] = True

    # Check API key for active provider (warn only, don't block local providers)
    if not has_api_key(config):
        pname = detect_provider(config["model"])
        prov  = PROVIDERS.get(pname, {})
        env   = prov.get("api_key_env", "")
        if env:   # local providers like ollama have no env key requirement
            warn(f"No API key found for provider '{pname}'. "
                 f"Set {env} or run: /config {pname}_api_key=YOUR_KEY")

    initial = " ".join(args.prompt) if args.prompt else None
    if args.print_mode and not initial:
        err("--print requires a prompt argument")
        sys.exit(1)

    repl(config, initial_prompt=initial)


if __name__ == "__main__":
    main()
