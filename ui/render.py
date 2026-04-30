"""
ui/render.py — All terminal rendering for CheetahClaws.

Provides:
  - ANSI color helpers (C, clr, info, ok, warn, err)
  - Rich Markdown streaming (stream_text, flush_response)
  - Spinner management
  - Tool call display (print_tool_start, print_tool_end)
  - Diff rendering (render_diff)
"""
from __future__ import annotations

import sys
import json
import threading

# ── Optional rich for markdown rendering ──────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.live import Live
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None
    Live = None
    Markdown = None

# ── ANSI helpers ───────────────────────────────────────────────────────────
C = {
    "cyan":    "\033[36m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "red":     "\033[31m",
    "blue":    "\033[34m",
    "magenta": "\033[35m",
    "white":   "\033[37m",
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

def _truncate_err_global(s: str, max_len: int = 200) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


# ── Diff rendering ─────────────────────────────────────────────────────────

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
_current_live = None                # active Rich Live instance (one at a time)
_RICH_LIVE = True                   # set False (via config rich_live=false) to disable

def set_rich_live(enabled: bool) -> None:
    """Called from repl.py to apply the rich_live config setting."""
    global _RICH_LIVE
    _RICH_LIVE = _RICH and enabled

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

_LIVE_LINE_LIMIT = 80  # auto-switch to plain streaming beyond this many lines


def stream_text(chunk: str) -> None:
    """Buffer chunk; render formatted output at flush_response() time.

    When Rich Live is enabled: update in-place on each chunk.
    When disabled (patch_stdout active): buffer silently, render once at flush.
    """
    global _current_live
    _accumulated_text.append(chunk)

    if _RICH and _RICH_LIVE:
        full = "".join(_accumulated_text)
        line_count = full.count("\n")

        # Safety: too many lines → kill Live and fall back to plain streaming
        if _current_live is not None and line_count > _LIVE_LINE_LIMIT:
            _current_live.stop()
            _current_live = None
            console.print(_make_renderable(full))
            return

        if line_count <= _LIVE_LINE_LIMIT:
            if _current_live is None:
                _start_live()
            _current_live.update(_make_renderable(full), refresh=True)
        else:
            print(chunk, end="", flush=True)
    # Non-Live: buffer only — flush_response() renders the complete text once

def stream_thinking(chunk: str, verbose: bool):
    if verbose:
        clean_chunk = chunk.replace("\n", " ")
        if clean_chunk:
            print(f"{C['dim']}{clean_chunk}", end="", flush=True)

def flush_response() -> None:
    """Commit buffered text to screen with Rich Markdown formatting."""
    global _current_live
    full = "".join(_accumulated_text)
    _accumulated_text.clear()
    if _current_live is not None:
        _current_live.stop()
        _current_live = None
    elif _RICH and full.strip():
        console.print(_make_renderable(full))
    elif full:
        print(full, end="", flush=True)
        print()


# ── Spinner ────────────────────────────────────────────────────────────────

_TOOL_SPINNER_PHRASES = [
    "⚡ Rewriting light speed...",
    "🏁 Winning a race against light...",
    "🤔 Who is Barry Allen?...",
    "🐆 Outrunning the compiler...",
    "💨 Leaving electrons behind...",
    "🌍 Orbiting the codebase...",
    "⏱️ Breaking the sound barrier...",
    "🔥 Faster than a hot reload...",
    "🚀 Terminal velocity reached...",
    "🐾 Claw marks on the stack...",
    "🏎️ Shifting to 6th gear...",
    "⚡ Speed force activated...",
    "🌪️ Blitzing through the AST...",
    "💫 Bending spacetime...",
    "🐆 Cheetah mode engaged...",
]

_DEBATE_SPINNER_PHRASES = [
    "⚔️  Experts taking their positions...",
    "🧠  Experts formulating arguments...",
    "🗣️  Debate in progress...",
    "⚖️  Weighing the evidence...",
    "💡  Building counter-arguments...",
    "🔥  Debate heating up...",
    "📜  Drafting the consensus...",
    "🎯  Finding common ground...",
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
        return
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

def set_spinner_phrase(phrase: str) -> None:
    """Set a specific spinner phrase (used by SSJ debate mode)."""
    global _spinner_phrase
    with _spinner_lock:
        _spinner_phrase = phrase

def _stop_tool_spinner():
    global _tool_spinner_thread
    if not _tool_spinner_thread:
        return
    _tool_spinner_stop.set()
    _tool_spinner_thread.join(timeout=1)
    _tool_spinner_thread = None
    sys.stdout.write(f"\r{' ' * 50}\r")
    sys.stdout.flush()


# ── Tool call display ──────────────────────────────────────────────────────

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


def print_tool_start(name: str, inputs: dict, verbose: bool):
    if verbose:
        desc = _tool_desc(name, inputs)
        print(clr(f"  ⚙  {desc}", "dim", "cyan"), flush=True)
        print(clr(f"     inputs: {json.dumps(inputs, ensure_ascii=False)[:200]}", "dim"))
        return
    # Non-verbose: overwrite the spinner line with the tool name, no newline
    desc = _tool_desc(name, inputs)
    sys.stdout.write(f"\r  \033[2m⚙  {desc[:80]}\033[0m" + " " * 10 + "\r")
    sys.stdout.flush()

def print_tool_end(name: str, result: str, verbose: bool):
    if verbose:
        lines = result.count("\n") + 1
        size = len(result)
        summary = f"→ {lines} lines ({size} chars)"
        if not result.startswith("Error") and not result.startswith("Denied"):
            print(clr(f"  ✓ {summary}", "dim", "green"), flush=True)
            if name in ("Edit", "Write") and _has_diff(result):
                parts = result.split("\n\n", 1)
                if len(parts) == 2:
                    print(clr(f"  {parts[0]}", "dim"))
                    render_diff(parts[1])
            preview = result[:500] + ("…" if len(result) > 500 else "")
            print(clr(f"     {preview.replace(chr(10), chr(10)+'     ')}", "dim"))
        else:
            print(clr(f"  ✗ {result[:120]}", "dim", "red"), flush=True)
        return
    # Non-verbose: show diffs for Edit/Write (permanent output), errors; erase all else
    if name in ("Edit", "Write") and _has_diff(result):
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        parts = result.split("\n\n", 1)
        if len(parts) == 2:
            print(clr(f"  {parts[0]}", "dim"))
            render_diff(parts[1])
    elif result.startswith("Error") or result.startswith("Denied"):
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        print(clr(f"  ✗ {result[:120]}", "dim", "red"), flush=True)
    # Otherwise: just erase the tool-start line; spinner will overwrite next
