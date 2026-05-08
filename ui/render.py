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

def _rgb(hex_str: str) -> str:
    """Convert '#rrggbb' -> ANSI 24-bit foreground escape."""
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"


# Curated palettes — each theme defines four semantic roles:
#   accent : info / primary chrome (cyan, blue)
#   ok     : success / diff additions (green) — kept distinct from accent so
#            info() and ok() are visually distinguishable
#   warn   : warnings (yellow, magenta)
#   err    : errors / diff removals (red)
#   code   : Rich Markdown code-block style
# Add new entries here and they show up in `/theme` automatically.
THEMES: dict = {
    "default":     {"accent": "#00D7FF", "ok": "#00FF87", "warn": "#FFAF00", "err": "#FF5F5F", "code": "monokai"},
    "dracula":     {"accent": "#BD93F9", "ok": "#50FA7B", "warn": "#FFB86C", "err": "#FF5555", "code": "dracula"},
    "nord":        {"accent": "#88C0D0", "ok": "#A3BE8C", "warn": "#EBCB8B", "err": "#BF616A", "code": "nord"},
    "gruvbox":     {"accent": "#FABD2F", "ok": "#B8BB26", "warn": "#FE8019", "err": "#FB4934", "code": "gruvbox-dark"},
    "solarized":   {"accent": "#268BD2", "ok": "#859900", "warn": "#B58900", "err": "#DC322F", "code": "solarized-dark"},
    "tokyo-night": {"accent": "#7AA2F7", "ok": "#9ECE6A", "warn": "#E0AF68", "err": "#F7768E", "code": "one-dark"},
    "catppuccin":  {"accent": "#F5C2E7", "ok": "#A6E3A1", "warn": "#FAB387", "err": "#F38BA8", "code": "one-dark"},
    "matrix":      {"accent": "#00FF41", "ok": "#7FFF00", "warn": "#CCFF00", "err": "#FF0000", "code": "monokai"},
    "synthwave":   {"accent": "#FF00FF", "ok": "#39FF14", "warn": "#FFCC00", "err": "#FF3864", "code": "fruity"},
    "midnight":    {"accent": "#00BCD4", "ok": "#76FF03", "warn": "#FFC107", "err": "#FF1744", "code": "dracula"},
    "ocean":       {"accent": "#38BDF8", "ok": "#34D399", "warn": "#FBBF24", "err": "#F87171", "code": "nord"},
    "monokai":     {"accent": "#66D9EF", "ok": "#A6E22E", "warn": "#E6DB74", "err": "#F92672", "code": "monokai"},
    "cheetah":     {"accent": "#FFB000", "ok": "#76FF03", "warn": "#FF6F00", "err": "#D50000", "code": "monokai"},
    "mono":        {"accent": "#E0E0E0", "ok": "#C0C0C0", "warn": "#A0A0A0", "err": "#FFFFFF", "code": "bw"},
    "none":        {"disable_color": True, "code": "default"},
}

# Active code-block style for Rich Markdown rendering. Read by _make_renderable.
CODE_THEME: str = "monokai"

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


def apply_theme(name: str) -> bool:
    """Mutate the global ANSI color map in-place to a named theme."""
    global CODE_THEME
    p = THEMES.get(name)
    if not p:
        return False

    # The "none" theme: strip every escape so output is plain text.
    if p.get("disable_color"):
        for k in list(C.keys()):
            C[k] = ""
        CODE_THEME = p.get("code", "default")
        return True

    accent = _rgb(p["accent"])
    ok_col = _rgb(p.get("ok", p["accent"]))
    warn_c = _rgb(p["warn"])
    err_c  = _rgb(p.get("err", "#FF5555"))

    C["cyan"]    = accent
    C["blue"]    = accent
    C["green"]   = ok_col
    C["yellow"]  = warn_c
    C["magenta"] = warn_c
    C["red"]     = err_c
    C["white"]   = "\033[97m"
    C["bold"]    = "\033[1m"
    C["dim"]     = "\033[2m"
    C["reset"]   = "\033[0m"
    CODE_THEME   = p["code"]
    return True

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
        return Markdown(text, code_theme=CODE_THEME)
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
    """Buffer chunk; update Live in-place when Rich available, else print directly.

    Safety: if accumulated text exceeds _LIVE_LINE_LIMIT lines, auto-switch
    from Rich Live to plain streaming to prevent terminal re-render duplication
    on terminals that can't handle large Live areas (macOS Terminal, etc.).
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
            # Print the full text once (Live already displayed partial content,
            # but stopping Live clears it — so we re-print cleanly)
            console.print(_make_renderable(full))
            return

        if line_count <= _LIVE_LINE_LIMIT:
            if _current_live is None:
                _start_live()
            _current_live.update(_make_renderable(full), refresh=True)
        else:
            # Already past limit, no Live — just append new chunk
            print(chunk, end="", flush=True)
    else:
        print(chunk, end="", flush=True)

def stream_thinking(chunk: str, verbose: bool):
    if verbose:
        clean_chunk = chunk.replace("\n", " ")
        if clean_chunk:
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
        console.print(_make_renderable(full))
    else:
        print()  # ensure newline after plain-text stream


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
    if name == "AskUserQuestion":
        questions = inputs.get("questions", [])
        if questions:
            first = questions[0].get("question", "") if isinstance(questions[0], dict) else str(questions[0])
            return f"AskUserQuestion({first[:60]}{'…' if len(first) > 60 else ''})"
        return "AskUserQuestion()"
    return f"{name}({list(inputs.values())[:1]})"


def print_tool_start(name: str, inputs: dict, verbose: bool):
    """Show tool invocation."""
    if name == "AskUserQuestion":
        return
    desc = _tool_desc(name, inputs)
    print(clr(f"  ⚙  {desc}", "dim", "cyan"), flush=True)
    if verbose:
        print(clr(f"     inputs: {json.dumps(inputs, ensure_ascii=False)[:200]}", "dim"))

def print_tool_end(name: str, result: str, verbose: bool):
    if name == "AskUserQuestion":
        return
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
    else:
        print(clr(f"  ✗ {result[:120]}", "dim", "red"), flush=True)
    if verbose and not result.startswith("Denied"):
        preview = result[:500] + ("…" if len(result) > 500 else "")
        print(clr(f"     {preview.replace(chr(10), chr(10)+'     ')}", "dim"))
