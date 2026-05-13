"""Tmux integration tools for CheetahClaws.

Gives the AI model direct control over tmux sessions: create panes,
send commands, read output, and manage layouts.  Auto-detected at
startup — tools are only registered when tmux is available on the host.
"""
import os
import re
import sys
import subprocess
import shutil
from tool_registry import ToolDef, register_tool

# ── Detection ────────────────────────────────────────────────────────────────

def _find_tmux() -> str | None:
    """Locate a tmux-compatible binary (tmux on Unix, psmux/tmux.exe on Windows)."""
    found = shutil.which("tmux") or shutil.which("psmux")
    if found:
        return found
    if sys.platform == "win32":
        candidates = [
            os.path.expanduser(r"~\.cargo\bin\psmux.exe"),
            os.path.expanduser(r"~\.cargo\bin\tmux.exe"),
        ]
        # Allow override via env var for custom install locations
        custom = os.environ.get("CHEETAH_PSMUX_PATH")
        if custom:
            candidates.insert(0, custom)
        # Search common install locations
        for base in [os.path.expanduser("~\\Desktop"), os.path.expanduser("~")]:
            for name in ("psmux.exe", "tmux.exe"):
                p = os.path.join(base, "psmux", "target", "release", name)
                candidates.append(p)
                p2 = os.path.join(base, "localtest", "psmux", "target", "release", name)
                candidates.append(p2)
        for c in candidates:
            if os.path.isfile(c):
                return c
    return None


_TMUX_BIN: str | None = _find_tmux()

# Sanitize pattern: only allow alphanumerics, underscores, hyphens, dots, colons
_SAFE_NAME = re.compile(r'^[a-zA-Z0-9_.:-]+$')

# Direction flag constants
_RESIZE_FLAGS = {"up": "-U", "down": "-D", "left": "-L", "right": "-R"}
_READ_ONLY_TOOLS = frozenset(("TmuxListSessions", "TmuxCapture", "TmuxListPanes", "TmuxListWindows"))


def tmux_available() -> bool:
    """Return True if a tmux-compatible binary exists on the system."""
    return _TMUX_BIN is not None


def _safe(value: str) -> str:
    """Sanitize a tmux target/session name to prevent shell injection."""
    if not value or not _SAFE_NAME.match(value):
        raise ValueError(f"Invalid tmux identifier: {value!r}")
    return value


def _t_args(params: dict, key: str = "target") -> list[str]:
    """Build the ['-t', <target>] argv pair, or [] if absent."""
    val = params.get(key, "")
    return ["-t", _safe(val)] if val else []


def _run(argv: list[str], timeout: int = 10) -> str:
    """Run a tmux command as argv (no shell). Returns combined stdout+stderr.

    The first arg may be the literal string "tmux"; it is replaced with the
    detected tmux binary path. Unsets nesting guards ($TMUX / $PSMUX_SESSION)
    so commands work from inside an existing session.
    """
    if not argv:
        return "Error: empty tmux command"
    if argv[0] == "tmux":
        argv = [_TMUX_BIN, *argv[1:]]
    try:
        env = dict(os.environ)
        env.pop("TMUX", None)
        env.pop("PSMUX_SESSION", None)
        r = subprocess.run(
            argv, shell=False, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        stdout = r.stdout.strip()
        stderr = r.stderr.strip()
        if r.returncode != 0 and stderr:
            return f"FAILED (exit {r.returncode}): {stderr}"
        out = (stdout + ("\n" + stderr if stderr else "")).strip()
        return out if out else "(ok)"
    except subprocess.TimeoutExpired:
        return "Error: tmux command timed out"
    except Exception as e:
        return f"Error: {e}"


# ── Tool implementations ────────────────────────────────────────────────────

def _tmux_list_sessions(params: dict, config: dict) -> str:
    return _run(["tmux", "list-sessions"])


def _tmux_new_session(params: dict, config: dict) -> str:
    name = _safe(params.get("session_name", "cheetah"))
    argv = ["tmux", "new-session"]
    if params.get("detached", True):
        argv.append("-d")
    argv += ["-s", name]
    cmd = params.get("command", "")
    if cmd:
        argv.append(cmd)
    return _run(argv)


def _tmux_split_window(params: dict, config: dict) -> str:
    direction = "-v" if params.get("direction", "vertical") == "vertical" else "-h"
    argv = ["tmux", "split-window", direction, *_t_args(params)]
    cmd = params.get("command", "")
    if cmd:
        argv.append(cmd)
    return _run(argv)


def _tmux_send_keys(params: dict, config: dict) -> str:
    keys = params["keys"]
    argv = ["tmux", "send-keys", *_t_args(params), keys]
    if params.get("press_enter", True):
        argv.append("Enter")
    return _run(argv)


def _tmux_capture_pane(params: dict, config: dict) -> str:
    lines = int(params.get("lines", 50))
    return _run(["tmux", "capture-pane", *_t_args(params), "-p", "-S", f"-{lines}"])


_PANE_FMT = "#{pane_index}: #{pane_current_command} [#{pane_width}x#{pane_height}] #{?pane_active,(active),}"


def _tmux_list_panes(params: dict, config: dict) -> str:
    return _run(["tmux", "list-panes", *_t_args(params), "-F", _PANE_FMT])


def _tmux_select_pane(params: dict, config: dict) -> str:
    return _run(["tmux", "select-pane", "-t", _safe(params["target"])])


def _tmux_kill_pane(params: dict, config: dict) -> str:
    return _run(["tmux", "kill-pane", *_t_args(params)])


def _tmux_new_window(params: dict, config: dict) -> str:
    argv = ["tmux", "new-window", *_t_args(params, "target_session")]
    name = params.get("window_name", "")
    if name:
        argv += ["-n", _safe(name)]
    cmd = params.get("command", "")
    if cmd:
        argv.append(cmd)
    return _run(argv)


_WINDOW_FMT = "#{window_index}: #{window_name} [#{window_width}x#{window_height}] #{?window_active,(active),}"


def _tmux_list_windows(params: dict, config: dict) -> str:
    return _run(["tmux", "list-windows", *_t_args(params, "target_session"), "-F", _WINDOW_FMT])


def _tmux_resize_pane(params: dict, config: dict) -> str:
    direction = params.get("direction", "down")
    amount = int(params.get("amount", 10))
    d_flag = _RESIZE_FLAGS.get(direction, "-D")
    return _run(["tmux", "resize-pane", *_t_args(params), d_flag, str(amount)])


# ── Schemas ──────────────────────────────────────────────────────────────────

TMUX_TOOL_SCHEMAS = [
    {
        "name": "TmuxListSessions",
        "description": "List all active tmux sessions.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "TmuxNewSession",
        "description": "Create a new tmux session. Use detached=true (default) to keep it in the background.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_name": {"type": "string", "description": "Session name (default: cheetah)"},
                "detached":     {"type": "boolean", "description": "Start detached (default: true)"},
                "command":      {"type": "string", "description": "Optional command to run in the new session"},
            },
        },
    },
    {
        "name": "TmuxSplitWindow",
        "description": "Split the current tmux pane into two. Creates a new visible terminal pane. (Hint: to run a command and keep the pane open, omit 'command' here and use TmuxSendKeys afterwards).",
        "input_schema": {
            "type": "object",
            "properties": {
                "target":    {"type": "string", "description": "Target pane (e.g. session:window.pane)"},
                "direction": {"type": "string", "enum": ["vertical", "horizontal"], "description": "Split direction (default: vertical)"},
                "command":   {"type": "string", "description": "Optional command to run in the new pane"},
            },
        },
    },
    {
        "name": "TmuxSendKeys",
        "description": "Send keystrokes/commands to a tmux pane. The command runs visibly in that pane.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keys":        {"type": "string", "description": "The text or command to send"},
                "target":      {"type": "string", "description": "Target pane (e.g. session:window.pane)"},
                "press_enter": {"type": "boolean", "description": "Press Enter after sending keys (default: true)"},
            },
            "required": ["keys"],
        },
    },
    {
        "name": "TmuxCapture",
        "description": "Capture and return the visible text content of a tmux pane. Use this to read command output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target pane (e.g. session:window.pane)"},
                "lines":  {"type": "integer", "description": "Number of history lines to capture (default: 50)"},
            },
        },
    },
    {
        "name": "TmuxListPanes",
        "description": "List all panes in the current session/window with their index, command, and size.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target session or window"},
            },
        },
    },
    {
        "name": "TmuxSelectPane",
        "description": "Switch focus to a specific tmux pane.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target pane (e.g. 0, 1, or session:window.pane)"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "TmuxKillPane",
        "description": "Close/kill a tmux pane.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target pane to kill"},
            },
        },
    },
    {
        "name": "TmuxNewWindow",
        "description": "Create a new tmux window (tab) in a session. (Hint: to run a command and keep the window open, omit 'command' here and use TmuxSendKeys afterwards).",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_session": {"type": "string", "description": "Session to add the window to"},
                "window_name":    {"type": "string", "description": "Name for the new window"},
                "command":        {"type": "string", "description": "Optional command to run"},
            },
        },
    },
    {
        "name": "TmuxListWindows",
        "description": "List all windows in a tmux session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_session": {"type": "string", "description": "Session name"},
            },
        },
    },
    {
        "name": "TmuxResizePane",
        "description": "Resize a tmux pane in a given direction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target":    {"type": "string", "description": "Target pane"},
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "Resize direction"},
                "amount":    {"type": "integer", "description": "Number of cells to resize (default: 10)"},
            },
        },
    },
]

# ── Registration ─────────────────────────────────────────────────────────────

_TOOL_FUNCS = {
    "TmuxListSessions": _tmux_list_sessions,
    "TmuxNewSession":   _tmux_new_session,
    "TmuxSplitWindow":  _tmux_split_window,
    "TmuxSendKeys":     _tmux_send_keys,
    "TmuxCapture":      _tmux_capture_pane,
    "TmuxListPanes":    _tmux_list_panes,
    "TmuxSelectPane":   _tmux_select_pane,
    "TmuxKillPane":     _tmux_kill_pane,
    "TmuxNewWindow":    _tmux_new_window,
    "TmuxListWindows":  _tmux_list_windows,
    "TmuxResizePane":   _tmux_resize_pane,
}


def register_tmux_tools() -> int:
    """Register all tmux tools. Returns number of tools registered."""
    if not tmux_available():
        return 0

    schema_map = {s["name"]: s for s in TMUX_TOOL_SCHEMAS}
    count = 0
    for name, func in _TOOL_FUNCS.items():
        register_tool(ToolDef(
            name=name,
            schema=schema_map[name],
            func=func,
            read_only=name in _READ_ONLY_TOOLS,
            concurrent_safe=True,
        ))
        count += 1
    return count
