#!/usr/bin/env python3
"""
CheetahClaws — A Fast, Easy-to-Use, Python-Native Personal AI Assistant for Any Model, Built to Work for You Autonomously 24/7.

Usage:
  python cheetahclaws.py [options] [prompt]

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
  /resume [f] Resume last auto-saved session (or a named file)
  /history    Print conversation history
  /context    Show context window usage
  /cost       Show API cost this session
  /status     Show current session status (model, mode, tokens, cost)
  /verbose    Toggle verbose mode
  /thinking   Toggle extended thinking
  /permissions [mode]  Set permission mode
  /cwd [path] Show or change working directory
  /compact    Compact conversation history to save context space
  /init       Initialize a CLAUDE.md file in the current directory
  /export [f] Export conversation history to a Markdown file
  /copy       Copy the last assistant response to clipboard
  /doctor     Diagnose installation health and tool connectivity
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
  /checkpoint       List checkpoints or restore one (/checkpoint restore <id>)
  /rewind [id]      Rewind conversation to a checkpoint
  /plan <desc>      Enter plan mode (write-protect everything except plan file)
  /plan done        Exit plan mode and restore permissions
  /plan status      Show plan mode status
  /brainstorm <topic>  Multi-persona iterative brainstorming session
  /worker           Auto-implement tasks from todo_list.txt
  /agent start <template> [args]  Autonomous agent loop (research_assistant / auto_bug_fixer / paper_writer / auto_coder)
  /agent stop <name>    Stop a running agent
  /agent list           List running agents
  /agent templates      List available task templates
  /ssj              SSJ Developer Mode — power menu (brainstorm, debate, worker, review…)
  /image [prompt]   Send clipboard image to vision model
  /voice            Record voice input, transcribe, and submit
  /voice status     Show available recording and STT backends
  /voice lang <code>  Set STT language (e.g. zh, en, ja — default: auto)
  /proactive [dur]  Background sentinel polling (e.g. /proactive 5m)
  /proactive off    Disable proactive polling
  /cloudsave setup <token>   Configure GitHub token for cloud sync
  /cloudsave        Upload current session to GitHub Gist
  /cloudsave push [desc]     Upload with optional description
  /cloudsave auto on|off     Toggle auto-upload on exit
  /cloudsave list   List your cheetahclaws Gists
  /cloudsave load <gist_id>  Download and load a session from Gist
  /subscribe <topic> [schedule] [--telegram] [--slack]
                    Subscribe to AI-monitored topic (ai_research, stock_TSLA, crypto_BTC, world_news, custom:<query>)
  /subscriptions    List active subscriptions
  /unsubscribe <topic>  Remove a subscription
  /monitor run [topic]  Run monitor(s) now and print AI report
  /monitor start    Start background scheduler (runs subscriptions on schedule)
  /monitor stop     Stop background scheduler
  /monitor status   Show scheduler status and subscription overview
  /monitor set telegram <token> <chat_id>  Configure Telegram delivery
  /monitor set slack <token> <channel_id>  Configure Slack delivery
  /monitor topics   List available built-in topics
  /telegram <bot_token> <chat_id>  Start Telegram bridge
  /telegram stop|status             Stop or check Telegram bridge
  /wechat login                     Authenticate WeChat via QR code
  /wechat stop|status               Stop or check WeChat bridge
  /slack <token> <channel_id>       Start Slack bridge (Web API)
  /slack stop|status|logout         Stop, check, or clear Slack bridge
  /exit /quit Exit
"""
from __future__ import annotations

# ── Standard library ───────────────────────────────────────────────────────
import os
import re
import sys
import uuid
if sys.platform == "win32":
    os.system("")  # Enable ANSI escape codes on Windows CMD
import json
try:
    import readline
except ImportError:
    readline = None  # Windows compatibility
import atexit
import argparse
import time
import traceback
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

# ── UI / rendering ─────────────────────────────────────────────────────────
from ui.render import (
    C, clr, info, ok, warn, err, _truncate_err_global,
    render_diff, _has_diff,
    stream_text, stream_thinking, flush_response,
    _start_tool_spinner, _stop_tool_spinner, _change_spinner_phrase,
    set_spinner_phrase, set_rich_live,
    print_tool_start, print_tool_end,
    _RICH, console,
)

# ── Input layer (prompt_toolkit with readline fallback) ──────────────────
import ui.input as _ui_input
_pt_read_line = _ui_input.read_line
HAS_PROMPT_TOOLKIT = _ui_input.HAS_PROMPT_TOOLKIT

# ── Bridge commands ────────────────────────────────────────────────────────
import bridges.telegram as _btg
import bridges.wechat   as _bwx
import bridges.slack    as _bslk
from bridges.telegram import cmd_telegram, _tg_send
from bridges.wechat   import cmd_wechat, _wx_start_bridge
from bridges.slack    import cmd_slack, _slack_start_bridge

# ── Session commands ───────────────────────────────────────────────────────
from commands.session import (
    cmd_save, cmd_load, cmd_resume, cmd_history,
    cmd_cloudsave, cmd_exit, save_latest,
)

# ── Config commands ────────────────────────────────────────────────────────
from commands.config_cmd import (
    cmd_model, cmd_config, cmd_verbose, cmd_thinking,
    cmd_permissions, cmd_cwd, _interactive_ollama_picker,
)

# ── Core commands ──────────────────────────────────────────────────────────
from commands.core import (
    cmd_help, cmd_clear, cmd_context, cmd_cost, cmd_compact,
    cmd_init, cmd_export, cmd_copy, cmd_status, cmd_doctor,
    cmd_proactive, cmd_image, run_setup_wizard,
)

# ── Checkpoint / Plan commands ─────────────────────────────────────────────
from commands.checkpoint_plan import cmd_checkpoint, cmd_rewind, cmd_plan

# ── Advanced commands ──────────────────────────────────────────────────────
from commands.advanced import (
    cmd_brainstorm, cmd_worker, cmd_ssj,
    cmd_memory, cmd_agents, cmd_skills, cmd_mcp, cmd_plugin, cmd_tasks,
    _save_synthesis, _print_background_notifications,
)

# ── Agent (autonomous loop) command ───────────────────────────────────────
from commands.agent_cmd import cmd_agent

# ── Monitor / Subscribe commands ──────────────────────────────────────────
from commands.monitor_cmd import cmd_subscribe, cmd_subscriptions, cmd_unsubscribe, cmd_monitor

# ── Tools / thread-local bridge state ─────────────────────────────────────
from tools import (
    ask_input_interactive,
    _tg_thread_local, _is_in_tg_turn,
    _wx_thread_local, _is_in_wx_turn,
    _slack_thread_local, _is_in_slack_turn,
)

# ── Live session context (replaces config["_run_query_callback"] etc.) ─────
import runtime

def _read_version() -> str:
    """Read version from pyproject.toml (single source of truth)."""
    try:
        _toml = Path(__file__).resolve().parent / "pyproject.toml"
        for _line in _toml.read_text(encoding="utf-8").splitlines():
            if _line.startswith("version"):
                return _line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("cheetahclaws")
    except Exception:
        return "0.0.0"


VERSION = _read_version()

# ── Load feature modules from modular/ ecosystem ───────────────────────────
# Commands from modular/ are merged into COMMANDS after the dict is built.
# Each module is optional — missing modules degrade gracefully.
try:
    from modular import load_all_commands as _modular_load_commands
    _MODULAR_AVAILABLE = True
except ImportError:
    _MODULAR_AVAILABLE = False
    def _modular_load_commands(): return {}  # type: ignore[misc]

# Quick availability checks for UI (help text, menus)
def _modular_has(cmd_name: str) -> bool:
    if not _MODULAR_AVAILABLE:
        return False
    try:
        import modular
        cmds = modular.load_all_commands()
        return cmd_name in cmds
    except Exception:
        return False

_VIDEO_AVAILABLE = _modular_has("video")
_VOICE_MODULAR   = _modular_has("voice")   # voice from modular (has its own cmd)

# Fallback stubs shown when a module is absent
def _missing_module_cmd(name: str):
    def _stub(args: str, _state, config) -> bool:
        warn(f"'{name}' module not available. Check modular/{name}/.")
        return True
    _stub.__name__ = f"cmd_{name}"
    return _stub


# ── Permission prompt ──────────────────────────────────────────────────────

def ask_permission_interactive(desc: str, config: dict) -> bool:
    text = ask_input_interactive(f"  Allow: {desc}  [y/N/a(ccept-all)] ", config).strip().lower()

    if text == "a" or text == "accept all" or text == "accept-all":
        config["permission_mode"] = "accept-all"
        if _is_in_tg_turn(config):
            token = config.get("telegram_token")
            chat_id = config.get("telegram_chat_id")
            _tg_send(token, chat_id, "✅ Permission mode set to accept-all for this session.")
        else:
            ok("  Permission mode set to accept-all for this session.")
        return True

    return text in ("y", "yes")


# ── Proactive watcher ──────────────────────────────────────────────────────

def _proactive_watcher_loop(config):
    """Background daemon that fires a wake-up prompt after a period of inactivity."""
    while True:
        time.sleep(1)
        sctx = runtime.get_ctx(config)
        if not sctx.proactive_enabled:
            continue
        try:
            now = time.time()
            interval = sctx.proactive_interval
            last = sctx.last_interaction_time
            if now - last >= interval:
                sctx.last_interaction_time = now
                cb = sctx.run_query
                if cb:
                    cb(f"(System Automated Event) You have been inactive for {interval} seconds. "
                       "Before doing anything else, review your previous messages in this conversation. "
                       "If you said you would implement, fix, or do something and didn't finish it, "
                       "continue and complete that work now. "
                       "Otherwise, check if you have any pending tasks to execute or simply say 'No pending tasks'.")
        except Exception as e:
            import logging_utils as _log
            _log.error("proactive_watcher_error", error=str(e)[:200])


# ── Slash commands ─────────────────────────────────────────────────────────

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
    # "voice" and "video" are loaded from modular/ by _load_external_commands_into()
    "image":       cmd_image,
    "img":         cmd_image,
    "brainstorm":  cmd_brainstorm,
    "worker":      cmd_worker,
    "agent":       cmd_agent,
    "ssj":         cmd_ssj,
    "telegram":    cmd_telegram,
    "wechat":      cmd_wechat,
    "weixin":      cmd_wechat,
    "slack":       cmd_slack,
    "checkpoint":  cmd_checkpoint,
    "rewind":      cmd_rewind,
    "plan":        cmd_plan,
    "subscribe":   cmd_subscribe,
    "subscriptions": cmd_subscriptions,
    "subs":        cmd_subscriptions,
    "unsubscribe": cmd_unsubscribe,
    "monitor":     cmd_monitor,
    "compact":     cmd_compact,
    "init":        cmd_init,
    "export":      cmd_export,
    "copy":        cmd_copy,
    "status":      cmd_status,
    "doctor":      cmd_doctor,
    "setup":       lambda a, s, c: (run_setup_wizard(c), True)[1],
    "exit":        cmd_exit,
    "quit":        cmd_exit,
    "resume":      cmd_resume,
}

# ── Load commands from modular/ ecosystem + installed plugins ──────────────
def _load_external_commands_into(commands_dict: dict) -> None:
    """Merge commands from modular/ modules and user-installed plugins into COMMANDS."""
    # 1. modular/ ecosystem (auto-discovered, ships with the project)
    try:
        for cmd_name, cmd_def in _modular_load_commands().items():
            if cmd_name not in commands_dict and callable(cmd_def.get("func")):
                commands_dict[cmd_name] = cmd_def["func"]
                for alias in cmd_def.get("aliases", []):
                    commands_dict.setdefault(alias, cmd_def["func"])
    except Exception:
        pass

    # 2. user-installed plugins (via /plugin install)
    try:
        from plugin.loader import load_plugin_commands
        for cmd_name, cmd_def in load_plugin_commands().items():
            if cmd_name not in commands_dict and callable(cmd_def.get("func")):
                commands_dict[cmd_name] = cmd_def["func"]
                for alias in cmd_def.get("aliases", []):
                    commands_dict.setdefault(alias, cmd_def["func"])
    except Exception:
        pass

_load_external_commands_into(COMMANDS)


def __getattr__(name: str):
    """Module-level __getattr__ for backward-compatible access to modular attributes.

    Exposes cmd_voice and _voice_language from modular/voice/cmd.py so that
    external code and tests can access cheetahclaws.cmd_voice and
    cheetahclaws._voice_language without the voice module being hard-coded here.
    """
    if name == "cmd_voice":
        if "voice" in COMMANDS:
            return COMMANDS["voice"]
        return _missing_module_cmd("voice")
    if name == "_voice_language":
        try:
            import modular.voice.cmd as _vc
            return _vc._voice_language
        except Exception:
            return "auto"
    raise AttributeError(f"module 'cheetahclaws' has no attribute {name!r}")


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
        # cmd_voice/cmd_image/cmd_brainstorm/cmd_plan return sentinels to ask the REPL to run_query
        if isinstance(result, tuple) and result[0] in ("__voice__", "__image__", "__brainstorm__", "__worker__", "__ssj_cmd__", "__ssj_query__", "__ssj_debate__", "__ssj_passthrough__", "__ssj_promote_worker__", "__plan__"):
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
    **({"voice": ("Voice input (record → STT)", ["lang", "status", "device"])} if _VOICE_MODULAR else {}),
    **({"tts": ("AI voice generator: text → any style → audio file", ["status"])} if _VOICE_MODULAR else {}),
    "image":       ("Send clipboard image to model",      []),
    "img":         ("Send clipboard image (alias)",       []),
    "brainstorm":  ("Multi-persona AI debate + auto tasks", []),
    "worker":      ("Auto-implement pending tasks",       []),
    "agent":       ("Autonomous agent loop (task templates)", ["start", "stop", "list", "status", "templates"]),
    "ssj":         ("SSJ Developer Mode — power menu",    []),
    "telegram":    ("Telegram bot bridge",                ["stop", "status"]),
    "wechat":      ("WeChat bridge (iLink Bot API)",      ["stop", "status"]),
    "slack":       ("Slack bot bridge (Web API)",         ["stop", "status", "logout"]),
    **({"video": ("AI video factory: story→voice→images→mp4", ["status", "niches"])} if _VIDEO_AVAILABLE else {}),
    "checkpoint":  ("List / restore checkpoints",          ["clear"]),
    "rewind":      ("Rewind to checkpoint (alias)",        ["clear"]),
    "plan":        ("Enter/exit plan mode",                ["done", "status"]),
    "compact":     ("Compact conversation history",         []),
    "init":        ("Initialize CLAUDE.md template",        []),
    "export":      ("Export conversation to file",          []),
    "copy":        ("Copy last response to clipboard",      []),
    "status":      ("Show session status and model info",   []),
    "doctor":      ("Diagnose installation health",         []),
    "setup":       ("Run interactive setup wizard",         []),
    "exit":        ("Exit cheetahclaws",              []),
    "quit":        ("Exit (alias for /exit)",             []),
    "resume":      ("Resume last session",                []),
}


_rl_current_prompt = ""   # set by _read_input before each input() call


def setup_readline(history_file: Path):
    global _rl_current_prompt
    if readline is None:
        return
    try:
        readline.read_history_file(str(history_file))
    except (FileNotFoundError, PermissionError, OSError):
        pass
    readline.set_history_length(1000)
    def _save_history():
        try:
            readline.write_history_file(str(history_file))
        except Exception:
            pass
    atexit.register(_save_history)

    # Allow "/" to be part of a completion token so "/model" is one word
    delims = readline.get_completer_delims().replace("/", "")
    readline.set_completer_delims(delims)

    def completer(text: str, state: int):
        line = readline.get_line_buffer()

        # ── Completing a command name: line starts with "/" and no space yet ──
        if line.startswith("/") and " " not in line:
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
        is_cmd = line.startswith("/") and " " not in line

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
        # Redisplay prompt + current buffer so typing continues on the prompt line
        sys.stdout.write(_rl_current_prompt + readline.get_line_buffer())
        sys.stdout.flush()

    readline.set_completion_display_matches_hook(display_matches)
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


# ── Main REPL ──────────────────────────────────────────────────────────────

def repl(config: dict, initial_prompt: str = None):
    from config import HISTORY_FILE
    from context import build_system_prompt
    from agent import AgentState, run, TextChunk, ThinkingChunk, ToolStart, ToolEnd, TurnDone, PermissionRequest

    if HAS_PROMPT_TOOLKIT:
        # Inject live providers so ui.input's completer enumerates the same
        # command set the dispatcher accepts (includes plugin/modular adds).
        _ui_input.setup(lambda: COMMANDS, lambda: _CMD_META)
    else:
        setup_readline(HISTORY_FILE)

    # prompt_toolkit's FileHistory uses an incompatible format to readline's
    # history file, so give it a sibling path. Both persist across sessions;
    # toggling CHEETAH_PT_INPUT only switches which file is active.
    PT_HISTORY_FILE = HISTORY_FILE.with_name("input_history_pt.txt")

    state = AgentState()
    verbose = config.get("verbose", False)

    # Create the per-session RuntimeContext early so all wiring uses it, not
    # the global singleton.  session_id must be set in config before any
    # bridge or tool code runs so they can look up the right context.
    import checkpoint as ckpt
    session_id = uuid.uuid4().hex[:8]
    config["_session_id"] = session_id
    session_ctx = runtime.get_session_ctx(session_id)
    session_ctx.tg_send = _tg_send
    session_ctx.agent_state = state

    ckpt.set_session(session_id)
    ckpt.cleanup_old_sessions()
    # Initial snapshot: capture the "blank slate" before any prompts
    ckpt.make_snapshot(session_id, state, config, "(initial state)", tracked_edits=None)

    # Banner
    if not initial_prompt:
        from providers import detect_provider

        # ── Cheetah startup animation ──
        _CHEETAH_FRAMES = [
            "     ✦",
            "    ✦ ·",
            "   ✦ · ·",
            "  ✦ · · ·",
            " ✦ · · · ·",
            "✦ · · · · ·",
        ]
        _CHEETAH_LOGO = [
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⣶⣶⢦⣤⣤⣤⣄⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣼⣿⣶⣿⣾⣿⣿⣿⣿⣿⣿⣶⣦⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠙⣿⣿⣟⠛⠛⠛⠛⠛⠛⠛⠛⠛⠻⠿⠛⠛⠛⠛⠉⠉⠉⠙⠛⠛⠛⠿⠿⣿⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⢈⣻⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣐⣒⣲⡦⣠⣤⣀⠀⠀⠀⠀⠉⠙⠻⣿⣿⣿⣦⣄⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⢀⣴⣾⣿⣿⠿⠿⠿⠦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⣽⣿⣿⣿⣿⣿⣴⣤⣀⠀⠀⠀⠙⠻⣿⣿⣷⣄⠀⠀⠀⠀",
            "⠀⠀⣠⣶⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣠⣄⣀⡈⠙⠩⠽⢻⣿⣿⣶⣀⠀⠀⠀⠈⢿⣿⣿⡆⠀⠀⠀",
            "⢀⠼⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣿⣯⡉⠙⠲⠦⣤⣌⣙⣻⣿⣿⣦⣤⣴⣿⣿⣿⡇⠀⠀⠀",
            "⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⣿⣿⣿⣿⣿⣿⠿⠿⠛⠛⠋⠉⠉⠁⠈⠉⠻⣿⡄⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣶⣶⣶⣶⣶⣶⣦⣬⣀⡀⠀⠉⠛⠻⠯⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢈⣿ ⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣶⣿⣿⣿⠿⠛⡉⠁⠐⠈⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣤⣤⣤⣴⣶⣾⣿⣿⣿⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⠯⠐⠀⠈⠀⠀⠀⠀⠀⠀⢀⣀⣤⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⣿⣿⡿⠃⣿⡇",
            "⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⣿⣿⣿⣿⣿⣶⣦⣀⠀⠀⠀⠀⠀⠀⢰⣿⡟⠋⠉⠉⠉⠙⣿⣿⠶⣦⣄⣀⠀⠀⠀⠀⠀⣠⣿⣿⣦⣿⠁",
            "⠀⠀⠀⠀⠀⠀⠀⠐⠛⠛⠛⠛⠛⠿⠿⣿⣿⣿⣿⣦⡀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⢀⠘⠏⠀⠺⣿⡿⠻⣶⣶⣶⡾⠟⠛⢩⣿⡏⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⣿⣿⣦⡀⠀⠸⣿⡶⠀⠀⠀⠀⠸⡏⠀⠀⠀⠛⠁⠀⣿⣯⡟⠀⠀⠀⢸⡟⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣿⣿⣄⠀⢿⡇⠀⠀⠀⠀⠀⢷⣀⠀⠀⠀⠀⠀⢸⠏⠀⠀⠀⠀⠈⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⢿⣟⢿⣿⡆⠈⣿⣿⠋⠀⠀⠀⡞⣿⡄⠀⠀⣴⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣜⣟⢽⢿⣿⣟⣿⣿⡀⠘⣷⡀⠀⣀⣾⡇⠘⠿⢷⣾⣿⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣺⣿⣽⣿⣿⣿⣹⢿⣻⣿⡇⠀⠈⢿⣾⣿⣿⡷⠾⠟⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣽⣻⣫⡼⠛⠉⠁⢸⣿⡇⠀⠀⠀⠛⠉⠀⠀⣠⣾⡿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⢀⣰⣾⣿⣿⣽⠽⠋⠉⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀⠀⢀⣠⣴⣿⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⣸⣿⠽⠟⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⣷⣶⣶⣿⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠘⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠼⠿⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        ]

        # Spinning galaxy animation
        _GALAXY_FRAMES = ["◜", "◝", "◞", "◟"]
        try:
            for i in range(8):
                frame = _GALAXY_FRAMES[i % 4]
                sys.stdout.write(f"\r  {clr(frame, 'cyan', 'bold')} Initializing Cheetah...")
                sys.stdout.flush()
                time.sleep(0.12)
            sys.stdout.write(f"\r{' ' * 40}\r")
            sys.stdout.flush()
        except Exception:
            pass

        # Print logo
        for line in _CHEETAH_LOGO:
            print(clr(line, "cyan", "bold"))
        print()

        model    = config["model"]
        pname    = detect_provider(model)
        model_clr = clr(model, "cyan", "bold")
        prov_clr  = clr(f"({pname})", "dim")
        pmode     = clr(config.get("permission_mode", "auto"), "yellow")
        ver_clr   = clr(f"v{VERSION}", "green")

        print(clr("  ╭─ ", "dim") + clr("CheetahClaws ", "cyan", "bold") + ver_clr + clr(" ─────────────────────────────────╮", "dim"))
        print(clr("  │", "dim") + clr("  Model: ", "dim") + model_clr + " " + prov_clr)
        print(clr("  │", "dim") + clr("  Permissions: ", "dim") + pmode)
        print(clr("  │", "dim") + clr("  /model to switch · /help for commands", "dim"))
        print(clr("  ╰──────────────────────────────────────────────────────╯", "dim"))

        # Show active non-default settings
        active_flags = []
        if config.get("verbose"):
            active_flags.append("verbose")
        if config.get("thinking"):
            active_flags.append("thinking")
        if session_ctx.proactive_enabled:
            active_flags.append("proactive")
        if config.get("telegram_token") and config.get("telegram_chat_id"):
            active_flags.append("telegram")
        if config.get("wechat_token"):
            active_flags.append("wechat")
        if config.get("slack_token") and config.get("slack_channel"):
            active_flags.append("slack")
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
    set_rich_live(config.get("rich_live", _rich_live_default))

    # Initialize proactive polling state via RuntimeContext (defaults already set)
    session_ctx.last_interaction_time = time.time()
    if session_ctx.proactive_thread is None:
        t = threading.Thread(target=_proactive_watcher_loop, args=(config,), daemon=True)
        session_ctx.proactive_thread = t
        t.start()

    def run_query(user_input: str, is_background: bool = False):
        nonlocal verbose

        with query_lock:
            verbose = config.get("verbose", False)

            # Rebuild system prompt each turn (picks up cwd changes, etc.)
            system_prompt = build_system_prompt(config)

            if is_background and not session_ctx.telegram_incoming:
                print(clr("\n\n[Background Event Triggered]", "yellow"))
            session_ctx.in_telegram_turn = session_ctx.telegram_incoming
            session_ctx.telegram_incoming = False

            print(clr("\n╭─ CheetahClaws ", "dim") + clr("●", "green") + clr(" ─────────────────────────", "dim"))

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
                        # Fire bridge streaming hook
                        _hook = session_ctx.on_text_chunk
                        if _hook:
                            try:
                                _hook(event.text)
                            except Exception:
                                pass

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
                        _hook = session_ctx.on_tool_start
                        if _hook:
                            try:
                                _hook(event.name, event.inputs or {})
                            except Exception:
                                pass

                    elif isinstance(event, PermissionRequest):
                        _stop_tool_spinner()
                        flush_response()
                        event.granted = ask_permission_interactive(event.description, config)
                        # Live will restart automatically on next TextChunk

                    elif isinstance(event, ToolEnd):
                        print_tool_end(event.name, event.result, verbose)
                        _hook = session_ctx.on_tool_end
                        if _hook:
                            try:
                                _hook(event.name, str(event.result or "")[:500])
                            except Exception:
                                pass
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
                flush_response()
                import urllib.error
                # Catch 404 Not Found (Ollama model missing)
                if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                    from providers import detect_provider
                    if detect_provider(config["model"]) == "ollama":
                        err(f"Ollama model '{config['model']}' not found.")
                        if _interactive_ollama_picker(config):
                            if state.messages and state.messages[-1]["role"] == "user":
                                state.messages.pop()
                            return run_query(user_input, is_background)
                        return
                # ── Actionable error messages for common API failures ─────
                err_str = str(e).lower()
                err_cls = type(e).__name__
                hint = ""
                if "auth" in err_cls.lower() or "authentication" in err_str or "invalid.*api.key" in err_str or "401" in err_str:
                    hint = "Check your API key: /config or set the appropriate env var (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)"
                elif isinstance(e, (ConnectionError, OSError)) or "connection" in err_str or isinstance(e, urllib.error.URLError):
                    hint = "Network error — check your internet connection or the API endpoint URL."
                    if "ollama" in err_str or "localhost" in err_str or "11434" in err_str:
                        hint = "Cannot connect to Ollama. Is it running? Start with: ollama serve"
                elif "rate_limit" in err_str or "rate limit" in err_str or err_cls == "RateLimitError":
                    hint = "Rate limited by the API. Wait a moment and retry."
                elif "not_found" in err_str or "model" in err_str and "not found" in err_str or "does not exist" in err_str:
                    hint = f"Model '{config.get('model', '?')}' not found. Check available models with /model"
                elif "insufficient" in err_str and ("quota" in err_str or "balance" in err_str or "credit" in err_str):
                    hint = "Insufficient API credits. Check your billing at your provider's dashboard."
                err(f"Error: {err_cls}: {_truncate_err_global(str(e))}")
                if hint:
                    warn(f"Hint: {hint}")
                warn("Your conversation is intact. You can retry or type a new message.")

            _stop_tool_spinner()
            flush_response()  # stop Live, commit any remaining text
            print(clr("╰──────────────────────────────────────────────", "dim"))
            print()

            # If this was a background task, we redraw the prompt for the user
            if is_background:
                print(clr(f"\n[{Path.cwd().name}] » ", "yellow"), end="", flush=True)

                # If Telegram is connected and this background task didn't originate from a live Telegram query,
                # forward the alert to the Telegram user so they are notified!
                is_tg_turn = session_ctx.in_telegram_turn
                ttok = config.get("telegram_token")
                tchat = config.get("telegram_chat_id")
                if not is_tg_turn and ttok and tchat:
                    if state.messages and state.messages[-1].get("role") == "assistant":
                        ans_content = state.messages[-1].get("content", "")
                        if isinstance(ans_content, list):
                            parts = [b["text"] if isinstance(b, dict) else str(b) for b in ans_content if (isinstance(b, dict) and b.get("type") == "text") or isinstance(b, str)]
                            ans_content = "\n".join(parts)
                        if ans_content:
                            _tg_send(ttok, tchat, ans_content)

        # Drain any AskUserQuestion prompts raised during this turn
        from tools import drain_pending_questions
        drain_pending_questions(config)

        # ── Auto-snapshot after each turn ──
        try:
            tracked = ckpt.get_tracked_edits()
            # Throttle: skip snapshot only if no files changed AND no new messages
            last_snaps = ckpt.list_snapshots(session_id)
            skip = False
            if not tracked and last_snaps:
                if len(state.messages) == last_snaps[-1].get("message_index", -1):
                    skip = True
            if not skip:
                ckpt.make_snapshot(session_id, state, config, user_input, tracked_edits=tracked)
            ckpt.reset_tracked()
        except Exception:
            pass  # never let checkpoint errors break the REPL

        session_ctx.last_interaction_time = time.time()

    session_ctx.run_query = lambda msg: run_query(msg, is_background=True)

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
            _todo_path = str(Path(brain_out_file).parent / "todo_list.txt")
            run_query(
                f"Based on the Master Plan you just synthesized, generate a todo list file at {_todo_path}. "
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

    session_ctx.handle_slash = _handle_slash_from_telegram

    # ── Auto-start Telegram bridge if configured ──────────────────────
    if config.get("telegram_token") and config.get("telegram_chat_id"):
        if not (_btg._telegram_thread and _btg._telegram_thread.is_alive()):
            _btg._telegram_stop.clear()
            _btg._telegram_thread = threading.Thread(
                target=_btg._tg_poll_loop,
                args=(config["telegram_token"], config["telegram_chat_id"], config),
                daemon=True
            )
            _btg._telegram_thread.start()

    # ── Auto-start WeChat bridge if configured ────────────────────────
    if config.get("wechat_token"):
        if not (_bwx._wechat_thread and _bwx._wechat_thread.is_alive()):
            _wx_start_bridge(config)

    # ── Auto-start Slack bridge if configured ─────────────────────────
    if config.get("slack_token") and config.get("slack_channel"):
        if not (_bslk._slack_thread and _bslk._slack_thread.is_alive()):
            _slack_start_bridge(config)

    # ── Rapid Ctrl+C force-quit ─────────────────────────────────────────
    # 3 Ctrl+C presses within 2 seconds → immediate hard exit
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
    _PASTE_START = "\x1b[200~"
    _PASTE_END   = "\x1b[201~"
    _bpm_active  = sys.stdin.isatty() and sys.platform != "win32"

    if _bpm_active:
        sys.stdout.write("\x1b[?2004h")   # enable bracketed paste mode
        sys.stdout.flush()

    def _read_input(prompt: str) -> str:
        """Read one user turn, collecting multi-line pastes as a single string."""
        global _rl_current_prompt
        import select as _sel

        # ── Phase 1a: prompt_toolkit (TTY + library available + not opted out) ─
        # Handles bracketed paste natively, so phase-2/3 are skipped on success.
        # Preserves the "(pasted N lines)" notification for parity with the
        # readline-based paste handling in phase 2/3.
        if (
            HAS_PROMPT_TOOLKIT
            and sys.stdin.isatty()
            and os.environ.get("CHEETAH_PT_INPUT", "1") != "0"
        ):
            try:
                result = _pt_read_line(prompt, PT_HISTORY_FILE)
                if "\n" in result:
                    n = result.count("\n") + 1
                    info(f"  (pasted {n} line{'s' if n > 1 else ''})")
                return result
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception as _pt_err:
                warn(
                    f"prompt_toolkit failed ({type(_pt_err).__name__}: {_pt_err}); "
                    "falling back to readline"
                )
                _ui_input.reset_session()
                # fall through to phase 1b

        # ── Phase 1b: get first line via readline (history, line-edit intact) ──
        # Wrap ANSI codes so readline counts them as zero-width (#29/#31).
        rl_prompt = re.sub(r'(\x1b\[[0-9;]*m)', r'\001\1\002', prompt)
        _rl_current_prompt = prompt   # for display_matches to redisplay
        first = input(rl_prompt)

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
            # Context usage indicator in prompt
            ctx_hint = ""
            try:
                from compaction import estimate_tokens, get_context_limit
                used = estimate_tokens(state.messages)
                limit = get_context_limit(config.get("model", ""))
                pct = int(used / limit * 100) if limit else 0
                if pct >= 70:
                    ctx_hint = clr(f" {pct}%", "red")
                elif pct >= 40:
                    ctx_hint = clr(f" {pct}%", "yellow")
                elif state.messages:
                    ctx_hint = clr(f" {pct}%", "dim")
            except Exception:
                pass
            prompt = clr(f"\n[{cwd_short}]", "dim") + ctx_hint + clr(" ", "dim") + clr("» ", "cyan", "bold")
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

        # ── Shell escape: !command runs directly in the system shell ──
        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if shell_cmd:
                print(clr(f"  $ {shell_cmd}", "dim"))
                try:
                    import subprocess as _sp
                    _sp.run(shell_cmd, shell=True)
                except Exception as e:
                    warn(f"Shell error: {e}")
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

            # Plan sentinel: ("__plan__", description)
            if result[0] == "__plan__":
                _, plan_desc = result
                try:
                    run_query(f"Please analyze the codebase and create a detailed implementation plan for: {plan_desc}")
                except KeyboardInterrupt:
                    _track_ctrl_c()
                    print(clr("\n  (interrupted)", "yellow"))
                break

            # SSJ passthrough: user typed a /command inside SSJ menu
            if result[0] == "__ssj_passthrough__":
                _, slash_line = result
                # Guard against /ssj re-entering itself infinitely
                if slash_line.strip().lower() == "/ssj":
                    result = handle_slash("/ssj", state, config)
                    continue
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
                    _todo_path = str(Path(brain_out_file).parent / "todo_list.txt")
                    print(clr("\n  ── Generating TODO List from Master Plan ──", "dim"))
                    run_query(
                        f"Based on the Master Plan you just synthesized, generate a todo list file at {_todo_path}. "
                        "Format: one task per line, each starting with '- [ ] '. "
                        "Order by priority. Include ALL actionable items from the plan. "
                        "Use the Write tool to create the file. Do NOT explain, just write the file now."
                    )
                    info(f"TODO list saved to {_todo_path}. Edit it freely, then use /worker to start implementing.")
                except KeyboardInterrupt:
                    _track_ctrl_c()
                    print(clr("\n  (interrupted)", "yellow"))
                if _from_ssj_flag:
                    result = handle_slash("/ssj", state, config)
                    continue
                break

            # Promote-then-Worker: generate todo_list.txt from brainstorm .md, then run worker
            if result[0] == "__ssj_promote_worker__":
                _, md_path, todo_path_str, task_nums_str, max_workers_str = result
                promote_prompt = (
                    f"Read the brainstorm file {md_path} and extract all actionable ideas. "
                    f"Convert each idea into a task with checkbox format (- [ ] task description). "
                    f"Write them to {todo_path_str} using the Write tool. Prioritize by impact. "
                    f"Do NOT explain, just write the file now."
                )
                print(clr(f"\n  ── Generating TODO list from {Path(md_path).name} ──", "dim"))
                try:
                    run_query(promote_prompt)
                except KeyboardInterrupt:
                    _track_ctrl_c()
                    print(clr("\n  (interrupted)", "yellow"))
                    result = handle_slash("/ssj", state, config)
                    continue
                # Now run worker on the newly created file
                worker_args = f"--path {todo_path_str}"
                if task_nums_str:
                    worker_args += f" --tasks {task_nums_str}"
                if max_workers_str and max_workers_str.isdigit():
                    worker_args += f" --workers {max_workers_str}"
                inner = handle_slash(f"/worker {worker_args}".strip(), state, config)
                if isinstance(inner, tuple):
                    result = ("__ssj_wrap__", inner)
                    continue
                result = handle_slash("/ssj", state, config)
                continue

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

            # Debate sentinel: ("__ssj_debate__", filepath, nagents, rounds, out_file)
            # Drives the debate round-by-round, showing a spinner before each expert's turn.
            if result[0] == "__ssj_debate__":
                _, _dfile, _nagents, _rounds, _debate_out = result
                import random as _random

                # ── Stdout wrapper: stops spinner on first real (non-\r) output ──
                class _DebateSpinnerWrapper:
                    def __init__(self, real_out):
                        self._real = real_out
                        self._stopped = False
                    def write(self, s):
                        if not self._stopped and s and not s.startswith('\r'):
                            self._stopped = True
                            _stop_tool_spinner()
                            self._real.write('\n')
                        return self._real.write(s)
                    def flush(self):   return self._real.flush()
                    def __getattr__(self, name): return getattr(self._real, name)

                def _spin_and_query(phrase, prompt):
                    """Show spinner with phrase, stop it on first model output, run query."""
                    set_spinner_phrase(phrase)
                    _start_tool_spinner()
                    _orig = sys.stdout
                    sys.stdout = _DebateSpinnerWrapper(sys.stdout)
                    try:
                        run_query(prompt)
                    finally:
                        _stop_tool_spinner()
                        sys.stdout = _orig

                try:
                    # ── Step 1: Read file and assign expert personas ──────────
                    _spin_and_query(
                        "⚔️  Assembling expert panel...",
                        f"Read the file {_dfile}. Then introduce the {_nagents} expert debaters you will "
                        f"role-play, each with a distinct focus area chosen to best challenge each other "
                        f"(e.g. architecture, performance, security, UX, testing, maintainability). "
                        f"List their names and focus areas. Do NOT debate yet."
                    )

                    # ── Step 2: Each round, each expert takes a turn ──────────
                    for _r in range(1, _rounds + 1):
                        for _e in range(1, _nagents + 1):
                            _phase = "opening argument" if _r == 1 else f"round {_r} response"
                            _spin_and_query(
                                _random.choice([
                                    f"⚔️  Round {_r}/{_rounds} — Expert {_e} thinking...",
                                    f"💬  Round {_r}/{_rounds} — Expert {_e} formulating...",
                                    f"🧠  Round {_r}/{_rounds} — Expert {_e} responding...",
                                ]),
                                f"Now speak as Expert {_e}. Give your {_phase}. "
                                f"Be specific, reference the file content, and directly address "
                                f"the previous arguments. Be concise (3-5 key points)."
                            )

                    # ── Step 3: Consensus + save ──────────────────────────────
                    _spin_and_query(
                        "📜  Drafting final consensus...",
                        f"Based on this entire debate, write a final consensus that all experts agree on. "
                        f"List the top actionable changes ranked by impact. "
                        f"Then use the Write tool to save the complete debate transcript and this consensus "
                        f"to: {_debate_out}"
                    )
                    ok(f"Debate complete. Saved to {_debate_out}")

                except KeyboardInterrupt:
                    _track_ctrl_c()
                    _stop_tool_spinner()
                    sys.stdout = sys.__stdout__
                    print(clr("\n  (debate interrupted)", "yellow"))

                result = handle_slash("/ssj", state, config)
                continue

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
        prog="cheetahclaws",
        description="CheetahClaws — minimal Python Claude Code implementation",
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
    parser.add_argument("--setup", action="store_true", help="Run interactive setup wizard")
    parser.add_argument("-h", "--help", action="store_true", help="Show help")

    args = parser.parse_args()

    if args.version:
        print(f"cheetahclaws v{VERSION}")
        sys.exit(0)

    if args.help:
        print(__doc__)
        sys.exit(0)

    from config import load_config, save_config, has_api_key
    from providers import detect_provider, PROVIDERS

    config = load_config()

    # Explicit bootstrap: configure logging, ensure tool registry is ready,
    # and start the optional health-check server.
    from bootstrap import bootstrap as _bootstrap
    _bootstrap(config)

    # Apply CLI overrides first (so key check uses the right provider)
    if args.model:
        m = args.model
        # Convert "provider:model" → "provider/model" only when left side is a known provider
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

    # ── Setup wizard: --setup flag or first-run auto-trigger ─────────────
    from config import CONFIG_FILE
    is_first_run = not CONFIG_FILE.exists() or os.path.getsize(CONFIG_FILE) < 5
    if args.setup or (is_first_run and sys.stdin.isatty() and not args.print_mode):
        run_setup_wizard(config)
        # Reload after wizard may have changed config
        config = load_config()
    elif not has_api_key(config):
        # Check API key for active provider (warn only, don't block local providers)
        pname = detect_provider(config["model"])
        prov  = PROVIDERS.get(pname, {})
        env   = prov.get("api_key_env", "")
        if env:   # local providers like ollama have no env key requirement
            warn(f"No API key found for provider '{pname}'. "
                 f"Set {env} or run: /config {pname}_api_key=YOUR_KEY"
                 f"\n  Or run: cheetahclaws --setup")

    initial = " ".join(args.prompt) if args.prompt else None
    if args.print_mode and not initial:
        err("--print requires a prompt argument")
        sys.exit(1)

    repl(config, initial_prompt=initial)


if __name__ == "__main__":
    main()
