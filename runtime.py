"""
runtime.py — Live session context for CheetahClaws.

Each REPL session (and each bridge connection) gets its own RuntimeContext
keyed by session_id.  This prevents concurrent sessions from corrupting
each other's callbacks, input events, and agent state.

Use get_session_ctx(session_id) to obtain the context for a specific session.
Use release_session_ctx(session_id) when a session ends to free the entry.
Use get_ctx(config) as a shortcut: reads config["_session_id"] and returns
the corresponding RuntimeContext.

The module-level `ctx` alias points to the "default" session and exists only
for backward compatibility with single-session CLI usage.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent import AgentState


@dataclass
class RuntimeContext:
    """Live references wired up when the REPL starts.  Not persisted to disk."""

    # Unique identifier for this session (matches config["_session_id"])
    session_id: str = "default"

    # Fire a background query from any thread (set by repl())
    run_query: Optional[Callable[[str], None]] = None

    # Process a /slash command coming in from a bridge (set by repl())
    handle_slash: Optional[Callable[[str], str]] = None

    # The active AgentState — message history, token counts, turn count
    agent_state: Optional["AgentState"] = None

    # Low-level Telegram send helper (from bridges.telegram._tg_send)
    tg_send: Optional[Callable] = None

    # Low-level Slack send helper: (channel, text) → None  (set by _slack_poll_loop)
    slack_send: Optional[Callable] = None

    # Low-level WeChat send helper: (user_id, text) → None  (set by _wx_poll_loop)
    wx_send: Optional[Callable] = None

    # Per-bridge synchronous-input synchronisation.
    # ask_input_interactive() sets the event, the poll loop fires it with the
    # user-supplied text.  Using RuntimeContext keeps these out of the config dict
    # and makes the coupling between tools.py and each bridge explicit.
    tg_input_event:    Optional[threading.Event] = None
    tg_input_value:    str = ""
    # Short opaque id baked into inline_keyboard callback_data so a stale
    # click on an older prompt does not deliver the wrong value.  Empty
    # string means "no inline-keyboard prompt is currently waiting".
    tg_callback_prompt_id: str = ""
    # message_id of the most recent inline-keyboard prompt — set so the
    # callback handler can edit it (strip the keyboard, append "✓ <choice>")
    # for clear visual feedback once the user clicks.
    tg_callback_message_id: int = 0
    slack_input_event: Optional[threading.Event] = None
    slack_input_value: str = ""
    wx_input_event:    Optional[threading.Event] = None
    wx_input_value:    str = ""

    # ── QQ bridge ───────────────────────────────────────────────────────────────
    qq_send: Optional[Callable] = None
    qq_input_event: Optional[threading.Event] = None
    qq_input_value: str = ""
    qq_input_target_id: str = ""
    in_qq_turn: bool = False
    qq_current_target_id: str = ""
    qq_current_msg_type: str = ""   # "group" or "c2c"

    # ── Live-streaming hooks (set by bridges before run_query; cleared after) ──
    # on_text_chunk(text)          — called for every TextChunk as it streams
    # on_tool_start(name, inputs)  — called when a tool call begins
    # on_tool_end(name, result)    — called when a tool call finishes
    on_text_chunk:  Optional[Callable[[str], None]] = None
    on_tool_start:  Optional[Callable[[str, dict], None]] = None
    on_tool_end:    Optional[Callable[[str, str], None]] = None

    # ── Runtime state (previously stored in config["_xxx"]) ──────────────────

    # Proactive polling
    proactive_enabled:  bool = False
    proactive_interval: int = 300
    proactive_thread:   Optional[threading.Thread] = None
    last_interaction_time: float = 0.0

    # Bridge turn flags
    in_telegram_turn: bool = False
    in_wechat_turn:   bool = False
    in_slack_turn:    bool = False
    telegram_incoming: bool = False
    qq_incoming: bool = False
    wx_current_user_id:   str = ""
    slack_current_channel: str = ""

    # Web (chat API) bridge synchronization
    web_input_event:  Optional[threading.Event] = None
    web_input_value:  str = ""
    in_web_turn:      bool = False

    # Transient per-turn data
    pending_image: Optional[str] = None

    # Plan mode
    plan_file: Optional[str] = None
    prev_permission_mode: Optional[str] = None

    # Voice
    voice_device_index: Optional[int] = None


# ── Per-session registry ───────────────────────────────────────────────────

_registry: dict[str, RuntimeContext] = {}
_registry_lock = threading.Lock()


def get_session_ctx(session_id: str = "default") -> RuntimeContext:
    """Return (creating if needed) the RuntimeContext for the given session."""
    with _registry_lock:
        if session_id not in _registry:
            _registry[session_id] = RuntimeContext(session_id=session_id)
        return _registry[session_id]


def release_session_ctx(session_id: str) -> None:
    """Remove the RuntimeContext for a session that has ended."""
    with _registry_lock:
        _registry.pop(session_id, None)


def get_ctx(config: dict) -> RuntimeContext:
    """Shortcut: return the RuntimeContext for the session stored in config."""
    return get_session_ctx(config.get("_session_id", "default"))


# ── Backward-compat alias ──────────────────────────────────────────────────
# Single-session CLI code that does `import runtime; runtime.ctx.xxx` still works.
ctx = get_session_ctx("default")
