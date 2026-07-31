"""Predicted next prompt — Claude-Code-style ghost text at the input line.

After each foreground turn the auxiliary (cheap/fast) model drafts the single
line the user is most likely to type next. It is pushed into ui.input, which
renders it dim/italic in the empty prompt; Tab (or →) accepts it in full,
anything else simply types over it.

Everything here is best-effort: the draft runs on a background daemon thread,
never blocks the REPL, and stays silent on any failure. Turn off with
`/config input_suggest false` or CHEETAH_SUGGEST=0.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from cheetahclaws.ui import input as ui_input

# Longer than this and the ghost wraps the terminal line, which reads as
# clutter rather than a hint.
MAX_LEN = 90

_SYSTEM = (
    "You predict the NEXT message a developer will type to their coding "
    "assistant, given the conversation so far.\n"
    "Rules:\n"
    "- Reply with that message ONLY: one line, no quotes, no explanation, "
    "no leading dash or bullet.\n"
    "- Keep it under 12 words and phrase it as the user (imperative, "
    "first-person), never as the assistant.\n"
    "- Write it in the same language the user has been using.\n"
    "- Make it the most probable concrete follow-up (run the tests, fix the "
    "failure, commit it, explain a specific part), not a generic pleasantry.\n"
    "- If nothing is plausibly next, reply with exactly: NONE"
)

# Generation counter: a slow draft from turn N must not overwrite the ghost
# that turn N+1 already published.
_lock = threading.Lock()
_generation = 0


def enabled(config: dict) -> bool:
    """True when ghost-text prediction should run for this session."""
    if os.environ.get("CHEETAH_SUGGEST", "1") == "0":
        return False
    if not config.get("input_suggest", True):
        return False
    return bool(ui_input.HAS_PROMPT_TOOLKIT)


def _text_of(content) -> str:
    """Flatten a message `content` (str or Anthropic-style block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def _recent_exchange(messages: list, limit: int = 4) -> list:
    """Last few user/assistant turns, flattened and truncated for the drafter."""
    out = []
    for msg in reversed(messages or []):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _text_of(msg.get("content")).strip()
        if not text:
            continue
        out.append({"role": role, "content": text[:1500]})
        if len(out) >= limit:
            break
    return list(reversed(out))


def _clean(raw: str) -> str:
    """Reduce the model's reply to a usable one-liner, or "" if unusable."""
    line = (raw or "").strip().splitlines()[0].strip() if (raw or "").strip() else ""
    line = line.lstrip("-•*").strip()
    for quote in ('"', "'", "`", "“", "”"):
        line = line.strip(quote)
    line = line.strip()
    if not line or line.upper() == "NONE":
        return ""
    if len(line) > MAX_LEN:
        return ""
    # A drafter that starts explaining itself is not producing a user message.
    if line.lower().startswith(("sure,", "here", "the user", "as an ai")):
        return ""
    return line


def predict(messages: list, config: dict) -> str:
    """Draft the likely next user message. Returns "" when unavailable."""
    exchange = _recent_exchange(messages)
    if not exchange:
        return ""
    try:
        from cheetahclaws.auxiliary import stream_auxiliary
        raw = stream_auxiliary(_SYSTEM, exchange, config)
    except Exception:
        return ""
    return _clean(raw)


def schedule(messages: list, config: dict) -> Optional[threading.Thread]:
    """Draft the next-prompt ghost text in the background. Non-blocking.

    Returns the worker thread (mostly for tests), or None when prediction is
    disabled or there is nothing to work from.
    """
    global _generation
    if not enabled(config):
        return None
    snapshot = _recent_exchange(messages)
    if not snapshot:
        return None

    with _lock:
        _generation += 1
        mine = _generation

    # Clear any leftover ghost from the previous turn straight away — a stale
    # prediction is worse than none while the new one is being drafted.
    ui_input.set_pending_suggestion("")

    def _work():
        text = predict(snapshot, config)
        if not text:
            return
        with _lock:
            if mine != _generation:
                return  # a newer turn already superseded this draft
        ui_input.set_pending_suggestion(text)

    thread = threading.Thread(target=_work, name="cheetah-suggest", daemon=True)
    thread.start()
    return thread
