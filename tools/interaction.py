"""
tools_interaction.py — Interactive input tools: AskUserQuestion, SleepTimer,
and bridge-routing helpers (Telegram / WeChat / Slack).
"""
from __future__ import annotations

import threading
from typing import Optional

# ── Bridge turn-detection (thread-local) ──────────────────────────────────

_tg_thread_local    = threading.local()
_wx_thread_local    = threading.local()
_slack_thread_local = threading.local()
_qq_thread_local    = threading.local()


def _is_in_tg_turn(config: dict) -> bool:
    import runtime
    return (getattr(_tg_thread_local, "active", False)
            or bool(runtime.get_ctx(config).in_telegram_turn))


def _is_in_wx_turn(config: dict) -> bool:
    import runtime
    return (getattr(_wx_thread_local, "active", False)
            or bool(runtime.get_ctx(config).in_wechat_turn))


def _is_in_slack_turn(config: dict) -> bool:
    import runtime
    return (getattr(_slack_thread_local, "active", False)
            or bool(runtime.get_ctx(config).in_slack_turn))


def _is_in_web_turn(config: dict) -> bool:
    import runtime
    return bool(getattr(runtime.get_ctx(config), 'in_web_turn', False))


def _is_in_qq_turn(config: dict) -> bool:
    import runtime
    return (getattr(_qq_thread_local, "active", False)
            or bool(runtime.get_ctx(config).in_qq_turn))


# ── options=… helpers (shared menu rendering + reply resolution) ─────────

def _strip_emojis_punct(s: str) -> str:
    """Reduce a label token to lowercase ASCII letters/digits for matching."""
    import re as _re
    return _re.sub(r'[^a-z0-9]+', '', s.lower())


def _format_menu_block(options) -> str:
    """Render an `options` list as a numbered menu suffix.

    Each row reads ``  [N] <label>  (reply N or <value>)`` so users on
    text-only bridges (Slack / WeChat / terminal fallback) know they can
    tap-or-type either the digit, the canonical value, or a leading word
    of the label.
    """
    if not options:
        return ""
    lines: list[str] = []
    for i, (label, value) in enumerate(options, 1):
        lines.append(f"  [{i}] {label}  (reply `{i}` or `{value}`)")
    return "\n".join(lines)


def _build_value_map(options) -> dict:
    """Lookup table for `_resolve_choice`.

    Keys are lowercase strings the user might send — digit ("1"), the
    canonical value ("y"), or any single token in the label
    ("approve" / "reject" / "accept" / "all"). Values are the canonical
    return value the caller should see.

    First-write-wins: if two options would map the same label-word to
    different values (rare), the earlier option keeps the binding so the
    table stays unambiguous.
    """
    if not options:
        return {}
    table: dict[str, str] = {}
    def _put(key: str, value: str) -> None:
        if key and key not in table:
            table[key] = value
    for i, (label, value) in enumerate(options, 1):
        v = str(value)
        _put(str(i), v)
        _put(v.lower(), v)
        # Also accept individual label tokens (so "approve" / "reject" /
        # "accept" / "all" all work for the standard permission options).
        for token in str(label).split():
            _put(_strip_emojis_punct(token), v)
    return table


def _resolve_choice(raw: str, value_map: dict) -> str:
    """Translate a user reply into the option's canonical value.

    Pass-through when no map (caller didn't pass `options`) or when the
    reply isn't a recognized alias — preserves existing behavior for
    free-text questions.
    """
    if not value_map or not isinstance(raw, str):
        return raw
    key = raw.strip().lower()
    return value_map.get(key, raw)


# ── AskUserQuestion ───────────────────────────────────────────────────────

_INPUT_WAIT_TIMEOUT = 300  # seconds before a remote input request times out


def _ask_user_question(
    question: str,
    options: list[dict] | None = None,
    allow_freetext: bool = True,
    config: dict | None = None,
) -> str:
    """Render a question to the user and synchronously return their answer.

    Runs in the agent thread that invoked the tool: prints the question,
    then delegates to ``ask_input_interactive`` so terminal/Telegram/WeChat/
    Slack/Web bridges all read input through their normal path.
    """
    config = config or {}
    options = options or []

    import re as _re
    from ui.render import clr
    _clean = _re.sub(r'\*\*(.+?)\*\*', r'\1', question)
    _clean = _re.sub(r'`(.+?)`', r'\1', _clean)
    _clean = _re.sub(r'\*(.+?)\*', r'\1', _clean)

    print()
    print(clr("❓ ", "magenta", "bold") + clr(_clean, "bold"))

    if options:
        print()
        for i, opt in enumerate(options, 1):
            label = opt.get("label", "")
            desc  = opt.get("description", "")
            print(clr(f"  [{i}] ", "cyan") + label + (clr(f" — {desc}", "dim") if desc else ""))
        if allow_freetext:
            print(clr("  [0] ", "dim") + clr("Type a custom answer", "dim"))
        print()

        while True:
            raw = ask_input_interactive(
                "Your choice (number or text): ", config
            ).strip()
            if not raw:
                return ""
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(options):
                    return options[idx - 1].get("label", "")
                if idx == 0 and allow_freetext:
                    return ask_input_interactive("Your answer: ", config).strip()
                print(f"Invalid option: {idx}")
                continue
            if allow_freetext:
                return raw
            print("Please choose a number from the list.")

    print()
    return ask_input_interactive("Your answer: ", config).strip()


# ── ask_input_interactive (bridge routing) ────────────────────────────────

def ask_input_interactive(prompt: str, config: dict,
                          menu_text: str = None,
                          options: list[tuple[str, str]] | None = None) -> str:
    """Route input prompt to Telegram / WeChat / Slack bridge or terminal.

    `options` (optional) is a list of ``(button_label, return_value)`` pairs.
    When set, every bridge gives the user a structured way to pick one:

      - **Telegram**: real inline_keyboard buttons; click delivers the value.
      - **Slack / WeChat**: numbered menu rendered into the message; reply
        with the digit, the canonical value, or a label word — all three
        resolve to the value before the caller sees them.
      - **Terminal**: numbered menu printed before the input prompt; same
        digit / value / label-word reply normalization.
      - **Web (chat API)**: existing browser UI handles approval, untouched.

    When ``options`` is None (default), every existing call site keeps its
    current free-text behavior — the helper is purely additive.
    """
    import re as _re
    import threading as _threading
    import runtime as _runtime

    _session_ctx = _runtime.get_session_ctx(config.get("_session_id", "default"))

    # Pre-compute the menu block + value map once so every bridge branch
    # sees the same UX. These are no-ops when options is falsy.
    _menu_block = _format_menu_block(options) if options else ""
    _value_map  = _build_value_map(options) if options else {}

    # ── Slack ──────────────────────────────────────────────────────────────
    if _is_in_slack_turn(config) and _session_ctx.slack_send is not None:
        clean_prompt = _re.sub(r'\x1b\[[0-9;]*m', '', prompt).strip()
        payload = ""
        if menu_text:
            payload += _re.sub(r'\x1b\[[0-9;]*m', '', menu_text).strip() + "\n\n"
        payload += f"❓ Input Required\n{clean_prompt}"
        if _menu_block:
            payload += "\n\n" + _menu_block
        slack_channel = (_runtime.get_ctx(config).slack_current_channel
                         or config.get("slack_channel", ""))
        _session_ctx.slack_send(slack_channel, payload)
        evt = _threading.Event()
        _session_ctx.slack_input_event = evt
        if not evt.wait(timeout=_INPUT_WAIT_TIMEOUT):
            _session_ctx.slack_input_event = None
            return "(timeout: no input received)"
        text = _session_ctx.slack_input_value.strip()
        _session_ctx.slack_input_event = None
        _session_ctx.slack_input_value = ""
        return _resolve_choice(text, _value_map)

    # ── WeChat ─────────────────────────────────────────────────────────────
    if _is_in_wx_turn(config) and _session_ctx.wx_send is not None:
        clean_prompt = _re.sub(r'\x1b\[[0-9;]*m', '', prompt).strip()
        payload = ""
        if menu_text:
            payload += _re.sub(r'\x1b\[[0-9;]*m', '', menu_text).strip() + "\n\n"
        payload += f"❓ 需要输入\n{clean_prompt}"
        if _menu_block:
            payload += "\n\n" + _menu_block
        _session_ctx.wx_send(_runtime.get_ctx(config).wx_current_user_id or "", payload)
        evt = _threading.Event()
        _session_ctx.wx_input_event = evt
        if not evt.wait(timeout=_INPUT_WAIT_TIMEOUT):
            _session_ctx.wx_input_event = None
            return "(timeout: no input received)"
        text = _session_ctx.wx_input_value.strip()
        _session_ctx.wx_input_event = None
        _session_ctx.wx_input_value = ""
        return _resolve_choice(text, _value_map)

    # ── QQ ────────────────────────────────────────────────────────────────
    if _is_in_qq_turn(config) and _session_ctx.qq_send is not None:
        clean_prompt = _re.sub(r'\x1b\[[0-9;]*m', '', prompt).strip()
        payload = ""
        if menu_text:
            payload += _re.sub(r'\x1b\[[0-9;]*m', '', menu_text).strip() + "\n\n"
        payload += f"❓ 需要输入\n{clean_prompt}"
        if _menu_block:
            payload += "\n\n" + _menu_block
        # Echo to terminal for visibility
        print(f"\n  📩 QQ 权限请求: {clean_prompt}")
        if _menu_block:
            for line in _menu_block.splitlines():
                print(f"  {line}")
        sctx = _runtime.get_ctx(config)
        # Prefer thread-local target_id to avoid race conditions with concurrent handlers
        target = (getattr(_qq_thread_local, "target_id", "")
                  or getattr(sctx, "qq_current_target_id", "") or "")
        if not target:
            print(f"\n  ⚠ QQ 权限请求无法发送：target_id 为空")
            return "(error: no QQ target_id)"
        evt = _threading.Event()
        _session_ctx.qq_input_target_id = target
        _session_ctx.qq_input_event = evt
        _session_ctx.qq_send(target, payload)
        if not evt.wait(timeout=120):
            _session_ctx.qq_input_event = None
            _session_ctx.qq_input_target_id = ""
            return "(timeout: no input received)"
        text = _session_ctx.qq_input_value.strip()
        _session_ctx.qq_input_event = None
        _session_ctx.qq_input_value = ""
        _session_ctx.qq_input_target_id = ""
        return _resolve_choice(text, _value_map)

    # ── Web (chat API) ────────────────────────────────────────────────────
    if getattr(_session_ctx, 'in_web_turn', False):
        # Permission event is already pushed to WS by ChatSession._run_agent.
        # Just block here until the browser responds via /api/approve.
        evt = _threading.Event()
        _session_ctx.web_input_event = evt
        if not evt.wait(timeout=_INPUT_WAIT_TIMEOUT):
            _session_ctx.web_input_event = None
            return "(timeout: no input received)"
        text = _session_ctx.web_input_value.strip()
        _session_ctx.web_input_event = None
        _session_ctx.web_input_value = ""
        return text

    # ── Telegram ───────────────────────────────────────────────────────────
    if _is_in_tg_turn(config) and _session_ctx.tg_send is not None:
        token   = config.get("telegram_token")
        chat_id = config.get("telegram_chat_id")
        clean_prompt = _re.sub(r'\x1b\[[0-9;]*m', '', prompt).strip()
        payload = ""
        if menu_text:
            payload += _re.sub(r'\x1b\[[0-9;]*m', '', menu_text).strip() + "\n\n"
        payload += f"❓ *Input Required*\n{clean_prompt}"
        if _menu_block:
            # Embed the menu in the prompt body too — buttons render normally,
            # but the text serves as a fallback if the keyboard ever fails to
            # show (very old clients, narrow web preview) and lets users type
            # `1` / `y` / `approve` instead of clicking if they prefer.
            payload += "\n\n" + _menu_block

        if options:
            # Inline-keyboard path: render real Telegram buttons. callback_data
            # carries a short prompt id so a click on a stale prompt cannot
            # deliver to the current waiting agent.
            import uuid as _uuid
            from bridges.telegram import _tg_send_keyboard
            prompt_id = _uuid.uuid4().hex[:8]
            keyboard = [
                [{"text": str(label),
                  "callback_data": f"cc:{prompt_id}:{value}"[:64]}]
                for (label, value) in options
            ]
            evt = _threading.Event()
            # Set the wiring BEFORE sending so a fast click cannot race in
            # before tg_input_event / tg_callback_prompt_id are visible.
            _session_ctx.tg_input_event = evt
            _session_ctx.tg_callback_prompt_id = prompt_id
            msg_id = _tg_send_keyboard(token, chat_id, payload, keyboard)
            _session_ctx.tg_callback_message_id = msg_id or 0
        else:
            _session_ctx.tg_send(token, chat_id, payload)
            evt = _threading.Event()
            _session_ctx.tg_input_event = evt

        if not evt.wait(timeout=_INPUT_WAIT_TIMEOUT):
            _session_ctx.tg_input_event = None
            _session_ctx.tg_callback_prompt_id = ""
            _session_ctx.tg_callback_message_id = 0
            return "(timeout: no input received)"
        text = _session_ctx.tg_input_value.strip()
        _session_ctx.tg_input_event = None
        _session_ctx.tg_input_value = ""
        _session_ctx.tg_callback_prompt_id = ""
        _session_ctx.tg_callback_message_id = 0
        # Click on inline_keyboard delivers the canonical value already, so
        # _resolve_choice is a no-op there. If the user typed `1` / `approve`
        # instead of clicking, this is what normalizes their reply.
        return _resolve_choice(text, _value_map)

    # ── Terminal ────────────────────────────────────────────────────────────
    try:
        if _menu_block:
            # Print on a fresh line so the menu sits cleanly above the
            # input cursor; the original prompt text (which already shows
            # the canonical hint, e.g. "[y/N/a]") becomes the input prompt
            # below.
            print()
            print(_menu_block)
        rl_prompt = _re.sub(r'(\x1b\[[0-9;]*m)', r'\001\1\002', prompt)
        return _resolve_choice(input(rl_prompt), _value_map)
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


# ── SleepTimer ────────────────────────────────────────────────────────────

def _sleeptimer(seconds: int, config: dict) -> str:
    import runtime
    session_ctx = runtime.get_session_ctx(config.get("_session_id", "default"))
    cb = session_ctx.run_query
    if not cb:
        return "Error: No active REPL session (run_query not set for this session)"

    def worker():
        import time
        time.sleep(seconds)
        cb(
            "(System Automated Event): The timer has finished. "
            "Please wake up, perform any pending monitoring checks "
            "and report to the user now."
        )

    threading.Thread(target=worker, daemon=True).start()
    return (
        f"Timer successfully scheduled for {seconds} seconds. "
        "You can output your final thoughts and end your turn. "
        "You will be automatically awakened."
    )
