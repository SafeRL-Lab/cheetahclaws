"""Tests for the options=… menu UX added to ask_input_interactive.

Covers:
  - _format_menu_block / _build_value_map / _resolve_choice helpers
  - Per-bridge end-to-end: Slack, WeChat, terminal each render the
    menu and accept digit / canonical-value / label-word replies.

No real network calls; bridge send helpers and `builtins.input` are
mocked. Driven from a worker thread so the synchronous `evt.wait()`
inside ask_input_interactive resolves on the test's signal.
"""
from __future__ import annotations

import io
import threading
import time
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cheetahclaws import runtime
from cheetahclaws.tools import interaction as itx


# ── Helpers ───────────────────────────────────────────────────────────────


PERM_OPTIONS = [
    ("✅ Approve",       "y"),
    ("❌ Reject",        "n"),
    ("✅✅ Accept all",  "a"),
]


def _fresh_session_ctx(sid: str):
    """Reset and return a clean RuntimeContext for the given session id."""
    sctx = runtime.get_session_ctx(sid)
    # Wipe any state another test may have left behind on this id.
    for attr in (
        "slack_send", "wx_send", "tg_send",
        "slack_input_event", "wx_input_event", "tg_input_event",
        "slack_input_value", "wx_input_value", "tg_input_value",
        "tg_callback_prompt_id", "tg_callback_message_id",
        "in_slack_turn", "in_wechat_turn", "in_telegram_turn", "in_web_turn",
    ):
        if hasattr(sctx, attr):
            cur = getattr(sctx, attr)
            if isinstance(cur, str):
                setattr(sctx, attr, "")
            elif isinstance(cur, int):
                setattr(sctx, attr, 0)
            elif isinstance(cur, bool):
                setattr(sctx, attr, False)
            else:
                setattr(sctx, attr, None)
    return sctx


# ── Pure helpers ──────────────────────────────────────────────────────────


class TestFormatMenuBlock:
    def test_returns_empty_when_no_options(self):
        assert itx._format_menu_block(None) == ""
        assert itx._format_menu_block([]) == ""

    def test_each_option_becomes_one_numbered_row(self):
        out = itx._format_menu_block(PERM_OPTIONS)
        lines = out.splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("  [1]")
        assert "✅ Approve" in lines[0]
        assert "reply `1` or `y`" in lines[0]
        assert lines[1].startswith("  [2]")
        assert "❌ Reject" in lines[1]
        assert "reply `2` or `n`" in lines[1]
        assert lines[2].startswith("  [3]")
        assert "✅✅ Accept all" in lines[2]
        assert "reply `3` or `a`" in lines[2]


class TestBuildValueMap:
    def test_empty_options_yields_empty_map(self):
        assert itx._build_value_map(None) == {}
        assert itx._build_value_map([]) == {}

    def test_digit_value_label_word_aliases_for_perm_options(self):
        m = itx._build_value_map(PERM_OPTIONS)
        # Digits
        assert m["1"] == "y"
        assert m["2"] == "n"
        assert m["3"] == "a"
        # Canonical values
        assert m["y"] == "y"
        assert m["n"] == "n"
        assert m["a"] == "a"
        # Label words (emojis stripped, lowercased)
        assert m["approve"] == "y"
        assert m["reject"]  == "n"
        assert m["accept"]  == "a"
        assert m["all"]     == "a"

    def test_first_write_wins_on_collision(self):
        # Two options whose labels would both reduce to "go" — first stays.
        opts = [("Go forward", "f"), ("Go back", "b")]
        m = itx._build_value_map(opts)
        assert m["go"] == "f"
        # Each option still gets its digit + value alias.
        assert m["1"] == "f" and m["f"] == "f"
        assert m["2"] == "b" and m["b"] == "b"


class TestResolveChoice:
    def test_empty_map_passes_through_unchanged(self):
        assert itx._resolve_choice("anything", {}) == "anything"
        assert itx._resolve_choice("", {}) == ""

    def test_digit_resolves_to_value(self):
        m = itx._build_value_map(PERM_OPTIONS)
        assert itx._resolve_choice("1", m) == "y"
        assert itx._resolve_choice("3", m) == "a"

    def test_canonical_value_passes_through(self):
        m = itx._build_value_map(PERM_OPTIONS)
        assert itx._resolve_choice("y", m) == "y"
        assert itx._resolve_choice("Y", m) == "y"  # case-insensitive

    def test_label_word_resolves(self):
        m = itx._build_value_map(PERM_OPTIONS)
        assert itx._resolve_choice("Approve", m) == "y"
        assert itx._resolve_choice("reject", m) == "n"
        assert itx._resolve_choice("ALL",    m) == "a"

    def test_unknown_input_returned_verbatim(self):
        # Pass-through preserves backwards-compat for callers that ask
        # questions with options but want to accept free-text too.
        m = itx._build_value_map(PERM_OPTIONS)
        assert itx._resolve_choice("custom answer", m) == "custom answer"

    def test_whitespace_trimmed_before_lookup(self):
        m = itx._build_value_map(PERM_OPTIONS)
        assert itx._resolve_choice("  1  ", m) == "y"
        assert itx._resolve_choice("\tApprove\n", m) == "y"

    def test_non_string_input_returned_unchanged(self):
        # Defensive: callers occasionally pass through tool results.
        m = itx._build_value_map(PERM_OPTIONS)
        assert itx._resolve_choice(None, m) is None
        assert itx._resolve_choice(42, m) == 42


# ── Per-bridge end-to-end (worker-thread driven) ─────────────────────────


def _drive_bridge(setup_ctx, deliver_reply, options=PERM_OPTIONS,
                  raw_reply="1", expected="y", timeout=2.0):
    """Generic runner: drives ask_input_interactive on a worker thread,
    waits until the bridge has registered its input event, delivers the
    raw reply via the bridge-specific path, and returns (worker_return,
    captured_payload). `setup_ctx(sctx, config)` flips the bridge flag
    and registers `*_send` so the function routes to that bridge;
    `deliver_reply(sctx, raw)` sets the `*_input_value` and fires the
    event."""
    sid = f"test-{id(setup_ctx)}"
    sctx = _fresh_session_ctx(sid)
    captured = []

    def capture_send(*args):
        captured.append(args)

    config = {"_session_id": sid}
    setup_ctx(sctx, config, capture_send)
    holder = {}

    def worker():
        holder["v"] = itx.ask_input_interactive("Allow: rm -rf /tmp",
                                                config, options=options)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    # Wait until the bridge registered its event before delivering.
    deadline = time.monotonic() + timeout
    evt_attr = setup_ctx.evt_attr  # type: ignore[attr-defined]
    while time.monotonic() < deadline:
        if getattr(sctx, evt_attr, None) is not None:
            break
        time.sleep(0.005)
    assert getattr(sctx, evt_attr, None) is not None, \
        f"worker never registered {evt_attr}"
    deliver_reply(sctx, raw_reply)
    t.join(timeout=timeout)
    assert not t.is_alive(), "worker did not unblock after reply"

    assert holder["v"] == expected, (
        f"resolved value mismatch: got {holder['v']!r}, expected {expected!r}"
    )
    return holder["v"], captured


# Slack ─────────────────────────────────────────────────

def _setup_slack(sctx, config, capture):
    sctx.in_slack_turn = True
    sctx.slack_send = capture
    config["slack_channel"] = "C1"
_setup_slack.evt_attr = "slack_input_event"  # type: ignore[attr-defined]


def _deliver_slack(sctx, raw):
    sctx.slack_input_value = raw
    sctx.slack_input_event.set()


class TestSlackOptions:
    def test_digit_reply_resolves_to_value(self):
        v, sends = _drive_bridge(_setup_slack, _deliver_slack,
                                 raw_reply="1", expected="y")
        assert len(sends) == 1
        channel, payload = sends[0]
        assert channel == "C1"
        # Menu rows present in the message body.
        assert "[1]" in payload and "✅ Approve" in payload
        assert "[2]" in payload and "❌ Reject" in payload
        assert "[3]" in payload and "✅✅ Accept all" in payload

    def test_label_word_reply_resolves(self):
        _drive_bridge(_setup_slack, _deliver_slack,
                      raw_reply="Approve", expected="y")

    def test_canonical_value_reply_passes_through(self):
        _drive_bridge(_setup_slack, _deliver_slack,
                      raw_reply="a", expected="a")

    def test_unknown_reply_returned_verbatim(self):
        # Pass-through means a free-text fallback still works for callers
        # that combine options with free input.
        _drive_bridge(_setup_slack, _deliver_slack,
                      raw_reply="something else", expected="something else")


# WeChat ────────────────────────────────────────────────

def _setup_wechat(sctx, config, capture):
    sctx.in_wechat_turn = True
    # wx_send takes (user_id, payload); the first positional is the user id,
    # so capture stores both for assertion.
    sctx.wx_send = capture
    sctx.wx_current_user_id = "U1"
_setup_wechat.evt_attr = "wx_input_event"  # type: ignore[attr-defined]


def _deliver_wechat(sctx, raw):
    sctx.wx_input_value = raw
    sctx.wx_input_event.set()


class TestWeChatOptions:
    def test_digit_reply_resolves_to_value(self):
        v, sends = _drive_bridge(_setup_wechat, _deliver_wechat,
                                 raw_reply="2", expected="n")
        assert len(sends) == 1
        user_id, payload = sends[0]
        assert user_id == "U1"
        assert "需要输入" in payload  # WeChat header is Chinese
        assert "[2]" in payload and "❌ Reject" in payload

    def test_label_word_reply_resolves(self):
        _drive_bridge(_setup_wechat, _deliver_wechat,
                      raw_reply="all", expected="a")


# Terminal ──────────────────────────────────────────────


class TestTerminalOptions:
    def test_input_resolves_digit_and_menu_printed(self):
        sid = "test-terminal"
        sctx = _fresh_session_ctx(sid)
        # No bridge active → falls through to terminal.
        config = {"_session_id": sid}

        buf = io.StringIO()
        with patch("builtins.input", return_value="3"), redirect_stdout(buf):
            out = itx.ask_input_interactive("Allow: rm -rf /tmp",
                                            config, options=PERM_OPTIONS)
        assert out == "a"
        printed = buf.getvalue()
        # Menu printed before the input cursor.
        assert "[1]" in printed and "✅ Approve" in printed
        assert "[3]" in printed and "✅✅ Accept all" in printed

    def test_label_word_resolves(self):
        sid = "test-terminal-label"
        sctx = _fresh_session_ctx(sid)
        config = {"_session_id": sid}
        with patch("builtins.input", return_value="approve"):
            out = itx.ask_input_interactive("Allow: rm -rf",
                                            config, options=PERM_OPTIONS)
        assert out == "y"

    def test_canonical_value_passes_through(self):
        sid = "test-terminal-y"
        sctx = _fresh_session_ctx(sid)
        config = {"_session_id": sid}
        with patch("builtins.input", return_value="y"):
            out = itx.ask_input_interactive("Allow: rm -rf",
                                            config, options=PERM_OPTIONS)
        assert out == "y"

    def test_no_options_keeps_existing_behavior(self):
        # Backwards compat: with options=None the menu must NOT print
        # and input is returned verbatim.
        sid = "test-terminal-noopts"
        sctx = _fresh_session_ctx(sid)
        config = {"_session_id": sid}
        buf = io.StringIO()
        with patch("builtins.input", return_value="hello"), redirect_stdout(buf):
            out = itx.ask_input_interactive("name? ", config)
        assert out == "hello"
        # No menu artefacts in stdout.
        assert "[1]" not in buf.getvalue()
        assert "reply `" not in buf.getvalue()
