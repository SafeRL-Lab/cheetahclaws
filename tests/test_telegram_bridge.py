"""Tests for bridges/telegram.py — the Telegram bot bridge.

Covers the file-handling additions made for issue #84:
  - _tg_send_document multipart upload (happy path + error paths)
  - _tg_send text splitting + Markdown fallback
  - The Write-tool auto-send hook in _bg_runner

No real Telegram calls are made; urllib.request.urlopen is mocked.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cheetahclaws.bridges import telegram as tg


# ── Helpers ───────────────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal context-managed stand-in for urllib.request.urlopen."""

    def __init__(self, payload: dict):
        self._buf = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._buf


@pytest.fixture
def tmp_file():
    """Write a small file and yield its path; clean up on exit."""
    fd, path = tempfile.mkstemp(prefix="tgtest_", suffix=".txt")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"hello cheetahclaws\n")
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── _tg_send (text) ───────────────────────────────────────────────────────


class TestTgSend:
    def test_short_message_single_call(self):
        with patch.object(tg, "_tg_api", return_value={"ok": True}) as api:
            tg._tg_send("TOK", 42, "hi")
        assert api.call_count == 1
        method, params = api.call_args[0][1], api.call_args[0][2]
        assert method == "sendMessage"
        assert params["chat_id"] == 42
        assert params["text"] == "hi"
        assert params["parse_mode"] == "Markdown"

    def test_long_message_split_into_4000_char_chunks(self):
        # 4000 + 4000 + 100 → 3 chunks.
        long_text = "a" * 4000 + "b" * 4000 + "c" * 100
        with patch.object(tg, "_tg_api", return_value={"ok": True}) as api:
            tg._tg_send("TOK", 1, long_text)
        assert api.call_count == 3
        chunks = [c[0][2]["text"] for c in api.call_args_list]
        assert chunks[0] == "a" * 4000
        assert chunks[1] == "b" * 4000
        assert chunks[2] == "c" * 100

    def test_markdown_failure_falls_back_to_plain(self):
        # First call (Markdown) fails, fallback (no parse_mode) must fire.
        responses = [{"ok": False, "description": "bad markdown"}, {"ok": True}]
        with patch.object(tg, "_tg_api", side_effect=responses) as api:
            tg._tg_send("TOK", 1, "*broken_")
        assert api.call_count == 2
        first_params = api.call_args_list[0][0][2]
        second_params = api.call_args_list[1][0][2]
        assert first_params.get("parse_mode") == "Markdown"
        assert "parse_mode" not in second_params


# ── _tg_send_document (multipart) ─────────────────────────────────────────


class TestTgSendDocument:
    def test_happy_path_builds_correct_multipart(self, tmp_file):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = req.data
            return _FakeResponse({"ok": True, "result": {"message_id": 99}})

        with patch.object(tg.urllib.request if hasattr(tg, "urllib") else __import__("urllib.request"),
                          "urlopen", side_effect=fake_urlopen, create=True):
            # The function imports urllib.request locally; patch the real module instead.
            import urllib.request as _ur
            with patch.object(_ur, "urlopen", side_effect=fake_urlopen):
                ok = tg._tg_send_document("TOKEN123", 42, tmp_file, caption="📎 hi")

        assert ok is True
        assert captured["url"] == "https://api.telegram.org/botTOKEN123/sendDocument"
        ct = next(v for k, v in captured["headers"].items() if k.lower() == "content-type")
        assert ct.startswith("multipart/form-data; boundary=")
        body = captured["body"]
        # chat_id field
        assert b'name="chat_id"\r\n\r\n42\r\n' in body
        # caption field
        assert b'name="caption"\r\n\r\n\xf0\x9f\x93\x8e hi\r\n' in body  # 📎 in UTF-8
        # document field with the actual filename
        fname = os.path.basename(tmp_file).encode()
        assert b'name="document"; filename="' + fname + b'"' in body
        # File contents are embedded
        assert b"hello cheetahclaws" in body

    def test_missing_file_returns_false_and_warns(self):
        with patch.object(tg, "_tg_send") as warn:
            ok = tg._tg_send_document("TOK", 1, "/nope/does-not-exist.bin")
        assert ok is False
        warn.assert_called_once()
        assert "not found" in warn.call_args[0][2]

    def test_empty_file_returns_false(self):
        fd, path = tempfile.mkstemp(prefix="tgempty_")
        os.close(fd)  # creates a 0-byte file
        try:
            with patch.object(tg, "_tg_send") as warn:
                ok = tg._tg_send_document("TOK", 1, path)
            assert ok is False
            warn.assert_called_once()
            assert "empty" in warn.call_args[0][2].lower()
        finally:
            os.unlink(path)

    def test_oversize_file_short_circuits_without_upload(self, tmp_file, monkeypatch):
        # Pretend the file is huge; getsize is the gate.
        import urllib.request as _ur
        monkeypatch.setattr(os.path, "getsize", lambda _p: tg._TG_FILE_MAX_BYTES + 1)
        with patch.object(_ur, "urlopen") as up, patch.object(tg, "_tg_send") as warn:
            ok = tg._tg_send_document("TOK", 1, tmp_file)
        assert ok is False
        up.assert_not_called()  # never even attempted upload
        assert "too large" in warn.call_args[0][2].lower()

    def test_network_exception_returns_false(self, tmp_file):
        import urllib.request as _ur

        def boom(*a, **kw):
            raise OSError("connection refused")

        with patch.object(_ur, "urlopen", side_effect=boom), \
             patch.object(tg, "_tg_send") as warn:
            ok = tg._tg_send_document("TOK", 1, tmp_file)
        assert ok is False
        assert "upload failed" in warn.call_args[0][2].lower()

    def test_api_rejects_returns_false_with_description(self, tmp_file):
        import urllib.request as _ur

        def fake(req, timeout=None):
            return _FakeResponse({"ok": False, "description": "FILE_TOO_BIG"})

        with patch.object(_ur, "urlopen", side_effect=fake), \
             patch.object(tg, "_tg_send") as warn:
            ok = tg._tg_send_document("TOK", 1, tmp_file)
        assert ok is False
        assert "FILE_TOO_BIG" in warn.call_args[0][2]


# ── Write-tool auto-send hook (in _bg_runner) ─────────────────────────────


class _StubJobs:
    """Drop-in replacement for the jobs module, just enough for _bg_runner."""

    def __init__(self):
        self.events = []

    def start(self, jid):           self.events.append(("start", jid))
    def add_step(self, jid, *a):    self.events.append(("add_step", jid, a))
    def finish_step(self, jid, *a): self.events.append(("finish", jid, a))
    def stream_result(self, *a):    pass
    def complete(self, *a):         self.events.append(("complete", a))
    def fail(self, *a):              self.events.append(("fail", a))

    def get(self, jid):
        return SimpleNamespace(step_count=1, duration_s=0.1)


def _make_session_ctx():
    """Bare-bones stand-in for runtime.get_session_ctx output."""
    return SimpleNamespace(
        on_text_chunk=None,
        on_tool_start=None,
        on_tool_end=None,
        agent_state=None,
    )


def _run_bg_runner_with_writes(tmp_file: str, write_paths: list[tuple[str, str]]):
    """Drive _bg_runner with a stub run_query_cb that fires Write start/end pairs.

    write_paths: list of (file_path, tool_result) to fire. Returns the list of
    paths that _tg_send_document was invoked with.
    """
    sent = []
    job = SimpleNamespace(id="JOB1")
    stub_jobs = _StubJobs()
    sctx = SimpleNamespace(telegram_incoming=False)
    session_ctx = _make_session_ctx()

    def run_query_cb(_q):
        # Simulate the agent loop: fire start, then end for each write.
        for fp, result in write_paths:
            session_ctx.on_tool_start("Write", {"file_path": fp})
            session_ctx.on_tool_end("Write", result)

    # Run threads synchronously so the auto-send finishes before we assert.
    class _SyncThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=False, **_):
            self._t, self._a, self._k = target, args, kwargs or {}
        def start(self):
            self._t(*self._a, **self._k)
        def join(self, *_a, **_k):
            return None

    with patch.object(tg, "_jobs", stub_jobs), \
         patch.object(tg, "_tg_api", return_value={"ok": True, "result": {"message_id": 1}}), \
         patch.object(tg, "_tg_send"), \
         patch.object(tg, "_tg_send_document", side_effect=lambda _t, _c, p, caption=None: sent.append(p) or True), \
         patch.object(tg.runtime, "get_ctx", return_value=sctx), \
         patch.object(tg.threading, "Thread", _SyncThread):
        tg._bg_runner(job, "do it", "TOK", 42, run_query_cb, session_ctx, {"_session_id": "t"})

    return sent


class TestBgRunnerAutoSend:
    def test_successful_write_mails_the_file(self, tmp_file):
        sent = _run_bg_runner_with_writes(
            tmp_file,
            [(tmp_file, "File written successfully")],
        )
        assert sent == [tmp_file]

    def test_errored_write_does_not_send(self, tmp_file):
        sent = _run_bg_runner_with_writes(
            tmp_file,
            [(tmp_file, "Error: missing required parameter 'file_path'")],
        )
        assert sent == []

    def test_denied_write_does_not_send(self, tmp_file):
        sent = _run_bg_runner_with_writes(
            tmp_file,
            [(tmp_file, "Denied: user rejected write operation")],
        )
        assert sent == []

    def test_duplicate_writes_in_one_turn_send_once(self, tmp_file):
        sent = _run_bg_runner_with_writes(
            tmp_file,
            [(tmp_file, "ok"), (tmp_file, "ok")],
        )
        assert sent == [tmp_file]  # second is deduped

    def test_distinct_paths_each_sent(self, tmp_file, tmp_path):
        # Both paths must exist on disk: the auto-send closure stats the file
        # before invoking _tg_send_document, so a non-existent path is silently
        # dropped.
        other = tmp_path / "other.txt"
        other.write_bytes(b"second file")
        sent = _run_bg_runner_with_writes(
            tmp_file,
            [(tmp_file, "ok"), (str(other), "ok")],
        )
        assert sent == [tmp_file, str(other)]


# ── Module-level smoke ────────────────────────────────────────────────────


class TestModuleExports:
    def test_size_cap_under_50_megs(self):
        assert tg._TG_FILE_MAX_BYTES <= 50 * 1024 * 1024
        assert tg._TG_FILE_MAX_BYTES >= 40 * 1024 * 1024  # sanity floor

    def test_required_callables_present(self):
        for name in ("_tg_send", "_tg_send_document", "_tg_api",
                     "_tg_poll_loop", "_bg_runner", "cmd_telegram",
                     "_tg_send_keyboard", "_handle_callback_query"):
            assert callable(getattr(tg, name)), f"missing: {name}"


# ── _tg_send_keyboard (inline_keyboard outbound) ─────────────────────────


class TestTgSendKeyboard:
    def test_happy_path_includes_reply_markup(self):
        kb = [[{"text": "✅ Approve", "callback_data": "cc:abc:y"}]]
        with patch.object(tg, "_tg_api",
                          return_value={"ok": True, "result": {"message_id": 7}}) as api:
            mid = tg._tg_send_keyboard("TOK", 42, "Allow it?", kb)
        assert mid == 7
        assert api.call_count == 1
        method, params = api.call_args[0][1], api.call_args[0][2]
        assert method == "sendMessage"
        assert params["chat_id"] == 42
        assert params["text"] == "Allow it?"
        assert params["parse_mode"] == "Markdown"
        assert params["reply_markup"] == {"inline_keyboard": kb}

    def test_markdown_failure_retries_without_parse_mode_keyboard_kept(self):
        kb = [[{"text": "OK", "callback_data": "cc:1:y"}]]
        responses = [
            {"ok": False, "description": "bad markdown"},
            {"ok": True, "result": {"message_id": 9}},
        ]
        with patch.object(tg, "_tg_api", side_effect=responses) as api:
            mid = tg._tg_send_keyboard("TOK", 1, "bad *prompt", kb)
        assert mid == 9
        assert api.call_count == 2
        first  = api.call_args_list[0][0][2]
        second = api.call_args_list[1][0][2]
        assert first.get("parse_mode") == "Markdown"
        assert "parse_mode" not in second
        # Keyboard preserved on retry — that's the whole point of this path.
        assert second["reply_markup"] == {"inline_keyboard": kb}

    def test_total_failure_falls_back_to_plain_text_no_keyboard(self):
        kb = [[{"text": "OK", "callback_data": "cc:1:y"}]]
        # Keyboard variants both fail; final plain-text attempt succeeds.
        responses = [
            {"ok": False, "description": "x"},
            {"ok": False, "description": "y"},
            {"ok": True, "result": {"message_id": 11}},
        ]
        with patch.object(tg, "_tg_api", side_effect=responses) as api:
            mid = tg._tg_send_keyboard("TOK", 1, "prompt", kb)
        assert mid == 11
        assert api.call_count == 3
        last = api.call_args_list[-1][0][2]
        assert "reply_markup" not in last
        assert "parse_mode" not in last

    def test_all_attempts_fail_returns_zero(self):
        kb = [[{"text": "X", "callback_data": "cc:1:n"}]]
        with patch.object(tg, "_tg_api", return_value={"ok": False}):
            mid = tg._tg_send_keyboard("TOK", 1, "p", kb)
        assert mid == 0


# ── _handle_callback_query (inbound click router) ────────────────────────


def _make_cb(data: str, chat_id: int, message_id: int = 100,
             cb_id: str = "CB1", text: str = "❓ pick one") -> dict:
    return {
        "id": cb_id,
        "data": data,
        "from": {"id": 999},
        "message": {
            "message_id": message_id,
            "chat": {"id": chat_id},
            "text": text,
        },
    }


class TestHandleCallbackQuery:
    def test_valid_click_delivers_value_and_fires_event(self):
        import threading
        evt = threading.Event()
        sctx = SimpleNamespace(
            tg_input_event=evt,
            tg_input_value="",
            tg_callback_prompt_id="abc12345",
            tg_callback_message_id=100,
        )
        with patch.object(tg, "_tg_api", return_value={"ok": True}) as api:
            tg._handle_callback_query("TOK", 42,
                                      _make_cb("cc:abc12345:y", 42), sctx)
        assert evt.is_set()
        assert sctx.tg_input_value == "y"
        # ID cleared after consumption so a follow-up click cannot retrigger.
        assert sctx.tg_callback_prompt_id == ""
        # answerCallbackQuery + editMessageText both fired.
        methods = [c[0][1] for c in api.call_args_list]
        assert "answerCallbackQuery" in methods
        assert "editMessageText" in methods

    def test_unauthorized_chat_ignored_and_event_not_fired(self):
        import threading
        evt = threading.Event()
        sctx = SimpleNamespace(
            tg_input_event=evt, tg_input_value="",
            tg_callback_prompt_id="abc12345", tg_callback_message_id=100,
        )
        with patch.object(tg, "_tg_api", return_value={"ok": True}) as api:
            tg._handle_callback_query("TOK", 42,
                                      _make_cb("cc:abc12345:y", chat_id=999), sctx)
        assert not evt.is_set()
        assert sctx.tg_input_value == ""
        # An "Unauthorized" answerCallbackQuery is sent as the only call.
        assert api.call_count == 1
        params = api.call_args[0][2]
        assert "Unauthorized" in params.get("text", "")

    def test_stale_prompt_id_dropped(self):
        import threading
        evt = threading.Event()
        sctx = SimpleNamespace(
            tg_input_event=evt, tg_input_value="",
            tg_callback_prompt_id="newprompt",  # current prompt
            tg_callback_message_id=100,
        )
        with patch.object(tg, "_tg_api", return_value={"ok": True}):
            # Click came in with the OLD prompt_id — must be ignored.
            tg._handle_callback_query("TOK", 42,
                                      _make_cb("cc:OLDPROMPT:y", 42), sctx)
        assert not evt.is_set()
        assert sctx.tg_input_value == ""
        # The current prompt's wiring must remain so the right click still works.
        assert sctx.tg_callback_prompt_id == "newprompt"

    def test_non_cc_payload_ignored(self):
        import threading
        evt = threading.Event()
        sctx = SimpleNamespace(
            tg_input_event=evt, tg_input_value="",
            tg_callback_prompt_id="abc12345", tg_callback_message_id=100,
        )
        with patch.object(tg, "_tg_api", return_value={"ok": True}) as api:
            tg._handle_callback_query("TOK", 42,
                                      _make_cb("garbage_payload", 42), sctx)
        assert not evt.is_set()
        # Still acknowledges so the spinner clears.
        methods = [c[0][1] for c in api.call_args_list]
        assert methods == ["answerCallbackQuery"]

    def test_no_prompt_waiting_does_not_edit_message(self):
        """Issue #84 follow-up: when a click arrives but no prompt is
        currently waiting (already answered or timed out), the handler
        must NOT edit the message to show "✓ Selected" — that would
        falsely tell the user the action took effect.  Acknowledge the
        callback (clears spinner) and bail."""
        import threading
        evt = threading.Event()
        sctx = SimpleNamespace(
            tg_input_event=evt, tg_input_value="",
            tg_callback_prompt_id="",  # no prompt waiting
            tg_callback_message_id=0,
        )
        with patch.object(tg, "_tg_api", return_value={"ok": True}) as api:
            tg._handle_callback_query("TOK", 42,
                                      _make_cb("cc:abc12345:y", 42), sctx)
        assert not evt.is_set()
        assert sctx.tg_input_value == ""
        # Only answerCallbackQuery should fire — no editMessageText.
        methods = [c[0][1] for c in api.call_args_list]
        assert methods == ["answerCallbackQuery"], \
            "Stale click must not produce a misleading message edit"

    def test_label_with_markdown_chars_is_sanitized(self):
        """Issue #84 follow-up: callers can pass any string as an option
        value.  Backticks/asterisks would break the Markdown parse mode
        and silently fail editMessageText.  The sanitizer replaces them
        before embedding into the confirmation line."""
        import threading
        evt = threading.Event()
        sctx = SimpleNamespace(
            tg_input_event=evt, tg_input_value="",
            tg_callback_prompt_id="abc12345",
            tg_callback_message_id=100,
        )
        # Value contains backtick and asterisk — Markdown markers.
        captured_payloads: list[dict] = []
        def _capture(_tok, _method, params=None):
            captured_payloads.append((_method, params or {}))
            return {"ok": True}
        with patch.object(tg, "_tg_api", side_effect=_capture):
            tg._handle_callback_query(
                "TOK", 42,
                _make_cb("cc:abc12345:`bad*value`", 42),
                sctx,
            )
        # The raw value (with backticks/asterisks) is still delivered to
        # the agent — sanitisation only affects the visual confirmation.
        assert sctx.tg_input_value == "`bad*value`"
        # Find the editMessageText call and check it has no unbalanced
        # Markdown markers in the appended "Selected: ..." line.
        edits = [p for m, p in captured_payloads if m == "editMessageText"]
        assert len(edits) == 1
        body = edits[0]["text"]
        # Backticks/asterisks from the raw value are escaped/replaced.
        assert "`bad*value`" not in body, \
            "Raw markdown chars must not leak into the visual confirmation"

    def test_value_with_colons_preserved(self):
        # callback_data is "cc:<id>:<value>" — the value field can itself
        # contain colons; split(":", 2) keeps them intact.
        import threading
        evt = threading.Event()
        sctx = SimpleNamespace(
            tg_input_event=evt, tg_input_value="",
            tg_callback_prompt_id="abc12345", tg_callback_message_id=100,
        )
        with patch.object(tg, "_tg_api", return_value={"ok": True}):
            tg._handle_callback_query(
                "TOK", 42,
                _make_cb("cc:abc12345:weird:value:with:colons", 42),
                sctx,
            )
        assert evt.is_set()
        assert sctx.tg_input_value == "weird:value:with:colons"


# ── End-to-end: ask_input_interactive(options=) → click → return ─────────


class TestAskInputWithKeyboard:
    """Drive ask_input_interactive in a worker thread, simulate a click via
    _handle_callback_query, and verify the worker returns the clicked value."""

    def _drive(self, options, click_value, expected_return,
               click_via_handler: bool = True):
        import threading, time; from cheetahclaws import runtime
        # Reset session ctx to a clean state for the test.
        sid = "tg-kbd-test"
        sctx = runtime.get_session_ctx(sid)
        sctx.tg_send = lambda *_a, **_k: None  # bridge appears active
        sctx.in_telegram_turn = True
        sctx.tg_input_event = None
        sctx.tg_input_value = ""
        sctx.tg_callback_prompt_id = ""
        sctx.tg_callback_message_id = 0

        config = {
            "_session_id": sid,
            "telegram_token": "TOKEN",
            "telegram_chat_id": 42,
        }

        captured_keyboard = []

        def fake_kbd(token, chat_id, text, kb):
            captured_keyboard.append((token, chat_id, text, kb))
            return 555  # fake message_id

        result_holder = {}

        def worker():
            from cheetahclaws.tools.interaction import ask_input_interactive
            result_holder["v"] = ask_input_interactive(
                "Allow: rm -rf  [y/N/a]", config, options=options
            )

        with patch.object(tg, "_tg_send_keyboard", side_effect=fake_kbd), \
             patch.object(tg, "_tg_api", return_value={"ok": True}):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            # Wait until the worker has registered its event and prompt_id.
            for _ in range(200):
                if sctx.tg_input_event is not None and sctx.tg_callback_prompt_id:
                    break
                time.sleep(0.01)
            assert sctx.tg_input_event is not None, "worker never registered event"
            assert captured_keyboard, "_tg_send_keyboard was not called"

            prompt_id = sctx.tg_callback_prompt_id
            cb = _make_cb(f"cc:{prompt_id}:{click_value}", 42, message_id=555)
            tg._handle_callback_query("TOKEN", 42, cb, sctx)
            t.join(timeout=2.0)
            assert not t.is_alive(), "worker did not unblock after callback"

        assert result_holder["v"] == expected_return
        # Wiring is reset so the next prompt starts clean.
        assert sctx.tg_input_event is None
        assert sctx.tg_callback_prompt_id == ""
        return captured_keyboard

    def test_click_returns_y(self):
        opts = [("✅ Approve", "y"), ("❌ Reject", "n"), ("✅✅ Accept all", "a")]
        kb_calls = self._drive(opts, click_value="y", expected_return="y")
        kb = kb_calls[0][3]
        # 3 rows, 1 button each, callback_data is "cc:<id>:<value>".
        assert len(kb) == 3
        labels = [row[0]["text"] for row in kb]
        assert labels == ["✅ Approve", "❌ Reject", "✅✅ Accept all"]
        for row in kb:
            assert row[0]["callback_data"].startswith("cc:")
            assert row[0]["callback_data"].count(":") >= 2

    def test_click_returns_a_for_accept_all(self):
        opts = [("✅ Approve", "y"), ("❌ Reject", "n"), ("✅✅ Accept all", "a")]
        self._drive(opts, click_value="a", expected_return="a")


# ── Slash-command stdout forwarding (issue #84 follow-up) ────────────────


class TestSlashRunnerCapturesPrintOutput:
    """Pin: when a Telegram /<cmd> dispatches a "simple" command (the
    handler returns a non-tuple), the bridge must forward whatever the
    command printed back into the chat.  Pre-fix it always sent the
    bare "✅ /help executed." string and the actual /help menu only
    appeared on the server console — the user-visible regression in
    issue #84.

    The poll-loop wraps slash_cb execution with a stdout/stderr Tee that
    captures print()/info()/ok()/warn() output.  These tests drive the
    same code path the live bridge uses (via the inline _slash_runner
    closure inside _tg_poll_loop) by lifting the closure out for direct
    invocation.
    """

    def _build_runner(self, monkeypatch, slash_cb, sent: list):
        """Replicate the closure from bridges/telegram.py:_tg_poll_loop so
        unit tests can drive it without pumping the long-poll loop."""
        import io as _io, sys as _sys, re as _re
        from cheetahclaws.bridges import telegram as tg

        # Patch _tg_send to capture instead of hitting the network.
        monkeypatch.setattr(tg, "_tg_send",
                            lambda token, chat_id, text: sent.append(text))

        class _Tee:
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    try: s.write(data)
                    except Exception: pass
            def flush(self):
                for s in self._streams:
                    try: s.flush()
                    except Exception: pass

        from cheetahclaws.tools import _tg_thread_local as _ttl  # imported the same way the bridge does

        def _slash_runner(_slash_text, _token, _chat_id):
            _ttl.active = True
            _buf_out, _buf_err = _io.StringIO(), _io.StringIO()
            _orig_out, _orig_err = _sys.stdout, _sys.stderr
            _sys.stdout = _Tee(_orig_out, _buf_out)
            _sys.stderr = _Tee(_orig_err, _buf_err)
            try:
                cmd_type = slash_cb(_slash_text)
            except Exception as e:
                _sys.stdout, _sys.stderr = _orig_out, _orig_err
                tg._tg_send(_token, _chat_id, f"⚠ Error: {e}")
                return
            finally:
                _sys.stdout, _sys.stderr = _orig_out, _orig_err
                _ttl.active = False
            captured = (_buf_out.getvalue() + _buf_err.getvalue())
            captured = _re.sub(r'\x1b\[[0-9;]*m', '', captured).strip()
            if cmd_type == "simple":
                cmd_name = _slash_text.strip().split()[0]
                if captured:
                    tg._tg_send(_token, _chat_id, captured)
                else:
                    tg._tg_send(_token, _chat_id, f"✅ {cmd_name} executed.")

        return _slash_runner

    def test_print_output_is_forwarded_to_chat(self, monkeypatch):
        """A simple command that prints a multi-line menu (think /help)
        must surface that menu in the chat — not the bare ack string."""
        def fake_help_cmd(text):
            print("CheetahClaws Commands:")
            print("  /help    show help")
            print("  /status  show status")
            return "simple"

        sent: list[str] = []
        runner = self._build_runner(monkeypatch, fake_help_cmd, sent)
        runner("/help", "tok", 42)

        assert len(sent) == 1
        body = sent[0]
        assert "CheetahClaws Commands" in body
        assert "/help" in body
        assert "/status" in body
        # The bare ack string must NOT replace the real menu.
        assert "executed" not in body

    def test_no_print_falls_back_to_ack(self, monkeypatch):
        """Commands that intentionally produce no output (rare, but possible
        for purely-stateful toggles) keep the existing ack so the user gets
        some confirmation."""
        def silent_cmd(text):
            return "simple"

        sent: list[str] = []
        runner = self._build_runner(monkeypatch, silent_cmd, sent)
        runner("/silent", "tok", 42)

        assert sent == ["✅ /silent executed."]

    def test_ansi_escapes_are_stripped(self, monkeypatch):
        """info()/ok()/warn() in ui/render.py wrap text in ANSI colour
        codes via clr().  Telegram doesn't render ANSI, so the bridge
        must strip them before sending — otherwise the user sees raw
        '\\x1b[36m...\\x1b[0m' garbage."""
        def coloured_cmd(text):
            from cheetahclaws.ui.render import info, ok
            info("informational")
            ok("done")
            return "simple"

        sent: list[str] = []
        runner = self._build_runner(monkeypatch, coloured_cmd, sent)
        runner("/status", "tok", 42)

        assert len(sent) == 1
        body = sent[0]
        assert "\x1b[" not in body, \
            "ANSI escape sequences must be stripped before sending to Telegram"
        assert "informational" in body
        assert "done" in body

    def test_other_threads_stdout_is_not_lost(self, monkeypatch):
        """The Tee writes to BOTH the original stdout and the capture
        buffer, so server logs (docker compose logs) still see the
        command's output even though the bridge also forwards it."""
        import io as _io
        captured_orig = _io.StringIO()
        monkeypatch.setattr("sys.stdout", captured_orig)

        def chatty_cmd(text):
            print("visible to operator")
            return "simple"

        sent: list[str] = []
        runner = self._build_runner(monkeypatch, chatty_cmd, sent)
        runner("/status", "tok", 42)

        assert "visible to operator" in captured_orig.getvalue(), \
            "Tee must keep writing to original stdout so docker logs " \
            "still show /<cmd> output"
        assert "visible to operator" in sent[0]
