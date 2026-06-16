"""Tests for cheetahclaws._make_bridge_slash_handler.

Issue #84 follow-up: in headless deployments (Docker, --web), bridges'
slash-command path silently dropped /<cmd> messages because
session_ctx.handle_slash was only wired in the interactive REPL.
The fix extracts the handler into a module-level factory used by both
the REPL and the headless bootstrap.

These tests pin the factory's contract so future changes to either
call site can't reintroduce the regression.
"""
from __future__ import annotations

from unittest.mock import patch

from cheetahclaws import cli as cheetahclaws


def _make_handler(handle_slash_return, run_query_calls):
    """Build the handler under test with handle_slash mocked to a fixed return.

    run_query_calls is a list that records every prompt run_query is called
    with — lets tests assert on the prompts dispatched for sentinel flows.
    """
    state = object()        # opaque — only handle_slash inspects it
    config = {}             # opaque — same
    run_query = lambda prompt, *a, **kw: run_query_calls.append(prompt)
    with patch.object(cheetahclaws, "handle_slash",
                      return_value=handle_slash_return):
        return cheetahclaws._make_bridge_slash_handler(state, config,
                                                        run_query), state, config


def test_simple_command_returns_simple_and_does_not_invoke_run_query():
    """/help, /status, /model, /cost — handle_slash returns a non-tuple
    (typically True/False); the handler should report "simple" and not
    spawn a background agent run."""
    calls: list[str] = []
    with patch.object(cheetahclaws, "handle_slash", return_value=True):
        handler = cheetahclaws._make_bridge_slash_handler(
            object(), {}, lambda p, *a, **kw: calls.append(p)
        )
        assert handler("/status") == "simple"
        assert handler("/help") == "simple"
        assert handler("/model") == "simple"
    assert calls == [], "run_query must not fire for simple commands"


def test_brainstorm_sentinel_dispatches_payload_through_run_query(tmp_path):
    """A __brainstorm__ sentinel returns (sentinel, payload, out_file).
    The handler should call run_query exactly once with the payload plus
    the strict todo-write rules, and report "query"."""
    out_file = tmp_path / "brainstorm" / "out.md"
    sentinel = ("__brainstorm__", "PAYLOAD_GOES_HERE", str(out_file))
    calls: list[str] = []
    with patch.object(cheetahclaws, "handle_slash", return_value=sentinel):
        handler = cheetahclaws._make_bridge_slash_handler(
            object(), {}, lambda p, *a, **kw: calls.append(p)
        )
        assert handler("/brainstorm") == "query"
    assert len(calls) == 1, "exactly one run_query call expected"
    body = calls[0]
    assert "PAYLOAD_GOES_HERE" in body
    # The strict-rules block carries the todo path so the agent writes
    # the file in one shot.
    assert "todo_list.txt" in body
    assert "STRICT RULES" in body


def test_worker_sentinel_dispatches_one_run_query_per_task():
    """A __worker__ sentinel returns (sentinel, [(idx, name, prompt), ...]).
    The handler should fire run_query once per task in order."""
    tasks = [
        (0, "task-a", "PROMPT_A"),
        (1, "task-b", "PROMPT_B"),
        (2, "task-c", "PROMPT_C"),
    ]
    sentinel = ("__worker__", tasks)
    calls: list[str] = []
    with patch.object(cheetahclaws, "handle_slash", return_value=sentinel):
        handler = cheetahclaws._make_bridge_slash_handler(
            object(), {}, lambda p, *a, **kw: calls.append(p)
        )
        assert handler("/worker") == "query"
    assert calls == ["PROMPT_A", "PROMPT_B", "PROMPT_C"]


def test_unknown_sentinel_still_returns_query_without_running():
    """An unrecognised sentinel tuple shouldn't crash the handler — the
    bridge should still see "query" and the user gets no spurious reply.
    Future sentinels (e.g. __ssj_cmd__, __plan__) should grow handlers
    here; until then they're a no-op rather than a hard error."""
    sentinel = ("__some_future_sentinel__", "ignored")
    calls: list[str] = []
    with patch.object(cheetahclaws, "handle_slash", return_value=sentinel):
        handler = cheetahclaws._make_bridge_slash_handler(
            object(), {}, lambda p, *a, **kw: calls.append(p)
        )
        assert handler("/whatever") == "query"
    assert calls == []


def test_handler_is_assigned_in_headless_bridges_bootstrap(monkeypatch):
    """End-to-end pin: when _start_headless_bridges runs with bridge
    config present, session_ctx.handle_slash must be set to a callable.
    Pre-fix this attribute stayed None, which is the actual user bug."""
    from cheetahclaws import runtime
    sid = "test-headless-slash-wire"
    config = {
        "_session_id": sid,
        "telegram_token": "FAKE_TOKEN",
        "telegram_chat_id": 12345,
    }
    # Stub the actual bridge thread spawn so we don't make HTTP calls.
    from cheetahclaws import cli as cc
    monkeypatch.setattr(cc._btg, "_telegram_thread", None)

    class _NoopThread:
        def start(self): pass
        def is_alive(self): return False
    monkeypatch.setattr("threading.Thread", lambda *a, **kw: _NoopThread())

    cc._start_headless_bridges(config)

    ctx = runtime.get_session_ctx(sid)
    assert callable(ctx.handle_slash), \
        "handle_slash must be wired in headless bootstrap (issue #84 follow-up)"
    assert callable(ctx.run_query), \
        "run_query must be wired in headless bootstrap"


def test_headless_bootstrap_wires_tg_send(monkeypatch):
    """Issue #84 follow-up: ask_input_interactive only routes to Telegram when
    session_ctx.tg_send is non-None.  Headless bootstrap previously left it
    unset, so inline-keyboard approval prompts never reached the user — the
    bridge silently fell through to terminal input()."""
    from cheetahclaws import runtime; from cheetahclaws import cli as cc
    sid = "test-headless-tgsend-wire"
    config = {
        "_session_id": sid,
        "telegram_token": "FAKE_TOKEN",
        "telegram_chat_id": 9999,
    }
    monkeypatch.setattr(cc._btg, "_telegram_thread", None)

    class _NoopThread:
        def start(self): pass
        def is_alive(self): return False
    monkeypatch.setattr("threading.Thread", lambda *a, **kw: _NoopThread())

    cc._start_headless_bridges(config)

    ctx = runtime.get_session_ctx(sid)
    assert callable(ctx.tg_send), \
        "tg_send must be wired in headless bootstrap so ask_input_interactive " \
        "can render Telegram inline-keyboard prompts (issue #84)"


def test_headless_run_query_handles_permission_request(monkeypatch):
    """Issue #84 follow-up: in headless mode the agent loop yields a
    PermissionRequest event for sensitive tools.  Pre-fix that event was
    dropped, leaving event.granted=False, so every approval-required tool
    silently denied without ever asking the user."""
    from cheetahclaws import runtime; from cheetahclaws import cli as cc
    from cheetahclaws.agent import PermissionRequest

    sid = "test-headless-permission-event"
    config = {
        "_session_id": sid,
        "telegram_token": "FAKE_TOKEN",
        "telegram_chat_id": 7777,
    }
    monkeypatch.setattr(cc._btg, "_telegram_thread", None)

    class _NoopThread:
        def start(self): pass
        def is_alive(self): return False
    monkeypatch.setattr("threading.Thread", lambda *a, **kw: _NoopThread())

    # Stub the agent loop: yield exactly one PermissionRequest, then stop.
    captured: list[PermissionRequest] = []
    def _fake_run(prompt, state, cfg, system_prompt):
        req = PermissionRequest(description="mkdir test_folder")
        captured.append(req)
        yield req

    monkeypatch.setattr("cheetahclaws.agent.run", _fake_run)
    monkeypatch.setattr(cc, "ask_permission_interactive",
                        lambda desc, cfg: True)
    monkeypatch.setattr("cheetahclaws.context.build_system_prompt", lambda c: "sys")

    cc._start_headless_bridges(config)
    ctx = runtime.get_session_ctx(sid)

    ctx.run_query("please make a folder")

    assert len(captured) == 1
    assert captured[0].granted is True, \
        "_headless_run_query must consult ask_permission_interactive on " \
        "PermissionRequest events (issue #84)"


def test_headless_run_query_promotes_telegram_incoming(monkeypatch):
    """When a Telegram message triggers the agent, _bg_runner sets
    telegram_incoming=True before calling run_query.  _headless_run_query
    must promote that to in_telegram_turn so _is_in_tg_turn() returns True
    while ask_input_interactive routes the prompt — otherwise prompts fall
    through to terminal input()."""
    from cheetahclaws import runtime; from cheetahclaws import cli as cc

    sid = "test-headless-turn-promotion"
    config = {
        "_session_id": sid,
        "telegram_token": "FAKE_TOKEN",
        "telegram_chat_id": 4242,
    }
    monkeypatch.setattr(cc._btg, "_telegram_thread", None)

    class _NoopThread:
        def start(self): pass
        def is_alive(self): return False
    monkeypatch.setattr("threading.Thread", lambda *a, **kw: _NoopThread())

    seen_in_turn: list[bool] = []
    def _fake_run(prompt, state, cfg, system_prompt):
        seen_in_turn.append(runtime.get_session_ctx(sid).in_telegram_turn)
        return iter(())  # generator yielding nothing

    monkeypatch.setattr("cheetahclaws.agent.run", _fake_run)
    monkeypatch.setattr("cheetahclaws.context.build_system_prompt", lambda c: "sys")

    cc._start_headless_bridges(config)
    ctx = runtime.get_session_ctx(sid)

    ctx.telegram_incoming = True
    ctx.run_query("hi")

    assert seen_in_turn == [True], \
        "in_telegram_turn must be True during the agent run, not just before " \
        "and after (issue #84)"
    assert ctx.in_telegram_turn is False, \
        "in_telegram_turn must be cleared after the run so the next " \
        "non-Telegram event isn't misrouted"
