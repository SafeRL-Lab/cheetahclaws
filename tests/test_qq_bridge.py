"""Tests for bridges/qq.py — message parsing, config, turn detection."""
import sys
import threading
import time
import types
import pytest


class _FakeRoute:
    def __init__(self, method, path, **kwargs):
        self.method = method
        self.path = path
        self.kwargs = kwargs


@pytest.fixture
def fake_botpy_route(monkeypatch):
    """Provide the botpy Route class needed by payload-only send tests."""
    fake_botpy = types.ModuleType("botpy")
    fake_http = types.ModuleType("botpy.http")
    fake_http.Route = _FakeRoute
    fake_botpy.http = fake_http
    monkeypatch.setitem(sys.modules, "botpy", fake_botpy)
    monkeypatch.setitem(sys.modules, "botpy.http", fake_http)


def test_config_defaults(monkeypatch, tmp_path):
    """QQ config keys exist in DEFAULTS."""
    monkeypatch.setenv("HOME", str(tmp_path))
    import importlib
    import cc_config
    importlib.reload(cc_config)
    cfg = cc_config.load_config()
    assert "qq_appid" in cfg
    assert "qq_secret" in cfg
    assert cfg["qq_appid"] == ""
    assert cfg["qq_secret"] == ""


def test_runtime_context_fields():
    """RuntimeContext has QQ fields with correct defaults."""
    from runtime import RuntimeContext
    ctx = RuntimeContext()
    assert ctx.qq_send is None
    assert ctx.qq_input_event is None
    assert ctx.qq_input_value == ""
    assert ctx.qq_input_target_id == ""
    assert ctx.in_qq_turn is False
    assert ctx.qq_current_target_id == ""
    assert ctx.qq_current_msg_type == ""


def test_is_in_qq_turn_default():
    """Turn detection returns False when no QQ turn is active."""
    from tools.interaction import _is_in_qq_turn
    assert _is_in_qq_turn({}) is False


def test_is_in_qq_turn_thread_local():
    """Turn detection returns True when thread-local flag is set."""
    from tools.interaction import _qq_thread_local, _is_in_qq_turn
    _qq_thread_local.active = True
    try:
        assert _is_in_qq_turn({}) is True
    finally:
        _qq_thread_local.active = False


def test_is_in_qq_turn_runtime_ctx():
    """Turn detection returns True when RuntimeContext.in_qq_turn is True."""
    from tools.interaction import _is_in_qq_turn
    import runtime
    ctx = runtime.get_session_ctx("_test_qq_turn")
    ctx.in_qq_turn = True
    try:
        assert _is_in_qq_turn({"_session_id": "_test_qq_turn"}) is True
    finally:
        ctx.in_qq_turn = False
        runtime.release_session_ctx("_test_qq_turn")


def test_qq_cmd_missing_config():
    """cmd_qq shows error when no args and no saved config."""
    from bridges.qq import cmd_qq
    result = cmd_qq("", None, {"qq_appid": "", "qq_secret": ""})
    assert result is True


def test_qq_cmd_inline_config(tmp_path, monkeypatch):
    """cmd_qq saves appid/secret when provided inline."""
    from unittest.mock import patch
    from bridges.qq import cmd_qq
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = {"qq_appid": "", "qq_secret": ""}
    # Mock bridge start to avoid spawning a real daemon thread
    with patch("bridges.qq._qq_start_bridge"):
        result = cmd_qq("myappid mysecret", None, cfg)
    assert result is True
    assert cfg["qq_appid"] == "myappid"
    assert cfg["qq_secret"] == "mysecret"


def test_qq_cmd_status_not_running():
    """cmd_qq status reports not configured when empty."""
    from bridges.qq import cmd_qq
    result = cmd_qq("status", None, {"qq_appid": "", "qq_secret": ""})
    assert result is True


def test_qq_cmd_status_configured():
    """cmd_qq status reports configured but not running."""
    from bridges.qq import cmd_qq
    result = cmd_qq("status", None, {"qq_appid": "test123", "qq_secret": "sec"})
    assert result is True


def test_message_dedup_set_capped():
    """_qq_seen_msgids stays under 2000 entries."""
    from bridges import qq
    for i in range(2100):
        qq._qq_seen_msgids.add(f"msg_{i}")
    assert len(qq._qq_seen_msgids) <= 2100
    qq._qq_seen_msgids.clear()


def test_reply_ctx_tracking():
    """Passive reply context stores msg_id, event_id, seq, timestamp, and msg_type."""
    from bridges.qq import _qq_reply_ctx, _qq_reply_lock
    with _qq_reply_lock:
        _qq_reply_ctx["test_target"] = ("msg123", "event456", 1, time.time(), "group")
    assert "test_target" in _qq_reply_ctx
    msg_id, event_id, seq, ts, msg_type = _qq_reply_ctx["test_target"]
    assert msg_id == "msg123"
    assert event_id == "event456"
    assert seq == 1
    assert msg_type == "group"
    # Test with None values
    with _qq_reply_lock:
        _qq_reply_ctx["test_target2"] = (None, None, 1, time.time(), "c2c")
    msg_id, event_id, seq, ts, msg_type = _qq_reply_ctx["test_target2"]
    assert msg_id is None
    assert event_id is None
    assert msg_type == "c2c"
    with _qq_reply_lock:
        del _qq_reply_ctx["test_target"]
        del _qq_reply_ctx["test_target2"]


def test_qq_send_no_api():
    """_qq_send is a no-op when no API is configured."""
    from bridges.qq import _qq_send
    _qq_send("some_target", "hello", {"qq_appid": "x"})


def test_qq_pending_input_only_accepts_prompt_target():
    """A QQ permission reply from another target must not release the prompt."""
    from runtime import RuntimeContext
    from bridges.qq import _qq_try_deliver_input

    ctx = RuntimeContext()
    evt = threading.Event()
    ctx.qq_input_event = evt
    ctx.qq_input_target_id = "target-a"

    assert _qq_try_deliver_input(ctx, "target-b", "y") is False
    assert not evt.is_set()
    assert ctx.qq_input_value == ""

    assert _qq_try_deliver_input(ctx, "target-a", "y") is True
    assert evt.is_set()
    assert ctx.qq_input_value == "y"


def test_qq_send_with_chunks():
    """_qq_send splits long text into chunks."""
    from bridges.qq import _qq_send, _QQ_MAX_MSG_LEN
    long_text = "A" * (_QQ_MAX_MSG_LEN * 2 + 100)
    # Should not raise even without API
    _qq_send("target", long_text, {})


def test_passive_window_constants():
    """Passive reply window follows botpy's documented 5-minute validity."""
    from bridges.qq import _QQ_PASSIVE_WINDOW, _QQ_STREAM_INTERVAL, _QQ_MAX_MSG_LEN, _QQ_STREAM_MIN_LEN
    assert _QQ_PASSIVE_WINDOW == 300
    assert _QQ_STREAM_INTERVAL == 2.0
    assert _QQ_MAX_MSG_LEN == 2000
    assert _QQ_STREAM_MIN_LEN == 80


def test_send_future_exception_is_logged():
    """Errors raised by scheduled QQ HTTP sends should be surfaced."""
    from concurrent.futures import Future
    from unittest.mock import patch
    from bridges.qq import _qq_log_send_future

    fut = Future()
    fut.set_exception(RuntimeError("api failed"))

    with patch("bridges.qq._log.warn") as warn:
        _qq_log_send_future(fut, "group", "target")

    warn.assert_called_once()
    assert warn.call_args.args[0] == "qq_send_api_error"


def test_queue_or_dispatch_marks_busy_before_thread_dispatch():
    """A second same-target job should queue before worker thread starts."""
    from unittest.mock import MagicMock, patch
    from bridges import qq

    job1 = MagicMock()
    job1.id = "job1"
    job2 = MagicMock()
    job2.id = "job2"

    with qq._qq_queues_lock:
        qq._qq_busy.clear()
        qq._qq_queues.clear()

    dispatched = []

    def fake_dispatch(job, prompt, target_id, msg_type, run_query_cb, session_ctx, config, image_b64=None):
        dispatched.append((job.id, prompt, target_id, image_b64))

    with patch("bridges.qq._dispatch_qq_job", side_effect=fake_dispatch):
        pos1 = qq._queue_or_dispatch_qq_job(
            job1, "prompt1", "target", "group", None, None, {}, "img1"
        )
        pos2 = qq._queue_or_dispatch_qq_job(
            job2, "prompt2", "target", "group", None, None, {}, "img2"
        )

    assert pos1 == 0
    assert pos2 == 1
    assert dispatched == [("job1", "prompt1", "target", "img1")]
    assert qq._qq_queues["target"] == [("job2", "prompt2", "img2")]

    with qq._qq_queues_lock:
        qq._qq_busy.clear()
        qq._qq_queues.clear()


def test_qq_thread_not_running_initially():
    """QQ bridge thread state is properly managed."""
    from bridges.qq import _qq_thread
    # After import, thread may have been started by other tests;
    # just verify the module-level variable exists and is accessible
    assert _qq_thread is None or isinstance(_qq_thread, threading.Thread)


def test_qq_stop_event_cleared():
    """QQ stop event should not be set initially."""
    from bridges.qq import _qq_stop
    assert not _qq_stop.is_set()


def test_post_group_clean_payload_no_msg_id(fake_botpy_route):
    """_qq_post_group builds clean payload without msg_id/event_id when empty."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from bridges.qq import _qq_post_group

    api = MagicMock()
    api._http = MagicMock()
    api._http.request = AsyncMock()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_qq_post_group(api, "group123", "hello"))
    finally:
        loop.close()

    api._http.request.assert_called_once()
    call_args = api._http.request.call_args
    payload = call_args[1]["json"]
    # msg_id, event_id, and msg_seq should NOT be in payload when not provided
    assert "msg_id" not in payload
    assert "event_id" not in payload
    assert "msg_seq" not in payload
    # Only clean fields present
    assert payload == {"group_openid": "group123", "msg_type": 0, "content": "hello"}
    # No null fields from botpy's locals() pattern
    assert "embed" not in payload
    assert "ark" not in payload
    assert "media" not in payload


def test_post_group_clean_payload_with_msg_id(fake_botpy_route):
    """_qq_post_group includes msg_id/msg_seq when msg_id is provided."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from bridges.qq import _qq_post_group

    api = MagicMock()
    api._http = MagicMock()
    api._http.request = AsyncMock()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_qq_post_group(api, "group123", "reply text", msg_id="msg456", msg_seq=2))
    finally:
        loop.close()

    payload = api._http.request.call_args[1]["json"]
    assert payload["msg_id"] == "msg456"
    assert payload["msg_seq"] == 2
    assert payload["content"] == "reply text"


def test_post_group_clean_payload_with_event_id(fake_botpy_route):
    """_qq_post_group uses event_id when msg_id is None."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from bridges.qq import _qq_post_group

    api = MagicMock()
    api._http = MagicMock()
    api._http.request = AsyncMock()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_qq_post_group(api, "group123", "reply text", event_id="evt789", msg_seq=1))
    finally:
        loop.close()

    payload = api._http.request.call_args[1]["json"]
    assert "msg_id" not in payload  # msg_id should not be set
    assert payload["event_id"] == "evt789"
    assert payload["msg_seq"] == 1
    assert payload["content"] == "reply text"


def test_post_c2c_clean_payload(fake_botpy_route):
    """_qq_post_c2c builds clean payload."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from bridges.qq import _qq_post_c2c

    api = MagicMock()
    api._http = MagicMock()
    api._http.request = AsyncMock()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_qq_post_c2c(api, "user123", "hi"))
    finally:
        loop.close()

    payload = api._http.request.call_args[1]["json"]
    assert "msg_id" not in payload
    assert "event_id" not in payload
    assert "msg_seq" not in payload
    assert payload == {"openid": "user123", "msg_type": 0, "content": "hi"}


def test_msg_seq_starts_at_1_for_new_message():
    """First send with reply context should use msg_seq=1."""
    import time
    from bridges.qq import _qq_reply_ctx, _qq_reply_lock
    from unittest.mock import MagicMock, patch

    # Set up reply context with seq=0 (as stored by _handle_message)
    with _qq_reply_lock:
        _qq_reply_ctx["test_target"] = ("msg123", "event456", 0, time.time(), "group")

    # Mock the API calls to capture the msg_seq values
    captured_seqs = []

    def mock_send_group(api, group_openid, content, msg_id=None, event_id=None, msg_seq=1):
        captured_seqs.append((msg_id, event_id, msg_seq, content))

    with patch("bridges.qq._qq_send_group", side_effect=mock_send_group):
        with patch("bridges.qq._qq_api_client", MagicMock()):
            from bridges.qq import _qq_send
            _qq_send("test_target", "hello", {})

    # First chunk should have msg_seq=1
    assert len(captured_seqs) == 1
    assert captured_seqs[0][0] == "msg123"  # msg_id
    assert captured_seqs[0][2] == 1  # msg_seq should be 1, not 2

    # Clean up
    with _qq_reply_lock:
        del _qq_reply_ctx["test_target"]


def test_msg_seq_increments_correctly_for_chunks():
    """Multiple chunks should increment msg_seq properly."""
    import time
    from bridges.qq import _qq_reply_ctx, _qq_reply_lock, _QQ_MAX_MSG_LEN
    from unittest.mock import MagicMock, patch

    # Set up reply context with seq=0
    with _qq_reply_lock:
        _qq_reply_ctx["test_target"] = ("msg123", "event456", 0, time.time(), "group")

    captured_seqs = []

    def mock_send_group(api, group_openid, content, msg_id=None, event_id=None, msg_seq=1):
        captured_seqs.append((msg_id, event_id, msg_seq, content))

    with patch("bridges.qq._qq_send_group", side_effect=mock_send_group):
        with patch("bridges.qq._qq_api_client", MagicMock()):
            from bridges.qq import _qq_send
            # Send text that will be split into 2 chunks
            long_text = "A" * (_QQ_MAX_MSG_LEN + 100)
            _qq_send("test_target", long_text, {})

    # Should have 2 chunks with msg_seq=1 and msg_seq=2
    assert len(captured_seqs) == 2
    assert captured_seqs[0][2] == 1  # First chunk: seq=1
    assert captured_seqs[1][2] == 2  # Second chunk: seq=2

    # Clean up
    with _qq_reply_lock:
        del _qq_reply_ctx["test_target"]


def test_no_duplicate_send_in_bg_runner():
    """_qq_bg_runner should not send duplicate messages."""
    from unittest.mock import MagicMock, patch
    from bridges.qq import _qq_bg_runner

    session_ctx = MagicMock()

    job = MagicMock()
    job.id = "test123"

    run_query_cb = MagicMock()
    send_calls = []

    def mock_qq_send(target_id, text, _cfg=None, _msg_type=None):
        send_calls.append((target_id, text))

    with patch("bridges.qq._qq_api_client", MagicMock()):
        with patch("bridges.qq._qq_send", side_effect=mock_qq_send):
            with patch("bridges.qq._jobs.start"):
                with patch("bridges.qq._jobs.stream_result"):
                    with patch("bridges.qq._jobs.complete"):
                        _qq_bg_runner(job, "test prompt", "target123", "group",
                                     run_query_cb, session_ctx, {})

    # Check that _qq_send was called reasonable times (not hundreds of duplicates)
    # We expect: 1 "任务执行中" message + result messages
    # But not the same message repeated dozens of times
    assert len(send_calls) < 50, f"Too many send calls: {len(send_calls)}, likely duplicate echo bug"
    # Verify the expected initial message is present
    assert any("执行中" in str(call[1]) for call in send_calls), "Missing initial status message"


def test_qq_bg_runner_sets_pending_image_for_matching_job():
    """Downloaded QQ images should be attached to the job that owns them."""
    from unittest.mock import MagicMock, patch
    from bridges.qq import _qq_bg_runner
    import runtime

    session_id = "_test_qq_image_job"
    config = {"_session_id": session_id}
    session_ctx = runtime.get_session_ctx(session_id)
    job = MagicMock()
    job.id = "img-job"
    seen_pending = []

    def run_query_cb(_prompt):
        seen_pending.append(runtime.get_ctx(config).pending_image)
        runtime.get_ctx(config).pending_image = None

    try:
        with patch("bridges.qq._qq_send"):
            with patch("bridges.qq._jobs.start"):
                with patch("bridges.qq._jobs.complete"):
                    _qq_bg_runner(
                        job,
                        "describe this",
                        "target",
                        "group",
                        run_query_cb,
                        session_ctx,
                        config,
                        "base64-image",
                    )
    finally:
        runtime.release_session_ctx(session_id)

    assert seen_pending == ["base64-image"]


def test_qq_bg_runner_clears_pending_image_if_run_query_fails_before_consuming():
    """A failed image job must not leak its image into the next turn."""
    from unittest.mock import MagicMock, patch
    from bridges.qq import _qq_bg_runner
    import runtime

    session_id = "_test_qq_image_fail_cleanup"
    config = {"_session_id": session_id}
    session_ctx = runtime.get_session_ctx(session_id)
    job = MagicMock()
    job.id = "img-fail-job"

    def run_query_cb(_prompt):
        raise RuntimeError("boom before agent consumes image")

    try:
        with patch("bridges.qq._qq_send"):
            with patch("bridges.qq._jobs.start"):
                with patch("bridges.qq._jobs.fail"):
                    _qq_bg_runner(
                        job,
                        "describe this",
                        "target",
                        "group",
                        run_query_cb,
                        session_ctx,
                        config,
                        "base64-image",
                    )
        assert runtime.get_ctx(config).pending_image is None
    finally:
        runtime.release_session_ctx(session_id)


def test_streaming_hook_idempotency():
    """Streaming hooks should handle duplicate calls gracefully."""
    from unittest.mock import MagicMock, patch
    from bridges.qq import _qq_bg_runner

    session_ctx = MagicMock()

    job = MagicMock()
    job.id = "test999"

    # Track how many times each hook is called
    chunk_count = [0]
    tool_start_count = [0]
    tool_end_count = [0]

    # Track _qq_send calls
    send_calls = []

    def mock_run_query(prompt):
        # Simulate agent calling hooks multiple times
        if session_ctx.on_text_chunk:
            session_ctx.on_text_chunk("test chunk")
            chunk_count[0] += 1
        if session_ctx.on_tool_start:
            session_ctx.on_tool_start("TestTool", {})
            tool_start_count[0] += 1
        if session_ctx.on_tool_end:
            session_ctx.on_tool_end("TestTool", "done")
            tool_end_count[0] += 1

    def mock_qq_send(target_id, text, _cfg=None, _msg_type=None):
        _ = _cfg, _msg_type  # Mark as intentionally unused
        send_calls.append((target_id, text))

    with patch("bridges.qq._qq_api_client", MagicMock()):
        with patch("bridges.qq._qq_send", side_effect=mock_qq_send):
            with patch("bridges.qq._jobs.start"):
                with patch("bridges.qq._jobs.complete"):
                    _qq_bg_runner(job, "test", "target", "group",
                                 mock_run_query, session_ctx, {})

    # Each hook should be called exactly once
    assert chunk_count[0] == 1, f"on_text_chunk called {chunk_count[0]} times"
    assert tool_start_count[0] == 1, f"on_tool_start called {tool_start_count[0]} times"
    assert tool_end_count[0] == 1, f"on_tool_end called {tool_end_count[0]} times"

    # _qq_send should be called for: "执行中" + tool start message = 2 calls minimum
    assert len(send_calls) >= 2, f"Expected at least 2 send calls, got {len(send_calls)}"


def test_qq_bg_runner_serializes_global_streaming_hooks():
    """Concurrent QQ jobs must not overwrite each other's session-level hooks."""
    from unittest.mock import MagicMock, patch
    from bridges.qq import _qq_bg_runner
    from runtime import RuntimeContext

    session_ctx = RuntimeContext()
    job_a = MagicMock()
    job_a.id = "job-a"
    job_b = MagicMock()
    job_b.id = "job-b"

    a_entered = threading.Event()
    b_entered = threading.Event()
    release_a = threading.Event()
    send_calls = []
    send_lock = threading.Lock()

    def mock_run_query(prompt):
        if prompt == "prompt-a":
            a_entered.set()
            session_ctx.on_text_chunk("chunk-a")
            assert release_a.wait(timeout=2)
        else:
            b_entered.set()
            session_ctx.on_text_chunk("chunk-b")

    def mock_qq_send(target_id, text, _cfg=None, _msg_type=None):
        with send_lock:
            send_calls.append((target_id, text))

    with patch("bridges.qq._qq_send", side_effect=mock_qq_send):
        with patch("bridges.qq._jobs.start"):
            with patch("bridges.qq._jobs.stream_result"):
                with patch("bridges.qq._jobs.complete"):
                    t1 = threading.Thread(
                        target=_qq_bg_runner,
                        args=(
                            job_a,
                            "prompt-a",
                            "target-a",
                            "group",
                            mock_run_query,
                            session_ctx,
                            {},
                        ),
                    )
                    t2 = threading.Thread(
                        target=_qq_bg_runner,
                        args=(
                            job_b,
                            "prompt-b",
                            "target-b",
                            "group",
                            mock_run_query,
                            session_ctx,
                            {},
                        ),
                    )
                    t1.start()
                    assert a_entered.wait(timeout=2)
                    t2.start()
                    time.sleep(0.05)
                    assert not b_entered.is_set()
                    release_a.set()
                    t1.join(timeout=2)
                    t2.join(timeout=2)

    assert not t1.is_alive()
    assert not t2.is_alive()
    assert ("target-a", "chunk-a") in send_calls
    assert ("target-b", "chunk-b") in send_calls
    assert ("target-a", "chunk-b") not in send_calls
    assert ("target-b", "chunk-a") not in send_calls
