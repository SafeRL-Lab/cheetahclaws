"""Tests for cc_daemon/session_methods.py (RFC 0002 F-6 Phase 2).

Exercises the three message-passing primitives:
  * ``session.send`` publishes ``session_inbound`` events and records LRU
  * ``session.reply`` publishes ``session_outbound`` events
  * ``session.list_recent`` reflects the LRU snapshot

Plus param-validation across all three.

The event bus is a real :class:`cc_daemon.events.EventBus` instance —
we don't mock it because the publish path is what we want to verify
(observers should be able to subscribe and pick up the events).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _setup_bus(tmp_path: Path):
    """Reinit the event bus on a tmpdir-backed SQLite so the in-memory
    LRU and the SSE feed don't bleed between tests."""
    from cc_daemon import schema, events
    schema.set_db_path(tmp_path / "test.db")
    schema._local.conn = None
    events.reset_bus_for_tests()


def _teardown_bus():
    from cc_daemon import schema, events
    events.reset_bus_for_tests()
    if hasattr(schema._local, "conn") and schema._local.conn is not None:
        try:
            schema._local.conn.close()
        except Exception:
            pass
        schema._local.conn = None
    schema._db_path = None


class _FakeState:
    def __init__(self, config=None):
        self.config = config or {}


def _build_registry():
    from cc_daemon.rpc import RpcRegistry
    from cc_daemon import session_methods
    reg = RpcRegistry()
    session_methods.register(reg, _FakeState())
    return reg


def _ctx(client_id="bridge:tg:99"):
    from cc_daemon.rpc import CallContext
    return CallContext(client_id=client_id, transport="unix", api_version="0")


def _call(reg, method, params=None, ctx=None):
    envelope = {"jsonrpc": "2.0", "id": 1, "method": method,
                "params": params or {}}
    response, _ = reg.dispatch(envelope, ctx or _ctx())
    return response.get("result"), response.get("error")


class _SessionTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        _setup_bus(Path(self._tmpdir.name))
        # Wipe the LRU so test ordering doesn't leak.
        from cc_daemon import session_methods
        session_methods._RECENT_LRU.clear()

    def tearDown(self):
        _teardown_bus()
        self._tmpdir.cleanup()


class TestSessionSend(_SessionTestBase):

    def test_send_publishes_session_inbound(self):
        from cc_daemon import events
        bus = events.get_bus()
        q = bus.subscribe()
        try:
            reg = _build_registry()
            result, err = _call(reg, "session.send", {
                "session_id": "tg:42",
                "text":       "hello",
            })
            self.assertIsNone(err)
            self.assertEqual(result["session_id"], "tg:42")
            self.assertTrue(result["message_id"].startswith("msg_"))

            ev = q.get(timeout=2.0)
            self.assertEqual(ev["type"], "session_inbound")
            self.assertEqual(ev["data"]["session_id"], "tg:42")
            self.assertEqual(ev["data"]["text"], "hello")
            # client_id of the test caller is used as origin when omitted.
            self.assertEqual(ev["data"]["origin"], "bridge:tg:99")
        finally:
            bus.unsubscribe(q)

    def test_send_explicit_origin_overrides_client_id(self):
        from cc_daemon import events
        bus = events.get_bus()
        q = bus.subscribe()
        try:
            reg = _build_registry()
            _call(reg, "session.send", {
                "session_id": "tg:42",
                "text":       "hi",
                "origin":     "telegram:tg:42",
            })
            ev = q.get(timeout=2.0)
            self.assertEqual(ev["data"]["origin"], "telegram:tg:42")
        finally:
            bus.unsubscribe(q)

    def test_send_requires_session_id(self):
        reg = _build_registry()
        _, err = _call(reg, "session.send", {"text": "x"})
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)
        self.assertIn("session_id", err["message"])

    def test_send_requires_text(self):
        reg = _build_registry()
        _, err = _call(reg, "session.send", {"session_id": "tg:1"})
        self.assertIsNotNone(err)
        self.assertIn("text", err["message"])

    def test_send_rejects_non_string_origin(self):
        reg = _build_registry()
        _, err = _call(reg, "session.send", {
            "session_id": "tg:1", "text": "x", "origin": 123,
        })
        self.assertIsNotNone(err)
        self.assertIn("origin", err["message"])

    def test_send_returns_custom_message_id(self):
        reg = _build_registry()
        result, _ = _call(reg, "session.send", {
            "session_id": "tg:1", "text": "x", "message_id": "abc-123",
        })
        self.assertEqual(result["message_id"], "abc-123")


class TestSessionReply(_SessionTestBase):

    def test_reply_publishes_session_outbound(self):
        from cc_daemon import events
        bus = events.get_bus()
        q = bus.subscribe()
        try:
            reg = _build_registry()
            result, err = _call(reg, "session.reply", {
                "session_id":     "tg:42",
                "text":           "ack",
                "target_bridges": ["telegram"],
            })
            self.assertIsNone(err)
            self.assertEqual(result["session_id"], "tg:42")

            ev = q.get(timeout=2.0)
            self.assertEqual(ev["type"], "session_outbound")
            self.assertEqual(ev["data"]["target_bridges"], ["telegram"])
        finally:
            bus.unsubscribe(q)

    def test_reply_target_bridges_optional(self):
        reg = _build_registry()
        result, err = _call(reg, "session.reply", {
            "session_id": "tg:1", "text": "broadcast",
        })
        self.assertIsNone(err)
        self.assertEqual(result["session_id"], "tg:1")

    def test_reply_rejects_non_list_target(self):
        reg = _build_registry()
        _, err = _call(reg, "session.reply", {
            "session_id": "tg:1", "text": "x",
            "target_bridges": "telegram",       # should be a list
        })
        self.assertIsNotNone(err)
        self.assertIn("target_bridges", err["message"])

    def test_reply_rejects_non_string_target_element(self):
        reg = _build_registry()
        _, err = _call(reg, "session.reply", {
            "session_id": "tg:1", "text": "x",
            "target_bridges": ["telegram", 123],
        })
        self.assertIsNotNone(err)


class TestSessionListRecent(_SessionTestBase):

    def test_list_recent_reflects_recent_sends(self):
        reg = _build_registry()
        _call(reg, "session.send", {"session_id": "tg:1", "text": "a"})
        _call(reg, "session.send", {"session_id": "sl:#ops", "text": "b"})
        result, err = _call(reg, "session.list_recent", {"limit": 10})
        self.assertIsNone(err)
        ids = [s["session_id"] for s in result["sessions"]]
        # Newest-first.
        self.assertEqual(ids[0], "sl:#ops")
        self.assertEqual(ids[1], "tg:1")

    def test_list_recent_dedups_by_session(self):
        reg = _build_registry()
        for _ in range(3):
            _call(reg, "session.send", {"session_id": "tg:1", "text": "x"})
        result, _ = _call(reg, "session.list_recent", {"limit": 10})
        ids = [s["session_id"] for s in result["sessions"]]
        self.assertEqual(ids.count("tg:1"), 1)

    def test_list_recent_rejects_zero_limit(self):
        reg = _build_registry()
        _, err = _call(reg, "session.list_recent", {"limit": 0})
        self.assertIsNotNone(err)
        self.assertIn("limit", err["message"])


if __name__ == "__main__":
    unittest.main()
