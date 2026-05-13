"""Tests for RFC 0002 F-6 Phase 2 inbound refactor.

The Phase 2 worker has two independent pieces:

  1. **Outbound subscriber** — listens on the daemon event bus for
     ``session_outbound`` events matching the bridge's session_id,
     calls ``handle.sender`` to deliver.
  2. **Inbound poller** — per-kind transport function (re-uses
     ``bridges/<kind>.py`` HTTP helpers); for every new message
     publishes ``session_inbound`` on the bus with ``origin=<kind>:<sid>``.

These tests cover (1) end-to-end (real EventBus + a stub sender) and
the BridgeHandle.session_id() formatting for each kind.

End-to-end Telegram inbound is exercised via a stubbed ``_tg_api``
that returns a single message then EOF, so we observe one
``session_inbound`` publish and then a clean stop.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


pytestmark_skipif_windows = sys.platform.startswith("win")


def _setup_isolated(tmp_path: Path):
    from cc_daemon import schema, events, bridge_supervisor as bs
    schema.set_db_path(tmp_path / "test.db")
    schema._local.conn = None
    events.reset_bus_for_tests()
    with bs._handles_lock:
        for h in list(bs._handles.values()):
            try:
                h.stop_event.set()
            except Exception:
                pass
        bs._handles.clear()


def _teardown_isolated():
    from cc_daemon import schema, events, bridge_supervisor as bs
    with bs._handles_lock:
        for h in list(bs._handles.values()):
            try:
                h.stop_event.set()
            except Exception:
                pass
            try:
                h.thread.join(timeout=2.0)
            except Exception:
                pass
        bs._handles.clear()
    events.reset_bus_for_tests()
    if hasattr(schema._local, "conn") and schema._local.conn is not None:
        try:
            schema._local.conn.close()
        except Exception:
            pass
        schema._local.conn = None
    schema._db_path = None


class _Phase2Base(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        _setup_isolated(Path(self._tmpdir.name))
        self._saved_env = {
            k: os.environ.pop(k, None)
            for k in ("CHEETAHCLAWS_ENABLE_F6",
                      "CHEETAHCLAWS_ENABLE_F7",
                      "CHEETAHCLAWS_ENABLE_F8")
        }

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v
        _teardown_isolated()
        self._tmpdir.cleanup()


# ── 1. session_id formatting ────────────────────────────────────────────


class TestSessionIdFormat(unittest.TestCase):

    def test_telegram_session_id(self):
        from cc_daemon.bridge_supervisor import BridgeHandle
        h = BridgeHandle(
            kind="telegram",
            config={"telegram_chat_id": 12345},
            started_at=0.0, stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        self.assertEqual(h.session_id(), "tg:12345")

    def test_slack_session_id(self):
        from cc_daemon.bridge_supervisor import BridgeHandle
        h = BridgeHandle(
            kind="slack",
            config={"slack_channel": "C123ABC"},
            started_at=0.0, stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        self.assertEqual(h.session_id(), "sl:C123ABC")

    def test_wechat_session_id(self):
        from cc_daemon.bridge_supervisor import BridgeHandle
        h = BridgeHandle(
            kind="wechat",
            config={"wechat_user_id": "u_xyz"},
            started_at=0.0, stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        self.assertEqual(h.session_id(), "wc:u_xyz")


# ── 2. Outbound delivery via session.reply ──────────────────────────────


class TestPhase2OutboundDelivery(_Phase2Base):

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_session_reply_forwards_to_sender(self):
        from cc_daemon import bridge_supervisor as bs
        from cc_daemon import events as _events
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"

        sent: list[str] = []

        # Stub the Telegram supervisor (legacy path) — we won't reach it
        # because we're enabling Phase 2, but the import path runs.
        with patch("bridges.telegram._tg_supervisor",
                   side_effect=lambda *a, **kw: threading.Event().wait(60)):
            # Patch _tg_api to return no updates so the inbound poller
            # spins quietly while we exercise the outbound path.
            with patch("bridges.telegram._tg_api",
                       return_value={"ok": True, "result": []}):
                handle = bs.start("telegram", {
                    "telegram_token":   "fake",
                    "telegram_chat_id": 42,
                }, daemon_phase2=True)
                handle.sender = lambda cfg, text: (sent.append(text) or True)
                try:
                    # Let the worker subscribe to the bus.
                    time.sleep(0.2)
                    bus = _events.get_bus()
                    bus.publish("session_outbound", {
                        "session_id": "tg:42",
                        "text":       "out-payload",
                        "target_bridges": ["telegram"],
                    })
                    # Wait for delivery.
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        if sent:
                            break
                        time.sleep(0.05)
                    self.assertEqual(sent, ["out-payload"])
                finally:
                    bs.stop("telegram", timeout_s=3.0)

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_outbound_ignores_other_sessions(self):
        from cc_daemon import bridge_supervisor as bs
        from cc_daemon import events as _events
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"

        sent: list[str] = []

        with patch("bridges.telegram._tg_api",
                   return_value={"ok": True, "result": []}):
            handle = bs.start("telegram", {
                "telegram_token":   "fake",
                "telegram_chat_id": 42,
            }, daemon_phase2=True)
            handle.sender = lambda cfg, text: (sent.append(text) or True)
            try:
                time.sleep(0.2)
                bus = _events.get_bus()
                # Different session_id — should be ignored.
                bus.publish("session_outbound", {
                    "session_id": "tg:99",
                    "text":       "for-another-chat",
                })
                # Different target_bridges — also ignored.
                bus.publish("session_outbound", {
                    "session_id": "tg:42",
                    "text":       "for-slack",
                    "target_bridges": ["slack"],
                })
                time.sleep(0.3)
                self.assertEqual(sent, [])
            finally:
                bs.stop("telegram", timeout_s=3.0)


# ── 3. Inbound publishes session_inbound ────────────────────────────────


class TestPhase2InboundPublish(_Phase2Base):

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_telegram_inbound_publishes_event(self):
        from cc_daemon import bridge_supervisor as bs
        from cc_daemon import events as _events
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"

        # First call to _tg_api is flush (offset=-1, returns latest).
        # Subsequent calls (offset > 0) return one message then empty.
        call_count = {"n": 0}

        def fake_tg_api(token, method, params=None):
            call_count["n"] += 1
            if method != "getUpdates":
                return {"ok": True, "result": []}
            # Flush: return latest update so offset advances cleanly.
            if params and params.get("offset") == -1:
                return {"ok": True, "result": [{"update_id": 100}]}
            # First real poll: return one message.
            if call_count["n"] == 2:
                return {"ok": True, "result": [{
                    "update_id": 101,
                    "message": {
                        "chat": {"id": 42},
                        "text": "from-phone",
                    },
                }]}
            return {"ok": True, "result": []}

        with patch("bridges.telegram._tg_api", side_effect=fake_tg_api):
            # Subscribe BEFORE start so we capture the inbound event.
            bus = _events.get_bus()
            q = bus.subscribe()
            handle = bs.start("telegram", {
                "telegram_token":   "fake",
                "telegram_chat_id": 42,
            }, daemon_phase2=True)
            try:
                # Wait up to 5 s for the inbound event.
                inbound: dict | None = None
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    try:
                        ev = q.get(timeout=0.5)
                    except Exception:
                        continue
                    if ev.get("type") == "session_inbound":
                        inbound = ev
                        break
                self.assertIsNotNone(inbound,
                                     "no session_inbound event published")
                data = inbound["data"]
                self.assertEqual(data["session_id"], "tg:42")
                self.assertEqual(data["text"], "from-phone")
                self.assertTrue(data["origin"].startswith("telegram:tg:42"))
            finally:
                bus.unsubscribe(q)
                bs.stop("telegram", timeout_s=3.0)


# ── 4. bridge.start RPC accepts daemon_phase2 ───────────────────────────


class TestBridgeStartRpcPhase2(_Phase2Base):

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_rpc_passes_daemon_phase2_through(self):
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"
        from cc_daemon.rpc import RpcRegistry, CallContext
        from cc_daemon import bridge_methods, bridge_supervisor as bs

        class _State:
            config = {}

        reg = RpcRegistry()
        bridge_methods.register(reg, _State())

        with patch("bridges.telegram._tg_api",
                   return_value={"ok": True, "result": []}):
            envelope = {"jsonrpc": "2.0", "id": 1, "method": "bridge.start",
                        "params": {
                            "kind": "telegram",
                            "config": {"telegram_token": "t",
                                       "telegram_chat_id": 7},
                            "daemon_phase2": True,
                        }}
            ctx = CallContext(client_id="x", transport="unix", api_version="0")
            response, _ = reg.dispatch(envelope, ctx)
            try:
                self.assertIn("result", response, msg=str(response))
                self.assertTrue(response["result"]["daemon_phase2"])
                self.assertEqual(response["result"]["session_id"], "tg:7")
            finally:
                bs.stop("telegram", timeout_s=3.0)


if __name__ == "__main__":
    unittest.main()
