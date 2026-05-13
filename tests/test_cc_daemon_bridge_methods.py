"""Tests for cc_daemon/bridge_methods.py (RFC 0002 F-6/7/8).

Exercises the five JSON-RPC methods (bridge.start / stop / list / send /
status) end-to-end through an ``RpcRegistry``, the same way ``server.py``
dispatches them.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeDaemonState:
    def __init__(self, config=None):
        self.config = config or {}


def _build_registry(state=None):
    from cc_daemon.rpc import RpcRegistry
    from cc_daemon import bridge_methods
    reg = RpcRegistry()
    bridge_methods.register(reg, state or _FakeDaemonState())
    return reg


def _ctx():
    from cc_daemon.rpc import CallContext
    return CallContext(client_id="tester", transport="unix", api_version="0")


def _call(reg, method, params=None):
    envelope = {"jsonrpc": "2.0", "id": 1, "method": method,
                "params": params or {}}
    response, _ = reg.dispatch(envelope, _ctx())
    return response.get("result"), response.get("error")


class _BridgeMethodsBase(unittest.TestCase):

    def setUp(self):
        from cc_daemon import schema
        from cc_daemon import bridge_supervisor as bs

        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        schema.set_db_path(self._db_path)
        schema._local.conn = None

        self._saved_env = {
            k: os.environ.pop(k, None)
            for k in ("CHEETAHCLAWS_ENABLE_F6",
                      "CHEETAHCLAWS_ENABLE_F7",
                      "CHEETAHCLAWS_ENABLE_F8")
        }
        with bs._handles_lock:
            for h in list(bs._handles.values()):
                try:
                    h.stop_event.set()
                except Exception:
                    pass
            bs._handles.clear()

    def tearDown(self):
        from cc_daemon import schema
        from cc_daemon import bridge_supervisor as bs

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

        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v

        if hasattr(schema._local, "conn") and schema._local.conn is not None:
            try:
                schema._local.conn.close()
            except Exception:
                pass
            schema._local.conn = None
        schema._db_path = None
        self._tmpdir.cleanup()


class TestRegistration(_BridgeMethodsBase):

    def test_all_methods_registered(self):
        reg = _build_registry()
        names = set(reg.methods())
        expected = {"bridge.start", "bridge.stop", "bridge.list",
                    "bridge.send", "bridge.status"}
        self.assertTrue(expected.issubset(names))


class TestParamValidation(_BridgeMethodsBase):

    def test_start_requires_kind(self):
        reg = _build_registry()
        _, err = _call(reg, "bridge.start", {})
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)
        self.assertIn("kind", err["message"])

    def test_start_rejects_non_dict_config(self):
        reg = _build_registry()
        _, err = _call(reg, "bridge.start", {"kind": "telegram", "config": 1})
        self.assertIsNotNone(err)
        self.assertIn("config", err["message"])

    def test_stop_requires_kind(self):
        reg = _build_registry()
        _, err = _call(reg, "bridge.stop", {})
        self.assertIsNotNone(err)
        self.assertIn("kind", err["message"])

    def test_send_requires_text(self):
        reg = _build_registry()
        _, err = _call(reg, "bridge.send", {"kind": "telegram"})
        self.assertIsNotNone(err)
        self.assertIn("text", err["message"])

    def test_status_unknown_kind_returns_not_found(self):
        reg = _build_registry()
        result, err = _call(reg, "bridge.status", {"kind": "telegram"})
        self.assertIsNone(err)
        self.assertEqual(result, {"kind": "telegram", "found": False})


class TestStartStopRoundTrip(_BridgeMethodsBase):

    def test_start_requires_feature_flag(self):
        reg = _build_registry()
        _, err = _call(reg, "bridge.start", {
            "kind": "telegram",
            "config": {"telegram_token": "t", "telegram_chat_id": 1},
        })
        self.assertIsNotNone(err)
        self.assertIn("CHEETAHCLAWS_ENABLE_F6", err["message"])

    def test_start_list_stop_with_flag_on(self):
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"

        ev = threading.Event()
        with patch("bridges.telegram._tg_supervisor",
                   side_effect=lambda *a, **kw: ev.wait()):
            reg = _build_registry()
            try:
                # start.
                result, err = _call(reg, "bridge.start", {
                    "kind": "telegram",
                    "config": {"telegram_token": "fake",
                               "telegram_chat_id": 9},
                })
                self.assertIsNone(err, err)
                self.assertEqual(result["kind"], "telegram")
                self.assertTrue(result["alive"])
                # Token redacted in the response.
                self.assertTrue(
                    result["config"]["telegram_token"].startswith("***"))

                # list returns it.
                result, err = _call(reg, "bridge.list", {})
                self.assertIsNone(err)
                kinds = [b["kind"] for b in result["bridges"]]
                self.assertIn("telegram", kinds)

                # status returns it.
                result, err = _call(reg, "bridge.status", {"kind": "telegram"})
                self.assertIsNone(err)
                self.assertTrue(result["found"])
                self.assertTrue(result["alive"])
            finally:
                ev.set()
                _call(reg, "bridge.stop", {"kind": "telegram"})

            # stop returns stopped: True.
            result, err = _call(reg, "bridge.stop", {"kind": "telegram"})
            # second stop is a no-op but should not crash.
            self.assertIsNone(err)


class TestSendOutbound(_BridgeMethodsBase):

    def test_send_with_no_bridge_returns_delivered_false(self):
        reg = _build_registry()
        result, err = _call(reg, "bridge.send",
                             {"kind": "telegram", "text": "x"})
        self.assertIsNone(err)
        self.assertEqual(result, {"kind": "telegram", "delivered": False})

    def test_send_with_running_bridge_calls_sender(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"

        ev = threading.Event()
        with patch("bridges.telegram._tg_supervisor",
                   side_effect=lambda *a, **kw: ev.wait()):
            handle = bs.start("telegram", {"telegram_token": "t",
                                            "telegram_chat_id": 1})
            sent = []
            handle.sender = lambda cfg, text: (sent.append(text) or True)
            try:
                reg = _build_registry()
                result, err = _call(reg, "bridge.send",
                                     {"kind": "telegram", "text": "hi"})
                self.assertIsNone(err)
                self.assertEqual(result, {"kind": "telegram",
                                          "delivered": True})
                self.assertEqual(sent, ["hi"])
            finally:
                ev.set()
                bs.stop("telegram", timeout_s=3.0)


if __name__ == "__main__":
    unittest.main()
