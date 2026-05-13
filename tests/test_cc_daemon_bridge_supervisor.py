"""Tests for cc_daemon/bridge_supervisor.py (RFC 0002 F-6/7/8 skeleton).

These tests stub out the network-y parts of ``bridges/<kind>.py`` so we
can exercise the supervisor's lifecycle, registry, SQLite persistence,
and outbound ``notify`` mailbox without making real HTTP calls.

Three layers of coverage (mirrors F-4):

  1. Feature flag — enabled() per kind, default off.
  2. Lifecycle — start / stop / list / status; idempotency.
  3. SQLite bridges table — INSERT/UPDATE on start/stop and listing
     persisted rows after a restart.
  4. notify() outbound mailbox — single-bridge dispatch + broadcast.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Shared setup helpers ──────────────────────────────────────────────────


class _BridgeTestBase(unittest.TestCase):
    """Resets the supervisor registry, the bridge-stop events, and the
    feature-flag env vars between cases. Points SQLite at a tmpdir so
    parallel test runs don't share state."""

    def setUp(self):
        from cc_daemon import schema
        from cc_daemon import bridge_supervisor as bs
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        schema.set_db_path(self._db_path)
        schema._local.conn = None

        # Save + clear env flags.
        self._saved_env = {
            k: os.environ.pop(k, None)
            for k in ("CHEETAHCLAWS_ENABLE_F6",
                      "CHEETAHCLAWS_ENABLE_F7",
                      "CHEETAHCLAWS_ENABLE_F8")
        }

        # Wipe live handles.
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

        # Restore env flags.
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


# ── 1. Feature flag ────────────────────────────────────────────────────────


class TestFeatureFlag(_BridgeTestBase):

    def test_enabled_default_off(self):
        from cc_daemon import bridge_supervisor as bs
        self.assertFalse(bs.enabled("telegram"))
        self.assertFalse(bs.enabled("slack"))
        self.assertFalse(bs.enabled("wechat"))

    def test_enabled_unknown_kind_is_false(self):
        from cc_daemon import bridge_supervisor as bs
        self.assertFalse(bs.enabled("discord"))
        self.assertFalse(bs.enabled(""))

    def test_enabled_via_env(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"
        self.assertTrue(bs.enabled("telegram"))
        # F-7 and F-8 stay off — flags are per-bridge.
        self.assertFalse(bs.enabled("slack"))
        self.assertFalse(bs.enabled("wechat"))

    def test_truthy_values(self):
        from cc_daemon import bridge_supervisor as bs
        for v in ("1", "true", "TRUE", "yes", "on", " 1 "):
            os.environ["CHEETAHCLAWS_ENABLE_F6"] = v
            self.assertTrue(bs.enabled("telegram"), v)
        for v in ("0", "false", "no", "", "junk"):
            os.environ["CHEETAHCLAWS_ENABLE_F6"] = v
            self.assertFalse(bs.enabled("telegram"), v)


# ── 2. Lifecycle ───────────────────────────────────────────────────────────


def _quiet_telegram_worker_stub(stop_event):
    """A pretend Telegram supervisor. Sits there waiting for the
    stop_event so the daemon's lifecycle exercises real Thread.join().
    Returns immediately when stop fires."""
    stop_event.wait()


class TestLifecycle(_BridgeTestBase):

    def test_start_without_flag_raises(self):
        from cc_daemon import bridge_supervisor as bs
        with self.assertRaises(RuntimeError) as ctx:
            bs.start("telegram", {"telegram_token": "x", "telegram_chat_id": 1})
        self.assertIn("CHEETAHCLAWS_ENABLE_F6", str(ctx.exception))

    def test_start_unsupported_kind_raises(self):
        from cc_daemon import bridge_supervisor as bs
        with self.assertRaises(ValueError):
            bs.start("discord", {})

    def test_start_slack_without_telegram_flag_raises(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F7"] = "1"
        with self.assertRaises(RuntimeError) as ctx:
            bs.start("slack", {"slack_token": "x", "slack_channel": "c"})
        self.assertIn("depends on F-6", str(ctx.exception))

    def test_start_and_stop_telegram(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"

        # Patch the inner supervisor so we don't actually hit Telegram.
        with patch("bridges.telegram._tg_supervisor",
                   side_effect=lambda *a, **kw: _quiet_telegram_worker_stub(
                       a[2].get("_test_stop", threading.Event())
                       if len(a) > 2 and isinstance(a[2], dict)
                       else threading.Event())):
            # The real bridge_worker rebinds _telegram_stop; pass the
            # bridge-supervisor's stop_event through ``config["_test_stop"]``
            # for our stub to consume.
            cfg = {"telegram_token": "fake", "telegram_chat_id": 99}
            handle = bs.start("telegram", cfg)
            # Surface the stop_event to the stub.
            handle.config["_test_stop"] = handle.stop_event
            self.assertTrue(handle.is_alive())
            self.assertEqual(handle.kind, "telegram")
            # bridges row inserted.
            rows = sqlite3.connect(str(self._db_path)).execute(
                "SELECT kind, enabled FROM bridges").fetchall()
            self.assertEqual(rows, [("telegram", 1)])

            # stop() joins the thread and flips enabled=0.
            self.assertTrue(bs.stop("telegram", timeout_s=3.0))
            rows = sqlite3.connect(str(self._db_path)).execute(
                "SELECT kind, enabled FROM bridges").fetchall()
            self.assertEqual(rows, [("telegram", 0)])

    def test_double_start_raises(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"

        ev = threading.Event()
        with patch("bridges.telegram._tg_supervisor",
                   side_effect=lambda *a, **kw: ev.wait()):
            try:
                bs.start("telegram", {"telegram_token": "fake",
                                      "telegram_chat_id": 1})
                with self.assertRaises(RuntimeError):
                    bs.start("telegram", {"telegram_token": "fake",
                                          "telegram_chat_id": 1})
            finally:
                ev.set()
                bs.stop("telegram", timeout_s=3.0)

    def test_stop_unknown_returns_false(self):
        from cc_daemon import bridge_supervisor as bs
        self.assertFalse(bs.stop("telegram"))


# ── 3. Notify (outbound mailbox) ───────────────────────────────────────────


class TestNotify(_BridgeTestBase):

    def test_notify_no_bridge_returns_false(self):
        from cc_daemon import bridge_supervisor as bs
        self.assertFalse(bs.notify("telegram", "hello"))

    def test_notify_calls_sender(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"

        sent: list[tuple[dict, str]] = []
        def fake_sender(cfg, text):
            sent.append((cfg, text))
            return True

        with patch("bridges.telegram._tg_supervisor",
                   side_effect=lambda *a, **kw: threading.Event().wait(0.5)):
            handle = bs.start("telegram", {"telegram_token": "x",
                                            "telegram_chat_id": 5})
            handle.sender = fake_sender   # swap real sender for the test
            try:
                self.assertTrue(bs.notify("telegram", "hello"))
                self.assertEqual(len(sent), 1)
                self.assertEqual(sent[0][1], "hello")
            finally:
                bs.stop("telegram", timeout_s=3.0)

    def test_notify_empty_text_is_dropped(self):
        from cc_daemon import bridge_supervisor as bs
        self.assertFalse(bs.notify("telegram", ""))

    def test_notify_broadcast_delivers_to_every_live_bridge(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"
        os.environ["CHEETAHCLAWS_ENABLE_F7"] = "1"

        seen: dict[str, list[str]] = {"telegram": [], "slack": []}

        def tg_sender(cfg, text):
            seen["telegram"].append(text); return True
        def sl_sender(cfg, text):
            seen["slack"].append(text); return True

        with patch("bridges.telegram._tg_supervisor",
                   side_effect=lambda *a, **kw: threading.Event().wait(0.5)), \
             patch("bridges.slack._slack_supervisor",
                   side_effect=lambda *a, **kw: threading.Event().wait(0.5)):
            tg = bs.start("telegram", {"telegram_token": "t",
                                        "telegram_chat_id": 1})
            sl = bs.start("slack", {"slack_token": "s",
                                     "slack_channel": "c"})
            tg.sender = tg_sender
            sl.sender = sl_sender
            try:
                self.assertTrue(bs.notify("*", "ping"))
                self.assertEqual(seen["telegram"], ["ping"])
                self.assertEqual(seen["slack"], ["ping"])
            finally:
                bs.stop_all(timeout_s=3.0)


# ── 4. SQLite persistence ──────────────────────────────────────────────────


class TestSqlitePersistence(_BridgeTestBase):

    def test_list_persisted_after_stop(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"

        ev = threading.Event()
        with patch("bridges.telegram._tg_supervisor",
                   side_effect=lambda *a, **kw: ev.wait()):
            bs.start("telegram", {"telegram_token": "abcdef",
                                   "telegram_chat_id": 7})
            ev.set()
            bs.stop("telegram", timeout_s=3.0)

        rows = bs.list_persisted()
        kinds = [r["kind"] for r in rows]
        self.assertIn("telegram", kinds)
        row = next(r for r in rows if r["kind"] == "telegram")
        self.assertFalse(row["enabled"])
        # Token redacted.
        cfg = row["config"]
        self.assertNotIn("abcdef", json.dumps(cfg))

    def test_db_failure_does_not_raise(self):
        """A broken SQLite handle must not prevent start/stop from
        running — bridges work degraded but don't take down the daemon."""
        from cc_daemon import bridge_supervisor as bs

        class _Handle:
            kind = "telegram"
            config = {"telegram_token": "t", "telegram_chat_id": 1}
            last_error = ""
        # Pass a partial handle so we exercise just the helper.
        with patch("cc_daemon.schema.get_conn",
                   side_effect=sqlite3.OperationalError("forced")):
            self.assertFalse(bs._db_upsert_bridge(_Handle(), enabled_flag=True))
            self.assertFalse(bs._db_finalize_bridge(_Handle()))
            self.assertEqual(bs.list_persisted(), [])


# ── 5. Config redaction ────────────────────────────────────────────────────


class TestConfigRedaction(unittest.TestCase):

    def test_token_redacted_chat_id_kept(self):
        from cc_daemon import bridge_supervisor as bs
        safe = bs._safe_cfg({
            "telegram_token":   "1234567890:abcdefghijklmnop",
            "telegram_chat_id": 99,
            "log_level":        "info",
        })
        self.assertEqual(safe["telegram_chat_id"], 99)
        self.assertEqual(safe["log_level"], "info")
        self.assertTrue(safe["telegram_token"].startswith("***"))
        # Only the last 4 of the token survive.
        self.assertTrue(safe["telegram_token"].endswith("mnop"))

    def test_provider_api_keys_also_redacted(self):
        """bridge.start merges daemon_state.config into the per-bridge
        config, so provider secrets (``anthropic_api_key`` /
        ``openai_api_key`` / ``password`` / ``*_secret`` / ``auth_*``)
        must also be redacted before they hit the bus or the bridges
        SQLite row."""
        from cc_daemon import bridge_supervisor as bs
        safe = bs._safe_cfg({
            "anthropic_api_key": "sk-ant-aaaabbbbccccdddd",
            "openai_api_key":    "sk-proj-eeeeffffgggghhhh",
            "user_password":     "hunter2-but-long",
            "client_secret":     "shhhhh",
            "auth_header":       "Bearer abc.def.ghi",
            "model":             "claude-opus-4-7",
        })
        # Provider keys redacted.
        self.assertTrue(safe["anthropic_api_key"].startswith("***"))
        self.assertTrue(safe["anthropic_api_key"].endswith("dddd"))
        self.assertTrue(safe["openai_api_key"].startswith("***"))
        self.assertTrue(safe["user_password"].startswith("***"))
        self.assertTrue(safe["client_secret"].startswith("***"))
        self.assertTrue(safe["auth_header"].startswith("***"))
        # Plain text non-secret stays intact.
        self.assertEqual(safe["model"], "claude-opus-4-7")


# ── 6. F-7 Slack-specific worker wiring ────────────────────────────────────


class TestSlackWorker(_BridgeTestBase):
    """F-7: same supervisor scaffolding, slack-specific imports + sender."""

    def test_slack_requires_f6_flag_too(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F7"] = "1"
        # No F-6 flag → bridge_supervisor.start refuses.
        with self.assertRaises(RuntimeError) as ctx:
            bs.start("slack", {"slack_token": "x", "slack_channel": "c"})
        self.assertIn("F-6", str(ctx.exception))

    def test_slack_worker_calls_slack_supervisor(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"
        os.environ["CHEETAHCLAWS_ENABLE_F7"] = "1"

        called: list[tuple] = []
        ev = threading.Event()
        def fake_supervisor(token, channel, config):
            called.append((token, channel))
            ev.wait()

        with patch("bridges.slack._slack_supervisor",
                   side_effect=fake_supervisor):
            handle = bs.start("slack", {"slack_token": "sl-tok",
                                         "slack_channel": "general"})
            # Give worker thread a beat to invoke the supervisor.
            time.sleep(0.1)
            self.assertEqual(called, [("sl-tok", "general")])
            ev.set()
            bs.stop("slack", timeout_s=3.0)

    def test_slack_sender_dispatches_outbound(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"
        os.environ["CHEETAHCLAWS_ENABLE_F7"] = "1"

        sent: list[tuple] = []
        ev = threading.Event()
        with patch("bridges.slack._slack_supervisor",
                   side_effect=lambda *a, **kw: ev.wait()), \
             patch("bridges.slack._slack_send",
                   side_effect=lambda tok, chan, text: sent.append(
                       (tok, chan, text))):
            handle = bs.start("slack", {"slack_token": "tok",
                                         "slack_channel": "ops"})
            try:
                self.assertTrue(bs.notify("slack", "halo"))
                self.assertEqual(sent, [("tok", "ops", "halo")])
            finally:
                ev.set()
                bs.stop("slack", timeout_s=3.0)


# ── 7. F-8 WeChat-specific worker wiring ───────────────────────────────────


class TestWechatWorker(_BridgeTestBase):
    """F-8: same supervisor scaffolding, wechat-specific imports + sender.
    WeChat needs token + base_url already set up by `_wx_start_bridge`'s
    QR-login path; the worker surfaces a clear error if either is missing."""

    def test_wechat_requires_f6_flag_too(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F8"] = "1"
        with self.assertRaises(RuntimeError) as ctx:
            bs.start("wechat", {"wechat_token": "x", "wechat_base_url": "u"})
        self.assertIn("F-6", str(ctx.exception))

    def test_wechat_worker_calls_wx_supervisor(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"
        os.environ["CHEETAHCLAWS_ENABLE_F8"] = "1"

        called: list[tuple] = []
        ev = threading.Event()
        def fake_supervisor(token, base_url, config):
            called.append((token, base_url))
            ev.wait()

        with patch("bridges.wechat._wx_supervisor", side_effect=fake_supervisor):
            handle = bs.start("wechat", {
                "wechat_token":    "wc-tok",
                "wechat_base_url": "http://localhost:1234",
            })
            time.sleep(0.1)
            self.assertEqual(called, [("wc-tok", "http://localhost:1234")])
            ev.set()
            bs.stop("wechat", timeout_s=3.0)

    def test_wechat_worker_reports_missing_config(self):
        """If a caller starts WeChat without populating the token /
        base_url (typical when /wechat login hasn't run), the worker
        exits cleanly with a clear last_error rather than blowing up
        deep inside _wx_supervisor's first HTTP call."""
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"
        os.environ["CHEETAHCLAWS_ENABLE_F8"] = "1"

        # Don't patch _wx_supervisor — the worker should never reach it.
        handle = bs.start("wechat", {})    # both fields missing
        # Wait briefly for the worker to log the error and exit.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not handle.thread.is_alive():
                break
            time.sleep(0.05)
        self.assertFalse(handle.thread.is_alive())
        self.assertIn("wechat config missing", handle.last_error)

    def test_wechat_sender_dispatches_outbound(self):
        from cc_daemon import bridge_supervisor as bs
        os.environ["CHEETAHCLAWS_ENABLE_F6"] = "1"
        os.environ["CHEETAHCLAWS_ENABLE_F8"] = "1"

        sent: list = []
        ev = threading.Event()
        with patch("bridges.wechat._wx_supervisor",
                   side_effect=lambda *a, **kw: ev.wait()), \
             patch("bridges.wechat._wx_send",
                   side_effect=lambda user_id, text, cfg: sent.append(
                       (user_id, text))):
            handle = bs.start("wechat", {
                "wechat_token":    "t",
                "wechat_base_url": "u",
                "wechat_user_id":  "user-42",
            })
            try:
                self.assertTrue(bs.notify("wechat", "nihao"))
                self.assertEqual(sent, [("user-42", "nihao")])
            finally:
                ev.set()
                bs.stop("wechat", timeout_s=3.0)


if __name__ == "__main__":
    unittest.main()
