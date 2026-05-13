"""Tests for RFC 0002 F-5 — proactive watcher in daemon.

Covers three layers:

  1. :mod:`cc_daemon.proactive_state` — schema_meta KV persistence.
  2. :mod:`cc_daemon.proactive_scheduler` — the daemon-owned background
     thread that publishes ``proactive_tick`` events when idle.
  3. :mod:`cc_daemon.proactive_methods` — the ``proactive.*`` RPCs that
     external clients (REPL, Web UI) drive the scheduler with.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _isolate_schema():
    """Point the schema at a fresh tmp DB. Returns (tmpdir, db_path)."""
    from cc_daemon import schema
    tmp = tempfile.TemporaryDirectory()
    db = Path(tmp.name) / "test.db"
    schema.set_db_path(db)
    if hasattr(schema._local, "conn") and schema._local.conn is not None:
        schema._local.conn.close()
        schema._local.conn = None
    schema.init_schema(db)
    return tmp, db


def _restore_schema():
    from cc_daemon import schema
    if hasattr(schema._local, "conn") and schema._local.conn is not None:
        schema._local.conn.close()
        schema._local.conn = None
    schema._db_path = None


# ── proactive_state: persistence ───────────────────────────────────────────


class TestProactiveStateRoundTrip(unittest.TestCase):
    def setUp(self):
        self._tmp, self._db = _isolate_schema()

    def tearDown(self):
        _restore_schema()
        self._tmp.cleanup()

    def test_get_returns_defaults_on_empty_table(self):
        from cc_daemon import proactive_state
        enabled, iv, last = proactive_state.get_state()
        self.assertFalse(enabled)
        self.assertEqual(iv, proactive_state.DEFAULT_INTERVAL_S)
        self.assertEqual(last, 0.0)

    def test_set_state_persists_and_round_trips(self):
        from cc_daemon import proactive_state
        proactive_state.set_state(enabled=True, interval_s=120)
        enabled, iv, last = proactive_state.get_state()
        self.assertTrue(enabled)
        self.assertEqual(iv, 120)
        # set_state resets last_tick_at to ~now so subsequent ticks
        # honor the full interval.
        self.assertGreater(last, time.time() - 5)

    def test_set_state_rejects_zero_or_negative_interval(self):
        from cc_daemon import proactive_state
        with self.assertRaises(ValueError):
            proactive_state.set_state(enabled=True, interval_s=0)
        with self.assertRaises(ValueError):
            proactive_state.set_state(enabled=True, interval_s=-5)

    def test_disable_keeps_interval(self):
        from cc_daemon import proactive_state
        proactive_state.set_state(enabled=True, interval_s=999)
        proactive_state.disable()
        enabled, iv, _ = proactive_state.get_state()
        self.assertFalse(enabled)
        self.assertEqual(iv, 999)

    def test_tickle_bumps_last_tick_at(self):
        from cc_daemon import proactive_state
        proactive_state.set_state(enabled=True, interval_s=60)
        time.sleep(0.05)
        old = proactive_state.get_state()[2]
        proactive_state.tickle()
        new = proactive_state.get_state()[2]
        self.assertGreater(new, old)

    def test_corrupt_value_falls_back_to_defaults(self):
        """A malformed row from a prior buggy writer must not crash the
        scheduler — get_state() heals with defaults instead."""
        from cc_daemon import proactive_state, schema
        conn = schema.get_conn()
        # Manually plant garbage in the interval field.
        conn.execute(
            "INSERT INTO schema_meta (key, value, updated_at) "
            "VALUES (?, ?, datetime('now'))",
            ("proactive.interval_s", "not-a-number"),
        )
        conn.commit()
        enabled, iv, _ = proactive_state.get_state()
        self.assertFalse(enabled)
        self.assertEqual(iv, proactive_state.DEFAULT_INTERVAL_S)


# ── proactive_scheduler: thread behaviour ──────────────────────────────────


class TestProactiveScheduler(unittest.TestCase):
    def setUp(self):
        self._tmp, self._db = _isolate_schema()
        from cc_daemon import events
        events.reset_bus_for_tests()
        # Sanity: make sure no scheduler from a prior test is still alive.
        from cc_daemon import proactive_scheduler as ps
        ps.stop()

    def tearDown(self):
        from cc_daemon import proactive_scheduler as ps
        ps.stop()
        _restore_schema()
        self._tmp.cleanup()

    def test_disabled_state_does_not_publish(self):
        from cc_daemon import (
            events, proactive_state, proactive_scheduler as ps,
        )
        proactive_state.set_state(enabled=False, interval_s=1)
        # Subscribe BEFORE start so we don't miss an early tick.
        q = events.get_bus().subscribe()
        try:
            self.assertTrue(ps.start(owned_by_daemon=True))
            time.sleep(2.0)
        finally:
            events.get_bus().unsubscribe(q)
        # Drain and confirm no proactive_tick.
        ticks = []
        while not q.empty():
            ev = q.get_nowait()
            if ev.get("type") == "proactive_tick":
                ticks.append(ev)
        self.assertEqual(ticks, [])

    def test_enabled_publishes_after_idle_interval(self):
        from cc_daemon import (
            events, proactive_state, proactive_scheduler as ps,
        )
        # Subscribe BEFORE start so we don't miss the first tick.
        q = events.get_bus().subscribe()
        try:
            proactive_state.set_state(enabled=True, interval_s=1)
            # Predate last_tick_at so the scheduler considers the user
            # already idle and fires on its next tick.
            proactive_state.record_tick(time.time() - 5.0)
            self.assertTrue(ps.start(owned_by_daemon=True))
            deadline = time.monotonic() + 5.0
            tick_event = None
            while time.monotonic() < deadline:
                try:
                    ev = q.get(timeout=0.5)
                except Exception:
                    continue
                if ev.get("type") == "proactive_tick":
                    tick_event = ev
                    break
            self.assertIsNotNone(
                tick_event,
                "scheduler never published proactive_tick within 5 s",
            )
            data = tick_event.get("data") or {}
            self.assertEqual(data.get("interval_s"), 1)
            self.assertIn("last_tick_at", data)
            self.assertIn("fired_at", data)
        finally:
            events.get_bus().unsubscribe(q)

    def test_owned_by_daemon_disables_foreign_check(self):
        """The daemon's own scheduler must not defer to itself when it
        wrote the discovery file. We exercise this by:
          1. starting with owned_by_daemon=True
          2. monkey-patching discovery.locate to return a fake foreign pid
          3. confirming a tick still fires
        """
        from cc_daemon import (
            events, proactive_state, proactive_scheduler as ps, discovery,
        )
        proactive_state.set_state(enabled=True, interval_s=1)
        proactive_state.record_tick(time.time() - 5.0)
        q = events.get_bus().subscribe()

        orig = discovery.locate
        # Even though `discovery.locate` reports a foreign pid, the
        # scheduler must keep ticking because we own it.
        discovery.locate = lambda: {"pid": 999999, "address": "x:0"}
        try:
            self.assertTrue(ps.start(owned_by_daemon=True))
            deadline = time.monotonic() + 5.0
            tick = None
            while time.monotonic() < deadline:
                try:
                    ev = q.get(timeout=0.5)
                except Exception:
                    continue
                if ev.get("type") == "proactive_tick":
                    tick = ev
                    break
            self.assertIsNotNone(tick, "owned daemon mistakenly deferred")
        finally:
            discovery.locate = orig
            events.get_bus().unsubscribe(q)

    def test_stop_joins_within_5s(self):
        from cc_daemon import proactive_scheduler as ps
        ps.start(owned_by_daemon=True)
        t0 = time.monotonic()
        self.assertTrue(ps.stop())
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 5.0)
        self.assertFalse(ps.is_running())

    def test_double_start_returns_false(self):
        from cc_daemon import proactive_scheduler as ps
        try:
            self.assertTrue(ps.start(owned_by_daemon=True))
            self.assertFalse(ps.start(owned_by_daemon=True))
        finally:
            ps.stop()


# ── proactive_methods: RPC layer ───────────────────────────────────────────


class TestProactiveRpc(unittest.TestCase):
    """End-to-end through the RPC dispatcher (no HTTP)."""

    def setUp(self):
        self._tmp, self._db = _isolate_schema()
        from cc_daemon import events
        events.reset_bus_for_tests()

    def tearDown(self):
        from cc_daemon import proactive_scheduler as ps
        ps.stop()
        _restore_schema()
        self._tmp.cleanup()

    def _registry(self):
        from cc_daemon import proactive_methods
        from cc_daemon.rpc import RpcRegistry

        class _FakeState:
            config = {}

        reg = RpcRegistry()
        proactive_methods.register(reg, _FakeState())
        return reg

    def _call(self, reg, method, params=None):
        from cc_daemon.rpc import CallContext
        envelope = {"jsonrpc": "2.0", "id": 1, "method": method,
                    "params": params or {}}
        ctx = CallContext(client_id="t", transport="unix", api_version="0")
        response, _http = reg.dispatch(envelope, ctx)
        return response.get("result"), response.get("error")

    def test_set_then_get_round_trips(self):
        reg = self._registry()
        result, err = self._call(reg, "proactive.set",
                                 {"enabled": True, "interval_s": 60})
        self.assertIsNone(err)
        self.assertEqual(result["interval_s"], 60)
        self.assertTrue(result["enabled"])

        result2, err2 = self._call(reg, "proactive.get")
        self.assertIsNone(err2)
        self.assertTrue(result2["enabled"])
        self.assertEqual(result2["interval_s"], 60)
        self.assertIn("scheduler_running", result2)

    def test_set_requires_enabled_field(self):
        reg = self._registry()
        result, err = self._call(reg, "proactive.set", {"interval_s": 60})
        self.assertIsNone(result)
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)
        self.assertIn("enabled", err["message"])

    def test_set_rejects_non_int_interval(self):
        reg = self._registry()
        result, err = self._call(reg, "proactive.set",
                                 {"enabled": True, "interval_s": "abc"})
        self.assertIsNone(result)
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)

    def test_set_rejects_zero_interval(self):
        reg = self._registry()
        result, err = self._call(reg, "proactive.set",
                                 {"enabled": True, "interval_s": 0})
        self.assertIsNone(result)
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)

    def test_tickle_bumps_last_tick_at(self):
        from cc_daemon import proactive_state
        reg = self._registry()
        proactive_state.set_state(enabled=True, interval_s=300)
        time.sleep(0.05)
        result, err = self._call(reg, "proactive.tickle")
        self.assertIsNone(err)
        # New last_tick_at must be close to "now".
        self.assertGreater(result["last_tick_at"], time.time() - 5)

    def test_get_reports_scheduler_state(self):
        from cc_daemon import proactive_scheduler as ps
        reg = self._registry()
        try:
            ps.start(owned_by_daemon=True)
            result, err = self._call(reg, "proactive.get")
            self.assertIsNone(err)
            self.assertTrue(result["scheduler_running"])
        finally:
            ps.stop()
        result2, _ = self._call(reg, "proactive.get")
        self.assertFalse(result2["scheduler_running"])


# ── REPL step-aside ────────────────────────────────────────────────────────


class TestReplStepAside(unittest.TestCase):
    """``_proactive_watcher_loop`` must skip firing when a foreign daemon
    is registered. We exercise the foreign-daemon helper directly — the
    full loop is timing-sensitive and covered by integration testing."""

    def test_foreign_daemon_helper_returns_false_when_none(self):
        from cc_daemon import discovery
        orig = discovery.locate
        discovery.locate = lambda: None
        try:
            import importlib
            import cheetahclaws
            importlib.reload(cheetahclaws)
            self.assertFalse(cheetahclaws._proactive_foreign_daemon_running())
        finally:
            discovery.locate = orig

    def test_foreign_daemon_helper_returns_true_for_other_pid(self):
        from cc_daemon import discovery
        orig = discovery.locate
        discovery.locate = lambda: {"pid": 1, "address": "x:0"}
        try:
            import cheetahclaws
            self.assertTrue(cheetahclaws._proactive_foreign_daemon_running())
        finally:
            discovery.locate = orig

    def test_foreign_daemon_helper_returns_false_for_own_pid(self):
        from cc_daemon import discovery
        orig = discovery.locate
        discovery.locate = lambda: {"pid": os.getpid(), "address": "x:0"}
        try:
            import cheetahclaws
            self.assertFalse(cheetahclaws._proactive_foreign_daemon_running())
        finally:
            discovery.locate = orig


if __name__ == "__main__":
    unittest.main()
