"""Tests for cc_daemon/runner_supervisor.py restart policy (RFC 0002 F-4 #3).

Three layers of coverage:

  1. Pure-function tests for ``RestartPolicy.next_delay`` and
     ``RestartPolicy.from_params`` — fast, deterministic, no I/O.
  2. Reader-loop integration: a crashed inline runner trips the restart
     hook; we patch the supervisor's spawn factory to assert the
     respawn call was issued with the right kwargs and the new
     ``restart_count``.
  3. ``stop()`` cancels a pending restart timer so a deliberate
     shutdown beats a scheduled respawn.

The tests deliberately avoid spawning ``python -m agent_runner`` (slow,
pulls the whole agent stack). The supervisor's pre-existing inline
runner pattern (a ``-c`` subprocess that just speaks the protocol)
covers the IPC side; the restart machinery is exercised against a
``RestartPolicy`` + factory stub.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


pytestmark_skipif_windows = sys.platform.startswith("win")


# ── 1. Pure-function tests: RestartPolicy ─────────────────────────────────


class TestRestartPolicyPure(unittest.TestCase):
    """``next_delay`` and ``from_params`` — no I/O, no Timer."""

    def test_disabled_returns_none_regardless_of_count(self):
        from cc_daemon.runner_supervisor import RestartPolicy
        p = RestartPolicy.disabled()
        self.assertIsNone(p.next_delay(0))
        self.assertIsNone(p.next_delay(5))
        self.assertIsNone(p.next_delay(1_000_000))

    def test_mode_none_with_max_restarts_still_disabled(self):
        """A misconfigured caller that sets max_restarts but forgets
        mode='on-crash' must not silently auto-restart."""
        from cc_daemon.runner_supervisor import RestartPolicy
        p = RestartPolicy(mode="none", max_restarts=5)
        self.assertIsNone(p.next_delay(0))

    def test_exponential_backoff_capped(self):
        """Without jitter: 1, 2, 4, 8, … capped at cap_s."""
        from cc_daemon.runner_supervisor import RestartPolicy
        p = RestartPolicy(mode="on-crash", max_restarts=10,
                          backoff_base_s=1.0, backoff_cap_s=5.0,
                          backoff_jitter_s=0.0)
        self.assertEqual(p.next_delay(0), 1.0)
        self.assertEqual(p.next_delay(1), 2.0)
        self.assertEqual(p.next_delay(2), 4.0)
        # 8 would be the unclamped value; cap drops it to 5.
        self.assertEqual(p.next_delay(3), 5.0)
        self.assertEqual(p.next_delay(4), 5.0)

    def test_jitter_stays_within_bounds(self):
        """Jitter must never produce a negative or huge delay."""
        import random
        from cc_daemon.runner_supervisor import RestartPolicy
        p = RestartPolicy(mode="on-crash", max_restarts=100,
                          backoff_base_s=1.0, backoff_cap_s=2.0,
                          backoff_jitter_s=0.5)
        # Lock the RNG so the assertion is deterministic.
        random.seed(0)
        for c in range(50):
            d = p.next_delay(c)
            assert d is not None
            self.assertGreaterEqual(d, 0.0)
            # cap=2.0 + jitter=0.5 = 2.5 is the theoretical max
            self.assertLessEqual(d, 2.5)

    def test_exhausted_after_max_restarts(self):
        from cc_daemon.runner_supervisor import RestartPolicy
        p = RestartPolicy(mode="on-crash", max_restarts=3,
                          backoff_base_s=1.0, backoff_cap_s=10.0,
                          backoff_jitter_s=0.0)
        # restart_count is the count *already used*; after 3 attempts
        # the 4th call (count=3) returns None.
        self.assertIsNotNone(p.next_delay(0))
        self.assertIsNotNone(p.next_delay(1))
        self.assertIsNotNone(p.next_delay(2))
        self.assertIsNone(p.next_delay(3))

    def test_from_params_defaults(self):
        from cc_daemon.runner_supervisor import RestartPolicy
        p = RestartPolicy.from_params({})
        self.assertEqual(p.mode, "none")
        self.assertEqual(p.max_restarts, 0)

    def test_from_params_round_trip(self):
        from cc_daemon.runner_supervisor import RestartPolicy
        p = RestartPolicy.from_params({
            "restart_policy":   "on-crash",
            "max_restarts":     3,
            "backoff_base_s":   0.5,
            "backoff_cap_s":    8.0,
            "backoff_jitter_s": 0.1,
        })
        self.assertEqual(p.mode, "on-crash")
        self.assertEqual(p.max_restarts, 3)
        self.assertAlmostEqual(p.backoff_base_s, 0.5)
        self.assertAlmostEqual(p.backoff_cap_s, 8.0)
        self.assertAlmostEqual(p.backoff_jitter_s, 0.1)

    def test_from_params_rejects_bad_mode(self):
        from cc_daemon.runner_supervisor import RestartPolicy
        with self.assertRaises(TypeError):
            RestartPolicy.from_params({"restart_policy": "always"})

    def test_from_params_rejects_negative_max_restarts(self):
        from cc_daemon.runner_supervisor import RestartPolicy
        with self.assertRaises(TypeError):
            RestartPolicy.from_params({"restart_policy": "on-crash",
                                       "max_restarts": -1})

    def test_from_params_rejects_cap_below_base(self):
        """The footgun: cap=0.1 < base=1.0 would mean every backoff
        gets clamped down and the policy "feels" disabled.  Catch it
        at config time instead."""
        from cc_daemon.runner_supervisor import RestartPolicy
        with self.assertRaises(TypeError):
            RestartPolicy.from_params({
                "restart_policy": "on-crash", "max_restarts": 3,
                "backoff_base_s": 1.0, "backoff_cap_s": 0.1,
            })

    def test_from_params_rejects_non_numeric_backoff(self):
        from cc_daemon.runner_supervisor import RestartPolicy
        with self.assertRaises(TypeError):
            RestartPolicy.from_params({"restart_policy": "on-crash",
                                       "max_restarts": 1,
                                       "backoff_base_s": "soon"})


# ── 2. Restart hook ────────────────────────────────────────────────────────


# A subprocess source that handshakes, immediately exits 1 — minimum
# viable "crashed" runner.  Used by the integration tests so we have a
# real `_reader_loop` finally to drive.
_MOCK_CRASH_SOURCE = textwrap.dedent("""
    import json, sys
    init = json.loads(sys.stdin.readline())
    sys.stdout.write(json.dumps({"op":"ready"}) + "\\n"); sys.stdout.flush()
    # Exit non-zero so the supervisor classifies this as 'crashed'.
    sys.exit(1)
""").strip()


def _spawn_crashing_runner_with_policy(name, policy):
    """Build a RunnerHandle pointing at the inline crashing runner, with
    the given restart policy.  Bypasses ``start()`` (which hard-codes
    ``-m agent_runner``) but does everything ``start()`` does after the
    handshake: stores ``_start_kwargs``, calls ``_register``, and spawns
    the reader thread.
    """
    from cc_daemon import runner_supervisor as rs
    from cc_daemon.runner_ipc import JsonLineChannel

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _MOCK_CRASH_SOURCE],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
    )
    chan = JsonLineChannel(proc.stdout, proc.stdin)
    chan.send({"op": "init", "payload": {"name": name}})
    reply = chan.recv(timeout=5.0)
    assert reply["op"] == "ready", reply

    # The same kwargs ``start()`` would have stashed if it had been used.
    start_kwargs = {
        "name":             name,
        "template_name":    "stub-template",
        "args":             "--noop",
        "config":           {},
        "interval":         2.0,
        "auto_approve":     True,
        "python":           sys.executable,
        "originator":       "test-originator",
        "permission_store": None,
        "restart_policy":   policy,
    }
    handle = rs.RunnerHandle(
        name=name, run_id=f"run_{name}", pid=proc.pid,
        started_at=time.time(), proc=proc, chan=chan,
        template_name="stub-template", args="--noop",
        auto_approve=True, originator="test-originator",
        permission_store=None,
        restart_policy=policy, restart_count=0,
        _start_kwargs=start_kwargs,
    )
    handle.status = "running"
    rs._register(handle)
    t = threading.Thread(target=rs._reader_loop, args=(handle,), daemon=True)
    t.start()
    handle._reader = t
    return handle


class TestRestartHookIntegration(unittest.TestCase):
    """The reader's finally invokes ``_maybe_schedule_restart`` which
    arms a Timer.  We replace the spawn factory so the Timer-fire does
    not actually fork another subprocess — just records the call."""

    def setUp(self):
        from cc_daemon import runner_supervisor as rs
        # Wipe any handles left over from earlier tests (modules are
        # process-global; flakes here usually trace back to stragglers).
        with rs._handles_lock:
            rs._handles.clear()
        self._restore_spawner = rs._RESTART_SPAWNER

    def tearDown(self):
        from cc_daemon import runner_supervisor as rs
        rs._RESTART_SPAWNER = self._restore_spawner
        with rs._handles_lock:
            for h in list(rs._handles.values()):
                t = h._restart_timer
                if t is not None:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                try:
                    h.proc.kill()
                except Exception:
                    pass
            rs._handles.clear()

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_disabled_policy_does_not_schedule_restart(self):
        """Default policy: crash leaves status='crashed', no Timer."""
        from cc_daemon import runner_supervisor as rs
        spawn = MagicMock()
        rs._RESTART_SPAWNER = spawn

        handle = _spawn_crashing_runner_with_policy(
            "disabled", rs.RestartPolicy.disabled())

        # Wait for the reader to observe the crash and run its finally.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if handle.status in {"crashed", "stopped"}:
                break
            time.sleep(0.05)
        self.assertEqual(handle.status, "crashed")
        self.assertIsNone(handle._restart_timer)
        spawn.assert_not_called()

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_on_crash_schedules_restart(self):
        """mode='on-crash' with restarts left: Timer fires _RESTART_SPAWNER."""
        from cc_daemon import runner_supervisor as rs

        called: list[dict] = []
        def fake_spawn(**kwargs):
            called.append(kwargs)
            # Return a stub handle so _do_restart's publish call has fields.
            class _Stub:
                name = kwargs["name"]
                run_id = "run_restarted"
                pid = 99999
                restart_count = kwargs.get("_restart_count_carry", 0)
            return _Stub()
        rs._RESTART_SPAWNER = fake_spawn

        policy = rs.RestartPolicy(
            mode="on-crash", max_restarts=2,
            backoff_base_s=0.01, backoff_cap_s=0.05,
            backoff_jitter_s=0.0,
        )
        handle = _spawn_crashing_runner_with_policy("respawn", policy)

        # Wait for restart_decided + Timer fired.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if called:
                break
            time.sleep(0.02)
        self.assertEqual(len(called), 1)
        kwargs = called[0]
        # The restart must propagate the lineage's counter +1.
        self.assertEqual(kwargs["_restart_count_carry"], 1)
        # And re-use the original start kwargs.
        self.assertEqual(kwargs["name"], "respawn")
        self.assertEqual(kwargs["template_name"], "stub-template")
        self.assertEqual(kwargs["originator"], "test-originator")
        self.assertIs(kwargs["restart_policy"], policy)

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_max_restarts_exhausted_emits_event(self):
        """After max_restarts the lineage stops respawning and publishes
        agent_runner_restart_exhausted."""
        from cc_daemon import runner_supervisor as rs

        events: list[tuple[str, dict]] = []

        def fake_publish(kind, payload):
            events.append((kind, payload))

        class _FakeBus:
            publish = staticmethod(fake_publish)

        # max_restarts=1 — first crash schedules one restart, second crash
        # finds the policy exhausted.
        policy = rs.RestartPolicy(
            mode="on-crash", max_restarts=1,
            backoff_base_s=0.01, backoff_cap_s=0.05,
            backoff_jitter_s=0.0,
        )

        # Patch the event bus accessor for the duration of the test.
        with patch.object(rs, "_get_event_bus", return_value=_FakeBus()):
            # Synthesise a "first attempt already used" handle to drive the
            # policy directly — the integration with a second real crash is
            # covered upstream by test_on_crash_schedules_restart; here we
            # only need the exhaustion path.
            handle = _spawn_crashing_runner_with_policy("exhausted", policy)
            handle.restart_count = 1   # pretend one restart already happened

            # Run the scheduler directly so we don't have to wait for two
            # real subprocess crashes.
            rs._maybe_schedule_restart(handle)

            # Wait briefly to let any (incorrectly scheduled) Timer fire.
            time.sleep(0.2)

        # The hook must NOT have scheduled another Timer.
        self.assertIsNone(handle._restart_timer)
        kinds = [k for k, _ in events]
        self.assertIn("agent_runner_restart_exhausted", kinds)


# ── 3. stop() cancels pending restart ──────────────────────────────────────


class TestStopCancelsRestart(unittest.TestCase):

    def setUp(self):
        from cc_daemon import runner_supervisor as rs
        with rs._handles_lock:
            rs._handles.clear()
        self._restore_spawner = rs._RESTART_SPAWNER

    def tearDown(self):
        from cc_daemon import runner_supervisor as rs
        rs._RESTART_SPAWNER = self._restore_spawner
        with rs._handles_lock:
            for h in list(rs._handles.values()):
                t = h._restart_timer
                if t is not None:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                try:
                    h.proc.kill()
                except Exception:
                    pass
            rs._handles.clear()

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_stop_cancels_pending_restart_timer(self):
        """A stop() arriving while a restart Timer is armed must cancel
        the Timer and avoid a respawn."""
        from cc_daemon import runner_supervisor as rs

        spawned = threading.Event()
        def fake_spawn(**_kwargs):
            spawned.set()
            raise RuntimeError("should NOT have been called after stop()")
        rs._RESTART_SPAWNER = fake_spawn

        # Build a handle that's already exited (poll returns 0) and arm
        # a restart Timer with a long delay so stop() has time to cancel.
        class _ExitedProc:
            returncode = 1
            def poll(self): return 1
            def wait(self, timeout=None): return 1
            def terminate(self): pass
            def kill(self): pass
        policy = rs.RestartPolicy(
            mode="on-crash", max_restarts=3,
            backoff_base_s=2.0, backoff_cap_s=2.0,
            backoff_jitter_s=0.0,
        )
        handle = rs.RunnerHandle(
            name="cancelme", run_id="run_cancelme",
            pid=os.getpid(),
            started_at=time.time(),
            proc=_ExitedProc(),  # type: ignore[arg-type]
            chan=None,           # type: ignore[arg-type]
            template_name="stub", args="",
            restart_policy=policy, restart_count=0,
            _start_kwargs={
                "name": "cancelme", "template_name": "stub",
                "args": "", "config": {}, "interval": 2.0,
                "auto_approve": True, "python": sys.executable,
                "originator": "", "permission_store": None,
                "restart_policy": policy,
            },
        )
        rs._register(handle)
        rs._maybe_schedule_restart(handle)
        self.assertIsNotNone(handle._restart_timer)

        # Stop the lineage before the Timer fires.
        ok = rs.stop("cancelme", timeout_s=1.0)
        self.assertTrue(ok)
        self.assertIsNone(handle._restart_timer)

        # Wait past when the Timer *would* have fired; confirm it didn't.
        self.assertFalse(spawned.wait(timeout=2.5),
                         "restart Timer fired despite stop() cancellation")


# ── 4. SQLite snapshot ─────────────────────────────────────────────────────


class TestRestartHandleSerialisation(unittest.TestCase):
    """agent_methods._handle_to_dict surfaces restart_count/policy."""

    def test_handle_dict_includes_restart_fields(self):
        from cc_daemon import runner_supervisor as rs
        from cc_daemon.agent_methods import _handle_to_dict

        class _FakeProc:
            def poll(self): return 0
        policy = rs.RestartPolicy(mode="on-crash", max_restarts=4,
                                  backoff_base_s=2.0, backoff_cap_s=10.0,
                                  backoff_jitter_s=0.1)
        handle = rs.RunnerHandle(
            name="x", run_id="r1", pid=1, started_at=0.0,
            proc=_FakeProc(),  # type: ignore[arg-type]
            chan=None,         # type: ignore[arg-type]
            restart_policy=policy, restart_count=2,
        )
        d = _handle_to_dict(handle)
        self.assertEqual(d["restart_count"], 2)
        self.assertEqual(d["restart_policy"]["mode"], "on-crash")
        self.assertEqual(d["restart_policy"]["max_restarts"], 4)
        self.assertAlmostEqual(d["restart_policy"]["backoff_base_s"], 2.0)


class TestUnregisterIdentityGuard(unittest.TestCase):
    """``_unregister(name, expected=handle)`` must NOT pop the slot when
    the registry holds a different handle for the same name. Otherwise a
    Timer-fired restart that respawned the runner mid-stop would have
    its successor handle silently deleted, leaking the new subprocess
    (it would still be running but the supervisor would forget about it).
    """

    def setUp(self):
        from cc_daemon import runner_supervisor as rs
        with rs._handles_lock:
            rs._handles.clear()

    def tearDown(self):
        from cc_daemon import runner_supervisor as rs
        with rs._handles_lock:
            rs._handles.clear()

    def _fake_handle(self, name, run_id):
        from cc_daemon import runner_supervisor as rs
        class _FakeProc:
            def poll(self): return 0
        return rs.RunnerHandle(
            name=name, run_id=run_id, pid=1, started_at=0.0,
            proc=_FakeProc(),  # type: ignore[arg-type]
            chan=None,         # type: ignore[arg-type]
        )

    def test_unregister_with_expected_only_pops_matching_handle(self):
        from cc_daemon import runner_supervisor as rs
        old = self._fake_handle("foo", "run_old")
        new = self._fake_handle("foo", "run_new")

        rs._register(old)
        # Simulate the race: a respawn replaced the slot before stop()
        # got to its terminal _unregister.
        with rs._handles_lock:
            rs._handles["foo"] = new

        # stop()-style cleanup must NOT remove the new handle.
        rs._unregister("foo", expected=old)
        self.assertIs(rs._handles.get("foo"), new)

        # Without ``expected`` (legacy callers), unconditional pop.
        rs._unregister("foo")
        self.assertNotIn("foo", rs._handles)


if __name__ == "__main__":
    unittest.main()
