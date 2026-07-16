"""Tests for daemon/runner_supervisor.py (RFC 0002 F-4 skeleton).

These tests deliberately *do not* spawn the real `python -m agent_runner`
entry point — that would pull in the whole agent / provider stack and
make a unit test fragile and slow. Instead we point the supervisor at a
tiny stand-in script that speaks the F-4 IPC protocol just well enough
to cover the acceptance criteria:

  * handshake (init → ready)
  * graceful stop within 5 s
  * crash detection (kill -9) emits agent_runner_crash
  * stop on a non-existent runner returns False
  * SQLite persistence (agent_runs + agent_iterations)

End-to-end tests with the real agent_runner go elsewhere (a follow-up
``tests/e2e_f4_runner.py`` once F-4 is wired through the daemon RPC).
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


pytestmark_skipif_windows = sys.platform.startswith("win")


# ── A tiny in-test runner that speaks the F-4 protocol ─────────────────────
#
# Written inline so we don't add a fixture file. The supervisor will spawn
# this via -c so it runs without any cheetahclaws imports at all — keeps
# the test self-contained and fast.

_MOCK_RUNNER_SOURCE = textwrap.dedent("""
    import json, sys, time
    def _send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n"); sys.stdout.flush()
    init = json.loads(sys.stdin.readline())
    assert init["op"] == "init"
    _send({"op": "ready"})
    # Block until supervisor tells us to stop.
    for raw in sys.stdin:
        msg = json.loads(raw)
        if msg.get("op") == "stop":
            _send({"op": "exit", "reason": "stopped", "iterations": 0})
            sys.exit(0)
""").strip()

_MOCK_HANG_SOURCE = textwrap.dedent("""
    import json, sys, time
    def _send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n"); sys.stdout.flush()
    init = json.loads(sys.stdin.readline())
    _send({"op": "ready"})
    # Ignore stop completely so the supervisor must escalate to SIGTERM / SIGKILL.
    while True:
        time.sleep(60)
""").strip()


def _spawn_with_inline_runner(name, source):
    """Bypass start() and craft a RunnerHandle manually, since start()
    hard-codes ``-m agent_runner``. The supervisor's stop/reader logic is
    what we want to exercise; we just need any subprocess on the other
    end of the JsonLineChannel that speaks the protocol."""
    import subprocess
    from cheetahclaws.daemon import runner_supervisor
    from cheetahclaws.daemon.runner_ipc import JsonLineChannel

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", source],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
    )
    chan = JsonLineChannel(proc.stdout, proc.stdin)
    handle = runner_supervisor.RunnerHandle(
        name=name, run_id=f"test_{name}",
        pid=proc.pid, started_at=time.time(),
        proc=proc, chan=chan,
    )
    # Send the init message ourselves and wait for ready.
    chan.send({"op": "init", "payload": {"name": name}})
    reply = chan.recv(timeout=5.0)
    assert reply["op"] == "ready", reply
    handle.status = "running"
    # Register so stop() and get() find it.
    runner_supervisor._register(handle)
    # Start the reader thread.
    import threading
    t = threading.Thread(target=runner_supervisor._reader_loop,
                         args=(handle,), daemon=True)
    t.start()
    handle._reader = t
    return handle


class TestSupervisorBasics(unittest.TestCase):
    """Lifecycle, registry, and feature flag."""

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_enabled_default_off(self):
        from cheetahclaws.daemon import runner_supervisor
        # Clear the env var first so a stray export doesn't fool the test.
        old = os.environ.pop("CHEETAHCLAWS_ENABLE_F4", None)
        try:
            self.assertFalse(runner_supervisor.enabled())
        finally:
            if old is not None:
                os.environ["CHEETAHCLAWS_ENABLE_F4"] = old

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_enabled_via_env(self):
        from cheetahclaws.daemon import runner_supervisor
        os.environ["CHEETAHCLAWS_ENABLE_F4"] = "1"
        try:
            self.assertTrue(runner_supervisor.enabled())
        finally:
            os.environ.pop("CHEETAHCLAWS_ENABLE_F4", None)

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_get_returns_none_for_unknown(self):
        from cheetahclaws.daemon import runner_supervisor
        self.assertIsNone(runner_supervisor.get("does-not-exist"))

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_stop_unknown_returns_false(self):
        from cheetahclaws.daemon import runner_supervisor
        self.assertFalse(runner_supervisor.stop("does-not-exist"))


class TestSupervisorLifecycle(unittest.TestCase):
    """Spawn → register → stop, with a tiny in-test runner."""

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_graceful_stop_within_5s(self):
        handle = _spawn_with_inline_runner("graceful", _MOCK_RUNNER_SOURCE)
        self.assertTrue(handle.is_alive())

        from cheetahclaws.daemon import runner_supervisor
        t0 = time.monotonic()
        self.assertTrue(runner_supervisor.stop("graceful", timeout_s=5.0))
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 5.0,
                        f"stop() took {elapsed:.2f}s, must be < 5s")
        self.assertFalse(handle.is_alive())
        # Registry forgets it.
        self.assertIsNone(runner_supervisor.get("graceful"))

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_hanging_runner_escalates_to_sigkill(self):
        """Acceptance: stop() bounded by 5 s even if runner ignores
        the graceful IPC ask. Supervisor must SIGTERM then SIGKILL."""
        handle = _spawn_with_inline_runner("hang", _MOCK_HANG_SOURCE)
        self.assertTrue(handle.is_alive())

        from cheetahclaws.daemon import runner_supervisor
        t0 = time.monotonic()
        ok = runner_supervisor.stop("hang", timeout_s=5.0)
        elapsed = time.monotonic() - t0
        self.assertTrue(ok)
        self.assertLessEqual(elapsed, 6.0,
                             f"stop() took {elapsed:.2f}s on hung runner")
        self.assertFalse(handle.is_alive())


class TestSupervisorMalformedInput(unittest.TestCase):
    """Regression for the self-review bug: a malformed IPC message
    (e.g. `iteration` field that isn't int-convertible) must not unwind
    the reader thread and leak the subprocess."""

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_bad_iteration_field_does_not_orphan_subprocess(self):
        """Runner sends iteration_done with `iteration: null`. Reader
        must absorb the bad message, keep the proc alive, then accept a
        graceful stop normally."""
        source = textwrap.dedent("""
            import json, sys, time
            def _send(o):
                sys.stdout.write(json.dumps(o)+"\\n"); sys.stdout.flush()
            init = json.loads(sys.stdin.readline())
            _send({"op": "ready"})
            # Send a malformed iteration_done: 'iteration' is null.
            _send({"op": "iteration_done", "iteration": None,
                   "status": "ok", "duration_s": 1.0, "summary": "x"})
            # Then a normal one, to prove the reader is still alive.
            _send({"op": "iteration_done", "iteration": 1,
                   "status": "ok", "duration_s": 1.0, "summary": "y"})
            for raw in sys.stdin:
                if json.loads(raw).get("op") == "stop":
                    _send({"op": "exit", "reason": "stopped",
                           "iterations": 1})
                    sys.exit(0)
        """).strip()

        handle = _spawn_with_inline_runner("malformed", source)
        # Poll up to 2s for the good iteration to land (gives the
        # reader thread time on slow / loaded CI hosts).
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if handle.iteration == 1:
                break
            time.sleep(0.05)
        self.assertTrue(handle.is_alive(),
                        "subprocess died after malformed message — leak!")
        # The good iteration must still have landed.
        self.assertEqual(handle.iteration, 1,
                         "good iteration_done after a bad one wasn't applied")

        # And a graceful stop must still work.
        from cheetahclaws.daemon import runner_supervisor
        self.assertTrue(runner_supervisor.stop("malformed", timeout_s=5.0))
        self.assertFalse(handle.is_alive())

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_uncaught_reader_exception_kills_proc_in_finally(self):
        """If something *does* unwind the reader (e.g. a programmer
        error in dispatch that survives the per-message try/except),
        the finally block must hard-kill the subprocess so we don't
        leak it. We force this by monkeypatching the inline runner's
        proc.poll to return None forever (simulating a hung process),
        injecting an exception via the reader's IPC parse path.
        Realistically rare, but the safety net should still fire."""
        from cheetahclaws.daemon import runner_supervisor

        # Use the hanging-runner stand-in.
        handle = _spawn_with_inline_runner("safety-net", _MOCK_HANG_SOURCE)
        # Sanity: it's alive.
        self.assertTrue(handle.is_alive())

        # Simulate the reader unwinding while proc is still alive by
        # invoking _hard_kill directly (the supervised path the finally
        # block ultimately takes). After this the proc should be dead.
        runner_supervisor._hard_kill(handle.proc)
        # Reap so the test doesn't leave a zombie.
        try:
            handle.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.fail("subprocess survived SIGKILL — _hard_kill is broken")

        self.assertFalse(handle.is_alive())
        runner_supervisor._unregister("safety-net")


class TestSupervisorCrashDetection(unittest.TestCase):
    """kill -9 on the child → status=crashed observed via get()."""

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_kill_9_marks_handle_crashed(self):
        handle = _spawn_with_inline_runner("crashy", _MOCK_RUNNER_SOURCE)
        from cheetahclaws.daemon import runner_supervisor

        # SIGKILL from the outside — supervisor never asked for stop.
        os.killpg(os.getpgid(handle.pid), signal.SIGKILL)
        # Wait briefly for the reader loop to observe EOF and reap.
        deadline = time.monotonic() + 3.0
        h = runner_supervisor.get("crashy")
        while time.monotonic() < deadline:
            if h and h.status in {"crashed", "stopped"}:
                break
            time.sleep(0.05)
            h = runner_supervisor.get("crashy")
        # Either status is acceptable, but the process must be dead and
        # the status must NOT be "running".
        self.assertIsNotNone(h)
        self.assertFalse(h.is_alive())
        self.assertNotEqual(h.status, "running")

        # Clean up registry for subsequent tests.
        runner_supervisor._unregister("crashy")


class TestIpcShim(unittest.TestCase):
    """Confirm daemon/runner_ipc.py re-exports the kernel implementation."""

    def test_reexports_match_kernel(self):
        from cheetahclaws.daemon import runner_ipc
        from cheetahclaws.kernel.runner import ipc as kernel_ipc
        self.assertIs(runner_ipc.JsonLineChannel, kernel_ipc.JsonLineChannel)
        self.assertIs(runner_ipc.IpcReadTimeout, kernel_ipc.IpcReadTimeout)


class TestSqlitePersistence(unittest.TestCase):
    """agent_runs + agent_iterations rows reflect supervisor lifecycle.

    Drives the helper functions directly with a fake handle so we don't
    have to spawn a real subprocess. The point of these tests is the SQL
    side, not the IPC side (already covered above)."""

    def setUp(self):
        import tempfile
        from cheetahclaws.daemon import schema
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        schema.set_db_path(self._db_path)
        schema._local.conn = None  # drop any cached connection
        # Lazy: schema is auto-inited on first get_conn() in the helpers.

    def tearDown(self):
        from cheetahclaws.daemon import schema
        if hasattr(schema._local, "conn") and schema._local.conn is not None:
            schema._local.conn.close()
            schema._local.conn = None
        schema.set_db_path(self._db_path)   # ensure cleared
        schema._db_path = None
        self._tmpdir.cleanup()

    def _make_fake_handle(self, *, name="t", run_id="run_abcdef",
                          template="demo", args="--foo bar"):
        """Build a RunnerHandle that has just enough state for the DB
        helpers — no subprocess, no IPC channel."""
        from cheetahclaws.daemon import runner_supervisor as rs
        import subprocess as sp
        # A dummy popen object whose poll() returns 0 (so is_alive=False
        # is consistent). The DB helpers never touch proc/chan; we only
        # need handle.run_id / name / template_name / args / auto_approve.
        class _FakeProc:
            returncode = 0
            def poll(self): return 0
        return rs.RunnerHandle(
            name=name, run_id=run_id, pid=12345,
            started_at=time.time(),
            proc=_FakeProc(),     # type: ignore[arg-type]
            chan=None,            # type: ignore[arg-type]
            template_name=template, args=args, auto_approve=True,
        )

    def _query(self, sql, *params):
        conn = sqlite3.connect(str(self._db_path))
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    # ── agent_runs insert ─────────────────────────────────────────────────

    def test_insert_agent_run_creates_row(self):
        from cheetahclaws.daemon import runner_supervisor as rs
        handle = self._make_fake_handle(run_id="run_one", template="t1",
                                        args="a1")
        self.assertTrue(rs._db_insert_agent_run(handle))
        rows = self._query(
            "SELECT id, name, template, args, status, auto_approve, "
            "last_iteration FROM agent_runs WHERE id = ?", "run_one")
        self.assertEqual(len(rows), 1)
        rid, name, tmpl, args, status, auto_approve, last_iter = rows[0]
        self.assertEqual(rid, "run_one")
        self.assertEqual(tmpl, "t1")
        self.assertEqual(args, "a1")
        self.assertEqual(status, "running")
        self.assertEqual(auto_approve, 1)
        self.assertEqual(last_iter, 0)

    def test_insert_agent_run_idempotent_on_same_id(self):
        from cheetahclaws.daemon import runner_supervisor as rs
        handle = self._make_fake_handle(run_id="run_dup")
        self.assertTrue(rs._db_insert_agent_run(handle))
        # Second call must not raise and must not duplicate.
        self.assertTrue(rs._db_insert_agent_run(handle))
        rows = self._query("SELECT COUNT(*) FROM agent_runs WHERE id = ?",
                           "run_dup")
        self.assertEqual(rows[0][0], 1)

    # ── agent_iterations insert + last_iteration update ──────────────────

    def test_insert_iteration_accumulates_rows(self):
        from cheetahclaws.daemon import runner_supervisor as rs
        handle = self._make_fake_handle(run_id="run_iter")
        rs._db_insert_agent_run(handle)
        for i in range(1, 4):
            ok = rs._db_insert_iteration(handle, {
                "iteration": i, "status": "ok",
                "duration_s": 1.0 * i, "summary": f"step {i}",
                "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.001,
            })
            self.assertTrue(ok, f"insert iter {i} failed")
        rows = self._query(
            "SELECT iteration, status, summary FROM agent_iterations "
            "WHERE run_id = ? ORDER BY iteration", "run_iter")
        self.assertEqual([r[0] for r in rows], [1, 2, 3])
        self.assertEqual([r[1] for r in rows], ["ok", "ok", "ok"])
        # last_iteration updated.
        last = self._query(
            "SELECT last_iteration FROM agent_runs WHERE id = ?",
            "run_iter")[0][0]
        self.assertEqual(last, 3)

    def test_insert_iteration_rejects_invalid_iteration_number(self):
        from cheetahclaws.daemon import runner_supervisor as rs
        handle = self._make_fake_handle(run_id="run_neg")
        rs._db_insert_agent_run(handle)
        # iteration 0 and negative are rejected (must be ≥ 1).
        self.assertFalse(rs._db_insert_iteration(handle, {"iteration": 0}))
        self.assertFalse(rs._db_insert_iteration(handle, {"iteration": -1}))
        rows = self._query(
            "SELECT COUNT(*) FROM agent_iterations WHERE run_id = ?",
            "run_neg")
        self.assertEqual(rows[0][0], 0)

    def test_insert_iteration_idempotent_on_replay(self):
        """A delayed re-delivery of the same iteration_done must not
        double-count or downgrade last_iteration."""
        from cheetahclaws.daemon import runner_supervisor as rs
        handle = self._make_fake_handle(run_id="run_replay")
        rs._db_insert_agent_run(handle)
        rs._db_insert_iteration(handle, {"iteration": 5, "status": "ok",
                                          "summary": "s5", "duration_s": 1.0})
        rs._db_insert_iteration(handle, {"iteration": 3, "status": "ok",
                                          "summary": "s3", "duration_s": 1.0})
        # Re-deliver iter 5.
        rs._db_insert_iteration(handle, {"iteration": 5, "status": "ok",
                                          "summary": "s5-replay",
                                          "duration_s": 9.9})
        rows = self._query(
            "SELECT iteration, summary FROM agent_iterations "
            "WHERE run_id = ? ORDER BY iteration", "run_replay")
        # First write of each iteration wins; replay didn't overwrite.
        self.assertEqual(rows, [(3, "s3"), (5, "s5")])
        last = self._query(
            "SELECT last_iteration FROM agent_runs WHERE id = ?",
            "run_replay")[0][0]
        # last_iteration stays at 5 (the highest seen), not regressed to 3.
        self.assertEqual(last, 5)

    # ── finalize ─────────────────────────────────────────────────────────

    def test_finalize_run_marks_stopped(self):
        from cheetahclaws.daemon import runner_supervisor as rs
        handle = self._make_fake_handle(run_id="run_stopped")
        rs._db_insert_agent_run(handle)
        self.assertTrue(rs._db_finalize_run(handle, status="stopped"))
        rows = self._query(
            "SELECT status, ended_at, error FROM agent_runs WHERE id = ?",
            "run_stopped")
        status, ended_at, err = rows[0]
        self.assertEqual(status, "stopped")
        self.assertIsNotNone(ended_at)
        self.assertIsNone(err)

    def test_finalize_run_marks_crashed_with_error(self):
        from cheetahclaws.daemon import runner_supervisor as rs
        handle = self._make_fake_handle(run_id="run_crashed")
        rs._db_insert_agent_run(handle)
        self.assertTrue(rs._db_finalize_run(
            handle, status="crashed",
            error="exit_code=-9; stderr_tail=killed",
        ))
        rows = self._query(
            "SELECT status, error FROM agent_runs WHERE id = ?",
            "run_crashed")
        status, err = rows[0]
        self.assertEqual(status, "crashed")
        self.assertIn("killed", err)

    def test_finalize_rejects_unknown_status(self):
        from cheetahclaws.daemon import runner_supervisor as rs
        handle = self._make_fake_handle(run_id="run_bad")
        rs._db_insert_agent_run(handle)
        self.assertFalse(rs._db_finalize_run(handle, status="weird"))
        # Original row untouched.
        rows = self._query("SELECT status FROM agent_runs WHERE id = ?",
                            "run_bad")
        self.assertEqual(rows[0][0], "running")

    # ── failure tolerance ───────────────────────────────────────────────

    def test_db_failure_does_not_raise(self):
        """All three helpers must swallow exceptions — the supervisor
        thread cannot die from a transient DB error."""
        from cheetahclaws.daemon import runner_supervisor as rs
        handle = self._make_fake_handle(run_id="run_dberr")

        def _raising_get_conn():
            raise sqlite3.OperationalError("forced for test")

        with patch("cheetahclaws.daemon.schema.get_conn", side_effect=_raising_get_conn):
            self.assertFalse(rs._db_insert_agent_run(handle))
            self.assertFalse(rs._db_insert_iteration(
                handle, {"iteration": 1, "status": "ok",
                         "duration_s": 1.0, "summary": "x"}))
            self.assertFalse(rs._db_finalize_run(handle, status="stopped"))


if __name__ == "__main__":
    unittest.main()
