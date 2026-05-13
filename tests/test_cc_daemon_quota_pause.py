"""End-to-end test for RFC 0002 §F-9 quota-pause / resume IPC roundtrip.

Verifies:
  1. A runner that ships ``{"op":"paused_budget", ...}`` makes the
     supervisor flip ``handle.status`` to ``paused_budget``, update
     SQLite, and publish ``quota_warn`` on the event bus.
  2. ``runner_supervisor.resume(name)`` sends ``{"op":"resume"}`` over
     IPC and returns True.
  3. After a follow-up ``{"op":"resumed"}`` from the runner the
     supervisor flips status back to ``running`` and publishes
     ``agent_runner_resumed``.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


pytestmark_skipif_windows = sys.platform.startswith("win")


# Runner that pauses, waits for `resume`, sends `resumed`, then waits for stop.
_PAUSE_RESUME_RUNNER = textwrap.dedent("""
    import json, sys
    def _send(o):
        sys.stdout.write(json.dumps(o)+"\\n"); sys.stdout.flush()
    init = json.loads(sys.stdin.readline())
    _send({"op": "ready"})
    _send({"op": "paused_budget", "reason": "session_token_budget reached"})
    # Block until supervisor sends resume.
    for raw in sys.stdin:
        m = json.loads(raw)
        if m.get("op") == "resume":
            _send({"op": "resumed"})
            break
    # Then loop until stop.
    for raw in sys.stdin:
        if json.loads(raw).get("op") == "stop":
            sys.exit(0)
""").strip()


class TestQuotaPauseIPC(unittest.TestCase):

    def setUp(self):
        from cc_daemon import schema
        from cc_daemon import runner_supervisor as rs
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        schema.set_db_path(self._db_path)
        schema._local.conn = None
        with rs._handles_lock:
            rs._handles.clear()

    def tearDown(self):
        from cc_daemon import schema
        from cc_daemon import runner_supervisor as rs
        for h in list(rs._handles.values()):
            try:
                h.proc.kill()
            except Exception:
                pass
        with rs._handles_lock:
            rs._handles.clear()
        if hasattr(schema._local, "conn") and schema._local.conn is not None:
            try:
                schema._local.conn.close()
            except Exception:
                pass
            schema._local.conn = None
        schema._db_path = None
        self._tmpdir.cleanup()

    def _spawn(self, name, source):
        from cc_daemon import runner_supervisor as rs
        from cc_daemon.runner_ipc import JsonLineChannel

        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
        )
        chan = JsonLineChannel(proc.stdout, proc.stdin)
        chan.send({"op": "init", "payload": {"name": name}})
        reply = chan.recv(timeout=5.0)
        assert reply["op"] == "ready"
        handle = rs.RunnerHandle(
            name=name, run_id=f"run_{name}", pid=proc.pid,
            started_at=time.time(), proc=proc, chan=chan,
            template_name="stub", args="",
        )
        handle.status = "running"
        rs._register(handle)
        rs._db_insert_agent_run(handle)
        t = threading.Thread(target=rs._reader_loop, args=(handle,), daemon=True)
        t.start()
        handle._reader = t
        return handle

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_paused_budget_then_resume_roundtrip(self):
        from cc_daemon import runner_supervisor as rs

        # Capture bus events so we can assert quota_warn fired.
        events: list[tuple[str, dict]] = []

        class _FakeBus:
            @staticmethod
            def publish(kind, payload):
                events.append((kind, payload))

        with patch.object(rs, "_get_event_bus", return_value=_FakeBus()):
            handle = self._spawn("paused", _PAUSE_RESUME_RUNNER)
            # Wait for the supervisor to consume paused_budget.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if handle.status == "paused_budget":
                    break
                time.sleep(0.05)
            self.assertEqual(handle.status, "paused_budget")
            self.assertIn("session_token_budget reached", handle.error)
            # SQLite reflects the pause. The reader-loop sets handle.status
            # *before* committing the DB update, so under load we may
            # observe status==paused_budget while the row still says
            # 'running'. Tolerate a short window.
            def _read_row():
                return sqlite3.connect(str(self._db_path)).execute(
                    "SELECT status, error FROM agent_runs WHERE id = ?",
                    (handle.run_id,)).fetchone()
            row = _read_row()
            db_deadline = time.monotonic() + 1.5
            while row[0] != "paused_budget" and time.monotonic() < db_deadline:
                time.sleep(0.05)
                row = _read_row()
            self.assertEqual(row[0], "paused_budget")
            self.assertIn("session_token_budget reached", row[1])
            kinds = [k for k, _ in events]
            self.assertIn("quota_warn", kinds)

            # Now resume.
            self.assertTrue(rs.resume("paused"))

            # Wait for the supervisor to receive `resumed`.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if handle.status == "running":
                    break
                time.sleep(0.05)
            self.assertEqual(handle.status, "running")
            self.assertEqual(handle.error, "")
            # Same DB-vs-handle race as above.
            row = _read_row()
            db_deadline = time.monotonic() + 1.5
            while row[0] != "running" and time.monotonic() < db_deadline:
                time.sleep(0.05)
                row = _read_row()
            self.assertEqual(row[0], "running")
            self.assertIsNone(row[1])

            # And agent_runner_resumed was published.
            kinds_after = [k for k, _ in events]
            self.assertIn("agent_runner_resumed", kinds_after)

            rs.stop("paused", timeout_s=3.0)

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_resume_unknown_runner_returns_false(self):
        from cc_daemon import runner_supervisor as rs
        self.assertFalse(rs.resume("no-such-runner"))


if __name__ == "__main__":
    unittest.main()
