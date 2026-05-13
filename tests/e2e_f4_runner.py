"""End-to-end tests for RFC 0002 F-4 — real ``python -m agent_runner --pipe``.

These tests differ from ``tests/test_cc_daemon_runner_supervisor.py`` in one
important way: they spawn the **actual** ``python -m agent_runner --pipe``
entry point (not an inline `-c` subprocess) via ``runner_supervisor.start``.
That means they exercise:

  * the real handshake in :func:`agent_runner._pipe_main`
  * the real :class:`_PipeAgentRunner` (iteration_done emission,
    permission_request IPC override)
  * the real :class:`AgentRunner._run_loop` body
  * the real :mod:`cc_daemon.runner_supervisor` reader thread + SQLite
    persistence helpers in :mod:`cc_daemon.schema`

To stay hermetic — no LLM provider, no network — the runner subprocess
swaps in a scripted ``agent.run`` when ``CHEETAHCLAWS_E2E_FAKE_AGENT=1``
is set in its env. The hook lives in
``agent_runner._pipe_main`` and is gated by an env var so production
paths cannot reach it.

These tests are POSIX-only (matches F-4 supervisor scope) and tagged
e2e so they can be skipped on slow CI tiers if needed.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


pytestmark_skipif_windows = sys.platform.startswith("win")


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_template(tmp_dir: Path, name: str = "e2e_stub") -> str:
    """Write a tiny template file and return its **absolute path** so the
    runner's ``load_template`` doesn't have to consult Path.home()."""
    p = tmp_dir / f"{name}.md"
    p.write_text(
        "# e2e stub\n\nPlaceholder consumed by the stubbed agent.run().\n",
        encoding="utf-8",
    )
    return str(p)


def _isolate_schema(tmp_path: Path) -> Path:
    """Point :mod:`cc_daemon.schema` at a fresh DB under tmp_path so the
    test sees its own agent_runs / agent_iterations rows."""
    from cc_daemon import schema

    db = tmp_path / "sessions.db"
    schema.set_db_path(db)
    if hasattr(schema._local, "conn") and schema._local.conn is not None:
        schema._local.conn.close()
        schema._local.conn = None
    schema.init_schema(db)
    return db


def _restore_schema_default():
    from cc_daemon import schema

    if hasattr(schema._local, "conn") and schema._local.conn is not None:
        schema._local.conn.close()
        schema._local.conn = None
    schema._db_path = None


def _query(db_path: Path, sql: str, *params):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# ── Tests ──────────────────────────────────────────────────────────────────


@unittest.skipIf(pytestmark_skipif_windows, "F-4 is POSIX-only")
class TestF4EndToEndRealRunner(unittest.TestCase):
    """Spawn the real ``python -m agent_runner --pipe`` via the supervisor."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self._tmp_path = Path(self._tmp.name)
        self._db_path = _isolate_schema(self._tmp_path)
        # Force F-4 enabled + the agent stub so we don't need a provider.
        self._prev_env = {
            k: os.environ.get(k)
            for k in (
                "CHEETAHCLAWS_ENABLE_F4",
                "CHEETAHCLAWS_E2E_FAKE_AGENT",
                "CHEETAHCLAWS_E2E_FAKE_PERMISSION",
            )
        }
        os.environ["CHEETAHCLAWS_ENABLE_F4"] = "1"
        os.environ["CHEETAHCLAWS_E2E_FAKE_AGENT"] = "1"
        # Default off for permission emission — the perm-routing test
        # opts in.
        os.environ.pop("CHEETAHCLAWS_E2E_FAKE_PERMISSION", None)
        self._template = _make_template(self._tmp_path)

    def tearDown(self):
        # Stop any leftover runner from this test method.
        from cc_daemon import runner_supervisor as rs

        for h in list(rs.list_all()):
            try:
                rs.stop(h.name, timeout_s=3.0)
            except Exception:
                pass
        _restore_schema_default()
        for k, v in self._prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    # ── #1: SQLite agent_runs row created on start ────────────────────────

    def test_start_creates_agent_runs_row(self):
        from cc_daemon import runner_supervisor as rs

        handle = rs.start(
            name="e2e-row",
            template_name=self._template,
            args="",
            config={},
            interval=0.5,
            auto_approve=True,
        )
        try:
            # The supervisor calls _db_insert_agent_run synchronously
            # before returning, so the row must already be present.
            rows = _query(
                self._db_path,
                "SELECT id, name, status, auto_approve FROM agent_runs "
                "WHERE id = ?",
                handle.run_id,
            )
            self.assertEqual(len(rows), 1)
            rid, name, status, aa = rows[0]
            self.assertEqual(rid, handle.run_id)
            self.assertEqual(name, "e2e-row")
            self.assertEqual(status, "running")
            self.assertEqual(aa, 1)
        finally:
            rs.stop("e2e-row", timeout_s=5.0)

    # ── #2: iteration_done lands in agent_iterations + updates last_iter ──

    def test_iteration_lands_in_sqlite_under_real_runner(self):
        from cc_daemon import runner_supervisor as rs

        handle = rs.start(
            name="e2e-iter",
            template_name=self._template,
            args="",
            config={},
            # Short interval so the runner produces multiple iterations
            # within the test budget.
            interval=0.1,
            auto_approve=True,
        )
        try:
            # Wait up to 15 s for at least one iteration to land. Long
            # ceiling because subprocess startup + Python import on a
            # cold tmpfs can be slow on shared CI.
            deadline = time.monotonic() + 15.0
            rows: list = []
            while time.monotonic() < deadline:
                rows = _query(
                    self._db_path,
                    "SELECT iteration, status, summary FROM agent_iterations "
                    "WHERE run_id = ? ORDER BY iteration",
                    handle.run_id,
                )
                if rows:
                    break
                time.sleep(0.1)
            self.assertTrue(
                rows,
                "no agent_iterations row landed within 15 s — the real "
                "runner subprocess never reported iteration_done. "
                f"handle.status={handle.status!r} "
                f"stderr_tail={bytes(handle.stderr_tail)[-512:]!r}",
            )
            # First row should be iteration 1 (1-indexed) and status ok.
            self.assertEqual(rows[0][0], 1)
            self.assertIn(rows[0][1], ("ok", "permission"))
            # Summary contains some text from the stubbed agent.run().
            self.assertIn("e2e", rows[0][2].lower())
            # last_iteration on agent_runs must keep up with the highest
            # observed iteration.
            last = _query(
                self._db_path,
                "SELECT last_iteration FROM agent_runs WHERE id = ?",
                handle.run_id,
            )[0][0]
            self.assertGreaterEqual(last, 1)
        finally:
            rs.stop("e2e-iter", timeout_s=5.0)

    # ── #3: agent_runs status is finalised on graceful stop ───────────────

    def test_graceful_stop_finalises_agent_runs_status(self):
        from cc_daemon import runner_supervisor as rs

        handle = rs.start(
            name="e2e-final",
            template_name=self._template,
            args="",
            config={},
            interval=0.1,
            auto_approve=True,
        )
        # Give it a beat so at least one iteration completes.
        time.sleep(1.5)
        self.assertTrue(rs.stop("e2e-final", timeout_s=5.0))

        # Poll for finalisation — the reader thread's `finally` block
        # writes ``ended_at`` and flips status after proc.wait() returns,
        # which races slightly with our stop() return.
        deadline = time.monotonic() + 5.0
        row = None
        while time.monotonic() < deadline:
            rows = _query(
                self._db_path,
                "SELECT status, ended_at FROM agent_runs WHERE id = ?",
                handle.run_id,
            )
            if rows and rows[0][1] is not None:
                row = rows[0]
                break
            time.sleep(0.1)
        self.assertIsNotNone(row, "agent_runs never got an ended_at")
        status, ended_at = row
        self.assertEqual(status, "stopped")
        self.assertIsNotNone(ended_at)

    # ── #4: Full permission round-trip through real subprocess ────────────

    def test_real_runner_permission_routing_round_trip(self):
        """auto_approve=False + a real PermissionStore + stubbed
        PermissionRequest from the agent. Originator's answer flows
        back to the runner via real IPC; the runner advances past it."""
        os.environ["CHEETAHCLAWS_E2E_FAKE_PERMISSION"] = "1"
        from cc_daemon import permission, runner_supervisor as rs

        store = permission.PermissionStore()
        store.start_janitor()
        try:
            handle = rs.start(
                name="e2e-perm",
                template_name=self._template,
                args="",
                config={},
                interval=0.1,
                auto_approve=False,
                originator="alice",
                permission_store=store,
            )
            try:
                # Wait up to 15 s for the runner to ask for permission.
                deadline = time.monotonic() + 15.0
                pending: list = []
                while time.monotonic() < deadline:
                    pending = store.list_pending_for("alice")
                    if pending:
                        break
                    time.sleep(0.05)
                self.assertTrue(
                    pending,
                    "real runner never raised a PermissionRequest into the "
                    "store. "
                    f"handle.status={handle.status!r} "
                    f"stderr_tail={bytes(handle.stderr_tail)[-512:]!r}",
                )
                pr = pending[0]
                # Originator answers approve.
                store.answer(pr.request_id, "alice", {"approve": True})

                # After approval, the runner's loop continues so the
                # iteration completes and an agent_iterations row lands.
                deadline = time.monotonic() + 10.0
                rows: list = []
                while time.monotonic() < deadline:
                    rows = _query(
                        self._db_path,
                        "SELECT iteration FROM agent_iterations "
                        "WHERE run_id = ?",
                        handle.run_id,
                    )
                    if rows:
                        break
                    time.sleep(0.1)
                self.assertTrue(
                    rows,
                    "iteration never finished after the originator's approval",
                )
            finally:
                rs.stop("e2e-perm", timeout_s=5.0)
        finally:
            store.stop()


if __name__ == "__main__":
    unittest.main()
