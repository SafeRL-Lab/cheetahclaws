"""Tests for the F-4 #2 path: a runner's `notify` IPC frame routes
through to the bridge supervisor's outbound mailbox.

Closes the gap that §F-4 "Still TODO" #2 documented: the supervisor's
reader used to drop ``{"op":"notify", "text": ...}`` on the floor.
With F-6 landed, the supervisor now forwards into
``bridge_supervisor.notify(...)`` and publishes an
``agent_runner_notify`` event so observers can see whether delivery
succeeded.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


pytestmark_skipif_windows = sys.platform.startswith("win")


# Inline runner that handshakes then emits one `notify` and exits cleanly.
_NOTIFY_RUNNER = textwrap.dedent("""
    import json, sys
    def _send(o):
        sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
    init = json.loads(sys.stdin.readline())
    _send({"op": "ready"})
    _send({"op": "notify", "text": "hello from runner", "bridge": "telegram"})
    # Wait for stop so the supervisor's reader has time to consume.
    for raw in sys.stdin:
        if json.loads(raw).get("op") == "stop":
            sys.exit(0)
""").strip()


def _spawn_inline_notify_runner(name="notifier"):
    from cc_daemon import runner_supervisor as rs
    from cc_daemon.runner_ipc import JsonLineChannel

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _NOTIFY_RUNNER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
    )
    chan = JsonLineChannel(proc.stdout, proc.stdin)
    chan.send({"op": "init", "payload": {"name": name}})
    reply = chan.recv(timeout=5.0)
    assert reply["op"] == "ready"
    handle = rs.RunnerHandle(
        name=name, run_id=f"run_{name}",
        pid=proc.pid, started_at=time.time(),
        proc=proc, chan=chan,
    )
    handle.status = "running"
    rs._register(handle)
    t = threading.Thread(target=rs._reader_loop, args=(handle,), daemon=True)
    t.start()
    handle._reader = t
    return handle


class TestNotifyRouting(unittest.TestCase):

    def setUp(self):
        from cc_daemon import runner_supervisor as rs
        from cc_daemon import bridge_supervisor as bs
        with rs._handles_lock:
            rs._handles.clear()
        with bs._handles_lock:
            for h in list(bs._handles.values()):
                try:
                    h.stop_event.set()
                except Exception:
                    pass
            bs._handles.clear()

    def tearDown(self):
        from cc_daemon import runner_supervisor as rs
        from cc_daemon import bridge_supervisor as bs
        for h in list(rs._handles.values()):
            try:
                h.proc.kill()
            except Exception:
                pass
        with rs._handles_lock:
            rs._handles.clear()
        with bs._handles_lock:
            bs._handles.clear()

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_notify_forwards_to_bridge_supervisor(self):
        from cc_daemon import runner_supervisor as rs
        from cc_daemon import bridge_supervisor as bs

        calls: list[tuple[str, str]] = []
        def fake_notify(kind, text):
            calls.append((kind, text))
            return True

        with patch.object(bs, "notify", side_effect=fake_notify):
            handle = _spawn_inline_notify_runner()
            # Wait a beat for the reader to consume the notify frame.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if calls:
                    break
                time.sleep(0.05)
            self.assertEqual(len(calls), 1, f"expected 1 notify, got {calls!r}")
            self.assertEqual(calls[0], ("telegram", "hello from runner"))

        rs.stop("notifier", timeout_s=3.0)

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_notify_broadcast_when_bridge_unspecified(self):
        """A runner that omits the ``bridge`` field defaults to ``*``
        broadcast, so the originator's bridge doesn't have to be
        threaded all the way down to the agent template."""
        from cc_daemon import runner_supervisor as rs
        from cc_daemon import bridge_supervisor as bs

        # Runner variant without "bridge" key.
        source = textwrap.dedent("""
            import json, sys
            def _send(o):
                sys.stdout.write(json.dumps(o)+"\\n"); sys.stdout.flush()
            init = json.loads(sys.stdin.readline())
            _send({"op": "ready"})
            _send({"op": "notify", "text": "hi"})
            for raw in sys.stdin:
                if json.loads(raw).get("op") == "stop":
                    sys.exit(0)
        """).strip()

        from cc_daemon.runner_ipc import JsonLineChannel
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
        )
        chan = JsonLineChannel(proc.stdout, proc.stdin)
        chan.send({"op": "init", "payload": {"name": "anon"}})
        chan.recv(timeout=5.0)

        handle = rs.RunnerHandle(
            name="anon", run_id="run_anon", pid=proc.pid,
            started_at=time.time(), proc=proc, chan=chan,
        )
        handle.status = "running"
        rs._register(handle)

        calls: list[tuple[str, str]] = []
        with patch.object(bs, "notify",
                          side_effect=lambda k, t: calls.append((k, t)) or True):
            t = threading.Thread(target=rs._reader_loop,
                                 args=(handle,), daemon=True)
            t.start()

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if calls:
                    break
                time.sleep(0.05)

        self.assertEqual(calls, [("*", "hi")])
        rs.stop("anon", timeout_s=3.0)

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_notify_with_empty_text_is_skipped(self):
        from cc_daemon import runner_supervisor as rs
        from cc_daemon import bridge_supervisor as bs

        source = textwrap.dedent("""
            import json, sys
            def _send(o):
                sys.stdout.write(json.dumps(o)+"\\n"); sys.stdout.flush()
            init = json.loads(sys.stdin.readline())
            _send({"op": "ready"})
            _send({"op": "notify", "text": ""})
            _send({"op": "notify"})           # no text key at all
            for raw in sys.stdin:
                if json.loads(raw).get("op") == "stop":
                    sys.exit(0)
        """).strip()

        from cc_daemon.runner_ipc import JsonLineChannel
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
        )
        chan = JsonLineChannel(proc.stdout, proc.stdin)
        chan.send({"op": "init", "payload": {"name": "empty"}})
        chan.recv(timeout=5.0)

        handle = rs.RunnerHandle(
            name="empty", run_id="run_empty", pid=proc.pid,
            started_at=time.time(), proc=proc, chan=chan,
        )
        handle.status = "running"
        rs._register(handle)

        calls: list = []
        with patch.object(bs, "notify",
                          side_effect=lambda k, t: calls.append((k, t)) or True):
            t = threading.Thread(target=rs._reader_loop,
                                 args=(handle,), daemon=True)
            t.start()
            time.sleep(0.4)  # let both frames be consumed
        self.assertEqual(calls, [], f"expected zero notify calls, got {calls!r}")
        rs.stop("empty", timeout_s=3.0)


if __name__ == "__main__":
    unittest.main()
