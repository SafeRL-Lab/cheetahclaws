"""Tests for daemon/agent_methods.py (RFC 0002 F-4).

These tests exercise the four JSON-RPC method functions
(agent.start / agent.stop / agent.list / agent.status) without spinning
up the full daemon HTTP server. We construct an ``RpcRegistry`` directly,
``register()`` the methods against a minimal stand-in for ``DaemonState``,
and dispatch envelopes the same way ``server.py`` does.

agent.start is exercised via the supervisor's inline-runner helper from
the supervisor tests (so we don't pay for spawning the real
``python -m agent_runner``).
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


pytestmark_skipif_windows = sys.platform.startswith("win")


class _FakeDaemonState:
    """Minimal DaemonState stand-in — agent_methods only reads .config."""

    def __init__(self, config=None):
        self.config = config or {}


def _build_registry(state=None):
    """Fresh RpcRegistry with agent.* methods registered."""
    from cheetahclaws.daemon.rpc import RpcRegistry
    from cheetahclaws.daemon import agent_methods
    reg = RpcRegistry()
    agent_methods.register(reg, state or _FakeDaemonState())
    return reg


def _ctx():
    from cheetahclaws.daemon.rpc import CallContext
    return CallContext(client_id="test", transport="unix", api_version="0")


def _call(reg, method, params=None):
    """Dispatch one RPC envelope and return (result, error) tuple."""
    envelope = {"jsonrpc": "2.0", "id": 1, "method": method,
                "params": params or {}}
    response, _http = reg.dispatch(envelope, _ctx())
    return response.get("result"), response.get("error")


class TestRegistration(unittest.TestCase):

    def test_all_four_methods_registered(self):
        reg = _build_registry()
        names = set(reg.methods())
        self.assertEqual(
            names & {"agent.start", "agent.stop", "agent.list", "agent.status"},
            {"agent.start", "agent.stop", "agent.list", "agent.status"},
        )


class TestParamValidation(unittest.TestCase):

    def test_start_requires_name(self):
        reg = _build_registry()
        result, err = _call(reg, "agent.start", {"template": "demo"})
        self.assertIsNone(result)
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)         # INVALID_PARAMS
        self.assertIn("name", err["message"])

    def test_start_requires_template(self):
        reg = _build_registry()
        result, err = _call(reg, "agent.start", {"name": "x"})
        self.assertIsNone(result)
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)
        self.assertIn("template", err["message"])

    def test_start_rejects_non_numeric_interval(self):
        reg = _build_registry()
        result, err = _call(reg, "agent.start",
                            {"name": "x", "template": "demo",
                             "interval": "not-a-number"})
        self.assertIsNone(result)
        self.assertEqual(err["code"], -32602)
        self.assertIn("interval", err["message"])

    def test_stop_requires_name(self):
        reg = _build_registry()
        result, err = _call(reg, "agent.stop", {})
        self.assertEqual(err["code"], -32602)
        self.assertIn("name", err["message"])

    def test_status_requires_name(self):
        reg = _build_registry()
        result, err = _call(reg, "agent.status", {})
        self.assertEqual(err["code"], -32602)


class TestListWhenEmpty(unittest.TestCase):

    def test_list_returns_empty(self):
        # Clear the registry first in case another test left something.
        from cheetahclaws.daemon import runner_supervisor
        for h in list(runner_supervisor.list_all()):
            runner_supervisor._unregister(h.name)
        reg = _build_registry()
        result, err = _call(reg, "agent.list", {})
        self.assertIsNone(err)
        self.assertEqual(result, {"runners": []})


class TestStatusUnknown(unittest.TestCase):

    def test_status_returns_not_found(self):
        reg = _build_registry()
        result, err = _call(reg, "agent.status", {"name": "no-such"})
        self.assertIsNone(err)
        self.assertEqual(result.get("found"), False)
        self.assertEqual(result.get("name"), "no-such")


class TestStopUnknown(unittest.TestCase):

    def test_stop_unknown_returns_false(self):
        reg = _build_registry()
        result, err = _call(reg, "agent.stop", {"name": "no-such"})
        self.assertIsNone(err)
        self.assertEqual(result, {"name": "no-such", "stopped": False})


# ── End-to-end with inline runner ─────────────────────────────────────────


class TestStopRunningEndToEnd(unittest.TestCase):
    """Spawn an inline runner via the same helper used in the supervisor
    tests, then verify agent.list shows it and agent.stop ends it."""

    @unittest.skipIf(pytestmark_skipif_windows, "POSIX only")
    def test_list_and_stop_round_trip(self):
        import textwrap, subprocess, threading
        from cheetahclaws.daemon import runner_supervisor as rs
        from cheetahclaws.daemon.runner_ipc import JsonLineChannel

        source = textwrap.dedent("""
            import json, sys
            def _send(o):
                sys.stdout.write(json.dumps(o)+"\\n"); sys.stdout.flush()
            init = json.loads(sys.stdin.readline())
            _send({"op":"ready"})
            for raw in sys.stdin:
                if json.loads(raw).get("op")=="stop":
                    _send({"op":"exit","reason":"stopped","iterations":0})
                    sys.exit(0)
        """).strip()

        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
        )
        chan = JsonLineChannel(proc.stdout, proc.stdin)
        handle = rs.RunnerHandle(
            name="rpc-roundtrip", run_id="rpc_round",
            pid=proc.pid, started_at=time.time(),
            proc=proc, chan=chan,
            template_name="inline", args="",
        )
        chan.send({"op": "init", "payload": {"name": "rpc-roundtrip"}})
        reply = chan.recv(timeout=5.0)
        self.assertEqual(reply["op"], "ready")
        handle.status = "running"
        rs._register(handle)
        t = threading.Thread(target=rs._reader_loop, args=(handle,),
                             daemon=True)
        t.start()
        handle._reader = t

        try:
            reg = _build_registry()

            # agent.list shows our runner.
            result, err = _call(reg, "agent.list", {})
            self.assertIsNone(err)
            names = [r["name"] for r in result["runners"]]
            self.assertIn("rpc-roundtrip", names)

            # agent.status finds it.
            result, err = _call(reg, "agent.status",
                                {"name": "rpc-roundtrip"})
            self.assertIsNone(err)
            self.assertTrue(result["found"])
            self.assertEqual(result["status"], "running")
            self.assertTrue(result["alive"])

            # agent.stop ends it within 5 s.
            t0 = time.monotonic()
            result, err = _call(reg, "agent.stop",
                                {"name": "rpc-roundtrip", "timeout_s": 5.0})
            elapsed = time.monotonic() - t0
            self.assertIsNone(err)
            self.assertEqual(result, {"name": "rpc-roundtrip", "stopped": True})
            self.assertLess(elapsed, 5.0,
                            f"stop took {elapsed:.2f}s, must be < 5s")

            # Registry no longer has it.
            result, err = _call(reg, "agent.status",
                                {"name": "rpc-roundtrip"})
            self.assertFalse(result["found"])
        finally:
            if handle.is_alive():
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    proc.kill()


if __name__ == "__main__":
    unittest.main()
