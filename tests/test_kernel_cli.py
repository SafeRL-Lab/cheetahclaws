"""Tests for kernel.cli — `cheetahclaws kernel ...` subcommands."""
from __future__ import annotations

import io
import json
import socket
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from cheetahclaws.daemon import discovery as _discovery
from cheetahclaws.daemon import events as _events
from cheetahclaws.daemon.server import make_tcp_server

from cheetahclaws.kernel import register_with_daemon
from cheetahclaws.kernel.cli import dispatch as kernel_dispatch
from cheetahclaws.kernel.integration import detach


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_kernel(tmp_path, monkeypatch):
    """Start a TCP daemon with kernel enabled, write discovery so the
    CLI can find it, and clean up on teardown.

    The CLI uses _discovery.locate() which reads
    ~/.cheetahclaws/daemon.json; we monkey-patch DEFAULT_TOKEN_PATH
    and the discovery module's path to point inside tmp_path so the
    test doesn't collide with any real running daemon."""
    _events.reset_bus_for_tests()
    port = _free_port()
    token = "t-" + uuid.uuid4().hex
    token_path = tmp_path / "daemon_token"
    token_path.write_text(token)
    token_path.chmod(0o600)
    discovery_path = tmp_path / "daemon.json"
    monkeypatch.setattr(_discovery, "get_default_path",
                         lambda: discovery_path)

    server = make_tcp_server(
        "127.0.0.1", port, data_dir=tmp_path, token=token,
        audit_enabled=True,
    )
    register_with_daemon(server.daemon_state, tmp_path / "kernel.db")

    # Use os.getpid() so discovery's liveness check (os.kill(pid, 0))
    # passes — the test process IS running.
    import os as _os
    info = _discovery.make_info(
        pid=_os.getpid(), transport="tcp",
        address=f"127.0.0.1:{port}",
        version="test",
        token_path=str(token_path),
    )
    discovery_path.write_text(json.dumps(info))

    t = threading.Thread(target=server.serve_forever,
                         kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)

    yield {
        "host": "127.0.0.1", "port": port, "token": token,
        "tmp_path": tmp_path, "server": server,
    }

    detach(server.daemon_state)
    server.daemon_state.shutdown()
    server.shutdown()
    server.server_close()
    t.join(timeout=2)


def _run_cli(*argv) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = kernel_dispatch(list(argv))
    return rc, out.getvalue(), err.getvalue()


# ── No-daemon path ──────────────────────────────────────────────────────


def test_cli_no_daemon_returns_1(monkeypatch, tmp_path):
    """When discovery is empty (no daemon), `summary` exits 1 with a
    friendly hint."""
    monkeypatch.setattr(_discovery, "get_default_path",
                         lambda: tmp_path / "no-such-daemon.json")
    rc, out, err = _run_cli("summary")
    assert rc == 1
    assert "not running" in err.lower() or "is `cheetahclaws serve" in err


# ── help / unknown ──────────────────────────────────────────────────────


def test_cli_help(tmp_path, monkeypatch):
    """--help prints usage and exits 0 even with no daemon."""
    monkeypatch.setattr(_discovery, "get_default_path",
                         lambda: tmp_path / "no.json")
    rc, out, err = _run_cli("--help")
    assert rc == 0
    assert "kernel" in err.lower() or "actions:" in err.lower()


def test_cli_unknown_subcommand(tmp_path, monkeypatch):
    monkeypatch.setattr(_discovery, "get_default_path",
                         lambda: tmp_path / "no.json")
    rc, out, err = _run_cli("not-a-real-action")
    assert rc == 2
    assert "unknown" in err.lower()


def test_cli_no_args(tmp_path, monkeypatch):
    monkeypatch.setattr(_discovery, "get_default_path",
                         lambda: tmp_path / "no.json")
    rc, out, err = _run_cli()
    assert rc == 2


# ── Live RPC path ───────────────────────────────────────────────────────


def test_cli_summary(running_kernel):
    rc, out, err = _run_cli("summary")
    assert rc == 0, err
    assert "kernel:" in out
    assert "schema:" in out
    assert "agents:" in out


def test_cli_summary_json(running_kernel):
    rc, out, err = _run_cli("summary", "--json")
    assert rc == 0, err
    parsed = json.loads(out)
    assert "schema_version" in parsed
    assert "agents" in parsed


def test_cli_info(running_kernel):
    rc, out, err = _run_cli("info")
    assert rc == 0, err
    assert "kernel_version:" in out
    assert "schema_version:" in out
    assert "method_count:" in out


def test_cli_agents_empty(running_kernel):
    rc, out, err = _run_cli("agents")
    assert rc == 0, err
    assert "no agents" in out or "(0 of 0)" in out


def test_cli_agents_with_data(running_kernel):
    """Create an agent through the daemon's kernel store, then list
    via CLI."""
    ks = running_kernel["server"].daemon_state.kernel_store
    a = ks.create(name="alice", template="t")
    rc, out, err = _run_cli("agents")
    assert rc == 0, err
    assert "alice" in out
    assert str(a.pid) in out


def test_cli_agents_state_filter(running_kernel):
    ks = running_kernel["server"].daemon_state.kernel_store
    a = ks.create(name="ready_one", template="t")
    b = ks.create(name="dead_one", template="t")
    ks.terminate(b.pid, exit_kind="completed")
    rc, out, err = _run_cli("agents", "--state", "READY")
    assert rc == 0
    assert "ready_one" in out
    assert "dead_one" not in out


def test_cli_proc(running_kernel):
    ds = running_kernel["server"].daemon_state
    a = ds.kernel_store.create(name="bob", template="t")
    ds.capability_store.create(pid=a.pid, tool_grants=["Read"])
    ds.ledger_store.create(pid=a.pid, grants={"tokens": 100})
    rc, out, err = _run_cli("proc", str(a.pid))
    assert rc == 0, err
    assert "name:       bob" in out
    assert "tools:" in out
    assert "ledger" in out


def test_cli_proc_unknown_pid(running_kernel):
    rc, out, err = _run_cli("proc", "9999")
    assert rc == 1
    assert "no process" in out


def test_cli_events(running_kernel):
    ks = running_kernel["server"].daemon_state.kernel_store
    a = ks.create(name="x", template="t")
    ks.transition(a.pid, "RUNNING")
    rc, out, err = _run_cli("events", "--pid", str(a.pid))
    assert rc == 0, err
    assert "kernel.process.created" in out
    assert "kernel.process.transitioned" in out


def test_cli_queue_empty(running_kernel):
    rc, out, err = _run_cli("queue")
    assert rc == 0, err
    assert "queue empty" in out


def test_cli_queue_with_entries(running_kernel):
    ds = running_kernel["server"].daemon_state
    a = ds.kernel_store.create(name="x", template="t")
    from cheetahclaws.kernel import ScheduleSpec
    sid = ds.scheduler_store.enqueue(ScheduleSpec(pid=a.pid, priority=5))
    rc, out, err = _run_cli("queue")
    assert rc == 0
    assert str(sid) in out
    assert "queued" in out


def test_cli_registry_empty(running_kernel):
    rc, out, err = _run_cli("registry")
    assert rc == 0, err
    assert "registry empty" in out


def test_cli_registry_with_entries(running_kernel):
    ds = running_kernel["server"].daemon_state
    a = ds.kernel_store.create(name="x", template="t")
    ds.registry_store.register(name="/agents/test/x", pid=a.pid,
                                tags=["test"])
    rc, out, err = _run_cli("registry")
    assert rc == 0
    assert "/agents/test/x" in out


def test_cli_methods(running_kernel):
    rc, out, err = _run_cli("methods")
    assert rc == 0, err
    assert "kernel.agent.create" in out
    assert "kernel.observe.summary" in out
    # Tier counts at the bottom.
    assert "stable=" in out


def test_cli_methods_tier_filter(running_kernel):
    rc, out, err = _run_cli("methods", "--tier", "stable")
    assert rc == 0
    assert "kernel.agent.create" in out


def test_cli_methods_invalid_tier(running_kernel):
    """argparse choices catches this — exits 2."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err), pytest.raises(SystemExit) as e:
        kernel_dispatch(["methods", "--tier", "bogus"])
    assert e.value.code == 2


def test_cli_prometheus(running_kernel):
    rc, out, err = _run_cli("prometheus")
    assert rc == 0, err
    assert "# HELP cheetahclaws_kernel_schema_version" in out
    assert "cheetahclaws_kernel_schema_version 5" in out
