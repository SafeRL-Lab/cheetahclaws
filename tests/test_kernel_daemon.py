"""Integration tests: kernel via the daemon RPC surface.

Pattern lifted from tests/test_daemon_spike.py — spin up a TCP daemon
on an ephemeral port for each test, drive kernel.* RPC calls through
HTTP. We register the kernel via ``kernel.register_with_daemon``
after the daemon is up; the cli.py --enable-kernel path is exercised
separately by tests/test_daemon_cli.py-style coverage.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import uuid
from pathlib import Path

import pytest

from cheetahclaws.daemon import API_VERSION, API_VERSION_HEADER, events
from cheetahclaws.daemon.originator import CLIENT_ID_HEADER, CLIENT_KIND_HEADER
from cheetahclaws.daemon.server import make_tcp_server

from cheetahclaws.kernel import register_with_daemon
from cheetahclaws.kernel.integration import detach
from cheetahclaws.kernel.process import AgentState
from cheetahclaws.kernel.store import EV_PROCESS_CREATED, EV_PROCESS_RECOVERED


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def daemon_with_kernel(tmp_path):
    """TCP daemon + kernel registered. Yields (host, port, token, kernel_db)."""
    events.reset_bus_for_tests()
    port = _free_port()
    token = "test-" + uuid.uuid4().hex
    server = make_tcp_server(
        "127.0.0.1", port,
        data_dir=tmp_path,
        token=token,
        audit_enabled=True,
    )
    kernel_db = tmp_path / "kernel.db"
    register_with_daemon(server.daemon_state, kernel_db)
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

    yield "127.0.0.1", port, token, kernel_db

    detach(server.daemon_state)
    server.daemon_state.shutdown()
    server.shutdown()
    server.server_close()
    t.join(timeout=2)


def _rpc(host, port, token, method, params, *, kind="test"):
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        CLIENT_KIND_HEADER: kind,
        API_VERSION_HEADER: API_VERSION,
        "Content-Length": str(len(body)),
    }
    conn = http.client.HTTPConnection(host, port, timeout=2)
    try:
        conn.request("POST", "/rpc", body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else None
    finally:
        conn.close()


# ── Method coverage ────────────────────────────────────────────────────────


def test_kernel_info_reports_zero_state(daemon_with_kernel):
    host, port, token, _db = daemon_with_kernel
    status, resp = _rpc(host, port, token, "kernel.info", {})
    assert status == 200, resp
    assert "result" in resp, resp
    info = resp["result"]
    # Track the live schema version rather than pinning a literal —
    # avoids a churn point on every additive schema bump.
    from cheetahclaws.kernel import SCHEMA_VERSION
    assert info["schema_version"] == SCHEMA_VERSION
    assert info["agent_count"] == 0
    assert info["event_count"] == 0


def test_agent_create_and_get_round_trip(daemon_with_kernel):
    host, port, token, _ = daemon_with_kernel
    status, resp = _rpc(host, port, token, "kernel.agent.create",
                        {"name": "alice", "template": "research/surveyor"})
    assert status == 200
    pid = resp["result"]["pid"]
    assert resp["result"]["state"] == AgentState.READY

    status, resp = _rpc(host, port, token, "kernel.agent.get", {"pid": pid})
    assert status == 200
    a = resp["result"]
    assert a["pid"] == pid
    assert a["name"] == "alice"
    assert a["state"] == AgentState.READY


def test_agent_transition_through_dead(daemon_with_kernel):
    host, port, token, _ = daemon_with_kernel
    pid = _rpc(host, port, token, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]

    s, r = _rpc(host, port, token, "kernel.agent.transition",
                {"pid": pid, "target_state": "RUNNING"})
    assert s == 200
    assert r["result"]["state"] == "RUNNING"

    s, r = _rpc(host, port, token, "kernel.agent.terminate",
                {"pid": pid, "exit_kind": "completed"})
    assert s == 200
    assert r["result"]["state"] == "DEAD"


def test_illegal_transition_returns_rpc_error(daemon_with_kernel):
    host, port, token, _ = daemon_with_kernel
    pid = _rpc(host, port, token, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    # READY -> WAITING is illegal.
    status, resp = _rpc(host, port, token, "kernel.agent.transition",
                        {"pid": pid, "target_state": "WAITING"})
    assert status == 200  # JSON-RPC errors come back as 200 with error envelope
    assert "error" in resp
    # The dispatcher's INTERNAL_ERROR path surfaces the class name in
    # the message; clients can match on it.
    assert "IllegalTransition" in resp["error"]["message"]


def test_unknown_pid_returns_rpc_error(daemon_with_kernel):
    host, port, token, _ = daemon_with_kernel
    status, resp = _rpc(host, port, token, "kernel.agent.get", {"pid": 9999})
    assert status == 200
    assert "error" in resp
    assert "UnknownPid" in resp["error"]["message"]


def test_invalid_payload_returns_invalid_params(daemon_with_kernel):
    host, port, token, _ = daemon_with_kernel
    # Missing 'name' field.
    status, resp = _rpc(host, port, token, "kernel.agent.create",
                        {"template": "t"})
    assert status == 200
    assert "error" in resp
    # InvalidPayload maps to TypeError -> INVALID_PARAMS (-32602).
    assert resp["error"]["code"] == -32602


def test_events_append_and_tail(daemon_with_kernel):
    host, port, token, _ = daemon_with_kernel
    pid = _rpc(host, port, token, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]

    # User-level event.
    s, r = _rpc(host, port, token, "kernel.events.append",
                {"pid": pid, "kind": "my.app.tool_call",
                 "payload": {"tool": "Bash", "ok": True}})
    assert s == 200 and "result" in r

    s, r = _rpc(host, port, token, "kernel.events.tail",
                {"pid": pid, "limit": 100})
    assert s == 200
    kinds = [e["kind"] for e in r["result"]["events"]]
    assert kinds == [EV_PROCESS_CREATED, "my.app.tool_call"]


def test_events_append_rejects_kernel_prefix_via_rpc(daemon_with_kernel):
    host, port, token, _ = daemon_with_kernel
    pid = _rpc(host, port, token, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    s, r = _rpc(host, port, token, "kernel.events.append",
                {"pid": pid, "kind": "kernel.foo", "payload": {}})
    assert s == 200
    assert "error" in r
    assert r["error"]["code"] == -32602


def test_agent_list_filters_via_rpc(daemon_with_kernel):
    host, port, token, _ = daemon_with_kernel
    pid1 = _rpc(host, port, token, "kernel.agent.create",
                {"name": "a1", "template": "t"})[1]["result"]["pid"]
    _pid2 = _rpc(host, port, token, "kernel.agent.create",
                 {"name": "a2", "template": "t"})[1]["result"]["pid"]
    _rpc(host, port, token, "kernel.agent.transition",
         {"pid": pid1, "target_state": "RUNNING"})

    s, r = _rpc(host, port, token, "kernel.agent.list", {"state": "RUNNING"})
    assert s == 200
    agents = r["result"]["agents"]
    assert len(agents) == 1
    assert agents[0]["pid"] == pid1


def test_kernel_methods_do_not_collide_with_existing(daemon_with_kernel):
    """Smoke test: existing methods (system.ping, echo.ping) still work."""
    host, port, token, _ = daemon_with_kernel
    s, r = _rpc(host, port, token, "system.ping", {})
    assert s == 200 and r["result"] == "pong"
    s, r = _rpc(host, port, token, "echo.ping", {"message": "hi"})
    assert s == 200 and r["result"]["pong"] is True


# ── Recovery via the daemon path ───────────────────────────────────────────


def test_recovery_runs_at_register_time(tmp_path):
    """register_with_daemon() runs recover() exactly once."""
    events.reset_bus_for_tests()
    db = tmp_path / "kernel.db"

    # Seed: one RUNNING agent left over from a prior daemon.
    from cheetahclaws.kernel import KernelStore
    seeded = KernelStore.open(db)
    try:
        a = seeded.create(name="x", template="t")
        seeded.transition(a.pid, AgentState.RUNNING)
        seeded_pid = a.pid
    finally:
        seeded.close()

    # Now bring the daemon up against the same DB.
    port = _free_port()
    token = "test-" + uuid.uuid4().hex
    server = make_tcp_server(
        "127.0.0.1", port, data_dir=tmp_path,
        token=token, audit_enabled=True,
    )
    register_with_daemon(server.daemon_state, db)
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
    try:
        s, r = _rpc("127.0.0.1", port, token, "kernel.agent.get",
                    {"pid": seeded_pid})
        assert s == 200
        assert r["result"]["state"] == AgentState.SUSPENDED
        assert r["result"]["state_reason"] == "daemon_restart"

        s, r = _rpc("127.0.0.1", port, token, "kernel.events.tail",
                    {"pid": seeded_pid, "kind": EV_PROCESS_RECOVERED})
        assert s == 200
        assert len(r["result"]["events"]) == 1
    finally:
        detach(server.daemon_state)
        server.daemon_state.shutdown()
        server.shutdown()
        server.server_close()
        t.join(timeout=2)
