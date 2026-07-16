"""End-to-end RPC tests for Phase-2 surface (kernel.cap.* + kernel.ledger.*).

Reuses the daemon-spinup pattern from tests/test_kernel_daemon.py.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import uuid

import pytest

from cheetahclaws.daemon import API_VERSION, API_VERSION_HEADER, events
from cheetahclaws.daemon.originator import CLIENT_KIND_HEADER
from cheetahclaws.daemon.server import make_tcp_server

from cheetahclaws.kernel import register_with_daemon
from cheetahclaws.kernel.integration import detach


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def daemon(tmp_path):
    events.reset_bus_for_tests()
    port = _free_port()
    token = "test-" + uuid.uuid4().hex
    server = make_tcp_server(
        "127.0.0.1", port,
        data_dir=tmp_path,
        token=token,
        audit_enabled=True,
    )
    register_with_daemon(server.daemon_state, tmp_path / "kernel.db")
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
    yield "127.0.0.1", port, token
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
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


# ── kernel.cap.* ──────────────────────────────────────────────────────────


def test_cap_create_and_get_round_trip(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "alice", "template": "t"})[1]["result"]["pid"]
    s, r = _rpc(h, p, t, "kernel.cap.create", {
        "pid": pid,
        "tool_grants": ["Read", "Bash"],
        "fs_grants": [{"prefix": "/work/", "mode": "rw"}],
        "net_grants": ["*.example.com"],
        "model_grants": ["claude-opus-4"],
        "sub_agent": True,
    })
    assert s == 200, r
    assert r["result"]["pid"] == pid
    cap = _rpc(h, p, t, "kernel.cap.get", {"pid": pid})[1]["result"]
    assert cap["tool_grants"] == ["Bash", "Read"]
    assert cap["sub_agent"] is True


def test_cap_check_endpoints(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.cap.create", {
        "pid": pid,
        "tool_grants": ["Read"],
        "fs_grants": [{"prefix": "/work/", "mode": "r"}],
        "net_grants": ["*.example.com"],
        "model_grants": ["claude-opus-4"],
    })
    assert _rpc(h, p, t, "kernel.cap.check_tool",
                {"pid": pid, "tool": "Read"})[1]["result"]["allowed"] is True
    assert _rpc(h, p, t, "kernel.cap.check_tool",
                {"pid": pid, "tool": "Bash"})[1]["result"]["allowed"] is False
    assert _rpc(h, p, t, "kernel.cap.check_fs",
                {"pid": pid, "path": "/work/a", "mode": "r"})[1]["result"]["allowed"] is True
    assert _rpc(h, p, t, "kernel.cap.check_fs",
                {"pid": pid, "path": "/work/a", "mode": "rw"})[1]["result"]["allowed"] is False
    assert _rpc(h, p, t, "kernel.cap.check_net",
                {"pid": pid, "host": "api.example.com"})[1]["result"]["allowed"] is True
    assert _rpc(h, p, t, "kernel.cap.check_model",
                {"pid": pid, "model": "claude-opus-4"})[1]["result"]["allowed"] is True


def test_cap_derive_via_rpc(daemon):
    h, p, t = daemon
    p1 = _rpc(h, p, t, "kernel.agent.create",
              {"name": "p", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.cap.create", {
        "pid": p1, "tool_grants": ["Read", "Bash"], "sub_agent": True,
    })
    c = _rpc(h, p, t, "kernel.agent.create",
             {"name": "c", "template": "t", "parent_pid": p1})[1]["result"]["pid"]
    s, r = _rpc(h, p, t, "kernel.cap.derive", {
        "parent_pid": p1, "child_pid": c,
        "tool_grants": ["Read"],
    })
    assert s == 200, r
    assert r["result"]["parent_cap_id"] is not None


def test_cap_derive_violation_returns_error(daemon):
    h, p, t = daemon
    p1 = _rpc(h, p, t, "kernel.agent.create",
              {"name": "p", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.cap.create", {"pid": p1, "tool_grants": ["Read"]})
    c = _rpc(h, p, t, "kernel.agent.create",
             {"name": "c", "template": "t", "parent_pid": p1})[1]["result"]["pid"]
    s, r = _rpc(h, p, t, "kernel.cap.derive", {
        "parent_pid": p1, "child_pid": c,
        "tool_grants": ["Read", "Bash"],   # Bash is broader than parent
    })
    assert s == 200
    assert "error" in r
    assert "CapabilityDerivationError" in r["error"]["message"]


# ── kernel.ledger.* ───────────────────────────────────────────────────────


def test_ledger_create_charge_round_trip(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    s, r = _rpc(h, p, t, "kernel.ledger.create", {
        "pid": pid, "grants": {"tokens": 1000},
    })
    assert s == 200
    assert r["result"]["dims"] == ["tokens"]
    s, r = _rpc(h, p, t, "kernel.ledger.charge", {
        "pid": pid, "dim": "tokens", "amount": 1500,
    })
    assert s == 200
    assert r["result"]["over_limit"]   is True
    assert r["result"]["first_breach"] is True
    assert r["result"]["used"]         == 1500


def test_ledger_check_no_mutate(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.ledger.create",
         {"pid": pid, "grants": {"tokens": 100}})
    chk = _rpc(h, p, t, "kernel.ledger.check",
               {"pid": pid, "dim": "tokens", "amount": 50})[1]["result"]
    assert chk["would_use"] == 50
    assert chk["would_exceed"] is False
    led = _rpc(h, p, t, "kernel.ledger.get", {"pid": pid})[1]["result"]
    assert led["entries"][0]["used"] == 0


def test_ledger_list_breached(daemon):
    h, p, t = daemon
    p1 = _rpc(h, p, t, "kernel.agent.create",
              {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.ledger.create",
         {"pid": p1, "grants": {"tokens": 100}})
    _rpc(h, p, t, "kernel.ledger.charge",
         {"pid": p1, "dim": "tokens", "amount": 200})
    s, r = _rpc(h, p, t, "kernel.ledger.list_breached", {})
    assert s == 200
    assert len(r["result"]["entries"]) == 1
    assert r["result"]["entries"][0]["pid"] == p1


def test_phase2_does_not_break_phase1(daemon):
    """Sanity: existing kernel.* methods still work alongside the new ones."""
    h, p, t = daemon
    s, r = _rpc(h, p, t, "kernel.info", {})
    assert s == 200
    from cheetahclaws.kernel import SCHEMA_VERSION
    assert r["result"]["schema_version"] == SCHEMA_VERSION
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.agent.transition",
         {"pid": pid, "target_state": "RUNNING"})
    a = _rpc(h, p, t, "kernel.agent.get", {"pid": pid})[1]["result"]
    assert a["state"] == "RUNNING"
