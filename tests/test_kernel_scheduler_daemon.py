"""End-to-end RPC tests for kernel.sched.* (RFC 0007)."""
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
        "127.0.0.1", port, data_dir=tmp_path,
        token=token, audit_enabled=True,
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
        "jsonrpc": "2.0", "id": str(uuid.uuid4()),
        "method": method, "params": params,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        CLIENT_KIND_HEADER: kind,
        API_VERSION_HEADER: API_VERSION,
        "Content-Length": str(len(body)),
    }
    c = http.client.HTTPConnection(host, port, timeout=2)
    try:
        c.request("POST", "/rpc", body=body, headers=headers)
        r = c.getresponse()
        return r.status, json.loads(r.read())
    finally:
        c.close()


def test_enqueue_claim_complete_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    s, r = _rpc(h, p, t, "kernel.sched.enqueue",
                {"pid": pid, "priority": 5, "trigger": "manual",
                 "payload": {"step": 1}})
    assert s == 200
    sid = r["result"]["sched_id"]

    s, r = _rpc(h, p, t, "kernel.sched.claim",
                {"worker_id": "sup-1", "max_n": 1})
    assert s == 200
    entries = r["result"]["entries"]
    assert len(entries) == 1
    assert entries[0]["sched_id"] == sid
    assert entries[0]["state"] == "dispatched"

    s, r = _rpc(h, p, t, "kernel.sched.complete",
                {"sched_id": sid, "exit_kind": "completed"})
    assert s == 200
    assert r["result"]["state"] == "completed"


def test_priority_order_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    s_lo = _rpc(h, p, t, "kernel.sched.enqueue",
                {"pid": pid, "priority": 1})[1]["result"]["sched_id"]
    s_hi = _rpc(h, p, t, "kernel.sched.enqueue",
                {"pid": pid, "priority": 10})[1]["result"]["sched_id"]
    r = _rpc(h, p, t, "kernel.sched.claim",
             {"worker_id": "sup-1", "max_n": 2})[1]["result"]
    assert [e["sched_id"] for e in r["entries"]] == [s_hi, s_lo]


def test_admission_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.ledger.create",
         {"pid": pid, "grants": {"tokens": 100}})
    _rpc(h, p, t, "kernel.ledger.charge",
         {"pid": pid, "dim": "tokens", "amount": 200})
    sid = _rpc(h, p, t, "kernel.sched.enqueue",
               {"pid": pid})[1]["result"]["sched_id"]
    r = _rpc(h, p, t, "kernel.sched.claim",
             {"worker_id": "sup-1", "max_n": 10})[1]["result"]
    assert r["entries"] == []
    # Disable admission and we get it back.
    r2 = _rpc(h, p, t, "kernel.sched.claim",
              {"worker_id": "sup-1", "max_n": 10,
               "admission_check": False})[1]["result"]
    assert len(r2["entries"]) == 1
    assert r2["entries"][0]["sched_id"] == sid


def test_cancel_dispatched_returns_error(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    sid = _rpc(h, p, t, "kernel.sched.enqueue",
               {"pid": pid})[1]["result"]["sched_id"]
    _rpc(h, p, t, "kernel.sched.claim",
         {"worker_id": "sup-1", "max_n": 1})
    s, r = _rpc(h, p, t, "kernel.sched.cancel", {"sched_id": sid})
    assert s == 200
    assert "error" in r
    assert "SchedIllegalTransition" in r["error"]["message"]


def test_gc_expired_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.sched.enqueue",
         {"pid": pid, "runnable_at": 0, "deadline": 100})
    _rpc(h, p, t, "kernel.sched.enqueue", {"pid": pid})
    s, r = _rpc(h, p, t, "kernel.sched.gc_expired", {"now": 200})
    assert s == 200
    assert r["result"]["swept"] == 1


def test_list_filters_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.sched.enqueue", {"pid": pid})
    _rpc(h, p, t, "kernel.sched.enqueue", {"pid": pid})
    s, r = _rpc(h, p, t, "kernel.sched.list",
                {"state": "queued"})
    assert s == 200
    assert r["result"]["total"] == 2
