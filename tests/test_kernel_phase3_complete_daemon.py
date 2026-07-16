"""End-to-end RPC smoke tests for Phase-3 mailbox + registry surfaces.

The unit-level tests cover most behaviour; this file confirms the RPC
wiring works end-to-end alongside the existing kernel.* methods.
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
    body = json.dumps({"jsonrpc": "2.0", "id": str(uuid.uuid4()),
                       "method": method, "params": params}).encode()
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


# ── Mailbox RPC ──────────────────────────────────────────────────────────


def test_mbox_send_recv_via_rpc(daemon):
    h, p, t = daemon
    sender = _rpc(h, p, t, "kernel.agent.create",
                  {"name": "sender", "template": "t"})[1]["result"]["pid"]
    recv   = _rpc(h, p, t, "kernel.agent.create",
                  {"name": "recv",   "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.mbox.create", {"pid": recv})
    s, r = _rpc(h, p, t, "kernel.mbox.send", {
        "sender_pid": sender, "recipient_pid": recv,
        "kind": "hello", "payload": {"msg": "hi"},
    })
    assert s == 200
    msg_id = r["result"]["msg_id"]

    s, r = _rpc(h, p, t, "kernel.mbox.recv", {"pid": recv})
    assert s == 200
    msgs = r["result"]["messages"]
    assert len(msgs) == 1
    assert msgs[0]["msg_id"] == msg_id
    assert msgs[0]["kind"] == "hello"


def test_mbox_publish_fanout_via_rpc(daemon):
    h, p, t = daemon
    a = _rpc(h, p, t, "kernel.agent.create",
             {"name": "a", "template": "t"})[1]["result"]["pid"]
    b = _rpc(h, p, t, "kernel.agent.create",
             {"name": "b", "template": "t"})[1]["result"]["pid"]
    for pid in (a, b):
        _rpc(h, p, t, "kernel.mbox.create", {"pid": pid})
        _rpc(h, p, t, "kernel.mbox.subscribe",
             {"pid": pid, "topic": "alerts"})
    s, r = _rpc(h, p, t, "kernel.mbox.publish",
                {"topic": "alerts", "kind": "ping", "payload": {}})
    assert s == 200
    assert r["result"]["delivered"] == 2


def test_mbox_full_returns_error(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.mbox.create",
         {"pid": pid, "queue_size": 1})
    _rpc(h, p, t, "kernel.mbox.send",
         {"recipient_pid": pid, "kind": "k", "payload": {}})
    s, r = _rpc(h, p, t, "kernel.mbox.send",
                {"recipient_pid": pid, "kind": "k", "payload": {}})
    assert s == 200
    assert "error" in r
    assert "MailboxFull" in r["error"]["message"]


# ── Registry RPC ─────────────────────────────────────────────────────────


def test_registry_register_lookup_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "alice", "template": "t"})[1]["result"]["pid"]
    s, _ = _rpc(h, p, t, "kernel.registry.register",
                {"name": "/agents/alice", "pid": pid,
                 "tags": ["research"]})
    assert s == 200
    s, r = _rpc(h, p, t, "kernel.registry.lookup",
                {"name": "/agents/alice"})
    assert s == 200
    assert r["result"]["pid"] == pid


def test_registry_list_filters_via_rpc(daemon):
    h, p, t = daemon
    a = _rpc(h, p, t, "kernel.agent.create",
             {"name": "a", "template": "t"})[1]["result"]["pid"]
    b = _rpc(h, p, t, "kernel.agent.create",
             {"name": "b", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.registry.register",
         {"name": "/agents/x", "pid": a, "tags": ["research"]})
    _rpc(h, p, t, "kernel.registry.register",
         {"name": "/agents/y", "pid": b, "tags": ["bridge"]})
    s, r = _rpc(h, p, t, "kernel.registry.list",
                {"tag": "research"})
    assert s == 200
    assert r["result"]["total"] == 1
    assert r["result"]["entries"][0]["name"] == "/agents/x"


def test_registry_duplicate_name_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.registry.register",
         {"name": "/x", "pid": pid})
    s, r = _rpc(h, p, t, "kernel.registry.register",
                {"name": "/x", "pid": pid})
    assert s == 200
    assert "error" in r
    assert "RegistryNameExists" in r["error"]["message"]


# ── Phase-3 surfaces don't break Phase-1/2 ───────────────────────────────


def test_phase3_does_not_break_earlier_phases(daemon):
    h, p, t = daemon
    s, r = _rpc(h, p, t, "kernel.info", {})
    from cheetahclaws.kernel import SCHEMA_VERSION
    assert r["result"]["schema_version"] == SCHEMA_VERSION
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "z", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.cap.create",
         {"pid": pid, "tool_grants": ["Read"]})
    assert _rpc(h, p, t, "kernel.cap.check_tool",
                {"pid": pid, "tool": "Read"})[1]["result"]["allowed"] is True
    _rpc(h, p, t, "kernel.ledger.create",
         {"pid": pid, "grants": {"tokens": 100}})
    sid = _rpc(h, p, t, "kernel.sched.enqueue",
               {"pid": pid})[1]["result"]["sched_id"]
    assert isinstance(sid, int)
