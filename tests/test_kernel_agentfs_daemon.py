"""End-to-end RPC smoke for kernel.fs.* (RFC 0011).

Covers base64 wire encoding round-trip + a few error paths.
"""
from __future__ import annotations

import base64
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


def test_write_read_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    # Binary payload must round-trip.
    payload = bytes(range(256))
    s, r = _rpc(h, p, t, "kernel.fs.write", {
        "pid": pid, "path": "/memory/x/note",
        "content": base64.b64encode(payload).decode("ascii"),
    })
    assert s == 200
    assert r["result"]["size"] == 256

    s, r = _rpc(h, p, t, "kernel.fs.read",
                {"path": "/memory/x/note"})
    assert s == 200
    decoded = base64.b64decode(r["result"]["content"])
    assert decoded == payload


def test_write_invalid_path_returns_error(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    s, r = _rpc(h, p, t, "kernel.fs.write", {
        "pid": pid, "path": "no-leading-slash",
        "content": base64.b64encode(b"x").decode("ascii"),
    })
    assert s == 200
    assert "error" in r
    assert r["error"]["code"] == -32602  # INVALID_PARAMS via TypeError


def test_list_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    for path in ("/memory/x/1", "/memory/x/2", "/skills/y"):
        _rpc(h, p, t, "kernel.fs.write", {
            "pid": pid, "path": path,
            "content": base64.b64encode(b"x").decode("ascii"),
        })
    s, r = _rpc(h, p, t, "kernel.fs.list",
                {"prefix": "/memory/"})
    assert s == 200
    paths = {e["path"] for e in r["result"]["entries"]}
    assert paths == {"/memory/x/1", "/memory/x/2"}


def test_quota_exceeded_via_rpc(daemon):
    h, p, t = daemon
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.ledger.create",
         {"pid": pid, "grants": {"fs_w_bytes": 50}})
    s, r = _rpc(h, p, t, "kernel.fs.write", {
        "pid": pid, "path": "/x",
        "content": base64.b64encode(b"x" * 100).decode("ascii"),
    })
    assert s == 200
    assert "error" in r
    assert "FsQuotaExceeded" in r["error"]["message"]
    # Write was rolled back.
    s, r = _rpc(h, p, t, "kernel.fs.exists", {"path": "/x"})
    assert r["result"]["exists"] is False


def test_phase4_does_not_break_earlier(daemon):
    h, p, t = daemon
    s, r = _rpc(h, p, t, "kernel.info", {})
    from cheetahclaws.kernel import SCHEMA_VERSION
    assert r["result"]["schema_version"] == SCHEMA_VERSION
    # Existing kernel.* surfaces still work.
    pid = _rpc(h, p, t, "kernel.agent.create",
               {"name": "x", "template": "t"})[1]["result"]["pid"]
    _rpc(h, p, t, "kernel.cap.create",
         {"pid": pid, "tool_grants": ["Read"]})
    _rpc(h, p, t, "kernel.sched.enqueue", {"pid": pid})
    _rpc(h, p, t, "kernel.mbox.create", {"pid": pid})
    _rpc(h, p, t, "kernel.registry.register",
         {"name": "/agents/x", "pid": pid})
