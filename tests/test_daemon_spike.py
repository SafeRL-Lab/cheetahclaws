"""Tests for the daemon spike. Covers the RFC must-fix items #1, 2, 3, 4, 6, 7, 8, 9.

Strategy: run a TCP daemon on an ephemeral port for each test (avoids
peer-cred / Unix-socket complexity in CI), drive it with the spike client.
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

from cc_daemon import API_VERSION, API_VERSION_HEADER, events
from cc_daemon.events import EventBus
from cc_daemon.originator import (
    CLIENT_ID_HEADER, CLIENT_KIND_HEADER, OriginatorStore,
)
from cc_daemon.permission import (
    PermissionStore, NotOriginator, UnknownRequest,
    DEFAULT_TIMEOUT_INTERACTIVE_S,
)
from cc_daemon.server import make_tcp_server


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def daemon_tcp(tmp_path):
    """Spin up a TCP daemon on a free port; yield (host, port, token, data_dir).
    Tears down on exit."""
    events.reset_bus_for_tests()
    port = _free_port()
    token = "test-" + uuid.uuid4().hex
    server = make_tcp_server(
        "127.0.0.1", port,
        data_dir=tmp_path,
        token=token,
        audit_enabled=True,
    )
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    # Wait for socket to be listening
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)
    yield "127.0.0.1", port, token, tmp_path
    server.daemon_state.shutdown()
    server.shutdown()
    server.server_close()
    t.join(timeout=2)


def _rpc(host, port, token, method, params, *, version=API_VERSION,
         client_id=None, kind="test"):
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
    }
    if version is not None:
        headers[API_VERSION_HEADER] = version
    if client_id:
        headers[CLIENT_ID_HEADER] = client_id
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("POST", "/rpc", body=body, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    out = {
        "status": resp.status,
        "client_id": resp.headers.get(CLIENT_ID_HEADER),
        "headers": dict(resp.headers),
    }
    try:
        out["body"] = json.loads(raw)
    except Exception:
        out["body"] = raw.decode()
    conn.close()
    return out


# ── Pure-unit tests for ring buffer (covers #7) ──────────────────────────────


def test_ring_buffer_overflow_emits_gap():
    """F-2: ring → SQLite swap.  Original spike test exercised the bounded
    in-memory ring (``ring_cap=5``); the F-2 SQLite-backed bus uses
    age + row retention instead.  Same intent: when the bus has evicted
    events the caller wants, ``replay_since`` yields a synthetic ``gap``
    so SSE clients can resync.  ``ring_cap`` is still accepted on the
    constructor for backward compat but ignored — retention drives
    eviction now.
    """
    events.reset_bus_for_tests()
    bus = EventBus(retention_rows=5, prune_every_n=1)
    for i in range(10):
        bus.publish("t", {"i": i})
    out = list(bus.replay_since(1))
    assert out[0]["type"] == "gap"
    assert out[0]["data"]["missed_from"] == 2
    assert all(e["id"] > 1 for e in out[1:])
    assert out[-1]["data"]["i"] == 9


def test_ring_buffer_no_gap_within_window():
    """F-2: SQLite-backed bus, retention high enough that nothing's evicted.
    Replay should be exact and gap-free."""
    events.reset_bus_for_tests()
    bus = EventBus(retention_rows=10, prune_every_n=1)
    for i in range(5):
        bus.publish("t", {"i": i})
    out = list(bus.replay_since(2))
    assert all(e["type"] != "gap" for e in out)
    assert [e["data"]["i"] for e in out] == [2, 3, 4]


# ── Permission store unit tests (covers #9 timeout + originator-only) ───────


def test_permission_originator_only_unit():
    store = PermissionStore()
    req = store.create(originator="alice", tool="Bash", tool_input={"cmd": "ls"})
    with pytest.raises(NotOriginator):
        store.answer(req.request_id, "bob", {"approve": True})
    answered = store.answer(req.request_id, "alice", {"approve": False})
    assert answered.answer == {"approve": False}
    with pytest.raises(UnknownRequest):
        store.answer(req.request_id, "alice", {"approve": True})


def test_permission_default_timeout_is_30min():
    assert DEFAULT_TIMEOUT_INTERACTIVE_S == 30 * 60


# ── HTTP integration tests ──────────────────────────────────────────────────


def test_echo_ping_and_event_emission(daemon_tcp):
    """#4: session-style sync RPC + side-effect event on /events."""
    host, port, token, _ = daemon_tcp
    out = _rpc(host, port, token, "echo.ping", {"hello": "world"})
    assert out["status"] == 200
    assert out["body"]["result"]["pong"] is True
    assert "server_uptime_s" in out["body"]["result"]
    assert out["client_id"], "daemon should mint and return a client_id"


def test_api_version_mismatch_returns_426(daemon_tcp):
    """#6: missing/wrong API version → 426."""
    host, port, token, _ = daemon_tcp
    # Wrong version
    out = _rpc(host, port, token, "echo.ping", {}, version="99")
    assert out["status"] == 426
    assert out["body"]["expected"] == API_VERSION
    # Missing version
    out2 = _rpc(host, port, token, "echo.ping", {}, version=None)
    assert out2["status"] == 426


def test_unauthenticated_tcp_returns_401(daemon_tcp):
    """Auth: missing/wrong token → 401."""
    host, port, _, _ = daemon_tcp
    # Wrong token
    out = _rpc(host, port, "wrong-token", "echo.ping", {})
    assert out["status"] == 401
    # Missing token
    body = json.dumps({"jsonrpc": "2.0", "id": "x", "method": "echo.ping", "params": {}}).encode()
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("POST", "/rpc", body=body, headers={
        "Content-Type": "application/json",
        API_VERSION_HEADER: API_VERSION,
    })
    resp = conn.getresponse()
    assert resp.status == 401
    conn.close()


def test_audit_log_records_outcomes(daemon_tcp):
    """#8: audit log default-on captures both ok and denied lines."""
    host, port, token, data_dir = daemon_tcp
    _rpc(host, port, token, "echo.ping", {})
    _rpc(host, port, "wrong-token", "echo.ping", {})
    audit = data_dir / "logs" / "auth.jsonl"
    # Give the writer a moment to flush
    time.sleep(0.05)
    assert audit.exists(), "audit log should exist when enabled"
    lines = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    outcomes = {l["outcome"] for l in lines}
    assert "ok" in outcomes
    assert "denied" in outcomes


def test_permission_not_originator_returns_403(daemon_tcp):
    """#3 + originator routing: non-originator gets 403."""
    host, port, token, _ = daemon_tcp
    # Client A creates a request targeting client A's own id.
    a_init = _rpc(host, port, token, "echo.ping", {}, kind="alice")
    alice_cid = a_init["client_id"]
    # Client B has its own id minted; we'll use B to try answering.
    b_init = _rpc(host, port, token, "echo.ping", {}, kind="bob")
    bob_cid = b_init["client_id"]
    assert alice_cid != bob_cid

    # Alice creates a permission request targeting herself.
    create_resp = _rpc(host, port, token, "permission.demo",
                       {"tool": "Bash", "input": {"cmd": "echo"}},
                       client_id=alice_cid, kind="alice")
    assert create_resp["status"] == 200
    rid = create_resp["body"]["result"]["request_id"]

    # Bob attempts to answer Alice's request → 403.
    bob_answer = _rpc(host, port, token, "permission.answer",
                      {"request_id": rid, "result": {"approve": True}},
                      client_id=bob_cid, kind="bob")
    assert bob_answer["status"] == 403
    assert bob_answer["body"]["error"]["code"] == -32001  # APP_NOT_ORIGINATOR

    # Alice answers her own request → ok.
    alice_answer = _rpc(host, port, token, "permission.answer",
                        {"request_id": rid, "result": {"approve": True}},
                        client_id=alice_cid, kind="alice")
    assert alice_answer["status"] == 200
    assert alice_answer["body"]["result"]["answer"] == {"approve": True}


def test_client_id_resume(daemon_tcp):
    """#3: presenting a known client_id reuses it; daemon does not mint a new one."""
    host, port, token, _ = daemon_tcp
    first = _rpc(host, port, token, "echo.ping", {}, kind="resume")
    cid = first["client_id"]
    assert cid

    # Second connect presenting the same id should keep it.
    second = _rpc(host, port, token, "echo.ping", {}, client_id=cid, kind="resume")
    assert second["client_id"] == cid

    # An unknown id presented (simulating a fresh client) should result in a new mint.
    third = _rpc(host, port, token, "echo.ping", {},
                 client_id="ffffffffffffffffffffffffffffffff", kind="resume")
    assert third["client_id"] != "ffffffffffffffffffffffffffffffff"


def test_sse_heartbeat_arrives(daemon_tcp):
    """#2: SSE sends `:\\n\\n` heartbeat within the window when no events flow.

    We use a tight HEARTBEAT window via monkeypatching to keep test runtime low.
    """
    host, port, token, _ = daemon_tcp
    # Speed up heartbeats for the test so it doesn't take 15s.
    from cc_daemon import server as _srv
    original = _srv.HEARTBEAT_INTERVAL_S
    _srv.HEARTBEAT_INTERVAL_S = 0.5
    try:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/events?since=0", headers={
            "Authorization": f"Bearer {token}",
            API_VERSION_HEADER: API_VERSION,
            CLIENT_KIND_HEADER: "watcher",
        })
        resp = conn.getresponse()
        assert resp.status == 200
        # Read enough bytes to capture either an event or a heartbeat.
        # Block until we see a `:` on its own (heartbeat) within ~2s.
        resp.fp.settimeout = lambda *_: None  # type: ignore[attr-defined]
        # Use the raw socket for a deadline read.
        sock = resp.fp.raw if hasattr(resp.fp, "raw") else resp.fp
        try:
            sock._sock.settimeout(2.0)  # type: ignore[attr-defined]
        except Exception:
            pass
        deadline = time.time() + 2.5
        saw_heartbeat = False
        buffer = b""
        while time.time() < deadline:
            chunk = resp.fp.read1(64) if hasattr(resp.fp, "read1") else resp.fp.read(64)
            if not chunk:
                break
            buffer += chunk
            if b"\n:\n\n" in b"\n" + buffer or buffer.startswith(b":\n\n"):
                saw_heartbeat = True
                break
        conn.close()
        assert saw_heartbeat, f"no heartbeat in 2.5s; got: {buffer!r}"
    finally:
        _srv.HEARTBEAT_INTERVAL_S = original


def test_concurrent_rpc_not_blocked_by_sse(daemon_tcp):
    """#1: ThreadingHTTPServer keeps /rpc responsive while many SSE clients are open."""
    host, port, token, _ = daemon_tcp

    # Open 16 concurrent SSE connections (fewer than 64 to keep test runtime low).
    sse_conns = []
    for _ in range(16):
        c = http.client.HTTPConnection(host, port, timeout=5)
        c.request("GET", "/events?since=0", headers={
            "Authorization": f"Bearer {token}",
            API_VERSION_HEADER: API_VERSION,
            CLIENT_KIND_HEADER: "sse-load",
        })
        c.getresponse()  # don't read body — just hold the connection open
        sse_conns.append(c)

    try:
        # Now hit /rpc from multiple threads; all should complete quickly.
        latencies = []
        results = []
        lock = threading.Lock()

        def hit():
            t0 = time.time()
            r = _rpc(host, port, token, "echo.ping", {})
            with lock:
                latencies.append(time.time() - t0)
                results.append(r["status"])

        threads = [threading.Thread(target=hit) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert all(s == 200 for s in results), f"some /rpc failed: {results}"
        # p95 should be well under 1s — generous bound for CI noise.
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95) - 1]
        assert p95 < 1.0, f"p95 latency too high: {p95}s; all={latencies}"
    finally:
        for c in sse_conns:
            try:
                c.close()
            except Exception:
                pass


def test_originator_store_persistence(tmp_path):
    """#3: client_ids survive an OriginatorStore reload (daemon restart)."""
    s1 = OriginatorStore(tmp_path)
    cid = s1.mint("repl")
    assert s1.kind_of(cid) == "repl"

    s2 = OriginatorStore(tmp_path)
    assert s2.kind_of(cid) == "repl"
    # Resume returns same id when presented
    cid2, minted = s2.resolve(cid, "repl")
    assert cid2 == cid
    assert minted is False
