"""End-to-end tests for `cheetahclaws serve` + `cheetahclaws daemon ...`.

These tests boot the *real* daemon as a subprocess, hit it from the test
process via http.client, and verify discovery / auth / RPC / SSE / shutdown
behaviour at the process boundary — i.e. the integration the in-process
``test_daemon_server.py`` cannot cover.

Each test gets an isolated ``$HOME`` so daemon.json / daemon_token / logs
don't collide with the developer's real state.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Fixture: boot a real daemon under an isolated HOME ─────────────────────

@pytest.fixture
def daemon_proc(tmp_path):
    """Start `cheetahclaws serve --listen tcp://...` in a subprocess.

    Yields ``(proc, home, address, token)``.  Tears the process down by
    sending the ``system.shutdown`` RPC; falls back to terminate() on
    timeout.
    """
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["USERPROFILE"] = str(tmp_path)  # Windows: Path.home() reads this
    # Pin XDG_RUNTIME_DIR under tmp_path so any UDS path lands here too.
    env["XDG_RUNTIME_DIR"] = str(tmp_path / "xdg")

    proc = subprocess.Popen(
        [sys.executable, "cheetahclaws.py", "serve",
         "--listen", "tcp://127.0.0.1:0"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    cheetah_dir = tmp_path / ".cheetahclaws"
    discovery_file = cheetah_dir / "daemon.json"
    token_file = cheetah_dir / "daemon_token"

    # Wait up to 10 s for the daemon to bind and write discovery.
    deadline = time.monotonic() + 10.0
    info = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            pytest.fail(
                "daemon exited before binding\n"
                f"stdout: {stdout.decode('utf-8', 'replace')}\n"
                f"stderr: {stderr.decode('utf-8', 'replace')}"
            )
        if discovery_file.exists() and token_file.exists():
            try:
                info = json.loads(discovery_file.read_text(encoding="utf-8"))
                break
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.1)
    if info is None:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("daemon did not write discovery file in time")

    address = info["address"]  # "host:port"
    token = token_file.read_text(encoding="utf-8").strip()

    yield proc, tmp_path, address, token

    # Teardown: graceful shutdown via RPC, then terminate as fallback.
    if proc.poll() is None:
        try:
            _post_rpc(address, token, "system.shutdown")
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5)


# ── Helpers ────────────────────────────────────────────────────────────────

def _post_rpc(address: str, token: str, method: str,
              params=None, *, timeout=3.0):
    from cc_daemon import API_VERSION, API_VERSION_HEADER
    host, port_s = address.rsplit(":", 1)
    body_obj = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body_obj["params"] = params
    body = json.dumps(body_obj).encode("utf-8")
    conn = http.client.HTTPConnection(host, int(port_s), timeout=timeout)
    try:
        conn.request("POST", "/rpc", body=body,
                     headers={"Authorization": f"Bearer {token}",
                              "Content-Type": "application/json",
                              API_VERSION_HEADER: API_VERSION})
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else None
    finally:
        conn.close()


def _get(address: str, path: str, *, token: str | None = None,
         api_version: bool = False, timeout=3.0):
    host, port_s = address.rsplit(":", 1)
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if api_version:
        from cc_daemon import API_VERSION, API_VERSION_HEADER
        headers[API_VERSION_HEADER] = API_VERSION
    conn = http.client.HTTPConnection(host, int(port_s), timeout=timeout)
    try:
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8", errors="replace")
    finally:
        conn.close()


def _run_subcommand(args: list[str], home: Path, *, timeout=10.0):
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["XDG_RUNTIME_DIR"] = str(home / "xdg")
    proc = subprocess.run(
        [sys.executable, "cheetahclaws.py", *args],
        cwd=str(REPO_ROOT), env=env, timeout=timeout,
        capture_output=True,
    )
    return (proc.returncode,
            proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"))


# ── Boot + discovery ───────────────────────────────────────────────────────

def test_daemon_writes_discovery_and_token(daemon_proc):
    _proc, home, address, token = daemon_proc
    info_path = home / ".cheetahclaws" / "daemon.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["transport"] == "tcp"
    assert info["address"] == address
    assert info["pid"] > 0
    assert info["schema"] == 1
    # secrets.token_urlsafe(32) yields ~43 base64-url chars; floor at 40 so
    # an accidental shrink to 16 raw bytes (~22 chars) breaks loudly.
    assert len(token) >= 40


# ── RPC ────────────────────────────────────────────────────────────────────

def test_rpc_system_ping(daemon_proc):
    _proc, _home, address, token = daemon_proc
    status, body = _post_rpc(address, token, "system.ping")
    assert status == 200
    assert body == {"jsonrpc": "2.0", "id": 1, "result": "pong"}


def test_rpc_method_not_found(daemon_proc):
    _proc, _home, address, token = daemon_proc
    status, body = _post_rpc(address, token, "no.such.method")
    assert status == 200  # JSON-RPC errors travel inside the envelope
    assert body["error"]["code"] == -32601


def test_rpc_without_token_returns_401(daemon_proc):
    _proc, _home, address, _token = daemon_proc
    host, port_s = address.rsplit(":", 1)
    body = json.dumps({"jsonrpc": "2.0", "id": 1,
                       "method": "system.ping"}).encode("utf-8")
    conn = http.client.HTTPConnection(host, int(port_s), timeout=3.0)
    try:
        conn.request("POST", "/rpc", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 401
    finally:
        conn.close()


# ── Health endpoints ───────────────────────────────────────────────────────

def test_healthz_requires_auth_by_default(daemon_proc):
    _proc, _home, address, _token = daemon_proc
    status, _ = _get(address, "/healthz")  # no token
    assert status == 401


def test_healthz_with_token_returns_real_payload(daemon_proc):
    _proc, _home, address, token = daemon_proc
    status, body = _get(address, "/healthz", token=token)
    assert status == 200
    payload = json.loads(body)
    assert payload["status"] == "ok"
    assert "uptime_s" in payload
    assert "active_sessions" in payload
    assert "model" in payload  # health.py reads from cc_config


def test_metrics_with_token_returns_real_payload(daemon_proc):
    _proc, _home, address, token = daemon_proc
    status, body = _get(address, "/metrics", token=token)
    assert status == 200
    payload = json.loads(body)
    for key in ("uptime_s", "model", "active_sessions", "circuits",
                "daily_tokens", "daily_cost_usd"):
        assert key in payload, f"missing {key}"


# ── SSE ────────────────────────────────────────────────────────────────────

def test_events_stream_emits_heartbeat(daemon_proc):
    from cc_daemon import API_VERSION, API_VERSION_HEADER
    _proc, _home, address, token = daemon_proc
    host, port_s = address.rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port_s), timeout=25.0)
    try:
        conn.request("GET", "/events",
                     headers={"Authorization": f"Bearer {token}",
                              API_VERSION_HEADER: API_VERSION})
        resp = conn.getresponse()
        assert resp.status == 200
        assert "text/event-stream" in resp.getheader("Content-Type", "")
        # spike's heartbeat is 15s; wait up to 22s.
        deadline = time.monotonic() + 22.0
        buf = b""
        while time.monotonic() < deadline:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if b":" in buf and b"\n\n" in buf:
                # SSE heartbeat is a comment line (`:\n\n` in spike).
                break
        # Heartbeat presence: spike emits `:\n\n`; our format used
        # `:heartbeat ts\n\n`.  Accept either as a heartbeat marker.
        assert (b":heartbeat" in buf) or (b":\n\n" in buf), \
            f"no heartbeat in stream: {buf!r}"
    finally:
        conn.close()


# ── daemon subcommands ─────────────────────────────────────────────────────

def test_daemon_status_subcommand_when_running(daemon_proc):
    _proc, home, _address, _token = daemon_proc
    rc, stdout, stderr = _run_subcommand(["daemon", "status"], home)
    assert rc == 0, f"stderr: {stderr}"
    assert "transport:" in stdout
    assert "address:" in stdout
    assert "pong" in stdout


def test_daemon_status_when_not_running(tmp_path):
    rc, _stdout, stderr = _run_subcommand(["daemon", "status"], tmp_path)
    assert rc == 1
    assert "not running" in stderr.lower()


def test_daemon_stop_clears_discovery(daemon_proc):
    proc, home, _address, _token = daemon_proc
    rc, _stdout, stderr = _run_subcommand(["daemon", "stop"], home)
    assert rc == 0, f"stderr: {stderr}"
    assert "stopped" in (_stdout + stderr).lower()
    proc.wait(timeout=5)
    assert not (home / ".cheetahclaws" / "daemon.json").exists()


def test_daemon_logs_subcommand(daemon_proc):
    _proc, home, _address, _token = daemon_proc
    rc, stdout, _stderr = _run_subcommand(
        ["daemon", "logs", "-n", "20"], home)
    assert rc == 0
    # Bootstrap emits structured JSON log lines under serve mode.
    assert "bootstrap_done" in stdout or "daemon_listening" in stdout


def test_daemon_rotate_token_changes_file(daemon_proc):
    _proc, home, _address, token = daemon_proc
    token_path = home / ".cheetahclaws" / "daemon_token"
    rc, stdout, _stderr = _run_subcommand(["daemon", "rotate-token"], home)
    assert rc == 0
    assert "rotated" in stdout.lower()
    assert token_path.read_text().strip() != token


# ── F-2: SQLite persistence + cross-restart replay ────────────────────────

def _start_daemon(home: Path, *, wait_s: float = 10.0) -> tuple[subprocess.Popen, str, str]:
    """Boot a daemon under *home*; return (proc, address, token)."""
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["XDG_RUNTIME_DIR"] = str(home / "xdg")
    proc = subprocess.Popen(
        [sys.executable, "cheetahclaws.py", "serve",
         "--listen", "tcp://127.0.0.1:0"],
        cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    discovery_file = home / ".cheetahclaws" / "daemon.json"
    token_file = home / ".cheetahclaws" / "daemon_token"
    deadline = time.monotonic() + wait_s
    info = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            pytest.fail(
                "daemon exited before binding\n"
                f"stdout: {stdout.decode('utf-8', 'replace')}\n"
                f"stderr: {stderr.decode('utf-8', 'replace')}"
            )
        if discovery_file.exists() and token_file.exists():
            try:
                info = json.loads(discovery_file.read_text(encoding="utf-8"))
                break
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.1)
    if info is None:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("daemon did not write discovery file in time")
    return proc, info["address"], token_file.read_text(encoding="utf-8").strip()


def _stop_daemon(proc: subprocess.Popen, address: str, token: str) -> None:
    if proc.poll() is None:
        try:
            _post_rpc(address, token, "system.shutdown")
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5)


def test_sessions_db_initialised_on_first_serve(tmp_path):
    """F-2 schema: cheetahclaws serve initialises ~/.cheetahclaws/sessions.db
    with the daemon tables before accepting any RPC."""
    proc, address, token = _start_daemon(tmp_path)
    try:
        # Hit ping so we know the daemon is fully up.
        status, _ = _post_rpc(address, token, "system.ping")
        assert status == 200
    finally:
        _stop_daemon(proc, address, token)

    db_path = tmp_path / ".cheetahclaws" / "sessions.db"
    assert db_path.exists()
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    finally:
        conn.close()
    for t in ("daemon_events", "agent_runs", "agent_iterations", "jobs",
              "monitor_subscriptions", "monitor_reports", "bridges",
              "schema_meta"):
        assert t in names, f"missing table after serve: {t}"


def test_monitor_subscribe_via_rpc_survives_daemon_restart(tmp_path):
    """F-3 headline:  /monitor subscribe via RPC persists into SQLite,
    daemon stops, daemon starts again, subscription is still listed."""
    proc1, addr1, token1 = _start_daemon(tmp_path)
    try:
        status, body = _post_rpc(addr1, token1, "monitor.subscribe",
                                  params={"topic": "arxiv",
                                          "schedule": "daily",
                                          "channels": ["console"]})
        assert status == 200
        assert body["result"]["topic"] == "arxiv"

        status, body = _post_rpc(addr1, token1, "monitor.list")
        assert status == 200
        topics = {s["topic"] for s in body["result"]["subscriptions"]}
        assert "arxiv" in topics
    finally:
        _stop_daemon(proc1, addr1, token1)

    proc2, addr2, token2 = _start_daemon(tmp_path)
    try:
        status, body = _post_rpc(addr2, token2, "monitor.list")
        assert status == 200
        topics = {s["topic"] for s in body["result"]["subscriptions"]}
        assert "arxiv" in topics, \
            f"subscription did not survive restart: {topics}"
    finally:
        _stop_daemon(proc2, addr2, token2)


def test_events_persist_in_sqlite_across_daemon_restart(tmp_path):
    """Publish events on daemon A (echo.ping fires `ping_received`),
    stop daemon A, start daemon B against the same data dir, and
    verify GET /events?since=0 replays the events from SQLite.
    This is the headline F-2 user-visible win for SSE clients
    (Web UI / future bridges) that survive daemon restarts.
    """
    from cc_daemon import API_VERSION, API_VERSION_HEADER

    # Boot A, publish a few events via echo.ping.
    proc1, addr1, token1 = _start_daemon(tmp_path)
    try:
        for i in range(3):
            status, body = _post_rpc(addr1, token1, "echo.ping",
                                     params={"i": i})
            assert status == 200
            assert body["result"]["pong"] is True
    finally:
        _stop_daemon(proc1, addr1, token1)

    # Boot B against the same HOME → same sessions.db.
    proc2, addr2, token2 = _start_daemon(tmp_path)
    try:
        host, port_s = addr2.rsplit(":", 1)
        conn = http.client.HTTPConnection(host, int(port_s), timeout=10.0)
        conn.request("GET", "/events?since=0",
                     headers={"Authorization": f"Bearer {token2}",
                              API_VERSION_HEADER: API_VERSION})
        resp = conn.getresponse()
        assert resp.status == 200
        # Read enough to capture all replayed events; daemon may also
        # emit a heartbeat — bail when we've seen all 3 pings or 5 s pass.
        deadline = time.monotonic() + 5.0
        buf = b""
        while time.monotonic() < deadline:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if buf.count(b"event: ping_received") >= 3:
                break
        conn.close()
    finally:
        _stop_daemon(proc2, addr2, token2)

    text = buf.decode("utf-8", errors="replace")
    assert text.count("event: ping_received") >= 3, \
        f"missing replayed events:\n{text[:500]}"

