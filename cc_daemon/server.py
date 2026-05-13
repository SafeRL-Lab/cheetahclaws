"""server.py — HTTP request handler + threaded server (TCP and Unix-socket).

The handler dispatches by URL path: /rpc, /events, /healthz, /readyz.
Both transports share the same handler; only setup (auth, peer addressing)
differs.

Concurrency: ThreadingMixIn handles each request on its own thread. SSE
connections live indefinitely and pull events from a per-subscriber Queue
with a 15s blocking timeout, sending an SSE comment heartbeat on each
timeout. /rpc requests run independently and never block on /events.
"""
from __future__ import annotations

import json
import os
import queue as _queue
import socket
import socketserver
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

from . import API_VERSION, API_VERSION_HEADER, events, permission
from .auth import AuditLog, AuthGate, AuthInfo, RateLimited, Unauthenticated
from .events import HEARTBEAT_INTERVAL_S, format_sse, heartbeat_frame
from .methods import register as register_methods
from .originator import CLIENT_ID_HEADER, CLIENT_KIND_HEADER, OriginatorStore
from .rpc import CallContext, RpcRegistry


# ── Shared daemon state ──────────────────────────────────────────────────────


class DaemonState:
    """The mutable singletons a request handler needs. One instance per
    daemon process, attached to the server via `server.daemon_state`."""

    def __init__(
        self,
        *,
        transport: str,
        data_dir: Path,
        token: Optional[str],
        expected_uid: Optional[int],
        audit_enabled: bool,
        unauthenticated_metrics: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        self.transport = transport
        self.data_dir = data_dir
        self.unauthenticated_metrics = unauthenticated_metrics
        self.config = config or {}
        self.audit = AuditLog(data_dir / "logs" / "auth.jsonl", enabled=audit_enabled)
        self.gate = AuthGate(
            transport,
            token=token,
            expected_uid=expected_uid,
            audit=self.audit,
        )
        self.originators = OriginatorStore(data_dir)
        self.permissions = permission.PermissionStore()
        self.permissions.start_janitor()
        self.rpc = RpcRegistry()
        register_methods(self.rpc, self.permissions)
        from . import (
            system_methods, monitor_methods, agent_methods, proactive_methods,
            bridge_methods, session_methods,
        )
        system_methods.register(self.rpc, self)
        monitor_methods.register(self.rpc, self)
        agent_methods.register(self.rpc, self)
        proactive_methods.register(self.rpc, self)
        # RFC 0002 F-6/7/8 — bridge_methods are registered unconditionally
        # so a caller probing for `bridge.list` always gets a response, but
        # `bridge.start` itself enforces the per-kind CHEETAHCLAWS_ENABLE_F<n>
        # flag so a daemon that's not opted in stays REPL-equivalent.
        bridge_methods.register(self.rpc, self)
        # RFC 0002 F-6 Phase 2 — session.send / session.reply /
        # session.list_recent.  The methods are I/O-free message-passing
        # primitives, safe to register on any daemon (no feature flag).
        session_methods.register(self.rpc, self)
        self.shutdown_event = threading.Event()

    def shutdown(self) -> None:
        # Wake up SSE loops so connections close cleanly
        events.get_bus().publish("shutdown", {"reason": "graceful"})
        self.shutdown_event.set()
        self.permissions.stop()


# ── Request handler ──────────────────────────────────────────────────────────


class DaemonRequestHandler(BaseHTTPRequestHandler):
    server_version = "cheetahclaws-daemon/0"
    # Quiet stdlib stderr logging; project should route through its own log
    # facility if desired.
    def log_message(self, fmt, *args):  # noqa
        return

    @property
    def state(self) -> DaemonState:
        return self.server.daemon_state  # type: ignore[attr-defined]

    # -- entry points -------------------------------------------------------

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        is_health_path = path in ("/healthz", "/readyz", "/metrics")

        # Health endpoints can opt out of auth (Prometheus scrapers, etc.).
        if is_health_path and self.state.unauthenticated_metrics:
            self._serve_health(path)
            return

        try:
            auth = self._authenticate()
        except RateLimited:
            self._send_error(429, "rate limited")
            return
        except Unauthenticated:
            self._send_error(401, "unauthenticated")
            return

        if is_health_path:
            # Auth passed; skip the API-version gate so a monitoring tool
            # that doesn't speak our protocol can still scrape.
            self._serve_health(path)
            return

        if not self._check_api_version():
            return
        client_id = self._resolve_client_id(auth)
        if path == "/events":
            self._handle_events(client_id)
            return
        self._send_error(404, "not found")

    # ── Health helpers ────────────────────────────────────────────────────

    def _serve_health(self, path: str) -> None:
        try:
            import health as _health
            payload = _health.payload_for(path, self.state.config)
        except Exception:
            payload = {"status": "ok"}
        code = 200
        if path == "/readyz" and payload.get("status") == "degraded":
            code = 503
        self._send_json(code, payload)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        try:
            auth = self._authenticate()
        except RateLimited:
            self._send_error(429, "rate limited")
            return
        except Unauthenticated:
            self._send_error(401, "unauthenticated")
            return
        if not self._check_api_version():
            return
        client_id = self._resolve_client_id(auth)
        if path == "/rpc":
            self._handle_rpc(client_id, auth)
            return
        self._send_error(404, "not found")

    # -- auth + headers -----------------------------------------------------

    def _authenticate(self) -> AuthInfo:
        return self.state.gate.authenticate(
            self.connection,
            self.client_address,
            self.headers,
        )

    def _check_api_version(self) -> bool:
        v = self.headers.get(API_VERSION_HEADER, "")
        if v == API_VERSION:
            return True
        # 426 Upgrade Required — RFC §6 patch we asked for.
        body = json.dumps({
            "error": "api_version_mismatch",
            "expected": API_VERSION,
            "got": v or None,
            "header": API_VERSION_HEADER,
        }).encode()
        self.send_response(426)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(API_VERSION_HEADER, API_VERSION)
        self.end_headers()
        self.wfile.write(body)
        return False

    def _resolve_client_id(self, auth: AuthInfo) -> str:
        presented = self.headers.get(CLIENT_ID_HEADER)
        kind = self.headers.get(CLIENT_KIND_HEADER, "unknown")
        cid, _ = self.state.originators.resolve(presented, kind)
        return cid

    # -- /rpc ---------------------------------------------------------------

    def _handle_rpc(self, client_id: str, auth: AuthInfo) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error(400, "bad content-length")
            return
        if length <= 0:
            self._send_error(400, "empty body")
            return
        raw = self.rfile.read(length)
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(200, {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            })
            return

        ctx = CallContext(
            client_id=client_id,
            transport=auth.transport,
            api_version=API_VERSION,
        )
        response, status = self.state.rpc.dispatch(envelope, ctx)
        if response is None:
            self.send_response(204)
            self.send_header(CLIENT_ID_HEADER, client_id)
            self.end_headers()
            return
        body = json.dumps(response).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(CLIENT_ID_HEADER, client_id)
        self.end_headers()
        self.wfile.write(body)

    # -- /events ------------------------------------------------------------

    def _handle_events(self, client_id: str) -> None:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        try:
            since = int(qs.get("since", ["0"])[0])
        except ValueError:
            since = 0

        bus = events.get_bus()
        sub = bus.subscribe()
        # Important: open the SSE response BEFORE backfilling, so the client
        # sees headers immediately and replay events stream in order.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header(CLIENT_ID_HEADER, client_id)
        self.end_headers()

        try:
            # Backfill
            for evt in bus.replay_since(since):
                if not self._sse_send(format_sse(evt)):
                    return
            # Tail
            while not self.state.shutdown_event.is_set():
                try:
                    evt = sub.get(timeout=HEARTBEAT_INTERVAL_S)
                except _queue.Empty:
                    if not self._sse_send(heartbeat_frame()):
                        return
                    continue
                if not self._sse_send(format_sse(evt)):
                    return
                if evt.get("type") == "shutdown":
                    return
        finally:
            bus.unsubscribe(sub)

    def _sse_send(self, frame: bytes) -> bool:
        try:
            self.wfile.write(frame)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    # -- helpers ------------------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(API_VERSION_HEADER, API_VERSION)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Server classes ───────────────────────────────────────────────────────────


class ThreadedTCPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Default 5 starves us under bursts; long-lived SSE plus a wave of /rpc
    # easily overflows the kernel listen backlog and triggers SYN retries.
    request_queue_size = 256
    daemon_state: DaemonState  # set after construction


# UnixStreamServer is unavailable on Windows; build the subclass only where
# socketserver exposes it.  Code paths that try to construct one on Windows
# raise from the helpers below instead.
ThreadedUnixServer = None  # type: ignore[assignment]
if hasattr(socketserver, "UnixStreamServer"):
    class ThreadedUnixServer(socketserver.ThreadingMixIn,                    # type: ignore[no-redef]
                              socketserver.UnixStreamServer):
        daemon_threads = True
        request_queue_size = 256
        daemon_state: DaemonState

        # BaseHTTPRequestHandler reads `client_address` to fill log entries; for
        # Unix sockets `accept()` returns ("", None). Synthesize a stable repr.
        def get_request(self):
            sock, _ = self.socket.accept()
            return sock, ("unix-socket", 0)

        def server_bind(self):
            # Remove leftover socket file from a prior crash, then bind.
            path = self.server_address
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            # Ensure parent dir is 0700 before bind.
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            os.chmod(Path(path).parent, 0o700)
            super().server_bind()
            os.chmod(path, 0o600)


# ── Construction helpers ─────────────────────────────────────────────────────


def make_tcp_server(
    host: str,
    port: int,
    *,
    data_dir: Path,
    token: str,
    audit_enabled: bool = True,
    unauthenticated_metrics: bool = False,
    config: Optional[dict] = None,
) -> ThreadedTCPServer:
    server = ThreadedTCPServer((host, port), DaemonRequestHandler)
    server.daemon_state = DaemonState(
        transport="tcp",
        data_dir=data_dir,
        token=token,
        expected_uid=None,
        audit_enabled=audit_enabled,
        unauthenticated_metrics=unauthenticated_metrics,
        config=config,
    )
    return server


def make_unix_server(
    socket_path: Path,
    *,
    data_dir: Path,
    expected_uid: Optional[int] = None,
    audit_enabled: bool = True,
    unauthenticated_metrics: bool = False,
    config: Optional[dict] = None,
):
    if ThreadedUnixServer is None:
        raise RuntimeError(
            "Unix-socket transport is unavailable on this platform "
            "(socketserver.UnixStreamServer missing); use TCP loopback instead."
        )
    server = ThreadedUnixServer(str(socket_path), DaemonRequestHandler)
    if expected_uid is None:
        expected_uid = os.geteuid()
    server.daemon_state = DaemonState(
        transport="unix",
        data_dir=data_dir,
        token=None,
        expected_uid=expected_uid,
        audit_enabled=audit_enabled,
        unauthenticated_metrics=unauthenticated_metrics,
        config=config,
    )
    return server
