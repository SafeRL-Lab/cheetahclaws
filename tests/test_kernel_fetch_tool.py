"""Tests for kernel.tools.fetch_tool (RFC 0025).

Most security-sensitive HTTP boundary the kernel ships. Tests
focus on: scheme allowlist, capability + DNS + IP gate, redirect
re-validation + auth-strip, size cap, deterministic local server
end-to-end.
"""
from __future__ import annotations

import http.server
import socket
import socketserver
import threading
import time
from urllib.parse import urlparse

import pytest

from cheetahclaws.kernel import (
    FETCH_TOOL,
    Kernel,
    ToolNetDenied,
    ToolRegistry,
    register_builtin_tools,
    register_fetch_tool,
)
from cheetahclaws.kernel.tools.fetch_tool import (
    DEFAULT_MAX_BYTES,
    DEFAULT_TIMEOUT_S,
    _is_private_ip,
    _validate_max_bytes,
    _validate_max_redirects,
    _validate_method,
    _validate_timeout,
    _validate_url,
    fetch_handler,
)
from cheetahclaws.kernel.tools.registry import (
    ToolContext,
    ToolFailed,
    ToolInvalidArgs,
    dispatch_tool_call,
)


# ── Opt-in invariant ─────────────────────────────────────────────────


def test_fetch_not_in_builtin():
    r = ToolRegistry()
    register_builtin_tools(r)
    assert "Fetch" not in r.list()


def test_register_fetch_adds_it():
    r = ToolRegistry()
    register_builtin_tools(r)
    name = register_fetch_tool(r)
    assert name == "Fetch"
    assert "Fetch" in r.list()


# ── _is_private_ip ──────────────────────────────────────────────────


@pytest.mark.parametrize("ip,expected", [
    ("127.0.0.1",         True),
    ("127.255.255.255",   True),
    ("10.0.0.1",          True),
    ("10.255.255.255",    True),
    ("172.16.0.1",        True),
    ("172.31.255.255",    True),
    ("192.168.0.1",       True),
    ("192.168.255.255",   True),
    ("169.254.169.254",   True),    # AWS / GCP metadata
    ("169.254.1.1",       True),    # link-local
    ("0.0.0.0",           True),    # unspecified
    ("224.0.0.1",         True),    # multicast
    ("::1",               True),    # IPv6 loopback
    ("fc00::1",           True),    # IPv6 unique local
    ("fe80::1",           True),    # IPv6 link local
    # Public IPs:
    ("1.1.1.1",           False),
    ("8.8.8.8",           False),
    ("104.21.45.10",      False),
])
def test_is_private_ip_classification(ip, expected):
    assert _is_private_ip(ip) is expected


def test_is_private_ip_unparseable_is_private():
    """Defensively classify unparseable IPs as private."""
    assert _is_private_ip("not-an-ip") is True
    assert _is_private_ip("") is True


# ── URL validation ──────────────────────────────────────────────────


def test_url_valid_https():
    assert _validate_url("https://example.com/x") == "https://example.com/x"


def test_url_valid_http():
    assert _validate_url("http://example.com/") == "http://example.com/"


@pytest.mark.parametrize("bad", [
    "file:///etc/passwd",
    "ftp://example.com",
    "gopher://example.com",
    "data:text/html,<script>...",
    "javascript:alert(1)",
])
def test_url_rejects_non_http_scheme(bad):
    with pytest.raises(ToolInvalidArgs):
        _validate_url(bad)


def test_url_rejects_empty():
    with pytest.raises(ToolInvalidArgs):
        _validate_url("")


def test_url_rejects_no_host():
    with pytest.raises(ToolInvalidArgs):
        _validate_url("https://")


def test_url_rejects_non_string():
    with pytest.raises(ToolInvalidArgs):
        _validate_url(123)


# ── Method validation ───────────────────────────────────────────────


def test_method_get_post_head():
    assert _validate_method("GET")  == "GET"
    assert _validate_method("get")  == "GET"
    assert _validate_method("POST") == "POST"
    assert _validate_method("HEAD") == "HEAD"


@pytest.mark.parametrize("bad", ["PUT", "DELETE", "PATCH", "OPTIONS",
                                    "CONNECT", "TRACE"])
def test_method_rejects_others(bad):
    with pytest.raises(ToolInvalidArgs):
        _validate_method(bad)


def test_method_rejects_non_string():
    with pytest.raises(ToolInvalidArgs):
        _validate_method(123)


# ── Numeric validation ──────────────────────────────────────────────


def test_max_bytes_default():
    assert _validate_max_bytes(None) == DEFAULT_MAX_BYTES


def test_max_bytes_within_range():
    assert _validate_max_bytes(2048) == 2048


def test_max_bytes_too_low():
    with pytest.raises(ToolInvalidArgs):
        _validate_max_bytes(100)


def test_max_bytes_too_high():
    with pytest.raises(ToolInvalidArgs):
        _validate_max_bytes(100 * 1024 * 1024)


def test_timeout_default():
    assert _validate_timeout(None) == DEFAULT_TIMEOUT_S


def test_timeout_zero_rejected():
    with pytest.raises(ToolInvalidArgs):
        _validate_timeout(0)


def test_timeout_too_high():
    with pytest.raises(ToolInvalidArgs):
        _validate_timeout(1000)


def test_max_redirects_default():
    from cheetahclaws.kernel.tools.fetch_tool import DEFAULT_MAX_REDIRECTS
    assert _validate_max_redirects(None) == DEFAULT_MAX_REDIRECTS


def test_max_redirects_zero_ok():
    assert _validate_max_redirects(0) == 0


def test_max_redirects_too_high():
    with pytest.raises(ToolInvalidArgs):
        _validate_max_redirects(100)


# ── Local HTTP server fixture ───────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CannedHandler(http.server.BaseHTTPRequestHandler):
    """Echo + redirect + various status responses."""
    canned_responses: dict = {}

    def log_message(self, format, *args):
        return  # silent

    def do_GET(self):
        self._respond("GET")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self._body = self.rfile.read(length) if length > 0 else b""
        self._respond("POST")

    def do_HEAD(self):
        self._respond("HEAD", body=False)

    def _respond(self, method: str, body: bool = True):
        path = self.path
        # Special path: /redirect/<count> redirects to /
        if path.startswith("/redirect/"):
            try:
                n = int(path.rsplit("/", 1)[1])
            except ValueError:
                n = 0
            target = "/" if n <= 1 else f"/redirect/{n - 1}"
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # Special: /redirect-to-private redirects to internal IP.
        if path == "/redirect-to-private":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # Special: /huge — emit 1 MB of 'X'
        if path == "/huge":
            data = b"X" * (1024 * 1024)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if body:
                self.wfile.write(data)
            return
        # Special: /echo-headers — return incoming headers as text
        if path == "/echo-headers":
            buf = []
            for k in self.headers:
                buf.append(f"{k}: {self.headers[k]}")
            data = "\n".join(buf).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if body:
                self.wfile.write(data)
            return
        # Special: /echo-body
        if path == "/echo-body":
            data = getattr(self, "_body", b"")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if body:
                self.wfile.write(data)
            return
        # Default: 200 OK with text.
        data = b"hello from canned server"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("X-Test-Marker", "yes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if body:
            self.wfile.write(data)


@pytest.fixture
def http_server():
    """Start a local HTTP server. Yields (port, base_url)."""
    port = _free_port()
    server = socketserver.TCPServer(("127.0.0.1", port), _CannedHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield port, f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()
    t.join(timeout=2)


# ── Capability + IP gates ───────────────────────────────────────────


class _AllowAllKernel:
    """Minimal kernel-like stub: cap.check_net always True."""
    class _Cap:
        def check_net(self, pid, host): return True
        def check_tool(self, pid, tool): return True
        def check_fs(self, pid, path, mode): return True
    cap = _Cap()


class _DenyKernel:
    class _Cap:
        def check_net(self, pid, host): return False
        def check_tool(self, pid, tool): return True
        def check_fs(self, pid, path, mode): return True
    cap = _Cap()


def test_handler_capability_denied():
    """cap.check_net=False → ToolNetDenied."""
    ctx = ToolContext(pid=1, kernel=_DenyKernel())
    with pytest.raises(ToolNetDenied):
        fetch_handler({"url": "https://example.com/"}, ctx)


def test_handler_localhost_blocked(monkeypatch):
    """Even with cap allow, resolving to 127.0.0.1 → ToolNetDenied."""
    monkeypatch.setattr(socket, "gethostbyname", lambda h: "127.0.0.1")
    ctx = ToolContext(pid=1, kernel=_AllowAllKernel())
    with pytest.raises(ToolNetDenied) as e:
        fetch_handler({"url": "https://attacker.example/"}, ctx)
    assert "127.0.0.1" in str(e.value)


def test_handler_metadata_endpoint_blocked(monkeypatch):
    """169.254.169.254 (AWS/GCP metadata) blocked."""
    monkeypatch.setattr(socket, "gethostbyname",
                         lambda h: "169.254.169.254")
    ctx = ToolContext(pid=1, kernel=_AllowAllKernel())
    with pytest.raises(ToolNetDenied):
        fetch_handler({"url": "https://attacker.example/"}, ctx)


def test_handler_private_ip_blocked(monkeypatch):
    """10.0.0.1 blocked."""
    monkeypatch.setattr(socket, "gethostbyname", lambda h: "10.0.0.1")
    ctx = ToolContext(pid=1, kernel=_AllowAllKernel())
    with pytest.raises(ToolNetDenied):
        fetch_handler({"url": "https://attacker.example/"}, ctx)


def test_handler_dns_failure_raises(monkeypatch):
    def fail_dns(h):
        raise socket.gaierror("Name not known")
    monkeypatch.setattr(socket, "gethostbyname", fail_dns)
    ctx = ToolContext(pid=1, kernel=_AllowAllKernel())
    with pytest.raises(ToolFailed):
        fetch_handler({"url": "https://nonexistent.example/"}, ctx)


# ── End-to-end via local server ─────────────────────────────────────


class _OverrideDnsKernel:
    """Kernel stub that allows the real localhost server. Used with
    monkeypatch on _is_private_ip to allow the test's 127.0.0.1
    bypass."""
    class _Cap:
        def check_net(self, pid, host): return True
        def check_tool(self, pid, tool): return True
        def check_fs(self, pid, path, mode): return True
    cap = _Cap()


@pytest.fixture
def kernel_for_local(monkeypatch):
    """Allow localhost connections for local-server tests by stubbing
    the private-IP check (return False so 127.0.0.1 is treated as
    public). Restored automatically."""
    monkeypatch.setattr(
        "cheetahclaws.kernel.tools.fetch_tool._is_private_ip",
        lambda ip: False,
    )
    return _OverrideDnsKernel()


def test_e2e_get(http_server, kernel_for_local):
    port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)
    result = fetch_handler({"url": f"{base}/anything"}, ctx)
    assert result["status"] == 200
    assert result["body"] == "hello from canned server"
    assert result["encoding"] == "utf-8"
    assert "x-test-marker" in result["headers"]
    assert result["truncated"] is False


def test_e2e_head(http_server, kernel_for_local):
    port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)
    result = fetch_handler(
        {"url": f"{base}/anything", "method": "HEAD"}, ctx,
    )
    assert result["status"] == 200
    assert result["body"] == ""           # HEAD has no body


def test_e2e_post_with_body(http_server, kernel_for_local):
    port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)
    result = fetch_handler({
        "url":    f"{base}/echo-body",
        "method": "POST",
        "body":   "ping",
    }, ctx)
    assert result["status"] == 200
    assert result["body"] == "ping"


def test_e2e_size_cap_truncates(http_server, kernel_for_local):
    """/huge returns 1MB; cap at 1024 → truncated."""
    port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)
    result = fetch_handler({
        "url":       f"{base}/huge",
        "max_bytes": 4096,
    }, ctx)
    assert result["status"] == 200
    assert result["truncated"] is True
    # Body decoded as base64 since ‘X’ * N is text but with bytes it
    # might still decode UTF-8; the truncation is the assertion.


def test_e2e_redirect_disabled_returns_30x(http_server, kernel_for_local):
    """follow_redirects=False → caller sees the 302 directly."""
    port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)
    result = fetch_handler({"url": f"{base}/redirect/1"}, ctx)
    assert result["status"] == 302
    assert result["redirects"] == []


def test_e2e_redirect_followed(http_server, kernel_for_local):
    """follow_redirects=True with budget → final 200."""
    port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)
    result = fetch_handler({
        "url":              f"{base}/redirect/2",
        "follow_redirects": True,
    }, ctx)
    assert result["status"] == 200
    assert len(result["redirects"]) == 2


def test_e2e_redirect_to_private_blocked(http_server, monkeypatch):
    """Redirect to 127.0.0.1 → ToolNetDenied (DNS rebinding defence
    via per-hop check). Setup: allow first hop's 127.0.0.1, then
    real check kicks in for the redirected URL whose IP is 127.0.0.1
    by direct URL.

    To force this scenario, we need the SECOND hop to fail. The
    server's /redirect-to-private points at http://127.0.0.1:1/.
    On the second hop, _check_hop is called with the new URL whose
    hostname is "127.0.0.1" — DNS resolves to itself, IP check
    fires.
    """
    port, base = http_server

    # Allow only the first hop's check; real check on the second.
    call_count = {"n": 0}
    real_is_private = None
    from cheetahclaws.kernel.tools.fetch_tool import _is_private_ip as _real
    real_is_private = _real

    def faked(ip):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return False  # let the first hop through
        return real_is_private(ip)
    monkeypatch.setattr(
        "cheetahclaws.kernel.tools.fetch_tool._is_private_ip", faked,
    )

    ctx = ToolContext(pid=1, kernel=_OverrideDnsKernel())
    with pytest.raises(ToolNetDenied):
        fetch_handler({
            "url":              f"{base}/redirect-to-private",
            "follow_redirects": True,
        }, ctx)


def test_e2e_redirect_strips_auth(http_server, kernel_for_local):
    """Authorization header sent on first hop, redirected to
    /echo-headers — second hop's response shouldn't include it."""
    port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)
    # /redirect/1 → / (default) → no echo. Use a custom path that
    # we know our handler maps to redirect.
    # Actually our server doesn't redirect to /echo-headers. We'd
    # need a special endpoint. Simplify: send to /echo-headers
    # WITHOUT redirect, verify auth was sent. Then verify redirect
    # path strips by inspecting handler logic — done in unit test
    # below.

    # Without redirect, header IS sent.
    result = fetch_handler({
        "url":     f"{base}/echo-headers",
        "headers": {"Authorization": "Bearer secret-shouldnt-leak"},
    }, ctx)
    # The server echoed all headers including Authorization.
    assert "secret-shouldnt-leak" in result["body"]


def test_strip_sensitive_headers_unit(http_server, monkeypatch):
    """Direct unit test on the strip logic: when the handler
    follows a redirect, sensitive headers are removed from the
    second-hop request. We verify by a custom http server that
    records the headers it received."""
    port = _free_port()
    received_headers: list = []

    class _Recorder(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            received_headers.append(dict(self.headers))
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", "/end")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                body = b"end"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = socketserver.TCPServer(("127.0.0.1", port), _Recorder)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    monkeypatch.setattr(
        "cheetahclaws.kernel.tools.fetch_tool._is_private_ip", lambda ip: False,
    )
    try:
        ctx = ToolContext(pid=1, kernel=_OverrideDnsKernel())
        fetch_handler({
            "url":              f"http://127.0.0.1:{port}/start",
            "headers":          {"Authorization": "Bearer SHOULD_BE_STRIPPED",
                                  "Cookie":        "session=SHOULD_BE_STRIPPED",
                                  "X-Custom":      "ok"},
            "follow_redirects": True,
        }, ctx)
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=2)

    assert len(received_headers) == 2
    # First hop has the auth header.
    assert "Bearer SHOULD_BE_STRIPPED" in received_headers[0].get(
        "Authorization", "",
    )
    # Second hop does NOT.
    assert "Authorization" not in received_headers[1] or \
        "SHOULD_BE_STRIPPED" not in received_headers[1].get("Authorization", "")
    # Cookie also stripped.
    assert "Cookie" not in received_headers[1] or \
        "SHOULD_BE_STRIPPED" not in received_headers[1].get("Cookie", "")
    # Custom header kept.
    assert received_headers[1].get("X-Custom") == "ok"


# ── Headers validation ──────────────────────────────────────────────


def test_headers_reject_reserved():
    ctx = ToolContext(pid=1, kernel=_AllowAllKernel())
    with pytest.raises(ToolInvalidArgs):
        fetch_handler({
            "url":     "https://example.com/",
            "headers": {"Host": "spoofed"},
        }, ctx)


def test_headers_reject_crlf():
    ctx = ToolContext(pid=1, kernel=_AllowAllKernel())
    with pytest.raises(ToolInvalidArgs):
        fetch_handler({
            "url":     "https://example.com/",
            "headers": {"X-Custom": "value\r\nHost: bad"},
        }, ctx)


def test_headers_reject_non_string():
    ctx = ToolContext(pid=1, kernel=_AllowAllKernel())
    with pytest.raises(ToolInvalidArgs):
        fetch_handler({
            "url":     "https://example.com/",
            "headers": {"X-Custom": 123},
        }, ctx)


# ── Body validation ─────────────────────────────────────────────────


def test_body_only_with_post():
    ctx = ToolContext(pid=1, kernel=_AllowAllKernel())
    with pytest.raises(ToolInvalidArgs):
        fetch_handler({
            "url":  "https://example.com/",
            "method": "GET",
            "body":   "data",
        }, ctx)


def test_body_b64_envelope_decoded(http_server, kernel_for_local):
    """body={'_b64': '...'} sends base64-decoded bytes."""
    port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)
    import base64 as _b64
    raw = b"\x00\x01\x02binary\xff"
    encoded = _b64.b64encode(raw).decode("ascii")
    result = fetch_handler({
        "url":    f"{base}/echo-body",
        "method": "POST",
        "body":   {"_b64": encoded},
    }, ctx)
    assert result["status"] == 200
    # Server echoed bytes; our decoder picks base64 fallback.
    decoded_back = _b64.b64decode(result["body"])
    assert decoded_back == raw


def test_body_b64_invalid():
    ctx = ToolContext(pid=1, kernel=_AllowAllKernel())
    with pytest.raises(ToolInvalidArgs):
        fetch_handler({
            "url":    "https://example.com/",
            "method": "POST",
            "body":   {"_b64": "not-valid-base64!!!"},
        }, ctx)


# ── Dispatch via supervisor (smoke) ─────────────────────────────────


def test_dispatch_via_supervisor(tmp_path, http_server, monkeypatch):
    """The full kernel + dispatch path with a granted Fetch."""
    port, base = http_server
    monkeypatch.setattr(
        "cheetahclaws.kernel.tools.fetch_tool._is_private_ip", lambda ip: False,
    )
    k = Kernel.open(tmp_path / "kernel.db")
    try:
        a = k.create_agent(name="x", template="t")
        k.cap.create(
            pid=a.pid, tool_grants=["Fetch"], net_grants=["*"],
        )
        r = ToolRegistry()
        register_fetch_tool(r)
        resp = dispatch_tool_call(
            msg={"tool": "Fetch", "tool_call_id": "T",
                  "args": {"url": f"{base}/anything"}},
            pid=a.pid, registry=r, kernel=k,
        )
        assert resp["ok"] is True
        assert resp["result"]["status"] == 200
    finally:
        k.close()


def test_dispatch_capability_denied(tmp_path):
    """Without Fetch in tool_grants → permission_denied."""
    k = Kernel.open(tmp_path / "kernel.db")
    try:
        a = k.create_agent(name="x", template="t")
        k.cap.create(pid=a.pid, tool_grants=["Read"], net_grants=["*"])
        r = ToolRegistry()
        register_fetch_tool(r)
        resp = dispatch_tool_call(
            msg={"tool": "Fetch", "tool_call_id": "T",
                  "args": {"url": "https://example.com/"}},
            pid=a.pid, registry=r, kernel=k,
        )
        assert resp["ok"] is False
        assert resp["error"] == "permission_denied"
    finally:
        k.close()


def test_dispatch_net_denied(tmp_path):
    """net_grants doesn't cover host → net_denied via handler."""
    k = Kernel.open(tmp_path / "kernel.db")
    try:
        a = k.create_agent(name="x", template="t")
        k.cap.create(
            pid=a.pid, tool_grants=["Fetch"],
            net_grants=["*.allowed.com"],
        )
        r = ToolRegistry()
        register_fetch_tool(r)
        resp = dispatch_tool_call(
            msg={"tool": "Fetch", "tool_call_id": "T",
                  "args": {"url": "https://forbidden.example/"}},
            pid=a.pid, registry=r, kernel=k,
        )
        assert resp["ok"] is False
        assert resp["error"] == "net_denied"
    finally:
        k.close()
