"""Tests for Fetch streaming (RFC 0029)."""
from __future__ import annotations

import http.server
import socket
import socketserver
import threading

import pytest

from cheetahclaws.kernel.tools.fetch_tool import fetch_handler
from cheetahclaws.kernel.tools.registry import ToolContext, ToolInvalidArgs


# ── Local HTTP server fixture ───────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StreamingHandler(http.server.BaseHTTPRequestHandler):
    """Endpoints geared at streaming tests."""

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/text16k":
            data = b"abcdefghijklmnop" * 1024     # 16384 bytes
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/text20k":
            data = b"L" * 20480                   # 20 KB
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/redirect-to-text":
            self.send_response(302)
            self.send_header("Location", "/text16k")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # default
        data = b"hello"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def http_server():
    port = _free_port()
    server = socketserver.TCPServer(("127.0.0.1", port), _StreamingHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield port, f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()
    t.join(timeout=2)


class _AllowAllKernel:
    class _Cap:
        def check_net(self, pid, host): return True
        def check_tool(self, pid, tool): return True
        def check_fs(self, pid, path, mode): return True
    cap = _Cap()


@pytest.fixture
def kernel_for_local(monkeypatch):
    monkeypatch.setattr(
        "cheetahclaws.kernel.tools.fetch_tool._is_private_ip",
        lambda ip: False,
    )
    return _AllowAllKernel()


# ── Validation ───────────────────────────────────────────────────────


def test_stream_arg_must_be_bool(http_server, kernel_for_local):
    _port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)
    with pytest.raises(ToolInvalidArgs):
        fetch_handler({"url": f"{base}/", "stream": "yes"}, ctx)


# ── Default stream=False emits zero chunks ──────────────────────────


def test_stream_default_no_chunks(http_server, kernel_for_local):
    _port, base = http_server
    received: list = []
    ctx = ToolContext(
        pid=1, kernel=kernel_for_local,
        on_chunk=lambda x: received.append(x),
    )
    result = fetch_handler({"url": f"{base}/text16k"}, ctx)
    assert result["status"] == 200
    assert len(result["body"]) == 16384
    assert received == []


def test_stream_true_no_callback_no_chunks(http_server, kernel_for_local):
    """stream=True but on_chunk None → buffered (no chunks
    possible because there's no sink)."""
    _port, base = http_server
    ctx = ToolContext(pid=1, kernel=kernel_for_local)   # no on_chunk
    result = fetch_handler({"url": f"{base}/text16k",
                              "stream": True}, ctx)
    assert result["status"] == 200
    assert len(result["body"]) == 16384


# ── Streaming path emits chunks ─────────────────────────────────────


def test_stream_emits_8k_blocks(http_server, kernel_for_local):
    _port, base = http_server
    received: list = []
    ctx = ToolContext(
        pid=1, kernel=kernel_for_local,
        on_chunk=lambda x: received.append(x),
    )
    result = fetch_handler({
        "url": f"{base}/text16k", "stream": True,
    }, ctx)
    assert result["status"] == 200
    assert len(result["body"]) == 16384
    # 16K body, 8K read chunks → 2 chunks expected.
    assert len(received) >= 2
    for c in received:
        assert c["op"] == "chunk"
        assert c["kind"] == "body"
        assert c["metadata"]["tool"] == "Fetch"
        assert c["metadata"]["url"] == f"{base}/text16k"
        assert c["metadata"]["status"] == 200
        assert isinstance(c["metadata"]["bytes_so_far"], int)


def test_stream_chunks_concatenate_to_body(http_server, kernel_for_local):
    _port, base = http_server
    received: list = []
    ctx = ToolContext(
        pid=1, kernel=kernel_for_local,
        on_chunk=lambda x: received.append(x),
    )
    result = fetch_handler({
        "url": f"{base}/text16k", "stream": True,
    }, ctx)
    streamed = "".join(c["content"] for c in received)
    assert streamed == result["body"]


def test_stream_bytes_so_far_monotonic(http_server, kernel_for_local):
    _port, base = http_server
    received: list = []
    ctx = ToolContext(
        pid=1, kernel=kernel_for_local,
        on_chunk=lambda x: received.append(x),
    )
    fetch_handler({"url": f"{base}/text16k", "stream": True}, ctx)
    bsf = [c["metadata"]["bytes_so_far"] for c in received]
    assert bsf == sorted(bsf)
    assert bsf[-1] <= 16384


# ── Truncation stops chunks ─────────────────────────────────────────


def test_stream_truncation_stops_chunks(http_server, kernel_for_local):
    """max_bytes < body size → chunks emitted only up to the cap,
    truncated=True."""
    _port, base = http_server
    received: list = []
    ctx = ToolContext(
        pid=1, kernel=kernel_for_local,
        on_chunk=lambda x: received.append(x),
    )
    result = fetch_handler({
        "url": f"{base}/text20k",
        "stream": True,
        "max_bytes": 8192,
    }, ctx)
    assert result["truncated"] is True
    assert len(result["body"]) == 8192
    streamed = "".join(c["content"] for c in received)
    # Streamed bytes must not exceed cap.
    assert len(streamed.encode("utf-8")) <= 8192
    assert all(
        c["metadata"]["bytes_so_far"] <= 8192 for c in received
    )


# ── Redirect intermediate body does NOT stream ──────────────────────


def test_stream_only_terminal_hop(http_server, kernel_for_local):
    """A redirect's empty intermediate body must not emit chunks;
    only the final hop streams."""
    _port, base = http_server
    received: list = []
    ctx = ToolContext(
        pid=1, kernel=kernel_for_local,
        on_chunk=lambda x: received.append(x),
    )
    result = fetch_handler({
        "url": f"{base}/redirect-to-text",
        "stream": True,
        "follow_redirects": True,
        "max_redirects": 3,
    }, ctx)
    assert result["status"] == 200
    assert len(result["body"]) == 16384
    # All chunks should have the FINAL url, not the redirect source.
    assert received, "expected chunks from the terminal hop"
    for c in received:
        assert c["metadata"]["url"] == f"{base}/text16k"
    assert result["redirects"][0]["from"] == f"{base}/redirect-to-text"


# ── Bad callback can't crash the fetch ──────────────────────────────


def test_stream_bad_callback_swallowed(http_server, kernel_for_local):
    _port, base = http_server

    def bad_cb(payload):
        raise RuntimeError("boom")

    ctx = ToolContext(
        pid=1, kernel=kernel_for_local, on_chunk=bad_cb,
    )
    result = fetch_handler({
        "url": f"{base}/text16k", "stream": True,
    }, ctx)
    assert result["status"] == 200
    assert len(result["body"]) == 16384
