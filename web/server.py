"""Lightweight web terminal server for CheetahClaws.

Spawns a CheetahClaws REPL in a PTY and bridges it to the browser.
Supports two transport modes on the same port:

  1. WebSocket (direct TCP connections)
  2. SSE + POST fallback (works through VS Code port forwarding
     and other HTTP-level proxies that break WebSocket upgrades)

Pure-stdlib — no Flask, no aiohttp, no external deps.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import pty
import secrets
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ── Config ───────────────────────────────────────────────────────────────

DEFAULT_PORT = 8080
_WEB_DIR = Path(__file__).resolve().parent

_server_password: Optional[str] = None
_server_no_auth = False
_server_cmd: list[str] = []

_MIME = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".ico": "image/x-icon",
    ".png": "image/png",
}


def _generate_password() -> str:
    return secrets.token_urlsafe(6)[:6]


class _BufferedSocket:
    """Thin wrapper that prepends leftover bytes from HTTP header parsing
    before delegating to the real socket.  Avoids monkey-patching sock.recv."""

    __slots__ = ("_sock", "_buf")

    def __init__(self, sock: socket.socket, extra: bytes = b""):
        self._sock = sock
        self._buf = extra

    def recv(self, n: int, _flags: int = 0) -> bytes:
        if self._buf:
            chunk = self._buf[:n]
            self._buf = self._buf[n:]
            return chunk
        return self._sock.recv(n)

    def sendall(self, data: bytes) -> None:
        self._sock.sendall(data)

    def settimeout(self, t) -> None:
        self._sock.settimeout(t)

    def close(self) -> None:
        self._sock.close()


# ── PTY session registry (for SSE mode) ─────────────────────────────────

_sessions: dict[str, "_PtySession"] = {}
_sessions_lock = threading.Lock()


_SESSION_TIMEOUT = 30  # seconds before an unattached SSE session is reaped


class _PtySession:
    """A PTY session shared between SSE stream and POST input."""

    def __init__(self):
        self.master_fd, slave_fd = pty.openpty()
        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["COLUMNS"] = "120"
        env["LINES"] = "30"
        env["CHEETAHCLAWS_WEB_TERMINAL"] = "1"
        self.proc = subprocess.Popen(
            _server_cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env=env, preexec_fn=os.setsid,
        )
        os.close(slave_fd)
        self.lock = threading.Lock()
        self.closed = False
        self.created_at = time.monotonic()
        self.attached = False  # True once an SSE stream connects

    def write(self, data: bytes) -> None:
        if not self.closed:
            with self.lock:
                try:
                    os.write(self.master_fd, data)
                except OSError:
                    pass

    def resize(self, rows: int, cols: int) -> None:
        if self.closed:
            return
        try:
            import fcntl, termios
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
            os.killpg(os.getpgid(self.proc.pid), signal.SIGWINCH)
        except (OSError, ProcessLookupError):
            pass

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            os.close(self.master_fd)
        except OSError:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ── HTML page ────────────────────────────────────────────────────────────

def _build_html(no_auth: bool = False) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>CheetahClaws Web Terminal</title>
<link rel="stylesheet" href="/xterm.min.css">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0a0a; overflow:hidden; height:100vh; display:flex; flex-direction:column; }}
  #topbar {{
    background:#111; border-bottom:1px solid #222; padding:8px 16px;
    display:flex; align-items:center; justify-content:space-between;
    font-family:-apple-system,sans-serif; color:#888; font-size:13px;
  }}
  #topbar .logo {{ color:#22d3ee; font-weight:700; font-size:15px; }}
  #topbar .status {{ display:flex; align-items:center; gap:8px; }}
  #topbar .dot {{ width:8px; height:8px; border-radius:50%; background:#4ade80; }}
  #topbar .dot.disconnected {{ background:#f87171; }}
  #terminal {{ flex:1; }}
  #login {{
    position:fixed; top:0; left:0; width:100%; height:100%;
    background:rgba(0,0,0,0.95); display:flex; align-items:center;
    justify-content:center; z-index:100; font-family:-apple-system,sans-serif;
  }}
  #login.hidden {{ display:none; }}
  #login form {{
    background:#111; border:1px solid #333; border-radius:12px;
    padding:2rem; width:320px; text-align:center;
  }}
  #login h2 {{ color:#22d3ee; margin-bottom:1rem; font-size:1.2rem; }}
  #login input {{
    width:100%; padding:10px; background:#1a1a1a; border:1px solid #333;
    border-radius:8px; color:#fff; font-size:1rem; margin-bottom:1rem;
    text-align:center; letter-spacing:2px;
  }}
  #login input:focus {{ outline:none; border-color:#22d3ee; }}
  #login button {{
    background:#22d3ee; color:#000; border:none; padding:10px 24px;
    border-radius:8px; font-weight:700; font-size:0.9rem; cursor:pointer; width:100%;
  }}
  #login .error {{ color:#f87171; font-size:0.85rem; margin-top:0.5rem; }}
</style>
</head>
<body>
<div id="login" class="{'hidden' if no_auth else ''}">
  <form onsubmit="doLogin(event)">
    <h2>CheetahClaws</h2>
    <input type="password" id="pwd" placeholder="Enter password" autofocus>
    <button type="submit">Connect</button>
    <div class="error" id="login-err"></div>
  </form>
</div>
<div id="topbar">
  <span class="logo">CheetahClaws Web Terminal</span>
  <span class="status">
    <span class="dot" id="status-dot"></span>
    <span id="status-text">connecting...</span>
  </span>
</div>
<div id="terminal"></div>
<script src="/xterm.min.js"></script>
<script src="/addon-fit.min.js"></script>
<script src="/addon-web-links.min.js"></script>
<script>
const term = new window.Terminal({{
  cursorBlink: true, fontSize: 14,
  fontFamily: "'JetBrains Mono','Fira Code','Cascadia Code',monospace",
  theme: {{ background:'#0a0a0a', foreground:'#e4e4e7', cursor:'#22d3ee',
            selectionBackground:'rgba(34,211,238,0.3)' }},
}});
const fitAddon = new window.FitAddon.FitAddon();
term.loadAddon(fitAddon);
term.loadAddon(new window.WebLinksAddon.WebLinksAddon());
term.open(document.getElementById('terminal'));
fitAddon.fit();

let authToken = '';
let sessionId = '';
let mode = ''; // 'ws' or 'sse'
let _dataSub = null, _resizeSub = null;
function bindInput(dataFn, resizeFn) {{
  if (_dataSub) _dataSub.dispose();
  if (_resizeSub) _resizeSub.dispose();
  _dataSub = term.onData(dataFn);
  _resizeSub = term.onResize(resizeFn);
}}

function setStatus(connected, label) {{
  document.getElementById('status-dot').className = 'dot' + (connected ? '' : ' disconnected');
  document.getElementById('status-text').textContent = label || (connected ? 'connected' : 'disconnected');
}}

// ── SSE + POST fallback (works through any HTTP proxy) ──────────────

function connectSSE() {{
  mode = 'sse';
  term.clear();
  term.reset();
  setStatus(false, 'connecting (http)...');

  // Create PTY session (cookie carries auth)
  fetch('/api/session', {{
    method: 'POST',
    credentials: 'same-origin',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{cols: term.cols, rows: term.rows}})
  }})
  .then(r => r.json())
  .then(data => {{
    sessionId = data.session_id;

    // Open SSE stream for terminal output (cookie carries auth)
    const evtSource = new EventSource('/api/stream?sid=' + sessionId);
    evtSource.onopen = () => setStatus(true, 'connected (http)');
    evtSource.onmessage = (e) => {{
      // Data is base64-encoded binary
      const bytes = Uint8Array.from(atob(e.data), c => c.charCodeAt(0));
      term.write(bytes);
    }};
    evtSource.onerror = () => {{
      setStatus(false);
      evtSource.close();
      term.write('\\r\\n\\x1b[33m[disconnected — refresh to reconnect]\\x1b[0m\\r\\n');
    }};

    // Bind input (replaces any previous WS handlers)
    bindInput(
      d => fetch('/api/input', {{
        method: 'POST',
        credentials: 'same-origin',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{sid: sessionId, data: d}})
      }}).catch(() => {{}}),
      ({{cols, rows}}) => fetch('/api/resize', {{
        method: 'POST',
        credentials: 'same-origin',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{sid: sessionId, cols, rows}})
      }}).catch(() => {{}})
    );
  }})
  .catch(err => {{
    setStatus(false, 'connection failed');
    term.write('\\r\\n\\x1b[31m[failed to connect: ' + err.message + ']\\x1b[0m\\r\\n');
  }});
}}

// ── WebSocket (direct connections) ──────────────────────────────────

function connectWS() {{
  mode = 'ws';
  setStatus(false, 'connecting (ws)...');

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = proto + '//' + location.host + '/ws';
  const ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';

  let wsOpened = false;
  let wsAuthed = false;
  const wsTimeout = setTimeout(() => {{
    if (!wsOpened) ws.close();
  }}, 3000);

  ws.onopen = () => {{
    wsOpened = true;
    clearTimeout(wsTimeout);
    // First frame: authenticate (cookie may already suffice, but send
    // explicit auth in case cookie is not available — e.g. cross-origin).
    ws.send(JSON.stringify({{type:'auth', token:authToken}}));
    setStatus(true, 'connected (ws)');
    ws.send(JSON.stringify({{type:'resize', cols:term.cols, rows:term.rows}}));

    bindInput(
      d => {{ if (ws.readyState === 1) ws.send(d); }},
      ({{cols, rows}}) => {{ if (ws.readyState === 1) ws.send(JSON.stringify({{type:'resize', cols, rows}})); }}
    );
  }};
  ws.onmessage = (e) => {{
    if (e.data instanceof ArrayBuffer) term.write(new Uint8Array(e.data));
    else term.write(e.data);
  }};
  ws.onclose = () => {{
    clearTimeout(wsTimeout);
    if (!wsOpened) {{
      connectSSE();
    }} else {{
      setStatus(false);
      term.write('\\r\\n\\x1b[33m[disconnected — refresh to reconnect]\\x1b[0m\\r\\n');
    }}
  }};
  ws.onerror = () => {{ }};
}}

// ── Connect (try WebSocket first, fall back to SSE) ─────────────────

function connect() {{
  connectWS();
}}

window.addEventListener('resize', () => fitAddon.fit());

function doLogin(e) {{
  e.preventDefault();
  authToken = document.getElementById('pwd').value;
  const errEl = document.getElementById('login-err');
  errEl.textContent = '';
  // Authenticate via POST (sets HttpOnly cookie for subsequent requests)
  fetch('/api/auth', {{
    method: 'POST',
    credentials: 'same-origin',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{token: authToken}})
  }}).then(r => {{
    if (!r.ok) {{ errEl.textContent = 'Wrong password'; return; }}
    document.getElementById('login').classList.add('hidden');
    connect();
  }}).catch(() => {{ errEl.textContent = 'Connection error'; }});
}}

if (document.getElementById('login').classList.contains('hidden')) connect();
</script>
</body>
</html>"""


# ── Raw HTTP helpers ─────────────────────────────────────────────────────

def _recv_until(sock: socket.socket, sentinel: bytes, max_bytes: int = 65536) -> bytes:
    buf = b""
    while sentinel not in buf and len(buf) < max_bytes:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf


def _cors_origin(request_origin: str = "") -> str:
    """Return a safe CORS origin: echo back the request Origin if present,
    otherwise omit the header entirely.  Never emit '*' so that
    credentialed (cookie-based) requests are allowed by browsers."""
    if request_origin:
        return request_origin
    return ""


def _send_http(sock: socket.socket, status: str, content_type: str,
               body: bytes, extra_headers: str = "",
               request_origin: str = "") -> None:
    origin = _cors_origin(request_origin)
    cors = ""
    if origin:
        cors = (
            f"Access-Control-Allow-Origin: {origin}\r\n"
            f"Access-Control-Allow-Credentials: true\r\n"
            f"Access-Control-Allow-Methods: GET, POST, PATCH, OPTIONS\r\n"
            f"Access-Control-Allow-Headers: Content-Type\r\n"
            f"Vary: Origin\r\n"
        )
    header = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"{cors}"
        f"Connection: close\r\n"
        f"{extra_headers}"
        f"\r\n"
    )
    sock.sendall(header.encode() + body)


def _send_json(sock: socket.socket, obj: dict,
               request_origin: str = "") -> None:
    body = json.dumps(obj).encode()
    _send_http(sock, "200 OK", "application/json", body,
               request_origin=request_origin)


def _check_auth(query: str = "", body_token: str = "",
                cookie_str: str = "") -> bool:
    """Check auth token from JSON body, cookie, or query (timing-safe).

    Preference order: body_token > cookie > query string.
    Cookie and body are preferred because they don't leak into access logs.
    """
    if _server_no_auth:
        return True
    token = body_token
    if not token and cookie_str:
        for part in cookie_str.split(";"):
            part = part.strip()
            if part.startswith("cctoken="):
                from urllib.parse import unquote
                token = unquote(part[8:])
                break
    if not token:
        for param in query.split("&"):
            if param.startswith("token="):
                from urllib.parse import unquote
                token = unquote(param[6:])
    if not token or not _server_password:
        return False
    return hmac.compare_digest(token, _server_password)


# ── WebSocket frame helpers (RFC 6455) ───────────────────────────────────

def _ws_send(sock, data: bytes, opcode: int = 0x02,
             lock: Optional[threading.Lock] = None) -> None:
    length = len(data)
    if length < 126:
        hdr = bytes([0x80 | opcode, length])
    elif length < 65536:
        hdr = bytes([0x80 | opcode, 126]) + struct.pack("!H", length)
    else:
        hdr = bytes([0x80 | opcode, 127]) + struct.pack("!Q", length)
    payload = hdr + data
    if lock:
        with lock:
            sock.sendall(payload)
    else:
        sock.sendall(payload)


def _ws_recv(sock,
             lock: Optional[threading.Lock] = None) -> Optional[str | bytes]:
    def recv_exact(n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except (OSError, ConnectionResetError, TimeoutError):
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    # Loop instead of recursion to avoid stack overflow on
    # sustained ping/pong sequences.
    while True:
        head = recv_exact(2)
        if not head:
            return None

        opcode = head[0] & 0x0F
        masked = bool(head[1] & 0x80)
        length = head[1] & 0x7F

        if length == 126:
            ext = recv_exact(2)
            if not ext:
                return None
            length = struct.unpack("!H", ext)[0]
        elif length == 127:
            ext = recv_exact(8)
            if not ext:
                return None
            length = struct.unpack("!Q", ext)[0]

        if masked:
            mask = recv_exact(4)
            if not mask:
                return None
            raw = bytearray(recv_exact(length) or b"")
            for i in range(len(raw)):
                raw[i] ^= mask[i % 4]
            raw = bytes(raw)
        else:
            raw = recv_exact(length) or b""

        if opcode == 0x8:  # close
            try:
                _ws_send(sock, struct.pack("!H", 1000), opcode=0x08, lock=lock)
            except OSError:
                pass
            return None
        if opcode == 0x9:  # ping → pong
            try:
                _ws_send(sock, raw, opcode=0x0A, lock=lock)
            except OSError:
                pass
            continue  # read next frame
        if opcode == 0xA:  # pong — ignore, read next frame
            continue

        if opcode == 0x1:
            return raw.decode(errors="replace")
        return raw


# ── WebSocket ↔ PTY bridge ───────────────────────────────────────────────

def _handle_websocket(sock: socket.socket, extra: bytes,
                      pre_authed: bool = False) -> None:
    bsock = _BufferedSocket(sock, extra)
    send_lock = threading.Lock()

    # ── First-frame authentication ──────────────────────────────────
    # If not pre-authenticated via cookie/query, require the first WS
    # message to be JSON {"type":"auth","token":"..."}.
    if not pre_authed:
        msg = _ws_recv(bsock, lock=send_lock)
        if msg is None:
            return
        authed = False
        if isinstance(msg, str):
            try:
                obj = json.loads(msg)
                if obj.get("type") == "auth":
                    authed = _check_auth(body_token=obj.get("token", ""))
            except (json.JSONDecodeError, KeyError):
                pass
        if not authed:
            try:
                _ws_send(bsock, b'{"error":"auth required"}', opcode=0x01,
                         lock=send_lock)
                _ws_send(bsock, struct.pack("!H", 1008), opcode=0x08,
                         lock=send_lock)
            except OSError:
                pass
            return

    master_fd, slave_fd = pty.openpty()
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["COLUMNS"] = "120"
    env["LINES"] = "30"
    env["CHEETAHCLAWS_WEB_TERMINAL"] = "1"
    proc = subprocess.Popen(
        _server_cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        env=env, preexec_fn=os.setsid,
    )
    os.close(slave_fd)

    def pty_to_ws():
        try:
            while True:
                r, _, _ = select.select([master_fd], [], [], 1.0)
                if not r:
                    if proc.poll() is not None:
                        break
                    continue
                data = os.read(master_fd, 16384)
                if not data:
                    break
                _ws_send(bsock, data, lock=send_lock)
        except (OSError, BrokenPipeError):
            pass

    reader_t = threading.Thread(target=pty_to_ws, daemon=True)
    reader_t.start()

    try:
        while True:
            msg = _ws_recv(bsock, lock=send_lock)
            if msg is None:
                break
            if isinstance(msg, str):
                try:
                    obj = json.loads(msg)
                    if obj.get("type") == "resize":
                        import fcntl, termios
                        rows, cols = int(obj["rows"]), int(obj["cols"])
                        winsize = struct.pack("HHHH", rows, cols, 0, 0)
                        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGWINCH)
                        except (OSError, ProcessLookupError):
                            pass
                        continue
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
                os.write(master_fd, msg.encode())
            elif isinstance(msg, bytes):
                os.write(master_fd, msg)
    except (OSError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        reader_t.join(timeout=2)


# ── SSE stream handler ──────────────────────────────────────────────────

def _handle_sse_stream(sock: socket.socket, sid: str,
                       request_origin: str = "") -> None:
    """Stream PTY output as Server-Sent Events."""
    with _sessions_lock:
        session = _sessions.get(sid)
    if not session or session.closed:
        _send_http(sock, "404 Not Found", "text/plain", b"session not found",
                   request_origin=request_origin)
        return

    # Send SSE headers (keep connection open)
    cors_origin = _cors_origin(request_origin)
    cors = ""
    if cors_origin:
        cors = (
            f"Access-Control-Allow-Origin: {cors_origin}\r\n"
            f"Access-Control-Allow-Credentials: true\r\n"
            f"Vary: Origin\r\n"
        )
    header = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        f"{cors}"
        "\r\n"
    )
    sock.sendall(header.encode())
    session.attached = True

    # Stream PTY output as SSE events
    try:
        while not session.closed:
            r, _, _ = select.select([session.master_fd], [], [], 1.0)
            if not r:
                # Send SSE comment as keepalive
                sock.sendall(b": keepalive\n\n")
                if session.proc.poll() is not None:
                    break
                continue
            data = os.read(session.master_fd, 16384)
            if not data:
                break
            # Encode as base64 for safe SSE transport
            b64 = base64.b64encode(data).decode()
            sse_msg = f"data: {b64}\n\n"
            sock.sendall(sse_msg.encode())
    except (OSError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        # Clean up session when stream ends
        session.close()
        with _sessions_lock:
            _sessions.pop(sid, None)


# ── Chat WebSocket handler (structured events) ─────────────────────────

def _handle_chat_websocket(sock: socket.socket, extra: bytes) -> None:
    """Handle /api/events WebSocket: stream ChatEvents to browser.

    Auth is already verified at the HTTP layer before the WS upgrade.
    First frame from the client must be: {"session_id": "..."}
    """
    bsock = _BufferedSocket(sock, extra)
    send_lock = threading.Lock()

    # First frame: {session_id: "..."} to identify the chat session
    msg = _ws_recv(bsock, lock=send_lock)
    if msg is None:
        return
    session_id = ""
    if isinstance(msg, str):
        try:
            obj = json.loads(msg)
            session_id = obj.get("session_id", "")
        except (json.JSONDecodeError, KeyError):
            pass

    from web.api import get_chat_session
    chat_session = get_chat_session(session_id)
    if not chat_session:
        try:
            _ws_send(bsock, json.dumps({"error": "session not found"}).encode(),
                     opcode=0x01, lock=send_lock)
        except OSError:
            pass
        return

    # Subscribe to the session's event queue
    event_queue = chat_session.subscribe()

    # Reader thread: handle incoming WS messages (approve, prompt, etc.)
    reader_alive = threading.Event()
    reader_alive.set()

    def _ws_reader():
        try:
            while reader_alive.is_set():
                msg = _ws_recv(bsock, lock=send_lock)
                if msg is None:
                    reader_alive.clear()
                    break
                if isinstance(msg, str):
                    try:
                        obj = json.loads(msg)
                        msg_type = obj.get("type", "")
                        if msg_type == "approve":
                            chat_session.approve_permission(
                                obj.get("granted", False))
                        elif msg_type == "prompt":
                            chat_session.submit_prompt(
                                obj.get("prompt", ""))
                    except (json.JSONDecodeError, KeyError):
                        pass
        except (OSError, ConnectionResetError):
            reader_alive.clear()

    reader_t = threading.Thread(target=_ws_reader, daemon=True)
    reader_t.start()

    # Main loop: drain event queue → send to WS client
    try:
        while reader_alive.is_set():
            try:
                event = event_queue.get(timeout=30)
                payload = event.to_json().encode()
                _ws_send(bsock, payload, opcode=0x01, lock=send_lock)
            except Exception:
                # queue.Empty on timeout → send WS ping as keepalive
                try:
                    _ws_send(bsock, b"", opcode=0x09, lock=send_lock)
                except OSError:
                    break
    except (OSError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        reader_alive.clear()
        chat_session.unsubscribe(event_queue)
        reader_t.join(timeout=2)


# ── Connection handler ───────────────────────────────────────────────────

def _handle_connection(sock: socket.socket, addr: tuple) -> None:
    try:
        sock.settimeout(30)
        raw = _recv_until(sock, b"\r\n\r\n")
        if not raw:
            sock.close()
            return

        header_end = raw.find(b"\r\n\r\n")
        header_bytes = raw[:header_end]
        extra = raw[header_end + 4:]

        header_str = header_bytes.decode(errors="replace")
        lines = header_str.split("\r\n")
        if not lines:
            sock.close()
            return

        request_line = lines[0]
        parts = request_line.split(" ")
        if len(parts) < 2:
            sock.close()
            return

        method, raw_path = parts[0], parts[1]
        path = raw_path.split("?")[0]
        query = raw_path.split("?", 1)[1] if "?" in raw_path else ""

        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        origin = headers.get("origin", "")
        cookie = headers.get("cookie", "")

        # Parse JSON body for POST requests
        body_str = ""
        body_json = {}
        if method in ("POST", "PATCH"):
            content_len = int(headers.get("content-length", 0))
            if content_len > 0:
                body_bytes = extra
                while len(body_bytes) < content_len:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    body_bytes += chunk
                body_str = body_bytes[:content_len].decode(errors="replace")
                try:
                    body_json = json.loads(body_str)
                except json.JSONDecodeError:
                    pass

        # ── CORS preflight ───────────────────────────────────────────
        if method == "OPTIONS":
            cors_origin = _cors_origin(origin)
            cors_hdrs = ""
            if cors_origin:
                cors_hdrs = (
                    f"Access-Control-Allow-Origin: {cors_origin}\r\n"
                    f"Access-Control-Allow-Credentials: true\r\n"
                    f"Access-Control-Allow-Methods: GET, POST, PATCH, OPTIONS\r\n"
                    f"Access-Control-Allow-Headers: Content-Type\r\n"
                    f"Vary: Origin\r\n"
                )
            _send_http(sock, "204 No Content", "text/plain", b"",
                       cors_hdrs)
            sock.close()
            return

        # ── POST /api/auth — login, set cookie ──────────────────────
        if path == "/api/auth" and method == "POST":
            if _check_auth(body_token=body_json.get("token", "")):
                from urllib.parse import quote
                set_cookie = (
                    f"Set-Cookie: cctoken={quote(_server_password or '')}; "
                    f"Path=/; HttpOnly; SameSite=Strict; Max-Age=86400\r\n"
                )
                body = b'{"ok":true}'
                _send_http(sock, "200 OK", "application/json", body,
                           extra_headers=set_cookie,
                           request_origin=origin)
            else:
                _send_http(sock, "401 Unauthorized", "application/json",
                           b'{"error":"bad password"}',
                           request_origin=origin)
            sock.close()
            return

        # ── WebSocket upgrade ────────────────────────────────────────
        if path == "/ws" and "upgrade" in headers.get("connection", "").lower():
            # Accept upgrade first; authenticate via first message.
            # Cookie auth is also accepted for convenience.
            pre_authed = _check_auth(query, cookie_str=cookie)

            ws_key = headers.get("sec-websocket-key", "")
            accept = base64.b64encode(
                hashlib.sha1(
                    (ws_key + "258EAFA5-E914-47DA-95CA-5AB9DC11B5AB").encode()
                ).digest()
            ).decode()
            handshake = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            )
            sock.sendall(handshake.encode())
            sock.settimeout(None)
            _handle_websocket(sock, extra, pre_authed=pre_authed)
            try:
                sock.close()
            except OSError:
                pass
            return

        # ── SSE: Create session ──────────────────────────────────────
        if path == "/api/session" and method == "POST":
            if not _check_auth(query, body_json.get("token", ""),
                               cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return

            sid = secrets.token_urlsafe(16)
            session = _PtySession()
            cols = body_json.get("cols", 120)
            rows = body_json.get("rows", 30)
            session.resize(rows, cols)
            with _sessions_lock:
                _sessions[sid] = session
            _send_json(sock, {"session_id": sid}, request_origin=origin)
            sock.close()
            return

        # ── SSE: Stream output ───────────────────────────────────────
        if path == "/api/stream" and method == "GET":
            sid = ""
            for param in query.split("&"):
                if param.startswith("sid="):
                    sid = param[4:]
            if not _check_auth(query, cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return

            sock.settimeout(None)  # Long-lived connection
            _handle_sse_stream(sock, sid, request_origin=origin)
            try:
                sock.close()
            except OSError:
                pass
            return

        # ── SSE: Terminal input ──────────────────────────────────────
        if path == "/api/input" and method == "POST":
            sid = body_json.get("sid", "")
            if not _check_auth(query, body_json.get("token", ""),
                               cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return
            with _sessions_lock:
                session = _sessions.get(sid)
            if session:
                session.write(body_json.get("data", "").encode())
                _send_json(sock, {"ok": True}, request_origin=origin)
            else:
                _send_http(sock, "404 Not Found", "text/plain",
                           b"session not found", request_origin=origin)
            sock.close()
            return

        # ── SSE: Resize ──────────────────────────────────────────────
        if path == "/api/resize" and method == "POST":
            sid = body_json.get("sid", "")
            if not _check_auth(query, body_json.get("token", ""),
                               cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return
            with _sessions_lock:
                session = _sessions.get(sid)
            if session:
                session.resize(body_json.get("rows", 30),
                               body_json.get("cols", 120))
                _send_json(sock, {"ok": True}, request_origin=origin)
            else:
                _send_http(sock, "404 Not Found", "text/plain",
                           b"session not found", request_origin=origin)
            sock.close()
            return

        # ── Chat UI page ────────────────────────────────────────────
        if path == "/chat" and method == "GET":
            # Serve chat.html always — it has its own login handling.
            # API endpoints enforce auth; the page itself is just static HTML.
            chat_path = _WEB_DIR / "chat.html"
            if chat_path.exists():
                body = chat_path.read_bytes()
                _send_http(sock, "200 OK", "text/html; charset=utf-8", body,
                           request_origin=origin)
            else:
                _send_http(sock, "404 Not Found", "text/plain",
                           b"chat.html not found", request_origin=origin)
            sock.close()
            return

        # ── POST /api/prompt — submit prompt to chat session ────────
        if path == "/api/prompt" and method == "POST":
            if not _check_auth(query, body_json.get("token", ""),
                               cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return
            from web.api import create_chat_session, get_chat_session
            sid = body_json.get("session_id", "")
            chat_sess = get_chat_session(sid) if sid else None
            if not chat_sess:
                from cc_config import load_config
                chat_sess = create_chat_session(load_config())
            prompt = body_json.get("prompt", "")
            if prompt and prompt.startswith("/"):
                # Check if client wants SSE streaming
                accept_hdr = headers.get("accept", "")
                wants_stream = "text/event-stream" in accept_hdr

                if wants_stream:
                    # SSE: keep connection open, stream events as they happen
                    cors_origin = _cors_origin(origin)
                    cors = ""
                    if cors_origin:
                        cors = (
                            f"Access-Control-Allow-Origin: {cors_origin}\r\n"
                            f"Access-Control-Allow-Credentials: true\r\n"
                            f"Vary: Origin\r\n"
                        )
                    sse_header = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/event-stream\r\n"
                        "Cache-Control: no-cache\r\n"
                        "Connection: keep-alive\r\n"
                        f"{cors}"
                        "\r\n"
                    )
                    sock.sendall(sse_header.encode())
                    sock.settimeout(None)
                    # Stream session_id first
                    sock.sendall(f"data: {json.dumps({'type':'session','data':{'session_id':chat_sess.session_id}})}\n\n".encode())
                    def _sse_callback(evt_dict):
                        try:
                            sock.sendall(f"data: {json.dumps(evt_dict)}\n\n".encode())
                        except (OSError, BrokenPipeError):
                            pass
                    try:
                        chat_sess.handle_slash_stream(prompt, _sse_callback)
                    except (OSError, BrokenPipeError):
                        pass
                    finally:
                        # Send done marker
                        try:
                            sock.sendall(b"data: {\"type\":\"done\"}\n\n")
                        except (OSError, BrokenPipeError):
                            pass
                        try:
                            sock.close()
                        except OSError:
                            pass
                    return

                # Regular POST: return events inline
                events = chat_sess.handle_slash_sync(prompt)
                _send_json(sock, {
                    "session_id": chat_sess.session_id,
                    "events": events,
                }, request_origin=origin)
                sock.close()
                return
            accepted = True
            if prompt:
                accepted = chat_sess.submit_prompt(prompt)
            if not accepted:
                _send_http(sock, "409 Conflict", "application/json",
                           json.dumps({"error": "agent is busy",
                                       "session_id": chat_sess.session_id}).encode(),
                           request_origin=origin)
                sock.close()
                return
            _send_json(sock, {"session_id": chat_sess.session_id},
                       request_origin=origin)
            sock.close()
            return

        # ── WS /api/events — structured event stream ────────────────
        if path == "/api/events" and "upgrade" in headers.get("connection", "").lower():
            # Auth check BEFORE upgrade — reject with 401, no WS handshake
            if not _check_auth(query, cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return
            ws_key = headers.get("sec-websocket-key", "")
            accept = base64.b64encode(
                hashlib.sha1(
                    (ws_key + "258EAFA5-E914-47DA-95CA-5AB9DC11B5AB").encode()
                ).digest()
            ).decode()
            handshake = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            )
            sock.sendall(handshake.encode())
            sock.settimeout(None)
            _handle_chat_websocket(sock, extra)
            try:
                sock.close()
            except OSError:
                pass
            return

        # ── POST /api/approve — respond to permission request ───────
        if path == "/api/approve" and method == "POST":
            if not _check_auth(query, body_json.get("token", ""),
                               cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return
            from web.api import get_chat_session
            sid = body_json.get("session_id", "")
            granted = body_json.get("granted", False)
            chat_sess = get_chat_session(sid)
            if chat_sess:
                chat_sess.approve_permission(granted)
                _send_json(sock, {"ok": True}, request_origin=origin)
            else:
                _send_http(sock, "404 Not Found", "text/plain",
                           b"session not found", request_origin=origin)
            sock.close()
            return

        # ── GET /api/sessions — list or get chat sessions ───────────
        if path.startswith("/api/sessions") and method == "GET":
            if not _check_auth(query, cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return
            from web.api import list_chat_sessions, get_chat_session
            parts_path = path.rstrip("/").split("/")
            if len(parts_path) == 3:
                # GET /api/sessions → list all
                _send_json(sock, {"sessions": list_chat_sessions()},
                           request_origin=origin)
            elif len(parts_path) == 4:
                # GET /api/sessions/{id} → get one
                chat_sess = get_chat_session(parts_path[3])
                if chat_sess:
                    _send_json(sock, {
                        "id": chat_sess.session_id,
                        "messages": chat_sess.get_messages(),
                        "config": chat_sess.get_safe_config(),
                        "busy": not chat_sess.is_idle(),
                    }, request_origin=origin)
                else:
                    _send_http(sock, "404 Not Found", "text/plain",
                               b"session not found", request_origin=origin)
            else:
                _send_http(sock, "404 Not Found", "text/plain",
                           b"Not Found", request_origin=origin)
            sock.close()
            return

        # ── GET/PATCH /api/config — read/write session config ───────
        if path == "/api/config":
            if not _check_auth(query, body_json.get("token", ""),
                               cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return
            from web.api import get_chat_session
            sid = body_json.get("session_id", "") or \
                  (query.split("sid=")[1].split("&")[0]
                   if "sid=" in query else "")
            chat_sess = get_chat_session(sid) if sid else None
            if method == "GET" and chat_sess:
                _send_json(sock, chat_sess.get_safe_config(),
                           request_origin=origin)
            elif method == "PATCH" and chat_sess:
                updated = chat_sess.update_config(body_json.get("config", {}))
                _send_json(sock, updated, request_origin=origin)
            else:
                _send_http(sock, "404 Not Found", "text/plain",
                           b"session not found", request_origin=origin)
            sock.close()
            return

        # ── GET /api/models — list available providers and models ────
        if path == "/api/models" and method == "GET":
            if not _check_auth(query, cookie_str=cookie):
                _send_http(sock, "401 Unauthorized", "text/plain",
                           b"Unauthorized", request_origin=origin)
                sock.close()
                return
            from web.api import get_available_models
            _send_json(sock, {"providers": get_available_models()},
                       request_origin=origin)
            sock.close()
            return

        # ── HTTP: serve page or static files ─────────────────────────
        if method != "GET":
            _send_http(sock, "405 Method Not Allowed", "text/plain",
                       b"Method Not Allowed", request_origin=origin)
            sock.close()
            return

        if path in ("/", "/index.html"):
            body = _build_html(no_auth=_server_no_auth).encode()
            _send_http(sock, "200 OK", "text/html; charset=utf-8", body,
                       request_origin=origin)
        else:
            fname = path.lstrip("/")
            fpath = _WEB_DIR / fname
            if (fpath.exists() and fpath.parent == _WEB_DIR
                    and not fname.startswith(".")):
                body = fpath.read_bytes()
                ctype = _MIME.get(fpath.suffix, "application/octet-stream")
                _send_http(sock, "200 OK", ctype, body,
                           "Cache-Control: public, max-age=86400\r\n",
                           request_origin=origin)
            else:
                _send_http(sock, "404 Not Found", "text/plain", b"Not Found",
                           request_origin=origin)

        sock.close()

    except (TimeoutError, ConnectionResetError, BrokenPipeError):
        pass  # normal for idle/dropped connections
    except Exception as exc:
        import traceback
        print(f"\033[31m[web] {addr} error: {exc}\033[0m", file=sys.stderr,
              flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        try:
            sock.close()
        except OSError:
            pass


# ── Entry point ──────────────────────────────────────────────────────────

def _reap_stale_sessions() -> None:
    """Periodically clean up leaked PTY sessions and stale chat sessions."""
    while True:
        time.sleep(10)
        # PTY sessions (SSE mode)
        now = time.monotonic()
        stale: list[str] = []
        with _sessions_lock:
            for sid, sess in _sessions.items():
                if not sess.attached and (now - sess.created_at) > _SESSION_TIMEOUT:
                    stale.append(sid)
                elif sess.proc.poll() is not None and sess.closed:
                    stale.append(sid)
            for sid in stale:
                sess = _sessions.pop(sid)
                sess.close()
        # Chat sessions (structured API)
        try:
            from web.api import reap_stale_chat_sessions
            reap_stale_chat_sessions()
        except ImportError:
            pass


def start_web_server(
    port: int = DEFAULT_PORT,
    host: str = "127.0.0.1",
    no_auth: bool = False,
) -> None:
    global _server_password, _server_no_auth, _server_cmd

    # Guard against recursive startup (e.g. shell alias maps
    # cheetahclaws → cheetahclaws --web)
    if os.environ.get("CHEETAHCLAWS_WEB_SERVER") == "1":
        print("\033[31mError: recursive --web launch detected. "
              "Check shell aliases.\033[0m", file=sys.stderr)
        sys.exit(1)
    os.environ["CHEETAHCLAWS_WEB_SERVER"] = "1"

    _server_password = None if no_auth else _generate_password()
    _server_no_auth = no_auth

    cc_bin = shutil.which("cheetahclaws")
    if cc_bin:
        _server_cmd = [cc_bin]
    else:
        cc_script = Path(__file__).resolve().parent.parent / "cheetahclaws.py"
        _server_cmd = [sys.executable, str(cc_script)]

    # Start background reaper for orphaned SSE sessions
    threading.Thread(target=_reap_stale_sessions, daemon=True).start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)

    print(f"\n  \033[36mCheetahClaws Web Terminal\033[0m", flush=True)
    print(f"  \033[2m{'─' * 40}\033[0m", flush=True)
    print(f"  Terminal: \033[1mhttp://localhost:{port}\033[0m", flush=True)
    print(f"  Chat UI:  \033[1mhttp://localhost:{port}/chat\033[0m", flush=True)
    if host == "0.0.0.0":
        print(f"  Host:     \033[33m0.0.0.0 (network accessible)\033[0m", flush=True)
    if not no_auth:
        print(f"  Password: \033[1;33m{_server_password}\033[0m", flush=True)
    else:
        print(f"  Auth:     \033[33mdisabled\033[0m", flush=True)
    print(f"  \033[2m{'─' * 40}\033[0m", flush=True)
    print(f"  \033[2mPress Ctrl+C to stop\033[0m\n", flush=True)

    try:
        while True:
            client, addr = srv.accept()
            t = threading.Thread(target=_handle_connection, args=(client, addr),
                                 daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n\033[2mWeb terminal stopped.\033[0m", flush=True)
    finally:
        srv.close()
