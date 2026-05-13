"""commands/daemon_cmd.py — `cheetahclaws daemon {status, stop, logs, rotate-token}`.

Dispatched from :func:`cheetahclaws.main` when the first positional argv
is ``daemon``.  All actions read the discovery file written by
``cheetahclaws serve``; absence of that file means "no daemon running".

Auth + token storage live in :mod:`cc_daemon.auth`; discovery in
:mod:`cc_daemon.discovery`.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from cc_daemon import auth as _auth
from cc_daemon import discovery as _discovery


LOG_DIR_NAME = "logs"
LOG_FILENAME = "daemon.log"
DEFAULT_TAIL_LINES = 50
RPC_TIMEOUT_S = 2.0
STOP_WAIT_S = 5.0

# Default token path matches cc_daemon.cli.DEFAULT_TOKEN_PATH; resolved
# lazily via _default_token_path() so unit tests can monkeypatch it.


def _default_token_path() -> Path:
    from cc_daemon.cli import DEFAULT_TOKEN_PATH
    return DEFAULT_TOKEN_PATH


def _resolve_token_path(info: Optional[dict]) -> Path:
    """Prefer the token path the daemon recorded in discovery (set when
    `serve --token-path` overrode the default); fall back to the default
    location otherwise."""
    if info is not None:
        recorded = info.get("token_path")
        if isinstance(recorded, str) and recorded:
            return Path(recorded).expanduser()
    return _default_token_path()


# ── Top-level dispatch ─────────────────────────────────────────────────────

def dispatch(argv: list[str]) -> int:
    if not argv:
        print("usage: cheetahclaws daemon {status|stop|logs|rotate-token} [options]",
              file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "status":
        return _status(rest)
    if cmd == "stop":
        return _stop(rest)
    if cmd == "logs":
        return _logs(rest)
    if cmd == "rotate-token":
        return _rotate_token(rest)
    print(f"unknown daemon action: {cmd}", file=sys.stderr)
    return 2


# ── status ─────────────────────────────────────────────────────────────────

def _status(argv: list[str]) -> int:
    info = _discovery.locate()
    if info is None:
        print("cheetahclaws daemon: not running", file=sys.stderr)
        return 1
    started = info.get("started_at", "?")
    uptime_s = _seconds_since(started)
    print(f"pid:         {info.get('pid', '?')}")
    print(f"transport:   {info.get('transport', '?')}")
    print(f"address:     {info.get('address', '?')}")
    print(f"version:     {info.get('version', '?')}")
    print(f"started_at:  {started}")
    if uptime_s is not None:
        print(f"uptime:      {_format_duration(uptime_s)}")

    ok, payload = _call_rpc("system.ping")
    if ok and isinstance(payload, dict) and payload.get("result") == "pong":
        print("ping:        pong")
    else:
        print(f"ping:        FAILED ({payload})", file=sys.stderr)
        return 2

    # F-9 (RFC 0002 §F-9) — surface the live serve-mode budgets so the
    # operator can sanity-check the defaults are wired in.  system.status
    # is best-effort: an older daemon that pre-dates F-9 returns
    # METHOD_NOT_FOUND, which we treat as "no budgets to display".
    ok, payload = _call_rpc("system.status")
    if ok and isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        result = payload["result"]
        budgets = result.get("budgets") or {}
        if any(v is not None for v in budgets.values()):
            print("budgets:")
            for k in ("session_token_budget", "session_cost_budget",
                     "daily_token_budget",   "daily_cost_budget"):
                v = budgets.get(k)
                if v is None:
                    rendered = "unlimited"
                elif "cost" in k:
                    rendered = f"${float(v):.2f}"
                else:
                    rendered = f"{int(v):,} tokens"
                print(f"  {k:<22} {rendered}")
        runners = result.get("runners", 0)
        bridges = result.get("bridges", 0)
        if runners or bridges:
            print(f"runners:     {runners}")
            print(f"bridges:     {bridges}")
    return 0


# ── stop ───────────────────────────────────────────────────────────────────

def _stop(argv: list[str]) -> int:
    info = _discovery.locate()
    if info is None:
        print("cheetahclaws daemon: not running", file=sys.stderr)
        return 0  # already in the desired state

    pid = info.get("pid")

    # Preferred path: graceful shutdown via RPC.
    rpc_ok, rpc_result = _call_rpc("system.shutdown")
    if not rpc_ok and isinstance(pid, int) and pid > 0:
        # Fallback: SIGTERM (POSIX) / TerminateProcess (Windows).
        try:
            if os.name == "nt":
                _terminate_windows(pid)
            else:
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            print(f"warning: signal delivery failed: {exc}", file=sys.stderr)

    # Wait for the daemon to clear its discovery file.
    deadline = time.monotonic() + STOP_WAIT_S
    while time.monotonic() < deadline:
        if _discovery.locate() is None:
            print("cheetahclaws daemon: stopped")
            return 0
        time.sleep(0.1)

    print("cheetahclaws daemon: did not stop within timeout", file=sys.stderr)
    return 1


# ── logs ───────────────────────────────────────────────────────────────────

def _logs(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="cheetahclaws daemon logs",
                                      description="Print recent daemon log lines.")
    parser.add_argument("-n", "--lines", type=int, default=DEFAULT_TAIL_LINES,
                        help=f"Number of trailing lines to print (default {DEFAULT_TAIL_LINES}).")
    args = parser.parse_args(argv)

    path = _log_path()
    if not path.exists():
        print(f"cheetahclaws daemon: no log file at {path}", file=sys.stderr)
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"error: cannot read log: {exc}", file=sys.stderr)
        return 1
    lines = text.splitlines()
    for line in lines[-max(args.lines, 0):]:
        print(line)
    return 0


# ── rotate-token ───────────────────────────────────────────────────────────

def _rotate_token(argv: list[str]) -> int:
    info = _discovery.locate()
    token_path = _resolve_token_path(info)
    _auth.rotate_token(token_path)
    print(f"cheetahclaws: rotated token at {token_path}")
    if info and info.get("transport") == "tcp":
        print("note: existing TCP clients will receive 401 on next request "
              "until they re-read the token file.")
    return 0


# ── RPC client ─────────────────────────────────────────────────────────────

def _call_rpc(method: str, params: Any = None) -> Tuple[bool, Any]:
    """Call a JSON-RPC method on the running daemon.

    Returns ``(ok, payload)``.  ``payload`` is the parsed JSON envelope on
    success or an error string.
    """
    info = _discovery.locate()
    if info is None:
        return False, "not running"
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": method,
        **({"params": params} if params is not None else {}),
    }).encode("utf-8")

    transport = info.get("transport")
    address = info.get("address", "")
    # cc_daemon's server enforces the API-version header; sending it lets
    # the request through the version gate.
    from cc_daemon import API_VERSION, API_VERSION_HEADER
    headers = {"Content-Type": "application/json",
               "Content-Length": str(len(body)),
               "Host": "localhost",
               API_VERSION_HEADER: API_VERSION}

    if transport == "tcp":
        token = _auth.load_or_create_token(_resolve_token_path(info))
        headers["Authorization"] = f"Bearer {token}"
        return _post_tcp(address, "/rpc", body, headers)
    if transport == "unix":
        return _post_unix(address, "/rpc", body, headers)
    return False, f"unknown transport: {transport}"


def _post_tcp(address: str, path: str, body: bytes,
              headers: dict) -> Tuple[bool, Any]:
    host, port_s = address.rsplit(":", 1)
    try:
        port = int(port_s)
    except ValueError:
        return False, f"bad address: {address}"
    conn = http.client.HTTPConnection(host, port, timeout=RPC_TIMEOUT_S)
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status >= 400:
            return False, f"http {resp.status}: {raw[:200].decode('utf-8','replace')}"
        try:
            return True, json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return False, "non-JSON response"
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()


def _post_unix(sock_path: str, path: str, body: bytes,
               headers: dict) -> Tuple[bool, Any]:
    """Send a one-shot HTTP request over a Unix domain socket."""
    if not hasattr(socket, "AF_UNIX"):
        return False, "Unix sockets not supported on this platform"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(RPC_TIMEOUT_S)
    try:
        sock.connect(sock_path)
        request_lines = [f"POST {path} HTTP/1.1"]
        for k, v in headers.items():
            request_lines.append(f"{k}: {v}")
        request_lines.append("Connection: close")
        request_lines.append("")
        request_lines.append("")
        sock.sendall("\r\n".join(request_lines).encode("ascii") + body)
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    except Exception as exc:
        return False, str(exc)
    finally:
        sock.close()

    head, _sep, tail = raw.partition(b"\r\n\r\n")
    if not _sep:
        return False, "malformed response"
    status_line = head.split(b"\r\n", 1)[0].decode("ascii", "replace")
    parts = status_line.split(" ", 2)
    if len(parts) < 2:
        return False, f"malformed status line: {status_line!r}"
    try:
        status = int(parts[1])
    except ValueError:
        return False, f"non-numeric status: {parts[1]!r}"
    if status >= 400:
        return False, f"http {status}: {tail[:200].decode('utf-8','replace')}"
    try:
        return True, json.loads(tail.decode("utf-8"))
    except json.JSONDecodeError:
        return False, "non-JSON response"


# ── Helpers ────────────────────────────────────────────────────────────────

def _log_path() -> Path:
    from cc_config import CONFIG_DIR
    return CONFIG_DIR / LOG_DIR_NAME / LOG_FILENAME


def _seconds_since(iso_ts: str) -> Optional[float]:
    """Best-effort ISO 8601 → seconds-since-now."""
    try:
        import datetime as _dt
        # Strip trailing Z, parse, treat as UTC.
        ts = iso_ts.rstrip("Z")
        started = _dt.datetime.fromisoformat(ts).replace(tzinfo=_dt.timezone.utc)
        now = _dt.datetime.now(_dt.timezone.utc)
        return max(0.0, (now - started).total_seconds())
    except Exception:
        return None


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{h}h {m}m {sec}s"


def _terminate_windows(pid: int) -> None:
    """Best-effort TerminateProcess on Windows."""
    import ctypes
    PROCESS_TERMINATE = 0x0001
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
    if not handle:
        return
    try:
        kernel32.TerminateProcess(handle, 1)
    finally:
        kernel32.CloseHandle(handle)
