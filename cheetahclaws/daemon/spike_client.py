"""spike_client.py — Manual smoke client for the daemon spike.

Stdlib-only (uses http.client). Useful when curl is awkward or you want to
exercise client_id resume.

Subcommands:
  ping                                Call echo.ping
  watch [--since <id>]                Tail /events
  request --tool <T> [--input '{...}']
  answer --request-id <id> [--approve]
  list                                List own pending requests
"""
from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from . import API_VERSION, API_VERSION_HEADER
from .originator import CLIENT_ID_HEADER, CLIENT_KIND_HEADER

DEFAULT_KIND = "spike-cli"


def _client_id_path(kind: str) -> Path:
    return Path.home() / ".cheetahclaws" / "clients" / f"{kind}.id"


def _load_client_id(kind: str) -> Optional[str]:
    p = _client_id_path(kind)
    if p.exists():
        return p.read_text().strip() or None
    return None


def _save_client_id(kind: str, cid: str) -> None:
    p = _client_id_path(kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cid)
    try:
        p.chmod(0o600)
    except OSError:
        pass


# ── Connection helpers ──────────────────────────────────────────────────────


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 30):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._socket_path)


def _open_conn(target: str) -> http.client.HTTPConnection:
    if target.startswith("unix://"):
        return _UnixHTTPConnection(target[len("unix://"):])
    if target.startswith("tcp://"):
        host_port = target[len("tcp://"):]
        host, port = host_port.rsplit(":", 1)
        return http.client.HTTPConnection(host, int(port))
    raise ValueError(f"unknown target {target!r}")


def _headers(kind: str, token: Optional[str]) -> dict:
    h = {
        API_VERSION_HEADER: API_VERSION,
        CLIENT_KIND_HEADER: kind,
        "Content-Type": "application/json",
    }
    cid = _load_client_id(kind)
    if cid:
        h[CLIENT_ID_HEADER] = cid
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _persist_returned_client_id(kind: str, response_headers) -> None:
    cid = response_headers.get(CLIENT_ID_HEADER)
    if cid:
        existing = _load_client_id(kind)
        if existing != cid:
            _save_client_id(kind, cid)


def _rpc(target: str, kind: str, token: Optional[str], method: str, params: dict) -> dict:
    conn = _open_conn(target)
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }).encode()
    conn.request("POST", "/rpc", body=body, headers=_headers(kind, token))
    resp = conn.getresponse()
    raw = resp.read()
    _persist_returned_client_id(kind, resp.headers)
    out = {"status": resp.status}
    try:
        out["body"] = json.loads(raw)
    except Exception:
        out["body"] = raw.decode("utf-8", "replace")
    conn.close()
    return out


# ── Subcommands ─────────────────────────────────────────────────────────────


def cmd_ping(args) -> int:
    out = _rpc(args.target, args.kind, args.token, "echo.ping",
               {"hello": args.message})
    print(json.dumps(out, indent=2))
    return 0


def cmd_request(args) -> int:
    params = {"tool": args.tool}
    if args.input:
        params["input"] = json.loads(args.input)
    if args.timeout_s is not None:
        params["timeout_s"] = args.timeout_s
    out = _rpc(args.target, args.kind, args.token, "permission.demo", params)
    print(json.dumps(out, indent=2))
    return 0


def cmd_answer(args) -> int:
    result = {"approve": args.approve}
    out = _rpc(args.target, args.kind, args.token, "permission.answer",
               {"request_id": args.request_id, "result": result})
    print(json.dumps(out, indent=2))
    return 0


def cmd_list(args) -> int:
    out = _rpc(args.target, args.kind, args.token, "permission.list", {})
    print(json.dumps(out, indent=2))
    return 0


def cmd_watch(args) -> int:
    conn = _open_conn(args.target)
    path = f"/events?since={args.since}"
    conn.request("GET", path, headers=_headers(args.kind, args.token))
    resp = conn.getresponse()
    _persist_returned_client_id(args.kind, resp.headers)
    if resp.status != 200:
        print(f"HTTP {resp.status}: {resp.read().decode()}")
        return 1
    print("# tailing /events; ctrl-c to stop")
    try:
        while True:
            line = resp.fp.readline()
            if not line:
                print("# server closed connection")
                return 0
            sys.stdout.write(line.decode("utf-8", "replace"))
            sys.stdout.flush()
    except KeyboardInterrupt:
        return 0
    finally:
        conn.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="spike-client")
    p.add_argument("--target", default=f"unix://{Path.home()}/.cheetahclaws/run/daemon.sock")
    p.add_argument("--kind", default=DEFAULT_KIND)
    p.add_argument("--token", default=None,
                   help="Bearer token for TCP targets (or read $CHEETAHCLAWS_TOKEN)")
    p.add_argument("--since", type=int, default=0)

    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("ping")
    s.add_argument("--message", default="hi")
    s.set_defaults(func=cmd_ping)

    s = sp.add_parser("watch")
    s.set_defaults(func=cmd_watch)

    s = sp.add_parser("request")
    s.add_argument("--tool", default="Bash")
    s.add_argument("--input", default=None)
    s.add_argument("--timeout-s", type=float, default=None, dest="timeout_s")
    s.set_defaults(func=cmd_request)

    s = sp.add_parser("answer")
    s.add_argument("--request-id", required=True, dest="request_id")
    s.add_argument("--approve", action="store_true")
    s.set_defaults(func=cmd_answer)

    s = sp.add_parser("list")
    s.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    if args.token is None:
        import os as _os
        args.token = _os.environ.get("CHEETAHCLAWS_TOKEN")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
