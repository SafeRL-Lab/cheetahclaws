"""kernel/cli.py — `cheetahclaws kernel ...` subcommand.

Talks to a running daemon (started via ``cheetahclaws serve
--enable-kernel``) over the existing daemon RPC channel and pretty-
prints kernel.* RPC results to the terminal. Read-only inspection
verbs only — mutations (create / spawn / charge) belong on the
Python API or future RFC-defined verbs.

Exit codes:
  0  success
  1  daemon not running OR kernel not enabled
  2  argv error / RPC error
"""
from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from cheetahclaws.daemon import API_VERSION, API_VERSION_HEADER
from cheetahclaws.daemon import auth as _auth
from cheetahclaws.daemon import discovery as _discovery


RPC_TIMEOUT_S = 5.0


# ── Top-level dispatch ────────────────────────────────────────────────────


def dispatch(argv: list[str]) -> int:
    if not argv:
        _print_usage()
        return 2
    cmd, rest = argv[0], argv[1:]
    handlers = {
        "summary":    _cmd_summary,
        "info":       _cmd_info,
        "agents":     _cmd_agents,
        "proc":       _cmd_proc,
        "events":     _cmd_events,
        "queue":      _cmd_queue,
        "registry":   _cmd_registry,
        "methods":    _cmd_methods,
        "prometheus": _cmd_prometheus,
    }
    handler = handlers.get(cmd)
    if handler is None:
        if cmd in ("-h", "--help", "help"):
            _print_usage()
            return 0
        print(f"unknown kernel action: {cmd!r}", file=sys.stderr)
        _print_usage()
        return 2
    return handler(rest)


def _print_usage() -> None:
    print(
        "usage: cheetahclaws kernel <action> [options]\n"
        "\n"
        "Actions:\n"
        "  summary             Show kernel-wide rollup (uptime, agents, queue, …)\n"
        "  info                Show kernel version / schema / API surface counts\n"
        "  agents [--state S]  List agents (filter by state)\n"
        "  proc <pid>          Combined per-agent view (cap, ledger, mailbox, …)\n"
        "  events [--pid P]    Tail recent events\n"
        "  queue [--state S]   List scheduler queue entries\n"
        "  registry [--prefix P] [--tag T]  List registry entries\n"
        "  methods [--tier T]  List documented kernel.* methods\n"
        "  prometheus          Print Prometheus exposition text\n"
        "\n"
        "Daemon must be running with `cheetahclaws serve --enable-kernel`.",
        file=sys.stderr,
    )


# ── Command handlers ───────────────────────────────────────────────────────


def _cmd_summary(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cheetahclaws kernel summary",
                                add_help=True)
    p.add_argument("--json", action="store_true",
                   help="Print raw JSON instead of formatted text")
    args = p.parse_args(argv)

    ok, resp = _call_rpc("kernel.observe.summary", {})
    if not ok:
        return _print_rpc_error(resp)
    result = _extract_result(resp)
    if result is None:
        return _print_rpc_error("no result in response")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"kernel:        {result.get('kernel_version', '?')}")
    print(f"schema:        v{result.get('schema_version', '?')}")
    print(f"uptime:        {_format_duration(result.get('uptime_s', 0))}")
    agents = result.get("agents", {})
    print(f"agents:        total={agents.get('total', 0)}  "
          f"READY={agents.get('READY', 0)}  "
          f"RUNNING={agents.get('RUNNING', 0)}  "
          f"WAITING={agents.get('WAITING', 0)}  "
          f"SUSPENDED={agents.get('SUSPENDED', 0)}  "
          f"DEAD={agents.get('DEAD', 0)}")
    events = result.get("events", {})
    print(f"events:        total={events.get('total', 0)}  "
          f"max_id={events.get('max_event_id', 0)}")
    sched = result.get("scheduler", {})
    print(f"scheduler:     queued={sched.get('queued', 0)}  "
          f"dispatched={sched.get('dispatched', 0)}  "
          f"completed={sched.get('completed', 0)}  "
          f"expired={sched.get('expired', 0)}  "
          f"cancelled={sched.get('cancelled', 0)}")
    led = result.get("ledger", {})
    print(f"ledger:        agents_with_budgets={led.get('agents_with_budgets', 0)}  "
          f"breached={led.get('breached', 0)}")
    mb = result.get("mailbox", {})
    print(f"mailbox:       mailboxes={mb.get('mailboxes', 0)}  "
          f"pending={mb.get('pending_messages', 0)}")
    fs = result.get("fs", {})
    print(f"fs:            objects={fs.get('objects', 0)}  "
          f"bytes={_format_bytes(fs.get('total_bytes', 0))}")
    reg = result.get("registry", {})
    print(f"registry:      entries={reg.get('entries', 0)}")
    return 0


def _cmd_info(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cheetahclaws kernel info")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    ok, resp = _call_rpc("kernel.api.version_info", {})
    if not ok:
        return _print_rpc_error(resp)
    result = _extract_result(resp)
    if result is None:
        return _print_rpc_error("no result")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"kernel_version:   {result.get('kernel_version', '?')}")
    print(f"schema_version:   v{result.get('schema_version', '?')}")
    print(f"api_version:      {result.get('api_version', '?')}")
    print(f"method_count:     {result.get('method_count', 0)}")
    tiers = result.get("tier_counts", {})
    print(f"tier_counts:      stable={tiers.get('stable', 0)}  "
          f"experimental={tiers.get('experimental', 0)}  "
          f"deprecated={tiers.get('deprecated', 0)}")
    rfcs = result.get("rfcs_implemented", [])
    print(f"rfcs_implemented: {', '.join(str(n) for n in rfcs)}")
    return 0


def _cmd_agents(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cheetahclaws kernel agents")
    p.add_argument("--state", help="Filter by state (READY|RUNNING|...)")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    params: dict = {"limit": args.limit}
    if args.state:
        params["state"] = args.state

    ok, resp = _call_rpc("kernel.agent.list", params)
    if not ok:
        return _print_rpc_error(resp)
    result = _extract_result(resp)
    if result is None:
        return _print_rpc_error("no result")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    agents = result.get("agents", [])
    total = result.get("total", 0)
    if not agents:
        print(f"no agents (total={total})")
        return 0
    print(f"{'PID':>6}  {'STATE':<10}  {'NAME':<24}  TEMPLATE")
    print("-" * 70)
    for a in agents:
        print(f"{a['pid']:>6}  {a['state']:<10}  "
              f"{a['name'][:24]:<24}  {a.get('template', '')}")
    print(f"\n({len(agents)} of {total})")
    return 0


def _cmd_proc(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cheetahclaws kernel proc")
    p.add_argument("pid", type=int)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    ok, resp = _call_rpc("kernel.observe.proc", {"pid": args.pid})
    if not ok:
        return _print_rpc_error(resp)
    result = _extract_result(resp)
    if result is None:
        return _print_rpc_error("no result")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    proc = result.get("process")
    if proc is None:
        print(f"no process for pid={args.pid}")
        return 1
    print(f"pid:        {proc['pid']}")
    print(f"name:       {proc['name']}")
    print(f"template:   {proc['template']}")
    print(f"state:      {proc['state']}"
          + (f"  ({proc['state_reason']})" if proc.get('state_reason') else ""))
    print(f"parent_pid: {proc.get('parent_pid', '—')}")
    if proc.get("exit_kind"):
        print(f"exit_kind:  {proc['exit_kind']}")

    cap = result.get("capability")
    if cap:
        print(f"\ncapability (cap_id={cap['cap_id']}, "
              f"sub_agent={cap['sub_agent']}):")
        print(f"  tools:  {cap.get('tool_grants', [])}")
        print(f"  fs:     {cap.get('fs_grants', [])}")
        print(f"  net:    {cap.get('net_grants', [])}")
        print(f"  models: {cap.get('model_grants', [])}")
    else:
        print("\ncapability: (none)")

    ledger = result.get("ledger", [])
    if ledger:
        print("\nledger:")
        for e in ledger:
            pct = (100.0 * e['used'] / e['granted']) if e['granted'] else 0
            print(f"  {e['dim']:<12}  used={e['used']}/{e['granted']}  "
                  f"({pct:.1f}%)")
    else:
        print("\nledger: (no budgets)")

    mb = result.get("mailbox", {})
    if mb.get("exists"):
        print(f"\nmailbox: queue_size={mb['queue_size']}  "
              f"pending={mb['pending']}  "
              f"subscriptions={mb.get('subscriptions', [])}")
    else:
        print("\nmailbox: (none)")

    sched = result.get("scheduler", {})
    print(f"\nscheduler: queued={sched.get('queued', 0)}  "
          f"dispatched={sched.get('dispatched', 0)}  "
          f"completed={sched.get('completed', 0)}")

    fs = result.get("fs", {})
    print(f"\nfs: objects={fs.get('object_count', 0)}  "
          f"bytes={_format_bytes(fs.get('total_bytes', 0))}")

    reg = result.get("registry", {})
    if reg.get("names"):
        print(f"\nregistry names: {reg['names']}")

    recent = result.get("recent_events", [])
    if recent:
        print(f"\nrecent events ({len(recent)}):")
        for e in recent[-10:]:  # last 10
            print(f"  [{e['event_id']}] {e['kind']}")
    return 0


def _cmd_events(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cheetahclaws kernel events")
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--kind", default=None)
    p.add_argument("--since", type=int, default=0)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    params: dict = {"since_event_id": args.since, "limit": args.limit}
    if args.pid is not None: params["pid"] = args.pid
    if args.kind: params["kind"] = args.kind

    ok, resp = _call_rpc("kernel.events.tail", params)
    if not ok:
        return _print_rpc_error(resp)
    result = _extract_result(resp)
    if result is None:
        return _print_rpc_error("no result")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    events = result.get("events", [])
    if not events:
        print("(no events)")
        return 0
    for e in events:
        ts = e.get("ts", 0)
        print(f"[{e['event_id']:>5}] pid={e['pid']:>4} "
              f"{e['kind']:<32} payload={json.dumps(e.get('payload') or {})[:120]}")
    print(f"\nnext_cursor={result.get('next_cursor', 0)}")
    return 0


def _cmd_queue(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cheetahclaws kernel queue")
    p.add_argument("--state",
                   help="queued|dispatched|completed|expired|cancelled")
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    params: dict = {"limit": args.limit}
    if args.state: params["state"] = args.state
    if args.pid is not None: params["pid"] = args.pid

    ok, resp = _call_rpc("kernel.sched.list", params)
    if not ok:
        return _print_rpc_error(resp)
    result = _extract_result(resp)
    if result is None:
        return _print_rpc_error("no result")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    entries = result.get("entries", [])
    total = result.get("total", 0)
    if not entries:
        print(f"queue empty (total={total})")
        return 0
    print(f"{'SCHED':>6}  {'PID':>4}  {'PRI':>4}  {'STATE':<11}  {'TRIGGER':<10}")
    print("-" * 50)
    for e in entries:
        print(f"{e['sched_id']:>6}  {e['pid']:>4}  {e['priority']:>4}  "
              f"{e['state']:<11}  {e.get('trigger', '')[:10]}")
    print(f"\n({len(entries)} of {total})")
    return 0


def _cmd_registry(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cheetahclaws kernel registry")
    p.add_argument("--prefix", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    params: dict = {"limit": args.limit}
    if args.prefix: params["prefix"] = args.prefix
    if args.tag:    params["tag"] = args.tag

    ok, resp = _call_rpc("kernel.registry.list", params)
    if not ok:
        return _print_rpc_error(resp)
    result = _extract_result(resp)
    if result is None:
        return _print_rpc_error("no result")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    entries = result.get("entries", [])
    total = result.get("total", 0)
    if not entries:
        print(f"registry empty (total={total})")
        return 0
    print(f"{'PID':>4}  {'NAME':<40}  TAGS")
    print("-" * 70)
    for e in entries:
        print(f"{e['pid']:>4}  {e['name'][:40]:<40}  {','.join(e.get('tags', []))}")
    print(f"\n({len(entries)} of {total})")
    return 0


def _cmd_methods(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cheetahclaws kernel methods")
    p.add_argument("--tier",
                   choices=("stable", "experimental", "deprecated"))
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    params: dict = {}
    if args.tier: params["tier"] = args.tier

    ok, resp = _call_rpc("kernel.api.list_methods", params)
    if not ok:
        return _print_rpc_error(resp)
    result = _extract_result(resp)
    if result is None:
        return _print_rpc_error("no result")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    methods = result.get("methods", [])
    for m in methods:
        print(m)
    counts = result.get("tier_counts", {})
    print(f"\ntotal: stable={counts.get('stable', 0)} "
          f"experimental={counts.get('experimental', 0)} "
          f"deprecated={counts.get('deprecated', 0)}")
    return 0


def _cmd_prometheus(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cheetahclaws kernel prometheus")
    p.parse_args(argv)
    ok, resp = _call_rpc("kernel.observe.prometheus", {})
    if not ok:
        return _print_rpc_error(resp)
    result = _extract_result(resp)
    if result is None:
        return _print_rpc_error("no result")
    text = result.get("text", "")
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


# ── Helpers ───────────────────────────────────────────────────────────────


def _extract_result(resp: Any) -> Optional[dict]:
    """JSON-RPC envelope → result dict, or None on error envelope."""
    if not isinstance(resp, dict):
        return None
    if "result" in resp:
        return resp["result"]
    if "error" in resp:
        err = resp["error"] or {}
        msg = err.get("message", "?")
        code = err.get("code", "?")
        print(f"kernel rpc error [{code}]: {msg}", file=sys.stderr)
        return None
    return None


def _print_rpc_error(reason: Any) -> int:
    """Translate connection errors into a friendly stderr message and
    return exit code 1 for daemon-not-running, 2 for everything else."""
    msg = str(reason)
    print(f"cheetahclaws kernel: {msg}", file=sys.stderr)
    if "not running" in msg or "Connection refused" in msg or \
       "No such file or directory" in msg:
        print("hint: is `cheetahclaws serve --enable-kernel` running?",
              file=sys.stderr)
        return 1
    if "Method not found" in msg or "method_not_found" in msg \
       or "method 'kernel." in msg or "kernel.observe" in msg \
       and "not found" in msg.lower():
        print("hint: daemon is running but kernel is not enabled. "
              "restart with --enable-kernel.", file=sys.stderr)
        return 1
    return 2


def _format_duration(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def _format_bytes(n: int) -> str:
    if not isinstance(n, (int, float)) or n < 0:
        return "?"
    if n < 1024:           return f"{int(n)}B"
    if n < 1024**2:        return f"{n/1024:.1f}KB"
    if n < 1024**3:        return f"{n/1024**2:.1f}MB"
    return f"{n/1024**3:.2f}GB"


# ── RPC client (mirrors commands/daemon_cmd.py pattern) ──────────────────


def _call_rpc(method: str, params: Optional[dict] = None) -> Tuple[bool, Any]:
    info = _discovery.locate()
    if info is None:
        return False, "daemon not running"
    body_obj: dict = {
        "jsonrpc": "2.0", "id": 1, "method": method,
    }
    if params is not None:
        body_obj["params"] = params
    body = json.dumps(body_obj).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "Host": "localhost",
        API_VERSION_HEADER: API_VERSION,
    }
    transport = info.get("transport")
    address = info.get("address", "")

    if transport == "tcp":
        token_path = _resolve_token_path(info)
        token = _auth.load_or_create_token(token_path)
        headers["Authorization"] = f"Bearer {token}"
        return _post_tcp(address, "/rpc", body, headers)
    if transport == "unix":
        return _post_unix(address, "/rpc", body, headers)
    return False, f"unknown transport: {transport}"


def _resolve_token_path(info: Optional[dict]) -> Path:
    if info is not None:
        recorded = info.get("token_path")
        if isinstance(recorded, str) and recorded:
            return Path(recorded).expanduser()
    from cheetahclaws.daemon.cli import DEFAULT_TOKEN_PATH
    return DEFAULT_TOKEN_PATH


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
            try:
                err = json.loads(raw.decode("utf-8"))
                return False, f"http {resp.status}: {err.get('error', raw[:200])}"
            except json.JSONDecodeError:
                return False, f"http {resp.status}: {raw[:200].decode('utf-8','replace')}"
        try:
            return True, json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return False, "non-JSON response"
    except (ConnectionRefusedError, OSError) as exc:
        return False, str(exc)
    finally:
        conn.close()


def _post_unix(sock_path: str, path: str, body: bytes,
               headers: dict) -> Tuple[bool, Any]:
    if not hasattr(socket, "AF_UNIX"):
        return False, "Unix sockets not supported on this platform"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(RPC_TIMEOUT_S)
    try:
        sock.connect(sock_path)
        request_lines = [f"POST {path} HTTP/1.1"]
        for k, v in headers.items():
            request_lines.append(f"{k}: {v}")
        request_lines.append("")
        request_lines.append("")
        head = "\r\n".join(request_lines).encode("utf-8")
        sock.sendall(head + body)
        # Read response.
        chunks = bytearray()
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            chunks.extend(chunk)
        return _parse_http_response(bytes(chunks))
    except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
        return False, str(exc)
    finally:
        sock.close()


def _parse_http_response(raw: bytes) -> Tuple[bool, Any]:
    sep = b"\r\n\r\n"
    if sep not in raw:
        return False, "malformed http response"
    header, body = raw.split(sep, 1)
    first_line = header.split(b"\r\n", 1)[0].decode("utf-8", "replace")
    parts = first_line.split(" ", 2)
    if len(parts) < 2:
        return False, f"malformed status line: {first_line!r}"
    try:
        status = int(parts[1])
    except ValueError:
        return False, f"non-numeric status: {parts[1]!r}"
    if status >= 400:
        return False, f"http {status}: {body[:200].decode('utf-8','replace')}"
    try:
        return True, json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return False, "non-JSON response body"


# ── Allow `python -m kernel.cli ...` for debug / scripting ────────────


if __name__ == "__main__":
    sys.exit(dispatch(sys.argv[1:]))
