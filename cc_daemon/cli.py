"""cli.py — `cheetahclaws serve` entry point.

The interactive daemon-control verbs (`cheetahclaws daemon status / stop /
logs / rotate-token`) live in :mod:`commands.daemon_cmd`; this module is
just the long-running serve loop.

Layered on top of the spike's `make_tcp_server` / `make_unix_server`
constructors, with these additions for the foundation:

* Calls :func:`bootstrap.bootstrap` so logging / tool registry are wired
  up the same way as the REPL.
* Pins ``log_file`` to ``<data_dir>/logs/daemon.log`` (overridable via
  user config) so ``cheetahclaws daemon logs`` has signal to tail.
* Threads the loaded ``config`` and ``unauthenticated_metrics`` flag
  through ``DaemonState`` so ``/healthz`` / ``/readyz`` / ``/metrics``
  return real ``health.py`` payloads.
* Writes ``~/.cheetahclaws/daemon.json`` (discovery) on bind and removes
  it on exit, in addition to the spike's pid file.
* Watches ``DaemonState.shutdown_event`` so ``system.shutdown`` over RPC
  triggers graceful exit cross-platform (Windows can't deliver SIGTERM
  cleanly to another Python process).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from . import discovery
from .auth import load_or_create_token


DEFAULT_DATA_DIR = Path.home() / ".cheetahclaws"
DEFAULT_RUN_DIR = DEFAULT_DATA_DIR / "run"
DEFAULT_UNIX_SOCKET = DEFAULT_RUN_DIR / "daemon.sock"
DEFAULT_TOKEN_PATH = DEFAULT_DATA_DIR / "daemon_token"
DEFAULT_PID_FILE = DEFAULT_RUN_DIR / "daemon.pid"


# ── F-9: serve-mode cost-guardrail defaults ───────────────────────────────
#
# REPL ``--in-process`` mode keeps the all-None defaults from cc_config so
# existing users see no surprise.  Headless ``cheetahclaws serve`` mode
# applies *conservative* defaults instead: a daemon often runs unattended
# for hours/days, an unbounded agent that's quietly compounding costs
# while no one is watching is the failure mode F-9 (#68) addresses.
#
# Operators who want a different ceiling override via:
#   * cc_config keys (`session_token_budget` etc.), or
#   * the agent.resume RPC's `budget_overrides` argument.
F9_SERVE_BUDGET_DEFAULTS = {
    "session_token_budget":  200_000,
    "session_cost_budget":   2.0,
    "daily_token_budget":    2_000_000,
    "daily_cost_budget":     20.0,
}


def _apply_serve_defaults(config: dict) -> dict:
    """Flip any ``None`` budget keys to the F-9 conservative defaults.

    Pure function for testability — returns the mutated config dict so
    tests can assert on the resulting state without poking module-level
    globals.  Bypassed entirely when an operator has already set the
    key to a real number (zero or above), so explicit user choices
    always win.
    """
    for key, default in F9_SERVE_BUDGET_DEFAULTS.items():
        if config.get(key) is None:
            config[key] = default
    return config


# ── --listen parsing ───────────────────────────────────────────────────────

def parse_listen(spec: str) -> tuple[str, object]:
    """Return ``("unix", Path)`` or ``("tcp", (host, port))``."""
    if spec.startswith("unix://"):
        return "unix", Path(spec[len("unix://"):]).expanduser()
    if spec.startswith("tcp://"):
        host_port = spec[len("tcp://"):]
        if ":" not in host_port:
            raise ValueError(f"tcp listen must be tcp://host:port, got {spec!r}")
        host, port_s = host_port.rsplit(":", 1)
        if not host:
            raise ValueError(f"tcp listen host empty: {spec!r}")
        try:
            port = int(port_s)
        except ValueError as exc:
            raise ValueError(f"tcp listen port not int: {spec!r}") from exc
        if not (0 <= port <= 65535):
            raise ValueError(f"tcp listen port out of range: {spec!r}")
        return "tcp", (host, port)
    raise ValueError(
        f"unknown listen spec {spec!r}; use unix://path or tcp://host:port"
    )


# ── argparse for `cheetahclaws serve` ─────────────────────────────────────

def _build_serve_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cheetahclaws serve",
        description="Run the headless cheetahclaws daemon.",
    )
    p.add_argument("--listen", default=None,
                   help=f"unix://path or tcp://host:port "
                        f"(default unix://{DEFAULT_UNIX_SOCKET})")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                   help="Directory for token / pid / discovery / audit files.")
    p.add_argument("--token-path", default=str(DEFAULT_TOKEN_PATH),
                   help="TCP bearer-token file path (TCP transport only).")
    p.add_argument("--no-audit", action="store_true",
                   help="Disable audit log (default: on for both transports).")
    p.add_argument("--print-token", action="store_true",
                   help="Print the TCP bearer token to stdout (TCP only).")
    p.add_argument("--unauthenticated-metrics", action="store_true",
                   help="Serve /healthz, /readyz, /metrics without auth "
                        "(off by default; opt-in for Prometheus scrapers).")
    # ── cc_kernel (RFC 0003) — opt-in only, default off. ──────────────────
    # When absent, cc_kernel is never imported and the daemon behaviour is
    # byte-for-byte identical to the pre-RFC build (existing users see no
    # change). When present, kernel.db is opened, startup recovery runs,
    # and the kernel.* RPC methods join the registry.
    p.add_argument("--enable-kernel", action="store_true",
                   help="Activate cc_kernel (RFC 0003: AgentProcess + EventLog). "
                        "Off by default; existing users see no change.")
    p.add_argument("--kernel-db", default=None,
                   help="Path to kernel.db (default: <data-dir>/kernel.db). "
                        "Only used with --enable-kernel.")
    p.add_argument("--kernel-recovery", choices=("suspend", "mark-dead"),
                   default="suspend",
                   help="What to do with stale RUNNING/WAITING rows on "
                        "startup. 'suspend' (default) is safe and "
                        "reversible; 'mark-dead' is unconditional. "
                        "Only used with --enable-kernel.")
    return p


def serve_main(argv: Optional[list[str]] = None) -> int:
    """Entry point used by ``cheetahclaws serve`` (dispatched from cheetahclaws.py)."""
    parser = _build_serve_parser()
    args = parser.parse_args(argv)
    return cmd_serve(args)


# ── The actual daemon loop ────────────────────────────────────────────────

def cmd_serve(args: argparse.Namespace) -> int:
    from .server import make_tcp_server, make_unix_server

    listen = args.listen or f"unix://{DEFAULT_UNIX_SOCKET}"
    try:
        transport, addr = parse_listen(listen)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if transport == "unix" and os.name == "nt":
        print("error: Unix sockets unavailable on Windows; "
              "use --listen tcp://host:port instead.", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir).expanduser()
    pid_file = (DEFAULT_PID_FILE if args.data_dir == str(DEFAULT_DATA_DIR)
                else data_dir / "run" / "daemon.pid")

    existing = _read_pidfile(pid_file)
    if existing and discovery.pid_alive(existing):
        print(f"daemon already running (pid={existing})", file=sys.stderr)
        return 1

    # ── Load config + bootstrap (logging, tool registry) ──────────────────
    from cc_config import load_config
    from bootstrap import bootstrap as _bootstrap
    config = load_config()
    if not config.get("log_file"):
        log_dir = data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        config["log_file"] = str(log_dir / "daemon.log")
    # Bump default log level so `daemon logs` has signal in serve mode.
    if config.get("log_level", "warn") == "warn":
        config["log_level"] = "info"
    # F-9 (RFC 0002 §F-9) — apply conservative cost-guardrail defaults
    # in serve mode.  Done BEFORE _bootstrap so the quota module sees the
    # final values on first init.
    _apply_serve_defaults(config)
    _bootstrap(config)

    # F-2: ensure the daemon's SQLite tables exist before the listener
    # starts publishing events / accepting jobs / writing monitor state.
    # init_schema is idempotent so re-running on an existing DB is a no-op.
    from . import schema as _schema
    _schema.init_schema()

    # Pin the health.py module-level config so default-arg payload helpers
    # see the model name on every call.
    import health as _health
    _health.install_config(config)

    audit_enabled = not args.no_audit
    token_path_for_discovery: Optional[str] = None
    if transport == "unix":
        server = make_unix_server(
            addr, data_dir=data_dir,
            audit_enabled=audit_enabled,
            unauthenticated_metrics=args.unauthenticated_metrics,
            config=config,
        )
        listen_repr = f"unix://{addr}"
        actual_address = str(addr)
    else:
        token_file = Path(args.token_path).expanduser()
        token = load_or_create_token(token_file)
        host, port = addr  # type: ignore[misc]
        server = make_tcp_server(
            host, port, data_dir=data_dir, token=token,
            audit_enabled=audit_enabled,
            unauthenticated_metrics=args.unauthenticated_metrics,
            config=config,
        )
        # If port=0 was passed, capture the actual kernel-chosen port.
        actual_port = server.server_address[1]
        listen_repr = f"tcp://{host}:{actual_port}"
        actual_address = f"{host}:{actual_port}"
        # Record token_path in discovery only when --token-path overrides
        # the default; daemon-control verbs use it to load the right token.
        if token_file.resolve() != DEFAULT_TOKEN_PATH.resolve():
            token_path_for_discovery = str(token_file)
        if args.print_token:
            print(f"token: {token}", flush=True)

    # ── cc_kernel activation (RFC 0003) — strictly opt-in. ───────────────
    # Importing cc_kernel is gated on the flag so the no-flag default
    # path imports nothing new and pays no startup cost.
    if getattr(args, "enable_kernel", False):
        kernel_db = Path(args.kernel_db).expanduser() if args.kernel_db \
            else (data_dir / "kernel.db")
        try:
            from cc_kernel import register_with_daemon as _register_kernel
            _register_kernel(
                server.daemon_state, kernel_db,
                recovery=args.kernel_recovery,
            )
        except Exception as exc:
            # Failing to bring up the kernel must not silently downgrade
            # the daemon; better to refuse to start than to lie about
            # serving the kernel surface.
            print(f"error: --enable-kernel: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            try:
                server.server_close()
            except Exception:
                pass
            return 3

    _write_pidfile(pid_file)

    # Discovery file — REPL/Web/bridge clients look here to find us.
    info = discovery.make_info(
        pid=os.getpid(), transport=transport,
        address=actual_address, version=_lookup_version(),
        token_path=token_path_for_discovery,
    )
    try:
        discovery.write(info)
    except OSError as exc:
        print(f"warning: discovery write failed: {exc}", file=sys.stderr, flush=True)

    print(f"cheetahclaws daemon listening on {listen_repr} (pid={os.getpid()})", flush=True)
    if audit_enabled:
        print(f"audit log: {data_dir / 'logs' / 'auth.jsonl'}", flush=True)

    # F-3: take ownership of the monitor scheduler — only after the listener
    # has bound, the pid/discovery files are on disk, and we've printed the
    # ready banner.  Order matters: if a due subscription fires before the
    # daemon is reachable, an LLM/network error in fetch/summarize/deliver
    # would surface in the log before the user sees the listening line, and
    # external clients couldn't yet act on the resulting `monitor_report`
    # SSE event.  ``owned_by_daemon=True`` opts the loop out of the
    # REPL-side step-aside check (otherwise the daemon would defer to its
    # own discovery entry and never run a subscription).
    try:
        from monitor.scheduler import start as _monitor_start
        _monitor_start(config, on_report=None, owned_by_daemon=True)
    except Exception as exc:
        print(f"warning: monitor scheduler did not start: {exc}",
              file=sys.stderr, flush=True)

    # F-5: proactive watcher. Same ordering rationale as F-3 — start
    # AFTER bind + discovery so external clients can subscribe to the
    # `proactive_tick` SSE feed before the first tick lands.
    # ``owned_by_daemon=True`` keeps the loop from deferring to its
    # own discovery file.
    try:
        from . import proactive_scheduler as _proactive_sched
        _proactive_sched.start(owned_by_daemon=True)
    except Exception as exc:
        print(f"warning: proactive scheduler did not start: {exc}",
              file=sys.stderr, flush=True)

    # Graceful-shutdown watcher: when DaemonState.shutdown_event fires
    # (set by system.shutdown RPC or the signal handler below), stop
    # the monitor + proactive schedulers and trigger server.shutdown()
    # from a side thread (the spike's invariant: the same thread as
    # serve_forever cannot call shutdown).
    def _watch_shutdown():
        server.daemon_state.shutdown_event.wait()
        try:
            from monitor.scheduler import stop as _monitor_stop
            _monitor_stop()
        except Exception:
            pass
        try:
            from . import proactive_scheduler as _proactive_stop_ref
            _proactive_stop_ref.stop()
        except Exception:
            pass
        # RFC 0002 F-6/7/8 — stop any running daemon-owned bridges so
        # their HTTP poll threads don't outlive the listener.
        try:
            from . import bridge_supervisor as _bs
            _bs.stop_all(timeout_s=5.0)
        except Exception:
            pass
        threading.Thread(target=server.shutdown, daemon=True).start()
    threading.Thread(target=_watch_shutdown,
                      daemon=True, name="daemon-shutdown-watch").start()

    def _signal_shutdown(_signo, _frame):
        server.daemon_state.shutdown()

    try:
        signal.signal(signal.SIGTERM, _signal_shutdown)
        signal.signal(signal.SIGINT, _signal_shutdown)
        if hasattr(signal, "SIGBREAK"):
            try:
                signal.signal(signal.SIGBREAK, _signal_shutdown)  # type: ignore[arg-type]
            except (ValueError, OSError):
                pass
    except (ValueError, OSError):
        pass

    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        try:
            server.server_close()
        except Exception:
            pass
        if transport == "unix":
            try:
                Path(addr).unlink()
            except FileNotFoundError:
                pass
        try:
            pid_file.unlink()
        except FileNotFoundError:
            pass
        try:
            discovery.clear()
        except OSError:
            pass
    return 0


# ── Helpers ────────────────────────────────────────────────────────────────

def _read_pidfile(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def _write_pidfile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()))


def _lookup_version() -> str:
    try:
        import cheetahclaws as _root
        return getattr(_root, "VERSION", "unknown")
    except Exception:
        return "unknown"


# ── Backward-compat entry: `python -m cc_daemon.cli` ──────────────────────
#
# The Cheetahclaws spike branch (RFC 0001-spike-notes.md §"How to run it")
# documented a subparser CLI with verbs ``serve``, ``status``, ``stop``,
# and ``rotate-token``.  Foundation moves the canonical surface to
# ``cheetahclaws serve`` / ``cheetahclaws daemon <action>``, but anyone
# following the spike notes should still be able to invoke
# ``python -m cc_daemon.cli ...``.
#
# We handle that here by dispatching:
#   * ``serve``  → the same :func:`serve_main` used by ``cheetahclaws serve``
#   * ``status`` / ``stop`` / ``logs`` / ``rotate-token``
#                → :func:`commands.daemon_cmd.dispatch` (the same code path
#                  used by ``cheetahclaws daemon <action>``)
#
# Output / exit codes match the new surface; a few flags from the old
# spike CLI (``--token-path`` / ``--print-token`` on rotate-token) are
# silently accepted as a courtesy and ignored.

_USAGE = (
    "usage: python -m cc_daemon.cli {serve|status|stop|logs|rotate-token} [options]\n"
    "\n"
    "Subcommands:\n"
    "  serve         Run the headless daemon. See `serve --help` for flags.\n"
    "  status        Print pid / transport / address / uptime / ping outcome.\n"
    "  stop          Graceful shutdown via system.shutdown RPC + signal fallback.\n"
    "  logs [-n N]   Tail ~/.cheetahclaws/logs/daemon.log.\n"
    "  rotate-token  Regenerate the TCP bearer token.\n"
)


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        # Top-level help goes to stdout (the usual convention) and exits 0;
        # only the no-args / unknown-subcommand cases go to stderr with code 2.
        if not argv:
            print(_USAGE, file=sys.stderr)
            return 2
        print(_USAGE)
        return 0

    cmd = argv[0]
    if cmd == "serve":
        return serve_main(argv[1:])

    if cmd in ("status", "stop", "logs", "rotate-token"):
        # daemon_cmd.dispatch reads cmd from argv[0]
        from commands.daemon_cmd import dispatch as _daemon_dispatch
        return _daemon_dispatch(argv)

    print(f"unknown subcommand: {cmd}\n", file=sys.stderr)
    print(_USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
