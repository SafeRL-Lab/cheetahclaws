"""
health.py — Lightweight HTTP health-check server for CheetahClaws.

Disabled by default.  Enable in config:
  /config health_check_port=8765

Endpoints:
  GET /healthz   → 200 JSON (always healthy while server is up)
  GET /readyz    → 200 JSON (healthy) or 503 (circuit breakers open)
  GET /metrics   → 200 JSON (sessions, token usage, circuit states)

Config keys:
  health_check_port : int  — TCP port to listen on (null = disabled)

The HTTP server runs in a daemon thread so it never blocks the REPL.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


_start_time: float = time.monotonic()
_server_thread: threading.Thread | None = None
# Reference to the config dict passed at startup (for live model reads)
_config: dict = {}


# ── Request handler ───────────────────────────────────────────────────────

class _HealthHandler(BaseHTTPRequestHandler):

    def log_message(self, *_args):
        pass  # silence access logs; structured logging handles this

    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path == "/healthz":
            self._send_json(200, self._healthz())
        elif path == "/readyz":
            body = self._readyz()
            code = 200 if body["status"] == "ok" else 503
            self._send_json(code, body)
        elif path == "/metrics":
            self._send_json(200, self._metrics())
        else:
            self._send_json(404, {"error": "not found"})

    # ── Payload builders ──────────────────────────────────────────────────
    # The instance methods below delegate to module-level functions so that
    # other listeners (e.g. daemon/server.py) can reuse the same payload
    # logic without booting a second http.server.

    def _healthz(self) -> dict:
        return healthz_payload(_config)

    def _readyz(self) -> dict:
        return readyz_payload(_config)

    def _metrics(self) -> dict:
        return metrics_payload(_config)


# ── Module-level payload helpers ──────────────────────────────────────────

def uptime_seconds() -> float:
    return round(time.monotonic() - _start_time, 1)


def _circuit_states() -> dict[str, str]:
    try:
        from cheetahclaws.circuit_breaker import _registry as _cb_reg, _registry_lock as _cb_lock
        with _cb_lock:
            return {p: b.state.value for p, b in _cb_reg.items()}
    except Exception:
        return {}


def _active_sessions() -> int:
    try:
        from cheetahclaws.runtime import _registry as _rt_reg, _registry_lock as _rt_lock
        with _rt_lock:
            return len(_rt_reg)
    except Exception:
        return 0


def healthz_payload(config: dict | None = None) -> dict:
    cfg = config if config is not None else _config
    return {
        "status":          "ok",
        "uptime_s":        uptime_seconds(),
        "model":           (cfg or {}).get("model", ""),
        "active_sessions": _active_sessions(),
    }


def readyz_payload(config: dict | None = None) -> dict:
    cfg = config if config is not None else _config
    circuits = _circuit_states()
    open_circuits = [p for p, s in circuits.items() if s == "open"]
    status = "degraded" if open_circuits else "ok"
    body: dict = {
        "status":   status,
        "uptime_s": uptime_seconds(),
        "circuits": circuits,
    }
    if open_circuits:
        body["open_circuits"] = open_circuits
    body["model"] = (cfg or {}).get("model", "")
    return body


def metrics_payload(config: dict | None = None) -> dict:
    cfg = config if config is not None else _config
    circuits = _circuit_states()
    # Today's quota usage (read from file — best effort)
    daily_tokens = daily_cost = 0
    try:
        from cheetahclaws.quota import _load_daily, _lock as _q_lock
        with _q_lock:
            daily_tokens, daily_cost = _load_daily()
    except Exception:
        pass
    return {
        "uptime_s":        uptime_seconds(),
        "model":           (cfg or {}).get("model", ""),
        "active_sessions": _active_sessions(),
        "circuits":        circuits,
        "daily_tokens":    daily_tokens,
        "daily_cost_usd":  round(daily_cost, 6),
    }


def payload_for(path: str, config: dict | None = None) -> dict:
    """Dispatch helper: route an HTTP path to its payload builder."""
    if path == "/healthz":
        return healthz_payload(config)
    if path == "/readyz":
        return readyz_payload(config)
    if path == "/metrics":
        return metrics_payload(config)
    return {}


def install_config(config: dict) -> None:
    """Pin the module-level config used by start_health_server / server.

    Idempotent — daemon/cli.py calls this when starting the daemon listener
    so the default-arg payload helpers see the right model/version data.
    """
    global _config
    _config = config


# ── Server lifecycle ──────────────────────────────────────────────────────

def start_health_server(port: int, config: dict) -> None:
    """Start the health-check HTTP server in a daemon thread.

    Safe to call multiple times — a second call while the server is already
    running is silently ignored.
    """
    global _server_thread, _config, _start_time

    if _server_thread and _server_thread.is_alive():
        return   # already running

    _config     = config
    _start_time = time.monotonic()

    server = HTTPServer(("", port), _HealthHandler)

    def _serve():
        from cheetahclaws import logging_utils as _log
        _log.info("health_server_listening", port=port)
        try:
            server.serve_forever()
        except Exception as exc:
            _log.error("health_server_error", error=str(exc)[:200])

    _server_thread = threading.Thread(target=_serve, daemon=True,
                                      name="health-check-server")
    _server_thread.start()
