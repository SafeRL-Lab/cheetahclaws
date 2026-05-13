"""agent_methods.py — `agent.*` JSON-RPC methods (RFC 0002 F-4).

Thin wrappers over :mod:`cc_daemon.runner_supervisor` so external clients
(REPL `/agent` command, future Web UI, third-party tools) can manage
agent runners through the daemon's RPC channel instead of importing the
supervisor directly.

Exposed methods:

    agent.start(name, template, args="", interval=2.0, auto_approve=True,
                restart_policy="none"|"on-crash", max_restarts=0,
                backoff_base_s=1.0, backoff_cap_s=60.0,
                backoff_jitter_s=0.5)
        Spawn a runner subprocess and return its handle dict.  The five
        restart_* params (RFC 0002 F-4 #3) control whether the supervisor
        auto-restarts a crashed runner: default ``restart_policy="none"``
        keeps today's behaviour (crashed handles stay crashed).

    agent.stop(name, timeout_s=5.0)
        Stop a runner. Returns {"name", "stopped": bool}.  Also cancels
        any pending restart timer for this lineage.

    agent.list()
        Return all currently-tracked runners.

    agent.status(name)
        Return one runner's status dict. 404-equivalent (returns
        {"name", "found": False}) if no runner with that name.

F-4 keeps these methods open to any authenticated caller — same
single-user threat model as F-3's monitor.* methods. Per-method
authorisation arrives with the originator routing in a follow-up.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .rpc import RpcRegistry

if TYPE_CHECKING:
    from .server import DaemonState


def _handle_to_dict(handle) -> dict:
    """Serialise a RunnerHandle for the wire. Drops process / channel
    references (not JSON-serialisable) and exposes only the fields a
    caller can act on."""
    policy = handle.restart_policy
    return {
        "name":            handle.name,
        "run_id":          handle.run_id,
        "pid":             handle.pid,
        "status":          handle.status,
        "iteration":       handle.iteration,
        "started_at":      handle.started_at,
        "template":        handle.template_name,
        "args":            handle.args,
        "auto_approve":    handle.auto_approve,
        "originator":      handle.originator,
        "alive":           handle.is_alive(),
        "error":           handle.error,
        # RFC 0002 F-4 #3 — surface restart status so callers can see
        # whether a lineage is auto-restarting, how many times, and
        # what the policy was.
        "restart_count":   handle.restart_count,
        "restart_policy":  {
            "mode":             policy.mode,
            "max_restarts":     policy.max_restarts,
            "backoff_base_s":   policy.backoff_base_s,
            "backoff_cap_s":    policy.backoff_cap_s,
            "backoff_jitter_s": policy.backoff_jitter_s,
        },
    }


def register(registry: RpcRegistry, daemon_state: "DaemonState") -> None:

    def agent_start(params: dict, ctx) -> dict:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError("agent.start requires non-empty 'name'")
        template = params.get("template")
        if not isinstance(template, str) or not template:
            raise TypeError("agent.start requires non-empty 'template'")
        args = str(params.get("args", "") or "")
        try:
            interval = float(params.get("interval", 2.0))
        except (TypeError, ValueError) as e:
            raise TypeError(f"agent.start: 'interval' must be numeric: {e}")
        auto_approve = bool(params.get("auto_approve", True))

        # Pull config from the daemon's own loaded config; the runner
        # subprocess inherits a JSON-safe subset (see
        # runner_supervisor._strip_unserialisable).
        config = dict(daemon_state.config or {})

        # RFC 0002 F-4 #3 — restart policy. The originator picks the
        # policy at start time; bad values raise TypeError which the RPC
        # frame converts into a JSON-RPC error (-32602 invalid params).
        from . import runner_supervisor as rs
        try:
            policy = rs.RestartPolicy.from_params(params)
        except TypeError:
            # Re-raise as-is so the RPC layer reports a 400-ish error
            # rather than a generic 500.
            raise

        # RFC 0002 F-4 #1: the caller's client_id is stamped on every
        # PermissionRequest minted by this runner, so only the same client
        # can answer via `permission.answer`. Pass the daemon's store so
        # the supervisor can route permission_request IPC through it.
        handle = rs.start(
            name=name, template_name=template, args=args,
            config=config, interval=interval, auto_approve=auto_approve,
            originator=getattr(ctx, "client_id", "") or "",
            permission_store=getattr(daemon_state, "permissions", None),
            restart_policy=policy,
        )
        return _handle_to_dict(handle)

    def agent_stop(params: dict, _ctx) -> dict:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError("agent.stop requires non-empty 'name'")
        try:
            timeout_s = float(params.get("timeout_s", 5.0))
        except (TypeError, ValueError) as e:
            raise TypeError(f"agent.stop: 'timeout_s' must be numeric: {e}")
        from . import runner_supervisor as rs
        return {"name": name, "stopped": rs.stop(name, timeout_s=timeout_s)}

    def agent_list(_params: dict, _ctx) -> dict:
        from . import runner_supervisor as rs
        return {"runners": [_handle_to_dict(h) for h in rs.list_all()]}

    def agent_status(params: dict, _ctx) -> dict:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError("agent.status requires non-empty 'name'")
        from . import runner_supervisor as rs
        h = rs.get(name)
        if h is None:
            return {"name": name, "found": False}
        d = _handle_to_dict(h)
        d["found"] = True
        return d

    def agent_resume(params: dict, _ctx) -> dict:
        """RFC 0002 §F-9 — bump the daemon-level budget AND/OR wake a
        paused runner.

        Params:
          ``budget_overrides`` — dict, any subset of:
            ``session_token_budget`` / ``session_cost_budget``
            / ``daily_token_budget``  / ``daily_cost_budget``
            Values are coerced to int (tokens) or float (cost). A value
            of ``null`` resets that budget to unlimited.
          ``name`` (optional) — runner name. When present, the
            supervisor sends a ``resume`` IPC frame to that runner so
            ``_on_quota_exceeded`` unblocks. The runner re-checks the
            quota against the *updated* daemon_state.config; if the
            ceiling is still too low it pauses again.

        Returns ``{"budgets": {…}, "resumed": bool|null}`` — ``resumed``
        is ``null`` when no ``name`` was supplied (no per-runner action
        taken), True when the IPC was delivered, False when the named
        runner isn't tracked or its channel is dead.
        """
        overrides = params.get("budget_overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise TypeError(
                "agent.resume: 'budget_overrides' must be an object")
        allowed = {
            "session_token_budget": int,
            "session_cost_budget":  float,
            "daily_token_budget":   int,
            "daily_cost_budget":    float,
        }
        for k, v in overrides.items():
            if k not in allowed:
                raise TypeError(
                    f"agent.resume: unknown budget key {k!r}; allowed: "
                    f"{sorted(allowed)}")
            if v is None:
                daemon_state.config[k] = None  # unlimited
                continue
            try:
                daemon_state.config[k] = allowed[k](v)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"agent.resume: {k}={v!r} not coercible to "
                    f"{allowed[k].__name__}: {e}")

        # Optional per-runner wake-up.
        resumed: Optional[bool] = None
        name = params.get("name")
        if name is not None:
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "agent.resume: 'name' must be a non-empty string")
            from . import runner_supervisor as rs
            resumed = rs.resume(name)
        return {
            "budgets": {k: daemon_state.config.get(k) for k in allowed},
            "resumed": resumed,
        }

    registry.register("agent.start",  agent_start)
    registry.register("agent.stop",   agent_stop)
    registry.register("agent.list",   agent_list)
    registry.register("agent.status", agent_status)
    registry.register("agent.resume", agent_resume)
