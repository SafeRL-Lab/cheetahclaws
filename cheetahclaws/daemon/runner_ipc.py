"""runner_ipc.py — IPC channel re-exports for the agent-runner supervisor (RFC 0002 F-4).

The supervisor (daemon/runner_supervisor.py) and the runner entry point
(`python -m agent_runner --pipe`) speak newline-delimited JSON over a pair
of pipes. The wire format and stream wrapper already exist for the kernel
LLM runner — we re-export them here so the daemon side doesn't grow a
duplicate implementation and stay-in-sync becomes automatic.

Message types on this channel (see runner_supervisor.py docstring for the
full state machine):

  supervisor → runner:
    {"op": "init",   "payload": {template, args, config, name, auto_approve}}
    {"op": "permission_response", "request_id": str, "granted": bool}
    {"op": "stop"}                              # graceful stop

  runner → supervisor:
    {"op": "ready"}                             # handshake reply to init
    {"op": "iteration_start", "iteration": int}
    {"op": "iteration_done",  "iteration": int, "status": str,
                              "duration_s": float, "summary": str,
                              "tokens_in": int, "tokens_out": int}
    {"op": "permission_request", "request_id": str, "description": str,
                                  "tool": str, "input": dict}
    {"op": "log", "level": str, "msg": str}
    {"op": "notify", "text": str}               # what send_fn would have sent
    {"op": "exit", "reason": str, "iterations": int}
"""
from __future__ import annotations

from cheetahclaws.kernel.runner.ipc import IpcReadTimeout, JsonLineChannel

__all__ = ["IpcReadTimeout", "JsonLineChannel"]
