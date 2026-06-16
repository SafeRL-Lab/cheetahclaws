"""kernel.runner — subprocess agent runner substrate (RFC 0016).

The bridge between kernel primitives and OS processes. Spawns
sandboxed subprocesses, drives the agent state machine through their
lifecycle, charges the ledger, and reports through the kernel's audit
log.

This package is purely additive: existing ``agent_runner.py`` and
related modules are not modified. The runner here is a parallel,
kernel-managed surface that future code (and a future RFC migrating
``agent_runner.py``) can build on.

Public surface:

    RunnerSupervisor   — Python API for spawn / wait / stop / list
    RunnerHandle       — frozen dataclass for one live subprocess
    RunnerExitInfo     — frozen dataclass for completed subprocess
    JsonLineChannel    — IPC primitive (test-friendly)

    RunnerIllegalState, RunnerHandshakeFailed,
    RunnerUnknownPid, RunnerIpcTimeout
"""
from __future__ import annotations

from ..errors import (
    RunnerHandshakeFailed,
    RunnerIllegalState,
    RunnerIpcTimeout,
    RunnerUnknownPid,
)
from .ipc import JsonLineChannel, IpcReadTimeout
from .supervisor import (
    RunnerExitInfo,
    RunnerHandle,
    RunnerSupervisor,
)

__all__ = [
    "RunnerSupervisor",
    "RunnerHandle",
    "RunnerExitInfo",
    "JsonLineChannel",
    "IpcReadTimeout",
    "RunnerIllegalState",
    "RunnerHandshakeFailed",
    "RunnerUnknownPid",
    "RunnerIpcTimeout",
]
