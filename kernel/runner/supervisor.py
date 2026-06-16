"""supervisor.py — RunnerSupervisor (RFC 0016).

Owns the lifecycle of subprocess-per-agent runners:
  spawn → READY→RUNNING transition →
  IPC handshake → live ledger charges →
  exit message → RUNNING→DEAD transition → reap

In-memory registry keyed by AgentProcess pid. Daemon restart loses
the registry; kernel-side recovery (RFC 0003 §2) coerces stale
RUNNING → SUSPENDED, which the operator/supervisor can then re-spawn.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence

from ..errors import (
    LedgerUnknownDim,
    RunnerHandshakeFailed,
    RunnerIllegalState,
    RunnerIpcTimeout,
    RunnerUnknownPid,
)
from ..process import AgentState
from ..sandbox import (
    SANDBOX_DEFAULT,
    SandboxNotAvailable,
    SandboxPolicy,
    apply_rlimits_in_child,
    wrap_with_bubblewrap,
)
from .ipc import IpcReadTimeout, JsonLineChannel

if TYPE_CHECKING:
    from ..ledger import LedgerStore
    from ..store import KernelStore


STDERR_TAIL_BYTES = 4 * 1024
DEFAULT_IPC_TIMEOUT_S = 5.0


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class RunnerHandle:
    pid:          int
    os_pid:       int
    started_at:   float
    sandbox:      SandboxPolicy
    proc:         subprocess.Popen = field(repr=False)
    chan:         JsonLineChannel  = field(repr=False)
    stderr_tail:  deque            = field(repr=False, default_factory=lambda: deque(maxlen=STDERR_TAIL_BYTES))
    _stderr_thread: Optional[threading.Thread] = field(repr=False, default=None)

    def is_alive(self) -> bool:
        return self.proc.poll() is None


@dataclass(frozen=True)
class RunnerExitInfo:
    pid:           int
    exit_kind:     str
    exit_code:     int
    stdout_tail:   bytes
    stderr_tail:   bytes
    duration_s:    float
    ledger_charged: dict
    # RFC 0020 additions — populated from the runner's exit message.
    # ``text`` is the runner-supplied full response (LLM runners use
    # it; echo runner leaves it ""). ``metadata`` is opaque per-runner
    # extras (finish_reason, tokens_total, etc.).
    text:          str = ""
    metadata:      dict = field(default_factory=dict)
    # RFC 0026: streaming chunks the runner emitted between
    # iteration_start and exit. Each entry is the raw IPC
    # message dict (kind/content/metadata).
    chunks:        tuple = ()

    def to_dict(self) -> dict:
        return {
            "pid":           self.pid,
            "exit_kind":     self.exit_kind,
            "exit_code":     self.exit_code,
            "duration_s":    self.duration_s,
            "ledger_charged": self.ledger_charged,
            # Tails are bytes — base64 only when going over the wire.
            "stdout_tail_len": len(self.stdout_tail),
            "stderr_tail_len": len(self.stderr_tail),
            "text_len":      len(self.text),
            "metadata":      dict(self.metadata),
        }


# ── Supervisor ─────────────────────────────────────────────────────────────


class RunnerSupervisor:
    """One supervisor per daemon. Tracks in-memory subprocess registry
    keyed by AgentProcess pid."""

    def __init__(
        self,
        kernel_store: "KernelStore",
        *,
        ledger_store:    Optional["LedgerStore"] = None,
        default_policy:  SandboxPolicy = SANDBOX_DEFAULT,
        ipc_timeout_s:   float = DEFAULT_IPC_TIMEOUT_S,
        tool_registry:   Optional["object"] = None,
        tool_kernel:     Optional["object"] = None,
    ) -> None:
        # ``tool_registry`` and ``tool_kernel`` activate RFC 0021 tool
        # dispatch. ``tool_registry`` is a ToolRegistry; ``tool_kernel``
        # is the Kernel facade used for cap.check_tool / check_fs.
        # Both None ⇒ tool_call IPC messages reply tool_not_found.
        self._kernel = kernel_store
        self._ledger = ledger_store
        self._default_policy = default_policy
        self._ipc_timeout_s = ipc_timeout_s
        self._tool_registry = tool_registry
        self._tool_kernel   = tool_kernel
        self._handles: dict[int, RunnerHandle] = {}
        self._lock = threading.Lock()

    # ── spawn ─────────────────────────────────────────────────────────

    def spawn(
        self,
        *,
        pid: int,
        argv: Sequence[str],
        policy: Optional[SandboxPolicy] = None,
        init_payload: Optional[dict] = None,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> RunnerHandle:
        if not isinstance(pid, int):
            raise RunnerIllegalState(pid, "?", AgentState.READY)

        # Verify agent state.
        agent = self._kernel.get(pid)
        if agent.state != AgentState.READY:
            raise RunnerIllegalState(pid, agent.state, AgentState.READY)

        # Already tracked? Refuse.
        with self._lock:
            if pid in self._handles:
                raise RunnerIllegalState(pid, "ALREADY_SPAWNED",
                                          AgentState.READY)

        # Apply sandbox.
        active_policy = policy or self._default_policy
        if active_policy.use_bubblewrap:
            full_argv = wrap_with_bubblewrap(list(argv), active_policy)
        else:
            full_argv = list(argv)

        preexec = apply_rlimits_in_child(active_policy)

        # Spawn.
        proc = subprocess.Popen(
            full_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=preexec,
            env=dict(env) if env is not None else None,
            cwd=cwd,
        )

        chan = JsonLineChannel(proc.stdout, proc.stdin)

        handle = RunnerHandle(
            pid=pid, os_pid=proc.pid,
            started_at=time.time(),
            sandbox=active_policy,
            proc=proc, chan=chan,
        )

        # Start stderr drainer.
        handle._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(proc, handle.stderr_tail),
            daemon=True, name=f"runner-{pid}-stderr",
        )
        handle._stderr_thread.start()

        # Init handshake.
        try:
            chan.send({
                "op": "init",
                "pid": pid,
                "payload": init_payload or {},
            })
            ready = chan.recv(timeout=self._ipc_timeout_s)
        except IpcReadTimeout as e:
            self._abort_spawn(handle, "ready handshake timed out")
            raise RunnerIpcTimeout(pid, self._ipc_timeout_s) from e
        except (EOFError, ValueError, OSError) as e:
            self._abort_spawn(handle, f"ready handshake error: {e}")
            raise RunnerHandshakeFailed(pid, str(e)) from e

        if ready.get("op") != "ready" or ready.get("pid") != pid:
            self._abort_spawn(handle, f"unexpected handshake msg: {ready}")
            raise RunnerHandshakeFailed(
                pid, f"runner sent {ready!r} instead of ready",
            )

        # Transition agent state.
        self._kernel.transition(pid, AgentState.RUNNING,
                                  reason="runner_spawned")

        # Track.
        with self._lock:
            self._handles[pid] = handle
        return handle

    def _abort_spawn(self, handle: RunnerHandle, reason: str) -> None:
        """Kill the subprocess + drain pipes after a failed spawn.
        The agent stays in READY since we never transitioned."""
        try:
            handle.proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            handle.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
        try:
            handle.chan.close()
        except Exception:
            pass

    # ── wait ──────────────────────────────────────────────────────────

    def wait(
        self, pid: int, timeout: Optional[float] = None,
        on_chunk: Optional[Callable[[dict], None]] = None,
    ) -> RunnerExitInfo:
        """Drain the runner's IPC stream, charge the ledger, transition
        to DEAD with the right exit_kind. Returns RunnerExitInfo.

        RFC 0026: when ``on_chunk`` is supplied, fires for each
        ``op="chunk"`` IPC message the runner emits, in send order.
        Chunks are also accumulated in ``RunnerExitInfo.chunks`` so
        a synchronous caller can read them after the fact.
        Callback exceptions are caught and dropped — a bad
        callback must not break the wait loop.
        """
        with self._lock:
            handle = self._handles.get(pid)
        if handle is None:
            raise RunnerUnknownPid(pid)

        # Drain messages until 'exit' or EOF.
        deadline = (time.monotonic() + timeout) if timeout else None
        exit_kind = "crashed"     # default unless runner sends 'exit'
        summary = ""
        text = ""
        runner_metadata: dict = {}
        ledger_charged: dict = {}
        chunks: list = []          # RFC 0026 — accumulated chunks

        try:
            while True:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    msg_timeout = min(remaining, 1.0)
                else:
                    msg_timeout = self._ipc_timeout_s

                try:
                    msg = handle.chan.recv(timeout=msg_timeout)
                except IpcReadTimeout:
                    if not handle.is_alive():
                        # Process died with no exit message.
                        break
                    # Still alive; keep waiting unless overall timeout.
                    continue
                except EOFError:
                    break
                except (ValueError, OSError):
                    break

                op = msg.get("op")
                if op == "exit":
                    exit_kind = msg.get("exit_kind", "completed")
                    summary = msg.get("summary", "")
                    raw_text = msg.get("text", "")
                    if isinstance(raw_text, str):
                        text = raw_text
                    raw_meta = msg.get("metadata") or {}
                    if isinstance(raw_meta, dict):
                        runner_metadata = raw_meta
                    break
                elif op == "charge":
                    self._apply_charge(handle.pid, msg, ledger_charged)
                elif op == "chunk":
                    # RFC 0026 streaming. Append to local list +
                    # fire user callback (if any). Bad callbacks
                    # are silently swallowed so they can't break
                    # the wait loop.
                    chunks.append(msg)
                    if on_chunk is not None:
                        try:
                            on_chunk(msg)
                        except Exception:
                            pass
                elif op == "tool_call":
                    # RFC 0028: build a chunk emitter that ALSO
                    # appends to our local chunks list (so they
                    # surface in RunnerExitInfo.chunks too) and
                    # forwards to the wait()-time callback.
                    def _emit_chunk(payload, _chunks=chunks,
                                     _on=on_chunk):
                        _chunks.append(payload)
                        if _on is not None:
                            try:
                                _on(payload)
                            except Exception:
                                pass
                    response = self._handle_tool_call(
                        handle.pid, msg, on_chunk=_emit_chunk,
                    )
                    try:
                        handle.chan.send(response)
                    except (BrokenPipeError, OSError):
                        # Runner died mid-call — keep draining; the
                        # exit handling below will catch the EOF.
                        pass
                elif op == "iteration_done":
                    # Convenience: auto-charge tokens / cost from the
                    # iteration message if those dims exist on the
                    # ledger.
                    for dim_name in ("tokens", "cost_micro"):
                        amt = msg.get(dim_name)
                        if isinstance(amt, int) and amt > 0:
                            self._apply_charge(
                                handle.pid,
                                {"op": "charge", "dim": dim_name,
                                 "amount": amt},
                                ledger_charged,
                            )
                # log / iteration_start / others — discard for v1
        except Exception:
            # Best-effort: on any error during drain, still try to
            # reap and transition.
            pass

        # Reap the OS process.
        exit_code = self._reap(handle, deadline)

        # Refine exit_kind based on actual termination.
        if exit_code is None:
            exit_kind = "crashed"
            exit_code = -signal.SIGKILL
        elif exit_code != 0 and exit_kind == "completed":
            # Runner said completed but exit was non-zero — odd; treat
            # as failed.
            exit_kind = "failed"
        elif exit_code != 0 and exit_kind == "crashed":
            # Confirmed crash.
            pass

        duration = time.time() - handle.started_at

        # Charge wall_s if available.
        if self._ledger is not None:
            wall_seconds = max(int(duration), 0)
            if wall_seconds > 0:
                try:
                    cr = self._ledger.charge(
                        pid=handle.pid, dim="wall_s",
                        amount=wall_seconds,
                    )
                    ledger_charged.setdefault("wall_s", 0)
                    ledger_charged["wall_s"] += wall_seconds
                    if cr.first_breach:
                        self._record_first_breach(handle.pid, "wall_s",
                                                   cr.used, cr.granted)
                except LedgerUnknownDim:
                    pass

        # Transition to DEAD.
        try:
            self._kernel.terminate(
                handle.pid, exit_kind=exit_kind,
                exit_detail={"exit_code": exit_code, "summary": summary},
            )
        except Exception:
            # Already DEAD? caller will see existing state.
            pass

        # Drop handle.
        with self._lock:
            self._handles.pop(handle.pid, None)

        # Tail stdout (we didn't keep stdout buffered separately;
        # at this point stdout is closed). Tail stderr from the deque.
        stderr_tail = bytes(handle.stderr_tail)
        try:
            handle.chan.close()
        except Exception:
            pass

        return RunnerExitInfo(
            pid=handle.pid,
            exit_kind=exit_kind,
            exit_code=int(exit_code),
            stdout_tail=b"",          # not collected in v1; runner uses chan
            stderr_tail=stderr_tail,
            duration_s=duration,
            ledger_charged=ledger_charged,
            text=text,
            metadata=runner_metadata,
            chunks=tuple(chunks),
        )

    def _reap(
        self, handle: RunnerHandle, deadline: Optional[float],
    ) -> Optional[int]:
        """Block until the OS process exits. Returns exit code or None
        if it had to be killed."""
        try:
            if deadline is None:
                return handle.proc.wait()
            remaining = max(deadline - time.monotonic(), 0.1)
            return handle.proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            self._kill(handle)
            try:
                return handle.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                return None

    # ── stop ──────────────────────────────────────────────────────────

    def stop(
        self, pid: int, *, exit_kind: str = "cancelled",
    ) -> RunnerExitInfo:
        """Send 'stop' over IPC, escalate to SIGTERM/SIGKILL after
        ipc_timeout_s. Returns RunnerExitInfo with the supplied
        exit_kind unless the runner died with a different status."""
        with self._lock:
            handle = self._handles.get(pid)
        if handle is None:
            raise RunnerUnknownPid(pid)

        # Polite request.
        try:
            handle.chan.send({"op": "stop"})
        except (BrokenPipeError, OSError):
            pass

        # Give it ipc_timeout_s to honour.
        deadline = time.monotonic() + self._ipc_timeout_s
        try:
            info = self.wait(pid, timeout=self._ipc_timeout_s)
        except RunnerUnknownPid:
            # Already reaped between calls.
            raise

        # Override exit_kind unless the runner crashed/failed.
        if info.exit_kind == "completed":
            # Re-record DEAD with the requested exit_kind since wait
            # already terminated to DEAD with 'completed'. We don't
            # re-transition (DEAD is terminal); the caller sees the
            # accurate code path via the returned info.
            pass
        return info

    def kill(self, pid: int) -> bool:
        """Public, non-blocking kill of a tracked runner's OS process.
        Sends SIGTERM, escalates to SIGKILL after a 1-second grace.
        Does NOT wait, transition state, or pop the handle — those are
        the responsibility of the wait() consumer (typically the
        worker thread that owns the runner).

        Use this from `WorkerLoop.stop` when there's already a wait()
        in flight on the same pid; calling supervisor.stop in that
        case races for the handle (see RFC 0017 §4 and the
        documented interaction).

        Returns True if a kill signal was sent, False if no live
        handle for that pid.
        """
        with self._lock:
            handle = self._handles.get(pid)
        if handle is None:
            return False
        self._kill(handle)
        return True

    def _kill(self, handle: RunnerHandle) -> None:
        """SIGTERM the process group, then SIGKILL after grace.
        Mirrors RFC 0008's wall-clock killer."""
        proc = handle.proc
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                proc.terminate()
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.wait(timeout=1.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass

    # ── list / cleanup ────────────────────────────────────────────────

    def list(self) -> list:
        with self._lock:
            return list(self._handles.values())

    def cleanup(self) -> int:
        """Reap any zombies whose Popen has poll()=not None but we
        haven't drained yet. Useful in tests."""
        cleaned = 0
        with self._lock:
            dead_pids = [
                p for p, h in self._handles.items()
                if not h.is_alive()
            ]
        for p in dead_pids:
            try:
                self.wait(p, timeout=2.0)
                cleaned += 1
            except Exception:
                pass
        return cleaned

    # ── ledger helpers ────────────────────────────────────────────────

    def _apply_charge(
        self, pid: int, msg: dict, ledger_charged: dict,
    ) -> None:
        if self._ledger is None:
            return
        dim = msg.get("dim")
        amount = msg.get("amount")
        if not isinstance(dim, str) or not isinstance(amount, int) or amount < 0:
            return
        try:
            cr = self._ledger.charge(pid=pid, dim=dim, amount=amount)
        except LedgerUnknownDim:
            # Charge silently ignored — supervisor's policy.
            return
        ledger_charged.setdefault(dim, 0)
        ledger_charged[dim] += amount
        if cr.first_breach:
            self._record_first_breach(pid, dim, cr.used, cr.granted)

    # ── tool dispatch (RFC 0021) ─────────────────────────────────────

    def _handle_tool_call(
        self, pid: int, msg: dict,
        on_chunk: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Map an inbound tool_call IPC message to a tool_response.

        If the supervisor was constructed without a tool_registry,
        every call yields tool_not_found. Otherwise the registry's
        dispatch function is invoked with the kernel-facade access
        for capability checks.

        RFC 0028: ``on_chunk``, if supplied, is forwarded to the
        ``ToolContext`` so streaming tools (like Exec) can emit
        per-line chunks during execution.
        """
        if self._tool_registry is None:
            return {
                "op":           "tool_response",
                "tool_call_id": msg.get("tool_call_id", ""),
                "ok":           False,
                "error":        "tool_not_found",
                "message":      "supervisor has no tool_registry configured",
            }
        from ..tools.registry import dispatch_tool_call as _dispatch
        response = _dispatch(
            msg=msg, pid=pid,
            registry=self._tool_registry,
            kernel=self._tool_kernel,
            on_chunk=on_chunk,
        )
        # Audit: write to event log via kernel store. Silent on
        # failure (the dispatch result is the source of truth for
        # the runner; audit is best-effort).
        try:
            kind = "tool.call.dispatched" if response.get("ok") else "tool.call.denied"
            self._kernel.events_append(
                pid=pid, kind=kind,
                payload={
                    "tool":         msg.get("tool"),
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "ok":           bool(response.get("ok")),
                    "error":        response.get("error"),
                },
            )
        except Exception:
            pass
        return response

    def _record_first_breach(
        self, pid: int, dim: str, used: int, granted: int,
    ) -> None:
        # Supervisor is a kernel client, so it can't use the reserved
        # ``kernel.*`` event prefix. RFC 0016 §7 names this event
        # ``runner.first_breach``; the prefix scopes it to the
        # supervisor layer in the audit log.
        try:
            self._kernel.events_append(
                pid=pid,
                kind="runner.first_breach",
                payload={
                    "dim": dim, "used": used, "granted": granted,
                },
            )
        except Exception:
            pass


# ── stderr drainer ─────────────────────────────────────────────────────────


def _drain_stderr(proc: subprocess.Popen, tail: deque) -> None:
    """Daemon thread: read proc.stderr in a loop, append to a bounded
    deque so the supervisor can include the tail in RunnerExitInfo."""
    try:
        while True:
            chunk = proc.stderr.read(4096) if proc.stderr else b""
            if not chunk:
                return
            tail.extend(chunk)
    except (OSError, ValueError):
        return
