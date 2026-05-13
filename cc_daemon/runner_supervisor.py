"""runner_supervisor.py — agent-runner subprocess supervision (RFC 0002 F-4).

Owns the lifecycle of one or more `python -m agent_runner --pipe` subprocesses
on behalf of the daemon. Each AgentRunner that today lives in a Python thread
becomes its own OS process so that:

  * a leak / hang / OOM in one runner doesn't take down the daemon,
  * `kill -9 <runner_pid>` is observable as an `agent_runner_crash` event,
  * `agent.stop` RPC delivers a graceful stop within 5 s.

Scope of this initial cut (F-4 skeleton, RFC 0002):

  * POSIX only — `subprocess.Popen` with stdin/stdout pipes, `JsonLineChannel`
    framing. Windows fallback is out of scope; callers must check `enabled()`.
  * Iteration log written via stdout dump (the runner side emits one
    `iteration_done` IPC message per iteration; supervisor persists to
    ``~/.cheetahclaws/agents/<name>/log.jsonl`` for parity with the in-thread
    path). SQLite persistence to the `agent_iterations` table is deferred.
  * Permission flow: when the runner sends ``permission_request``, the
    supervisor's reader thread auto-approves (matches today's
    ``auto_approve=True`` REPL default). Routing to a real PermissionStore
    is deferred to a follow-up.

Acceptance (RFC 0002 §F-4 "Acceptance"):

  ✓ Runner crash (kill -9): supervisor detects exit via proc.poll(), emits
    ``agent_runner_crash`` event with stderr tail.
  ✓ Runner OOM: same code path as kill -9 (process exits with non-zero
    code), supervisor stays up.
  ✓ Runner subprocess stops within 5 s of stop(): graceful "stop" IPC →
    SIGTERM at 2 s → SIGKILL at 5 s.
  ⚠ Iteration-log parity: jsonl format matches today's
    AgentRunner._persist_record. SQLite agent_iterations population is
    follow-up (see schema.py line 74-85 comment "populated in F-4").
"""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .runner_ipc import IpcReadTimeout, JsonLineChannel

# Lazy import — events module pulls in SQLite; tests that exercise the
# supervisor in isolation shouldn't trigger schema init.
def _get_event_bus():
    try:
        from . import events
        return events.get_bus()
    except Exception:
        return None


def _iso_now() -> str:
    """ISO 8601 UTC timestamp with microsecond precision and Z suffix.
    Same shape as cc_daemon.events._epoch_to_iso so the two columns sort
    consistently when joined on time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


STDERR_TAIL_BYTES = 4 * 1024
HANDSHAKE_TIMEOUT_S = 5.0
GRACEFUL_STOP_TIMEOUT_S = 2.0      # IPC "stop" → SIGTERM after this
SIGTERM_GRACE_S = 3.0              # SIGTERM → SIGKILL after this
# Total upper bound on stop(): HANDSHAKE_TIMEOUT_S irrelevant here;
# 2 + 3 = 5 s matches the F-4 acceptance criterion.

_LOG_DIR = Path.home() / ".cheetahclaws" / "agents"


# ── Restart policy (RFC 0002 F-4 #3) ──────────────────────────────────────


@dataclass(frozen=True)
class RestartPolicy:
    """How the supervisor should react to a crashed runner.

    The decision to restart lives with the *originator* that started the
    runner (RFC 0002 §F-4 #3 — "lives with the originator"). The originator
    picks a policy at ``agent.start`` time; the supervisor then honours it
    autonomously until ``max_restarts`` is exhausted.

    Only ``mode='on-crash'`` triggers a respawn. A graceful ``stop()`` is
    never followed by a restart.

    Backoff is ``min(backoff_base_s * 2**restart_count, backoff_cap_s)``
    plus optional jitter (``±backoff_jitter_s`` uniform), so a pathological
    boot-loop doesn't pin one core at 100 % CPU.
    """
    mode:              str   = "none"      # "none" | "on-crash"
    max_restarts:      int   = 0           # 0 disables restart even if mode='on-crash'
    backoff_base_s:    float = 1.0
    backoff_cap_s:     float = 60.0
    backoff_jitter_s:  float = 0.5

    @classmethod
    def disabled(cls) -> "RestartPolicy":
        return cls(mode="none", max_restarts=0)

    @classmethod
    def from_params(cls, params: dict) -> "RestartPolicy":
        """Construct from a dict supplied to ``agent.start`` over RPC.
        Unknown keys are ignored; missing keys take the default. Bad
        types raise TypeError so the RPC layer can surface a 400."""
        mode = str(params.get("restart_policy", "none") or "none").lower()
        if mode not in {"none", "on-crash"}:
            raise TypeError(
                f"restart_policy must be 'none' or 'on-crash', got {mode!r}")
        try:
            max_restarts = int(params.get("max_restarts", 0) or 0)
        except (TypeError, ValueError) as e:
            raise TypeError(f"max_restarts must be int: {e}")
        if max_restarts < 0:
            raise TypeError(f"max_restarts must be ≥ 0, got {max_restarts}")
        try:
            base = float(params.get("backoff_base_s", 1.0))
            cap  = float(params.get("backoff_cap_s",  60.0))
            jitter = float(params.get("backoff_jitter_s", 0.5))
        except (TypeError, ValueError) as e:
            raise TypeError(f"backoff fields must be numeric: {e}")
        if base < 0 or cap < 0 or jitter < 0:
            raise TypeError("backoff fields must be ≥ 0")
        if cap < base:
            # Catch the obvious config mistake before it traps the user
            # in an "exhausted on attempt 1 because cap < base" loop.
            raise TypeError(
                f"backoff_cap_s ({cap}) must be ≥ backoff_base_s ({base})")
        return cls(mode=mode, max_restarts=max_restarts,
                   backoff_base_s=base, backoff_cap_s=cap,
                   backoff_jitter_s=jitter)

    def next_delay(self, restart_count: int) -> Optional[float]:
        """How long to wait before the next restart attempt.

        Returns None when:
          * mode != 'on-crash' (restarts disabled), or
          * ``restart_count`` already at or above ``max_restarts``.

        Pure function — no I/O, no clock reads — so tests can drive the
        full decision matrix without spawning subprocesses or sleeping.
        """
        if self.mode != "on-crash":
            return None
        if restart_count >= self.max_restarts:
            return None
        delay = self.backoff_base_s * (2 ** max(0, restart_count))
        delay = min(delay, self.backoff_cap_s)
        if self.backoff_jitter_s > 0:
            # Symmetric jitter, clipped at 0 so we never schedule into the past.
            delay = max(0.0, delay + random.uniform(
                -self.backoff_jitter_s, self.backoff_jitter_s))
        return delay


# Factory hook for tests: a `Callable[[dict], RunnerHandle]` that respawns
# a runner from the previous handle's start_kwargs. The default points at
# ``start`` further down the file; tests inject a stub that records the
# call and returns a fake handle without actually forking a subprocess.
_RESTART_SPAWNER: Optional[Callable[..., "RunnerHandle"]] = None


# ── Feature flag ──────────────────────────────────────────────────────────


def enabled() -> bool:
    """Return True iff F-4 subprocess-per-runner is active.

    Sources (any one truthy is enough):
      * ``CHEETAHCLAWS_ENABLE_F4`` env var
      * config key ``agent_runner_subprocess`` (callers pass via start())

    Defaults to False. Windows is unsupported regardless.
    """
    if sys.platform.startswith("win"):
        return False
    flag = os.environ.get("CHEETAHCLAWS_ENABLE_F4", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


# ── Handle / status ───────────────────────────────────────────────────────


@dataclass
class RunnerHandle:
    name:        str
    run_id:      str
    pid:         int
    started_at:  float
    proc:        subprocess.Popen = field(repr=False)
    chan:        JsonLineChannel  = field(repr=False)
    stderr_tail: deque            = field(repr=False,
                                          default_factory=lambda: deque(maxlen=STDERR_TAIL_BYTES))
    _reader:        Optional[threading.Thread] = field(repr=False, default=None)
    _stderr_reader: Optional[threading.Thread] = field(repr=False, default=None)
    iteration:   int   = 0
    status:      str   = "starting"   # starting | running | stopping | stopped | crashed
    error:       str   = ""
    # RFC 0002 F-4 — kept for the agent.list RPC and SQLite agent_runs row.
    template_name: str = ""
    args:          str = ""
    auto_approve:  bool = True
    # RFC 0002 F-4 #1 — permission routing.
    # When `auto_approve` is False AND `permission_store` is set, the reader
    # loop routes inbound `permission_request` IPC through the store so the
    # originator (the client_id that called agent.start) can answer via
    # `permission.answer`. When either is missing, the reader falls back to
    # the today's auto-approve fast path (consistent with the in-process
    # AgentRunner's default).
    originator:       str = ""
    permission_store: Optional["object"] = field(default=None, repr=False)
    # RFC 0002 F-4 #3 — restart policy.
    # ``restart_policy`` decides what the reader's `finally` block does
    # when the process exited non-zero (and not via graceful stop).
    # ``restart_count`` is the running tally for *this lineage* — every
    # successor handle inherits + 1, so the policy's max_restarts caps
    # the whole sequence, not each respawn in isolation.
    # ``_start_kwargs`` captures the exact arguments the supervisor used
    # to spawn this handle so a follow-up restart is byte-for-byte
    # equivalent. We do not snapshot the live config dict — the daemon's
    # config is mutable, and a restart that picks up the *current* values
    # is the behaviour callers expect.
    restart_policy:   "RestartPolicy" = field(
        default_factory=lambda: RestartPolicy.disabled())
    restart_count:    int  = 0
    _start_kwargs:    dict = field(default_factory=dict, repr=False)
    _restart_timer:   Optional[threading.Timer] = field(default=None, repr=False)
    # Set to True when the reader's `finally` has already scheduled (or
    # decided not to schedule) a restart. Prevents a race where stop()
    # cancels the timer slot just as the reader is about to write to it.
    _restart_decided: bool = False

    def is_alive(self) -> bool:
        return self.proc.poll() is None


# ── Registry ──────────────────────────────────────────────────────────────


_handles: dict[str, RunnerHandle] = {}
_handles_lock = threading.Lock()


def get(name: str) -> Optional[RunnerHandle]:
    with _handles_lock:
        h = _handles.get(name)
        if h and not h.is_alive() and h.status not in {"crashed", "stopped"}:
            # Process died before we noticed — reflect that.
            h.status = "crashed"
        return h


def list_all() -> list[RunnerHandle]:
    with _handles_lock:
        return list(_handles.values())


def _register(handle: RunnerHandle) -> None:
    with _handles_lock:
        # Stop any prior runner with the same name first.
        old = _handles.get(handle.name)
        if old and old.is_alive():
            # Caller is expected to have called stop() already; this is
            # just a safety net.
            try:
                old.proc.terminate()
            except ProcessLookupError:
                pass
        _handles[handle.name] = handle


def _unregister(name: str, expected: Optional["RunnerHandle"] = None) -> None:
    """Remove ``name`` from the registry.

    With ``expected=None`` (today's callers that don't care about the
    lineage), the slot is popped unconditionally. When ``expected`` is
    passed, the slot is popped *only if* the currently-registered
    handle is the same object — this closes a F-4 #3 race where a
    Timer-fired ``_do_restart`` spawns a new handle while a concurrent
    ``stop()`` is still cleaning up the previous one. Without the
    identity check the stop's terminal ``_unregister`` would silently
    delete the freshly-spawned successor and leak its subprocess.
    """
    with _handles_lock:
        current = _handles.get(name)
        if expected is not None and current is not expected:
            return
        _handles.pop(name, None)


# ── Spawn ─────────────────────────────────────────────────────────────────


def start(
    name: str,
    template_name: str,
    args: str,
    config: dict,
    *,
    interval: float = 2.0,
    auto_approve: bool = True,
    python: str = sys.executable,
    originator: str = "",
    permission_store: Optional["object"] = None,
    restart_policy: Optional[RestartPolicy] = None,
    _restart_count_carry: int = 0,
) -> RunnerHandle:
    """Spawn `python -m agent_runner --pipe` as a child process and return
    its handle after the IPC handshake completes.

    Args:
        originator: client_id of the RPC caller that started this runner.
            Stamped on the PermissionRequest so only that client can answer
            via `permission.answer`. Empty string disables the per-client
            check (back-compat path for callers that don't go through the
            RPC layer — e.g. unit tests, in-process REPL).
        permission_store: PermissionStore instance to route `permission_request`
            IPC through. None means the reader uses today's auto-approve
            fast path regardless of `auto_approve` (back-compat).
        restart_policy: RFC 0002 F-4 #3. ``None`` is equivalent to
            ``RestartPolicy.disabled()`` (today's behaviour — a crashed
            runner stays crashed). When ``mode='on-crash'`` and
            ``max_restarts > 0`` the supervisor respawns the runner after
            an exponential backoff via the reader's `finally` block.
        _restart_count_carry: internal — used by the restart machinery to
            propagate the lineage's restart counter to the new handle.
            Not exposed over RPC.

    Raises:
        RuntimeError on handshake failure or if F-4 is disabled.
    """
    if sys.platform.startswith("win"):
        raise RuntimeError("F-4 supervisor is POSIX-only in this skeleton")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    log_dir = _LOG_DIR / name
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [python, "-u", "-m", "agent_runner", "--pipe", "--name", name]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        # New session so a SIGTERM to the supervisor doesn't take the
        # runner with it; we manage the runner's lifetime explicitly.
        start_new_session=True,
        env={**os.environ, "CHEETAHCLAWS_F4_CHILD": "1"},
    )

    chan = JsonLineChannel(proc.stdout, proc.stdin)
    effective_policy = restart_policy or RestartPolicy.disabled()
    # Capture the exact kwargs we'd need to spawn a successor handle from
    # this lineage. Stored on the handle so the reader's restart hook
    # doesn't have to thread them through the closure.
    start_kwargs = {
        "name":             name,
        "template_name":    template_name,
        "args":             args,
        "config":           config,
        "interval":         float(interval),
        "auto_approve":     bool(auto_approve),
        "python":           python,
        "originator":       str(originator or ""),
        "permission_store": permission_store,
        "restart_policy":   effective_policy,
    }
    handle = RunnerHandle(
        name=name, run_id=run_id, pid=proc.pid,
        started_at=time.time(), proc=proc, chan=chan,
        template_name=template_name, args=args,
        auto_approve=bool(auto_approve),
        originator=str(originator or ""),
        permission_store=permission_store,
        restart_policy=effective_policy,
        restart_count=int(_restart_count_carry),
        _start_kwargs=start_kwargs,
    )

    # Capture stderr in a background thread so a chatty runner can't
    # block on a full stderr pipe. Tail kept for crash diagnostics.
    def _drain_stderr():
        try:
            for line in iter(proc.stderr.readline, b""):
                handle.stderr_tail.extend(line[-STDERR_TAIL_BYTES:])
        except Exception:
            pass

    t_err = threading.Thread(target=_drain_stderr, daemon=True,
                             name=f"f4-stderr-{name}")
    t_err.start()
    handle._stderr_reader = t_err

    # Send init; the runner must reply with {"op": "ready"} within
    # HANDSHAKE_TIMEOUT_S, else we kill it and raise.
    try:
        chan.send({
            "op": "init",
            "payload": {
                "name":         name,
                "run_id":       run_id,
                "template":     template_name,
                "args":         args,
                "config":       _strip_unserialisable(config),
                "interval":     float(interval),
                "auto_approve": bool(auto_approve),
                "log_dir":      str(log_dir),
            },
        })
        reply = chan.recv(timeout=HANDSHAKE_TIMEOUT_S)
    except (IpcReadTimeout, EOFError, ValueError, BrokenPipeError) as e:
        _hard_kill(proc)
        raise RuntimeError(
            f"agent runner handshake failed: {type(e).__name__}: {e}; "
            f"stderr tail: {bytes(handle.stderr_tail)[-512:]!r}"
        ) from e

    if reply.get("op") != "ready":
        _hard_kill(proc)
        raise RuntimeError(f"agent runner replied {reply!r}, expected 'ready'")

    handle.status = "running"

    # Register and insert the DB row BEFORE starting the reader thread.
    # Otherwise an immediate runner exit observed by the reader's `finally`
    # would race ahead of these calls — publishing `agent_runner_crash`
    # before `agent_runner_start`, and finalising a row that hadn't been
    # inserted yet.
    _register(handle)
    _db_insert_agent_run(handle)
    bus = _get_event_bus()
    if bus is not None:
        try:
            bus.publish("agent_runner_start", {
                "name": name, "run_id": run_id, "pid": proc.pid,
                "template": template_name,
            })
        except Exception:
            pass

    # Now safe to spawn the reader.
    t_read = threading.Thread(target=_reader_loop, args=(handle,),
                              daemon=True, name=f"f4-reader-{name}")
    t_read.start()
    handle._reader = t_read

    return handle


# ── Reader loop (one thread per runner) ───────────────────────────────────


def _reader_loop(handle: RunnerHandle) -> None:
    """Pump IPC messages from the runner. Auto-approves permission
    requests (matches today's default). On EOF, classify as graceful
    exit if proc.returncode == 0, else crash."""
    log_path = _LOG_DIR / handle.name / "log.jsonl"
    bus = _get_event_bus()

    try:
        while True:
            try:
                msg = handle.chan.recv(timeout=1.0)
            except IpcReadTimeout:
                # Periodic poll — keeps the loop responsive to proc death
                # even when the runner is mid-iteration and quiet.
                if not handle.is_alive():
                    break
                continue
            except EOFError:
                break
            except (ValueError, OSError) as e:
                handle.error = f"ipc parse error: {e}"
                break

            # Wrap message dispatch in its own try/except so a malformed
            # field from a buggy runner (e.g. non-int "iteration") can't
            # unwind the reader thread and leave the subprocess orphaned.
            try:
                op = msg.get("op", "")
                if op == "iteration_start":
                    try:
                        handle.iteration = int(msg.get("iteration", handle.iteration))
                    except (TypeError, ValueError):
                        pass    # ignore bad iteration counter, keep going
                elif op == "iteration_done":
                    try:
                        handle.iteration = int(msg.get("iteration", handle.iteration))
                    except (TypeError, ValueError):
                        pass
                    # Persist BEFORE the bus broadcast so any subscriber that
                    # immediately queries the DB sees the row.
                    _persist_iteration_jsonl(log_path, msg)
                    _db_insert_iteration(handle, msg)
                    if bus is not None:
                        try:
                            bus.publish("agent_iteration_done", {
                                "name":       handle.name,
                                "run_id":     handle.run_id,
                                "iteration":  msg.get("iteration"),
                                "status":     msg.get("status"),
                                "duration_s": msg.get("duration_s"),
                            })
                        except Exception:
                            pass
                elif op == "permission_request":
                    runner_rid = str(msg.get("request_id", ""))
                    # Fast path: auto-approve runner or no store wired in.
                    # Matches the in-thread AgentRunner default (auto_approve=True)
                    # and the back-compat behaviour for callers that don't
                    # pass a PermissionStore (unit tests, REPL).
                    if handle.auto_approve or handle.permission_store is None:
                        try:
                            handle.chan.send({
                                "op":         "permission_response",
                                "request_id": runner_rid,
                                "granted":    True,
                            })
                        except (BrokenPipeError, OSError):
                            break
                    else:
                        # Route through PermissionStore so the originator
                        # answers via `permission.answer`. The callback
                        # forwards the result back to the runner. We use
                        # the runner's request_id as the IPC correlation
                        # token even though the store mints its own —
                        # the runner doesn't need to learn about the
                        # store's id at all.
                        chan_ref = handle.chan
                        def _on_answer(req, _runner_rid=runner_rid,
                                       _chan=chan_ref):
                            ans = req.answer or {}
                            granted = bool(ans.get("approve"))
                            try:
                                _chan.send({
                                    "op":         "permission_response",
                                    "request_id": _runner_rid,
                                    "granted":    granted,
                                })
                            except (BrokenPipeError, OSError):
                                pass
                        try:
                            handle.permission_store.create(
                                originator=handle.originator,
                                tool=str(msg.get("tool", "")),
                                tool_input=msg.get("input", {}) or {},
                                rationale=str(msg.get("rationale", "")),
                                on_answer=_on_answer,
                            )
                        except Exception as e:
                            handle.error = (
                                f"permission_routing: {type(e).__name__}: {e}"
                            )[:512]
                            try:
                                handle.chan.send({
                                    "op":         "permission_response",
                                    "request_id": runner_rid,
                                    "granted":    False,
                                })
                            except (BrokenPipeError, OSError):
                                break
                elif op == "exit":
                    handle.status = "stopping"   # waits for the proc to actually exit below
                    # Note: don't break here; proc.wait() will be observed.
                elif op == "notify":
                    # RFC 0002 F-4 #2 — forward the runner's `notify` IPC
                    # payload to the bridge mailbox. The runner can target
                    # a specific bridge by setting ``msg["bridge"]``
                    # (e.g. "telegram") or omit it for a broadcast to every
                    # live bridge ("*"). Best-effort: a missing/disabled
                    # bridge does not raise — the runner shouldn't have to
                    # know which channels its originator owns.
                    text = msg.get("text") or msg.get("msg") or ""
                    target = str(msg.get("bridge", "*") or "*")
                    if text:
                        try:
                            from . import bridge_supervisor as _bs
                            delivered = _bs.notify(target, str(text))
                        except Exception as e:
                            handle.error = (
                                f"notify: {type(e).__name__}: {e}"[:512])
                            delivered = False
                        if bus is not None:
                            try:
                                bus.publish("agent_runner_notify", {
                                    "name":      handle.name,
                                    "run_id":    handle.run_id,
                                    "bridge":    target,
                                    "delivered": bool(delivered),
                                    # Truncate aggressively — chat-sized
                                    # notifications could be many KB and
                                    # the event bus has retention bounds.
                                    "text":      str(text)[:500],
                                })
                            except Exception:
                                pass
                elif op == "log":
                    # Forward through the daemon's logger when available.
                    # For the skeleton we just bus-publish at info level.
                    if bus is not None:
                        try:
                            bus.publish("agent_runner_log", {
                                "name":  handle.name,
                                "level": msg.get("level", "info"),
                                "msg":   msg.get("msg", ""),
                            })
                        except Exception:
                            pass
                elif op == "paused_budget":
                    # RFC 0002 §F-9 — runner blocked on a budget cap.
                    # Flip SQLite + publish quota_warn so the originator
                    # can react (typically: agent.resume with bumped
                    # budget_overrides).
                    handle.status = "paused_budget"
                    reason = str(msg.get("reason", ""))[:300]
                    handle.error = reason     # surfaced via agent.status
                    _db_update_run_status(handle, "paused_budget", reason)
                    if bus is not None:
                        try:
                            bus.publish("quota_warn", {
                                "name":   handle.name,
                                "run_id": handle.run_id,
                                "reason": reason,
                            })
                        except Exception:
                            pass
                elif op == "resumed":
                    # Mirror of paused_budget — runner woke from the
                    # pause and is about to attempt the next iteration.
                    handle.status = "running"
                    handle.error = ""
                    _db_update_run_status(handle, "running", None)
                    if bus is not None:
                        try:
                            bus.publish("agent_runner_resumed", {
                                "name":   handle.name,
                                "run_id": handle.run_id,
                            })
                        except Exception:
                            pass
            except Exception as e:
                # Malformed payload, programmer error in dispatch, etc.
                # Don't propagate — record the most recent error on the
                # handle (visible via agent.status) and continue reading.
                handle.error = f"reader: {type(e).__name__}: {e}"[:512]

            if not handle.is_alive():
                break
    finally:
        # If the reader unwound while the subprocess is still alive
        # (e.g., an uncaught exception above), kill it so we don't leak
        # a runner that the supervisor no longer monitors.
        if handle.proc.poll() is None:
            _hard_kill(handle.proc)
        # Reap and classify.
        try:
            rc = handle.proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            rc = -1
        stderr_text = bytes(handle.stderr_tail)[-1024:].decode(
            "utf-8", errors="replace")
        prev_status = handle.status
        if rc == 0 or prev_status == "stopping":
            handle.status = "stopped"
            ev = "agent_runner_stopped"
            db_error = None
        else:
            handle.status = "crashed"
            ev = "agent_runner_crash"
            # Truncate to keep the agent_runs.error column reasonable.
            db_error = (f"exit_code={rc}; stderr_tail={stderr_text}"
                        if stderr_text else f"exit_code={rc}")
            db_error = db_error[:1024]
        # Finalize SQLite first so the bus subscribers' DB queries are
        # consistent with the event they just received.
        _db_finalize_run(handle, status=handle.status, error=db_error)
        if bus is not None:
            try:
                bus.publish(ev, {
                    "name":        handle.name,
                    "run_id":      handle.run_id,
                    "pid":         handle.pid,
                    "exit_code":   rc,
                    "iterations":  handle.iteration,
                    "stderr_tail": stderr_text,
                })
            except Exception:
                pass
        # RFC 0002 F-4 #3 — restart policy. Only crashes trigger respawn;
        # a graceful stop must never be followed by a restart (the user
        # explicitly asked the runner to go away). The hook is best-effort:
        # any failure inside _maybe_schedule_restart is logged on the
        # handle but never re-raised, so the reader thread always exits
        # cleanly.
        if handle.status == "crashed":
            try:
                _maybe_schedule_restart(handle)
            except Exception as e:
                handle.error = (handle.error + " | "
                                if handle.error else ""
                                ) + f"restart_hook: {type(e).__name__}: {e}"


# ── Restart hook (RFC 0002 F-4 #3) ────────────────────────────────────────


def _maybe_schedule_restart(prev: RunnerHandle) -> None:
    """Called from the reader's `finally` after a crash. If the lineage's
    restart_policy says so, kick off a Timer that respawns the runner.

    Sequence:
      1. Compute next_delay() from the policy + restart_count.
      2. If None: emit ``agent_runner_restart_exhausted`` (when the
         lineage actually attempted at least one restart) so observers
         can take over.  No event for "policy never enabled in the
         first place" — that's the default and not worth the noise.
      3. Otherwise: arm a threading.Timer for ``delay`` seconds; on fire,
         call ``_do_restart`` which respawns via the global factory.

    The Timer reference is stored on the handle so :func:`stop` can
    cancel a pending restart before the user re-enters this corner of
    the supervisor surface (otherwise a respawn could race past a
    deliberate stop).
    """
    prev._restart_decided = True
    policy = prev.restart_policy
    delay = policy.next_delay(prev.restart_count)
    bus = _get_event_bus()
    if delay is None:
        # Only emit "exhausted" when the lineage at least *tried* — for a
        # disabled policy the event would be spam on every crash.
        if policy.mode == "on-crash" and prev.restart_count > 0 and bus is not None:
            try:
                bus.publish("agent_runner_restart_exhausted", {
                    "name":          prev.name,
                    "run_id":        prev.run_id,
                    "restart_count": prev.restart_count,
                    "max_restarts":  policy.max_restarts,
                })
            except Exception:
                pass
        return

    if bus is not None:
        try:
            bus.publish("agent_runner_restart_scheduled", {
                "name":          prev.name,
                "run_id":        prev.run_id,
                "restart_count": prev.restart_count,
                "delay_s":       delay,
            })
        except Exception:
            pass

    timer = threading.Timer(delay, _do_restart, args=(prev,))
    timer.daemon = True
    timer.name = f"f4-restart-{prev.name}"
    prev._restart_timer = timer
    timer.start()


def _do_restart(prev: RunnerHandle) -> None:
    """Timer callback. Re-spawn the runner with the same start_kwargs and
    a bumped restart_count. A concurrent ``stop()`` may have unregistered
    the lineage first — in that case we abort silently (the user is in
    charge)."""
    # Race guard: if the registry no longer holds *this* handle, the user
    # called stop() between scheduling and firing. Don't respawn.
    with _handles_lock:
        current = _handles.get(prev.name)
        if current is None or current.run_id != prev.run_id:
            return
        # Atomically swap the slot to a "restart in progress" sentinel so a
        # second stop() arriving mid-restart doesn't double-fire.
        _handles.pop(prev.name, None)

    spawner = _RESTART_SPAWNER or start
    try:
        new_handle = spawner(
            **prev._start_kwargs,
            _restart_count_carry=prev.restart_count + 1,
        )
    except Exception as e:
        # Respawn itself failed (e.g. handshake timeout because the agent
        # template now blows up at import). Treat the lineage as
        # exhausted at this attempt — no further auto-restart, but
        # observers get a clear signal.
        bus = _get_event_bus()
        if bus is not None:
            try:
                bus.publish("agent_runner_restart_failed", {
                    "name":          prev.name,
                    "run_id":        prev.run_id,
                    "restart_count": prev.restart_count,
                    "error":         f"{type(e).__name__}: {e}"[:512],
                })
            except Exception:
                pass
        return

    bus = _get_event_bus()
    if bus is not None:
        try:
            bus.publish("agent_runner_restart", {
                "name":           new_handle.name,
                "old_run_id":     prev.run_id,
                "new_run_id":     new_handle.run_id,
                "restart_count":  new_handle.restart_count,
                "pid":            new_handle.pid,
            })
        except Exception:
            pass


# ── Stop ──────────────────────────────────────────────────────────────────


def stop(name: str, *, timeout_s: float = 5.0) -> bool:
    """Stop a runner. Returns True iff the process actually exited.

    Order:
      1. Send IPC "stop" (graceful — runner finishes its current iter and exits).
      2. After GRACEFUL_STOP_TIMEOUT_S: SIGTERM.
      3. After GRACEFUL_STOP_TIMEOUT_S + SIGTERM_GRACE_S: SIGKILL.

    Bounded by ``timeout_s`` (default 5 s — matches F-4 acceptance).

    Side-effect: cancels any pending restart timer first (RFC 0002 F-4 #3).
    A respawn that was scheduled because the runner crashed earlier must
    not fire after the user explicitly asked for shutdown.
    """
    handle = get(name)
    if handle is None:
        return False

    # Cancel any in-flight restart for this lineage. Doing this *before*
    # the alive-check matters: if the previous process already exited and
    # only a Timer is keeping the lineage going, we want to neuter the
    # Timer and then return success.
    timer = handle._restart_timer
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass
        handle._restart_timer = None

    if not handle.is_alive():
        _unregister(name, expected=handle)
        return True

    handle.status = "stopping"
    deadline = time.monotonic() + timeout_s

    # 1) Polite IPC ask.
    try:
        handle.chan.send({"op": "stop"})
    except (BrokenPipeError, OSError):
        pass

    if _wait_until(handle.proc, deadline=min(deadline,
                   time.monotonic() + GRACEFUL_STOP_TIMEOUT_S)):
        _unregister(name, expected=handle)
        return True

    # 2) SIGTERM.
    try:
        os.killpg(os.getpgid(handle.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            handle.proc.terminate()
        except (ProcessLookupError, OSError):
            pass

    if _wait_until(handle.proc, deadline=deadline):
        _unregister(name, expected=handle)
        return True

    # 3) SIGKILL.
    _hard_kill(handle.proc)
    handle.proc.wait(timeout=1.0)
    _unregister(name, expected=handle)
    return True


def _wait_until(proc: subprocess.Popen, *, deadline: float) -> bool:
    """Poll until proc exits or deadline reached. Returns True iff exited."""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return proc.poll() is not None


def _hard_kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


def stop_all(*, timeout_s: float = 5.0) -> int:
    """Stop every registered runner. Returns the number that exited."""
    names = [h.name for h in list_all()]
    n = 0
    for name in names:
        if stop(name, timeout_s=timeout_s):
            n += 1
    return n


def resume(name: str) -> bool:
    """RFC 0002 §F-9 — send a ``resume`` IPC frame to a paused runner.

    Returns True iff the frame was delivered. The runner's control loop
    sets ``_resume_event`` on receipt; `_on_quota_exceeded` then unblocks,
    re-checks the quota, and proceeds (or pauses again if the cap is
    still too low).

    A runner that wasn't paused will silently absorb the frame on the
    control-loop side — so spurious resumes are safe.
    """
    handle = get(name)
    if handle is None:
        return False
    if not handle.is_alive():
        return False
    try:
        handle.chan.send({"op": "resume"})
        return True
    except (BrokenPipeError, OSError):
        return False


# ── SQLite persistence (agent_runs + agent_iterations) ───────────────────
#
# Every DB write is best-effort: a failed insert/update is logged via
# returning False but never raises, so the supervisor can keep going even
# if the daemon DB is missing or read-only. The schema lives in
# cc_daemon/schema.py (tables created by F-2's init_schema).


def _db_insert_agent_run(handle: "RunnerHandle") -> bool:
    """INSERT one row into agent_runs at start(). Idempotent: a UNIQUE
    PRIMARY KEY violation (caller retried with the same run_id) is
    swallowed because the existing row is already correct. Returns True
    iff the row was inserted (or already present)."""
    try:
        from .schema import get_conn
        conn = get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO agent_runs "
            "(id, name, template, args, status, auto_approve, "
            " started_at, last_iteration) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                handle.run_id,
                handle.name,
                handle.template_name,
                handle.args,
                "running",
                1 if handle.auto_approve else 0,
                _iso_now(),
                0,
            ),
        )
        conn.commit()
        return True
    except Exception:
        return False


def _db_insert_iteration(handle: "RunnerHandle", msg: dict) -> bool:
    """INSERT one row into agent_iterations and UPDATE agent_runs.last_iteration.

    Both writes happen inside one transaction so a duplicate iteration_done
    (PK violation) leaves last_iteration untouched.
    """
    iteration = int(msg.get("iteration", 0) or 0)
    if iteration <= 0:
        return False
    try:
        from .schema import get_conn
        conn = get_conn()
        # INSERT OR IGNORE: re-delivery of the same iteration_done shouldn't
        # double-count. UPDATE only fires when the row was newly inserted
        # to avoid clobbering on retry.
        cur = conn.execute(
            "INSERT OR IGNORE INTO agent_iterations "
            "(run_id, iteration, ts, status, duration_s, summary, "
            " in_tokens, out_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                handle.run_id,
                iteration,
                _iso_now(),
                str(msg.get("status", "ok")),
                float(msg.get("duration_s", 0.0) or 0.0),
                str(msg.get("summary", ""))[:400],
                int(msg.get("tokens_in", 0) or 0),
                int(msg.get("tokens_out", 0) or 0),
                float(msg.get("cost_usd", 0.0) or 0.0),
            ),
        )
        if cur.rowcount > 0:
            conn.execute(
                "UPDATE agent_runs SET last_iteration = ? "
                "WHERE id = ? AND last_iteration < ?",
                (iteration, handle.run_id, iteration),
            )
        conn.commit()
        return True
    except Exception:
        return False


def _db_update_run_status(handle: "RunnerHandle", status: str,
                          error: Optional[str]) -> bool:
    """Best-effort agent_runs.status flip. Used by the F-9 paused_budget
    / resumed IPC paths so SQLite reflects the runner's live state
    without waiting for finalize. Idempotent: re-applying the same status
    is a no-op at the SQL level."""
    try:
        from .schema import get_conn
        conn = get_conn()
        conn.execute(
            "UPDATE agent_runs SET status = ?, error = ? WHERE id = ?",
            (status, error, handle.run_id),
        )
        conn.commit()
        return True
    except Exception:
        return False


def _db_finalize_run(handle: "RunnerHandle", *, status: str,
                     error: Optional[str] = None) -> bool:
    """UPDATE agent_runs at process exit. Idempotent — if the row is
    already in the terminal state we still bump ended_at, which is fine
    (a redundant finalize on the same handle is rare but harmless)."""
    if status not in {"stopped", "crashed"}:
        return False
    try:
        from .schema import get_conn
        conn = get_conn()
        conn.execute(
            "UPDATE agent_runs SET status = ?, ended_at = ?, error = ? "
            "WHERE id = ?",
            (status, _iso_now(), error, handle.run_id),
        )
        conn.commit()
        return True
    except Exception:
        return False


# ── Iteration-log persistence (jsonl parity) ──────────────────────────────


def _persist_iteration_jsonl(log_path: Path, msg: dict) -> None:
    """Mirror today's ``AgentRunner._persist_record`` so a runner under
    F-4 produces the same on-disk log as a runner under threads. Format
    locked by agent_runner.py:503-515."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "iteration":  int(msg.get("iteration", 0)),
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status":     str(msg.get("status", "ok")),
                "duration_s": float(msg.get("duration_s", 0.0) or 0.0),
                "summary":    str(msg.get("summary", "")[:400]),
            }) + "\n")
    except Exception:
        # Persistence failure must not crash the supervisor — the
        # runner is still chugging.
        pass


# ── Helpers ───────────────────────────────────────────────────────────────


def _strip_unserialisable(cfg: dict) -> dict:
    """Remove dict entries that won't survive JSON round-trip. Callbacks,
    file handles, threading primitives all live in the parent's config
    today; child subprocess doesn't need them."""
    out: dict = {}
    for k, v in cfg.items():
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    return out


__all__ = [
    "RestartPolicy",
    "RunnerHandle",
    "enabled",
    "get",
    "list_all",
    "resume",
    "start",
    "stop",
    "stop_all",
]
