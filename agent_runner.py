"""
agent_runner.py — Autonomous agent loop driven by task templates.

Design
------
* Each AgentRunner owns an isolated AgentState (separate from the main REPL).
* Templates are Markdown files (built-ins in agent_templates/ or user-supplied
  path) describing what the agent should do, inspired by Karpathy's autoresearch
  program.md pattern.
* The loop calls agent.run() for each iteration, draining the generator.
  PermissionRequests are auto-granted (autonomous mode) with a notification.
* After each iteration a ≤500-char summary is sent via send_fn (bridge / terminal).
* Iteration history is persisted to ~/.cheetahclaws/agents/<name>/log.jsonl.
* call stop() or send_fn receives "!agent-stop" to terminate the loop.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import logging_utils as _log

# ── Template resolution ────────────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).parent / "agent_templates"
_USER_TEMPLATES_DIR = Path.home() / ".cheetahclaws" / "agent_templates"


def list_templates() -> list[dict]:
    """Return all known templates (built-in + user-defined)."""
    result = []
    for d, source in [(_TEMPLATES_DIR, "built-in"), (_USER_TEMPLATES_DIR, "user")]:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.md")):
            result.append({"name": f.stem, "source": source, "path": str(f)})
    return result


def load_template(name_or_path: str) -> tuple[str, str]:
    """Load a template by name or file path.

    Returns (template_content, resolved_path).
    Raises FileNotFoundError if not found.
    """
    p = Path(name_or_path)
    if p.exists():
        return p.read_text(encoding="utf-8"), str(p)

    # Search built-in then user
    for d in [_USER_TEMPLATES_DIR, _TEMPLATES_DIR]:
        candidate = d / f"{name_or_path}.md"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8"), str(candidate)

    available = [t["name"] for t in list_templates()]
    raise FileNotFoundError(
        f"Template '{name_or_path}' not found. "
        f"Available: {', '.join(available) or '(none)'}"
    )


# ── Registry ───────────────────────────────────────────────────────────────

_runners: dict[str, "AgentRunner"] = {}
_runners_lock = threading.Lock()


def get_runner(name: str) -> "AgentRunner | None":
    with _runners_lock:
        r = _runners.get(name)
        if r and not r.is_alive:
            _runners.pop(name, None)
            return None
        return r


def list_runners() -> list["AgentRunner"]:
    with _runners_lock:
        return list(_runners.values())


def _should_use_subprocess(config: dict) -> bool:
    """Pick between the in-thread (legacy) and subprocess (F-4) execution
    path. The subprocess path is POSIX-only and gated by either:
        * ``CHEETAHCLAWS_ENABLE_F4`` env var (any truthy value), OR
        * ``agent_runner_subprocess: true`` in config.
    Default is False — REPL users see no behaviour change.
    """
    if sys.platform.startswith("win"):
        return False
    env_flag = os.environ.get("CHEETAHCLAWS_ENABLE_F4", "").strip().lower()
    if env_flag in {"1", "true", "yes", "on"}:
        return True
    return bool(config.get("agent_runner_subprocess", False))


def start_runner(
    name: str,
    template_name: str,
    args: str,
    config: dict,
    send_fn: Optional[Callable[[str], None]] = None,
    interval: float = 2.0,
    auto_approve: bool = True,
):
    """Create and start an AgentRunner; kill any previous runner with same name.

    Returns either an :class:`AgentRunner` (thread mode — legacy) or a
    ``RunnerHandle`` (subprocess mode — F-4). Both expose ``.name``,
    ``.status``, and ``.is_alive`` (callable on the handle, property on
    the AgentRunner) so light-touch callers don't need to branch.
    """
    if _should_use_subprocess(config):
        # F-4 path: hand off to the daemon-side supervisor. Note that
        # send_fn is ignored in subprocess mode for the skeleton —
        # ``notify`` IPC messages are dropped on the supervisor side
        # until F-6/7/8 wires bridge delivery in.
        from cc_daemon import runner_supervisor
        return runner_supervisor.start(
            name=name,
            template_name=template_name,
            args=args,
            config=config,
            interval=interval,
            auto_approve=auto_approve,
        )

    template_content, template_path = load_template(template_name)
    runner = AgentRunner(
        name=name,
        template_content=template_content,
        template_path=template_path,
        args=args,
        config=config,
        send_fn=send_fn,
        interval=interval,
        auto_approve=auto_approve,
    )
    with _runners_lock:
        old = _runners.get(name)
        if old:
            old.stop()
        _runners[name] = runner
    runner.start()
    return runner


def stop_runner(name: str) -> bool:
    # Thread mode.
    with _runners_lock:
        r = _runners.pop(name, None)
    if r:
        r.stop()
        return True
    # Subprocess mode (F-4): the handle lives in the daemon supervisor.
    try:
        from cc_daemon import runner_supervisor
    except Exception:
        return False
    return runner_supervisor.stop(name)


def stop_all() -> int:
    with _runners_lock:
        runners = list(_runners.values())
        _runners.clear()
    for r in runners:
        r.stop()
    count = len(runners)
    try:
        from cc_daemon import runner_supervisor
        count += runner_supervisor.stop_all()
    except Exception:
        pass
    return count


# ── AgentRunner ────────────────────────────────────────────────────────────

_LOG_DIR = Path.home() / ".cheetahclaws" / "agents"


def _normalize_summary(text: str) -> str:
    """Collapse a per-iteration summary into a comparable canonical form.

    Stagnation detection compares summaries across successive iterations to
    detect when the model is stuck repeating itself (e.g. "task complete, no
    more papers to process"). Whitespace and case differences are ignored;
    structural punctuation is preserved so "Done." and "Done!" still match if
    the rest is identical, but "Done." vs "I am done." don't.
    """
    if not text:
        return ""
    # Lowercase + collapse runs of whitespace to a single space.
    return " ".join(text.lower().split()).strip()


@dataclass
class _IterationRecord:
    iteration: int
    timestamp: str
    summary: str
    status: str  # "ok" | "error" | "permission"
    duration_s: float


class AgentRunner:
    """Runs an autonomous agent loop driven by a task template."""

    def __init__(
        self,
        name: str,
        template_content: str,
        template_path: str,
        args: str,
        config: dict,
        send_fn: Optional[Callable[[str], None]],
        interval: float = 2.0,
        auto_approve: bool = True,
    ) -> None:
        self.name = name
        self.template = template_content
        self.template_path = template_path
        self.args = args
        self._config = config.copy()
        self.send_fn = send_fn
        self.interval = interval
        self.auto_approve = auto_approve

        self.iteration = 0
        self.status = "idle"
        self._stop_event = threading.Event()
        self._history: list[_IterationRecord] = []
        self._thread: threading.Thread | None = None
        self._log_dir = _LOG_DIR / name
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # Public output dir: where templates that produce user-facing files
        # (research notes, paper drafts, generated code) should land. Lives
        # under ~/.cheetahclaws/agents/<name>/output/ so all agent artifacts
        # stay in one place — no more files dropped in the cheetahclaws
        # source directory.
        self.output_dir = self._log_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public interface ───────────────────────────────────────────────────

    def start(self) -> None:
        self.status = "starting"
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True,
            name=f"agent-{self.name}",
        )
        self._thread.start()
        _log.info("agent_runner_start", name=self.name,
                  template=self.template_path, args=self.args[:100])

    def stop(self) -> None:
        self._stop_event.set()
        self.status = "stopping"
        _log.info("agent_runner_stop", name=self.name, iteration=self.iteration)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def recent_log(self, n: int = 5) -> list[_IterationRecord]:
        return self._history[-n:]

    def summary_text(self) -> str:
        lines = [f"Agent: {self.name}  status={self.status}  iter={self.iteration}"]
        for rec in self.recent_log(3):
            lines.append(f"  [{rec.iteration}] {rec.status} ({rec.duration_s:.1f}s): {rec.summary[:120]}")
        return "\n".join(lines)

    # ── Internal loop ──────────────────────────────────────────────────────

    def _notify(self, text: str) -> None:
        """Send a message to the phone/terminal."""
        if self.send_fn:
            try:
                self.send_fn(text)
            except Exception:
                pass
        else:
            print(text)

    def _on_quota_exceeded(self, qe) -> None:
        """RFC 0002 §F-9 hook — called from ``_run_loop`` when the
        pre-iteration ``quota.check_quota`` call raised QuotaExceeded.

        Base impl is a **no-op** so today's REPL path keeps its current
        behaviour: the iteration proceeds, ``agent.run`` catches
        QuotaExceeded internally, yields a ``[Quota exceeded …]`` text
        chunk, and breaks; the runner sleeps ``interval`` seconds and
        tries again. Operators see the warning in the iteration log.

        Subclasses override this to *block* until the cap is lifted —
        ``_PipeAgentRunner`` ships ``paused_budget`` over IPC so the
        supervisor can flip ``agent_runs.status`` and the originator can
        decide what to do (typically ``agent.resume(name=…)`` with a new
        ceiling).
        """
        return

    def _handle_permission_request(self, event) -> str:
        """Decide a PermissionRequest. Sets ``event.granted`` and returns
        the iteration record status (always ``"permission"``).

        Default behaviour matches today's in-thread AgentRunner:
          * ``auto_approve=True``  → grant, notify, continue.
          * ``auto_approve=False`` → deny, notify, stop the loop.

        Subclasses (e.g. ``_PipeAgentRunner``) override this to route the
        request through external machinery — RFC 0002 F-4 #1 sends it over
        the supervisor IPC channel so the originator can answer.
        """
        if self.auto_approve:
            event.granted = True
            self._notify(
                f"🔐 [{self.name}] Auto-approved: {event.description[:120]}"
            )
        else:
            self._notify(
                f"🔐 [{self.name}] Permission needed (agent paused):\n"
                f"{event.description}\n\n"
                "The agent cannot continue without approval. "
                "Restart with `--auto-approve` to enable autonomous mode."
            )
            event.granted = False
            self._stop_event.set()
        return "permission"

    def _run_loop(self) -> None:
        from agent import AgentState, PermissionRequest, TurnDone
        from agent import TextChunk, ToolStart, ToolEnd

        state = AgentState()
        config = self._config.copy()
        config["_auto_agent"] = True
        config["_auto_approve"] = self.auto_approve

        system_prompt = (
            "You are an autonomous agent executing the following task program. "
            "Run it faithfully and autonomously. After completing each iteration, "
            "write a brief 1-2 sentence summary of what you did and what you'll do next.\n\n"
            f"=== TASK PROGRAM ===\n{self.template}\n=== END PROGRAM ==="
        )

        self.status = "running"
        self._notify(
            f"🚀 Agent **{self.name}** started.\n"
            f"Template: `{Path(self.template_path).name}`\n"
            f"Args: {self.args or '(none)'}\n"
            f"Auto-approve: {self.auto_approve}\n"
            "Send `!agent stop {name}` to stop."
        )

        iteration = 0
        # Consecutive-failure tracking — stop the agent if N iterations
        # in a row hit any failure, so a fundamentally broken request
        # (context overflow that compaction can't fix, missing API key,
        # unauthorized model, etc.) doesn't loop for hours.
        # Two parallel counters:
        #   - consecutive_same_failures: same signature N times → stop
        #   - consecutive_any_failures:  ANY failure marker N times → stop
        # The second one is needed because agent.py alternates between
        # `[Failed ...]` (during the retry budget) and
        # `[Circuit breaker ...]` (during the breaker's cooldown), so
        # signature-matched counter alone keeps resetting to 1 on every
        # alternation and never reaches the limit.
        consecutive_same_failures = 0
        consecutive_any_failures = 0
        last_failure_signature: str | None = None
        _SAME_ERROR_STOP_LIMIT = 3
        _ANY_ERROR_STOP_LIMIT = 4
        # ── Stagnation detection (separate from failure tracking) ───────────
        # When the model successfully completes its turn but emits the same
        # summary text N times in a row, the template's polling loop is asking
        # it to "do more" but it's already declared itself done (e.g.
        # "Task complete. No further papers to process."). Without this guard
        # the loop burns thousands of API calls producing identical "I'm done"
        # responses. Configurable via auto_agent_dup_summary_limit; 0 disables.
        _DUP_LIMIT = int(config.get("auto_agent_dup_summary_limit", 3) or 0)
        _recent_summaries: list[str] = []   # rolling window of normalized summaries
        # Circuit-breaker awareness — when an iteration's text contains
        # the standard "[Circuit breaker OPEN ... Cooldown: Xs]" marker,
        # honor that cooldown instead of the configured 2s interval.
        # Otherwise we burn 60+ wasted iterations per single 120s cooldown.
        import re as _re_runner
        _CIRCUIT_RE = _re_runner.compile(
            r"Circuit breaker OPEN.*?Cooldown:\s*(\d+(?:\.\d+)?)\s*s",
            _re_runner.IGNORECASE,
        )
        _FAILURE_RE = _re_runner.compile(
            r"\[(?:Failed|Circuit breaker)\b[^\]]*\]",
            _re_runner.IGNORECASE,
        )

        while not self._stop_event.is_set():
            iteration += 1
            self.iteration = iteration
            self.status = f"running (iter {iteration})"
            t_start = time.monotonic()

            # RFC 0002 §F-9 — pre-flight quota check. If a budget is
            # exhausted, hand the situation off to ``_on_quota_exceeded``
            # and re-check after it returns (the F-4 subprocess override
            # blocks there until ``agent.resume`` lifts the cap; the
            # base impl is a no-op so today's REPL behaviour — agent.run
            # catches it internally and yields a quota text chunk —
            # remains unchanged).
            try:
                import quota as _quota_mod
                _quota_mod.check_quota(
                    self._config.get("_session_id", "default"), self._config)
            except _quota_mod.QuotaExceeded as _qe:
                self._on_quota_exceeded(_qe)
                if self._stop_event.is_set():
                    break
                # Re-check after _on_quota_exceeded returned; if a new
                # budget didn't actually lift the cap, the override
                # should already have called _stop_event.set(). Falling
                # through here means the operator bumped the ceiling
                # and we can attempt the iteration.

            prompt = (
                f"Begin the program. Args: {self.args}" if iteration == 1 and self.args
                else "Begin the program." if iteration == 1
                else "Continue to the next iteration of the program."
            )

            text_chunks: list[str] = []
            rec_status = "ok"
            err_msg = ""

            try:
                for event in __import__("agent").run(
                    prompt, state, config, system_prompt
                ):
                    if self._stop_event.is_set():
                        break

                    if isinstance(event, TextChunk):
                        text_chunks.append(event.text)

                    elif isinstance(event, PermissionRequest):
                        rec_status = self._handle_permission_request(event)
                        if not event.granted and self._stop_event.is_set():
                            break

                    elif isinstance(event, ToolStart):
                        cmd_preview = str(
                            (event.inputs or {}).get("command",
                             (event.inputs or {}).get("file_path", ""))
                        ).strip()[:60]
                        _log.debug("agent_tool_start", name=self.name,
                                   tool=event.name, cmd=cmd_preview)

            except Exception as exc:
                rec_status = "error"
                err_msg = str(exc)[:300]
                text_chunks.append(f"\n[ERROR: {err_msg}]")
                self._notify(f"⚠ [{self.name}] iter {iteration} error:\n{err_msg}")
                _log.warn("agent_runner_error", name=self.name, iteration=iteration,
                          error=err_msg)
                # Brief pause before retrying
                self._stop_event.wait(10.0)

            duration = time.monotonic() - t_start
            summary = "".join(text_chunks).strip()[-400:] or "(no output)"

            rec = _IterationRecord(
                iteration=iteration,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                summary=summary[:400],
                status=rec_status,
                duration_s=round(duration, 1),
            )
            self._history.append(rec)
            self._persist_record(rec)

            # Report iteration result
            if rec_status != "error":
                self._notify(
                    f"✅ [{self.name}] iter {iteration} ({duration:.0f}s):\n"
                    f"{summary[:400]}"
                )

            _log.info("agent_runner_iter", name=self.name, iteration=iteration,
                      status=rec_status, duration_s=rec.duration_s)

            # ── Consecutive-failure tracking ────────────────────────────
            # An iteration "fails" if the catch above marked it error OR
            # if the streamed text contains a `[Failed ...]` / `[Circuit
            # breaker ...]` marker (agent.py emits these in its retry
            # loop when retries are exhausted or the breaker is open).
            full_text = "".join(text_chunks)
            failure_match = _FAILURE_RE.search(full_text)
            failed_this_iter = (rec_status == "error" or bool(failure_match))
            if failed_this_iter:
                # Build a short signature so "same error 3x in a row" is
                # robust against tiny phrasing differences (timestamps,
                # session IDs).
                sig = (failure_match.group(0) if failure_match else err_msg)[:80]
                if sig == last_failure_signature:
                    consecutive_same_failures += 1
                else:
                    last_failure_signature = sig
                    consecutive_same_failures = 1
                consecutive_any_failures += 1   # any failure, regardless of sig

                # Trip on either limit. ANY-failure limit catches the
                # "Failed → Circuit breaker → Failed → …" alternation
                # pattern that signature-matching alone misses.
                tripped_same = consecutive_same_failures >= _SAME_ERROR_STOP_LIMIT
                tripped_any  = consecutive_any_failures  >= _ANY_ERROR_STOP_LIMIT
                if tripped_same or tripped_any:
                    reason = (
                        f"{consecutive_same_failures} consecutive identical failures"
                        if tripped_same
                        else f"{consecutive_any_failures} consecutive failures (mixed signatures)"
                    )
                    self._notify(
                        f"⏹ [{self.name}] stopping — {reason}.\n"
                        f"Last signature: `{sig}`\n\n"
                        f"This is usually one of: a fundamentally broken "
                        f"request (context too big to compact), an exhausted "
                        f"API key / quota, or an upstream model that's down. "
                        f"Inspect the log: `/agent log {self.name}`"
                    )
                    _log.warn("agent_runner_consecutive_failure_stop",
                              name=self.name, iterations=iteration,
                              same_count=consecutive_same_failures,
                              any_count=consecutive_any_failures,
                              signature=sig)
                    self._stop_event.set()
                    break
            else:
                consecutive_same_failures = 0
                consecutive_any_failures  = 0
                last_failure_signature = None
                # Stagnation check: only on successful iterations, because
                # failure summaries are already tracked above.
                if _DUP_LIMIT >= 2:
                    norm = _normalize_summary(summary)
                    # Skip the trivial "(no output)" case — that means the model
                    # produced nothing this turn, which is a failure mode worth
                    # surfacing differently (loop-guard handled it elsewhere).
                    if norm and norm != "(no output)":
                        _recent_summaries.append(norm)
                        # Keep only the last _DUP_LIMIT entries
                        if len(_recent_summaries) > _DUP_LIMIT:
                            _recent_summaries = _recent_summaries[-_DUP_LIMIT:]
                        if (len(_recent_summaries) >= _DUP_LIMIT
                                and len(set(_recent_summaries)) == 1):
                            self._notify(
                                f"⏹ [{self.name}] stopping — model produced "
                                f"the same summary {_DUP_LIMIT} iterations in "
                                f"a row, likely the template's task is "
                                f"already complete.\n\n"
                                f"Last summary:\n{summary[:300]}\n\n"
                                f"If this is wrong, raise the limit via "
                                f"`/config auto_agent_dup_summary_limit=10` "
                                f"or set to 0 to disable."
                            )
                            _log.warn(
                                "agent_runner_stagnation_stop",
                                name=self.name, iterations=iteration,
                                duplicate_count=_DUP_LIMIT,
                                summary_preview=summary[:200],
                            )
                            self._stop_event.set()
                            break
                    else:
                        _recent_summaries.clear()

            # ── Circuit-breaker cooldown override ───────────────────────
            # When the iteration's output mentions a circuit-breaker
            # cooldown, sleep that long (capped at 5 min) instead of
            # the configured 2s interval. Avoids 60+ pointless retries
            # against an upstream that's already telling us "wait".
            wait_s = self.interval
            cb_match = _CIRCUIT_RE.search(full_text)
            if cb_match:
                try:
                    cooldown = float(cb_match.group(1))
                    wait_s = max(self.interval, min(cooldown + 1.0, 300.0))
                    _log.info("agent_runner_circuit_wait",
                              name=self.name, cooldown_s=wait_s)
                except ValueError:
                    pass

            # Wait before next iteration (stop event wakes it early)
            self._stop_event.wait(wait_s)

        self.status = "stopped"
        self._notify(f"⏹ Agent **{self.name}** stopped after {iteration} iterations.")
        _log.info("agent_runner_stopped", name=self.name, iterations=iteration)

    def _persist_record(self, rec: _IterationRecord) -> None:
        log_file = self._log_dir / "log.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "iteration": rec.iteration,
                    "timestamp": rec.timestamp,
                    "status": rec.status,
                    "duration_s": rec.duration_s,
                    "summary": rec.summary,
                }) + "\n")
        except Exception:
            pass


# ── Subprocess entry point (RFC 0002 F-4) ──────────────────────────────────
#
# When invoked as ``python -m agent_runner --pipe``, this module turns
# itself into a runner driven by a JSON-line IPC channel on stdin/stdout
# instead of the in-process thread + send_fn API. The supervisor side lives
# in ``cc_daemon/runner_supervisor.py``.
#
# Protocol (see cc_daemon/runner_ipc.py docstring for the full message
# catalogue):
#   1. Supervisor writes {"op": "init", "payload": {...}} on stdin.
#   2. We reply {"op": "ready"} on stdout, then run the existing
#      _run_loop body with send_fn / permission flow redirected to IPC.
#   3. Per iteration we emit iteration_start / iteration_done.
#   4. PermissionRequest events bounce out as permission_request and we
#      block until the supervisor sends back permission_response.
#   5. Either side may end the run: supervisor sends {"op": "stop"},
#      or the runner reaches its natural stop conditions and sends
#      {"op": "exit", "reason": "...", "iterations": N}.

def _pipe_main(name_arg: Optional[str] = None) -> int:
    """Subprocess entry point. Returns the process exit code."""
    import argparse
    import sys as _sys
    from cc_daemon.runner_ipc import IpcReadTimeout, JsonLineChannel

    parser = argparse.ArgumentParser(prog="agent_runner")
    parser.add_argument("--pipe", action="store_true",
                        help="run as a JSON-line IPC subprocess")
    parser.add_argument("--name", default=name_arg or "",
                        help="runner name (echoed in logs)")
    args = parser.parse_args()
    if not args.pipe:
        print("agent_runner: --pipe required for subprocess mode",
              file=_sys.stderr)
        return 2

    chan = JsonLineChannel(_sys.stdin.buffer, _sys.stdout.buffer)

    # ── 1) Init handshake ─────────────────────────────────────────────────
    try:
        init = chan.recv(timeout=10.0)
    except (IpcReadTimeout, EOFError, ValueError) as e:
        print(f"agent_runner: init failed: {e}", file=_sys.stderr)
        return 2

    if init.get("op") != "init":
        print(f"agent_runner: expected init, got {init!r}", file=_sys.stderr)
        return 2

    payload = init.get("payload") or {}
    runner_name   = str(payload.get("name") or args.name or "anon")
    template_arg  = str(payload.get("template", ""))
    runner_args   = str(payload.get("args", ""))
    runner_config = dict(payload.get("config") or {})
    runner_interval = float(payload.get("interval", 2.0))
    runner_auto   = bool(payload.get("auto_approve", True))

    try:
        template_content, template_path = load_template(template_arg)
    except FileNotFoundError as e:
        # Pre-handshake failure: write to stderr and exit non-zero.
        # The supervisor sees the handshake recv hit EOF and raises a
        # RuntimeError with the stderr tail attached, which is more
        # informative than sending an IPC `log`/`exit` here would be
        # (those would be misread as the runner's handshake reply).
        print(f"agent_runner: template not found: {e}", file=_sys.stderr)
        return 1

    chan.send({"op": "ready"})

    # ── 1.5) Optional e2e stub for ``agent.run`` ──────────────────────────
    # When ``CHEETAHCLAWS_E2E_FAKE_AGENT=1`` is set in the subprocess env,
    # replace ``agent.run`` with a scripted generator. This keeps the F-4
    # end-to-end test (tests/e2e_f4_runner.py) hermetic — it exercises the
    # real `python -m agent_runner --pipe` entry point, the real
    # `_PipeAgentRunner`, the real IPC, and the real SQLite agent_runs /
    # agent_iterations writes, without depending on an LLM provider being
    # configured or reachable. The stub is gated by env var so production
    # paths can never reach it. The test caller drives termination via
    # ``rs.stop()`` once it sees the iteration counter rise.
    if os.environ.get("CHEETAHCLAWS_E2E_FAKE_AGENT") == "1":
        import agent as _agent_mod
        from agent import (
            TextChunk as _StubTextChunk,
            TurnDone as _StubTurnDone,
            PermissionRequest as _StubPermissionRequest,
        )
        _stub_emit_perm = os.environ.get("CHEETAHCLAWS_E2E_FAKE_PERMISSION") == "1"
        _stub_state = {"perm_emitted": False}

        def _fake_run(prompt, state, config, system_prompt,
                      depth=0, cancel_check=None):
            yield _StubTextChunk("e2e iteration begin")
            if _stub_emit_perm and not _stub_state["perm_emitted"]:
                _stub_state["perm_emitted"] = True
                pr = _StubPermissionRequest(
                    description="e2e fake permission request: tool=Bash"
                )
                yield pr
                if not pr.granted:
                    yield _StubTextChunk("[denied]")
                    yield _StubTurnDone(input_tokens=1, output_tokens=1)
                    return
            yield _StubTextChunk("e2e iteration done")
            yield _StubTurnDone(input_tokens=1, output_tokens=1)

        _agent_mod.run = _fake_run

    # ── 2) Bridge send_fn → IPC notify ────────────────────────────────────
    def _ipc_send(text: str) -> None:
        try:
            chan.send({"op": "notify", "text": str(text)})
        except (BrokenPipeError, OSError):
            pass

    # ── 3) Construct the existing AgentRunner but DON'T spawn its thread.
    # We drive _run_loop directly on this process's main thread so the
    # subprocess stays single-purpose. The PermissionRequest hook is
    # patched in by overriding auto_approve handling inside a thin
    # subclass that defers to IPC.
    runner = _PipeAgentRunner(
        name=runner_name,
        template_content=template_content,
        template_path=template_path,
        args=runner_args,
        config=runner_config,
        send_fn=_ipc_send,
        interval=runner_interval,
        auto_approve=runner_auto,
        chan=chan,
    )

    # ── 4) Watch for supervisor "stop" on a background thread. The
    # blocking IPC recv would otherwise compete with the agent's
    # generator drain inside _run_loop. permission_response and stop
    # both arrive via the same channel; the runner needs both.
    pending_perms: dict[str, threading.Event] = {}
    pending_perms_results: dict[str, bool] = {}

    def _control_loop():
        while not runner._stop_event.is_set():
            try:
                msg = chan.recv(timeout=0.5)
            except IpcReadTimeout:
                continue
            except (EOFError, ValueError, OSError):
                runner._stop_event.set()
                break
            op = msg.get("op", "")
            if op == "stop":
                runner._stop_event.set()
                # Also wake any quota-pause waiter so the loop unblocks
                # cleanly instead of waiting up to _PERMISSION_WAIT_S.
                try:
                    runner._resume_event.set()
                except Exception:
                    pass
                break
            if op == "permission_response":
                rid = msg.get("request_id", "")
                pending_perms_results[rid] = bool(msg.get("granted"))
                ev = pending_perms.get(rid)
                if ev is not None:
                    ev.set()
            elif op == "resume":
                # RFC 0002 §F-9 — unblock _on_quota_exceeded after the
                # originator bumped the daemon-level budgets via
                # `agent.resume`. The runner re-runs the quota check at
                # the top of the next iteration; if the new ceiling is
                # *still* too low, it pauses again.
                try:
                    runner._resume_event.set()
                except Exception:
                    pass

    runner._pending_perms = pending_perms
    runner._pending_perms_results = pending_perms_results

    ctl = threading.Thread(target=_control_loop, daemon=True,
                           name=f"f4-ctl-{runner_name}")
    ctl.start()

    try:
        runner._run_loop()
    except Exception as e:
        chan.send({"op": "log", "level": "error",
                   "msg": f"_run_loop crashed: {type(e).__name__}: {e}"})
        chan.send({"op": "exit", "reason": "exception",
                   "iterations": runner.iteration})
        return 1

    chan.send({"op": "exit", "reason": "completed",
               "iterations": runner.iteration})
    return 0


class _PipeAgentRunner(AgentRunner):
    """AgentRunner driven from an IPC channel instead of an in-process
    send_fn / permission callback. Overrides the two seams in the parent
    class:

      * iteration boundary  → emit iteration_start / iteration_done
      * PermissionRequest   → emit permission_request, await response
    """

    # Cap the wait on the supervisor's permission response. Matches the
    # PermissionStore's interactive default so the runner's view of the
    # timeout stays consistent with the store's janitor.
    _PERMISSION_WAIT_S = 30 * 60

    def __init__(self, *, chan, **kw) -> None:
        super().__init__(**kw)
        self._chan = chan
        self._pending_perms: dict = {}
        self._pending_perms_results: dict = {}
        # RFC 0002 §F-9 — _on_quota_exceeded blocks on this until the
        # supervisor delivers a `resume` IPC frame (driven by the
        # `agent.resume` RPC). Re-armed on every iteration.
        self._resume_event = threading.Event()

    def _on_quota_exceeded(self, qe) -> None:
        """RFC 0002 §F-9 — block until the supervisor delivers a
        ``resume`` IPC frame, then return so ``_run_loop`` re-checks
        the quota and proceeds.

        Sequence:
          1. Send ``{"op":"paused_budget", "reason": …}`` IPC. The
             supervisor flips ``agent_runs.status='paused_budget'`` and
             publishes ``quota_warn`` on the bus so observers can react.
          2. Clear and then wait on ``_resume_event``. The wait honours
             ``_stop_event`` (the control loop sets the resume event
             before breaking on stop), so a stop arriving while paused
             unblocks the runner cleanly.
          3. After resume, send ``{"op":"resumed"}`` IPC so the
             supervisor can flip the SQLite status back to ``running``.
        """
        reason = getattr(qe, "reason", "") or str(qe)
        try:
            self._chan.send({
                "op":     "paused_budget",
                "reason": str(reason)[:300],
            })
        except (BrokenPipeError, OSError):
            self._stop_event.set()
            return

        self.status = "paused_budget"
        self._notify(
            f"⏸ [{self.name}] paused — {reason}. "
            f"Use `agent.resume` with new budget_overrides to unblock."
        )
        # Clear before wait so an old set() can't satisfy us instantly.
        self._resume_event.clear()
        self._resume_event.wait()
        # Re-arm for the next pause.
        self._resume_event.clear()
        try:
            self._chan.send({"op": "resumed"})
        except (BrokenPipeError, OSError):
            self._stop_event.set()
            return
        self.status = "running"
        self._notify(f"▶ [{self.name}] resumed.")

    def _persist_record(self, rec: _IterationRecord) -> None:
        # Keep the parent's jsonl write so on-disk behaviour is identical;
        # also push iteration_done over IPC so the supervisor learns about
        # iteration boundaries in real time.
        super()._persist_record(rec)
        try:
            self._chan.send({
                "op":         "iteration_done",
                "iteration":  rec.iteration,
                "status":     rec.status,
                "duration_s": rec.duration_s,
                "summary":    rec.summary,
                "tokens_in":  0,
                "tokens_out": 0,
            })
        except (BrokenPipeError, OSError):
            self._stop_event.set()

    def _handle_permission_request(self, event) -> str:
        """Route the request through the supervisor IPC channel and block
        until ``permission_response`` arrives.

        Fast path: ``auto_approve=True`` delegates to the parent so the
        runner doesn't bother the supervisor at all.

        Slow path: emit ``permission_request`` with a fresh correlation
        id, register an event in ``_pending_perms``, and wait up to
        ``_PERMISSION_WAIT_S``. On wait timeout or IPC error we deny and
        stop, matching the parent's "no approval ⇒ paused" stance.
        """
        if self.auto_approve:
            return super()._handle_permission_request(event)

        import uuid as _uuid

        rid = _uuid.uuid4().hex[:12]
        ev = threading.Event()
        # Register BEFORE sending so a fast response can't race ahead of
        # the wait setup.
        self._pending_perms[rid] = ev
        try:
            self._chan.send({
                "op":         "permission_request",
                "request_id": rid,
                "tool":       getattr(event, "name", "") or "",
                "input":      getattr(event, "inputs", {}) or {},
                "rationale":  getattr(event, "description", "") or "",
            })
        except (BrokenPipeError, OSError):
            self._pending_perms.pop(rid, None)
            event.granted = False
            self._stop_event.set()
            return "permission"

        if not ev.wait(timeout=self._PERMISSION_WAIT_S):
            # Supervisor never answered — treat as deny + stop.
            self._pending_perms.pop(rid, None)
            self._pending_perms_results.pop(rid, None)
            event.granted = False
            self._stop_event.set()
            self._notify(
                f"🔐 [{self.name}] Permission request timed out "
                f"(no response after {self._PERMISSION_WAIT_S}s)."
            )
            return "permission"

        granted = bool(self._pending_perms_results.pop(rid, False))
        self._pending_perms.pop(rid, None)
        event.granted = granted
        if not granted:
            self._notify(
                f"🔐 [{self.name}] Permission denied by originator — stopping."
            )
            self._stop_event.set()
        else:
            self._notify(
                f"🔐 [{self.name}] Approved: {event.description[:120]}"
            )
        return "permission"


if __name__ == "__main__":
    sys.exit(_pipe_main())
