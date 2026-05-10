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


def start_runner(
    name: str,
    template_name: str,
    args: str,
    config: dict,
    send_fn: Optional[Callable[[str], None]] = None,
    interval: float = 2.0,
    auto_approve: bool = True,
) -> "AgentRunner":
    """Create and start an AgentRunner; kill any previous runner with same name."""
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
    with _runners_lock:
        r = _runners.pop(name, None)
    if r:
        r.stop()
        return True
    return False


def stop_all() -> int:
    with _runners_lock:
        runners = list(_runners.values())
        _runners.clear()
    for r in runners:
        r.stop()
    return len(runners)


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

            prompt = (
                f"Begin the program. Args: {self.args}" if iteration == 1 and self.args
                else "Begin the program." if iteration == 1
                else "Continue to the next iteration of the program."
            )

            text_chunks: list[str] = []
            rec_status = "ok"

            try:
                for event in __import__("agent").run(
                    prompt, state, config, system_prompt
                ):
                    if self._stop_event.is_set():
                        break

                    if isinstance(event, TextChunk):
                        text_chunks.append(event.text)

                    elif isinstance(event, PermissionRequest):
                        if self.auto_approve:
                            event.granted = True
                            self._notify(
                                f"🔐 [{self.name}] Auto-approved: {event.description[:120]}"
                            )
                            rec_status = "permission"
                        else:
                            self._notify(
                                f"🔐 [{self.name}] Permission needed (agent paused):\n"
                                f"{event.description}\n\n"
                                "The agent cannot continue without approval. "
                                "Restart with `--auto-approve` to enable autonomous mode."
                            )
                            event.granted = False
                            self._stop_event.set()
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
