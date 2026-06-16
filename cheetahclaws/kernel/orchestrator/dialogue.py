"""dialogue.py — multi-turn LLM conversation orchestrator (RFC 0020).

Lives in the daemon's Python process. The conversation is anchored
to a long-lived **owner** agent (``agent_pid``); each turn spawns a
fresh **child** agent that does one LLM call and is then transitioned
to DEAD. This sidesteps the agent state machine's
"can't-respawn-DEAD" rule (RFC 0003 §1) — the owner stays in its
original state forever, while turn children come and go.

State plane: AgentFS at ``/conversations/<owner_pid>/history.json``
(path overridable). The orchestrator is stateless between turns —
it always reloads from AgentFS — so a daemon restart, an in-flight
crash, or a parallel inspection tool sees the same data.

Per-turn errors raise typed exceptions; the orchestrator does not
silently swallow runner failures.
"""
from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from ..errors import (
    FsNotFound,
    LedgerExists,
    RunnerHandshakeFailed,
    RunnerIpcTimeout,
)
from ..sandbox import SandboxPolicy

if TYPE_CHECKING:
    from ..api import Kernel
    from ..runner.supervisor import RunnerExitInfo


HISTORY_SCHEMA_VERSION = 1
_DEFAULT_RUNNER_ARGV = (sys.executable, "-m", "cheetahclaws.kernel.runner.llm")


# ── Errors ────────────────────────────────────────────────────────────────


class DialogueTurnFailed(RuntimeError):
    """Raised when the LLM runner exits with a non-completed
    exit_kind. ``info`` carries the full RunnerExitInfo for the
    caller to inspect."""

    def __init__(self, info: "RunnerExitInfo", msg: str = ""):
        super().__init__(
            msg or f"runner exited {info.exit_kind} (code {info.exit_code})"
        )
        self.info = info


class DialogueTurnTimeout(DialogueTurnFailed):
    """Specifically: wall-clock timeout. Subclass so callers can
    catch one or both."""


class DialogueQuotaBreached(RuntimeError):
    """Raised at the end of a turn when the supervisor's ledger
    charge crossed a hard_limit. The user message AND the assistant
    response are persisted (the call already happened and was
    charged); subsequent turns will fail until the budget is bumped
    via ``kernel.ledger.update_grant``."""

    def __init__(self, dim: str, used: int, hard_limit: int):
        super().__init__(
            f"ledger dim {dim!r} crossed hard_limit: used={used} "
            f"hard_limit={hard_limit}",
        )
        self.dim = dim
        self.used = used
        self.hard_limit = hard_limit


# ── Helpers ──────────────────────────────────────────────────────────────


def _default_history_path(agent_pid: int) -> str:
    return f"/conversations/{agent_pid}/history.json"


def _default_policy() -> SandboxPolicy:
    return SandboxPolicy(
        memory_bytes=2 * 1024**3,
        wall_seconds=300.0,
        nofile=1024,
    )


# ── Orchestrator ─────────────────────────────────────────────────────────


class DialogueOrchestrator:
    """Multi-turn LLM conversation tied to one ``agent_pid``.

    The orchestrator is stateless between turns. Each ``turn()``
    call:

        load history from AgentFS
        append user message
        create CHILD agent (parent_pid=owner)
        spawn LLM runner against child
        wait → RunnerExitInfo.text
        (child is now DEAD)
        append assistant response
        save history

    The owner pid never has its state mutated by the orchestrator.

    Concurrent turns on the same instance race for the history
    file — caller serialises (typical pattern: one orchestrator per
    chat).
    """

    def __init__(
        self,
        kernel: "Kernel",
        *,
        agent_pid: int,
        model:        str = "claude-opus-4-7",
        system:       str = "",
        max_tokens:   int = 1024,
        temperature: float = 0.7,
        runner_argv:  Optional[Sequence[str]] = None,
        runner_policy: Optional[SandboxPolicy] = None,
        runner_env:   Optional[Mapping[str, str]] = None,
        history_path: Optional[str] = None,
        wait_timeout_s: float = 300.0,
        child_grants: Optional[Mapping[str, int]] = None,
    ) -> None:
        if not isinstance(agent_pid, int):
            raise ValueError(
                f"agent_pid must be int, got {type(agent_pid).__name__}",
            )
        # Verify the agent exists; raises kernel UnknownPid if not.
        kernel.process.get(agent_pid)
        self._kernel = kernel
        self._pid = agent_pid
        self._model = model
        self._system = system
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._runner_argv = (
            tuple(runner_argv) if runner_argv else _DEFAULT_RUNNER_ARGV
        )
        self._runner_policy = runner_policy or _default_policy()
        self._runner_env = (
            dict(runner_env) if runner_env is not None else None
        )
        self._history_path = (
            history_path or _default_history_path(agent_pid)
        )
        self._wait_timeout_s = float(wait_timeout_s)
        self._child_grants = (
            dict(child_grants) if child_grants is not None else None
        )

    # ── public API ──────────────────────────────────────────────────

    def history(self) -> list[dict]:
        """Snapshot of the conversation messages (no system prompt).
        Loads from AgentFS each call."""
        return list(self._load_state()["messages"])

    def stats(self) -> dict:
        return dict(self._load_state()["stats"])

    def reset(self, *, keep_system: bool = True) -> None:
        """Wipe the conversation. AgentFS file is overwritten with
        an empty messages list."""
        new_state = self._empty_state(
            system=self._system if keep_system else "",
        )
        self._save_state(new_state)

    def turn(self, user_message: str) -> str:
        """Run one round-trip. Returns the assistant text.

        Raises:
          DialogueTurnFailed — runner exited with non-completed
            kind. History is NOT updated.
          DialogueQuotaBreached — ledger crossed hard_limit on this
            turn. History IS updated through assistant response;
            caller must bump budget before next turn.
        """
        if not isinstance(user_message, str):
            raise ValueError("user_message must be str")
        if not user_message:
            raise ValueError("user_message must be non-empty")

        # 1) Load history (or empty if first turn).
        state = self._load_state()
        prior_messages = list(state["messages"])
        tentative = prior_messages + [
            {"role": "user", "content": user_message},
        ]

        # 2) Create per-turn child agent.
        turn_index = state["stats"]["turns"] + 1
        child = self._kernel.create_agent(
            name=f"turn-{turn_index}",
            template="dialogue/turn",
            parent_pid=self._pid,
            metadata={"orchestrator": "dialogue", "turn": turn_index},
        )

        # 3) Apply per-turn ledger grants if configured.
        if self._child_grants:
            try:
                self._kernel.ledger.create(
                    pid=child.pid, grants=self._child_grants,
                )
            except LedgerExists:
                # Shouldn't happen for a fresh child, but harmless.
                pass

        # 4) Spawn the LLM runner.
        sup = self._kernel.make_supervisor()
        init_payload = {
            "model":       self._model,
            "system":      state.get("system", self._system),
            "messages":    tentative,
            "max_tokens":  self._max_tokens,
            "temperature": self._temperature,
        }
        # The runner can fail before the handshake completes (e.g.
        # CC_LLM_PROVIDER unset, missing SDK, bad init payload).
        # The supervisor raises RunnerHandshakeFailed /
        # RunnerIpcTimeout in that case; map to DialogueTurnFailed
        # so callers see one exception type for any spawn / wait
        # failure.
        try:
            sup.spawn(
                pid=child.pid,
                argv=list(self._runner_argv),
                policy=self._runner_policy,
                init_payload=init_payload,
                env=self._runner_env,
            )
        except RunnerIpcTimeout as e:
            # Synthesise a minimal failed-info object so the
            # exception carries the same shape as a wait()-time
            # failure.
            from ..runner.supervisor import RunnerExitInfo
            stub = RunnerExitInfo(
                pid=child.pid, exit_kind="failed", exit_code=-1,
                stdout_tail=b"", stderr_tail=b"",
                duration_s=0.0, ledger_charged={},
                text="", metadata={"handshake_error": str(e)},
            )
            raise DialogueTurnTimeout(stub, str(e)) from e
        except RunnerHandshakeFailed as e:
            from ..runner.supervisor import RunnerExitInfo
            stub = RunnerExitInfo(
                pid=child.pid, exit_kind="failed", exit_code=-1,
                stdout_tail=b"", stderr_tail=b"",
                duration_s=0.0, ledger_charged={},
                text="", metadata={"handshake_error": str(e)},
            )
            raise DialogueTurnFailed(stub, str(e)) from e

        # 5) Wait for completion.
        info = sup.wait(child.pid, timeout=self._wait_timeout_s)

        # 6) Map exit_kind → exception or success.
        if info.exit_kind != "completed":
            # Rollback: don't persist the user message either.
            if info.exit_kind == "crashed" and info.exit_code in (-9, -15):
                # Wall-killer or external SIGTERM.
                raise DialogueTurnTimeout(info)
            raise DialogueTurnFailed(info)

        # 7) Append assistant response, accumulate stats, persist.
        assistant_text = info.text or ""
        new_messages = tentative + [
            {"role": "assistant", "content": assistant_text},
        ]
        state["messages"] = new_messages
        state["stats"]["turns"] = turn_index
        meta = info.metadata or {}
        tokens_total = int(meta.get("tokens_total")
                           or info.ledger_charged.get("tokens", 0)
                           or 0)
        cost_micro = int(meta.get("cost_micro")
                         or info.ledger_charged.get("cost_micro", 0)
                         or 0)
        state["stats"]["total_tokens"]     += tokens_total
        state["stats"]["total_cost_micro"] += cost_micro
        state["stats"]["last_turn_at"]      = time.time()
        # Last-turn details for debugging.
        state["stats"]["last_turn_child_pid"] = child.pid
        state["stats"]["last_turn_finish"] = meta.get("finish_reason", "")

        self._save_state(state)
        return assistant_text

    # ── helpers ─────────────────────────────────────────────────────

    def _empty_state(self, *, system: str = "") -> dict:
        return {
            "version":  HISTORY_SCHEMA_VERSION,
            "model":    self._model,
            "system":   system,
            "messages": [],
            "stats": {
                "turns":             0,
                "total_tokens":      0,
                "total_cost_micro":  0,
                "started_at":        time.time(),
                "last_turn_at":      None,
            },
        }

    def _load_state(self) -> dict:
        try:
            content, _ = self._kernel.fs.read(self._history_path)
        except FsNotFound:
            return self._empty_state(system=self._system)
        try:
            data = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._empty_state(system=self._system)
        if not isinstance(data, dict) or "messages" not in data:
            return self._empty_state(system=self._system)
        # Ensure required nested fields exist (from older histories).
        data.setdefault("system", self._system)
        data.setdefault("model", self._model)
        stats = data.setdefault("stats", {})
        stats.setdefault("turns", 0)
        stats.setdefault("total_tokens", 0)
        stats.setdefault("total_cost_micro", 0)
        stats.setdefault("started_at", time.time())
        stats.setdefault("last_turn_at", None)
        return data

    def _save_state(self, state: dict) -> None:
        encoded = json.dumps(state, ensure_ascii=False).encode("utf-8")
        self._kernel.fs.write(
            pid=self._pid,
            path=self._history_path,
            content=encoded,
        )
