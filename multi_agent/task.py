"""Sub-agent task execution and git worktree management for cheetahclaws."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .definitions import AgentDefinition

log = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a sub-agent task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubAgentTask:
    """Tracks a spawned sub-agent's lifecycle."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    agent_type: str = "general-purpose"
    name: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    worktree_path: Optional[Path] = None
    worktree_branch: Optional[str] = None
    thread: Optional[threading.Thread] = None
    model: Optional[str] = None
    _messages: list[str] = field(default_factory=list)

    def send_message(self, message: str) -> None:
        """Queue a follow-up message for this agent."""
        self._messages.append(message)

    def get_pending_messages(self) -> list[str]:
        """Retrieve and clear pending messages."""
        msgs = list(self._messages)
        self._messages.clear()
        return msgs

    def to_dict(self) -> dict[str, Any]:
        """Serialize task state for reporting."""
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt[:100],
            "agent_type": self.agent_type,
            "status": self.status.value,
            "result": self.result[:200] if self.result else None,
            "error": self.error,
            "worktree_branch": self.worktree_branch,
        }


def _git_root(cwd: Path | None = None) -> Path | None:
    """Find the git repository root from cwd."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=cwd or Path.cwd(),
            timeout=10,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _create_worktree(root: Path, branch_name: str) -> Path:
    """Create a git worktree for isolated agent work."""
    worktree_dir = root / ".agent_worktrees" / branch_name
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, str(worktree_dir)],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    log.info("Created worktree at %s (branch: %s)", worktree_dir, branch_name)
    return worktree_dir


def _remove_worktree(wt_path: Path) -> None:
    """Remove a git worktree and clean up."""
    root = _git_root(wt_path)
    if root:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt_path)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    if wt_path.exists():
        shutil.rmtree(wt_path, ignore_errors=True)
    log.info("Removed worktree at %s", wt_path)


def _extract_final_text(output: str) -> str:
    """Extract the final assistant text from agent output."""
    lines = output.strip().splitlines()
    result_lines: list[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            if result_lines:
                break
            continue
        result_lines.append(line)
    result_lines.reverse()
    return "\n".join(result_lines) if result_lines else output[-500:]


def _agent_run(
    task: SubAgentTask,
    agent_def: AgentDefinition,
    working_dir: Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Execute a sub-agent in a subprocess. Runs in a background thread."""
    task.status = TaskStatus.RUNNING
    env = {**os.environ, **(extra_env or {})}

    try:
        cmd = [
            "python",
            "-m",
            "cheetahclaws",
            "--agent-mode",
            "--agent-type",
            task.agent_type,
        ]
        if task.model:
            cmd.extend(["--model", task.model])

        result = subprocess.run(
            cmd,
            input=task.prompt,
            capture_output=True,
            text=True,
            cwd=working_dir,
            env=env,
            timeout=300,
        )

        if result.returncode == 0:
            task.result = _extract_final_text(result.stdout)
            task.status = TaskStatus.COMPLETED
        else:
            task.error = result.stderr[-500:] if result.stderr else "Non-zero exit"
            task.result = result.stdout[-500:] if result.stdout else None
            task.status = TaskStatus.FAILED

    except subprocess.TimeoutExpired:
        task.error = "Agent timed out after 300s"
        task.status = TaskStatus.FAILED
    except Exception as exc:
        task.error = str(exc)
        task.status = TaskStatus.FAILED
    finally:
        if task.worktree_path and task.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        ):
            log.debug(
                "Worktree %s preserved for review (branch: %s)",
                task.worktree_path,
                task.worktree_branch,
            )
