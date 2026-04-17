"""Sub-agent manager — orchestrates spawning, tracking, and messaging agents."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from .definitions import AgentDefinition, get_agent_definition, load_agent_definitions
from .task import (
    SubAgentTask,
    TaskStatus,
    _agent_run,
    _create_worktree,
    _git_root,
    _remove_worktree,
)

log = logging.getLogger(__name__)


class SubAgentManager:
    """Manages the lifecycle of sub-agents."""

    def __init__(self, config_dir: str | Path | None = None):
        self.config_dir = config_dir
        self._tasks: dict[str, SubAgentTask] = {}
        self._lock = threading.Lock()

    def spawn(
        self,
        prompt: str,
        agent_type: str = "general-purpose",
        name: Optional[str] = None,
        model: Optional[str] = None,
        isolation: Optional[str] = None,
        wait: bool = True,
        working_dir: Path | None = None,
    ) -> SubAgentTask:
        """Spawn a new sub-agent task."""
        agent_def = get_agent_definition(agent_type, self.config_dir)

        task = SubAgentTask(
            prompt=prompt,
            agent_type=agent_type,
            name=name,
            model=model or agent_def.model,
        )

        work_dir = working_dir or Path.cwd()

        if isolation == "worktree":
            git_root = _git_root(work_dir)
            if git_root:
                branch = f"agent/{task.id}"
                wt_path = _create_worktree(git_root, branch)
                task.worktree_path = wt_path
                task.worktree_branch = branch
                work_dir = wt_path
            else:
                log.warning("No git repo found — running without worktree isolation")

        with self._lock:
            self._tasks[task.id] = task

        if wait:
            _agent_run(task, agent_def, work_dir)
        else:
            thread = threading.Thread(
                target=_agent_run,
                args=(task, agent_def, work_dir),
                daemon=True,
                name=f"agent-{task.id}",
            )
            task.thread = thread
            thread.start()

        return task

    def get_task(self, task_id: str) -> SubAgentTask | None:
        """Retrieve a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def find_by_name(self, name: str) -> SubAgentTask | None:
        """Find a running task by its human-readable name."""
        with self._lock:
            for task in self._tasks.values():
                if task.name == name:
                    return task
        return None

    def list_tasks(self) -> list[dict[str, Any]]:
        """List all tasks with their current status."""
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    def send_message(self, to: str, message: str) -> bool:
        """Send a message to a named or ID-referenced agent."""
        task = self.find_by_name(to) or self.get_task(to)
        if not task:
            return False
        task.send_message(message)
        return True

    def cleanup(self, task_id: str) -> None:
        """Remove a completed task and its worktree."""
        with self._lock:
            task = self._tasks.pop(task_id, None)
        if task and task.worktree_path:
            _remove_worktree(task.worktree_path)

    def list_agent_types(self) -> list[dict[str, str]]:
        """Return available agent type definitions."""
        agents = load_agent_definitions(self.config_dir)
        return [
            {"name": a.name, "description": a.description}
            for a in agents.values()
        ]
