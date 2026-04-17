"""Backward-compatibility shim -- import from multi_agent directly."""

from .definitions import (
    AgentDefinition,
    get_agent_definition,
    load_agent_definitions,
)
from .manager import SubAgentManager
from .task import SubAgentTask, TaskStatus

__all__ = [
    "AgentDefinition",
    "SubAgentManager",
    "SubAgentTask",
    "TaskStatus",
    "get_agent_definition",
    "load_agent_definitions",
]
