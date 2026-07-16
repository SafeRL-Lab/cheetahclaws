"""kernel.orchestrator — multi-step LLM workflows on the kernel.

A thin layer above ``kernel.runner.llm`` that chains N
single-turn calls into higher-level patterns. Currently ships:

  DialogueOrchestrator     multi-turn conversation (RFC 0020)

Future RFCs will add:
  - Tool-using orchestrator (with permission routing)
  - DAG / step-graph orchestrator
  - Fan-out / aggregator
"""
from __future__ import annotations

from .dialogue import (
    DialogueOrchestrator,
    DialogueQuotaBreached,
    DialogueTurnFailed,
    DialogueTurnTimeout,
)

__all__ = [
    "DialogueOrchestrator",
    "DialogueTurnFailed",
    "DialogueTurnTimeout",
    "DialogueQuotaBreached",
]
