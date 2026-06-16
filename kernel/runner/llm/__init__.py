"""kernel.runner.llm — first real workload runner (RFC 0019).

Single-turn LLM call wrapped as a kernel-managed subprocess. Speaks
the JSON-line protocol from RFC 0016 / runner_main; emits ledger
charge messages for ``tokens`` and ``cost_micro``.

Public surface::

    LlmRequest          — dataclass: model + system + user + ...
    LlmResponse         — dataclass: text + tokens + cost + ...
    Provider            — Callable[[LlmRequest], LlmResponse]  (a protocol)
    MockProvider        — for tests; reads response from env var or
                          constructor

    ProviderUnavailable     — raised for transient/setup errors
    ProviderInvalidRequest  — raised for malformed inputs

    AnthropicProvider   — real adapter (lazy-imports `anthropic`);
                          raises ProviderUnavailable if SDK missing

Run as a subprocess::

    python -m kernel.runner.llm

with ``CC_LLM_PROVIDER`` env var set to select the provider.
"""
from __future__ import annotations

from .provider import (
    LlmRequest,
    LlmResponse,
    MockProvider,
    Provider,
    ProviderInvalidRequest,
    ProviderUnavailable,
    ScriptedMockProvider,
)

__all__ = [
    "LlmRequest",
    "LlmResponse",
    "MockProvider",
    "Provider",
    "ProviderInvalidRequest",
    "ProviderUnavailable",
    "ScriptedMockProvider",
]
