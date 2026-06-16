"""provider.py — Provider protocol + dataclasses + MockProvider.

A Provider is a callable that maps an LlmRequest to an LlmResponse.
This module ships only the abstraction and a deterministic mock.
Real adapters live in their own modules (anthropic_provider.py etc.)
and are imported lazily so the absence of an SDK doesn't break this
module's import.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable


# ── Errors ────────────────────────────────────────────────────────────────


class ProviderUnavailable(RuntimeError):
    """Raised when a provider can't service a request right now —
    network failure, rate limit, missing credentials, missing SDK
    dependency, etc.

    The runner translates this into ``exit_kind=failed``.
    """


class ProviderInvalidRequest(ValueError):
    """Raised for malformed inputs — empty user message, unknown
    model name, etc. Distinct from ProviderUnavailable so callers
    can distinguish "your fault" from "their fault"."""


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LlmRequest:
    model:       str
    user:        str = ""            # single-turn convenience
    system:      str = ""            # optional system prompt
    messages:    tuple = ()          # canonical multi-turn (RFC 0020)
    tools:       tuple = ()          # provider-native tool defs (RFC 0022)
    stream:      bool = False        # RFC 0027 — opt-in token-by-token
    max_tokens:  int = 1024
    temperature: float = 0.7
    metadata:    dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model:
            raise ProviderInvalidRequest(
                f"model must be non-empty str, got {self.model!r}",
            )
        # Either messages OR user must be set. Both is allowed —
        # provider implementations prefer messages.
        has_messages = bool(self.messages)
        has_user = bool(self.user)
        if not has_messages and not has_user:
            raise ProviderInvalidRequest(
                "either 'messages' or 'user' must be non-empty",
            )
        if has_messages:
            for i, m in enumerate(self.messages):
                if not isinstance(m, dict):
                    raise ProviderInvalidRequest(
                        f"messages[{i}] must be a dict, got {type(m).__name__}",
                    )
                role = m.get("role")
                content = m.get("content")
                if role not in ("user", "assistant", "system"):
                    raise ProviderInvalidRequest(
                        f"messages[{i}].role must be 'user'|'assistant'|'system', "
                        f"got {role!r}",
                    )
                # Content may be either a plain string OR a list of
                # content blocks (Anthropic-style multi-content
                # messages — text + tool_use, or tool_result list).
                # See RFC 0022 §3 for the runner-side construction
                # of these shapes during a tool-calling loop.
                if isinstance(content, str):
                    pass
                elif isinstance(content, list):
                    for j, block in enumerate(content):
                        if not isinstance(block, dict):
                            raise ProviderInvalidRequest(
                                f"messages[{i}].content[{j}] must be a dict, "
                                f"got {type(block).__name__}",
                            )
                else:
                    raise ProviderInvalidRequest(
                        f"messages[{i}].content must be str or list, got "
                        f"{type(content).__name__}",
                    )
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ProviderInvalidRequest(
                f"max_tokens must be positive int, got {self.max_tokens!r}",
            )
        if not isinstance(self.temperature, (int, float)) or \
                not (0.0 <= self.temperature <= 2.0):
            raise ProviderInvalidRequest(
                f"temperature must be in [0, 2], got {self.temperature!r}",
            )
        if not isinstance(self.stream, bool):
            raise ProviderInvalidRequest(
                f"stream must be bool, got {type(self.stream).__name__}",
            )

    @property
    def has_messages(self) -> bool:
        return bool(self.messages)

    def to_dict(self) -> dict:
        return {
            "model":       self.model,
            "user":        self.user,
            "system":      self.system,
            "messages":    [dict(m) for m in self.messages],
            "tools":       [dict(t) for t in self.tools],
            "stream":      self.stream,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "metadata":    dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LlmRequest":
        if "model" not in d:
            raise ProviderInvalidRequest("missing 'model'")
        msgs_raw = d.get("messages") or ()
        if msgs_raw and not isinstance(msgs_raw, (list, tuple)):
            raise ProviderInvalidRequest(
                f"messages must be a list, got {type(msgs_raw).__name__}",
            )
        tools_raw = d.get("tools") or ()
        if tools_raw and not isinstance(tools_raw, (list, tuple)):
            raise ProviderInvalidRequest(
                f"tools must be a list, got {type(tools_raw).__name__}",
            )
        # Either messages or user is required.
        has_msgs = bool(msgs_raw)
        if not has_msgs and "user" not in d:
            raise ProviderInvalidRequest("either 'user' or 'messages' is required")
        return cls(
            model       = str(d["model"]),
            user        = str(d.get("user", "")),
            system      = str(d.get("system", "")),
            messages    = tuple(dict(m) for m in msgs_raw),
            tools       = tuple(dict(t) for t in tools_raw),
            stream      = bool(d.get("stream", False)),
            max_tokens  = int(d.get("max_tokens", 1024)),
            temperature = float(d.get("temperature", 0.7)),
            metadata    = dict(d.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LlmResponse:
    text:           str
    tokens_input:   int
    tokens_output:  int
    cost_micro:     int            # micro-USD = 10⁻⁶ USD
    model:          str
    finish_reason:  str = "stop"   # 'stop' | 'length' | 'tool_use' | 'error' | provider-specific
    metadata:       dict = field(default_factory=dict)
    # RFC 0022: function-calling support. Each entry:
    #   {"id": str, "name": str, "input": dict}
    tool_calls:     tuple = ()

    def __post_init__(self):
        for fld in ("tokens_input", "tokens_output", "cost_micro"):
            v = getattr(self, fld)
            if not isinstance(v, int) or v < 0:
                raise ProviderInvalidRequest(
                    f"{fld} must be non-negative int, got {v!r}",
                )
        if not isinstance(self.text, str):
            raise ProviderInvalidRequest("text must be str")
        if not isinstance(self.model, str) or not self.model:
            raise ProviderInvalidRequest("model must be non-empty str")
        # Validate tool_calls shape if any.
        for i, tc in enumerate(self.tool_calls):
            if not isinstance(tc, dict):
                raise ProviderInvalidRequest(
                    f"tool_calls[{i}] must be dict, got {type(tc).__name__}",
                )
            if not isinstance(tc.get("id"), str) or not tc["id"]:
                raise ProviderInvalidRequest(
                    f"tool_calls[{i}].id must be non-empty str",
                )
            if not isinstance(tc.get("name"), str) or not tc["name"]:
                raise ProviderInvalidRequest(
                    f"tool_calls[{i}].name must be non-empty str",
                )
            if not isinstance(tc.get("input", {}), dict):
                raise ProviderInvalidRequest(
                    f"tool_calls[{i}].input must be dict",
                )

    @property
    def tokens_total(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def is_tool_use(self) -> bool:
        """True iff the response includes at least one tool_call."""
        return bool(self.tool_calls)

    def to_dict(self) -> dict:
        return {
            "text":          self.text,
            "tokens_input":  self.tokens_input,
            "tokens_output": self.tokens_output,
            "cost_micro":    self.cost_micro,
            "model":         self.model,
            "finish_reason": self.finish_reason,
            "metadata":      dict(self.metadata),
            "tool_calls":    [dict(tc) for tc in self.tool_calls],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LlmResponse":
        return cls(
            text          = str(d.get("text", "")),
            tokens_input  = int(d.get("tokens_input", 0)),
            tokens_output = int(d.get("tokens_output", 0)),
            cost_micro    = int(d.get("cost_micro", 0)),
            model         = str(d.get("model", "")),
            finish_reason = str(d.get("finish_reason", "stop")),
            metadata      = dict(d.get("metadata") or {}),
            tool_calls    = tuple(dict(tc) for tc in (d.get("tool_calls") or ())),
        )


# ── Provider protocol ────────────────────────────────────────────────────


# Type alias rather than typing.Protocol — we intentionally accept
# any callable, including bare functions.
Provider = Callable[[LlmRequest], LlmResponse]


# ── MockProvider ─────────────────────────────────────────────────────────


class MockProvider:
    """Deterministic provider for tests. Returns the same
    LlmResponse on every call.

    Three constructors, in order of preference:

    1. ``MockProvider(response=LlmResponse(...))`` — explicit.
    2. ``MockProvider.from_env()`` — reads ``CC_LLM_MOCK_RESPONSE_JSON``
       env var (a JSON LlmResponse). Used by the subprocess entry
       point ``python -m kernel.runner.llm``.
    3. ``MockProvider.echo()`` — minimal default that echoes the user
       message back verbatim with token counts proportional to
       length.
    """

    ENV_RESPONSE = "CC_LLM_MOCK_RESPONSE_JSON"

    def __init__(self, response: LlmResponse) -> None:
        if not isinstance(response, LlmResponse):
            raise ProviderInvalidRequest(
                "response must be LlmResponse",
            )
        self._response = response
        self.calls: list[LlmRequest] = []

    def __call__(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        return self._response

    @classmethod
    def from_env(cls) -> "MockProvider":
        raw = os.environ.get(cls.ENV_RESPONSE)
        if not raw:
            raise ProviderUnavailable(
                f"{cls.ENV_RESPONSE} env var not set; "
                "MockProvider.from_env() needs a JSON LlmResponse",
            )
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ProviderUnavailable(
                f"{cls.ENV_RESPONSE} is not valid JSON: {e}",
            ) from e
        return cls(LlmResponse.from_dict(d))

    @classmethod
    def scripted(cls, responses) -> "ScriptedMockProvider":
        """Convenience constructor for ScriptedMockProvider."""
        return ScriptedMockProvider(list(responses))

    @classmethod
    def echo(cls, *, model: str = "mock-echo-1") -> "MockProvider":
        # We can't precompute the response without seeing the request,
        # so echo() returns a thin shim that customises per-call.
        class _EchoProvider(cls):
            def __init__(self) -> None:  # noqa: D401
                # Skip parent __init__; we don't have a fixed response.
                self.calls: list[LlmRequest] = []
                self._model = model

            def __call__(self, request: LlmRequest) -> LlmResponse:
                self.calls.append(request)
                # Token counts proportional to character length / 4.
                ti = max(1, (len(request.user) + len(request.system)) // 4)
                to = max(1, len(request.user) // 4)
                return LlmResponse(
                    text=f"echo: {request.user}",
                    tokens_input=ti,
                    tokens_output=to,
                    cost_micro=ti * 3 + to * 15,  # rough rate
                    model=request.model,
                    finish_reason="stop",
                )
        return _EchoProvider()


# ── ScriptedMockProvider ──────────────────────────────────────────────────


class ScriptedMockProvider:
    """Returns a sequence of pre-canned responses (RFC 0022).

    Useful for testing multi-iteration tool-calling flows: the
    first response can be a tool_use, the second the final text.
    Exhaustion (more calls than responses) raises
    ``ProviderUnavailable``.
    """

    ENV_RESPONSES = "CC_LLM_SCRIPTED_RESPONSES_JSON"

    def __init__(self, responses: list[LlmResponse]) -> None:
        if not isinstance(responses, list) or not responses:
            raise ProviderInvalidRequest(
                "responses must be a non-empty list of LlmResponse",
            )
        for i, r in enumerate(responses):
            if not isinstance(r, LlmResponse):
                raise ProviderInvalidRequest(
                    f"responses[{i}] must be LlmResponse, "
                    f"got {type(r).__name__}",
                )
        self._responses = list(responses)
        self._cursor = 0
        self.calls: list[LlmRequest] = []

    def __call__(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        if self._cursor >= len(self._responses):
            raise ProviderUnavailable(
                f"ScriptedMockProvider exhausted after "
                f"{len(self._responses)} calls",
            )
        resp = self._responses[self._cursor]
        self._cursor += 1
        return resp

    def stream(self, request: LlmRequest, on_delta) -> LlmResponse:
        """RFC 0027: emit each character of the next response's text
        via on_delta, then return the full response. Tool-use
        responses (no text) emit zero deltas; the caller can still
        process the tool_calls from the returned response."""
        # Pull next response (advances cursor like __call__ does).
        response = self(request)
        if not callable(on_delta):
            raise ProviderInvalidRequest("on_delta must be callable")
        for ch in response.text:
            on_delta(ch)
        return response

    @property
    def remaining(self) -> int:
        return len(self._responses) - self._cursor

    @classmethod
    def from_env(cls) -> "ScriptedMockProvider":
        raw = os.environ.get(cls.ENV_RESPONSES)
        if not raw:
            raise ProviderUnavailable(
                f"{cls.ENV_RESPONSES} env var not set; "
                "ScriptedMockProvider.from_env() needs a JSON list",
            )
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ProviderUnavailable(
                f"{cls.ENV_RESPONSES} is not valid JSON: {e}",
            ) from e
        if not isinstance(arr, list):
            raise ProviderUnavailable(
                f"{cls.ENV_RESPONSES} must decode to a JSON array, "
                f"got {type(arr).__name__}",
            )
        return cls([LlmResponse.from_dict(d) for d in arr])
