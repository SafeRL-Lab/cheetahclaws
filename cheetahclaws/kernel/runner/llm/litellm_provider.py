"""litellm_provider.py - LiteLLM adapter.

Routes to 100+ LLM providers (OpenAI, Anthropic, Google, Azure, Bedrock,
Ollama, etc.) via the litellm SDK. No proxy server needed.

Model strings use the provider/model format, e.g.
anthropic/claude-sonnet-4-20250514, azure/gpt-4o, openai/gpt-4o.

Install: pip install cheetahclaws[litellm]

See https://docs.litellm.ai/docs/providers for all supported models.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .provider import (
    LlmRequest,
    LlmResponse,
    ProviderInvalidRequest,
    ProviderUnavailable,
)


# USD → micro-USD conversion. cost_micro = USD * 1_000_000.
_USD_TO_MICRO = 1_000_000


class LiteLLMProvider:
    """Provider that routes to 100+ LLM providers via the litellm SDK.

    Construction does NOT import the SDK — that happens on first call.
    Tests can therefore exercise import paths without litellm installed,
    matching the AnthropicProvider contract.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._timeout_s = float(timeout_s)
        self._litellm = None  # lazy

    def _ensure_litellm(self):
        """Import litellm on first use. Raises ProviderUnavailable if
        the SDK is missing so the module itself stays importable on
        machines without litellm — matches AnthropicProvider's pattern."""
        if self._litellm is not None:
            return self._litellm
        try:
            import litellm  # type: ignore
        except ImportError as e:
            raise ProviderUnavailable(
                "the 'litellm' SDK is required for LiteLLMProvider; "
                "install with `pip install cheetahclaws[litellm]`"
            ) from e
        self._litellm = litellm
        return litellm

    def _build_messages(self, request: LlmRequest) -> list[dict]:
        """Convert canonical LlmRequest into OpenAI-style messages list."""
        if request.messages:
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in request.messages
            ]
            if request.system:
                messages.insert(
                    0, {"role": "system", "content": request.system}
                )
            return messages
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.user})
        return messages

    def _build_params(
        self,
        request: LlmRequest,
        *,
        stream: bool = False,
    ) -> dict:
        params: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "drop_params": True,
            "timeout": self._timeout_s,
        }
        if self._api_key:
            params["api_key"] = self._api_key
        if request.tools:
            params["tools"] = [dict(t) for t in request.tools]
        if stream:
            params["stream"] = True
            # Ask litellm to send a final chunk with usage stats so we
            # can populate token counts (otherwise streaming responses
            # would silently record 0/0 and break ledger accounting).
            params["stream_options"] = {"include_usage": True}
        return params

    def _map_exception(self, e: Exception) -> Exception:
        """Map a litellm SDK exception to our Provider* hierarchy.

        4xx-class (auth/bad-request) → ProviderInvalidRequest so
        callers can distinguish 'your fault' from 'their fault';
        everything else (rate limit, connection, timeout, server
        error) → ProviderUnavailable so the runner may retry."""
        # Read exception classes off the already-imported litellm
        # module rather than re-importing — keeps the mapper testable
        # without a real SDK installed.
        _le = getattr(self._litellm, "exceptions", None) if self._litellm else None
        if _le is None:
            return ProviderUnavailable(
                f"LiteLLM call failed: {type(e).__name__}: {e}"
            )
        invalid = (
            getattr(_le, "AuthenticationError", ()),
            getattr(_le, "BadRequestError", ()),
            getattr(_le, "NotFoundError", ()),
            getattr(_le, "UnsupportedParamsError", ()),
        )
        invalid = tuple(c for c in invalid if isinstance(c, type))
        if invalid and isinstance(e, invalid):
            return ProviderInvalidRequest(
                f"LiteLLM rejected request: {type(e).__name__}: {e}"
            )
        return ProviderUnavailable(
            f"LiteLLM call failed: {type(e).__name__}: {e}"
        )

    def _parse_tool_calls(self, message) -> list[dict]:
        """Pull tool_calls off a litellm message in canonical shape.

        Defensive on three known failure modes:
          • ``tc.function`` is None or missing  → skip
          • ``function.name`` is empty          → skip (LlmResponse would
                                                  reject it anyway)
          • ``function.arguments`` decodes to a non-dict (``"null"``,
            ``"[1,2,3]"``)                      → coerce to {} so the
                                                  LlmResponse validator
                                                  doesn't reject the whole
                                                  response."""
        out: list[dict] = []
        raw = getattr(message, "tool_calls", None) or []
        for tc in raw:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn is not None else None
            if not name:
                continue
            args_raw = getattr(fn, "arguments", None)
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            tc_id = getattr(tc, "id", None) or f"call_{len(out)}"
            out.append({"id": tc_id, "name": name, "input": args})
        return out

    def _compute_cost_micro(self, resp, model: str) -> tuple[int, bool]:
        """Use litellm.completion_cost when available; fall back to 0.

        Returns (cost_micro, known). When known is False the caller
        should surface metadata['cost_unknown']=True so downstream
        consumers can tell a real $0 (e.g. local Ollama) from an
        unpriced model.

        litellm has its own per-model price table covering 100+
        providers, so we delegate rather than duplicate it here. On
        any failure (unknown model, missing usage, table miss) we
        return (0, False)."""
        litellm = self._litellm
        if litellm is None:
            return 0, False
        try:
            usd = litellm.completion_cost(
                completion_response=resp, model=model
            )
        except Exception:
            return 0, False
        if usd is None:
            return 0, False
        if usd <= 0:
            # litellm returned a real 0 (free model). Cost is *known*
            # to be zero — not unknown.
            return 0, True
        return int(round(float(usd) * _USD_TO_MICRO)), True

    def _resp_provider_info(self, resp, model: str) -> dict:
        """Best-effort metadata about which underlying provider was hit,
        useful for cross-provider debugging."""
        meta: dict[str, Any] = {}
        litellm = self._litellm
        if litellm is None:
            return meta
        try:
            _, provider, _, _ = litellm.get_llm_provider(model)
            if provider:
                meta["litellm_provider"] = provider
        except Exception:
            pass
        actual = getattr(resp, "model", None)
        if actual and actual != model:
            meta["actual_model"] = actual
        return meta

    def __call__(self, request: LlmRequest) -> LlmResponse:
        if not isinstance(request, LlmRequest):
            raise ProviderInvalidRequest("request must be LlmRequest")
        litellm = self._ensure_litellm()
        try:
            resp = litellm.completion(**self._build_params(request))
        except Exception as e:
            raise self._map_exception(e) from e
        return self._convert_response(resp, request.model)

    def stream(self, request: LlmRequest, on_delta) -> LlmResponse:
        """Streaming text deltas via on_delta callback, return full
        LlmResponse at end of stream.

        Reassembles chunks via litellm.stream_chunk_builder so the
        final response carries usage stats, tool_calls, and the real
        finish_reason — without this, streaming calls would silently
        record 0/0 tokens and lose any tool_use the model emitted."""
        if not isinstance(request, LlmRequest):
            raise ProviderInvalidRequest("request must be LlmRequest")
        if not callable(on_delta):
            raise ProviderInvalidRequest("on_delta must be callable")
        litellm = self._ensure_litellm()
        chunks: list = []
        try:
            for chunk in litellm.completion(
                **self._build_params(request, stream=True)
            ):
                chunks.append(chunk)
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and getattr(delta, "content", None):
                    try:
                        on_delta(delta.content)
                    except Exception:
                        # Don't let user callback errors break the
                        # SDK's stream consumer.
                        pass
        except Exception as e:
            raise self._map_exception(e) from e
        try:
            final = litellm.stream_chunk_builder(
                chunks, messages=self._build_messages(request)
            )
        except Exception:
            final = None
        if final is None:
            # Builder failed (very old litellm, or no chunks). Return
            # a best-effort response with only the assembled text and
            # cost_unknown=True so the ledger / quota layer knows the
            # 0/0 token counts aren't a real "free" reading.
            text = "".join(
                (c.choices[0].delta.content or "")
                for c in chunks
                if c.choices
                and getattr(c.choices[0].delta, "content", None)
            )
            return LlmResponse(
                text=text,
                tokens_input=0,
                tokens_output=0,
                cost_micro=0,
                model=request.model,
                finish_reason="stop",
                metadata={"cost_unknown": True},
                tool_calls=(),
            )
        return self._convert_response(final, request.model)

    def _convert_response(self, resp, model: str) -> LlmResponse:
        """Convert a litellm ModelResponse → our LlmResponse."""
        message = resp.choices[0].message
        text = getattr(message, "content", "") or ""
        tool_calls = self._parse_tool_calls(message)
        usage = getattr(resp, "usage", None)
        ti = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        to = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        finish = getattr(resp.choices[0], "finish_reason", "stop") or "stop"
        cost_micro, cost_known = self._compute_cost_micro(resp, model)
        metadata = self._resp_provider_info(resp, model)
        if not cost_known:
            metadata["cost_unknown"] = True
        return LlmResponse(
            text=text,
            tokens_input=ti,
            tokens_output=to,
            cost_micro=cost_micro,
            model=model,
            finish_reason=finish,
            metadata=metadata,
            tool_calls=tuple(tool_calls),
        )


__all__ = ["LiteLLMProvider"]
