"""anthropic_provider.py — real Anthropic adapter (lazy import).

Wraps the ``anthropic`` SDK in a Provider-shaped callable. Imports
``anthropic`` lazily so ``import kernel.runner.llm`` succeeds on
machines without the SDK installed.

Cost calculation uses Anthropic's published per-million-token rates
as of 2026-05; if the user passes a model not in the table, the
provider falls back to a conservative pricing estimate (the
opus-4-7 rate) and warns via the `metadata` field.
"""
from __future__ import annotations

import os
from typing import Optional

from .provider import (
    LlmRequest,
    LlmResponse,
    ProviderInvalidRequest,
    ProviderUnavailable,
)


# Per-million-token USD prices (input, output) for known models.
# Conservative table — extend as needed; the runner doesn't crash on
# unknown models, just falls back to opus-4-7 rates.
_PRICING_USD_PER_M: dict[str, tuple[float, float]] = {
    "claude-opus-4-7":         (15.00, 75.00),
    "claude-opus-4-7-1m":      (15.00, 75.00),
    "claude-opus-4-6":         (15.00, 75.00),
    "claude-sonnet-4-6":        (3.00, 15.00),
    "claude-haiku-4-5":         (1.00,  5.00),
    "claude-haiku-4-5-20251001":(1.00,  5.00),
}
_FALLBACK_PRICING = (15.00, 75.00)


class AnthropicProvider:
    """Provider that calls the Anthropic Messages API.

    Construction does NOT import the SDK — that happens on first
    ``__call__``. Tests can therefore exercise import paths without
    needing the SDK installed.

    The Anthropic SDK reads ``ANTHROPIC_API_KEY`` from the
    environment by default; pass ``api_key`` explicitly to override.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._timeout_s = float(timeout_s)
        self._client = None  # lazy
        self._messages_module = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ProviderUnavailable(
                "the 'anthropic' SDK is required for AnthropicProvider; "
                "install with `pip install anthropic`"
            ) from e
        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderUnavailable(
                "ANTHROPIC_API_KEY env var not set and no api_key provided",
            )
        try:
            self._client = anthropic.Anthropic(
                api_key=api_key,
                timeout=self._timeout_s,
            )
        except Exception as e:
            raise ProviderUnavailable(
                f"failed to construct Anthropic client: {e}"
            ) from e

    def _build_kwargs(self, request: "LlmRequest") -> dict:
        """Shared kwargs builder used by both __call__ and stream()
        — keeps the message-shape conversion and tools-forwarding
        logic in one place."""
        if request.messages:
            payload_msgs = []
            inline_systems = []
            for m in request.messages:
                if m.get("role") == "system":
                    inline_systems.append(m.get("content", ""))
                else:
                    payload_msgs.append({
                        "role":    m["role"],
                        "content": m["content"],
                    })
            merged_system = request.system
            if inline_systems:
                merged_system = "\n\n".join(
                    ([merged_system] if merged_system else [])
                    + inline_systems
                )
        else:
            payload_msgs = [{"role": "user", "content": request.user}]
            merged_system = request.system
        kwargs = {
            "model":       request.model,
            "max_tokens":  request.max_tokens,
            "temperature": request.temperature,
            "messages":    payload_msgs,
        }
        if merged_system:
            kwargs["system"] = merged_system
        if request.tools:
            kwargs["tools"] = [dict(t) for t in request.tools]
        return kwargs

    def _convert_response(self, resp, model: str) -> "LlmResponse":
        """Convert SDK response object → our LlmResponse."""
        text_parts = []
        tool_calls = []
        for block in getattr(resp, "content", []):
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id":    getattr(block, "id", ""),
                    "name":  getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                })
        text = "".join(text_parts)
        usage = getattr(resp, "usage", None)
        ti = int(getattr(usage, "input_tokens", 0)) if usage else 0
        to = int(getattr(usage, "output_tokens", 0)) if usage else 0
        in_rate, out_rate = _PRICING_USD_PER_M.get(
            model, _FALLBACK_PRICING,
        )
        cost_micro = int(round(ti * in_rate)) + int(round(to * out_rate))
        finish = getattr(resp, "stop_reason", "stop") or "stop"
        metadata = {}
        if model not in _PRICING_USD_PER_M:
            metadata["pricing_fallback"] = True
        return LlmResponse(
            text=text, tokens_input=ti, tokens_output=to,
            cost_micro=cost_micro, model=model,
            finish_reason=finish, metadata=metadata,
            tool_calls=tuple(tool_calls),
        )

    def stream(self, request: LlmRequest, on_delta) -> LlmResponse:
        """RFC 0027: streaming text deltas via on_delta callback,
        return the full LlmResponse at end of stream.

        Uses ``client.messages.stream()`` context manager. Tool-use
        blocks remain whole (Anthropic emits them as a finished
        block at end of stream)."""
        if not isinstance(request, LlmRequest):
            raise ProviderInvalidRequest("request must be LlmRequest")
        if not callable(on_delta):
            raise ProviderInvalidRequest("on_delta must be callable")
        self._ensure_client()
        try:
            kwargs = self._build_kwargs(request)
            with self._client.messages.stream(**kwargs) as stream:
                for delta in stream.text_stream:
                    if delta:
                        try:
                            on_delta(delta)
                        except Exception:
                            # Don't let user callback errors break
                            # the SDK's stream consumer.
                            pass
                final = stream.get_final_message()
        except Exception as e:
            raise ProviderUnavailable(
                f"Anthropic streaming call failed: {type(e).__name__}: {e}"
            ) from e
        return self._convert_response(final, request.model)

    def __call__(self, request: LlmRequest) -> LlmResponse:
        if not isinstance(request, LlmRequest):
            raise ProviderInvalidRequest(
                "request must be LlmRequest",
            )
        self._ensure_client()
        try:
            # Multi-turn (RFC 0020): use the canonical messages list
            # if provided. Single-turn callers get a one-message list
            # built from request.user.
            if request.messages:
                # Anthropic's API expects 'system' as a top-level
                # field and 'messages' to alternate user/assistant.
                # System messages embedded in the messages list are
                # rejected — strip them and merge into `system`.
                payload_msgs = []
                inline_systems = []
                for m in request.messages:
                    if m.get("role") == "system":
                        inline_systems.append(m.get("content", ""))
                    else:
                        payload_msgs.append({
                            "role":    m["role"],
                            "content": m["content"],
                        })
                merged_system = request.system
                if inline_systems:
                    merged_system = "\n\n".join(
                        ([merged_system] if merged_system else [])
                        + inline_systems
                    )
            else:
                payload_msgs = [{"role": "user", "content": request.user}]
                merged_system = request.system

            kwargs = {
                "model":       request.model,
                "max_tokens":  request.max_tokens,
                "temperature": request.temperature,
                "messages":    payload_msgs,
            }
            if merged_system:
                kwargs["system"] = merged_system
            # RFC 0022: forward tool definitions if provided. The
            # Anthropic SDK accepts the same shape we use
            # ({"name", "description", "input_schema"}).
            if request.tools:
                kwargs["tools"] = [dict(t) for t in request.tools]
            resp = self._client.messages.create(**kwargs)
        except Exception as e:
            # Map every Anthropic SDK exception to ProviderUnavailable;
            # the runner's caller decides whether to retry.
            raise ProviderUnavailable(
                f"Anthropic API call failed: {type(e).__name__}: {e}"
            ) from e

        # Extract text + tool_use blocks. Anthropic returns content
        # as a list of typed blocks; we accumulate text and convert
        # tool_use blocks into our canonical tool_calls shape (RFC
        # 0022 §2 "tool_calls").
        text_parts = []
        tool_calls = []
        for block in getattr(resp, "content", []):
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id":    getattr(block, "id", ""),
                    "name":  getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                })
        text = "".join(text_parts)

        # Token usage.
        usage = getattr(resp, "usage", None)
        ti = int(getattr(usage, "input_tokens", 0)) if usage else 0
        to = int(getattr(usage, "output_tokens", 0)) if usage else 0

        # Cost calc.
        in_rate, out_rate = _PRICING_USD_PER_M.get(
            request.model, _FALLBACK_PRICING,
        )
        # micro-USD = (tokens / 1_000_000) * USD_per_M * 1_000_000
        #          = tokens * USD_per_M
        # Wait: micro-USD * 10⁻⁶ = USD. tokens * (rate per 1M) / 1M = USD.
        # micro-USD = USD * 1_000_000 = tokens * rate.
        cost_micro_in  = int(round(ti * in_rate))
        cost_micro_out = int(round(to * out_rate))
        cost_micro = cost_micro_in + cost_micro_out

        finish = getattr(resp, "stop_reason", "stop") or "stop"
        metadata = {}
        if request.model not in _PRICING_USD_PER_M:
            metadata["pricing_fallback"] = True

        return LlmResponse(
            text=text,
            tokens_input=ti,
            tokens_output=to,
            cost_micro=cost_micro,
            model=request.model,
            finish_reason=finish,
            metadata=metadata,
            tool_calls=tuple(tool_calls),
        )


__all__ = ["AnthropicProvider"]
