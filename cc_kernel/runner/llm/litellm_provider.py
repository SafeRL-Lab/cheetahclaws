"""litellm_provider.py - LiteLLM adapter.

Routes to 100+ LLM providers (OpenAI, Anthropic, Google, Azure, Bedrock,
Ollama, etc.) via the litellm SDK. No proxy server needed.

Model strings use the provider/model format, e.g.
anthropic/claude-sonnet-4-20250514, azure/gpt-4o, openai/gpt-4o.

Install: pip install cheetahclaws[litellm]

See https://docs.litellm.ai/docs/providers for all supported models.
"""

from __future__ import annotations

from typing import Optional

import litellm

from .provider import (
    LlmRequest,
    LlmResponse,
    ProviderInvalidRequest,
    ProviderUnavailable,
)


class LiteLLMProvider:
    """Provider that routes to 100+ LLM providers via the litellm SDK."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._timeout_s = float(timeout_s)

    def __call__(self, request: LlmRequest) -> LlmResponse:
        if not isinstance(request, LlmRequest):
            raise ProviderInvalidRequest("request must be LlmRequest")

        try:
            if request.messages:
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in request.messages
                ]
                if request.system:
                    messages.insert(0, {"role": "system", "content": request.system})
            else:
                messages = []
                if request.system:
                    messages.append({"role": "system", "content": request.system})
                messages.append({"role": "user", "content": request.user})

            params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "drop_params": True,
                "timeout": self._timeout_s,
            }
            if self._api_key:
                params["api_key"] = self._api_key
            if request.tools:
                params["tools"] = [dict(t) for t in request.tools]

            resp = litellm.completion(**params)

        except Exception as e:
            raise ProviderUnavailable(
                f"LiteLLM API call failed: {type(e).__name__}: {e}"
            ) from e

        message = resp.choices[0].message
        text = message.content or ""

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                import json

                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": args,
                    }
                )

        usage = getattr(resp, "usage", None)
        ti = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        to = getattr(usage, "completion_tokens", 0) or 0 if usage else 0

        finish = getattr(resp.choices[0], "finish_reason", "stop") or "stop"

        return LlmResponse(
            text=text,
            tokens_input=ti,
            tokens_output=to,
            cost_micro=0,
            model=request.model,
            finish_reason=finish,
            metadata={},
            tool_calls=tuple(tool_calls),
        )

    def stream(self, request: LlmRequest, on_delta) -> LlmResponse:
        """Streaming text deltas via on_delta callback, return full
        LlmResponse at end of stream."""
        if not isinstance(request, LlmRequest):
            raise ProviderInvalidRequest("request must be LlmRequest")
        if not callable(on_delta):
            raise ProviderInvalidRequest("on_delta must be callable")

        try:
            if request.messages:
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in request.messages
                ]
                if request.system:
                    messages.insert(0, {"role": "system", "content": request.system})
            else:
                messages = []
                if request.system:
                    messages.append({"role": "system", "content": request.system})
                messages.append({"role": "user", "content": request.user})

            params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "drop_params": True,
                "stream": True,
                "timeout": self._timeout_s,
            }
            if self._api_key:
                params["api_key"] = self._api_key

            text_parts = []
            for chunk in litellm.completion(**params):
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    text_parts.append(delta.content)
                    try:
                        on_delta(delta.content)
                    except Exception:
                        pass

            text = "".join(text_parts)

        except Exception as e:
            raise ProviderUnavailable(
                f"LiteLLM streaming call failed: {type(e).__name__}: {e}"
            ) from e

        return LlmResponse(
            text=text,
            tokens_input=0,
            tokens_output=0,
            cost_micro=0,
            model=request.model,
            finish_reason="stop",
            metadata={},
            tool_calls=(),
        )


__all__ = ["LiteLLMProvider"]
