"""Subprocess entry point for the LLM runner with tool calling
(RFC 0019 + RFC 0022).

Single-turn (RFC 0019) AND function-calling multi-iteration
(RFC 0022) flows go through this entry point. Selection happens
naturally:

  - tools=() in the init payload → first response is text →
    one provider call, exit.
  - tools=[...] in the init payload → if the response has
    tool_calls, dispatch via IPC tool_call → tool_response,
    append, call again. Repeat until text response or
    max_iterations.

Provider selection via env var ``CC_LLM_PROVIDER``:

  mock      — MockProvider.from_env()    (CC_LLM_MOCK_RESPONSE_JSON)
  scripted  — ScriptedMockProvider.from_env()  (CC_LLM_SCRIPTED_RESPONSES_JSON)
  anthropic — AnthropicProvider()        (ANTHROPIC_API_KEY)
  litellm   — LiteLLMProvider()          (provider-specific env vars,
                                          or CC_LLM_API_KEY override)
"""
from __future__ import annotations

import json
import os
import sys

from ..ipc import JsonLineChannel
from .provider import (
    LlmRequest,
    LlmResponse,
    MockProvider,
    Provider,
    ProviderInvalidRequest,
    ProviderUnavailable,
    ScriptedMockProvider,
)


DEFAULT_MAX_ITERATIONS = 10
DEFAULT_TOOL_RESPONSE_TIMEOUT_S = 60.0


def _select_provider() -> Provider:
    name = os.environ.get("CC_LLM_PROVIDER", "")
    if not name:
        raise ProviderUnavailable(
            "CC_LLM_PROVIDER env var not set "
            "(use 'mock', 'scripted', 'anthropic', or 'litellm')",
        )
    if name == "mock":
        return MockProvider.from_env()
    if name == "scripted":
        return ScriptedMockProvider.from_env()
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if name == "litellm":
        from .litellm_provider import LiteLLMProvider
        # litellm reads provider-specific keys (OPENAI_API_KEY,
        # ANTHROPIC_API_KEY, AZURE_API_KEY, ...) from env on its own.
        # CC_LLM_API_KEY is an optional explicit override.
        return LiteLLMProvider(api_key=os.environ.get("CC_LLM_API_KEY"))
    raise ProviderUnavailable(f"unknown CC_LLM_PROVIDER: {name!r}")


def _format_tool_result(response: dict) -> str:
    """Convert a supervisor's tool_response payload to the string
    form Anthropic expects in tool_result blocks.

    On success: serialise result dict as JSON.
    On failure: a short error message including the slug.
    """
    if response.get("ok"):
        try:
            return json.dumps(response.get("result") or {},
                              ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(response.get("result"))
    err = response.get("error", "unknown_error")
    msg = response.get("message", "")
    return f"ERROR: {err}: {msg}"


def main() -> int:
    chan = JsonLineChannel(sys.stdin.buffer, sys.stdout.buffer)

    # 1) Read init.
    try:
        init = chan.recv(timeout=10.0)
    except Exception as e:
        sys.stderr.write(f"llm-runner: init recv failed: {e}\n")
        return 2
    if init.get("op") != "init":
        sys.stderr.write(f"llm-runner: expected init, got {init!r}\n")
        return 2

    pid = init.get("pid")
    payload = init.get("payload") or {}
    max_iters = int(payload.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    tool_response_timeout_s = float(
        payload.get("tool_response_timeout_s",
                     DEFAULT_TOOL_RESPONSE_TIMEOUT_S),
    )

    # 2) Select provider.
    try:
        provider = _select_provider()
    except ProviderUnavailable as e:
        sys.stderr.write(f"llm-runner: {e}\n")
        return 2

    # 3) Send ready.
    chan.send({"op": "ready", "pid": pid})

    # 4) Build the working messages list.
    if "messages" in payload and payload["messages"]:
        messages = list(payload["messages"])
    elif "user" in payload and payload["user"]:
        messages = [{"role": "user", "content": payload["user"]}]
    else:
        chan.send({"op": "log", "level": "error",
                   "msg": "init payload missing 'messages' or 'user'"})
        chan.send({"op": "exit", "exit_kind": "failed",
                   "summary": "no messages in payload"})
        return 1

    tools = list(payload.get("tools") or ())
    system = str(payload.get("system", ""))
    model = payload.get("model", "")
    if not model:
        chan.send({"op": "log", "level": "error",
                   "msg": "init payload missing 'model'"})
        chan.send({"op": "exit", "exit_kind": "failed",
                   "summary": "no model in payload"})
        return 1

    total_tokens = 0
    total_cost_micro = 0
    last_response: LlmResponse | None = None
    final_text = ""
    iterations_used = 0

    # 5) Iteration loop.
    stream_requested = bool(payload.get("stream", False))
    provider_supports_stream = (
        hasattr(provider, "stream") and callable(getattr(provider, "stream"))
    )
    for it in range(1, max_iters + 1):
        iterations_used = it
        chan.send({"op": "iteration_start", "iter": it})
        try:
            request = LlmRequest(
                model=str(model),
                system=system,
                messages=tuple(messages),
                tools=tuple(tools),
                stream=stream_requested,
                max_tokens=int(payload.get("max_tokens", 1024)),
                temperature=float(payload.get("temperature", 0.7)),
            )
        except ProviderInvalidRequest as e:
            chan.send({"op": "log", "level": "error",
                       "msg": f"invalid request: {e}"})
            chan.send({"op": "exit", "exit_kind": "failed",
                       "summary": f"invalid request: {e}",
                       "metadata": {"iterations": iterations_used}})
            return 1

        try:
            if stream_requested and provider_supports_stream:
                # RFC 0027 streaming path. emit each text delta as
                # an IPC chunk message; the final response from
                # provider.stream() carries the assembled text +
                # token counts + tool_calls (if any).
                def _on_delta(delta: str, _it=it) -> None:
                    chan.send({
                        "op":       "chunk",
                        "kind":     "text",
                        "content":  delta,
                        "metadata": {"iter": _it},
                    })
                response = provider.stream(request, _on_delta)
            else:
                response = provider(request)
        except ProviderInvalidRequest as e:
            chan.send({"op": "log", "level": "error",
                       "msg": f"provider rejected request: {e}"})
            chan.send({"op": "exit", "exit_kind": "failed",
                       "summary": f"invalid request: {e}",
                       "metadata": {"iterations": iterations_used}})
            return 1
        except ProviderUnavailable as e:
            chan.send({"op": "log", "level": "error",
                       "msg": f"provider unavailable: {e}"})
            chan.send({"op": "exit", "exit_kind": "failed",
                       "summary": f"provider unavailable: {e}",
                       "metadata": {"iterations": iterations_used}})
            return 1
        except Exception as e:
            chan.send({"op": "log", "level": "error",
                       "msg": f"unexpected provider error: "
                              f"{type(e).__name__}: {e}"})
            chan.send({"op": "exit", "exit_kind": "failed",
                       "summary": f"unexpected error: {type(e).__name__}",
                       "metadata": {"iterations": iterations_used}})
            return 1

        last_response = response
        total_tokens     += response.tokens_total
        total_cost_micro += response.cost_micro

        # Emit per-iteration charges. Supervisor's auto-charge from
        # iteration_done is suppressed by zeroing those amounts (RFC
        # 0019 §4).
        if response.tokens_total > 0:
            chan.send({"op": "charge", "dim": "tokens",
                       "amount": response.tokens_total})
        if response.cost_micro > 0:
            chan.send({"op": "charge", "dim": "cost_micro",
                       "amount": response.cost_micro})
        chan.send({"op": "iteration_done", "iter": it,
                   "tokens": 0, "cost_micro": 0})

        # If the response has no tool calls, this is the final
        # answer. Exit.
        if not response.is_tool_use:
            final_text = response.text
            break

        # Tool use path. Append the assistant message in
        # multi-content shape, dispatch each call, append the
        # tool_result message, loop.
        chan.send({"op": "log", "level": "info",
                   "msg": f"iter {it}: {len(response.tool_calls)} "
                          f"tool_call(s)"})
        assistant_content = []
        # If the response also contains text alongside tool_use
        # (Anthropic does this — model thinks aloud before calling),
        # preserve it.
        if response.text:
            assistant_content.append({
                "type": "text", "text": response.text,
            })
        for tc in response.tool_calls:
            assistant_content.append({
                "type":  "tool_use",
                "id":    tc["id"],
                "name":  tc["name"],
                "input": tc.get("input") or {},
            })
        messages.append({
            "role":    "assistant",
            "content": assistant_content,
        })

        # Dispatch each tool call via IPC and collect results.
        tool_results = []
        any_dispatch_error = False
        for tc in response.tool_calls:
            chan.send({
                "op":           "tool_call",
                "tool_call_id": tc["id"],
                "tool":         tc["name"],
                "args":         tc.get("input") or {},
            })
            try:
                tool_resp = chan.recv(timeout=tool_response_timeout_s)
            except Exception as e:
                # Supervisor never responded; abort this iteration.
                chan.send({"op": "log", "level": "error",
                           "msg": f"tool_response recv failed: {e}"})
                any_dispatch_error = True
                tool_results.append({
                    "type":         "tool_result",
                    "tool_use_id":  tc["id"],
                    "content":      f"ERROR: ipc_failed: {e}",
                })
                continue
            tool_results.append({
                "type":         "tool_result",
                "tool_use_id":  tc["id"],
                "content":      _format_tool_result(tool_resp),
            })
            chan.send({"op": "log", "level": "info",
                       "msg": f"tool_response {tc['name']} ok={tool_resp.get('ok')}"})

        messages.append({
            "role":    "user",
            "content": tool_results,
        })

        if any_dispatch_error:
            # Continue to next iteration anyway so the model can
            # incorporate the error into its plan.
            pass

    # 6) Did we exit the loop with a final text answer, or hit the cap?
    if last_response is None or last_response.is_tool_use:
        # Cap hit — last response still wanted to call tools.
        chan.send({"op": "log", "level": "error",
                   "msg": f"max_iterations={max_iters} reached"})
        chan.send({
            "op":         "exit",
            "exit_kind":  "failed",
            "summary":    f"max_iterations={max_iters}",
            "metadata":   {
                "iterations":   iterations_used,
                "tokens_total": total_tokens,
                "cost_micro":   total_cost_micro,
                "error":        "max_iterations",
            },
        })
        return 1

    # 7) Emit final exit.
    summary = final_text
    if len(summary) > 500:
        summary = summary[:500] + "..."
    chan.send({"op": "log", "level": "info",
               "msg": f"completed in {iterations_used} iter(s), "
                      f"tokens={total_tokens}, cost_micro={total_cost_micro}"})
    chan.send({
        "op":         "exit",
        "exit_kind":  "completed",
        "summary":    summary,
        "text":       final_text,
        "metadata":   {
            "finish_reason":  last_response.finish_reason,
            # Per-call counts come from the LAST iteration only
            # (RFC 0019 single-turn callers expect these to be
            # the response's own counts). Aggregate counts for
            # multi-iteration callers go in tokens_total /
            # cost_micro.
            "tokens_input":   last_response.tokens_input,
            "tokens_output":  last_response.tokens_output,
            "tokens_total":   total_tokens,
            "cost_micro":     total_cost_micro,
            "model":          last_response.model,
            "iterations":     iterations_used,
        },
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
