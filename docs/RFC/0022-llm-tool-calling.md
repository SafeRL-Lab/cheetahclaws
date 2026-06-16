# Design Note: LLM tool calling — closing the loop

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0019-llm-runner.md`](./0019-llm-runner.md), [`0021-tool-dispatch.md`](./0021-tool-dispatch.md)

RFC 0019 gave us LLM calls. RFC 0021 gave us tool dispatch. This
RFC closes the loop: an LLM-driven runner that can **decide to
call a tool**, dispatch through the supervisor, get the result,
and continue the conversation — autonomously, in a single
subprocess.

This is the difference between "LLM chat wrapper" and "agent".

The scope is intentionally narrow:

- **Function-calling style only.** The runner accepts a `tools=`
  list, the provider returns `tool_calls`, the runner dispatches
  via IPC. No "ReAct" prompt parsing, no XML tag extraction.
- **Anthropic-shaped JSON.** Provider-native tool format mirrors
  Anthropic's `messages.tools=[...]` schema. The mock provider is
  pass-through; the anthropic adapter converts the response back
  into our shape. Other providers (OpenAI, Gemini) need their own
  adapters but the runner stays unchanged.
- **Bounded iterations.** A max_iterations cap (default 10) stops
  runaway loops. Hitting the cap surfaces as
  `exit_kind="failed"` with `error="max_iterations"`.
- **No streaming.** Each LLM call is whole-response. Streaming
  tool output is a future RFC.

The substrate for everything else (capability checks, fs gates,
audit) was built in RFC 0021 and works unchanged — the runner
just emits `tool_call` IPC messages and the supervisor dispatches
exactly as it does for the test runner.

## 1. Scope

**Goals:**

1. **`LlmRequest.tools`** — list of provider-native tool
   definitions. Pass-through to the provider.
2. **`LlmResponse.tool_calls`** — list of structured tool-call
   requests when the model decides to invoke. Empty list = pure
   text response.
3. **Iteration loop** in the LLM runner: while the response has
   tool_calls and the cap isn't hit, dispatch each call via IPC
   tool_call, append the result, call the provider again.
4. **`ScriptedMockProvider`** for deterministic multi-step tests
   without real model APIs.
5. **Anthropic adapter** converts: our `tools=[...]` →
   `tools=[...]` API field; Anthropic's `tool_use` content blocks
   → our `tool_calls`. The user-supplied tool descriptions are
   pass-through.

**Non-goals:**

- **Streaming** — separate RFC.
- **Parallel tool calls in one turn** — supported in shape (the
  field is a list) but the runner dispatches sequentially.
- **OpenAI / Gemini adapters** — substrate works for any
  provider; the adapters themselves are follow-up patches.

## 2. Data model

### `LlmRequest.tools`

```python
tools: tuple = ()      # opaque list of dicts in provider-native format
```

For Anthropic, the dicts look like:

```python
{
    "name": "Read",
    "description": "Read a file from host filesystem.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}
```

The runner does not validate or canonicalise — the provider does.

### `LlmResponse.tool_calls`

```python
tool_calls: tuple = ()
```

Each entry is a dict with our canonical shape:

```python
{
    "id":    "<unique-id>",       # provider-supplied
    "name":  "<tool name>",
    "input": {...},               # arguments
}
```

Note this is the **canonical shape** the runner uses internally.
Each provider adapter converts from its native format.

`LlmResponse.is_tool_use` property:

```python
@property
def is_tool_use(self) -> bool:
    return bool(self.tool_calls)
```

### `ScriptedMockProvider`

```python
ScriptedMockProvider(responses: list[LlmResponse])
```

Returns `responses[0]` on first call, `responses[1]` on second,
etc. Raises `ProviderUnavailable` if exhausted. Useful for
testing multi-iteration flows without needing real LLM behaviour.

`ScriptedMockProvider.from_env()` reads
`CC_LLM_SCRIPTED_RESPONSES_JSON` env var (a JSON list of
LlmResponse dicts).

## 3. Runner iteration loop

```
1. Read init payload, build LlmRequest with messages + tools.
2. messages = list(init.messages)
3. for iter in 1..max_iterations:
4.    request = LlmRequest(messages=messages, tools=tools, ...)
5.    response = provider(request)
6.    emit charge messages for tokens / cost
7.    if not response.is_tool_use:
8.        emit exit (completed, text=response.text, ...)
9.        return 0
10.   # Tool calls present.
11.   assistant_msg = {"role": "assistant", "content": [
12.       {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
13.       for tc in response.tool_calls
14.   ]}
15.   messages.append(assistant_msg)
16.   tool_results = []
17.   for tc in response.tool_calls:
18.       send IPC tool_call(tc.id, tc.name, tc.input)
19.       resp = recv IPC tool_response (timeout)
20.       result_str = format result or error
21.       tool_results.append({
22.           "type": "tool_result",
23.           "tool_use_id": tc.id,
24.           "content": result_str,
25.       })
26.   user_msg = {"role": "user", "content": tool_results}
27.   messages.append(user_msg)
28. # Cap hit.
29. emit exit (failed, error="max_iterations")
30. return 1
```

The runner emits a `log` message on each iteration; the
supervisor doesn't otherwise see what happened inside.

## 4. Anthropic adapter

When `request.tools` is non-empty, the adapter:

1. Passes `tools=[{...same shape...}]` to
   `client.messages.create()`.
2. After response, scans `response.content` blocks; non-text
   blocks of type `tool_use` are converted to:
   ```python
   {"id": block.id, "name": block.name, "input": block.input}
   ```
3. Text blocks stay as `text` field on the LlmResponse.
4. `finish_reason` is `tool_use` when there are tool calls,
   `stop` for plain text.

### Multi-turn tool result format

When the runner sends a follow-up turn with tool results, the
adapter must emit content blocks in Anthropic's expected shape:

```python
{
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "...", "content": "..."}
    ],
}
```

Our ``messages`` list already carries this shape (the runner
builds it). The adapter passes it through unchanged — no extra
conversion needed.

## 5. Backwards compatibility

- `LlmRequest.tools` defaults to `()`; existing single-turn
  callers see no change.
- `LlmResponse.tool_calls` defaults to `()`; existing pure-text
  responses are unaffected.
- The runner's behaviour when `tools=()` is identical to RFC
  0019: one provider call, emit charges, exit. The iteration
  loop short-circuits on the first iteration when
  `response.is_tool_use` is False.
- ScriptedMockProvider is a new class — no impact on existing
  MockProvider users.

## 6. Failure modes

| Cause | Runner behaviour | exit_kind |
|---|---|---|
| Provider returns text on first iter | normal flow, exit completed | completed |
| Provider returns tool_use, supervisor dispatches OK, next iter returns text | normal flow | completed |
| max_iterations exceeded | emit exit with error="max_iterations" | failed |
| tool_response.ok=false | result included in messages as error string; loop continues (model can decide to recover) | depends on later iters |
| ScriptedMockProvider exhausted | ProviderUnavailable → exit failed | failed |

## 7. Acceptance criteria

A PR claiming this RFC must:

1. `LlmRequest(tools=[...])` round-trips via to_dict/from_dict.
2. `LlmResponse(tool_calls=[...])` round-trips; is_tool_use
   reflects.
3. ScriptedMockProvider returns responses in order; exhaustion
   raises.
4. Runner with scripted provider [tool_use(Echo), text("done")]:
   - Iteration 1: tool_use → IPC tool_call(Echo) → response.
   - Iteration 2: text → exit completed with the final text.
   - Charges accumulate across iterations.
5. Runner with scripted provider that always returns tool_use
   hits max_iterations → exit failed.
6. The supervisor's tool dispatch invariants from RFC 0021
   still apply (cap + fs check, audit events).
7. No file outside `kernel/`, `tests/`, `docs/RFC/`
   modified.
