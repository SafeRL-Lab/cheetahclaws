# Design Note: LLM streaming — token-by-token output

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0019-llm-runner.md`](./0019-llm-runner.md), [`0022-llm-tool-calling.md`](./0022-llm-tool-calling.md), [`0026-ipc-streaming.md`](./0026-ipc-streaming.md)

RFC 0026 gave the IPC substrate for streaming chunks. This RFC
plugs the LLM runner into it: when ``stream=True`` is set in the
init payload, the provider's text deltas flow as IPC ``chunk``
messages to the supervisor's ``on_chunk`` callback in real time.
A 30-second LLM response now feels real-time instead of a 30-
second wait.

The design is minimal:

- **Optional**. ``stream=False`` (default) keeps the existing
  RFC 0019 single-shot path. No behaviour change for current
  callers.
- **Provider opt-in**. Providers that want to support streaming
  add a ``stream(request, on_delta)`` method. Providers without
  it are silently used non-streaming when stream=True is
  requested — graceful degradation, no error.
- **Text deltas only**. Tool-use blocks come as one piece. A
  future RFC may stream tool-use input deltas, but Anthropic's
  SDK already returns them whole.
- **Multi-iteration aware**. In a tool-calling loop (RFC 0022),
  streaming applies only to iterations that produce text
  output. Tool-use iterations run non-streaming.

## 1. Data model

### `LlmRequest.stream`

```python
stream: bool = False
```

Validated in `__post_init__` (must be bool). Round-trips via
`to_dict` / `from_dict`.

### Provider protocol

Providers MAY implement:

```python
def stream(self, request: LlmRequest,
           on_delta: Callable[[str], None]) -> LlmResponse:
    """Stream text deltas via on_delta, return final LlmResponse."""
```

`on_delta` is called for each chunk of text content as it
arrives. The final returned `LlmResponse` has the full text +
token counts + finish_reason.

A provider without ``stream()`` falls back to non-streaming
``__call__`` — no chunks are emitted but the call still works.

## 2. Provider implementations

### ScriptedMockProvider.stream

For deterministic tests. Emits the response's `text` one
character at a time via `on_delta`, then returns the full
response (matching the next entry in its scripted list).

```python
def stream(self, request, on_delta):
    response = self(request)             # advances cursor
    for ch in response.text:
        on_delta(ch)
    return response
```

### AnthropicProvider.stream

Uses ``client.messages.stream()`` context manager. The SDK's
``text_stream`` iterator yields incremental text. We pump each
delta to ``on_delta`` and assemble the final response from the
final message.

```python
def stream(self, request, on_delta):
    self._ensure_client()
    kwargs = self._build_kwargs(request)
    with self._client.messages.stream(**kwargs) as stream:
        for delta in stream.text_stream:
            on_delta(delta)
        final = stream.get_final_message()
    return self._convert(final, request.model)
```

(Tool-use blocks in the streamed response stay whole — Anthropic
emits them as a finished `tool_use` block at the end.)

## 3. LLM runner integration

In ``kernel/runner/llm/__main__.py``:

```python
stream = bool(payload.get("stream", False))
provider_supports_stream = hasattr(provider, "stream") and \
                            callable(provider.stream)

# In the iteration loop:
if stream and provider_supports_stream:
    def on_delta(text: str) -> None:
        chan.send({
            "op":      "chunk",
            "kind":    "text",
            "content": text,
            "metadata": {"iter": it},
        })
    response = provider.stream(request, on_delta)
else:
    response = provider(request)
```

The chunk's ``metadata.iter`` lets a UI distinguish text from
different iterations of a tool-calling loop.

## 4. Backwards compatibility

- ``LlmRequest.stream`` defaults to False → existing single-turn
  and tool-calling tests are unchanged.
- Providers without ``stream()`` are silently used
  non-streaming → no breakage of existing test mocks.
- ``RunnerExitInfo.text`` still reflects the full text
  regardless of streaming — same final output, just produced
  incrementally.
- ``info.chunks`` (RFC 0026) populates with per-delta entries
  when streaming.

## 5. Acceptance criteria

A PR claiming this RFC must:

1. ``LlmRequest(stream=True).to_dict()`` round-trips.
2. ``ScriptedMockProvider.stream("hello", ...)`` calls
   on_delta with each character: 'h', 'e', 'l', 'l', 'o'.
3. LLM runner with ``stream=True`` + scripted "hi" → 2 IPC
   chunk messages with content 'h', 'i' before the exit.
4. Supervisor's ``on_chunk`` callback receives them in order.
5. ``stream=False`` (default) sends NO chunk messages.
6. Provider without stream() method + stream=True still works
   (non-streaming fallback, no chunks).
7. Multi-iteration tool calling: tool_use iteration emits 0
   chunks; final text iteration emits per-delta chunks.
8. ``info.text`` matches the assembled deltas (same text either
   way).
9. No file outside ``kernel/``, ``tests/``, ``docs/RFC/``
   modified.
