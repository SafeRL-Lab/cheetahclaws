# Design Note: IPC streaming chunks

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0016-subprocess-agent-runner.md`](./0016-subprocess-agent-runner.md)

A 10-minute LLM response, a `git log` with 50 MB of output, a
multi-megabyte web fetch — each is currently invisible to the
caller until the runner ships its single ``exit`` message. This
RFC adds a streaming primitive so callers see incremental
output as it's produced.

The substrate is intentionally narrow:

- **One new IPC message kind**: ``chunk``. Emitted between
  ``iteration_start`` and ``exit``; carries an arbitrary
  payload.
- **`supervisor.wait` gets `on_chunk`**: callers can register
  a callback that fires on each chunk arrival. Synchronous,
  inline with the wait loop.
- **`RunnerExitInfo.chunks`**: a tuple of all received chunks,
  populated for after-the-fact inspection.
- **No protocol breakage**: runners that don't emit ``chunk``
  messages behave exactly as before. ``chunks`` defaults to
  empty tuple.

What's **not** in this RFC:

- LLM runner integration with anthropic streaming — separate
  follow-up. Substrate ships first.
- Exec / Fetch streaming output — same.
- Caller-driven cancellation mid-stream — already covered by
  ``stop()``.
- Chunk acknowledgement / backpressure — chunks are fire-and-
  forget. The receiver's callback has to be quick or the
  runner backs up at the pipe.

## 1. IPC message shape

```jsonc
{
  "op":      "chunk",
  "kind":    "text" | "tool_output" | "log" | <custom>,
  "content": "<the chunk payload — usually a string>",
  "metadata": { /* opaque */ }
}
```

``kind`` is a free-form classifier so callers can route
chunks (e.g., a UI surface separates "text" from "tool_output").
``content`` is typically a UTF-8 string but binary callers can
base64-encode and use a custom ``kind``.

The runner can emit zero or many ``chunk`` messages in any
iteration. They MUST appear before the iteration's
``iteration_done`` (or ``exit`` for runners that don't track
iterations). The supervisor doesn't enforce ordering — chunks
arrive in send order regardless of message kind.

## 2. Supervisor surface

```python
def wait(self, pid: int, *,
         timeout: float | None = None,
         on_chunk: Callable[[dict], None] | None = None,
         ) -> RunnerExitInfo:
    ...
```

When ``on_chunk`` is supplied, the supervisor calls it for
each ``chunk`` IPC message received. The callable is invoked
synchronously inside the wait loop — slow callbacks slow the
drain. Callers that need responsive UIs should hand off to a
queue inside the callback and process elsewhere.

Callback exceptions are caught and dropped (logged, not
reraised) — a bad UI thread shouldn't kill the runner.

```python
@dataclass(frozen=True)
class RunnerExitInfo:
    ...
    chunks: tuple = ()      # NEW
```

``chunks`` is the full sequence of received chunks (each a
dict in the wire shape). Populated even when ``on_chunk`` was
also provided — the same chunks appear in both.

## 3. runner_main testing path

A new ``CC_RUNNER_BEHAVIOR=chunks=N`` value emits ``N`` text
chunks (each ``"chunk-i"`` for i = 1..N), then exits cleanly.
This drives the supervisor end-to-end without needing an LLM
or real streaming source.

## 4. Backwards compatibility

- ``RunnerExitInfo.chunks`` defaults to ``()``. Existing
  fields and tests are unchanged.
- ``supervisor.wait`` ``on_chunk`` parameter is optional with
  default None. Existing callers see no change.
- Runners that don't emit ``chunk`` messages receive
  ``chunks=()`` in their info — same as before.

## 5. Acceptance criteria

A PR claiming this RFC must:

1. ``op="chunk"`` IPC message is recognised by the supervisor;
   sent to ``on_chunk`` if provided; appended to
   ``RunnerExitInfo.chunks``.
2. Order is preserved: callback fires in send order; tuple
   reflects same.
3. Runner emitting 5 chunks → callback fires 5 times → tuple
   has 5 entries.
4. Runner emitting no chunks → ``chunks == ()``.
5. Callback raising → next chunks still delivered + tuple
   still appended.
6. Existing tests with no chunks involvement keep passing.
7. No file outside ``kernel/``, ``tests/``, ``docs/RFC/``
   modified.
