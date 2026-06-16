# Design Note: Exec streaming — line-by-line stdout/stderr

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0021-tool-dispatch.md`](./0021-tool-dispatch.md), [`0023-shell-exec-tool.md`](./0023-shell-exec-tool.md), [`0026-ipc-streaming.md`](./0026-ipc-streaming.md)

A long-running build/test command is the worst UX in the
agent: the user sees nothing for 30s, then the whole wall of
output. RFC 0026 gave us a streaming chunk channel; this RFC
lets tools — starting with Exec — emit per-line chunks during
execution, so the supervisor's ``on_chunk`` callback gets
``stdout``/``stderr`` lines as they arrive.

The integration is small but cuts cleanly across two layers:

- **ToolContext** gains an optional ``on_chunk(payload)``
  callable. Set by the supervisor when the wait()-time caller
  passed ``on_chunk``; tools that don't need streaming ignore
  it.
- **Exec tool** gains an opt-in ``stream=True`` arg. Default
  ``False`` keeps RFC 0023's buffered behaviour byte-for-byte.

## 1. Tool-side contract change

### `ToolContext.on_chunk`

```python
@dataclass(frozen=True)
class ToolContext:
    pid:      int
    kernel:   Optional["Kernel"]
    on_chunk: Optional[Callable[[dict], None]] = None    # NEW
```

Handlers that want to stream MAY call ``ctx.on_chunk(payload)``
where ``payload`` is a dict with at least ``op="chunk"``,
``kind`` (free-form string the tool chooses, e.g. ``"stdout"``
/ ``"stderr"``), ``content`` (the delta), and optional
``metadata``. Callbacks that raise are caught at the supervisor
boundary — a misbehaving callback can't crash the tool.

### `dispatch_tool_call(...)` accepts on_chunk

```python
def dispatch_tool_call(
    *, msg, pid, registry, kernel=None,
    on_chunk: Optional[Callable[[dict], None]] = None,
) -> dict:
```

Passed through into the constructed ``ToolContext``.

### Supervisor wires it up

In ``Supervisor._handle_tool_call`` and the ``wait()`` loop's
``tool_call`` branch, build a chunk-emitter that ALSO appends
to the local ``chunks`` list (so ``RunnerExitInfo.chunks``
stays the unified source of truth) and forwards to the user's
``on_chunk`` callback:

```python
def emit(payload):
    chunks.append(payload)
    if on_chunk is not None:
        try: on_chunk(payload)
        except Exception: pass

response = self._handle_tool_call(handle.pid, msg,
                                    on_chunk=emit)
```

## 2. Exec tool: opt-in streaming

### New arg

```python
{ "argv": [...], ..., "stream": True }    # default False
```

Validated as bool. Streaming is enabled iff
``args.stream is True`` AND ``ctx.on_chunk is not None`` —
otherwise the tool falls back to the existing buffered path
(zero behaviour change).

### Streaming path

Instead of ``run_sandboxed`` (which buffers via
``proc.communicate()``), the streaming path uses
``subprocess.Popen`` directly with two reader threads:

```
                  ┌───── stdout reader ─────┐
Popen ──pipe──►   │   readline → q.put()    │   ─┐
                  └─────────────────────────┘    ▼
                                         main: q.get() →
                                         ctx.on_chunk + accumulate
                                         ▲
                  ┌───── stderr reader ─────┐    │
Popen ──pipe──►   │   readline → q.put()    │   ─┘
                  └─────────────────────────┘
```

- One ``queue.Queue`` serializes lines from both pipes — keeps
  ``ctx.on_chunk`` calls single-threaded so user callbacks
  don't need to be thread-safe.
- The wall-clock killer (RFC 0008) still runs as a separate
  daemon thread.
- RLIMITs (CPU / AS / FSIZE / NOFILE) still applied via the
  same ``apply_rlimits_in_child`` preexec_fn.
- Output cap (``max_output_bytes``) still enforced; once the
  cap is hit, further bytes still emit chunks (so a UI sees
  progress) but are NOT accumulated into the final
  ``stdout`` / ``stderr`` strings — same truncation flag.

### Chunk shape

```python
{
    "op":       "chunk",
    "kind":     "stdout",   # or "stderr"
    "content":  "<one decoded line, including trailing newline>",
    "metadata": {"tool": "Exec"},
}
```

## 3. Backwards compatibility

- ``ToolContext`` adds a field with default ``None`` — existing
  ``ToolContext(pid=..., kernel=...)`` constructions still
  type-check.
- ``dispatch_tool_call`` adds a kw-only arg with default
  ``None`` — existing callers unchanged.
- ``Supervisor._handle_tool_call`` adds a kw-only arg with
  default ``None`` — existing call sites in
  ``Supervisor.wait`` continue to compile.
- Exec ``stream`` defaults False; output shape, exit_code,
  truncation flags, timed_out — all identical.
- ``RunnerExitInfo.chunks`` already includes
  runner-emitted chunks (RFC 0026); now also includes
  tool-emitted chunks. Tests that asserted ``chunks == ()``
  for non-streaming runs continue to hold (no tool emits
  chunks unless ``stream=True``).

## 4. Acceptance criteria

A PR claiming this RFC must:

1. ``ToolContext`` accepts ``on_chunk=None`` and round-trips
   to handlers.
2. ``dispatch_tool_call(..., on_chunk=cb)`` reaches the
   handler's ``ctx.on_chunk``.
3. Exec with ``stream=False`` (default) and a captured
   ``on_chunk`` callback emits ZERO chunks; output unchanged.
4. Exec with ``stream=True`` AND captured ``on_chunk`` emits
   one chunk per line of stdout AND one chunk per line of
   stderr.
5. Lines are emitted in arrival order; concatenating
   ``content`` of all stdout chunks reproduces the full
   stdout string.
6. Both ``stdout`` and ``stderr`` final fields in the result
   match what was streamed (modulo truncation).
7. Wall-clock timeout still fires under streaming; the result
   ``timed_out=True`` and the chunks accumulated up to the
   kill point are preserved.
8. RLIMIT enforcement still works under streaming (e.g. CPU
   limit is hit when the binary spins).
9. End-to-end via supervisor.wait(on_chunk=...): streaming
   Exec output reaches the wait()-level callback in real
   time (test asserts at least one chunk arrives before
   process exit).
10. No file outside ``kernel/``, ``tests/``, ``docs/RFC/``
    modified.
