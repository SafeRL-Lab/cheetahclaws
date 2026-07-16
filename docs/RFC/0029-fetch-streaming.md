# Design Note: Fetch streaming — incremental HTTP body

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0025-fetch-tool.md`](./0025-fetch-tool.md), [`0026-ipc-streaming.md`](./0026-ipc-streaming.md), [`0028-exec-streaming.md`](./0028-exec-streaming.md)

A 4 MB log file streamed across the wire shouldn't sit in
buffer for 30 seconds. RFC 0026 gave us the chunk channel,
RFC 0028 plugged tool handlers into it; this one wires the
Fetch tool's response-body read loop to emit chunks per
8 KB block.

The integration is small:

- ``FetchRequest.stream: bool = False`` — opt-in like Exec.
- The existing `_read_capped` body-drain loop already reads
  in 8 KB chunks (``resp.read(8192)``); we just emit each
  one via ``ctx.on_chunk`` if streaming is enabled.
- All SSRF / DNS-rebind / auth-header-strip / max_bytes /
  timeout safety properties from RFC 0025 are preserved
  bit-for-bit.

## 1. Args

```python
{
  "url": "https://...", ..., "stream": True   # default False
}
```

Validated as bool. Streaming is enabled iff
``args.stream is True`` AND ``ctx.on_chunk is not None``.

## 2. Chunk shape

Per ``resp.read(8192)`` block:

```python
{
    "op":       "chunk",
    "kind":     "body",
    "content":  "<8KB decoded utf-8 with errors='replace'>",
    "metadata": {
        "tool":         "Fetch",
        "url":          "<current url>",
        "bytes_so_far": <int>,
        "status":       <int>,
    },
}
```

- ``content`` is utf-8-best-effort decoded. Multi-byte
  characters that straddle chunk boundaries are decoded as
  the U+FFFD replacement char in one chunk and recovered in
  the next; for log/JSONL/SSE workloads this is the right
  tradeoff. Consumers who need exact bytes can read the
  final ``body`` field.
- Chunks past the ``max_bytes`` cap stop emitting once the
  cap is hit (truncation flag set, body trimmed).
- Redirect intermediate responses do NOT stream — only the
  terminal response body.

## 3. Backwards compatibility

- ``stream`` defaults False — existing callers see no change.
- ``on_chunk`` is None for legacy callers (no
  ToolContext.on_chunk plumbed); streaming is silently
  disabled — buffered behaviour identical to RFC 0025.
- All RFC 0025 safety properties (SSRF, DNS-rebind,
  redirect auth-strip, max_bytes, timeout) preserved.
- Final result dict shape (``url``, ``status``, ``headers``,
  ``body``, ``encoding``, ``body_bytes``, ``truncated``,
  ``redirects``, ``duration_s``) unchanged.

## 4. Acceptance criteria

A PR claiming this RFC must:

1. ``Fetch`` accepts ``stream: bool``; non-bool raises
   ``invalid_args``.
2. ``stream=False`` (default) emits zero chunks even with
   ``on_chunk`` plumbed.
3. ``stream=True`` + ``ctx.on_chunk`` set emits one chunk
   per 8 KB block of body read.
4. Chunks have shape ``{op:"chunk", kind:"body",
   content:str, metadata:{tool:"Fetch", url, bytes_so_far,
   status}}``.
5. Concatenated chunk content reproduces the body string
   (modulo multi-byte boundary replacement chars).
6. Redirect intermediate bodies do NOT stream; only the
   terminal hop streams.
7. Truncation: once max_bytes is reached, no further chunks
   are emitted; ``truncated=True`` in result.
8. Bad ``on_chunk`` callback (raises) doesn't break the
   fetch — exception swallowed at the boundary.
9. No file outside ``kernel/``, ``tests/``,
   ``docs/RFC/`` modified.
