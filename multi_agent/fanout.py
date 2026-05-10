"""Auto-fanout for oversized tool outputs.

When a single tool result (Read on a 6.6 MB PDF, Grep over a huge tree, WebFetch
of a long article, …) returns more text than will fit in the active model's
context window, the cleanest user-visible behavior is *not* to surface a
"context too long" error. We instead split the result into chunks, dispatch
each chunk to a parallel sub-LLM call that extracts only the parts relevant to
the user's question, then merge the per-chunk summaries via a single reduce
call. The merged summary replaces the original tool result before it goes into
the conversation history.

This sits between tool execution and the conversation-history append in
agent.py, and is gated by a configurable threshold (default: tool output
> 40% of ctx window).

Distinct from compaction:
  • compaction shrinks the *conversation history* over many turns.
  • fanout shrinks a *single tool output* in one turn.

Compaction can't help here because the latest user/tool message is the
oversize one — compacting older history doesn't free room for it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable


# ── Decision: should this tool result be fanned out? ─────────────────────

# Tools whose outputs we are willing to summarize via sub-LLM. Excludes
# structured outputs (Task*, Memory*, etc.) where summarization would
# destroy machine-readable shape, and excludes already-cheap tools.
DEFAULT_FANOUT_TOOLS = frozenset({
    "Read", "ReadPDF", "ReadImage",
    "Grep", "Glob", "Bash",
    "WebFetch", "WebSearch",
    "SummarizeLargeFile",
})


def estimate_tokens_simple(text: str) -> int:
    """Same chars/2.8 + 1.1 multiplier as compaction.estimate_tokens, single
    string only — duplicated here to avoid circular imports."""
    if not text:
        return 0
    return int(len(text) / 2.8 * 1.1)


def should_fanout(tool_name: str, result, ctx_window: int, config: dict) -> bool:
    """Return True iff this tool result should be summarized via fanout.

    Triggers on:
      • auto_fanout_enabled is True (default)
      • tool name is in the fanout-eligible set
      • estimated tokens of result > ctx_window * threshold (default 0.4)
    """
    if not config.get("auto_fanout_enabled", True):
        return False
    if not isinstance(result, str) or not result:
        return False
    eligible = config.get("auto_fanout_tools") or DEFAULT_FANOUT_TOOLS
    if isinstance(eligible, (list, tuple, set, frozenset)):
        if tool_name not in eligible:
            return False
    threshold_pct = float(config.get("auto_fanout_threshold", 0.4))
    threshold_tokens = int(ctx_window * threshold_pct)
    return estimate_tokens_simple(result) > threshold_tokens


# ── Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str,
               max_chunk_tokens: int = 8000,
               overlap_tokens: int = 200) -> list[str]:
    """Split text into chunks each ≤ max_chunk_tokens, preferring paragraph
    boundaries, then sentences, then character-level as last resort. Carries
    `overlap_tokens` worth of trailing text from chunk N into the head of
    chunk N+1 so cross-boundary references survive.
    """
    if not text:
        return []
    # tokens → chars via the same 2.8 / 1.1 heuristic.
    max_chars = max(1024, int(max_chunk_tokens * 2.8 / 1.1))
    overlap_chars = max(0, int(overlap_tokens * 2.8 / 1.1))

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            piece = para
        else:
            piece = current + "\n\n" + para
        if len(piece) <= max_chars:
            current = piece
            continue
        # Flushing current; figure out what carries forward.
        if current:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            seed = (tail + "\n\n" + para) if tail else para
        else:
            seed = para
        # If the paragraph alone is bigger than max_chars, hard-split.
        while len(seed) > max_chars:
            chunks.append(seed[:max_chars])
            seed = seed[max_chars - overlap_chars:] if overlap_chars else seed[max_chars:]
        current = seed
    if current:
        chunks.append(current)
    return chunks


def coalesce_chunks(chunks: list[str], max_count: int) -> list[str]:
    """Reduce chunk count to ≤ max_count by merging adjacent chunks.

    When chunk_text produces more chunks than the configured max_subagents,
    we'd otherwise either drop chunks (lossy) or run more sub-calls than the
    user wants (slow / expensive). Coalescing adjacent ones into max_count
    groups preserves coverage at the cost of larger per-chunk size.
    """
    if not chunks or len(chunks) <= max_count:
        return chunks
    # Round-robin grouping by index, preserving order, then concatenate.
    groups: list[list[str]] = [[] for _ in range(max_count)]
    for i, c in enumerate(chunks):
        groups[i * max_count // len(chunks)].append(c)
    return ["\n\n".join(g) for g in groups if g]


# ── Map-reduce summarize ─────────────────────────────────────────────────

_MAP_SYSTEM = (
    "You are one of several parallel summarization workers. You see ONE chunk "
    "of a larger document along with the user's question. Extract ONLY the "
    "parts of THIS chunk relevant to that question. Be concise — your "
    "summary will be merged with summaries from other chunks. Quote specific "
    "numbers, names, and short verbatim snippets when they matter. If this "
    "chunk has nothing relevant, say so in one line. Stay under 400 words."
)

_REDUCE_SYSTEM = (
    "You are merging parallel chunk summaries into a single coherent answer. "
    "Each input was extracted from one chunk of a single source document. "
    "Combine them into a unified summary that answers the user's question. "
    "Preserve specific quotes and numbers. If chunk summaries conflict, note "
    "the conflict. If most chunks said 'nothing relevant', say so honestly "
    "rather than fabricating. Stay under 800 words."
)


def fanout_summarize(
    text: str,
    user_question: str,
    config: dict,
    llm_call: Callable[[str, str], str],
    ctx_window: int,
    max_subagents: int = 5,
) -> str:
    """Map-reduce summarize a large text into one that fits in ctx.

    Args:
        text: original tool result, may be very large.
        user_question: most recent user message (provides focus for extraction).
        config: agent config dict (read for chunk overlap, etc.).
        llm_call: pure (system_prompt, user_prompt) -> response_text. Caller
            injects this so we can stub it in tests and avoid coupling fanout
            to a specific provider here.
        ctx_window: model's total context window in tokens.
        max_subagents: cap on parallel chunk workers.

    Returns:
        merged summary string, suitable to substitute for the original tool
        result. Never raises — on internal failure it returns a truncated
        version of the input rather than blowing up the agent loop.
    """
    if not text:
        return text
    overlap = int(config.get("auto_fanout_chunk_overlap_tokens", 200))
    # Each chunk should leave ≥ ¾ of ctx for system+question+output overhead.
    target = max(2048, ctx_window // 4)
    chunks = chunk_text(text, max_chunk_tokens=target, overlap_tokens=overlap)
    if not chunks:
        return text
    chunks = coalesce_chunks(chunks, max_subagents)

    def _map_one(idx_chunk: tuple[int, str]) -> tuple[int, str]:
        i, chunk = idx_chunk
        prompt = (
            f"User question: {user_question or '(no explicit question; produce a general summary)'}\n\n"
            f"Document chunk {i+1} of {len(chunks)}:\n{chunk}\n\n"
            "Extract the relevant info from THIS chunk only."
        )
        try:
            return i, llm_call(_MAP_SYSTEM, prompt)
        except Exception as e:
            return i, f"[chunk {i+1} failed: {type(e).__name__}: {e}]"

    summaries: list[str] = [""] * len(chunks)
    with ThreadPoolExecutor(max_workers=min(len(chunks), max_subagents)) as pool:
        for i, summary in pool.map(_map_one, enumerate(chunks)):
            summaries[i] = summary

    reduce_user = f"User question: {user_question or '(no explicit question)'}\n\n"
    reduce_user += "Chunk summaries:\n\n"
    for i, s in enumerate(summaries):
        reduce_user += f"=== Chunk {i+1}/{len(chunks)} ===\n{s}\n\n"
    reduce_user += "Now produce the merged answer."

    try:
        merged = llm_call(_REDUCE_SYSTEM, reduce_user)
    except Exception:
        # Fallback: just concatenate per-chunk summaries with headers.
        merged = "\n\n".join(
            f"## Chunk {i+1}/{len(chunks)}\n{s}" for i, s in enumerate(summaries)
        )

    header = (
        f"[Auto-fanout summary: original tool output was "
        f"~{estimate_tokens_simple(text)} tokens, split into {len(chunks)} "
        f"parallel sub-summaries, merged below]\n\n"
    )
    return header + merged.strip()


# ── Provider-backed llm_call factory ─────────────────────────────────────

def make_llm_caller(config: dict) -> Callable[[str, str], str]:
    """Build a sync (system, user_text) -> response_text using the same
    provider/model as the parent agent. Tools are disabled for the sub-call
    so the response is plain text and one round-trip — no nested tool loops.
    """
    import providers

    model = config.get("model", "")
    provider = providers.detect_provider(model)

    def call(system: str, user: str) -> str:
        messages = [{"role": "user", "content": user}]
        # Disable tool injection on the sub-call: we want a plain summary.
        sub_config = dict(config)
        sub_config["no_tools"] = True
        # Lower max_tokens for the map step — chunk summaries are bounded by
        # the ≤400-word system prompt, so 2048 tokens is plenty and it keeps
        # latency predictable.
        sub_config["max_tokens"] = min(int(config.get("max_tokens", 8192) or 8192), 2048)

        if provider == "anthropic":
            api_key = providers.get_api_key("anthropic", config)
            stream = providers.stream_anthropic(
                api_key=api_key, model=model, system=system,
                messages=messages, tool_schemas=[], config=sub_config,
            )
        else:
            api_key = providers.get_api_key(provider, config)
            base_url = (
                providers.PROVIDERS.get(provider, {}).get("base_url")
                or config.get("custom_base_url", "")
            )
            stream = providers.stream_openai_compat(
                api_key=api_key, base_url=base_url, model=model,
                system=system, messages=messages, tool_schemas=[],
                config=sub_config,
            )
        text = ""
        for ev in stream:
            cls = ev.__class__.__name__
            if cls == "TextChunk":
                text += getattr(ev, "text", "")
            elif cls == "AssistantTurn":
                if not text:
                    text = getattr(ev, "text", "") or ""
        return text.strip()
    return call


# ── User-facing notification text ────────────────────────────────────────

def fanout_notice(tool_name: str, original_chars: int,
                  num_subagents: int, ctx_window: int) -> str:
    """One-line transparent notice shown to the user when fanout fires."""
    return (
        f"[Auto-fanout: {tool_name} returned ~{original_chars:,} chars "
        f"(>{int(ctx_window * 0.4):,} token threshold) → dispatching "
        f"{num_subagents} parallel sub-summaries to fit in {ctx_window:,}-tok "
        f"window]"
    )
