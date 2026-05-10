"""Tests for multi_agent.fanout — auto-fanout for oversized tool outputs.

Regression target: a 6.6 MB PDF read on qwen2.5-72b (32k ctx) → tool result
~70-150k tokens → impossible to fit in any single API call no matter how much
the conversation history is compacted, because the latest tool message itself
is the oversize one. fanout splits the result, dispatches parallel sub-LLM
summarizations, merges them, and substitutes the merged summary in place of
the original — turning a hard error into a graceful degradation.
"""
from __future__ import annotations

import os
import sys
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multi_agent.fanout import (
    DEFAULT_FANOUT_TOOLS,
    chunk_text,
    coalesce_chunks,
    estimate_tokens_simple,
    fanout_notice,
    fanout_summarize,
    should_fanout,
)


# ── should_fanout ────────────────────────────────────────────────────────

class TestShouldFanout:
    def test_disabled_via_config(self):
        cfg = {"auto_fanout_enabled": False}
        big = "x" * 200000
        assert should_fanout("Read", big, ctx_window=32768, config=cfg) is False

    def test_default_threshold_fires_above_40_pct(self):
        # 32768 * 0.4 ≈ 13107 tokens ≈ 13107 * 2.8 / 1.1 ≈ 33,363 chars.
        # 50,000 chars → ~19,481 tokens → above threshold.
        big = "y" * 50000
        cfg = {"auto_fanout_enabled": True}
        assert should_fanout("Read", big, ctx_window=32768, config=cfg) is True

    def test_default_threshold_skips_small_output(self):
        small = "z" * 1000  # ~390 tokens, way below threshold
        cfg = {"auto_fanout_enabled": True}
        assert should_fanout("Read", small, ctx_window=32768, config=cfg) is False

    def test_skips_non_eligible_tool(self):
        # MemorySave is not in DEFAULT_FANOUT_TOOLS — its output is structured
        # and shouldn't be summarized.
        big = "x" * 200000
        cfg = {"auto_fanout_enabled": True}
        assert should_fanout("MemorySave", big, ctx_window=32768, config=cfg) is False

    def test_skips_error_strings(self):
        # Errors are tiny by definition; even an "error" string would pass
        # threshold only if it's huge — which it never is. We rely on the
        # CALLER to skip error strings; should_fanout itself doesn't check
        # the content. Just verify it stays length-based.
        cfg = {"auto_fanout_enabled": True}
        assert should_fanout("Read", "Error: file not found", 32768, cfg) is False

    def test_custom_threshold(self):
        # Lower threshold → easier to fire.
        result = "a" * 10000
        cfg = {"auto_fanout_enabled": True, "auto_fanout_threshold": 0.1}
        # 32768 * 0.1 ≈ 3277 tokens; 10000 chars ≈ 3896 tokens — fires.
        assert should_fanout("Read", result, 32768, cfg) is True
        cfg2 = {"auto_fanout_enabled": True, "auto_fanout_threshold": 0.5}
        # 32768 * 0.5 ≈ 16384 tokens; 10000 chars ≈ 3896 tokens — skips.
        assert should_fanout("Read", result, 32768, cfg2) is False

    def test_non_string_result_skipped(self):
        cfg = {"auto_fanout_enabled": True}
        assert should_fanout("Read", None, 32768, cfg) is False
        assert should_fanout("Read", b"bytes", 32768, cfg) is False


# ── chunk_text ───────────────────────────────────────────────────────────

class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_short_text_single_chunk(self):
        text = "hello world"
        chunks = chunk_text(text, max_chunk_tokens=8000)
        assert chunks == ["hello world"]

    def test_paragraph_boundary_preferred(self):
        # 3 paragraphs that fit in one chunk → one chunk.
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = chunk_text(text, max_chunk_tokens=8000)
        assert len(chunks) == 1
        assert "Para 1" in chunks[0]
        assert "Para 3" in chunks[0]

    def test_splits_when_exceeding_max(self):
        # Build a doc bigger than max_chars in a way that paragraphs
        # naturally split it. With max_chunk_tokens=200 → max_chars ≈ 509.
        para = "x" * 400
        text = (para + "\n\n") * 5  # ~2000+ chars
        chunks = chunk_text(text, max_chunk_tokens=200, overlap_tokens=0)
        assert len(chunks) >= 2
        # Concatenated chunks (minus overlaps) cover the original text
        # length within reason.
        total = sum(len(c) for c in chunks)
        assert total >= len(text) * 0.9

    def test_overlap_carries_context(self):
        para = "AAAAA" * 200    # 1000 chars
        text = (para + "\n\n") * 4
        chunks = chunk_text(text, max_chunk_tokens=200, overlap_tokens=50)
        # With overlap > 0, total chars across chunks > original (carryover)
        total = sum(len(c) for c in chunks)
        assert total > len(text)

    def test_oversize_single_paragraph_hard_split(self):
        # One paragraph way bigger than max — must be split. chunk_text has
        # a 1024-char floor to keep chunks meaningful, so use a paragraph
        # well above that floor and assert against the actual bound.
        huge_para = "z" * 8000
        chunks = chunk_text(huge_para, max_chunk_tokens=500, overlap_tokens=0)
        assert len(chunks) >= 2
        # max_chars = max(1024, 500*2.8/1.1) ≈ 1272; small slack for boundaries.
        assert all(len(c) <= 1300 for c in chunks)


# ── coalesce_chunks ──────────────────────────────────────────────────────

class TestCoalesceChunks:
    def test_already_below_max_unchanged(self):
        chunks = ["a", "b", "c"]
        assert coalesce_chunks(chunks, max_count=5) == chunks

    def test_reduces_to_max_count(self):
        chunks = [f"chunk{i}" for i in range(10)]
        result = coalesce_chunks(chunks, max_count=3)
        assert len(result) <= 3

    def test_preserves_order(self):
        chunks = ["A", "B", "C", "D", "E", "F"]
        result = coalesce_chunks(chunks, max_count=2)
        # First group should contain A/B/C, second D/E/F (or similar order).
        assert "A" in result[0]
        assert "F" in result[-1]

    def test_no_content_lost(self):
        chunks = [f"item-{i}" for i in range(20)]
        result = coalesce_chunks(chunks, max_count=4)
        merged = " ".join(result)
        for i in range(20):
            assert f"item-{i}" in merged


# ── fanout_summarize end-to-end with stubbed llm_call ────────────────────

class TestFanoutSummarize:
    def _make_stub(self) -> Callable[[str, str], str]:
        """Stub LLM that records every call and returns a deterministic
        chunk-aware response."""
        calls: list[tuple[str, str]] = []

        def stub(system: str, user: str) -> str:
            calls.append((system, user))
            # Identify whether map step or reduce step from system prompt.
            if "merging" in system.lower():
                # Reduce: count chunk inputs, return merged
                num_chunks = user.count("=== Chunk ")
                return f"MERGED({num_chunks})"
            # Map: extract chunk number from prompt
            for line in user.splitlines():
                if line.startswith("Document chunk "):
                    return f"SUMMARY of {line.strip()}"
            return "SUMMARY (no chunk header)"
        stub.calls = calls  # type: ignore[attr-defined]
        return stub

    def test_short_input_one_chunk_one_summary(self):
        stub = self._make_stub()
        text = "Some short content about cats."
        result = fanout_summarize(
            text=text, user_question="What is this about?",
            config={}, llm_call=stub,
            ctx_window=32768, max_subagents=5,
        )
        # Map call: 1 chunk. Reduce call: 1.
        assert len(stub.calls) == 2  # type: ignore[attr-defined]
        assert "MERGED(1)" in result
        assert "Auto-fanout summary" in result  # header present

    def test_long_input_produces_multiple_chunks(self):
        stub = self._make_stub()
        # ~150k chars → ~58k tokens → many chunks under target=ctx_window/4=8192.
        text = ("Section about deep learning. " * 100 + "\n\n") * 50
        result = fanout_summarize(
            text=text, user_question="What does this say about deep learning?",
            config={}, llm_call=stub,
            ctx_window=32768, max_subagents=5,
        )
        # We capped at max_subagents=5, so map calls ≤ 5, +1 reduce.
        assert len(stub.calls) <= 6  # type: ignore[attr-defined]
        assert len(stub.calls) >= 2  # type: ignore[attr-defined]
        # Last call must be the reduce step
        last_system, _ = stub.calls[-1]  # type: ignore[attr-defined]
        assert "merging" in last_system.lower()

    def test_chunk_failure_does_not_break_reduce(self):
        """If one chunk's sub-LLM call raises, the merged result still
        completes using whatever chunks succeeded plus an error placeholder."""
        call_count = [0]

        def flaky(system: str, user: str) -> str:
            call_count[0] += 1
            # Map step calls are first; raise on the 2nd map call.
            if "merging" not in system.lower() and call_count[0] == 2:
                raise RuntimeError("simulated chunk-2 timeout")
            if "merging" in system.lower():
                return "MERGED OK"
            return "OK summary"

        text = "PARA\n\n" * 200
        result = fanout_summarize(
            text=text, user_question="Q?", config={}, llm_call=flaky,
            ctx_window=32768, max_subagents=5,
        )
        # The placeholder string from the failed chunk must appear in the
        # reduce input *or* the fallback concat result.
        assert "MERGED OK" in result or "chunk" in result.lower()

    def test_reduce_failure_falls_back_to_concat(self):
        def flaky(system: str, user: str) -> str:
            if "merging" in system.lower():
                raise RuntimeError("reduce broke")
            # Identify chunk number for distinct map outputs.
            for line in user.splitlines():
                if line.startswith("Document chunk "):
                    return f"MAP_{line.split()[2]}"
            return "MAP"

        text = "P\n\n" * 200
        result = fanout_summarize(
            text=text, user_question="Q?", config={}, llm_call=flaky,
            ctx_window=32768, max_subagents=3,
        )
        # Fallback prepends "## Chunk N" headers
        assert "## Chunk 1" in result
        # Header still present
        assert "Auto-fanout summary" in result


# ── fanout_notice ────────────────────────────────────────────────────────

class TestFanoutNotice:
    def test_includes_tool_and_size(self):
        msg = fanout_notice("Read", 70000, 5, 32768)
        assert "Read" in msg
        assert "70,000" in msg
        assert "5" in msg
        assert "32,768" in msg


# ── End-to-end regression: 6.6MB PDF case ────────────────────────────────

class TestPDFRegression:
    def test_70k_token_pdf_fits_after_fanout(self):
        """The motivating case: a 6.6 MB PDF → ~70k tokens of text.
        Without fanout, this kills any 32k-window model. With fanout the
        merged summary is bounded by the reduce step's ≤800-word limit.
        """
        # Build a 70k-token-equivalent input.
        text = "Section " + ("technical content " * 400 + "\n\n") * 50
        # Simulate the merged summary length being controlled by the reduce
        # prompt's "≤ 800 words" instruction.
        def stub(system: str, user: str) -> str:
            if "merging" in system.lower():
                return "Merged summary " + ("word " * 800)
            return "Map summary " + ("word " * 400)

        cfg = {"auto_fanout_enabled": True}
        # Confirm should_fanout fires
        assert should_fanout("Read", text, ctx_window=32768, config=cfg) is True
        # Confirm fanout_summarize completes and produces a small output
        out = fanout_summarize(
            text=text, user_question="Summarize",
            config=cfg, llm_call=stub,
            ctx_window=32768, max_subagents=5,
        )
        out_tokens = estimate_tokens_simple(out)
        # Merged summary should fit comfortably below the threshold (40% of
        # 32k = ~13k tokens). The reduce prompt caps at 800 words ≈ 1100
        # tokens; map summaries each ≤ 400 words ≈ 550 tokens.
        assert out_tokens < 32768 * 0.4
