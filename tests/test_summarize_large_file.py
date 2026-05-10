"""Tests for the SummarizeLargeFile multi-agent map-reduce tool.

The user reported: a ~24K-token PDF on qwen2.5-72b's 32K-context model
hit context-overflow → circuit breaker → infinite loop. Fix is to
chunk the file adaptively, summarize each chunk in parallel via sub-LLM
calls, then merge — making file size irrelevant.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import tools.files as f
from tools.files import (
    _estimate_text_tokens,
    _read_file_for_summary,
    _plan_chunks,
    _summarize_large_file,
)


# ── Token estimator ──────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected_min,expected_max", [
    ("",                                  0,    0),
    ("hello",                             1,    2),
    ("x" * 280,                          90,  110),    # 280/2.8 = 100
    ("x" * 32000,                     11000, 12000),   # ~11428
])
def test_estimate_text_tokens(text, expected_min, expected_max):
    n = _estimate_text_tokens(text)
    assert expected_min <= n <= expected_max


# ── Chunk planner: adaptive to file size + model context ─────────────────


def test_plan_chunks_tiny_file_single_chunk():
    """File that fits → 1 chunk (single-shot path)."""
    chunks = _plan_chunks("hello world this is small", 32768)
    assert len(chunks) == 1
    assert chunks[0] == "hello world this is small"


def test_plan_chunks_scales_with_file_size():
    """Number of chunks should grow with file size, not be capped."""
    # 32K-context model, ~8500 reserved → ~24K chunk budget
    # Files of increasing size should produce increasing chunk counts.
    sizes_to_chunks = []
    for kb in [10, 50, 200, 500]:
        text = "x" * (kb * 1024)
        sizes_to_chunks.append((kb, len(_plan_chunks(text, 32768))))
    # Monotone non-decreasing
    counts = [c for _, c in sizes_to_chunks]
    assert counts == sorted(counts), f"chunk counts not monotone: {sizes_to_chunks}"
    # 500KB should be quite a few chunks
    assert sizes_to_chunks[-1][1] >= 5


def test_plan_chunks_larger_context_means_fewer_chunks():
    """Same file in a 200K-context model → fewer chunks than in 32K."""
    text = "x" * (200 * 1024)   # 200KB
    n_32k = len(_plan_chunks(text, 32768))
    n_200k = len(_plan_chunks(text, 200000))
    assert n_200k < n_32k, (
        f"expected larger context → fewer chunks, got 32K={n_32k}, 200K={n_200k}"
    )


def test_plan_chunks_have_overlap():
    """Adjacent chunks should share characters (overlap for continuity)."""
    text = "x" * 100000
    chunks = _plan_chunks(text, 32768)
    if len(chunks) >= 2:
        # Last chars of chunk 0 should appear at start of chunk 1
        # (not testable by content since all 'x' — just verify total
        # chars > unique chars, indicating overlap exists)
        total_chunk_chars = sum(len(c) for c in chunks)
        assert total_chunk_chars > len(text), "no overlap between chunks"


def test_plan_chunks_covers_entire_content():
    """Every byte of input must be in at least one chunk."""
    text = "ABCDEFGH" * 10000  # 80K chars, deterministic content
    chunks = _plan_chunks(text, 32768)
    # Concat back together — should contain all content (with overlap dupes)
    concat = "".join(chunks)
    assert text[:1000] in concat
    assert text[-1000:] in concat


# ── File reader dispatch ─────────────────────────────────────────────────


def test_read_file_for_summary_text_file(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello world\nline 2", encoding="utf-8")
    content = _read_file_for_summary(str(p), {})
    assert "hello world" in content
    assert "line 2" in content


def test_read_file_for_summary_missing_file(tmp_path):
    out = _read_file_for_summary(str(tmp_path / "nope.txt"), {})
    assert out.startswith("Error")
    assert "not found" in out


def test_read_file_for_summary_directory_rejected(tmp_path):
    out = _read_file_for_summary(str(tmp_path), {})
    assert out.startswith("Error")
    assert "directory" in out.lower()


# ── Full pipeline (mocked LLM) ───────────────────────────────────────────


def test_summarize_small_file_uses_single_shot(tmp_path, monkeypatch):
    """Tiny file → one LLM call with mode='single'."""
    p = tmp_path / "small.txt"
    p.write_text("This is a short document.", encoding="utf-8")

    calls = []

    def fake_summarize_chunk(text, focus, config, mode="single", **kw):
        calls.append({"mode": mode, "text_len": len(text), "focus": focus})
        return f"SUMMARY-{mode}-of-{len(text)}-chars"

    monkeypatch.setattr(f, "_summarize_chunk_via_llm", fake_summarize_chunk)
    out = _summarize_large_file({"file_path": str(p)}, {"model": "claude-opus-4-7"})

    # Exactly one call, mode='single'
    assert len(calls) == 1
    assert calls[0]["mode"] == "single"
    # Output mentions single-shot
    assert "single-shot" in out


def test_summarize_large_file_does_map_then_reduce(tmp_path, monkeypatch):
    """Big file (relative to model context) → N map calls + 1 reduce call.
    Number of map calls matches the chunk planner's output."""
    # Force a small model context via monkeypatching so the test is robust
    # against future changes to provider context limits in compaction.py.
    import tools.files as _f
    monkeypatch.setattr(
        "compaction.get_context_limit", lambda m: 32768,
    )

    p = tmp_path / "huge.txt"
    p.write_text("X" * (200 * 1024), encoding="utf-8")  # 200KB

    map_calls = []
    reduce_calls = []

    def fake_summarize_chunk(text, focus, config, mode="single", **kw):
        if mode == "map":
            map_calls.append(kw.get("chunk_idx"))
        elif mode == "reduce":
            reduce_calls.append(len(text))
        return f"chunk-{kw.get('chunk_idx', 0)}-summary"

    monkeypatch.setattr(_f, "_summarize_chunk_via_llm", fake_summarize_chunk)
    out = _summarize_large_file({"file_path": str(p)}, {"model": "test-32k-model"})

    # Multiple map calls (for 200KB / ~73K tokens on a 32K-context model
    # with ~8.5K reserved → ~24K chunks → ~3-4 chunks, depending on overlap)
    assert len(map_calls) >= 3, f"expected ≥3 map calls for 200KB, got {len(map_calls)}"
    # Exactly one reduce
    assert len(reduce_calls) == 1
    # Output advertises map-reduce
    assert "map-reduce" in out.lower()
    assert f"{len(map_calls)} chunks" in out


def test_summarize_missing_file_error(monkeypatch):
    """Missing file → Error: ... returned, NOT a crash."""
    out = _summarize_large_file(
        {"file_path": "/nonexistent/path/abc.pdf"}, {"model": "claude-opus-4-7"},
    )
    assert out.startswith("Error")
    assert "not found" in out


def test_summarize_missing_required_param():
    out = _summarize_large_file({}, {"model": "claude-opus-4-7"})
    assert out.startswith("Error")
    assert "file_path" in out


# ── Tool registration ────────────────────────────────────────────────────


def test_summarize_large_file_tool_registered():
    import tool_registry
    tool = tool_registry.get_tool("SummarizeLargeFile")
    assert tool is not None
    assert tool.read_only is True
    assert tool.concurrent_safe is True


def test_summarize_large_file_schema_advertises_chunking():
    """The tool description must mention map-reduce / chunking so a model
    reading the tool list knows when to pick this tool over Read."""
    import tool_registry
    tool = tool_registry.get_tool("SummarizeLargeFile")
    desc = tool.schema["description"].lower()
    # Key marketing the model must see
    assert "chunk" in desc
    assert "context" in desc or "overflow" in desc
    assert "summar" in desc
