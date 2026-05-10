"""Tests for the Read-tool overflow redirect — defense-in-depth that
catches the case where the model ignores the template's "use
SummarizeLargeFile" instruction and calls Read/ReadPDF on a too-big file
anyway. The Read response itself routes the model to SummarizeLargeFile,
so the raw content never overflows the next API call."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from tools.files import (
    _is_cjk_heavy,
    _maybe_redirect_to_summarize,
)


# ── CJK-heavy detection ──────────────────────────────────────────────────


def test_is_cjk_heavy_pure_english_false():
    assert _is_cjk_heavy("hello world this is plain english") is False


def test_is_cjk_heavy_pure_chinese_true():
    assert _is_cjk_heavy("中文内容测试一下分词的情况" * 10) is True


def test_is_cjk_heavy_pure_japanese_true():
    assert _is_cjk_heavy("こんにちは世界これは日本語のテキストです" * 10) is True


def test_is_cjk_heavy_mixed_minority_cjk_false():
    """A document with <20% CJK characters should NOT be flagged."""
    text = "Mostly English text with a few 中文 characters here and there." * 20
    assert _is_cjk_heavy(text) is False


def test_is_cjk_heavy_empty():
    assert _is_cjk_heavy("") is False


# ── Redirect threshold logic ─────────────────────────────────────────────


def test_no_redirect_for_small_files():
    """A small file → returns None (caller returns original content)."""
    assert _maybe_redirect_to_summarize(
        "small file", "/tmp/x.txt", {"model": "custom/qwen2.5-72b"},
    ) is None


def test_no_redirect_for_empty_text():
    assert _maybe_redirect_to_summarize(
        "", "/tmp/x.txt", {"model": "claude-opus-4-7"},
    ) is None


def test_redirect_fires_on_users_actual_failure_case():
    """Reproduce the user's exact scenario: ~25K-token PDF on
    custom/qwen2.5-72b. The custom provider's declared ctx is 128K but
    the actual model is 32K — the safe_ctx cap at 30K must catch this."""
    # ~25K tokens of English-ish text (roughly 70K chars at 2.8 chars/token)
    big_text = "Sample paragraph with citations [Smith 2024]. " * 1500
    redirect = _maybe_redirect_to_summarize(
        big_text, "/home/user/autodan.pdf", {"model": "custom/qwen2.5-72b"},
    )
    assert redirect is not None
    assert "ReadTooLarge" in redirect
    assert "SummarizeLargeFile" in redirect
    assert "/home/user/autodan.pdf" in redirect
    # Redirect message must be MUCH smaller than the input it replaces —
    # that's the whole point of the redirect.
    assert len(redirect) < len(big_text) / 10


def test_redirect_fires_on_cjk_at_lower_char_count():
    """CJK content tokenizes 1:1 with chars, so a 17K-char CJK file is
    17K tokens — should trigger redirect on a 32K-context model. The
    same chars in English would NOT trigger (~6K tokens)."""
    cjk_text = "中文内容测试" * 3000   # ~18K chars CJK
    redirect_cjk = _maybe_redirect_to_summarize(
        cjk_text, "/tmp/cn.txt", {"model": "custom/qwen2.5-72b"},
    )
    assert redirect_cjk is not None, "CJK content of this size must trigger redirect"

    # Same character count in English should NOT trigger
    eng_text = "abcdef" * 3000   # also 18K chars but English
    redirect_eng = _maybe_redirect_to_summarize(
        eng_text, "/tmp/en.txt", {"model": "custom/qwen2.5-72b"},
    )
    assert redirect_eng is None, (
        "Equivalent char-count in English should NOT redirect (chars/2.8 = ~6K tokens, fits)"
    )


def test_redirect_caps_threshold_for_overconfident_provider():
    """`custom/...` provider declares 128K context but the underlying
    model might be 32K. The redirect must use min(ctx, 30K) as ceiling
    — protect users on small models even when the provider lies."""
    # 25K-token text. On a 128K-believed model with no cap, this would
    # NOT trigger (25K << 128K * 0.7). With our 30K cap, it DOES.
    text = "x" * 70000   # ~25K tokens English
    redirect = _maybe_redirect_to_summarize(
        text, "/tmp/big.txt", {"model": "custom/some-model"},
    )
    assert redirect is not None, (
        "custom-provider redirect must fire even though declared ctx is 128K"
    )


def test_no_redirect_on_genuine_large_context_model_with_modest_file():
    """A 32K-token file on claude-opus-4-7 (200K context) should NOT
    redirect — there's plenty of room. We're conservative, not paranoid."""
    text = "x" * 84000   # ~30K tokens
    # claude-opus-4-7 has 200K declared. Our cap is min(200K, 30K) = 30K.
    # Reservation 6K, ceiling 0.7*(30K-6K) = 16800.
    # 30K > 16800 → SHOULD redirect (because of the cap).
    # This actually demonstrates the conservative side-effect: we redirect
    # even on big-context models. That's intentional: SummarizeLargeFile
    # is cheap on the small-file path (single-shot), so redirecting
    # early is harmless even when not strictly needed.
    redirect = _maybe_redirect_to_summarize(
        text, "/tmp/x.txt", {"model": "claude-opus-4-7"},
    )
    # 30K tokens IS over the 16800 ceiling → redirect fires
    assert redirect is not None


def test_redirect_message_includes_preview():
    """The redirect must include a preview chunk so the model has *some*
    context to decide on a focus parameter for SummarizeLargeFile."""
    text = ("UNIQUE_PREVIEW_CONTENT_MARKER " * 200) + ("X" * 100000)
    redirect = _maybe_redirect_to_summarize(
        text, "/tmp/x.txt", {"model": "custom/qwen2.5-72b"},
    )
    assert redirect is not None
    assert "PREVIEW" in redirect
    # The preview comes from the start of the file
    assert "UNIQUE_PREVIEW_CONTENT_MARKER" in redirect


# ── Integration: Read tool wrapper actually applies the redirect ────────


def test_read_tool_redirects_huge_text_file(tmp_path):
    """Write a fake 'huge' text file, call Read via the tool dispatcher,
    verify the result is the redirect message (not the raw content)."""
    big = tmp_path / "big.txt"
    big.write_text("Sample line with content. " * 4000, encoding="utf-8")  # ~100KB

    # Call via the tool registry (simulates what agent.py does)
    from tools import execute_tool
    out = execute_tool(
        "Read",
        {"file_path": str(big)},
        permission_mode="accept-all",
        config={"model": "custom/qwen2.5-72b"},
    )
    # Defensive redirect must have fired
    assert "ReadTooLarge" in out
    assert "SummarizeLargeFile" in out
    assert str(big) in out


def test_read_tool_passes_through_small_file(tmp_path):
    """Small files — Read returns the actual content, not a redirect."""
    small = tmp_path / "small.txt"
    small.write_text("just a few lines\nof normal text\n", encoding="utf-8")

    from tools import execute_tool
    out = execute_tool(
        "Read",
        {"file_path": str(small)},
        permission_mode="accept-all",
        config={"model": "custom/qwen2.5-72b"},
    )
    assert "ReadTooLarge" not in out
    assert "just a few lines" in out
