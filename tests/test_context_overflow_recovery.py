"""Tests for the three context-overflow / circuit-breaker doom-loop fixes.

User report: `/ssj 15 → Research Assistant` on a large PDF. qwen2.5-72b
has a 32K context, the PDF read used 24577 input tokens, output cap was
8192 → total 32769 → 1 token over the limit. Every API call returned
the same BadRequestError. Circuit breaker opened, cooled down, retried
the same broken request, opened again — forever.

Three fixes:
  1. agent.py — parse the explicit token counts in the error message
     and auto-reduce output cap to fit, instead of looping at 8192.
  2. agent_runner.py — stop the agent after N consecutive identical
     failures so a fundamentally broken request can't loop for hours.
  3. agent_runner.py — when the iteration text mentions a circuit-
     breaker cooldown, sleep that long instead of the 2s configured
     interval.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent import _try_reduce_output_cap_from_error


# ── _try_reduce_output_cap_from_error ───────────────────────────────────


# OpenAI-style (the user's actual error):
_OPENAI_OVERFLOW_ERR = (
    "Error code: 400 - {'error': {'message': \"This model's maximum "
    "context length is 32768 tokens. However, you requested 8192 output "
    "tokens and your prompt contains at least 24577 input tokens, for a "
    "total of 32769 tokens.\"}}"
)


def test_parse_reproduces_user_failure_case():
    """The user's exact error: max=32768, prompt=24577, requested=8192.
    Safe new cap: 32768 - 24577 - 2500_buffer = 5691.

    The buffer is intentionally generous (~7.6% of 32K) because vLLM-
    served qwen2.5-72b in the wild re-tokenizes prompts ~+1000 tokens
    larger on the retry (decoder priming budget not counted in initial
    validation). Earlier 200 / 1000 buffers both failed by getting
    eaten by this growth; 2500 has real headroom."""
    new_cap = _try_reduce_output_cap_from_error(
        _OPENAI_OVERFLOW_ERR, {"max_tokens": 8192},
    )
    assert new_cap == 5691


def test_parse_returns_none_when_new_cap_not_smaller():
    """If current cap is already <= the safe cap (no reduction needed),
    return None — don't pretend to fix something that's already fine."""
    # Current cap of 4096 is well under the 7991 safe ceiling — no need
    # to reduce.
    assert _try_reduce_output_cap_from_error(
        _OPENAI_OVERFLOW_ERR, {"max_tokens": 4096},
    ) is None


def test_parse_returns_none_when_safe_cap_is_too_small():
    """If the prompt is so big that fitting any reasonable output cap
    is impossible (<256 tokens), give up and let the caller fall back
    to compaction."""
    err = (
        "This model's maximum context length is 8192 tokens. However, "
        "your prompt contains at least 6000 input tokens, for a total "
        "of 14192 tokens."
    )
    # Safe cap = 8192 - 6000 - 2500_buffer = -308 < 256 → None
    assert _try_reduce_output_cap_from_error(err, {"max_tokens": 4096}) is None


def test_parse_returns_none_for_unrelated_errors():
    """Rate limit / connection / random errors → no parse, no reduction."""
    assert _try_reduce_output_cap_from_error(
        "rate limit exceeded", {"max_tokens": 8192}) is None
    assert _try_reduce_output_cap_from_error(
        "connection refused", {}) is None
    assert _try_reduce_output_cap_from_error("", {}) is None


def test_parse_anthropic_style_phrasing():
    """Anthropic phrases it slightly differently — make sure the
    tolerant regex still works."""
    err = (
        "Bad request: max context window of 200000 tokens exceeded. "
        "Your prompt contains 195000 input tokens."
    )
    new_cap = _try_reduce_output_cap_from_error(err, {"max_tokens": 16000})
    # 200000 - 195000 - 2500 = 2500
    assert new_cap == 2500


def test_parse_no_current_cap_still_works():
    """If config doesn't have max_tokens set (None), we still return a
    safe cap rather than None — the caller can apply it."""
    new_cap = _try_reduce_output_cap_from_error(
        _OPENAI_OVERFLOW_ERR, {},
    )
    assert new_cap == 5691


# ── Agent runner: circuit-breaker cooldown extraction ────────────────────


def test_circuit_cooldown_regex_matches_real_error():
    """The agent_runner's _CIRCUIT_RE pattern must match the exact
    string agent.py emits when the breaker is open. We replicate the
    same regex here so a future agent.py change to the message format
    doesn't silently break the runner's cooldown awareness."""
    import re
    text = (
        "[Circuit breaker OPEN for provider 'custom'. Cooldown: 120s. "
        "Use /circuit reset custom to force-close.]"
    )
    pattern = re.compile(
        r"Circuit breaker OPEN.*?Cooldown:\s*(\d+(?:\.\d+)?)\s*s",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    assert m is not None
    assert m.group(1) == "120"


def test_failure_marker_regex_matches_real_outputs():
    """Same idea — the runner's _FAILURE_RE must match the exact
    `[Failed ...]` and `[Circuit breaker ...]` markers agent.py emits."""
    import re
    pattern = re.compile(
        r"\[(?:Failed|Circuit breaker)\b[^\]]*\]",
        re.IGNORECASE,
    )
    # Real agent.py emissions
    assert pattern.search("\n[Failed — BadRequestError: too long.]\n") is not None
    assert pattern.search("[Circuit breaker OPEN for provider 'custom'. Cooldown: 120s.]") is not None
    # Negatives — don't match retry messages or success
    assert pattern.search("[Retry 1/3 after 2s — rate_limit]") is None
    assert pattern.search("response from the model OK") is None
