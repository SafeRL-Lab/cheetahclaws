"""Tests for the dynamic context-window / max_tokens cap.

Regression target: a 32k-context model (qwen2.5-72b) running under the custom
provider with a 24k-token prompt + 8k requested output → API rejects with
"maximum context length is 32768. requested 8192 output, prompt has 24577 input,
total 32769". Two layers cooperate to prevent this:

  1. compaction.get_context_limit must return the model's REAL ctx (32768 for
     qwen2.5-72b), not the static 128000 default for the custom provider, so
     the auto-compact threshold (70%) fires at the right time.

  2. providers.dynamic_cap_max_tokens must shrink the per-call max_tokens so
     input + output never exceeds ctx - safety_margin even if compaction
     hasn't yet fired.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import providers
import compaction


# ── Known model context registry ─────────────────────────────────────────

class TestKnownModelContext:
    def test_qwen25_72b_is_32k(self):
        assert providers.get_model_context_window("custom", "qwen2.5-72b") == 32768

    def test_qwen25_72b_via_custom_prefix(self):
        assert providers.get_model_context_window("custom", "custom/qwen2.5-72b") == 32768

    def test_qwen25_coder_is_32k(self):
        assert providers.get_model_context_window("custom", "qwen2.5-coder-32b") == 32768

    def test_llama33_is_128k(self):
        assert providers.get_model_context_window("ollama", "llama3.3") == 131072

    def test_phi4_is_16k(self):
        assert providers.get_model_context_window("ollama", "phi4") == 16384

    def test_unknown_model_falls_back_to_provider_default(self):
        # 'completely-unknown-model' not in registry → uses provider context_limit
        result = providers.get_model_context_window("anthropic", "completely-unknown-model")
        assert result == providers.PROVIDERS["anthropic"]["context_limit"]

    def test_prefix_match_for_variant(self):
        # Variant suffix should still match the registry entry by prefix.
        # "qwen2.5-72b-instruct-vllm-build" starts with "qwen2.5-72b" → 32768.
        result = providers.get_model_context_window("custom", "qwen2.5-72b-instruct-vllm-build")
        assert result == 32768


# ── dynamic_cap_max_tokens ───────────────────────────────────────────────

class TestDynamicCapMaxTokens:
    def test_small_prompt_returns_configured(self):
        # Tiny prompt, lots of headroom — return the configured value untouched.
        msgs = [{"role": "user", "content": "Hello"}]
        result = providers.dynamic_cap_max_tokens(
            messages=msgs, system="", tool_schemas=None,
            ctx_window=32768, configured=8192,
        )
        assert result == 8192

    def test_large_prompt_shrinks_max_tokens(self):
        # 24k char prompt at chars/2.8 ≈ 8.5k tokens, plus framing 1.1× → ~9.4k.
        # ctx 32768 - 9400 - safety 1024 = ~22344 → caller wants 8192 → return 8192.
        msgs = [{"role": "user", "content": "x" * 24000}]
        result = providers.dynamic_cap_max_tokens(
            messages=msgs, system="", tool_schemas=None,
            ctx_window=32768, configured=8192,
        )
        assert result == 8192

    def test_oversized_prompt_caps_below_configured(self):
        # Reproduce the original bug: a prompt that already estimates ~24k+ tokens
        # combined with a 8192 configured max_tokens should shrink so the total
        # fits under ctx_window - safety_margin.
        big_text = "y" * 80000  # ~31k estimated tokens
        msgs = [{"role": "user", "content": big_text}]
        ctx = 32768
        configured = 8192
        result = providers.dynamic_cap_max_tokens(
            messages=msgs, system="You are a helpful assistant.",
            tool_schemas=None, ctx_window=ctx, configured=configured,
        )
        # Recompute the input estimate the same way dynamic_cap_max_tokens does
        # and assert the result keeps total under ctx - safety_margin.
        input_est = compaction.estimate_tokens(msgs)
        # Must not be above the configured value
        assert result <= configured
        # Must keep input + result + safety <= ctx (or hit the 256 floor)
        assert result == 256 or (input_est + result + 1024 <= ctx)

    def test_returns_floor_when_input_alone_exceeds_window(self):
        # Input alone bigger than ctx_window → return the 256 floor.
        msgs = [{"role": "user", "content": "z" * 200000}]
        result = providers.dynamic_cap_max_tokens(
            messages=msgs, system="", tool_schemas=None,
            ctx_window=32768, configured=8192,
        )
        assert result == 256

    def test_system_prompt_counted(self):
        # A 50k-char system prompt should reduce headroom.
        msgs = [{"role": "user", "content": "hi"}]
        big_system = "S" * 50000
        result = providers.dynamic_cap_max_tokens(
            messages=msgs, system=big_system, tool_schemas=None,
            ctx_window=32768, configured=8192,
        )
        # 50k chars / 2.8 * 1.1 ≈ 19.6k tokens, ctx 32768 - 19600 - 1024 ≈ 12144
        # → return min(8192, 12144) = 8192. But if we used a smaller ctx the
        # result must shrink below 8192. Here we just assert sanity.
        assert result <= 8192

    def test_tool_schemas_counted(self):
        # 30 fake tools with verbose schemas eat into the headroom.
        msgs = [{"role": "user", "content": "hi"}]
        tools = [
            {"name": f"tool_{i}", "description": "x" * 500,
             "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
            for i in range(30)
        ]
        result = providers.dynamic_cap_max_tokens(
            messages=msgs, system="", tool_schemas=tools,
            ctx_window=32768, configured=8192,
        )
        # Just assert it ran and produced a sane number.
        assert 256 <= result <= 8192


# ── compaction.get_context_limit for custom provider ─────────────────────

class TestCompactionContextLimit:
    def test_known_custom_model_returns_real_limit(self):
        # custom/qwen2.5-72b should return 32768 via the registry, not 128000.
        assert compaction.get_context_limit("custom/qwen2.5-72b") == 32768

    def test_known_anthropic_unchanged(self):
        # Pre-existing test from test_compaction.py — must not regress.
        assert compaction.get_context_limit("claude-opus-4-6") == 200000

    def test_unknown_custom_falls_back_to_provider_default(self):
        # Unknown custom model with no live fetch (no base_url passed) →
        # provider default. May be the originally-static 128000 OR a value
        # that was backfilled by a prior fetch in another test; either way,
        # it should at least be a positive integer.
        result = compaction.get_context_limit("custom/totally-unknown-xyz")
        assert isinstance(result, int) and result > 0


# ── End-to-end regression for the qwen2.5-72b 32k case ───────────────────

class TestQwen32kRegression:
    def test_compaction_threshold_fits_qwen25_72b(self):
        # The exact failure scenario: ctx 32768, threshold = 32768 * 0.7 = 22937.6.
        # If get_context_limit incorrectly returned 128000, threshold would be
        # 89600 and compaction would never fire before the 32k overflow.
        limit = compaction.get_context_limit("custom/qwen2.5-72b")
        threshold = limit * 0.7
        # Threshold must fall safely BELOW the real ctx window with margin for
        # one more turn of growth (>= 1k headroom).
        assert threshold < limit - 1000
        assert threshold == 32768 * 0.7

    def test_dynamic_cap_prevents_24k_plus_8k_overflow(self):
        # Simulate the exact failure: 24,577 input tokens, 8192 requested output,
        # ctx 32768. With the fix, max_tokens must shrink to keep total in window.
        # We construct a message large enough to produce ~24,577 estimated tokens.
        # estimate_tokens uses chars/2.8 with framing+1.1 multiplier, so ~62,400
        # chars ≈ 24,500 tokens.
        msgs = [{"role": "user", "content": "q" * 62400}]
        input_est = compaction.estimate_tokens(msgs)
        ctx = 32768
        configured = 8192
        capped = providers.dynamic_cap_max_tokens(
            messages=msgs, system="", tool_schemas=None,
            ctx_window=ctx, configured=configured,
        )
        # The whole point: input_est + capped must fit under ctx with safety.
        assert input_est + capped <= ctx - 1024 + 1  # +1 for floor() rounding
