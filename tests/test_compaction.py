"""Tests for compaction.py — token estimation, context limits, snipping, split point."""
from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compaction import (
    estimate_tokens,
    get_context_limit,
    snip_old_tool_results,
    find_split_point,
    sanitize_history,
    _respect_tool_pairs,
)


# ── estimate_tokens ───────────────────────────────────────────────────────

def _expected(chars: int, msgs: int) -> int:
    """Mirror the estimate_tokens formula: int((int(chars/2.8) + msgs*4) * 1.1)"""
    return int((int(chars / 2.8) + msgs * 4) * 1.1)


class TestEstimateTokens:
    def test_simple_messages(self):
        msgs = [
            {"role": "user", "content": "Hello world"},          # 11 chars
            {"role": "assistant", "content": "Hi there!"},       # 9 chars
        ]
        result = estimate_tokens(msgs)
        # 20 chars, 2 messages → _expected(20, 2)
        assert result == _expected(20, 2)

    def test_empty_messages(self):
        assert estimate_tokens([]) == 0

    def test_empty_content(self):
        msgs = [{"role": "user", "content": ""}]
        # 0 chars + 1 msg framing overhead
        assert estimate_tokens(msgs) == _expected(0, 1)

    def test_tool_result_messages(self):
        msgs = [
            {"role": "tool", "tool_call_id": "abc", "name": "Read", "content": "x" * 350},
        ]
        result = estimate_tokens(msgs)
        assert result == _expected(350, 1)

    def test_structured_content(self):
        """Content that is a list of dicts (e.g. Anthropic tool_result blocks)."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "id1", "content": "A" * 70},
                ],
            },
        ]
        result = estimate_tokens(msgs)
        # "tool_result" (11) + "id1" (3) + "A"*70 (70) = 84 chars, 1 message
        assert result == _expected(84, 1)

    def test_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        ]
        result = estimate_tokens(msgs)
        # content "ok" (2) + tool_calls recursively: "c1"(2) + "Bash"(4) + "ls"(2) = 10 chars
        # (nested input dict values ARE now counted)
        assert result == _expected(10, 1)

    def test_deeply_nested_tool_input(self):
        """Deeply nested tool input values are all counted."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "x", "name": "Write", "input": {"file_path": "/a/b", "content": "hello"}},
                ],
            },
        ]
        result = estimate_tokens(msgs)
        # "x"(1) + "Write"(5) + "/a/b"(4) + "hello"(5) = 15 chars, 1 msg
        assert result == _expected(15, 1)


# ── get_context_limit ─────────────────────────────────────────────────────

class TestGetContextLimit:
    def test_anthropic(self):
        assert get_context_limit("claude-opus-4-6") == 200000

    def test_gemini(self):
        assert get_context_limit("gemini-2.0-flash") == 1000000

    def test_deepseek(self):
        assert get_context_limit("deepseek-chat") == 64000

    def test_openai(self):
        assert get_context_limit("gpt-4o") == 128000

    def test_qwen(self):
        assert get_context_limit("qwen-max") == 1000000

    def test_unknown_model_fallback(self):
        # Unknown models fall back to openai provider which has 128000
        assert get_context_limit("some-random-model-xyz") == 128000

    def test_explicit_provider_prefix(self):
        assert get_context_limit("ollama/llama3.3") == 128000


# ── snip_old_tool_results ─────────────────────────────────────────────────

class TestSnipOldToolResults:
    def test_old_tool_results_get_truncated(self):
        long_content = "A" * 5000
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "let me check", "tool_calls": []},
            {"role": "tool", "tool_call_id": "t1", "name": "Read", "content": long_content},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "you're welcome"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "goodbye"},
            {"role": "user", "content": "wait"},
            {"role": "assistant", "content": "yes?"},
            {"role": "user", "content": "never mind"},
        ]
        result = snip_old_tool_results(msgs, max_chars=2000, preserve_last_n_turns=6)
        assert result is msgs  # mutated in place
        tool_msg = msgs[2]
        assert len(tool_msg["content"]) < 5000
        assert "snipped" in tool_msg["content"]

    def test_recent_tool_results_preserved(self):
        long_content = "B" * 5000
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok", "tool_calls": []},
            {"role": "tool", "tool_call_id": "t1", "name": "Read", "content": long_content},
        ]
        # All 3 messages are within preserve_last_n_turns=6
        result = snip_old_tool_results(msgs, max_chars=2000, preserve_last_n_turns=6)
        assert msgs[2]["content"] == long_content  # not truncated

    def test_short_tool_results_not_touched(self):
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "name": "Bash", "content": "short"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "user", "content": "d"},
            {"role": "user", "content": "e"},
            {"role": "user", "content": "f"},
        ]
        snip_old_tool_results(msgs, max_chars=2000, preserve_last_n_turns=6)
        assert msgs[0]["content"] == "short"

    def test_non_tool_messages_untouched(self):
        msgs = [
            {"role": "user", "content": "X" * 5000},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "user", "content": "d"},
            {"role": "user", "content": "e"},
            {"role": "user", "content": "f"},
        ]
        snip_old_tool_results(msgs, max_chars=2000, preserve_last_n_turns=6)
        assert msgs[0]["content"] == "X" * 5000


# ── find_split_point ──────────────────────────────────────────────────────

class TestFindSplitPoint:
    def test_returns_reasonable_index(self):
        msgs = [
            {"role": "user", "content": "A" * 1000},
            {"role": "assistant", "content": "B" * 1000},
            {"role": "user", "content": "C" * 1000},
            {"role": "assistant", "content": "D" * 1000},
            {"role": "user", "content": "E" * 1000},
        ]
        idx = find_split_point(msgs, keep_ratio=0.3)
        # With equal-size messages and keep_ratio=0.3, split should be around index 3-4
        assert 2 <= idx <= 4

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        idx = find_split_point(msgs, keep_ratio=0.3)
        assert idx == 0

    def test_empty_messages(self):
        idx = find_split_point([], keep_ratio=0.3)
        assert idx == 0

    def test_split_preserves_recent(self):
        # Recent portion should contain ~30% of tokens
        msgs = [{"role": "user", "content": "X" * 100} for _ in range(10)]
        idx = find_split_point(msgs, keep_ratio=0.3)
        total = estimate_tokens(msgs)
        recent = estimate_tokens(msgs[idx:])
        # Recent should be roughly 30% of total (allow some tolerance)
        assert recent >= total * 0.2
        assert recent <= total * 0.5

    def test_split_never_splits_tool_pair(self):
        # Large old portion, then one tool-call pair right where the natural
        # split would fall, followed by a small recent portion.
        msgs = (
            [{"role": "user", "content": "U" * 2000}]
            + [{"role": "assistant", "content": "A" * 2000} for _ in range(3)]
            + [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
                ]},
                {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "R" * 500},
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        for ratio in (0.2, 0.3, 0.4, 0.5):
            idx = find_split_point(msgs, keep_ratio=ratio)
            # The split must not leave an orphan 'tool' as messages[idx].
            if 0 < idx < len(msgs):
                assert msgs[idx].get("role") != "tool", (
                    f"split at {idx} orphans a tool message (ratio={ratio})")
                prev = msgs[idx - 1]
                if prev.get("role") == "assistant" and prev.get("tool_calls"):
                    # Must have kept the tool responses with it.
                    assert msgs[idx].get("role") != "tool"


class TestRespectToolPairs:
    def test_noop_when_not_on_tool_pair(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        assert _respect_tool_pairs(msgs, 2) == 2

    def test_advances_past_tool_response(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        # Natural split at 2 (between assistant and its tool response) → push to 3.
        assert _respect_tool_pairs(msgs, 2) == 3

    def test_advances_past_multiple_tool_responses(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
                {"id": "c2", "name": "Bash", "input": {"command": "b"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "1"},
            {"role": "tool", "tool_call_id": "c2", "name": "Bash", "content": "2"},
            {"role": "user", "content": "next"},
        ]
        assert _respect_tool_pairs(msgs, 1) == 3

    def test_orphan_tool_at_split_gets_skipped(self):
        # messages[split]=='tool' but messages[split-1] is not an assistant
        # with tool_calls → still push forward past the tool.
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "tool", "tool_call_id": "cX", "name": "Bash", "content": "x"},
            {"role": "user", "content": "b"},
        ]
        assert _respect_tool_pairs(msgs, 1) == 2


# ── sanitize_history ──────────────────────────────────────────────────────

class TestSanitizeHistory:
    def test_well_formed_history_unchanged(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        out = sanitize_history(msgs)
        assert out == msgs

    def test_drops_orphan_tool_message(self):
        # tool with no preceding assistant(tool_calls) — simulates post-compaction orphan.
        msgs = [
            {"role": "user", "content": "[summary]"},
            {"role": "assistant", "content": "understood"},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "orphan"},
            {"role": "user", "content": "thanks"},
        ]
        out = sanitize_history(msgs)
        assert all(m.get("role") != "tool" for m in out)
        assert len(out) == 3
        assert [m["role"] for m in out] == ["user", "assistant", "user"]

    def test_strips_unanswered_tool_calls(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
                {"id": "c2", "name": "Bash", "input": {"command": "b"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "A"},
            {"role": "user", "content": "next"},
        ]
        out = sanitize_history(msgs)
        asst = out[1]
        # c2 had no response → stripped; c1 remains.
        assert [tc["id"] for tc in asst["tool_calls"]] == ["c1"]
        assert out[2]["role"] == "tool" and out[2]["tool_call_id"] == "c1"

    def test_strips_all_tool_calls_when_none_answered(self):
        msgs = [
            {"role": "assistant", "content": "thinking", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
            ]},
            {"role": "user", "content": "never mind"},
        ]
        out = sanitize_history(msgs)
        assert "tool_calls" not in out[0]
        assert out[0]["content"] == "thinking"

    def test_unanswered_at_end_also_stripped(self):
        # No trailing user/assistant; unanswered tool_calls at the very end.
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
            ]},
        ]
        out = sanitize_history(msgs)
        assert "tool_calls" not in out[1]

    def test_input_not_mutated(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
            ]},
            {"role": "user", "content": "x"},
        ]
        import copy
        snapshot = copy.deepcopy(msgs)
        sanitize_history(msgs)
        assert msgs == snapshot

    def test_tool_with_wrong_id_dropped(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
            ]},
            {"role": "tool", "tool_call_id": "WRONG", "name": "Bash", "content": "bad"},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "good"},
            {"role": "user", "content": "ok"},
        ]
        out = sanitize_history(msgs)
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "c1"
