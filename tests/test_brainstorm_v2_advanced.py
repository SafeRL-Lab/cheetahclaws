"""Tests for the four post-output-quality additions to /brainstorm:

  A. Programmatic action-plan filter (`_filter_action_plan` against
     `_extract_ban_keywords`) — the deterministic backstop for the
     prompt-side SELF-CHECK that weak leads ignore.

  B. Synthesis ranking enforcement (`_consensus_is_ranked` +
     `_ensure_consensus_is_ranked` fallback LLM call).

  C. Background mode (`--bg`) — flag parsing + the bg registry helpers
     (`_bg_register / _bg_set_stage / _bg_complete / _bg_snapshot`).

  D. /brainstorm status subcommand (smoke — needs registry to work).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import commands.advanced as adv
from commands.advanced import (
    _DEFAULT_BAN_KEYWORDS,
    _extract_ban_keywords,
    _filter_action_plan,
    _consensus_is_ranked,
    _ensure_consensus_is_ranked,
    _parse_bg_flag,
    _bg_register,
    _bg_set_stage,
    _bg_complete,
    _bg_snapshot,
    _BG_BRAINSTORMS,
)


# ── A. Ban-keyword extraction + action-plan filter ───────────────────────


def test_extract_ban_keywords_includes_defaults():
    kws = _extract_ban_keywords("")
    assert "consult an advisor" in kws
    assert "diversify your portfolio" in kws
    assert "咨询财务顾问" in kws
    assert "定期监控" in kws


def test_extract_ban_keywords_pulls_from_opening_quotes():
    """Quoted strings in the opening become extra ban keywords."""
    opening = """### Lead Opening
1. Concrete artifact: name specific tickers.
2. We will NOT accept:
   - "vague macro takes"
   - 「random meme stocks」
"""
    kws = _extract_ban_keywords(opening)
    # Defaults still present
    assert "consult an advisor" in kws
    # Opening-extracted bans appended
    assert "vague macro takes" in kws
    assert "random meme stocks" in kws


def test_filter_action_plan_drops_banned_items():
    synthesis = """## Ranked Consensus
1. Buy NVDA.

## Concrete Action Plan
1. Buy 100 shares of NVDA at market open.
2. Set a 10% stop-loss on the position.
3. Consult a financial advisor before placing the order.
4. Diversify your portfolio across 5 sectors.
5. Set up alerts for major NVDA news.

## What Was Filler
- nothing
"""
    filtered, removed = _filter_action_plan(synthesis,
                                              _extract_ban_keywords(""))
    # The banned items (3 + 4) must be gone.
    assert "Consult a financial advisor" not in filtered
    assert "Diversify your portfolio" not in filtered
    # The kept items (1, 2, 5) must still be there.
    assert "Buy 100 shares of NVDA" in filtered
    assert "Set a 10% stop-loss" in filtered
    assert "Set up alerts for major NVDA news" in filtered
    # Removed log
    assert len(removed) == 2
    assert any("consult" in r.lower() for r in removed)
    assert any("diversify" in r.lower() for r in removed)
    # Note appended
    assert "programmatic self-check removed 2 action(s)" in filtered


def test_filter_action_plan_handles_chinese_bans():
    synthesis = """## Concrete Action Plan
1. 明天买入100股NVDA。
2. 定期监控投资组合。
3. 咨询金融顾问。
"""
    filtered, removed = _filter_action_plan(synthesis,
                                              _extract_ban_keywords(""))
    assert "明天买入100股NVDA" in filtered
    assert "定期监控投资组合" not in filtered
    assert "咨询金融顾问" not in filtered
    assert len(removed) == 2


def test_filter_action_plan_no_action_section_returns_unchanged():
    """If there's no 'Concrete Action Plan' section, nothing to filter."""
    synthesis = "## Consensus\n- buy AAPL\n## Dissents\nnone"
    filtered, removed = _filter_action_plan(synthesis,
                                              _extract_ban_keywords(""))
    assert filtered == synthesis
    assert removed == []


def test_filter_action_plan_all_clean_no_changes():
    synthesis = """## Concrete Action Plan
1. Buy NVDA at market open.
2. Set 10% stop loss on the position.
3. Sell when target price 200 is reached.
"""
    filtered, removed = _filter_action_plan(synthesis,
                                              _extract_ban_keywords(""))
    assert removed == []
    assert filtered == synthesis


# ── B. Ranking detector + fallback ───────────────────────────────────────


def test_consensus_is_ranked_detects_proper_ranking():
    synthesis = """## Ranked Consensus
**Ranked by: highest expected return**
1. NVDA — strong AI tailwinds.
2. AAPL — stable cash flows.
3. MSFT — Azure dominance.
"""
    assert _consensus_is_ranked(synthesis) is True


def test_consensus_is_ranked_misses_unranked_bullets():
    synthesis = """## Consensus
- NVDA backed by A, B
- AAPL backed by A, C
- MSFT backed by B, C
"""
    assert _consensus_is_ranked(synthesis) is False


def test_consensus_is_ranked_misses_no_section():
    assert _consensus_is_ranked("nothing here") is False
    assert _consensus_is_ranked("") is False


def test_consensus_is_ranked_needs_at_least_two_items():
    """One numbered item isn't a ranking."""
    synthesis = "## Consensus\n1. Only one item.\n"
    assert _consensus_is_ranked(synthesis) is False


def test_ensure_consensus_is_ranked_skips_when_already_ranked(monkeypatch):
    """If already ranked, no LLM call is made."""
    call_count = {"n": 0}

    def fake_oneshot(*a, **kw):
        call_count["n"] += 1
        return "rewritten"

    monkeypatch.setattr(adv, "_llm_oneshot", fake_oneshot)
    synthesis = """## Ranked Consensus
1. NVDA
2. AAPL
"""
    result = _ensure_consensus_is_ranked(synthesis, "topic", "lead", {})
    assert result == synthesis    # untouched
    assert call_count["n"] == 0   # no LLM call


def test_ensure_consensus_is_ranked_calls_llm_when_unranked(monkeypatch):
    """If unranked, do ONE LLM call asking for a ranking."""
    monkeypatch.setattr(adv, "_llm_oneshot",
                         lambda *a, **kw: """## Ranked Consensus
**Ranked by: highest expected return**
1. NVDA — top pick.
2. AAPL — second.
""")
    synthesis = "## Consensus\n- NVDA\n- AAPL\n"
    result = _ensure_consensus_is_ranked(synthesis, "topic", "lead", {})
    assert "## Ranked Consensus" in result
    assert _consensus_is_ranked(result)


def test_ensure_consensus_is_ranked_keeps_original_on_llm_failure(monkeypatch):
    """LLM returns garbage that's STILL not ranked → keep original."""
    monkeypatch.setattr(adv, "_llm_oneshot",
                         lambda *a, **kw: "still no ranking here")
    synthesis = "## Consensus\n- A\n- B\n"
    result = _ensure_consensus_is_ranked(synthesis, "topic", "lead", {})
    assert result == synthesis   # fall back to original


# ── C. --bg flag parsing ─────────────────────────────────────────────────


@pytest.mark.parametrize("args,expected_bg,expected_rest", [
    ("",                                False, ""),
    ("the topic",                       False, "the topic"),
    ("--bg the topic",                  True,  "the topic"),
    ("the topic --bg",                  True,  "the topic"),
    ("--background the topic",          True,  "the topic"),
    ("--bg --rounds 3 the topic",       True,  "--rounds 3 the topic"),
    # `--bg` must be a token boundary — not a prefix of another flag
    ("--bgmode topic",                  False, "--bgmode topic"),
])
def test_parse_bg_flag(args, expected_bg, expected_rest):
    bg, rest = _parse_bg_flag(args)
    assert bg is expected_bg
    assert rest == expected_rest


# ── C. bg registry (D. status subcommand uses these) ────────────────────


@pytest.fixture(autouse=True)
def _clear_bg_registry():
    """Each test starts with a clean registry."""
    _BG_BRAINSTORMS.clear()
    yield
    _BG_BRAINSTORMS.clear()


def test_bg_register_and_snapshot_returns_entry():
    _bg_register("bs-test1", "topic 1", "/tmp/out1.md")
    snap = _bg_snapshot()
    assert len(snap) == 1
    assert snap[0]["id"] == "bs-test1"
    assert snap[0]["topic"] == "topic 1"
    assert snap[0]["output"] == "/tmp/out1.md"
    assert snap[0]["stage"] == "starting"
    assert snap[0]["done"] is False
    assert snap[0]["error"] == ""


def test_bg_set_stage_updates_in_place():
    _bg_register("bs-test2", "topic 2", "/tmp/out2.md")
    _bg_set_stage("bs-test2", "round 2/3")
    snap = _bg_snapshot()
    assert snap[0]["stage"] == "round 2/3"


def test_bg_complete_marks_done():
    _bg_register("bs-test3", "topic 3", "/tmp/out3.md")
    _bg_complete("bs-test3")
    snap = _bg_snapshot()
    assert snap[0]["done"] is True
    assert snap[0]["error"] == ""
    assert snap[0]["stage"] == "complete"


def test_bg_complete_with_error_marks_failed():
    _bg_register("bs-test4", "topic 4", "/tmp/out4.md")
    _bg_complete("bs-test4", error="provider 429")
    snap = _bg_snapshot()
    assert snap[0]["done"] is True
    assert snap[0]["error"] == "provider 429"
    assert snap[0]["stage"] == "failed"


def test_bg_snapshot_sorted_by_start_time_desc():
    """Most recently started first."""
    _bg_register("old", "old topic", "/tmp/old.md")
    time.sleep(0.01)
    _bg_register("new", "new topic", "/tmp/new.md")
    snap = _bg_snapshot()
    assert snap[0]["id"] == "new"
    assert snap[1]["id"] == "old"


def test_bg_snapshot_drops_finished_older_than_1h():
    """Long-finished entries are pruned to keep the list useful."""
    _bg_register("ancient", "ancient topic", "/tmp/a.md")
    # Fake the start time to >1h ago AND mark done
    _BG_BRAINSTORMS["ancient"]["started"] = time.time() - 7200
    _BG_BRAINSTORMS["ancient"]["done"] = True
    _bg_register("fresh", "fresh topic", "/tmp/f.md")
    snap = _bg_snapshot()
    assert len(snap) == 1
    assert snap[0]["id"] == "fresh"


def test_bg_snapshot_keeps_running_entries_regardless_of_age():
    """A long-running brainstorm shouldn't be pruned just because it's been
    going for >1h."""
    _bg_register("long-running", "huge topic", "/tmp/lr.md")
    _BG_BRAINSTORMS["long-running"]["started"] = time.time() - 7200
    # still done=False
    snap = _bg_snapshot()
    assert len(snap) == 1
