"""Tests for the four post-output quality guards on /brainstorm:

  1. Sequential agent letters (A, B, C, … not all 'P')
  2. Stable identity per persona across rounds (Faker re-roll bug)
  3. Anti-copy-paste detector (Jaccard similarity on CHALLENGE blocks)
  4. Weak-lead-model warning

Each was a bug in the user-reported transcript at
`brainstorm_outputs/brainstorm_20260509_000935.md`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws.commands.advanced import (
    _extract_challenge_blocks,
    _jaccard_similarity,
    _is_redundant_challenge,
    _is_weak_lead_model,
)


# ── Challenge-block extraction ───────────────────────────────────────────


def test_extract_challenge_blocks_finds_one():
    text = """### [CHALLENGE → Agent B]
> "TSLA will hit $300"
**Why this fails:** ignores demand.
**Counter:** $200 max.

### Defense by C
unrelated text"""
    blocks = _extract_challenge_blocks(text)
    assert len(blocks) == 1
    assert "tsla will hit" in blocks[0]
    assert "200 max" in blocks[0]


def test_extract_challenge_blocks_finds_multiple():
    text = """### [CHALLENGE → Agent A]
body 1 here

### [CHALLENGE → Agent B]
body 2 here
"""
    assert len(_extract_challenge_blocks(text)) == 2


def test_extract_challenge_blocks_empty_on_no_match():
    assert _extract_challenge_blocks("nothing here") == []
    assert _extract_challenge_blocks("") == []


# ── Jaccard similarity ──────────────────────────────────────────────────


def test_jaccard_identical_strings():
    s = "the quick brown fox jumps over the lazy dog"
    assert _jaccard_similarity(s, s) == 1.0


def test_jaccard_disjoint_strings():
    assert _jaccard_similarity("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_jaccard_partial_overlap():
    sim = _jaccard_similarity("the quick brown fox", "the quick red fox")
    # Tokens: {the, quick, brown, fox} ∩ {the, quick, red, fox} = 3
    #         {the, quick, brown, fox} ∪ {the, quick, red, fox} = 5
    # 3/5 = 0.6
    assert abs(sim - 0.6) < 0.01


def test_jaccard_empty_strings():
    assert _jaccard_similarity("", "anything") == 0.0
    assert _jaccard_similarity("anything", "") == 0.0


# ── Redundancy detection (the user's exact bug) ──────────────────────────


def test_redundant_challenge_catches_verbatim_clone():
    """The user's exact bug: persona B copy-pastes persona A's challenge
    word-for-word. Must be flagged with very high similarity."""
    original = """### [CHALLENGE → Agent X]
> "BABA will grow"
**Why this fails:** Pinduoduo competition.
**Counter:** flat to 2026."""
    clone = original  # exact copy
    redundant, sim = _is_redundant_challenge(clone, [original])
    assert redundant is True
    assert sim >= 0.95


def test_redundant_challenge_catches_near_clone():
    """A near-clone with one word changed (the qwen2.5 pattern) must
    still be flagged at the 0.7 threshold."""
    original = """### [CHALLENGE → Agent X]
> "BABA will grow"
**Why this fails:** Pinduoduo competition heating up.
**Counter:** flat to 2026."""
    near_clone = """### [CHALLENGE → Agent Y]
> "BABA will grow"
**Why this fails:** Pinduoduo competition is heating up substantially.
**Counter:** flat to 2026."""
    redundant, sim = _is_redundant_challenge(near_clone, [original])
    assert redundant is True
    assert 0.7 <= sim < 1.0


def test_redundant_challenge_passes_genuinely_different():
    """A genuinely different challenge on a different agent / claim must
    NOT be flagged. This protects the legitimate case where two personas
    independently challenge two different things."""
    original = """### [CHALLENGE → Agent A]
> "TSLA will hit $300"
**Why this fails:** macro headwinds ignored.
**Counter:** $180 base case."""
    different = """### [CHALLENGE → Agent B]
> "BABA will grow 20%"
**Why this fails:** Beijing regulatory overhang on cloud.
**Counter:** sub-5% revenue growth, BABA underperforms HSI."""
    redundant, sim = _is_redundant_challenge(different, [original])
    assert redundant is False
    assert sim < 0.7


def test_redundant_no_blocks_in_new_text_is_safe():
    """If the new persona response doesn't contain any CHALLENGE block
    (defense-only round 2 reply, etc.), there's nothing to compare."""
    original = "### [CHALLENGE → Agent A]\nbody"
    defense_only = "### [Agent B, round 2 defense]\nI maintain my position."
    redundant, _ = _is_redundant_challenge(defense_only, [original])
    assert redundant is False


# ── Weak-lead-model detection ────────────────────────────────────────────


@pytest.mark.parametrize("model_id,expected", [
    # Strong models — must NOT trigger
    ("claude-opus-4-7",                              False),
    ("claude-sonnet-4-6",                            False),
    ("gpt-5",                                        False),
    ("o1",                                           False),
    ("nim/deepseek-ai/deepseek-r1",                  False),
    ("deepseek-chat",                                False),
    ("nim/meta/llama-3.3-70b-instruct",              False),  # 70B is fine
    # Weak models — MUST trigger
    ("custom/qwen2.5-72b",                           True),   # qwen family
    ("qwen-max",                                     True),
    ("ollama/qwen2.5-coder",                         True),
    ("qwq-32b-preview",                              True),
    ("ollama/gemma-2-9b",                            True),
    ("ollama/phi-3-mini",                            True),
    ("custom/llama-3.2-3b",                          True),
    # Edge cases
    ("",                                             False),
])
def test_is_weak_lead_model(model_id, expected):
    assert _is_weak_lead_model(model_id) is expected


# ── Synthesis signature accepts opening param ────────────────────────────


def test_lead_synthesis_accepts_opening(monkeypatch):
    """Backward compat: opening is optional. Existing callers passing
    only the original 4 args must still work."""
    import cheetahclaws.commands.advanced as adv
    monkeypatch.setattr(adv, "_llm_oneshot",
                         lambda *a, **kw: "## Consensus\n- ok")
    # Without opening
    out1 = adv._lead_synthesis("topic", "transcript", "lead-model", {})
    assert "Consensus" in out1
    # With opening
    out2 = adv._lead_synthesis("topic", "transcript", "lead-model", {},
                                 opening="### Lead Opening\n- ban filler")
    assert "Consensus" in out2


def test_lead_synthesis_passes_opening_to_prompt(monkeypatch):
    """When opening is provided, it MUST appear in the user message so
    the model can self-check its action plan against the ban list."""
    captured = {}
    import cheetahclaws.commands.advanced as adv

    def fake(model, sys, user, config, **kw):
        captured["user"] = user
        return "## Consensus\n- ok"

    monkeypatch.setattr(adv, "_llm_oneshot", fake)
    adv._lead_synthesis("topic", "transcript", "lead-model", {},
                         opening="### Lead Opening\n- ban: consult an advisor")
    assert "ban: consult an advisor" in captured["user"]
    assert "SELF-CHECK" in captured["user"]
