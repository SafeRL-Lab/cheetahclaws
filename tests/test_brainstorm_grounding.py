"""Tests for /brainstorm --ground (research-grounded brainstorm).

The brainstorm pipeline is pure-reasoning by default — personas only
know what their model knew at training time, which makes it useless
for data-hungry topics (stocks, current events, recent news). The
--ground flag pre-fetches a /research brief and inlines top results
into the snapshot personas see, so they cite real sources instead of
hallucinating.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import commands.advanced as adv
from commands.advanced import (
    _parse_ground_flag,
    _format_grounding_brief,
    _fetch_grounding,
)


# ── --ground flag parsing ────────────────────────────────────────────────


@pytest.mark.parametrize("args,expected_n,expected_remaining", [
    # No flag → off
    ("the topic",                                    0,  "the topic"),
    ("",                                             0,  ""),
    # Bare --ground → default 15
    ("--ground the topic",                           15, "the topic"),
    ("the topic --ground",                           15, "the topic"),
    # --ground=N → exact N (clamped to [3, 50])
    ("--ground=20 the topic",                        20, "the topic"),
    ("--ground=30 redesign auth",                    30, "redesign auth"),
    # Bound clamping
    ("--ground=1 topic",                             3,  "topic"),    # clamp up
    ("--ground=999 topic",                           50, "topic"),    # clamp down
])
def test_parse_ground_flag(args, expected_n, expected_remaining):
    n, remaining = _parse_ground_flag(args)
    assert n == expected_n
    assert remaining == expected_remaining


def test_parse_ground_composes_with_other_flags():
    """All four flags can stack — --ground / --rounds / --lead / --models."""
    args = ("--ground=20 --rounds 3 --lead claude-opus-4-7 "
            "--models gpt-5,nim/deepseek-ai/deepseek-r1 stocks 2026")
    n, rest1 = _parse_ground_flag(args)
    rounds, rest2 = adv._parse_rounds_flag(rest1)
    lead, rest3 = adv._parse_lead_flag(rest2)
    models, rest4 = adv._parse_models_flag(rest3)
    assert n == 20
    assert rounds == 3
    assert lead == "claude-opus-4-7"
    assert models == ["gpt-5", "nim/deepseek-ai/deepseek-r1"]
    assert rest4 == "stocks 2026"


# ── Brief formatting ─────────────────────────────────────────────────────


@dataclass
class _FakeResult:
    source: str
    title: str
    url: str
    snippet: str = ""
    domain: str = "web"
    engagement_score: float = 0.0


@dataclass
class _FakeBrief:
    topic: str = ""
    results: list = field(default_factory=list)
    statuses: list = field(default_factory=list)


def test_format_grounding_brief_basic_shape():
    brief = _FakeBrief(results=[
        _FakeResult("arxiv", "Paper A", "https://arxiv.org/a", "abstract A",
                     "academic", 0.9),
        _FakeResult("hn",    "Story B", "https://news.ycomb/b", "discussion B",
                     "tech", 0.7),
    ])
    out = _format_grounding_brief(brief)
    assert "### GROUNDING DATA" in out
    assert "[1] (arxiv · academic)" in out
    assert "Paper A" in out
    assert "https://arxiv.org/a" in out
    assert "[2] (hn · tech)" in out
    # Citation guidance must be in the suffix
    assert "[N]" in out
    assert "do not invent" in out.lower() or "do NOT invent" in out


def test_format_grounding_brief_sorted_by_engagement_desc():
    """Results are re-sorted by engagement_score so the top entries
    fit into the char budget when truncation kicks in."""
    brief = _FakeBrief(results=[
        _FakeResult("low",  "Low",  "u1", engagement_score=0.1),
        _FakeResult("high", "High", "u2", engagement_score=0.9),
        _FakeResult("mid",  "Mid",  "u3", engagement_score=0.5),
    ])
    out = _format_grounding_brief(brief)
    # The high-score one must appear first.
    high_idx = out.find("[1]")
    assert "High" in out[high_idx:high_idx + 100]


def test_format_grounding_brief_respects_char_budget():
    """A huge brief must be truncated rather than blow context window."""
    big_results = [
        _FakeResult("src", f"Title {i}", f"https://x/{i}",
                     "x" * 300, "web", 0.5)
        for i in range(100)
    ]
    out = _format_grounding_brief(_FakeBrief(results=big_results),
                                     max_chars=2000)
    # Keep budget honest — header + suffix add some chars but the body
    # must be bounded by max_chars budget.
    body_only = out.split("### GROUNDING DATA", 1)[1].split("_When you", 1)[0]
    assert len(body_only) <= 2200, (
        f"grounding body {len(body_only)} chars exceeds 2200 (budget 2000 + slack)"
    )


def test_format_grounding_brief_empty_results():
    assert _format_grounding_brief(_FakeBrief(results=[])) == ""
    assert _format_grounding_brief(None) == ""


# ── Fetch graceful degradation ───────────────────────────────────────────


def test_fetch_grounding_returns_empty_on_research_exception(monkeypatch):
    """A flaky network or missing API keys must not break the brainstorm
    — _fetch_grounding swallows the exception and returns ""."""
    import research.aggregator as _agg

    def raising_research(**kw):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(_agg, "research", raising_research)
    out = _fetch_grounding("topic", 15, {})
    assert out == ""


def test_fetch_grounding_returns_empty_on_empty_brief(monkeypatch):
    """A brief with zero results (every source 429'd, etc.) is also a
    'no grounding' case — return empty so the brainstorm continues
    un-grounded with a logged warning, not a crash."""
    import research.aggregator as _agg
    monkeypatch.setattr(_agg, "research",
                         lambda **kw: _FakeBrief(results=[]))
    out = _fetch_grounding("topic", 15, {})
    assert out == ""


def test_fetch_grounding_returns_formatted_block_on_success(monkeypatch):
    """Happy path: research returns a brief, _fetch_grounding returns
    the formatted markdown ready to inline."""
    import research.aggregator as _agg
    fake_brief = _FakeBrief(results=[
        _FakeResult("arxiv", "Real Paper", "https://arxiv.org/x",
                     "real snippet", "academic", 0.9),
    ])
    monkeypatch.setattr(_agg, "research", lambda **kw: fake_brief)
    out = _fetch_grounding("topic", 15, {})
    assert "### GROUNDING DATA" in out
    assert "Real Paper" in out


# ── Lead synthesis: optional grounding param ─────────────────────────────


def test_lead_synthesis_passes_grounding_to_prompt(monkeypatch):
    """When grounding is provided, the synthesis prompt must include it
    AND the traceability instruction so consensus claims are tied to
    either grounding [N] or persona claims."""
    captured = {}

    def fake_oneshot(model, sys, user, config, **kw):
        captured["user"] = user
        return "## Consensus\n- ok"

    monkeypatch.setattr(adv, "_llm_oneshot", fake_oneshot)
    adv._lead_synthesis(
        "topic", "transcript", "lead-model", {},
        opening="### Lead Opening\n- ban filler",
        grounding="### GROUNDING DATA\n[1] real source",
    )
    assert "GROUNDING DATA" in captured["user"]
    # Traceability instruction
    assert "trace to either" in captured["user"]


def test_lead_synthesis_grounding_optional_backward_compat(monkeypatch):
    """Existing callers passing only the original args (no grounding) must
    still work — grounding defaults to empty string and the prompt skips
    the grounding section."""
    captured = {}
    monkeypatch.setattr(adv, "_llm_oneshot",
                         lambda m, s, u, c, **k: captured.setdefault("user", u) or "ok")
    adv._lead_synthesis("topic", "transcript", "lead-model", {})
    assert "GROUNDING DATA" not in captured["user"]
