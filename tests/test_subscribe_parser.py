"""Tests for /subscribe argument parsing.

The previous parser took the first whitespace token as the topic and
dropped the rest, so trend-track on "Agent OS Benchmark" subscribed
the topic "research:7d:Agent" — silently corrupting the user's intent.

The new rule: schedule is the LAST non-flag token if it matches a known
schedule keyword; otherwise default. Everything between is the topic.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws.commands.monitor_cmd import _parse_subscribe_args


@pytest.mark.parametrize("args,expected_topic,expected_schedule,expected_channels", [
    # Single-word topic, default schedule
    ("ai_research",                                 "ai_research",                         "daily",  []),
    # Single-word topic with explicit schedule
    ("ai_research weekly",                          "ai_research",                         "weekly", []),
    # Multi-word custom topic + schedule (regression: was "custom:quantum")
    ("custom:quantum computing weekly",             "custom:quantum computing",            "weekly", []),
    # The reported bug: trend-track on "Agent OS Benchmark"
    ("research:7d:Agent OS Benchmark daily",        "research:7d:Agent OS Benchmark",      "daily",  []),
    # Same topic without explicit schedule → default
    ("research:7d:Agent OS Benchmark",              "research:7d:Agent OS Benchmark",      "daily",  []),
    # Multi-word topic with no schedule, just channels
    ("research:7d:foo bar --telegram",              "research:7d:foo bar",                 "daily",  ["telegram"]),
    # Channels mixed in middle (order should not matter)
    ("--slack research:7d:foo bar weekly",          "research:7d:foo bar",                 "weekly", ["slack"]),
    # Stock/crypto single-token topics still work
    ("stock_TSLA",                                  "stock_TSLA",                          "daily",  []),
    ("crypto_BTC 6h --telegram",                    "crypto_BTC",                          "6h",     ["telegram"]),
    # Edge: only flags → topic is None
    ("--telegram",                                  None,                                   "daily",  ["telegram"]),
    # Edge: empty args
    ("",                                            None,                                   "daily",  []),
])
def test_parse_subscribe_args(args, expected_topic, expected_schedule, expected_channels):
    topic, schedule, channels = _parse_subscribe_args(args)
    assert topic == expected_topic, f"topic for {args!r}"
    assert schedule == expected_schedule, f"schedule for {args!r}"
    assert channels == expected_channels, f"channels for {args!r}"


def test_schedule_keyword_in_middle_of_topic_NOT_extracted():
    """A schedule keyword in the middle of a multi-word topic must stay
    in the topic — only the LAST token is candidate for being a schedule.
    Otherwise topics like 'weekly_report' or 'daily standup metrics' would
    get mangled."""
    # 'weekly' here is part of the topic, the LAST token isn't a schedule
    topic, schedule, _ = _parse_subscribe_args("research:7d:weekly market data")
    assert topic == "research:7d:weekly market data"
    assert schedule == "daily"
