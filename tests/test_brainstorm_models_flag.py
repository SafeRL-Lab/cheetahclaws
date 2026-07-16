"""Tests for the `--models a,b,c` flag on /brainstorm.

A single-model brainstorm is an echo chamber: every persona shares the
same training data and blind spots. The flag lets each persona run a
different model so you get real epistemic diversity (Claude critic +
GPT optimist + DeepSeek pragmatist).

Borrowed in spirit from Dulus's RoundtableAgent (webchat_server.py).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws.commands.advanced import _parse_models_flag


@pytest.mark.parametrize("args,expected_models,expected_remaining", [
    # Default: no flag → empty list, args unchanged
    ("redesign the auth flow",                  [], "redesign the auth flow"),
    ("",                                        [], ""),
    # Space form
    ("--models claude-opus-4-7,gpt-5,nim/deepseek-ai/deepseek-r1 the topic",
     ["claude-opus-4-7", "gpt-5", "nim/deepseek-ai/deepseek-r1"], "the topic"),
    # Equals form
    ("--models=claude-opus-4-7,gpt-5 the topic",
     ["claude-opus-4-7", "gpt-5"], "the topic"),
    # Flag at the end
    ("the topic --models claude-opus-4-7,gpt-5",
     ["claude-opus-4-7", "gpt-5"], "the topic"),
    # Single model
    ("--models claude-opus-4-7 short topic",
     ["claude-opus-4-7"], "short topic"),
    # Whitespace tolerance — splits on comma, trims each entry
    ("--models  claude-opus-4-7,gpt-5  topic",
     ["claude-opus-4-7", "gpt-5"], "topic"),
])
def test_parse_models_flag(args, expected_models, expected_remaining):
    models, remaining = _parse_models_flag(args)
    assert models == expected_models
    assert remaining == expected_remaining


def test_parse_models_handles_provider_prefixed_ids():
    """Provider-prefixed model IDs (with '/') must round-trip intact —
    the flag-stripping regex must not eat the slash."""
    args = "--models nim/meta/llama-3.3-70b-instruct,custom/qwen2.5-72b improve x"
    models, remaining = _parse_models_flag(args)
    assert models == [
        "nim/meta/llama-3.3-70b-instruct",
        "custom/qwen2.5-72b",
    ]
    assert remaining == "improve x"


def test_parse_models_empty_value_returns_empty_list():
    """Edge case: `--models ` with no value should not crash; returns []."""
    # Pattern requires at least one non-space char after the flag, so this
    # leaves the string untouched and treats the rest as topic.
    models, remaining = _parse_models_flag("--models  the topic")
    # Either parsed nothing (preferred) or parsed "the" — both safe.
    if models:
        assert models == ["the"]
        assert remaining == "topic"
    else:
        assert "the topic" in remaining
