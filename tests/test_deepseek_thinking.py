"""Regression tests for DeepSeek v4 thinking-mode toggle semantics.

The bug: `providers.py` uses `config.get("thinking") is False` to decide whether
to inject `extra_body={"thinking":{"type":"disabled"}}`.  If `config.DEFAULTS`
sets `"thinking": False`, every default user gets thinking disabled — opposite
of the intended "provider default (ON)" stance.  These tests pin the contract
so the default cannot regress to a strict bool again.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws.config import DEFAULTS


class TestThinkingDefault:
    def test_default_is_none_not_false(self):
        # Tri-state sentinel: None = unset, True = ON, False = explicit OFF.
        # Must be None so the DeepSeek `is False` check doesn't fire by default.
        assert DEFAULTS["thinking"] is None
        assert DEFAULTS["thinking"] is not False

    def test_default_is_falsy_for_anthropic_path(self):
        # Anthropic path uses `if config.get("thinking"):` — None must be falsy
        # so existing Anthropic behaviour (thinking off unless toggled) survives.
        assert not DEFAULTS["thinking"]


class TestThinkingDisablePredicate:
    """Mirrors providers.py:627 — `if config.get("thinking") is False:`."""

    @staticmethod
    def _should_disable(config: dict) -> bool:
        return config.get("thinking") is False

    def test_unset_does_not_disable(self):
        assert self._should_disable({}) is False

    def test_none_does_not_disable(self):
        assert self._should_disable({"thinking": None}) is False

    def test_true_does_not_disable(self):
        assert self._should_disable({"thinking": True}) is False

    def test_explicit_false_disables(self):
        assert self._should_disable({"thinking": False}) is True


class TestThinkingToggle:
    """Mirrors commands/config_cmd.py:139 — first /thinking from default → ON."""

    @staticmethod
    def _toggle(config: dict) -> bool:
        return not config.get("thinking", False)

    def test_first_toggle_from_default_goes_on(self):
        # Default config has thinking=None.  `not None` is True.
        config = dict(DEFAULTS)
        assert self._toggle(config) is True

    def test_toggle_from_on_goes_off(self):
        assert self._toggle({"thinking": True}) is False

    def test_toggle_from_off_goes_on(self):
        assert self._toggle({"thinking": False}) is True
