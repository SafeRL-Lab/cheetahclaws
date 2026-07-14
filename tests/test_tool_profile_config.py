"""Compatibility and validation tests for tool-profile configuration."""
from __future__ import annotations

import json

import pytest

from cheetahclaws import config as config_module
from cheetahclaws.tool_registry import normalize_tool_profile


def test_legacy_saved_config_keeps_full_tool_surface(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "test"}), encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(config_module, "SESSIONS_DIR", tmp_path / "sessions")

    assert config_module.load_config()["tool_profile"] == "full"


def test_fresh_config_uses_compact_standard_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(config_module, "SESSIONS_DIR", tmp_path / "sessions")

    assert config_module.load_config()["tool_profile"] == "standard"


@pytest.mark.parametrize("value", [1, ["standard"], {"profile": "full"}])
def test_invalid_tool_profile_value_is_a_clean_validation_error(value):
    with pytest.raises(ValueError):
        normalize_tool_profile(value)
