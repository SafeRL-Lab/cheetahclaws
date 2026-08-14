"""Compatibility and validation tests for tool-profile configuration."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from cheetahclaws import config as config_module
from cheetahclaws.tool_registry import normalize_tool_profile


def test_legacy_saved_config_keeps_the_full_tool_surface(monkeypatch, tmp_path):
    # A config predating tool profiles must keep every capability it had before
    # (web, sub-agents, MCP, ...): upgrading never silently removes tools.
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "test"}), encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(config_module, "SESSIONS_DIR", tmp_path / "sessions")

    assert config_module.load_config()["tool_profile"] == "full"


def test_saved_full_profile_remains_an_explicit_opt_in(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"tool_profile": "full"}', encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(config_module, "SESSIONS_DIR", tmp_path / "sessions")

    assert config_module.load_config()["tool_profile"] == "full"


def test_fresh_config_uses_full_profile_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(config_module, "SESSIONS_DIR", tmp_path / "sessions")

    assert config_module.load_config()["tool_profile"] == "full"


def test_standard_profile_is_an_explicit_opt_in(monkeypatch, tmp_path):
    # Shrinking the surface to the compact coding set is opt-in, never implicit.
    config_file = tmp_path / "config.json"
    config_file.write_text('{"tool_profile": "standard"}', encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(config_module, "SESSIONS_DIR", tmp_path / "sessions")

    assert config_module.load_config()["tool_profile"] == "standard"


@pytest.mark.parametrize("value", [1, ["standard"], {"profile": "full"}])
def test_invalid_tool_profile_value_is_a_clean_validation_error(value):
    with pytest.raises(ValueError):
        normalize_tool_profile(value)


def test_web_session_exposes_and_updates_tool_profile(monkeypatch):
    from cheetahclaws.web import api

    persisted = {}
    fake_db = types.SimpleNamespace(
        repo=types.SimpleNamespace(
            upsert_session=lambda *args, **kwargs: persisted.update(kwargs),
        ),
    )
    import cheetahclaws.web as web_package
    monkeypatch.setattr(web_package, "db", fake_db, raising=False)

    session = api.ChatSession.__new__(api.ChatSession)
    session.config = {"tool_profile": "standard"}
    session.session_id = "profile-test"
    session.user_id = 1
    session.title = "Test"

    assert session.update_config({"tool_profile": "research"})["tool_profile"] == "research"
    assert session.config["tool_profile"] == "research"
    assert persisted["config"]["tool_profile"] == "research"

    with pytest.raises(ValueError):
        session.update_config({"tool_profile": "not-a-profile"})


def test_web_settings_expose_and_render_the_tool_profile_selector():
    root = Path(__file__).resolve().parent.parent
    markup = (root / "cheetahclaws/web/chat.html").read_text(encoding="utf-8")
    script = (root / "cheetahclaws/web/static/js/settings.js").read_text(encoding="utf-8")

    assert 'id="sp-tool-profile"' in markup
    assert "updateConfig('tool_profile', this.value)" in markup
    assert "sp-tool-profile').value = cfg.tool_profile || 'full'" in script


def test_terminal_config_rejects_invalid_tool_profile(monkeypatch):
    from cheetahclaws import config as config_module
    from cheetahclaws.commands.config_cmd import cmd_config

    monkeypatch.setattr(config_module, "save_config", lambda _config: None)
    config = {"tool_profile": "standard"}

    assert cmd_config("tool_profile=not-a-profile", None, config) is False
    assert config["tool_profile"] == "standard"
