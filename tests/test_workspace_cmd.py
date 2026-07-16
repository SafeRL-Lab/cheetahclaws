"""Unit tests for commands/workspace_cmd.

Coverage:
  * create / list / switch / delete round-trips against a tmp workspace root
  * current-workspace detection from cwd
  * startup workspace precedence (default > last > builtin fallback)
  * `default` and `switch` write DISTINCT config keys (regression: they used
    to share `workspace_last`, so setting a default was clobbered by a switch)
  * `_apply_workspace` is a no-op-safe boot helper that chdirs correctly

All filesystem state is redirected to tmp_path via the module-level
`_WORKSPACES_DIR` (computed from Path.home() at import time), and config
persistence is stubbed so tests never touch the real ~/.cheetahclaws.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws.commands import workspace_cmd as ws


@pytest.fixture
def wsroot(tmp_path, monkeypatch):
    """Redirect the workspace root to an isolated tmp dir and stub save_config.

    Restores the original cwd after each test (the command chdirs around).
    """
    root = tmp_path / "workspaces"
    monkeypatch.setattr(ws, "_WORKSPACES_DIR", root)
    # cmd_workspace / _activate_workspace import save_config from cheetahclaws.config
    import cheetahclaws.config as _cfg
    monkeypatch.setattr(_cfg, "save_config", lambda cfg: None)
    origin = Path.cwd()
    # Start outside any workspace so current-detection is unambiguous.
    monkeypatch.chdir(tmp_path)
    try:
        yield root
    finally:
        os.chdir(origin)


def _run(sub, config):
    return ws.cmd_workspace(sub, None, config)


# ── create / list ───────────────────────────────────────────────────────────

def test_create_and_list(wsroot):
    config = {}
    assert _run("create alpha", config) is True
    assert _run("create beta", config) is True
    assert (wsroot / "alpha").is_dir()
    assert ws._list_workspaces() == ["alpha", "beta"]


def test_create_requires_name(wsroot, capsys):
    assert _run("create", {}) is True
    assert not wsroot.exists() or not any(wsroot.iterdir())


# ── switch / current detection ───────────────────────────────────────────────

def test_switch_creates_and_chdirs(wsroot):
    config = {}
    _run("switch gamma", config)
    assert Path.cwd().resolve() == (wsroot / "gamma").resolve()
    # switch records last-used, NOT the sticky default
    assert config.get("workspace_last") == "gamma"
    assert config.get("workspace_default") is None


def test_current_workspace_detection(wsroot):
    config = {}
    _run("switch delta", config)
    assert ws._current_workspace_name() == "delta"


def test_current_none_outside_workspace(wsroot):
    assert ws._current_workspace_name() is None


# ── default vs switch: distinct keys (regression) ────────────────────────────

def test_default_and_switch_use_distinct_keys(wsroot):
    """Setting a default must survive a later switch."""
    config = {}
    _run("default proj", config)
    assert config["workspace_default"] == "proj"
    _run("switch scratch", config)
    # switch changed last-used but left the default intact
    assert config["workspace_default"] == "proj"
    assert config["workspace_last"] == "scratch"


def test_startup_precedence(wsroot):
    # default wins over last
    assert ws._startup_workspace(
        {"workspace_default": "d", "workspace_last": "l"}
    ) == "d"
    # last wins when no default
    assert ws._startup_workspace({"workspace_last": "l"}) == "l"
    # builtin fallback when neither set
    assert ws._startup_workspace({}) == ws._DEFAULT_WORKSPACE


# ── delete ───────────────────────────────────────────────────────────────────

def test_delete_empty_workspace(wsroot):
    config = {}
    _run("create trash", config)
    assert (wsroot / "trash").is_dir()
    _run("delete trash", config)
    assert not (wsroot / "trash").exists()


def test_cannot_delete_current_workspace(wsroot):
    config = {}
    _run("switch here", config)
    _run("delete here", config)
    # still present because we're inside it
    assert (wsroot / "here").exists()


def test_delete_missing_is_graceful(wsroot):
    assert _run("delete nope", {}) is True


# ── boot helper ──────────────────────────────────────────────────────────────

def test_apply_workspace_chdirs_to_default(wsroot):
    config = {"workspace_default": "boot"}
    ws._apply_workspace(config)
    assert Path.cwd().resolve() == (wsroot / "boot").resolve()


def test_apply_workspace_fallback_creates_builtin(wsroot):
    ws._apply_workspace({})
    assert (wsroot / ws._DEFAULT_WORKSPACE).is_dir()
    assert Path.cwd().resolve() == (wsroot / ws._DEFAULT_WORKSPACE).resolve()
