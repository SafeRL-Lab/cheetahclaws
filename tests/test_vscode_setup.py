"""Unit tests for the VS Code terminal-title auto-setup.

The behavior that matters: pick the settings file the *editor actually reads*
— server-side Machine settings under Remote-SSH / WSL / devcontainers, local
User settings for a plain local install, and neither (instructions instead)
when the UI lives on a machine we can't touch.
"""

from __future__ import annotations

import json

import pytest

from cheetahclaws.ui import vscode_setup as vs


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Strip every VS Code marker and point HOME/CONFIG_DIR at a temp tree."""
    for var in ("TERM_PROGRAM", "VSCODE_AGENT_FOLDER", "VSCODE_GIT_ASKPASS_NODE",
                "VSCODE_GIT_ASKPASS_MAIN", "VSCODE_IPC_HOOK_CLI",
                "XDG_CONFIG_HOME", "APPDATA"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(vs.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(vs, "CONFIG_DIR", tmp_path / "cheetah")
    (tmp_path / "cheetah").mkdir()
    return home


def _make_server(home, name=".vscode-server"):
    root = home / name
    (root / "cli" / "servers" / "Stable-abc" / "server").mkdir(parents=True)
    (root / "data").mkdir()
    return root


# ── app detection ───────────────────────────────────────────────────────────

def test_no_vscode_env_means_no_app(monkeypatch):
    assert vs._vscode_app() is None


def test_cursor_is_distinguished_from_code(monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    monkeypatch.setenv("CURSOR_TRACE_ID", "x")
    assert vs._vscode_app() == "Cursor"


# ── remote server-root detection ────────────────────────────────────────────

def test_server_root_found_via_askpass_path(monkeypatch, _isolated_env):
    root = _make_server(_isolated_env)
    monkeypatch.setenv(
        "VSCODE_GIT_ASKPASS_NODE",
        str(root / "cli" / "servers" / "Stable-abc" / "server" / "node"),
    )
    assert vs._remote_server_root("Code") == root


def test_server_root_found_via_home_fallback(_isolated_env):
    root = _make_server(_isolated_env)
    assert vs._remote_server_root("Code") == root


def test_server_root_matches_forks(monkeypatch, _isolated_env):
    root = _make_server(_isolated_env, ".cursor-server")
    assert vs._remote_server_root("Cursor") == root
    assert vs._remote_server_root("Code") is None  # wrong fork's fallback


def test_server_root_ignores_unrelated_dirs_named_server(monkeypatch, _isolated_env):
    stray = _isolated_env / "my-server"
    stray.mkdir()  # no data/cli/bin/extensions inside
    monkeypatch.setenv("VSCODE_GIT_ASKPASS_NODE", str(stray / "node"))
    assert vs._remote_server_root("Code") is None


def test_local_install_is_not_mistaken_for_a_server(_isolated_env):
    (_isolated_env / ".config" / "Code" / "User").mkdir(parents=True)
    assert vs._remote_server_root("Code") is None


# ── target resolution ───────────────────────────────────────────────────────

def test_remote_target_is_machine_settings(_isolated_env):
    root = _make_server(_isolated_env)
    target, scope, why = vs._resolve_target("Code")
    assert scope == "remote" and why == ""
    assert target == root / "data" / "Machine" / "settings.json"


def test_local_target_is_user_settings_when_the_editor_dir_exists(_isolated_env):
    user_dir = _isolated_env / ".config" / "Code" / "User"
    user_dir.mkdir(parents=True)
    target, scope, why = vs._resolve_target("Code")
    assert scope == "local" and target == user_dir / "settings.json"


def test_no_target_when_no_local_install_and_no_server(_isolated_env):
    """Regression: never fabricate ~/.config/Code/User/settings.json.

    That file is what a Remote-SSH session used to write — a path the editor
    never reads, leaving the user with a silently broken tab title.
    """
    target, scope, why = vs._resolve_target("Code")
    assert target is None and scope == ""
    assert "no local Code install" in why
    assert not (_isolated_env / ".config" / "Code").exists()


def test_server_wins_over_a_local_settings_dir(_isolated_env):
    """Server-side run: the local User dir here is not the UI's settings."""
    root = _make_server(_isolated_env)
    (_isolated_env / ".config" / "Code" / "User").mkdir(parents=True)
    target, scope, _ = vs._resolve_target("Code")
    assert scope == "remote" and target.is_relative_to(root)


# ── marker: one attempt per target ──────────────────────────────────────────

def test_marker_is_target_scoped(_isolated_env):
    vs._mark_attempted("/a/settings.json")
    assert vs._already_attempted("/a/settings.json")
    assert not vs._already_attempted("/b/settings.json")


def test_legacy_timestamp_marker_triggers_exactly_one_retry(_isolated_env):
    """Machines stuck by the old bare-timestamp marker must retry once."""
    (vs.CONFIG_DIR / vs._MARKER).write_text("1753034346")
    assert not vs._already_attempted("/a/settings.json")
    vs._mark_attempted("/a/settings.json")
    assert vs._already_attempted("/a/settings.json")


# ── end-to-end through the auto path ────────────────────────────────────────

def test_auto_setup_writes_remote_machine_settings(monkeypatch, _isolated_env, capsys):
    root = _make_server(_isolated_env)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    vs.maybe_setup_vscode_terminal_title({})

    written = json.loads((root / "data" / "Machine" / "settings.json").read_text())
    assert written[vs._TITLE_KEY] == vs._TITLE_VAL
    out = capsys.readouterr().out
    assert "NEW terminal" in out
    # Second launch is silent and does not rewrite anything.
    before = (root / "data" / "Machine" / "settings.json").read_text()
    vs.maybe_setup_vscode_terminal_title({})
    assert (root / "data" / "Machine" / "settings.json").read_text() == before
    assert capsys.readouterr().out == ""


def test_auto_setup_preserves_existing_machine_settings(monkeypatch, _isolated_env):
    root = _make_server(_isolated_env)
    machine = root / "data" / "Machine"
    machine.mkdir(parents=True)
    (machine / "settings.json").write_text(
        '{\n    // keep me\n    "files.autoSave": "off"\n}\n')
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    vs.maybe_setup_vscode_terminal_title({})

    raw = (machine / "settings.json").read_text()
    assert "// keep me" in raw                      # JSONC comments survive
    parsed = json.loads(vs._strip_jsonc(raw))
    assert parsed["files.autoSave"] == "off"
    assert parsed[vs._TITLE_KEY] == vs._TITLE_VAL


def test_auto_setup_prints_instructions_when_it_cannot_write(monkeypatch, _isolated_env, capsys):
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    vs.maybe_setup_vscode_terminal_title({})

    out = capsys.readouterr().out
    assert vs._TITLE_KEY in out and vs._TITLE_VAL in out
    assert "/terminal-setup" in out
    assert not (_isolated_env / ".config").exists()   # nothing fabricated


def test_auto_setup_respects_terminal_title_false(monkeypatch, _isolated_env, capsys):
    root = _make_server(_isolated_env)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    vs.maybe_setup_vscode_terminal_title({"terminal_title": False})

    assert not (root / "data" / "Machine" / "settings.json").exists()
    assert capsys.readouterr().out == ""


def test_terminal_setup_command_reports_remote_scope(monkeypatch, _isolated_env, capsys):
    _make_server(_isolated_env)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    vs.run_terminal_setup()

    out = capsys.readouterr().out
    assert "Machine" in out or "remote" in out


def test_terminal_setup_command_outside_vscode_says_nothing_to_do(_isolated_env, capsys):
    vs.run_terminal_setup()
    assert "no setup needed" in capsys.readouterr().out
