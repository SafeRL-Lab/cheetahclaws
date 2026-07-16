"""Tests for external plugin discovery and dependency helpers."""
from __future__ import annotations

import json
import os

import pytest

from cheetahclaws.plugin import store
from cheetahclaws.plugin.store import (
    PLUGIN_PATH_ENV,
    _dep_distribution_name,
    _external_plugin_dirs,
    _missing_dependencies,
    _scan_external_plugins,
    disable_plugin,
    enable_plugin,
    get_plugin,
    list_plugins,
)
from cheetahclaws.plugin.types import PluginScope


@pytest.fixture(autouse=True)
def _isolate_user_cfg(tmp_path, monkeypatch):
    """Redirect the user-level config so tests don't touch ~/.cheetahclaws."""
    fake_cfg = tmp_path / "user-plugins.json"
    monkeypatch.setattr(store, "USER_PLUGIN_CFG", fake_cfg)
    yield
    # No cleanup needed — tmp_path is auto-removed.


def _write_plugin_json(dirpath, name, **extra):
    dirpath.mkdir(parents=True, exist_ok=True)
    data = {"name": name, "version": "0.1.0", "description": f"{name} plugin"}
    data.update(extra)
    (dirpath / "plugin.json").write_text(json.dumps(data))


class TestExternalPluginDirs:
    def test_empty_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(PLUGIN_PATH_ENV, raising=False)
        assert _external_plugin_dirs() == []

    def test_skips_nonexistent_and_empty_segments(self, monkeypatch, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        fake = tmp_path / "missing"
        value = os.pathsep.join(["", str(real), str(fake), ""])
        monkeypatch.setenv(PLUGIN_PATH_ENV, value)
        assert _external_plugin_dirs() == [real]

    def test_preserves_env_order(self, monkeypatch, tmp_path):
        a = tmp_path / "a"; a.mkdir()
        b = tmp_path / "b"; b.mkdir()
        monkeypatch.setenv(PLUGIN_PATH_ENV, f"{b}{os.pathsep}{a}")
        assert _external_plugin_dirs() == [b, a]


class TestScanExternalPlugins:
    def test_finds_plugin_json(self, monkeypatch, tmp_path):
        _write_plugin_json(tmp_path / "alpha", "alpha")
        monkeypatch.setenv(PLUGIN_PATH_ENV, str(tmp_path))
        plugins = _scan_external_plugins()
        assert [p.name for p in plugins] == ["alpha"]
        assert plugins[0].scope == PluginScope.EXTERNAL
        assert plugins[0].enabled is False

    def test_finds_plugin_md(self, monkeypatch, tmp_path):
        d = tmp_path / "beta"
        d.mkdir()
        (d / "PLUGIN.md").write_text(
            "---\nname: beta\nversion: 1.0\n---\nbody\n"
        )
        monkeypatch.setenv(PLUGIN_PATH_ENV, str(tmp_path))
        plugins = _scan_external_plugins()
        assert [p.name for p in plugins] == ["beta"]

    def test_skips_dir_without_manifest(self, monkeypatch, tmp_path):
        (tmp_path / "not_a_plugin").mkdir()
        monkeypatch.setenv(PLUGIN_PATH_ENV, str(tmp_path))
        assert _scan_external_plugins() == []

    def test_malformed_json_does_not_crash(self, monkeypatch, tmp_path, capsys):
        d = tmp_path / "broken"
        d.mkdir()
        (d / "plugin.json").write_text("{not valid json")
        _write_plugin_json(tmp_path / "ok", "ok")
        monkeypatch.setenv(PLUGIN_PATH_ENV, str(tmp_path))
        plugins = _scan_external_plugins()
        assert [p.name for p in plugins] == ["ok"]

    def test_skips_hidden_dirs(self, monkeypatch, tmp_path):
        _write_plugin_json(tmp_path / ".hidden", "hidden")
        _write_plugin_json(tmp_path / "visible", "visible")
        monkeypatch.setenv(PLUGIN_PATH_ENV, str(tmp_path))
        assert [p.name for p in _scan_external_plugins()] == ["visible"]

    def test_first_entry_wins_on_name_collision(self, monkeypatch, tmp_path):
        path_a = tmp_path / "a"; path_b = tmp_path / "b"
        _write_plugin_json(path_a / "dup", "dup", description="from-a")
        _write_plugin_json(path_b / "dup", "dup", description="from-b")
        monkeypatch.setenv(PLUGIN_PATH_ENV, f"{path_a}{os.pathsep}{path_b}")
        plugins = _scan_external_plugins()
        assert len(plugins) == 1
        assert plugins[0].manifest.description == "from-a"


class TestListPluginsDedup:
    def test_installed_shadows_external(self, monkeypatch, tmp_path):
        # External plugin "shared"
        _write_plugin_json(tmp_path / "shared", "shared")
        monkeypatch.setenv(PLUGIN_PATH_ENV, str(tmp_path))
        # Same name installed at USER scope
        store._write_cfg(store.USER_PLUGIN_CFG, {
            "plugins": {
                "shared": {
                    "name": "shared",
                    "scope": "user",
                    "source": "/somewhere",
                    "install_dir": str(tmp_path / "installed"),
                    "enabled": True,
                }
            }
        })
        plugins = list_plugins()
        scopes = [p.scope for p in plugins if p.name == "shared"]
        assert scopes == [PluginScope.USER]


class TestEnableDisablePersistence:
    def test_enable_writes_external_state(self, monkeypatch, tmp_path):
        _write_plugin_json(tmp_path / "foo", "foo")
        monkeypatch.setenv(PLUGIN_PATH_ENV, str(tmp_path))

        ok, _ = enable_plugin("foo")
        assert ok
        cfg = store._read_cfg(store.USER_PLUGIN_CFG)
        assert cfg["external_enabled"]["foo"] is True

        # List should reflect the enabled state
        entry = get_plugin("foo")
        assert entry is not None
        assert entry.scope == PluginScope.EXTERNAL
        assert entry.enabled is True

    def test_disable_writes_false(self, monkeypatch, tmp_path):
        _write_plugin_json(tmp_path / "foo", "foo")
        monkeypatch.setenv(PLUGIN_PATH_ENV, str(tmp_path))
        enable_plugin("foo")
        ok, _ = disable_plugin("foo")
        assert ok
        cfg = store._read_cfg(store.USER_PLUGIN_CFG)
        assert cfg["external_enabled"]["foo"] is False


class TestDepsHelpers:
    def test_distribution_name_strips_version(self):
        assert _dep_distribution_name("requests>=2.28") == "requests"
        assert _dep_distribution_name("package[extra]>=1.0") == "package"
        assert _dep_distribution_name("bare") == "bare"
        assert _dep_distribution_name("  padded == 1.0  ") == "padded"

    def test_missing_deps_recognises_installed_pypi_name(self):
        # pytest is definitely installed in the test environment —
        # PyPI name 'pytest', import name also 'pytest' (easy case).
        assert _missing_dependencies(["pytest"]) == []

    def test_missing_deps_reports_absent_package(self):
        missing = _missing_dependencies(["no_such_package_abc_xyz_999"])
        assert missing == ["no_such_package_abc_xyz_999"]

    def test_missing_deps_checks_pypi_name_not_import_name(self):
        # Regression for the PR #49 bug: find_spec('pillow') returns None
        # even when Pillow is installed (import name is PIL). We don't
        # require Pillow to actually be installed — we only require that
        # the check key off the PyPI (distribution) name, not the module
        # name. So an uninstalled PyPI name should come back as missing
        # regardless of whether a same-named importable module exists.
        # 'os' is an importable module but not a PyPI distribution:
        missing = _missing_dependencies(["os"])
        assert missing == ["os"]
