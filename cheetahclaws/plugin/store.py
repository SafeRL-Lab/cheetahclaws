"""Plugin store: install/uninstall/enable/disable/update + config persistence."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .types import PluginEntry, PluginManifest, PluginScope, parse_plugin_identifier, sanitize_plugin_name

# ── Config paths ──────────────────────────────────────────────────────────────

USER_PLUGIN_DIR  = Path.home() / ".cheetahclaws" / "plugins"
USER_PLUGIN_CFG  = Path.home() / ".cheetahclaws" / "plugins.json"

# Colon-separated list of dirs that hold plugin subdirs discovered in-place
# (no install/copy). External plugins are disabled by default — the user must
# run `/plugin enable <name>` once, which persists under "external_enabled"
# in the user config.
PLUGIN_PATH_ENV     = "CHEETAHCLAWS_PLUGIN_PATH"
_EXTERNAL_ENABLED_KEY = "external_enabled"

def _project_plugin_dir() -> Path:
    return Path.cwd() / ".cheetahclaws" / "plugins"

def _project_plugin_cfg() -> Path:
    return Path.cwd() / ".cheetahclaws" / "plugins.json"


# ── Config read/write ─────────────────────────────────────────────────────────

def _read_cfg(cfg_path: Path) -> dict:
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            pass
    return {"plugins": {}}


def _write_cfg(cfg_path: Path, data: dict) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(data, indent=2))


def _plugin_dir_for(scope: PluginScope) -> Path:
    return USER_PLUGIN_DIR if scope == PluginScope.USER else _project_plugin_dir()


def _plugin_cfg_for(scope: PluginScope) -> Path:
    # External plugins store their enable state in the user config
    if scope == PluginScope.PROJECT:
        return _project_plugin_cfg()
    return USER_PLUGIN_CFG


# ── External plugin discovery ────────────────────────────────────────────────

def _external_plugin_dirs() -> list[Path]:
    """Return existing directories listed in $CHEETAHCLAWS_PLUGIN_PATH."""
    raw = os.environ.get(PLUGIN_PATH_ENV, "")
    result: list[Path] = []
    for segment in raw.split(os.pathsep):
        if not segment:
            continue
        p = Path(segment).expanduser()
        if p.is_dir():
            result.append(p)
    return result


def _scan_external_plugins() -> list[PluginEntry]:
    """Discover plugins sitting in-place under $CHEETAHCLAWS_PLUGIN_PATH.

    Each immediate subdirectory with a plugin.json or PLUGIN.md counts.
    Malformed manifests are skipped with a warning on stderr — one bad
    file must not take down the CLI.
    """
    enabled_map = _read_cfg(USER_PLUGIN_CFG).get(_EXTERNAL_ENABLED_KEY, {})
    seen: set[str] = set()
    results: list[PluginEntry] = []
    for root in _external_plugin_dirs():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            try:
                manifest = PluginManifest.from_plugin_dir(child)
            except Exception as exc:
                print(f"[plugin] skipping {child}: {exc}", file=sys.stderr)
                continue
            if manifest is None:
                continue
            name = sanitize_plugin_name(manifest.name or child.name)
            if name in seen:
                # First occurrence wins (earlier entry in PLUGIN_PATH)
                continue
            seen.add(name)
            results.append(PluginEntry(
                name=name,
                scope=PluginScope.EXTERNAL,
                source=str(child.resolve()),
                install_dir=child,
                enabled=bool(enabled_map.get(name, False)),
                manifest=manifest,
            ))
    return results


# ── List ──────────────────────────────────────────────────────────────────────

def list_plugins(scope: PluginScope | None = None) -> list[PluginEntry]:
    """Return all installed plugins (optionally filtered by scope).

    External plugins (from $CHEETAHCLAWS_PLUGIN_PATH) are included unless a
    USER or PROJECT plugin already holds the same name (installed wins).
    """
    entries: list[PluginEntry] = []
    installed_scopes = [PluginScope.USER, PluginScope.PROJECT]
    scopes = installed_scopes if scope is None else [scope]
    installed_names: set[str] = set()
    for sc in scopes:
        if sc == PluginScope.EXTERNAL:
            continue
        cfg = _read_cfg(_plugin_cfg_for(sc))
        for name, data in cfg.get("plugins", {}).items():
            entry = PluginEntry.from_dict(data)
            entry.manifest = PluginManifest.from_plugin_dir(entry.install_dir)
            entries.append(entry)
            installed_names.add(entry.name)

    if scope is None or scope == PluginScope.EXTERNAL:
        for ext in _scan_external_plugins():
            if ext.name in installed_names:
                continue
            entries.append(ext)
    return entries


def get_plugin(name: str, scope: PluginScope | None = None) -> PluginEntry | None:
    for entry in list_plugins(scope):
        if entry.name == name:
            return entry
    return None


# ── Install ───────────────────────────────────────────────────────────────────

def install_plugin(
    identifier: str,
    scope: PluginScope = PluginScope.USER,
    force: bool = False,
) -> tuple[bool, str]:
    """
    Install a plugin. identifier = 'name' | 'name@git_url' | 'name@local_path'.
    Returns (success, message).
    """
    name, source = parse_plugin_identifier(identifier)
    safe_name = sanitize_plugin_name(name)

    # Check if already installed
    existing = get_plugin(safe_name, scope)
    if existing and not force:
        return False, f"Plugin '{safe_name}' is already installed in {scope.value} scope. Use --force to reinstall."

    plugin_dir = _plugin_dir_for(scope) / safe_name

    try:
        if source is None:
            # No source → treat name as a local path if it exists, else error
            local = Path(name)
            if local.exists() and local.is_dir():
                source = str(local.resolve())
            else:
                return False, (
                    f"No source specified for '{name}'. "
                    "Provide 'name@git_url' or 'name@/local/path'."
                )

        # Install from local path or git
        if plugin_dir.exists() and force:
            shutil.rmtree(plugin_dir)

        if _is_git_url(source):
            ok, msg = _clone_plugin(source, plugin_dir)
            if not ok:
                return False, msg
        else:
            local_src = Path(source)
            if not local_src.exists():
                return False, f"Local path not found: {source}"
            shutil.copytree(str(local_src), str(plugin_dir))

        # Load and validate manifest
        manifest = PluginManifest.from_plugin_dir(plugin_dir)
        if manifest is None:
            manifest = PluginManifest(name=safe_name, description="(no manifest)")

        # Install pip dependencies
        if manifest.dependencies:
            dep_ok, dep_msg = _install_dependencies(manifest.dependencies)
            if not dep_ok:
                return False, dep_msg

        # Persist to config
        entry = PluginEntry(
            name=safe_name,
            scope=scope,
            source=source,
            install_dir=plugin_dir,
            enabled=True,
            manifest=manifest,
        )
        _save_entry(entry)
        return True, f"Plugin '{safe_name}' installed successfully ({scope.value} scope)."

    except Exception as e:
        return False, f"Install failed: {e}"


def _is_git_url(source: str) -> bool:
    return (
        source.startswith("https://")
        or source.startswith("git@")
        or source.startswith("http://")
        or source.endswith(".git")
    )


def _clone_plugin(url: str, dest: Path) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"git clone failed: {result.stderr.strip()}"
    return True, "cloned"


def _dep_distribution_name(spec: str) -> str:
    """Extract the bare distribution name from a PEP 508 requirement."""
    import re
    return re.split(r"[>=<!~;\[\s]", spec.strip(), maxsplit=1)[0].strip()


def _missing_dependencies(deps: list[str]) -> list[str]:
    """Return the subset of deps not already installed, keyed by distribution name.

    Uses importlib.metadata so the PyPI name is what matters — sidesteps the
    PyPI-vs-import-name trap (Pillow/PIL, PyYAML/yaml, etc.).
    """
    from importlib.metadata import PackageNotFoundError, distribution
    missing: list[str] = []
    for spec in deps:
        dist_name = _dep_distribution_name(spec)
        if not dist_name:
            continue
        try:
            distribution(dist_name)
        except PackageNotFoundError:
            missing.append(spec)
    return missing


def _install_dependencies(deps: list[str]) -> tuple[bool, str]:
    """Install deps via pip — caller is expected to have informed-consent
    from the user (explicit `install` / `enable`). Never call implicitly."""
    missing = _missing_dependencies(deps)
    if not missing:
        return True, "deps already satisfied"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"pip install failed: {result.stderr.strip()}"
    return True, f"installed: {', '.join(missing)}"


def _save_entry(entry: PluginEntry) -> None:
    cfg_path = _plugin_cfg_for(entry.scope)
    data = _read_cfg(cfg_path)
    if entry.scope == PluginScope.EXTERNAL:
        # Externals aren't stored as full entries — only their enable state is.
        data.setdefault(_EXTERNAL_ENABLED_KEY, {})[entry.name] = bool(entry.enabled)
    else:
        data.setdefault("plugins", {})[entry.name] = entry.to_dict()
    _write_cfg(cfg_path, data)


def _remove_entry(name: str, scope: PluginScope) -> None:
    cfg_path = _plugin_cfg_for(scope)
    data = _read_cfg(cfg_path)
    if scope == PluginScope.EXTERNAL:
        data.get(_EXTERNAL_ENABLED_KEY, {}).pop(name, None)
    else:
        data.get("plugins", {}).pop(name, None)
    _write_cfg(cfg_path, data)


# ── Uninstall ─────────────────────────────────────────────────────────────────

def uninstall_plugin(
    name: str,
    scope: PluginScope | None = None,
    keep_data: bool = False,
) -> tuple[bool, str]:
    entry = get_plugin(name, scope)
    if entry is None:
        return False, f"Plugin '{name}' not found."
    # External plugins live on user-owned paths we never copied; only drop
    # our enable-state record, never touch the source directory.
    if entry.scope != PluginScope.EXTERNAL:
        if not keep_data and entry.install_dir.exists():
            shutil.rmtree(entry.install_dir)
    _remove_entry(entry.name, entry.scope)
    return True, f"Plugin '{name}' uninstalled."


# ── Enable / Disable ──────────────────────────────────────────────────────────

def _set_enabled(name: str, scope: PluginScope | None, enabled: bool) -> tuple[bool, str]:
    entry = get_plugin(name, scope)
    if entry is None:
        return False, f"Plugin '{name}' not found."
    # External plugins aren't installed, so their declared deps haven't been
    # pip-installed yet. Enabling is the user's explicit consent to do so.
    if enabled and entry.scope == PluginScope.EXTERNAL and entry.manifest and entry.manifest.dependencies:
        missing = _missing_dependencies(entry.manifest.dependencies)
        if missing:
            dep_ok, dep_msg = _install_dependencies(missing)
            if not dep_ok:
                return False, f"Cannot enable '{name}': {dep_msg}"
    entry.enabled = enabled
    _save_entry(entry)
    state = "enabled" if enabled else "disabled"
    return True, f"Plugin '{name}' {state}."


def enable_plugin(name: str, scope: PluginScope | None = None) -> tuple[bool, str]:
    return _set_enabled(name, scope, True)


def disable_plugin(name: str, scope: PluginScope | None = None) -> tuple[bool, str]:
    return _set_enabled(name, scope, False)


def disable_all_plugins(scope: PluginScope | None = None) -> tuple[bool, str]:
    entries = list_plugins(scope)
    if not entries:
        return True, "No plugins to disable."
    for entry in entries:
        entry.enabled = False
        _save_entry(entry)
    return True, f"Disabled {len(entries)} plugin(s)."


# ── Update ────────────────────────────────────────────────────────────────────

def update_plugin(name: str, scope: PluginScope | None = None) -> tuple[bool, str]:
    entry = get_plugin(name, scope)
    if entry is None:
        return False, f"Plugin '{name}' not found."
    if entry.scope == PluginScope.EXTERNAL:
        return False, f"Plugin '{name}' is external — update the source directory directly."
    if not _is_git_url(entry.source):
        return False, f"Plugin '{name}' was installed from a local path; cannot auto-update."
    if not entry.install_dir.exists():
        return False, f"Install directory missing: {entry.install_dir}"
    result = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=str(entry.install_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"git pull failed: {result.stderr.strip()}"
    # Re-install dependencies if manifest changed
    manifest = PluginManifest.from_plugin_dir(entry.install_dir)
    if manifest and manifest.dependencies:
        _install_dependencies(manifest.dependencies)
    return True, f"Plugin '{name}' updated. {result.stdout.strip()}"
