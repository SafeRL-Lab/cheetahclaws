"""Plugin loader: discover and load tools/skills/mcp from installed plugins."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from .store import list_plugins
from .types import PluginEntry, PluginScope


def _plugins_disabled() -> bool:
    return os.environ.get("CHEETAHCLAWS_DISABLE_PLUGINS", "0") == "1"


def _plugin_allowlist() -> set[str] | None:
    raw = os.environ.get("CHEETAHCLAWS_PLUGIN_ALLOWLIST", "").strip()
    if not raw:
        return None
    return {name.strip() for name in raw.split(",") if name.strip()}


def load_all_plugins(scope: PluginScope | None = None) -> list[PluginEntry]:
    """Return enabled plugins (optionally filtered by scope).

    Gated by two env vars (defense in depth — plugins run arbitrary Python):
      * CHEETAHCLAWS_DISABLE_PLUGINS=1 — load nothing.
      * CHEETAHCLAWS_PLUGIN_ALLOWLIST=a,b,c — load only the named plugins.
    """
    if _plugins_disabled():
        return []
    plugins = [p for p in list_plugins(scope) if p.enabled]
    allow = _plugin_allowlist()
    if allow is not None:
        plugins = [p for p in plugins if p.name in allow]
    return plugins


def load_plugin_tools(scope: PluginScope | None = None) -> list[dict]:
    """
    Import tool modules from all enabled plugins and collect their TOOL_SCHEMAS.
    Returns combined list of tool schema dicts.
    """
    schemas: list[dict] = []
    for entry in load_all_plugins(scope):
        if not entry.manifest or not entry.manifest.tools:
            continue
        for module_name in entry.manifest.tools:
            mod = _import_plugin_module(entry, module_name)
            if mod and hasattr(mod, "TOOL_SCHEMAS"):
                schemas.extend(mod.TOOL_SCHEMAS)
    return schemas


def register_plugin_tools(scope: PluginScope | None = None) -> int:
    """
    Import tool modules from enabled plugins and register them into tool_registry.
    Returns number of tools registered.
    """
    from tool_registry import register_tool, ToolDef
    count = 0
    for entry in load_all_plugins(scope):
        if not entry.manifest or not entry.manifest.tools:
            continue
        for module_name in entry.manifest.tools:
            mod = _import_plugin_module(entry, module_name)
            if mod is None:
                continue
            # Register each ToolDef exported by the module
            if hasattr(mod, "TOOL_DEFS"):
                for tdef in mod.TOOL_DEFS:
                    register_tool(tdef)
                    count += 1
    return count


def load_plugin_skills(scope: PluginScope | None = None) -> list[Path]:
    """Return paths to skill markdown files from enabled plugins."""
    paths: list[Path] = []
    for entry in load_all_plugins(scope):
        if not entry.manifest or not entry.manifest.skills:
            continue
        for skill_rel in entry.manifest.skills:
            skill_path = entry.install_dir / skill_rel
            if skill_path.exists():
                paths.append(skill_path)
    return paths


def load_plugin_commands(scope: PluginScope | None = None) -> dict[str, dict]:
    """
    Import command modules from enabled plugins and collect their COMMAND_DEFS.

    Returns a merged dict:
        { "command_name": {"func": callable, "help": (desc, [aliases]), "aliases": [...]} }

    Plugin authors expose commands by adding a COMMAND_DEFS dict to their module.
    Example module (video/cmd.py):
        COMMAND_DEFS = {
            "video": {
                "func":    cmd_video,
                "help":    ("AI video factory", ["status"]),
                "aliases": [],
            }
        }
    """
    result: dict[str, dict] = {}
    for entry in load_all_plugins(scope):
        if not entry.manifest or not entry.manifest.commands:
            continue
        for module_name in entry.manifest.commands:
            mod = _import_plugin_module(entry, module_name)
            if mod and hasattr(mod, "COMMAND_DEFS"):
                for cmd_name, cmd_def in mod.COMMAND_DEFS.items():
                    result[cmd_name] = cmd_def
    return result


def load_plugin_mcp_configs(scope: PluginScope | None = None) -> dict:
    """Return mcp server configs contributed by enabled plugins."""
    configs: dict = {}
    for entry in load_all_plugins(scope):
        if not entry.manifest or not entry.manifest.mcp_servers:
            continue
        for server_name, server_cfg in entry.manifest.mcp_servers.items():
            # Prefix server name with plugin name to avoid collisions
            qualified = f"{entry.name}__{server_name}"
            configs[qualified] = server_cfg
    return configs


_warned_once: set[str] = set()


def _warn_external_once(entry: PluginEntry) -> None:
    """One-line stderr notice the first time an EXTERNAL-scope plugin is loaded.

    EXTERNAL plugins come from $CHEETAHCLAWS_PLUGIN_PATH and may live anywhere;
    surface this in the log so a stolen env var doesn't load code silently.
    """
    if entry.scope is not PluginScope.EXTERNAL:
        return
    key = f"{entry.name}@{entry.install_dir}"
    if key in _warned_once:
        return
    _warned_once.add(key)
    print(
        f"[plugin] Loading EXTERNAL plugin '{entry.name}' from "
        f"{entry.install_dir} — this executes arbitrary Python with your "
        f"privileges. Set CHEETAHCLAWS_DISABLE_PLUGINS=1 or "
        f"CHEETAHCLAWS_PLUGIN_ALLOWLIST=<names> to restrict.",
        file=sys.stderr, flush=True,
    )


def _import_plugin_module(entry: PluginEntry, module_name: str):
    """Dynamically import a module from a plugin directory."""
    _warn_external_once(entry)

    # Resolve and confine the module path inside the plugin's install_dir so
    # a malicious manifest cannot reach for `../../etc/passwd_loader.py` or
    # similar.
    install_dir = entry.install_dir.resolve()
    plugin_dir_str = str(install_dir)
    if plugin_dir_str not in sys.path:
        sys.path.insert(0, plugin_dir_str)

    # Build a unique module name to avoid collisions
    unique_name = f"_plugin_{entry.name}_{module_name}"
    if unique_name in sys.modules:
        return sys.modules[unique_name]

    # Try as a file
    candidates = [
        install_dir / f"{module_name}.py",
        install_dir / module_name / "__init__.py",
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(install_dir)
        except (ValueError, OSError):
            continue
        if resolved.exists():
            spec = importlib.util.spec_from_file_location(unique_name, resolved)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[unique_name] = mod
                try:
                    spec.loader.exec_module(mod)
                    return mod
                except Exception as e:
                    print(f"[plugin] Failed to load {module_name} from {entry.name}: {e}")
                    del sys.modules[unique_name]
    return None
