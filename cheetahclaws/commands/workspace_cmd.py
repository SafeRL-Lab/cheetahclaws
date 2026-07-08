"""commands/workspace_cmd.py — Workspace management for CheetahClaws.

Mirrors the Dulus /workspace slash command so users can switch, list, create,
and delete isolated working directories under ~/.cheetahclaws/workspaces.
"""
from __future__ import annotations

import os
from pathlib import Path

from cheetahclaws.ui.render import info, ok, err, warn


_WORKSPACES_DIR: Path = Path.home() / ".cheetahclaws" / "workspaces"
_DEFAULT_WORKSPACE: str = "workspace1"


def _workspace_path(name: str) -> Path:
    return _WORKSPACES_DIR / name


def _ensure_workspace(name: str) -> Path:
    """Create workspace dir if missing and return its path."""
    ws = _workspace_path(name)
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _list_workspaces() -> list[str]:
    if not _WORKSPACES_DIR.exists():
        return []
    return sorted(p.name for p in _WORKSPACES_DIR.iterdir() if p.is_dir())


def _current_workspace_name() -> str | None:
    """Return the workspace name if cwd is inside ~/.cheetahclaws/workspaces/."""
    try:
        cwd = Path.cwd().resolve()
        root = _WORKSPACES_DIR.resolve()
        if root in cwd.parents or cwd == root:
            rel = cwd.relative_to(root)
            first = rel.parts[0] if rel.parts else None
            if first and _workspace_path(first).is_dir():
                return first
    except Exception:
        pass
    return None


def _activate_workspace(name: str, config: dict) -> bool:
    """Change cwd into workspace, create it if missing, and persist as last used."""
    from cheetahclaws.config import save_config
    ws = _ensure_workspace(name)
    try:
        os.chdir(ws)
        config["workspace_last"] = name
        save_config(config)
        return True
    except Exception as e:
        err(f"Could not switch to workspace '{name}': {e}")
        return False


def _apply_workspace(config: dict) -> None:
    """At boot, move cwd into the last-used workspace (or workspace1)."""
    last = config.get("workspace_last") or _DEFAULT_WORKSPACE
    ws = _workspace_path(last)
    if not ws.exists():
        _ensure_workspace(last)
    try:
        os.chdir(ws)
        if config.get("verbose", False):
            info(f"Active workspace: {last}")
    except Exception as e:
        warn(f"Could not enter workspace '{last}': {e}")


def cmd_workspace(args: str, _state, config) -> bool:
    """Manage CheetahClaws workspaces under ~/.cheetahclaws/workspaces.

    /workspace                — show current workspace + cwd
    /workspace current        — same as above
    /workspace list           — list workspaces
    /workspace switch <name>  — change to workspace (creates if missing)
    /workspace default [name] — show or set the startup workspace
    /workspace create <name>  — create a workspace without switching
    /workspace delete <name>  — delete a workspace (must be empty)
    """
    from cheetahclaws.config import save_config
    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else "current"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if subcmd in ("current", "cwd", ""):
        current = _current_workspace_name()
        if current:
            info(f"Workspace: {current}")
        else:
            info("You are not inside a CheetahClaws workspace.")
        info(f"Working directory: {os.getcwd()}")
        return True

    if subcmd == "list":
        workspaces = _list_workspaces()
        current = _current_workspace_name()
        if not workspaces:
            info("No workspaces yet. Use /workspace create <name>.")
            return True
        info(f"Workspaces in {_WORKSPACES_DIR}:")
        for w in workspaces:
            mark = "  → " if w == current else "    "
            print(f"{mark}{w}")
        return True

    if subcmd == "switch":
        if not rest:
            err("Usage: /workspace switch <name>")
            return True
        name = rest.split()[0]
        if _activate_workspace(name, config):
            ok(f"Workspace switched to: {name}")
        return True

    if subcmd == "default":
        if not rest:
            current_default = config.get("workspace_last") or _DEFAULT_WORKSPACE
            info(f"Default workspace: {current_default}")
            return True
        name = rest.split()[0]
        _ensure_workspace(name)
        config["workspace_last"] = name
        save_config(config)
        ok(f"Default workspace set to: {name}")
        return True

    if subcmd == "create":
        if not rest:
            err("Usage: /workspace create <name>")
            return True
        name = rest.split()[0]
        _ensure_workspace(name)
        ok(f"Workspace created: {name}")
        return True

    if subcmd == "delete":
        if not rest:
            err("Usage: /workspace delete <name>")
            return True
        name = rest.split()[0]
        target = _workspace_path(name)
        if not target.exists():
            err(f"Workspace '{name}' does not exist.")
            return True
        current = _current_workspace_name()
        if name == current:
            err("You cannot delete the workspace you are currently in. Switch first with /workspace switch.")
            return True
        try:
            target.rmdir()
            ok(f"Workspace deleted: {name}")
        except OSError as e:
            err(f"Could not delete '{name}': {e}. Make sure it is empty.")
        return True

    err(f"Unknown subcommand: /workspace {subcmd}")
    return True
