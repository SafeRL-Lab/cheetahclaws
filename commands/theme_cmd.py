"""
commands/theme_cmd.py - /theme slash command for CheetahClaws.

Usage:
  /theme              List available themes (current marked with *)
  /theme <name>       Apply a theme and persist it to config
"""
from __future__ import annotations

from ui.render import THEMES, apply_theme, clr, info, ok, warn, err


def cmd_theme(args: str, _state, config) -> bool:
    name = (args or "").strip()
    current = config.get("theme", "default")

    if not name:
        info("Available themes:")
        for t, palette in THEMES.items():
            marker = "*" if t == current else " "
            swatch = clr("  accent  ", "cyan") + clr("  warn  ", "yellow")
            line = f"  {marker} {t:<14} {swatch} ({palette['code']})"
            print(line)
        info("\nUsage: /theme <name>")
        return True

    if name not in THEMES:
        err(f"Unknown theme: {name}")
        info("Run /theme to list available themes.")
        return True

    if not apply_theme(name):
        err(f"Failed to apply theme: {name}")
        return True

    config["theme"] = name
    try:
        from cc_config import save_config
        save_config(config)
    except Exception as e:
        warn(f"Theme applied but could not be saved: {e}")

    ok(f"Theme set to {clr(name, 'cyan')}.")
    return True
