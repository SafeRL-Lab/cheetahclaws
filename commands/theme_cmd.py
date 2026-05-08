"""
commands/theme_cmd.py - /theme slash command for CheetahClaws.

Usage:
  /theme              List available themes (current marked with *)
  /theme <name>       Apply a theme and persist it to config
"""
from __future__ import annotations

from ui.render import THEMES, apply_theme, clr, info, ok, warn, err, _rgb


_RESET = "\033[0m"


def _render_swatch(palette: dict) -> str:
    """Render a per-palette preview using the palette's own colors."""
    if palette.get("disable_color"):
        return "  (no color)  "
    parts = []
    for label, key, fallback in [
        ("info", "accent", "#FFFFFF"),
        ("ok",   "ok",     palette.get("accent", "#FFFFFF")),
        ("warn", "warn",   "#FFFFFF"),
        ("err",  "err",    "#FF5555"),
    ]:
        hex_val = palette.get(key, fallback)
        parts.append(f"{_rgb(hex_val)} {label} {_RESET}")
    return "".join(parts)


def cmd_theme(args: str, _state, config) -> bool:
    name = (args or "").strip()
    current = config.get("theme", "default")

    if not name:
        info("Available themes:")
        for t, palette in THEMES.items():
            marker = "*" if t == current else " "
            line = f"  {marker} {t:<14} {_render_swatch(palette)}  ({palette['code']})"
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
