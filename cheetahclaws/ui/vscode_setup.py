"""One-time VS Code / Cursor / Windsurf terminal-title setup.

CheetahClaws emits OSC 0 titles (a pulsing task while working) that show up
out-of-the-box in iTerm2 / Terminal.app, but VS Code's integrated terminal
hides program-set titles by default — its tab shows ``${process}`` and the
program title lands in ``${sequence}``, which the default template ignores.
Flipping ``terminal.integrated.tabs.title`` to include ``${sequence}`` makes
the tab reflect what CheetahClaws is doing.

Doing that by hand is a papercut, so this module offers to do it for the user
the first time they run inside a VS Code-family terminal. It is deliberately
conservative:

* runs at most once (a marker file under ~/.cheetahclaws), so we never nag;
* never overwrites a value the user already set for that key;
* backs the file up before writing;
* inserts the key textually (preserving comments / formatting of a JSONC
  settings file) and then re-parses the result, aborting if the edit would
  drop a key, change a value, or fail to parse — so a weird settings file is
  left untouched rather than corrupted;
* swallows every error: a nicety must never break startup.

The new setting only applies to terminals opened AFTER it is written, so the
current session still won't show it — the announced message says as much.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from cheetahclaws.config import CONFIG_DIR

_TITLE_KEY = "terminal.integrated.tabs.title"
_TITLE_VAL = "${sequence}${separator}${process}"
_MARKER = "vscode_terminal_title.done"


# ── editor / path detection ────────────────────────────────────────────────

def _vscode_app() -> str | None:
    """Return the VS Code-family app dir name ('Code'/'Cursor'/'Windsurf')
    when running inside its integrated terminal, else None. Cursor and
    Windsurf are VS Code forks and both export TERM_PROGRAM=vscode, so the
    specific fork is inferred from its env markers."""
    if os.environ.get("TERM_PROGRAM") != "vscode":
        return None
    blob = " ".join(f"{k}={v}" for k, v in os.environ.items()
                    if k.startswith(("VSCODE", "CURSOR", "WINDSURF"))).lower()
    if "cursor" in blob:
        return "Cursor"
    if "windsurf" in blob:
        return "Windsurf"
    return "Code"


def _settings_path(app: str) -> Path | None:
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support" / app / "User"
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return None
        base = Path(appdata) / app / "User"
    else:
        cfg = os.environ.get("XDG_CONFIG_HOME") or str(home / ".config")
        base = Path(cfg) / app / "User"
    return base / "settings.json"


# ── JSONC helpers ──────────────────────────────────────────────────────────

def _strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments (outside strings) and trailing commas,
    yielding text that ``json.loads`` accepts. Used only to validate."""
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    quote = ""
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                in_str = False
            i += 1
            continue
        if c in "\"'":
            in_str = True
            quote = c
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _find_object_start(text: str) -> int:
    """Index of the first structural '{' (skipping comments and strings)."""
    i, n = 0, len(text)
    in_str = False
    quote = ""
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                in_str = False
            i += 1
            continue
        if c in "\"'":
            in_str = True
            quote = c
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c == "{":
            return i
        i += 1
    return -1


def _apply_to_settings(path: Path) -> tuple[bool, str]:
    """Insert the title key into ``path``. Returns (changed, message).

    Copy-on-write with a re-parse safety net: the edited text must parse to
    exactly the old object plus our single key, or we abort and leave the
    original file untouched.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{\n    "%s": "%s"\n}\n' % (_TITLE_KEY, _TITLE_VAL))
        return True, f"created {path}"

    raw = path.read_text()
    try:
        old = json.loads(_strip_jsonc(raw)) if raw.strip() else {}
    except Exception:
        return False, f"skipped: couldn't parse {path.name} (edit it by hand)"
    if not isinstance(old, dict):
        return False, "skipped: settings root is not a JSON object"
    if _TITLE_KEY in old:
        return False, "already configured (left as-is)"

    bi = _find_object_start(raw)
    if bi == -1:
        return False, "skipped: no JSON object found"
    rest = raw[bi + 1:]
    if rest.lstrip().startswith("}"):          # empty object → no trailing comma
        entry = f'\n    "{_TITLE_KEY}": "{_TITLE_VAL}"\n'
    else:
        entry = f'\n    "{_TITLE_KEY}": "{_TITLE_VAL}",'
    candidate = raw[:bi + 1] + entry + rest

    try:
        new = json.loads(_strip_jsonc(candidate))
    except Exception:
        return False, "skipped: edit would not parse (left untouched)"
    if new != {**old, _TITLE_KEY: _TITLE_VAL}:
        return False, "skipped: safety check failed (left untouched)"

    backup = path.with_name(path.name + f".bak-cheetah-{int(time.time())}")
    backup.write_text(raw)
    path.write_text(candidate)
    return True, f"updated {path.name} (backup: {backup.name})"


# ── public entry points ────────────────────────────────────────────────────

def _print(msg: str) -> None:
    try:
        from cheetahclaws.ui.render import clr
        print(clr("  ⚙ " + msg, "dim"))
    except Exception:
        print("  " + msg)


def maybe_setup_vscode_terminal_title(config: dict) -> None:
    """Auto-run once on first launch inside a VS Code-family terminal.

    No-op unless: terminal_title is enabled, we're in VS Code/Cursor/Windsurf,
    and we haven't already tried. Any failure is swallowed."""
    try:
        if not config.get("terminal_title", True):
            return
        app = _vscode_app()
        if not app:
            return
        marker = CONFIG_DIR / _MARKER
        if marker.exists():
            return
        path = _settings_path(app)
        changed, msg = (False, "no settings path") if path is None \
            else _apply_to_settings(path)
        # Mark as attempted regardless, so we never re-touch the file on later
        # launches (manual /terminal-setup remains available to re-run).
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(int(time.time())))
        if changed:
            _print(f"Set up {app} terminal tab titles — {msg}.")
            _print("Reopen the terminal to see the task in the tab. "
                   "Disable any time with /config terminal_title=false.")
    except Exception:
        pass


def run_terminal_setup(force: bool = False) -> None:
    """Backing the /terminal-setup command: re-run the setup on demand and
    report clearly, ignoring the one-shot marker."""
    app = _vscode_app()
    if not app:
        _print("This terminal shows program titles natively (iTerm2 / "
               "Terminal.app / most terminals) — no setup needed.")
        _print("Nothing to configure here.")
        return
    path = _settings_path(app)
    if path is None:
        _print(f"Couldn't locate {app} settings.json on this platform.")
        return
    changed, msg = _apply_to_settings(path)
    # Refresh the marker so the auto-path stays quiet afterwards.
    try:
        (CONFIG_DIR / _MARKER).parent.mkdir(parents=True, exist_ok=True)
        (CONFIG_DIR / _MARKER).write_text(str(int(time.time())))
    except Exception:
        pass
    _print(f"{app}: {msg}")
    if changed:
        _print("Reopen the terminal (or window) for the tab title to update.")
    elif "already" in msg:
        _print("You're all set — reopen a terminal if the tab isn't showing it.")
