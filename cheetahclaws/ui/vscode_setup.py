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

* runs at most once *per settings target* (a marker file under
  ~/.cheetahclaws records which file was configured), so we never nag — but a
  machine whose target changes (e.g. a local run after a Remote-SSH one) is
  still handled;
* never overwrites a value the user already set for that key;
* backs the file up before writing;
* inserts the key textually (preserving comments / formatting of a JSONC
  settings file) and then re-parses the result, aborting if the edit would
  drop a key, change a value, or fail to parse — so a weird settings file is
  left untouched rather than corrupted;
* swallows every error: a nicety must never break startup.

Remote-SSH / WSL / devcontainers / Codespaces need care: there the editor UI
runs on the user's own machine and its User settings.json is unreachable from
here, so writing ~/.config/Code/User/settings.json on the *server* side configures
nothing (it just fabricates a file VS Code never reads). The window does,
however, also read the **remote Machine settings** that live on this side —
``<server-root>/data/Machine/settings.json`` — and the title key applies from
there, so that is what we write when a server install is detected.

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
    """Local (UI-side) User settings.json for `app` on this platform."""
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


# Server installs: ~/.vscode-server, ~/.vscode-server-insiders, ~/.cursor-server,
# ~/.windsurf-server, /vscode/vscode-server (devcontainers), … — matched by the
# "-server" component rather than a fixed list, so forks and odd layouts work.
_SERVER_HINT_VARS = (
    "VSCODE_AGENT_FOLDER",        # the server root itself, when exported
    "VSCODE_GIT_ASKPASS_NODE",    # absolute path *into* the running server
    "VSCODE_GIT_ASKPASS_MAIN",
)
_SERVER_FALLBACK_DIRS = {
    "Code":     (".vscode-server", ".vscode-server-insiders"),
    "Cursor":   (".cursor-server",),
    "Windsurf": (".windsurf-server",),
}


def _looks_like_server_root(p: Path) -> bool:
    if "-server" not in p.name:
        return False
    try:
        return p.is_dir() and any((p / sub).is_dir()
                                  for sub in ("data", "cli", "bin", "extensions"))
    except OSError:
        return False


def _remote_server_root(app: str) -> Path | None:
    """Root of the editor *server* install when the UI runs elsewhere.

    Covers Remote-SSH, WSL, devcontainers and Codespaces — anywhere the
    terminal is on this machine but the window (and its User settings) is not.
    Returns None for a plain local editor.
    """
    for var in _SERVER_HINT_VARS:
        raw = os.environ.get(var)
        if not raw:
            continue
        candidate = Path(raw)
        for p in (candidate, *candidate.parents):
            if _looks_like_server_root(p):
                return p
    home = Path.home()
    for name in _SERVER_FALLBACK_DIRS.get(app, ()):
        p = home / name
        if _looks_like_server_root(p):
            return p
    return None


def _machine_settings_path(server_root: Path) -> Path:
    """Remote (server-side) Machine settings — the 'Remote [SSH: host]' scope.

    The title key applies from here, which is what makes Remote-SSH setups
    configurable at all from the machine CheetahClaws is installed on.
    """
    return server_root / "data" / "Machine" / "settings.json"


def _resolve_target(app: str) -> tuple[Path | None, str, str]:
    """Where to write the setting: (path, scope, why-not).

    scope is ``"remote"`` (server-side Machine settings) or ``"local"`` (this
    machine's User settings). When path is None, `why-not` explains it and the
    caller prints copy-paste instructions instead.
    """
    server_root = _remote_server_root(app)
    if server_root is not None:
        return _machine_settings_path(server_root), "remote", ""
    path = _settings_path(app)
    if path is None:
        return None, "", f"couldn't locate {app} settings.json on this platform"
    if not path.parent.exists():
        # The editor creates its own User dir; if it isn't here, no local
        # install is either — never fabricate a settings.json nothing reads.
        return None, "", f"no local {app} install found on this machine"
    return path, "local", ""


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


def _target_key(target: Path | None, why: str) -> str:
    return str(target) if target is not None else f"none:{why}"


def _already_attempted(key: str) -> bool:
    """True only when we already ran for *this* target.

    The marker used to hold a bare timestamp, so any run that resolves a
    different (or newly resolvable) target retries exactly once — which is
    what rescues machines whose first attempt wrote a file the editor never
    reads, e.g. a Remote-SSH session configured before this was understood.
    """
    try:
        return json.loads((CONFIG_DIR / _MARKER).read_text()).get("target") == key
    except Exception:
        return False


def _mark_attempted(key: str) -> None:
    try:
        marker = CONFIG_DIR / _MARKER
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"ts": int(time.time()), "target": key}))
    except Exception:
        pass


def _print_manual_instructions(app: str, why: str) -> None:
    _print(f"{app} tab titles need one setting, but {why} — its User settings "
           "live on the machine running the window.")
    _print("Add this to that machine's settings.json "
           "(Preferences: Open User Settings (JSON)):")
    _print(f'    "{_TITLE_KEY}": "{_TITLE_VAL}"')
    _print("Then open a new terminal. Re-check with /terminal-setup, or turn "
           "the feature off with /config terminal_title=false.")


def _scope_label(scope: str) -> str:
    return ("remote (Machine) settings — the window reads them from this side"
            if scope == "remote" else "User settings")


def maybe_setup_vscode_terminal_title(config: dict) -> None:
    """Auto-run once per target on launch inside a VS Code-family terminal.

    No-op unless: terminal_title is enabled, we're in VS Code/Cursor/Windsurf,
    and we haven't already run for this settings target. Any failure is
    swallowed."""
    try:
        if not config.get("terminal_title", True):
            return
        app = _vscode_app()
        if not app:
            return
        target, scope, why = _resolve_target(app)
        key = _target_key(target, why)
        if _already_attempted(key):
            return
        if target is None:
            _mark_attempted(key)
            _print_manual_instructions(app, why)
            return
        changed, msg = _apply_to_settings(target)
        _mark_attempted(key)
        if changed:
            _print(f"Set up {app} terminal tab titles in {_scope_label(scope)} "
                   f"— {msg}.")
            _print("Open a NEW terminal to see the task in the tab. "
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
    target, scope, why = _resolve_target(app)
    if target is None:
        _print_manual_instructions(app, why)
        _mark_attempted(_target_key(target, why))
        return
    changed, msg = _apply_to_settings(target)
    # Refresh the marker so the auto-path stays quiet afterwards.
    _mark_attempted(_target_key(target, why))
    _print(f"{app} ({_scope_label(scope)}): {msg}")
    if changed:
        _print("Open a new terminal (or window) for the tab title to update.")
    elif "already" in msg:
        _print("You're all set — open a new terminal if the tab isn't showing it.")
