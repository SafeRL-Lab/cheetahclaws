"""prompt_toolkit-based REPL input with typing-time slash-command autosuggest.

Optional dependency: when prompt_toolkit is not installed, HAS_PROMPT_TOOLKIT
is False and callers should fall through to readline-based input.

Dependency-injected: callers register command/meta providers via setup()
before calling read_line(). This module never imports cheetahclaws — keeping
the dependency one-way and eliminating any circular-import risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


# ── Injected providers ───────────────────────────────────────────────────────
# Callers (cheetahclaws.repl) must call setup() before read_line().
_commands_provider: Optional[Callable[[], dict]] = None
_meta_provider: Optional[Callable[[], dict]] = None


def setup(
    commands_provider: Callable[[], dict],
    meta_provider: Callable[[], dict],
) -> None:
    """Register providers for the live command registry and metadata.

    `commands_provider` returns the dispatcher's COMMANDS dict.
    `meta_provider` returns the _CMD_META dict (descriptions + subcommands).
    """
    global _commands_provider, _meta_provider
    _commands_provider = commands_provider
    _meta_provider = meta_provider


# ── Completer ────────────────────────────────────────────────────────────────
if HAS_PROMPT_TOOLKIT:

    class SlashCompleter(Completer):
        """Two-level completer for slash commands.

        Level 1: /partial  (no space)  → command names.
        Level 2: /cmd partial           → subcommands listed in the meta dict.

        Providers default to the module-level ones registered via setup(),
        but can be injected via the constructor for testing.
        """

        def __init__(
            self,
            commands_provider: Optional[Callable[[], dict]] = None,
            meta_provider: Optional[Callable[[], dict]] = None,
        ):
            self._commands_override = commands_provider
            self._meta_override = meta_provider
            self._cache_key: Optional[tuple] = None
            self._cache_names: list[str] = []

        def _get_commands(self) -> dict:
            provider = self._commands_override or _commands_provider
            return (provider() if provider else {}) or {}

        def _get_meta(self) -> dict:
            provider = self._meta_override or _meta_provider
            return (provider() if provider else {}) or {}

        def _live_command_names(self) -> list[str]:
            keys = sorted(set(self._get_commands().keys()) | set(self._get_meta().keys()))
            sig = tuple(keys)
            if self._cache_key == sig:
                return self._cache_names
            self._cache_key = sig
            self._cache_names = keys
            return keys

        def get_completions(self, document, complete_event):  # type: ignore[override]
            text = document.text_before_cursor
            if not text.startswith("/"):
                return

            meta = self._get_meta()

            if " " not in text:
                word = text[1:]
                for name in self._live_command_names():
                    if not name.startswith(word):
                        continue
                    desc, subs = meta.get(name, ("", []))
                    hint = ""
                    if subs:
                        head = ", ".join(subs[:3])
                        more = "…" if len(subs) > 3 else ""
                        hint = f"  [{head}{more}]"
                    yield Completion(
                        "/" + name,
                        start_position=-len(text),
                        display=ANSI(f"\x1b[36m/{name}\x1b[0m"),
                        display_meta=(desc + hint) if desc else hint.strip(),
                    )
                return

            head, _, tail = text.partition(" ")
            cmd = head[1:]
            meta_entry = meta.get(cmd)
            if not meta_entry:
                return
            subs = meta_entry[1]
            if not subs:
                return
            partial = tail.rsplit(" ", 1)[-1]
            for sub in subs:
                if sub.startswith(partial):
                    yield Completion(
                        sub,
                        start_position=-len(partial),
                        display_meta=f"{cmd} subcommand",
                    )

else:  # pragma: no cover — unreachable when prompt_toolkit is installed
    class SlashCompleter:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("prompt_toolkit is not installed")


# ── Session cache ────────────────────────────────────────────────────────────
_SESSION = None
_SESSION_HISTORY_PATH: Optional[Path] = None


def reset_session() -> None:
    """Drop the cached session so the next read_line() rebuilds from scratch."""
    global _SESSION, _SESSION_HISTORY_PATH
    _SESSION = None
    _SESSION_HISTORY_PATH = None


def _build_session(history_path: Optional[Path]):
    if not HAS_PROMPT_TOOLKIT:
        raise RuntimeError("prompt_toolkit is not installed")
    completer = SlashCompleter()
    history = FileHistory(str(history_path)) if history_path else InMemoryHistory()
    style = Style.from_dict({
        "completion-menu.completion":              "bg:#222222 #cccccc",
        "completion-menu.completion.current":      "bg:#005f87 #ffffff bold",
        "completion-menu.meta.completion":         "bg:#222222 #808080",
        "completion-menu.meta.completion.current": "bg:#005f87 #eeeeee",
        "auto-suggestion":                         "#606060 italic",
    })
    return PromptSession(
        history=history,
        completer=completer,
        auto_suggest=AutoSuggestFromHistory(),
        complete_while_typing=True,
        enable_history_search=False,
        mouse_support=False,
        style=style,
    )


def read_line(prompt_ansi: str, history_path: Optional[Path] = None) -> str:
    """Read one line of input via prompt_toolkit; caches the session across calls.

    The history file passed here MUST NOT be the readline history file — the
    two line-editors use incompatible formats. See cheetahclaws.repl for the
    dedicated PT_HISTORY_FILE.
    """
    global _SESSION, _SESSION_HISTORY_PATH
    if _SESSION is not None and _SESSION_HISTORY_PATH != history_path:
        _SESSION = None
    if _SESSION is None:
        _SESSION = _build_session(history_path)
        _SESSION_HISTORY_PATH = history_path
    with patch_stdout(raw=True):
        return _SESSION.prompt(ANSI(prompt_ansi))
