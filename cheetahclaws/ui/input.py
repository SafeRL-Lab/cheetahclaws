"""prompt_toolkit-based REPL input with typing-time slash-command autosuggest.

Optional dependency: when prompt_toolkit is not installed, HAS_PROMPT_TOOLKIT
is False and callers should fall through to readline-based input.

Dependency-injected: callers register command/meta providers via setup()
before calling read_line(). This module never imports cheetahclaws — keeping
the dependency one-way and eliminating any circular-import risk.

Ghost text has two sources, in priority order:
  1. A predicted next prompt pushed in by ui.suggest via
     set_pending_suggestion() after each turn (Claude-Code style).
  2. The shell-history suggestion (prompt_toolkit's AutoSuggestFromHistory).
Both render dim/italic; Tab (or →) accepts the whole thing.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import (
        AutoSuggest, AutoSuggestFromHistory, Suggestion,
    )
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.application import get_app
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


# ── Injected providers ───────────────────────────────────────────────────────
# Callers (cheetahclaws.repl) must call setup() before read_line().
_commands_provider: Optional[Callable[[], dict]] = None
_meta_provider: Optional[Callable[[], dict]] = None
_dynamic_completions_provider: Optional[Callable[[], dict]] = None


def setup(
    commands_provider: Callable[[], dict],
    meta_provider: Callable[[], dict],
    dynamic_completions_provider: Optional[Callable[[], dict]] = None,
) -> None:
    """Register providers for the live command registry and metadata.

    `commands_provider` returns the dispatcher's COMMANDS dict.
    `meta_provider` returns the _CMD_META dict (descriptions + subcommands).
    `dynamic_completions_provider` returns a dict mapping command names to
    callables: fn(partial: str) -> Iterable[str]. Used for context-aware
    completions beyond static subcommand lists (e.g. /model providers).
    """
    global _commands_provider, _meta_provider, _dynamic_completions_provider
    _commands_provider = commands_provider
    _meta_provider = meta_provider
    _dynamic_completions_provider = dynamic_completions_provider


# ── Predicted next prompt (pushed in by ui.suggest) ──────────────────────────
# Written from a background thread, read from the prompt_toolkit event loop —
# hence the lock. One-shot: read_line() clears it once the user submits a line.
_pending_lock = threading.Lock()
_pending_suggestion: str = ""


def set_pending_suggestion(text: str) -> None:
    """Publish the ghost text shown at the next (empty) prompt."""
    global _pending_suggestion
    cleaned = (text or "").strip()
    with _pending_lock:
        _pending_suggestion = cleaned


def get_pending_suggestion() -> str:
    with _pending_lock:
        return _pending_suggestion


def clear_pending_suggestion() -> None:
    set_pending_suggestion("")


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
            dynamic_completions_provider: Optional[Callable[[], dict]] = None,
        ):
            self._commands_override = commands_provider
            self._meta_override = meta_provider
            self._dynamic_completions_override = dynamic_completions_provider
            self._cache_key: Optional[tuple] = None
            self._cache_names: list[str] = []

        def _get_commands(self) -> dict:
            provider = self._commands_override or _commands_provider
            return (provider() if provider else {}) or {}

        def _get_meta(self) -> dict:
            provider = self._meta_override or _meta_provider
            return (provider() if provider else {}) or {}

        def _get_dynamic_completions(self) -> dict:
            provider = self._dynamic_completions_override or _dynamic_completions_provider
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
            if meta_entry:
                subs = meta_entry[1]
                partial = tail.rsplit(" ", 1)[-1]
                if subs:
                    for sub in subs:
                        if sub.startswith(partial):
                            yield Completion(
                                sub,
                                start_position=-len(partial),
                                display_meta=f"{cmd} subcommand",
                            )
                    return

            # ── Dynamic command-specific completions (e.g. /model) ──
            dynamic = self._get_dynamic_completions()
            completer_fn = dynamic.get(cmd)
            if completer_fn:
                partial = tail.rsplit(" ", 1)[-1]
                for match in completer_fn(partial):
                    if match.startswith(partial):
                        yield Completion(
                            match,
                            start_position=-len(partial),
                            display_meta=f"{cmd} completion",
                        )

else:  # pragma: no cover — unreachable when prompt_toolkit is installed
    class SlashCompleter:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("prompt_toolkit is not installed")


# ── Auto-suggest ─────────────────────────────────────────────────────────────
if HAS_PROMPT_TOOLKIT:

    class PredictiveAutoSuggest(AutoSuggest):
        """Predicted next prompt first, shell history second.

        On an empty buffer the whole prediction is offered; once the user
        starts typing it survives only while it still prefixes what they
        typed, after which history takes over.
        """

        def __init__(self, provider: Optional[Callable[[], str]] = None):
            self._provider = provider or get_pending_suggestion
            self._history = AutoSuggestFromHistory()

        def get_suggestion(self, buffer, document):  # type: ignore[override]
            text = document.text
            # Only single-line, cursor-at-end states — otherwise the ghost
            # would render in the middle of a paste or a multi-line edit.
            if "\n" not in text and document.cursor_position == len(text):
                pending = (self._provider() or "").strip()
                if pending.startswith(text) and len(pending) > len(text):
                    return Suggestion(pending[len(text):])
            return self._history.get_suggestion(buffer, document)


# ── Key bindings ─────────────────────────────────────────────────────────────
if HAS_PROMPT_TOOLKIT:

    @Condition
    def _ghost_text_acceptable() -> bool:
        """True when a ghost-suggestion is shown and no slash menu is active."""
        buf = get_app().current_buffer
        if not (buf.suggestion and buf.suggestion.text):
            return False
        cs = buf.complete_state
        if cs and cs.completions:
            return False
        return True

    def _build_key_bindings() -> "KeyBindings":
        """Tab accepts the dim ghost-text (predicted prompt or history) shown.

        Falls through to the default Tab binding (slash-menu cycling) when the
        filter doesn't match, so `/cmd` completion behavior is unchanged.
        """
        kb = KeyBindings()

        @kb.add("tab", filter=_ghost_text_acceptable)
        def _(event):
            buf = event.current_buffer
            buf.insert_text(buf.suggestion.text)

        return kb

    def _apply_pending(buf) -> None:
        """Set the predicted ghost text on `buf`, synchronously.

        The auto-suggester runs as a background task and only on text
        *insert* — so an empty buffer is never asked at all, and a fast Tab
        right after typing can beat the async result. Applying the prediction
        inline on every text change (and at pre_run) keeps the ghost exact and
        immediate; the async path still handles the history fallback.
        """
        pending = get_pending_suggestion()
        if not pending or buf.suggestion:
            return
        text = buf.text
        if "\n" in text or buf.cursor_position != len(text):
            return
        if pending.startswith(text) and len(pending) > len(text):
            buf.suggestion = Suggestion(pending[len(text):])

    def _seed_suggestion() -> None:
        """pre_run hook: show the prediction before the user types anything."""
        try:
            buf = get_app().current_buffer
        except Exception:
            return
        _apply_pending(buf)


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
    session = PromptSession(
        history=history,
        completer=completer,
        auto_suggest=PredictiveAutoSuggest(),
        complete_while_typing=True,
        enable_history_search=False,
        mouse_support=False,
        style=style,
        key_bindings=_build_key_bindings(),
    )
    session.default_buffer.on_text_changed += _apply_pending
    return session


def read_line(prompt_ansi: str, history_path: Optional[Path] = None) -> str:
    """Read one line of input via prompt_toolkit; caches the session across calls.

    The history file passed here MUST NOT be the readline history file — the
    two line-editors use incompatible formats. See cheetahclaws.repl for the
    dedicated PT_HISTORY_FILE.

    A predicted next prompt published via set_pending_suggestion() is shown
    as dim ghost text and consumed by this call — it never carries over to a
    later prompt, where it would be stale.
    """
    global _SESSION, _SESSION_HISTORY_PATH
    if _SESSION is not None and _SESSION_HISTORY_PATH != history_path:
        _SESSION = None
    if _SESSION is None:
        _SESSION = _build_session(history_path)
        _SESSION_HISTORY_PATH = history_path
    try:
        with patch_stdout(raw=True):
            return _SESSION.prompt(ANSI(prompt_ansi), pre_run=_seed_suggestion)
    finally:
        clear_pending_suggestion()
