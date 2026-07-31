"""Unit tests for the predicted-next-prompt ghost text.

Covers the ui.input pending-suggestion store + PredictiveAutoSuggest, and the
ui.suggest drafting/cleanup/staleness logic (with the auxiliary model stubbed).
"""

from __future__ import annotations

import pytest

from cheetahclaws.ui.input import HAS_PROMPT_TOOLKIT

if not HAS_PROMPT_TOOLKIT:
    pytest.skip("prompt_toolkit not installed", allow_module_level=True)

from prompt_toolkit.document import Document

import cheetahclaws.ui.input as ui_input
import cheetahclaws.ui.suggest as ui_suggest


@pytest.fixture(autouse=True)
def _clean_pending():
    ui_input.clear_pending_suggestion()
    yield
    ui_input.clear_pending_suggestion()


def _suggest(text: str, pending: str):
    completer = ui_input.PredictiveAutoSuggest(provider=lambda: pending)
    doc = Document(text, cursor_position=len(text))
    return completer.get_suggestion(_FakeBuffer(), doc)


class _FakeBuffer:
    """Minimal stand-in — AutoSuggestFromHistory only reads `.history`."""

    class _History:
        def get_strings(self):
            return ["run the tests", "git commit -m fix"]

    history = _History()
    document = None


# ── ui.input: pending store ─────────────────────────────────────────────────

def test_pending_suggestion_roundtrip_and_strip():
    ui_input.set_pending_suggestion("  run the tests  ")
    assert ui_input.get_pending_suggestion() == "run the tests"
    ui_input.clear_pending_suggestion()
    assert ui_input.get_pending_suggestion() == ""


# ── ui.input: PredictiveAutoSuggest ─────────────────────────────────────────

def test_empty_buffer_offers_whole_prediction():
    s = _suggest("", "add a test for the parser")
    assert s is not None and s.text == "add a test for the parser"


def test_typed_prefix_offers_the_remainder():
    s = _suggest("add a ", "add a test for the parser")
    assert s is not None and s.text == "test for the parser"


def test_divergent_typing_falls_back_to_history():
    s = _suggest("git c", "add a test for the parser")
    assert s is not None and s.text == "ommit -m fix"  # from _FakeBuffer history


def test_fully_typed_prediction_yields_no_ghost():
    """Prediction exhausted and no history match → nothing dangling."""
    s = _suggest("add a test for the parser", "add a test for the parser")
    assert s is None


def test_no_pending_leaves_history_behavior_untouched():
    s = _suggest("run the", "")
    assert s is not None and s.text == " tests"


def test_cursor_not_at_end_suppresses_prediction():
    """Mid-line editing gets history behavior only — no predicted remainder.

    (prompt_toolkit's renderer hides any ghost while the cursor is not at the
    end, so the history fallback here is invisible either way.)
    """
    completer = ui_input.PredictiveAutoSuggest(provider=lambda: "run the parser suite")
    doc = Document("run", cursor_position=1)
    s = completer.get_suggestion(_FakeBuffer(), doc)
    assert s is not None and s.text == " the tests"  # history, not the prediction


def test_multiline_buffer_suppresses_prediction():
    text = "run\nthe"
    completer = ui_input.PredictiveAutoSuggest(provider=lambda: "run\nthe tests")
    doc = Document(text, cursor_position=len(text))
    assert completer.get_suggestion(_FakeBuffer(), doc) is None


# ── ui.suggest: cleaning ────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("run the tests", "run the tests"),
    ('  "run the tests"  ', "run the tests"),
    ("- run the tests", "run the tests"),
    ("`run the tests`", "run the tests"),
    ("run the tests\nthen commit", "run the tests"),
    ("给这个函数加个测试", "给这个函数加个测试"),
    ("NONE", ""),
    ("", ""),
    ("   ", ""),
    ("Sure, here is what they'd type", ""),
    ("x" * (ui_suggest.MAX_LEN + 1), ""),
])
def test_clean(raw, expected):
    assert ui_suggest._clean(raw) == expected


# ── ui.suggest: transcript extraction ───────────────────────────────────────

def test_recent_exchange_flattens_blocks_and_keeps_order():
    messages = [
        {"role": "user", "content": "fix the parser"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Fixed it."},
            {"type": "tool_use", "name": "Edit"},
        ]},
    ]
    assert ui_suggest._recent_exchange(messages) == [
        {"role": "user", "content": "fix the parser"},
        {"role": "assistant", "content": "Fixed it."},
    ]


def test_recent_exchange_skips_empty_and_non_chat_roles():
    messages = [
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "hello"},
    ]
    assert ui_suggest._recent_exchange(messages) == [
        {"role": "user", "content": "hello"},
    ]


# ── ui.suggest: scheduling ──────────────────────────────────────────────────

def _stub_auxiliary(monkeypatch, reply):
    import cheetahclaws.auxiliary as aux
    monkeypatch.setattr(aux, "stream_auxiliary",
                        lambda system, messages, config: reply)


MESSAGES = [
    {"role": "user", "content": "fix the parser"},
    {"role": "assistant", "content": "Fixed it."},
]


def test_schedule_publishes_prediction(monkeypatch):
    _stub_auxiliary(monkeypatch, "run the tests")
    thread = ui_suggest.schedule(MESSAGES, {})
    assert thread is not None
    thread.join(timeout=5)
    assert ui_input.get_pending_suggestion() == "run the tests"


def test_schedule_disabled_by_config(monkeypatch):
    _stub_auxiliary(monkeypatch, "run the tests")
    assert ui_suggest.schedule(MESSAGES, {"input_suggest": False}) is None
    assert ui_input.get_pending_suggestion() == ""


def test_schedule_disabled_by_env(monkeypatch):
    _stub_auxiliary(monkeypatch, "run the tests")
    monkeypatch.setenv("CHEETAH_SUGGEST", "0")
    assert ui_suggest.schedule(MESSAGES, {}) is None


def test_schedule_without_history_is_a_noop(monkeypatch):
    _stub_auxiliary(monkeypatch, "run the tests")
    assert ui_suggest.schedule([], {}) is None


def test_schedule_clears_the_previous_ghost_immediately(monkeypatch):
    """A stale prediction must not survive into the next turn's draft window."""
    ui_input.set_pending_suggestion("stale suggestion")
    _stub_auxiliary(monkeypatch, "NONE")   # nothing usable this turn
    thread = ui_suggest.schedule(MESSAGES, {})
    thread.join(timeout=5)
    assert ui_input.get_pending_suggestion() == ""


def test_auxiliary_failure_is_silent(monkeypatch):
    import cheetahclaws.auxiliary as aux

    def _boom(system, messages, config):
        raise RuntimeError("provider down")

    monkeypatch.setattr(aux, "stream_auxiliary", _boom)
    assert ui_suggest.predict(MESSAGES, {}) == ""


def test_stale_draft_does_not_overwrite_newer_one(monkeypatch):
    """Turn N finishing after turn N+1 must not clobber the newer ghost."""
    import cheetahclaws.auxiliary as aux
    monkeypatch.setattr(aux, "stream_auxiliary",
                        lambda system, messages, config: "old prediction")
    ui_suggest._generation += 1          # simulate a newer turn already queued
    thread = ui_suggest.schedule(MESSAGES, {})
    thread.join(timeout=5)
    published = ui_input.get_pending_suggestion()
    ui_suggest._generation += 1
    assert published == "old prediction"  # this draft IS the newest one


# ── End-to-end through a real prompt_toolkit session ────────────────────────

def _drive(keys: str):
    """Run read_line() against a piped input; return (submitted, rendered)."""
    import io
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.plain_text import PlainTextOutput

    screen = io.StringIO()
    ui_input.reset_session()
    try:
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            with create_app_session(input=pipe, output=PlainTextOutput(screen)):
                return ui_input.read_line("» "), screen.getvalue()
    finally:
        ui_input.reset_session()


def test_e2e_ghost_renders_on_empty_prompt_and_enter_ignores_it():
    ui_input.set_pending_suggestion("run the tests")
    submitted, screen = _drive("\r")
    assert "run the tests" in screen   # shown as ghost text…
    assert submitted == ""             # …but never submitted on its own


def test_e2e_prediction_is_consumed_by_the_prompt_it_was_shown_at():
    ui_input.set_pending_suggestion("run the tests")
    _drive("\r")
    assert ui_input.get_pending_suggestion() == ""


def test_e2e_tab_accepts_the_whole_prediction():
    ui_input.set_pending_suggestion("run the tests")
    submitted, _ = _drive("\t\r")
    assert submitted == "run the tests"


def test_e2e_tab_completes_after_a_matching_prefix():
    """Regression: the ghost must be exact even when Tab beats the async pass."""
    ui_input.set_pending_suggestion("run the tests")
    submitted, _ = _drive("run \t\r")
    assert submitted == "run the tests"


def test_e2e_typing_something_else_types_over_the_ghost():
    ui_input.set_pending_suggestion("run the tests")
    submitted, _ = _drive("hello\r")
    assert submitted == "hello"


def test_e2e_ghost_returns_after_erasing_back_to_empty():
    ui_input.set_pending_suggestion("run the tests")
    submitted, _ = _drive("x\x7f\t\r")
    assert submitted == "run the tests"


def test_e2e_ghost_does_not_hijack_tab_for_slash_commands():
    ui_input.setup(lambda: {"help": True}, lambda: {"help": ("Show help", [])})
    try:
        ui_input.set_pending_suggestion("run the tests")
        submitted, _ = _drive("/hel\t\r")
        assert submitted.startswith("/hel")   # slash menu, never the ghost
        assert submitted != "run the tests"
    finally:
        ui_input.setup(lambda: {}, lambda: {})


def test_superseded_draft_is_dropped(monkeypatch):
    import cheetahclaws.auxiliary as aux
    started = __import__("threading").Event()
    release = __import__("threading").Event()

    def _slow(system, messages, config):
        started.set()
        release.wait(timeout=5)
        return "first prediction"

    monkeypatch.setattr(aux, "stream_auxiliary", _slow)
    first = ui_suggest.schedule(MESSAGES, {})
    assert started.wait(timeout=5)

    monkeypatch.setattr(aux, "stream_auxiliary",
                        lambda system, messages, config: "second prediction")
    second = ui_suggest.schedule(MESSAGES, {})
    second.join(timeout=5)

    release.set()
    first.join(timeout=5)
    assert ui_input.get_pending_suggestion() == "second prediction"
