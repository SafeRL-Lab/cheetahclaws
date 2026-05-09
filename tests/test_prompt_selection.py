"""Tests for :mod:`prompts.select` — overlay routing + fragment loading.

The design is single-base + small overlays:
    final = prompts/base/default.md  +  prompts/overlays/<family>.md (if matched)

So every prompt **starts from the same default.md** and family-specific
content is appended only when the model has a documented quirk.  Tests
encode that invariant directly.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from prompts import pick_base_prompt, load_fragment
from prompts import select as _select


def _default_text() -> str:
    return (_select._BASE_DIR / "default.md").read_text(encoding="utf-8")


def _overlay_text(name: str) -> str:
    return (_select._OVERLAYS_DIR / name).read_text(encoding="utf-8")


# ── Core claim: every model gets default.md as its base ───────────────────
#
# The shared base is the invariant that lets us add general prompt-eng
# guidance once and have all models benefit.  Any prompt the selector
# returns must include the default.md body verbatim.

_ALL_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-5",
    "custom/anthropic/claude-sonnet-4-5",
    "gpt-5",
    "gpt-4o",
    "o1",
    "o3-mini",
    "gpt-5-codex",
    "gemini/gemini-2.5-pro",
    "gemini-3.1-pro-preview",
    "kimi/moonshot-v1-128k",
    "deepseek/deepseek-chat",
    "ollama/qwen2.5-coder:32b",
    "ollama/llama3.3",
    "ollama/gemma4:e4b",
    "custom/my-private-finetune",
    "",
]


@pytest.mark.parametrize("model_id", _ALL_MODELS,
                          ids=[m or "<empty>" for m in _ALL_MODELS])
def test_every_model_includes_default_base(model_id: str):
    """The default base must be the shared starting point for all models."""
    text = pick_base_prompt(model_id=model_id)
    assert _default_text() in text, (
        f"model_id={model_id!r} did not include default.md verbatim — "
        f"that's the shared baseline."
    )


# ── Overlay routing: documented family quirks get an overlay appended ─────

_OVERLAY_CASES = [
    # (model_id, expected_overlay_filename_or_None, comment)
    # --- Claude → claude.md overlay (XML-tag preference) ---
    ("claude-opus-4-7",                    "claude.md", "native Anthropic"),
    ("claude-sonnet-4-5",                  "claude.md", "native Anthropic"),
    ("custom/anthropic/claude-sonnet-4-5", "claude.md", "Claude via OpenRouter"),
    # --- Gemini → gemini.md overlay (explicit agentic-mode) ---
    ("gemini/gemini-2.5-pro",              "gemini.md", "native Gemini"),
    ("gemini-3.1-pro-preview",             "gemini.md", "Gemini 3"),
    ("custom/google/gemini-2.5-pro",       "gemini.md", "Gemini via OpenRouter"),
    # --- OpenAI reasoning models → openai-reasoning.md overlay ---
    ("o1",            "openai-reasoning.md", "o1 reasoning"),
    ("o3-mini",       "openai-reasoning.md", "o3 reasoning"),
    ("o4-preview",    "openai-reasoning.md", "o4 reasoning"),
    ("gpt-5-codex",   "openai-reasoning.md", "codex variant"),
    # --- Plain GPT chat models get NO overlay (default base only) ---
    ("gpt-5",                          None, "plain GPT — default only"),
    ("gpt-4o",                         None, "plain GPT — default only"),
    ("custom/openai/gpt-5",            None, "plain GPT via OpenRouter"),
    # --- Qwen / QwQ → qwen.md overlay (explicit tool-use stance) ---
    ("ollama/qwen2.5-coder:32b",       "qwen.md", "Qwen via Ollama"),
    ("custom/qwen2.5-72b",             "qwen.md", "Qwen via OpenAI-compat"),
    ("qwen/Qwen3-MAX",                 "qwen.md", "Qwen3 via DashScope"),
    ("qwq-32b-preview",                "qwen.md", "QwQ reasoning"),
    # --- No-overlay families: rely on default base ---
    ("kimi/moonshot-v1-128k",          None, "Kimi — no overlay yet"),
    ("deepseek/deepseek-chat",         None, "DeepSeek — no overlay yet"),
    ("ollama/llama3.3",                None, "Llama — no overlay"),
    ("ollama/gemma4:e4b",              None, "Gemma — no overlay"),
    ("custom/my-private-finetune",     None, "unknown model"),
    ("",                               None, "empty model id"),
]


@pytest.mark.parametrize("model_id,overlay,comment", _OVERLAY_CASES,
                          ids=[c[2] for c in _OVERLAY_CASES])
def test_overlay_routing(model_id: str, overlay: str | None, comment: str):
    """Each model resolves to the expected overlay (or no overlay)."""
    text = pick_base_prompt(model_id=model_id)
    if overlay is None:
        # No overlay → result must equal default.md verbatim (with assemble's
        # rstrip semantics) — i.e. no extra family content appended.
        assert text == _default_text()
    else:
        body = _overlay_text(overlay).strip()
        assert body in text, (
            f"[{comment}] model_id={model_id!r} expected {overlay} content "
            f"appended, but it wasn't found in the assembled prompt."
        )


# ── Runtime invariance: same model via different runtimes → same prompt ──


def test_runtime_is_irrelevant_for_family_routing():
    """Qwen served three different ways → same prompt (default + qwen overlay)."""
    via_ollama     = pick_base_prompt(model_id="ollama/qwen2.5-coder")
    via_dashscope  = pick_base_prompt(model_id="qwen/Qwen3-MAX")
    via_openrouter = pick_base_prompt(model_id="custom/qwen/Qwen3-MAX")
    assert via_ollama == via_dashscope == via_openrouter
    # And it should be picking up the qwen overlay regardless of runtime.
    assert _overlay_text("qwen.md").strip() in via_ollama


def test_claude_routing_is_runtime_agnostic():
    native     = pick_base_prompt(model_id="claude-opus-4-7")
    openrouter = pick_base_prompt(model_id="custom/anthropic/claude-opus-4-7")
    assert native == openrouter


def test_deepseek_via_anywhere_is_default_only():
    """DeepSeek currently has no overlay — every runtime path must give plain default."""
    a = pick_base_prompt(model_id="deepseek/deepseek-chat")
    b = pick_base_prompt(model_id="ollama/deepseek-r1:32b")
    c = pick_base_prompt(model_id="custom/deepseek/deepseek-chat-v3.2")
    assert a == b == c == _default_text()


# ── Provider fallback (only consulted when model_id is empty) ────────────


def test_provider_fallback_when_model_id_empty():
    """With no model_id, the provider kwarg picks the matching overlay."""
    claude_prompt = pick_base_prompt(provider="anthropic")
    assert _overlay_text("claude.md").strip() in claude_prompt
    gemini_prompt = pick_base_prompt(provider="gemini")
    assert _overlay_text("gemini.md").strip() in gemini_prompt


def test_openai_provider_without_model_gets_no_overlay():
    """openai provider w/o model_id can't tell chat-vs-reasoning, so default only."""
    assert pick_base_prompt(provider="openai") == _default_text()


def test_local_providers_do_not_pick_a_runtime_overlay():
    """ollama / lmstudio / custom without model_id must hit default — never a runtime-specific overlay."""
    assert pick_base_prompt(provider="ollama")   == _default_text()
    assert pick_base_prompt(provider="lmstudio") == _default_text()
    assert pick_base_prompt(provider="custom")   == _default_text()


def test_model_id_takes_precedence_over_provider():
    """If model_id carries a family keyword, provider fallback is ignored."""
    out = pick_base_prompt(provider="custom",
                            model_id="custom/anthropic/claude-sonnet-4-5")
    assert _overlay_text("claude.md").strip() in out


def test_unknown_provider_with_no_model_falls_back_to_default():
    assert pick_base_prompt(provider="some-unknown-provider") == _default_text()


def test_pick_base_prompt_no_args_returns_default():
    assert pick_base_prompt() == _default_text()


# ── Fragment loading ──────────────────────────────────────────────────────


def test_load_fragment_tmux():
    text = load_fragment("tmux")
    assert "tmux" in text.lower()
    assert "TmuxNewSession" in text


def test_load_fragment_plan_keeps_placeholder():
    """plan.md must keep the {plan_file} placeholder unformatted."""
    text = load_fragment("plan")
    assert "{plan_file}" in text, "plan fragment must carry its placeholder for caller to format"
    assert "Plan Mode" in text


def test_load_fragment_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_fragment("no-such-fragment-does-not-exist")


# ── Cache behavior ────────────────────────────────────────────────────────


def test_repeated_calls_hit_cache(monkeypatch):
    """Second call to pick_base_prompt must NOT re-read the file."""
    _select.clear_cache()

    call_count = {"n": 0}
    original_read_text = _select.Path.read_text

    def counting_read_text(self, *a, **kw):
        call_count["n"] += 1
        return original_read_text(self, *a, **kw)

    monkeypatch.setattr(_select.Path, "read_text", counting_read_text)

    pick_base_prompt(model_id="claude-opus-4-7")
    first = call_count["n"]
    pick_base_prompt(model_id="claude-opus-4-7")
    pick_base_prompt(model_id="claude-opus-4-7")
    assert call_count["n"] == first, "lru_cache should prevent further reads"


# ── Architectural regressions ────────────────────────────────────────────


def test_ollama_md_is_not_shipped():
    """No runtime-level prompt file may exist.

    Runtime ('ollama', 'lmstudio', 'custom') is never a valid prompt
    dimension — prompts depend on family, not on how the model is served.
    """
    for forbidden in ["ollama.md", "lmstudio.md", "custom.md", "vllm.md"]:
        assert not (_select._BASE_DIR / forbidden).exists(), (
            f"prompts/base/{forbidden} must not exist — see prompts/README.md."
        )
        assert not (_select._OVERLAYS_DIR / forbidden).exists(), (
            f"prompts/overlays/{forbidden} must not exist — see prompts/README.md."
        )


def test_dead_family_base_files_are_gone():
    """The pre-refactor full per-family base files must not be revived.

    They duplicated default.md content with minor edits, which silently
    denied general guidance to families without a dedicated file.  All
    family-specific content now lives in tiny overlays.
    """
    for old in ["anthropic.md", "openai.md", "gemini.md", "kimi.md", "deepseek.md"]:
        assert not (_select._BASE_DIR / old).exists(), (
            f"prompts/base/{old} should not exist — family content lives in "
            f"prompts/overlays/ now (see prompts/README.md)."
        )


def test_overlays_directory_has_expected_files():
    """Lock the current overlay set so accidental deletes / typos are caught."""
    expected = {"claude.md", "gemini.md", "openai-reasoning.md", "qwen.md"}
    actual = {p.name for p in _select._OVERLAYS_DIR.iterdir() if p.suffix == ".md"}
    assert expected.issubset(actual), (
        f"missing expected overlay files. expected: {expected}, found: {actual}"
    )
