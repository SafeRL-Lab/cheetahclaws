"""Prompt file loading and model-family routing.

Public API (signatures unchanged from the original family-file design):
    pick_base_prompt(provider: str = "", model_id: str = "") -> str
    load_fragment(name: str) -> str

Internally this module now follows a **single base + small overlay** design
instead of one full file per family.  Rationale:

    Most prompt-engineering guidance ("be concise", "parallel tool calls",
    "minimal scope", "stop conditions", "safe vs unsafe actions") applies
    to every model.  Putting it in family files duplicates content and
    silently denies that guidance to families without a dedicated file.

    Conversely, *truly* family-specific quirks are short and well-documented
    (Anthropic XML tags; Gemini explicit agentic-mode framing; OpenAI
    o-series "do not narrate reasoning").  These belong in tiny overlays.

So:

    final_prompt = base/default.md  +  overlays/<family>.md  (if matched)

``pick_base_prompt`` returns the **assembled** text; callers continue to
use it as if it were a single file.  Tests can introspect via
``_family_overlay_for_model`` / ``_BASE_DIR`` / ``_OVERLAYS_DIR`` if needed.

## Why route by model family, not by provider/runtime

``providers.detect_provider()`` returns the *runtime* — anthropic, openai,
ollama, lmstudio, custom (OpenRouter, vLLM, any OpenAI-compat endpoint).
That dimension is right for **API plumbing** but wrong for **prompts**:
Qwen-3 served by DashScope, Ollama, vLLM, or OpenRouter is the same model
and should get the same prompt.  Routing is therefore primarily a
substring match on ``model_id``; the ``provider`` argument is consulted
only as a fallback when the model ID is empty.

## Overlay file contract
- Path: ``prompts/overlays/<family>.md``
- Soft cap: 20 lines.  Hard rule: must cite an official prompting guide
  URL in a top-of-file ``<!-- Source: -->`` comment.
- Must NOT repeat content already in default.md.
- Adding a new overlay requires a new entry in ``_OVERLAY_RULES`` and a
  case in ``tests/test_prompt_selection.py``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR    = Path(__file__).parent
_BASE_DIR       = _PROMPTS_DIR / "base"
_OVERLAYS_DIR   = _PROMPTS_DIR / "overlays"
_FRAGMENTS_DIR  = _PROMPTS_DIR / "fragments"


# ── Family-overlay routing ───────────────────────────────────────────────
#
# Ordered list of (substring-keywords, overlay-filename).  First hit wins.
# Matching is case-insensitive on the *last path segment* of the model ID
# (so "custom/anthropic/claude-sonnet-4-5" → "claude-sonnet-4-5" → matches
# "claude").  No match ⇒ no overlay (just the default base).
#
# Adding a new family means: write the overlay file, add an entry here,
# and add a case to tests/test_prompt_selection.py.
_OVERLAY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("claude",),                                "claude.md"),
    (("gemini",),                                "gemini.md"),
    # OpenAI reasoning models (o1/o3/o4 + codex variant of GPT-5).
    # Plain "gpt-" without a reasoning suffix gets NO overlay — the
    # default-base guidance is already what GPT chat models want.
    (("o1", "o3", "o4", "gpt-5-codex", "codex"), "openai-reasoning.md"),
    # Qwen / QwQ — chat-tuned default is conversational, needs an explicit
    # "call the tool, don't ask the user" stance for agentic use.
    (("qwen", "qwq"),                            "qwen.md"),
    # Families without an overlay yet (kimi / llama / mistral / gemma /
    # phi / glm / minimax / deepseek) all rely on default.md. Add an
    # overlay file + entry here when a documented quirk emerges.
)

# Provider → overlay fallback.  Used only when model_id is empty.
_PROVIDER_OVERLAY_FALLBACK: dict[str, str] = {
    "anthropic": "claude.md",
    "gemini":    "gemini.md",
    # "openai" provider WITHOUT a model id intentionally omitted —
    # we can't tell chat-vs-reasoning, so default base only.
}


def _family_overlay_for_model(model_id: str) -> str | None:
    """Return the overlay filename for a model ID, or None."""
    if not model_id:
        return None
    tail = model_id.rsplit("/", 1)[-1].lower()
    for keywords, fname in _OVERLAY_RULES:
        if any(k in tail for k in keywords):
            return fname
    return None


@lru_cache(maxsize=None)
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _assemble(base_name: str, overlay_name: str | None) -> str:
    """Return base + (optional) overlay, joined by a blank line.

    Cached so identical (base, overlay) pairs are a dict lookup, not disk I/O.
    """
    base_text = _read(_BASE_DIR / base_name)
    if not overlay_name:
        return base_text
    overlay_path = _OVERLAYS_DIR / overlay_name
    if not overlay_path.exists():
        # Defensive: a rule referenced a not-yet-shipped overlay.  Fall
        # back silently rather than raising in production.
        return base_text
    return base_text.rstrip() + "\n\n" + _read(overlay_path).strip() + "\n"


def pick_base_prompt(provider: str = "", model_id: str = "") -> str:
    """Return the assembled base prompt (default + matched overlay).

    Args:
        provider: provider name from ``providers.detect_provider()``. Used
                  only as a fallback when ``model_id`` is empty.
        model_id: the full model identifier (may include a ``provider/``
                  or ``provider/vendor/`` prefix). Matched against
                  ``_OVERLAY_RULES`` case-insensitively on its last path
                  segment.

    Returns:
        ``default.md`` body, optionally followed by one matched overlay.
        Never raises for unknown models.
    """
    overlay = (
        _family_overlay_for_model(model_id)
        or _PROVIDER_OVERLAY_FALLBACK.get(provider)
    )
    return _assemble("default.md", overlay)


def load_fragment(name: str) -> str:
    """Return the raw Markdown body of a conditional fragment.

    Fragments are short reusable blocks appended to the system prompt
    under runtime conditions (e.g. tmux present, plan mode active).

    Raises:
        FileNotFoundError: if the fragment does not exist — this is a
            programming error, not a runtime condition, so it should be loud.
    """
    path = _FRAGMENTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt fragment not found: {path}")
    return _read(path)


def clear_cache() -> None:
    """Reset the prompt file cache. Intended for tests only."""
    _read.cache_clear()
    _assemble.cache_clear()
