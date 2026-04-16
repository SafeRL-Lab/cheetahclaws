"""Auxiliary model routing for cheap/fast side tasks.

Routes tasks like context compression, session title generation, and
vision analysis to a fast, inexpensive model instead of the user's
primary model. Falls back to the primary model if no auxiliary is available.

Config key: "auxiliary_model" (default: auto-detect)
"""
from __future__ import annotations

import os
from typing import Optional

import providers

# ── Fast model candidates (checked in order) ─────────────────────────────
# Each entry: (model_name, required_env_var_or_None)
_CANDIDATES = [
    ("gemini/gemini-2.0-flash",      "GEMINI_API_KEY"),
    ("gpt-4o-mini",                   "OPENAI_API_KEY"),
    ("deepseek/deepseek-chat",        "DEEPSEEK_API_KEY"),
    ("claude-haiku-4-5-20251001",     "ANTHROPIC_API_KEY"),
    ("qwen/qwen-turbo",              "DASHSCOPE_API_KEY"),
    ("zhipu/glm-4-flash",            "ZHIPU_API_KEY"),
]

_resolved: Optional[str] = None


def get_auxiliary_model(config: dict) -> str:
    """Return the best available auxiliary model.

    Priority:
    1. config["auxiliary_model"] if explicitly set
    2. Auto-detect from available API keys (cheapest/fastest first)
    3. Fall back to the user's primary model
    """
    global _resolved

    # Explicit config
    explicit = config.get("auxiliary_model")
    if explicit:
        return explicit

    # Cached auto-detection
    if _resolved is not None:
        return _resolved

    # Check which providers have keys available
    for model, env_var in _CANDIDATES:
        if env_var is None:
            _resolved = model
            return model
        # Check env var or config key
        pname = providers.detect_provider(model)
        key = providers.get_api_key(pname, config)
        if key:
            _resolved = model
            return model

    # Check if current model is local (Ollama) — use it directly
    primary = config.get("model", "")
    pname = providers.detect_provider(primary)
    if pname in ("ollama", "lmstudio", "custom"):
        _resolved = primary
        return primary

    # Final fallback: use the primary model
    _resolved = primary
    return primary


def reset_cache():
    """Clear the cached auxiliary model (for testing or config changes)."""
    global _resolved
    _resolved = None


def stream_auxiliary(
    system: str,
    messages: list,
    config: dict,
) -> str:
    """Run a simple text completion with the auxiliary model.

    Returns the full response text (no streaming to user, no tools).
    """
    model = get_auxiliary_model(config)
    text = ""
    try:
        for event in providers.stream(
            model=model,
            system=system,
            messages=messages,
            tool_schemas=[],
            config=config,
        ):
            if isinstance(event, providers.TextChunk):
                text += event.text
    except Exception:
        # Auxiliary model failure should not crash the caller.
        # Return whatever text was collected so far.
        pass
    return text
