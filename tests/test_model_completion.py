"""Tests for the /model dynamic completion helper in cli.py."""
from __future__ import annotations

from cheetahclaws.cli import _model_dynamic_completions
from cheetahclaws.providers import PROVIDERS


def _providers_with_static_models() -> set[str]:
    return {
        pname for pname, pdata in PROVIDERS.items()
        if pdata.get("models") and pname not in {"lmstudio", "custom"}
    }


def test_empty_partial_suggests_one_model_per_provider_plus_litellm_backends():
    suggestions = _model_dynamic_completions("")
    providers = _providers_with_static_models()
    providers.add("litellm")
    suggested_providers = set()
    for s in suggestions:
        if s.startswith("litellm/") and s.endswith("/"):
            suggested_providers.add("litellm")
        else:
            suggested_providers.add(s.split("/", 1)[0])
    assert suggested_providers == providers


def test_provider_prefix_completion():
    suggestions = _model_dynamic_completions("openai/g")
    assert all(s.startswith("openai/") for s in suggestions)
    assert any("gpt-4o" in s for s in suggestions)


def test_litellm_level1_backend_completion():
    suggestions = _model_dynamic_completions("litellm/")
    assert any(s == "litellm/openrouter/" for s in suggestions)
    assert any(s == "litellm/groq/" for s in suggestions)


def test_litellm_level2_model_completion():
    suggestions = _model_dynamic_completions("litellm/openrouter/anthropic")
    assert all(s.startswith("litellm/openrouter/anthropic") for s in suggestions)
    assert any("claude-3-5-sonnet" in s for s in suggestions)


def test_fallback_matches_full_provider_model():
    suggestions = _model_dynamic_completions("openai/gpt-4o")
    assert "openai/gpt-4o" in suggestions


def test_fallback_matches_litellm_full_string():
    suggestions = _model_dynamic_completions("litellm/openrouter/openai/gpt-4o")
    assert "litellm/openrouter/openai/gpt-4o" in suggestions


def test_no_dulus_only_providers():
    """Guard against accidentally shipping providers that are private to Falcon/Dulus."""
    suggestions = _model_dynamic_completions("")
    providers = {s.split("/", 1)[0] for s in suggestions}
    private = {"claude-web", "kimi-web", "deepseek-web", "qwen-web", "gemini-web",
               "claude-code", "xai-oauth", "xiaomi", "sakana", "cloudflare",
               "modelstudio", "amd", "nvidia-web", "azure-sponsored", "gcloud-sponsored"}
    assert providers & private == set()
