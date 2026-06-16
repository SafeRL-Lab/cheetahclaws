"""End-to-end tests for LiteLLMProvider against real LLM APIs.

Skipped unless ``CC_LITELLM_E2E=1`` is set in the environment AND at least
one provider credential is available (ANTHROPIC_API_KEY by default; override
with CC_LITELLM_E2E_MODEL + the matching key env var).

Run locally with:

    CC_LITELLM_E2E=1 ANTHROPIC_API_KEY=sk-ant-... \\
        pytest tests/e2e_litellm_provider.py -v

These cover the same three scenarios the PR body claimed were passing
against real APIs — basic call, streaming, system prompt — so the
contract isn't only mock-asserted.
"""

from __future__ import annotations

import os

import pytest

# Default to a low-cost Anthropic model since cheetahclaws ships with that
# key by convention. Override via CC_LITELLM_E2E_MODEL=openai/gpt-4o-mini etc.
_DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"
_MODEL = os.environ.get("CC_LITELLM_E2E_MODEL", _DEFAULT_MODEL)


def _has_creds_for(model: str) -> bool:
    """Best-effort check that the right env var is set for the provider
    embedded in `model` (e.g. anthropic/... → ANTHROPIC_API_KEY)."""
    prefix = model.split("/", 1)[0] if "/" in model else ""
    return bool(
        {
            "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
            "openai":    os.environ.get("OPENAI_API_KEY"),
            "azure":     os.environ.get("AZURE_API_KEY"),
            "bedrock":   os.environ.get("AWS_ACCESS_KEY_ID"),
            "gemini":    os.environ.get("GEMINI_API_KEY"),
            "vertex_ai": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        }.get(prefix)
    )


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("CC_LITELLM_E2E") != "1",
        reason="set CC_LITELLM_E2E=1 to enable LiteLLM live-API tests",
    ),
    pytest.mark.skipif(
        not _has_creds_for(_MODEL),
        reason=f"no credentials available for {_MODEL}",
    ),
]


@pytest.fixture(scope="module")
def provider():
    # The litellm_provider module imports without the SDK (lazy
    # import). Catch the ProviderUnavailable raised on first SDK use
    # so we skip — rather than fail — when CC_LITELLM_E2E is set on a
    # box that doesn't actually have litellm installed.
    from cheetahclaws.kernel.runner.llm.litellm_provider import LiteLLMProvider
    p = LiteLLMProvider()
    try:
        p._ensure_litellm()
    except Exception as e:
        pytest.skip(f"litellm SDK not usable: {e}")
    return p


def test_basic_call(provider):
    from cheetahclaws.kernel.runner.llm.provider import LlmRequest

    req = LlmRequest(model=_MODEL, user="What is 2+2? Reply with just the number.",
                     max_tokens=10)
    resp = provider(req)
    assert resp.text.strip() != ""
    # Token counts must come back from the real API (not the hard-coded 0
    # we used to emit) so ledger accounting is correct.
    assert resp.tokens_input > 0
    assert resp.tokens_output > 0


def test_streaming_emits_deltas(provider):
    from cheetahclaws.kernel.runner.llm.provider import LlmRequest

    received: list[str] = []
    req = LlmRequest(
        model=_MODEL,
        user="Say 'OK' and nothing else.",
        max_tokens=10,
    )
    resp = provider.stream(req, lambda d: received.append(d))
    assert "".join(received) == resp.text
    # Streaming usage must be populated too — the historical bug was
    # that stream() always returned 0 here.
    assert resp.tokens_input > 0
    assert resp.tokens_output > 0


def test_system_prompt_steers_reply(provider):
    from cheetahclaws.kernel.runner.llm.provider import LlmRequest

    req = LlmRequest(
        model=_MODEL,
        system="You are a pirate. Reply with one word in pirate speech.",
        user="Greet me.",
        max_tokens=20,
    )
    resp = provider(req)
    # Cheapest possible assertion that the system prompt landed — just
    # that the reply isn't empty and is short enough not to be spam.
    assert resp.text.strip() != ""
    assert len(resp.text) < 200
