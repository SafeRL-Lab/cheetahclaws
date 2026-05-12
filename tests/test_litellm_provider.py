"""Tests for LiteLLM provider."""

from unittest.mock import MagicMock, patch

import pytest

from cc_kernel.runner.llm.provider import (
    LlmRequest,
    LlmResponse,
    ProviderInvalidRequest,
    ProviderUnavailable,
)
from cc_kernel.runner.llm.litellm_provider import LiteLLMProvider


class TestLiteLLMProviderInit:
    def test_default_timeout(self):
        p = LiteLLMProvider()
        assert p._timeout_s == 60.0

    def test_custom_api_key(self):
        p = LiteLLMProvider(api_key="sk-test")
        assert p._api_key == "sk-test"

    def test_no_lazy_state(self):
        p = LiteLLMProvider()
        assert not hasattr(p, "_litellm")


class TestLiteLLMProviderCall:
    @patch("cc_kernel.runner.llm.litellm_provider.litellm")
    def test_calls_litellm_completion(self, mock_litellm):
        p = LiteLLMProvider(api_key="sk-test")
        mock_msg = MagicMock(content="hello", tool_calls=None)
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_choice = MagicMock(message=mock_msg, finish_reason="stop")
        mock_litellm.completion.return_value = MagicMock(
            choices=[mock_choice], usage=mock_usage
        )

        req = LlmRequest(model="openai/gpt-4o", user="hi")
        resp = p(req)

        assert resp.text == "hello"
        assert resp.tokens_input == 10
        assert resp.tokens_output == 5
        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["drop_params"] is True
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["api_key"] == "sk-test"

    @patch("cc_kernel.runner.llm.litellm_provider.litellm")
    def test_omits_api_key_when_none(self, mock_litellm):
        p = LiteLLMProvider()
        mock_msg = MagicMock(content="ok", tool_calls=None)
        mock_litellm.completion.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=None,
        )

        req = LlmRequest(model="openai/gpt-4o", user="hi")
        p(req)
        assert "api_key" not in mock_litellm.completion.call_args.kwargs

    @patch("cc_kernel.runner.llm.litellm_provider.litellm")
    def test_system_prompt_included(self, mock_litellm):
        p = LiteLLMProvider()
        mock_msg = MagicMock(content="ok", tool_calls=None)
        mock_litellm.completion.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=None,
        )

        req = LlmRequest(model="openai/gpt-4o", system="be helpful", user="hi")
        p(req)
        messages = mock_litellm.completion.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "be helpful"}
        assert messages[1] == {"role": "user", "content": "hi"}

    @patch("cc_kernel.runner.llm.litellm_provider.litellm")
    def test_multi_turn_messages(self, mock_litellm):
        p = LiteLLMProvider()
        mock_msg = MagicMock(content="ok", tool_calls=None)
        mock_litellm.completion.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=None,
        )

        msgs = (
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        )
        req = LlmRequest(model="openai/gpt-4o", messages=msgs)
        p(req)
        messages = mock_litellm.completion.call_args.kwargs["messages"]
        assert len(messages) == 3

    @patch("cc_kernel.runner.llm.litellm_provider.litellm")
    def test_returns_llm_response(self, mock_litellm):
        p = LiteLLMProvider()
        mock_msg = MagicMock(content="test", tool_calls=None)
        mock_litellm.completion.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=None,
        )

        req = LlmRequest(model="openai/gpt-4o", user="hi")
        resp = p(req)
        assert isinstance(resp, LlmResponse)
        assert resp.model == "openai/gpt-4o"


class TestLiteLLMProviderEdgeCases:
    def test_rejects_non_llm_request(self):
        p = LiteLLMProvider()
        with pytest.raises(ProviderInvalidRequest):
            p("not a request")

    @patch("cc_kernel.runner.llm.litellm_provider.litellm")
    def test_api_error_raises_provider_unavailable(self, mock_litellm):
        p = LiteLLMProvider()
        mock_litellm.completion.side_effect = Exception("401 Unauthorized")

        req = LlmRequest(model="openai/gpt-4o", user="hi")
        with pytest.raises(ProviderUnavailable, match="401"):
            p(req)

    @patch("cc_kernel.runner.llm.litellm_provider.litellm")
    def test_stream_rejects_non_callable_on_delta(self, mock_litellm):
        p = LiteLLMProvider()
        req = LlmRequest(model="openai/gpt-4o", user="hi")
        with pytest.raises(ProviderInvalidRequest, match="on_delta"):
            p.stream(req, "not callable")


class TestRegistration:
    def test_litellm_in_requirements(self):
        from pathlib import Path

        reqs = Path("requirements.txt").read_text()
        assert "litellm" in reqs
