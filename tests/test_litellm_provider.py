"""Tests for LiteLLM provider."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cheetahclaws.kernel.runner.llm.provider import (
    LlmRequest,
    LlmResponse,
    ProviderInvalidRequest,
    ProviderUnavailable,
)
from cheetahclaws.kernel.runner.llm.litellm_provider import LiteLLMProvider


def _make_provider_with_fake_litellm(
    completion_return=None,
    completion_side_effect=None,
    stream_chunks=None,
    completion_cost=None,
    completion_cost_side_effect=None,
    api_key=None,
):
    """Build a LiteLLMProvider whose `_litellm` is a pre-populated mock —
    bypasses the lazy import path entirely so tests don't need the real
    litellm SDK installed."""
    fake = MagicMock()
    if completion_side_effect is not None:
        fake.completion.side_effect = completion_side_effect
    elif stream_chunks is not None:
        fake.completion.return_value = iter(stream_chunks)
    else:
        fake.completion.return_value = completion_return
    if completion_cost_side_effect is not None:
        fake.completion_cost.side_effect = completion_cost_side_effect
    elif completion_cost is not None:
        fake.completion_cost.return_value = completion_cost
    else:
        fake.completion_cost.return_value = 0.0
    fake.get_llm_provider.return_value = ("model", "fake-provider", "key", "url")
    fake.stream_chunk_builder.side_effect = lambda chunks, messages=None: (
        _build_final_from_chunks(chunks)
    )
    # Real litellm has an `exceptions` submodule; tests that exercise the
    # exception-mapping path patch this. By default leave it empty so the
    # mapper falls through to ProviderUnavailable.
    fake.exceptions = MagicMock(spec=[])
    p = LiteLLMProvider(api_key=api_key)
    p._litellm = fake
    return p, fake


def _build_final_from_chunks(chunks):
    """Tiny stand-in for litellm.stream_chunk_builder: glues content
    deltas and surfaces the LAST chunk's usage / tool_calls /
    finish_reason. Mirrors litellm's behaviour closely enough for unit
    tests."""
    chunks = list(chunks)
    if not chunks:
        return None
    text = "".join(
        (c.choices[0].delta.content or "")
        for c in chunks
        if c.choices and getattr(c.choices[0].delta, "content", None)
    )
    last = chunks[-1]
    usage = getattr(last, "usage", None)
    tool_calls = None
    finish_reason = "stop"
    for c in chunks:
        if not c.choices:
            continue
        tc = getattr(c.choices[0].delta, "tool_calls", None)
        if tc:
            tool_calls = tc
        fr = getattr(c.choices[0], "finish_reason", None)
        if fr:
            finish_reason = fr
    msg = MagicMock(content=text, tool_calls=tool_calls)
    choice = MagicMock(message=msg, finish_reason=finish_reason)
    return MagicMock(choices=[choice], usage=usage, model="model")


class TestLiteLLMProviderInit:
    def test_default_timeout(self):
        p = LiteLLMProvider()
        assert p._timeout_s == 60.0

    def test_custom_api_key(self):
        p = LiteLLMProvider(api_key="sk-test")
        assert p._api_key == "sk-test"

    def test_lazy_import_state(self):
        """The SDK reference must start as None so construction works
        on machines without litellm installed."""
        p = LiteLLMProvider()
        assert p._litellm is None


class TestLazyImport:
    def test_module_imports_without_litellm(self, monkeypatch):
        """The module itself must import even when litellm is not
        installed — matches the AnthropicProvider contract."""
        import sys

        real_litellm = sys.modules.pop("litellm", None)
        try:
            # Force re-import: pop the cached module from sys.modules.
            import importlib

            import cheetahclaws.kernel.runner.llm.litellm_provider as mod

            importlib.reload(mod)
            # Construction must succeed too.
            mod.LiteLLMProvider()
        finally:
            if real_litellm is not None:
                sys.modules["litellm"] = real_litellm

    def test_ensure_litellm_raises_when_missing(self, monkeypatch):
        """Calling the provider with litellm uninstalled must raise
        ProviderUnavailable (not ImportError) so the runner reports
        a clean error."""
        p = LiteLLMProvider()

        # Simulate `import litellm` failing.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "litellm":
                raise ImportError("no litellm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        req = LlmRequest(model="openai/gpt-4o", user="hi")
        with pytest.raises(ProviderUnavailable, match="litellm"):
            p(req)


class TestLiteLLMProviderCall:
    def test_calls_litellm_completion(self):
        mock_msg = MagicMock(content="hello", tool_calls=None)
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_choice = MagicMock(message=mock_msg, finish_reason="stop")
        ret = MagicMock(choices=[mock_choice], usage=mock_usage, model="openai/gpt-4o")
        p, fake = _make_provider_with_fake_litellm(
            completion_return=ret, api_key="sk-test"
        )

        req = LlmRequest(model="openai/gpt-4o", user="hi")
        resp = p(req)

        assert resp.text == "hello"
        assert resp.tokens_input == 10
        assert resp.tokens_output == 5
        kwargs = fake.completion.call_args.kwargs
        assert kwargs["drop_params"] is True
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["api_key"] == "sk-test"

    def test_omits_api_key_when_none(self):
        mock_msg = MagicMock(content="ok", tool_calls=None)
        ret = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=None,
            model="m",
        )
        p, fake = _make_provider_with_fake_litellm(completion_return=ret)
        req = LlmRequest(model="openai/gpt-4o", user="hi")
        p(req)
        assert "api_key" not in fake.completion.call_args.kwargs

    def test_system_prompt_included(self):
        mock_msg = MagicMock(content="ok", tool_calls=None)
        ret = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=None,
            model="m",
        )
        p, fake = _make_provider_with_fake_litellm(completion_return=ret)
        req = LlmRequest(model="openai/gpt-4o", system="be helpful", user="hi")
        p(req)
        messages = fake.completion.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "be helpful"}
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_multi_turn_messages(self):
        mock_msg = MagicMock(content="ok", tool_calls=None)
        ret = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=None,
            model="m",
        )
        p, fake = _make_provider_with_fake_litellm(completion_return=ret)
        msgs = (
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        )
        req = LlmRequest(model="openai/gpt-4o", messages=msgs)
        p(req)
        messages = fake.completion.call_args.kwargs["messages"]
        assert len(messages) == 3

    def test_returns_llm_response(self):
        mock_msg = MagicMock(content="test", tool_calls=None)
        ret = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=None,
            model="openai/gpt-4o",
        )
        p, _ = _make_provider_with_fake_litellm(completion_return=ret)
        req = LlmRequest(model="openai/gpt-4o", user="hi")
        resp = p(req)
        assert isinstance(resp, LlmResponse)
        assert resp.model == "openai/gpt-4o"


class TestCostCalculation:
    def test_cost_micro_populated_from_litellm(self):
        """litellm.completion_cost returns USD; we convert to micro-USD
        so the kernel ledger records spend on every successful call."""
        mock_msg = MagicMock(content="hi", tool_calls=None)
        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        ret = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=mock_usage,
            model="openai/gpt-4o",
        )
        p, _ = _make_provider_with_fake_litellm(
            completion_return=ret,
            completion_cost=0.000375,  # 100 in * $2.50/M + 50 out * $10/M
        )
        req = LlmRequest(model="openai/gpt-4o", user="hi")
        resp = p(req)
        # 0.000375 USD * 1_000_000 = 375 micro-USD
        assert resp.cost_micro == 375

    def test_cost_falls_back_to_zero_on_unknown_model(self):
        """If litellm can't price the model it raises; we swallow that
        and emit cost=0 rather than crashing the call."""
        mock_msg = MagicMock(content="hi", tool_calls=None)
        ret = MagicMock(
            choices=[MagicMock(message=mock_msg, finish_reason="stop")],
            usage=MagicMock(prompt_tokens=1, completion_tokens=1),
            model="weird/model",
        )
        p, _ = _make_provider_with_fake_litellm(
            completion_return=ret,
            completion_cost_side_effect=Exception("unknown model"),
        )
        req = LlmRequest(model="weird/model", user="hi")
        resp = p(req)
        assert resp.cost_micro == 0


class TestStreaming:
    def test_stream_emits_deltas_and_returns_usage(self):
        """Streaming path must reassemble chunks via
        stream_chunk_builder so the final response carries token
        counts — without this, every streamed call would record 0/0
        and break ledger accounting."""
        # Build three text chunks + a final usage chunk.
        c1 = MagicMock(choices=[MagicMock(delta=MagicMock(content="hel", tool_calls=None), finish_reason=None)])
        c2 = MagicMock(choices=[MagicMock(delta=MagicMock(content="lo", tool_calls=None), finish_reason=None)])
        c3 = MagicMock(
            choices=[MagicMock(delta=MagicMock(content=None, tool_calls=None), finish_reason="stop")],
            usage=MagicMock(prompt_tokens=4, completion_tokens=2),
        )
        # First two chunks have no usage attribute → set to None.
        c1.usage = None
        c2.usage = None
        p, _ = _make_provider_with_fake_litellm(stream_chunks=[c1, c2, c3])

        received = []
        req = LlmRequest(model="openai/gpt-4o", user="hi")
        resp = p.stream(req, lambda d: received.append(d))

        assert received == ["hel", "lo"]
        assert resp.text == "hello"
        assert resp.tokens_input == 4
        assert resp.tokens_output == 2

    def test_stream_preserves_tool_calls(self):
        """If the model emits tool_use during streaming, the final
        response must include it — RFC 0022 multi-iteration loops
        depend on this."""
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock(name="search", arguments='{"q":"x"}')
        tool_call.function.name = "search"
        c1 = MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content=None, tool_calls=[tool_call]),
                    finish_reason="tool_calls",
                )
            ],
            usage=MagicMock(prompt_tokens=5, completion_tokens=3),
        )
        p, _ = _make_provider_with_fake_litellm(stream_chunks=[c1])

        req = LlmRequest(model="openai/gpt-4o", user="search for x")
        resp = p.stream(req, lambda d: None)

        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["name"] == "search"
        assert resp.tool_calls[0]["input"] == {"q": "x"}
        assert resp.finish_reason == "tool_calls"

    def test_stream_passes_include_usage_flag(self):
        c = MagicMock(choices=[MagicMock(delta=MagicMock(content="x", tool_calls=None), finish_reason="stop")])
        c.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        p, fake = _make_provider_with_fake_litellm(stream_chunks=[c])
        req = LlmRequest(model="openai/gpt-4o", user="hi")
        p.stream(req, lambda d: None)
        kwargs = fake.completion.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}


class TestExceptionMapping:
    def test_auth_error_maps_to_invalid_request(self):
        """4xx-class errors (auth, bad request) should surface as
        ProviderInvalidRequest so callers can distinguish 'your fault'
        from 'their fault'."""

        class FakeAuthError(Exception):
            pass

        p, fake = _make_provider_with_fake_litellm(
            completion_side_effect=FakeAuthError("invalid api key")
        )
        # Mapper reads `self._litellm.exceptions.AuthenticationError`,
        # so we attach the fake class there.
        fake.exceptions = MagicMock(spec=[])
        fake.exceptions.AuthenticationError = FakeAuthError

        req = LlmRequest(model="openai/gpt-4o", user="hi")
        with pytest.raises(ProviderInvalidRequest):
            p(req)

    def test_other_errors_map_to_provider_unavailable(self):
        p, _ = _make_provider_with_fake_litellm(
            completion_side_effect=RuntimeError("upstream 503")
        )
        req = LlmRequest(model="openai/gpt-4o", user="hi")
        with pytest.raises(ProviderUnavailable, match="503"):
            p(req)


class TestLiteLLMProviderEdgeCases:
    def test_rejects_non_llm_request(self):
        p = LiteLLMProvider()
        with pytest.raises(ProviderInvalidRequest):
            p("not a request")

    def test_stream_rejects_non_callable_on_delta(self):
        p = LiteLLMProvider()
        req = LlmRequest(model="openai/gpt-4o", user="hi")
        with pytest.raises(ProviderInvalidRequest, match="on_delta"):
            p.stream(req, "not callable")


class TestToolCallParsingDefensive:
    """Regression tests for parser bugs that the original PR shipped:
    a malformed tool_call shouldn't crash the entire response."""

    def _provider_with_tool_call(self, tool_call):
        msg = MagicMock(content="", tool_calls=[tool_call])
        ret = MagicMock(
            choices=[MagicMock(message=msg, finish_reason="tool_calls")],
            usage=MagicMock(prompt_tokens=1, completion_tokens=1),
            model="openai/gpt-4o",
        )
        return _make_provider_with_fake_litellm(completion_return=ret)

    def test_skips_tool_call_with_missing_function(self):
        """If `tc.function` is None (or absent), the call is skipped
        rather than crashing on `tc.function.name`. Some providers send
        a sentinel tool_call without a function block."""
        tc = MagicMock()
        tc.function = None
        tc.id = "call_1"
        p, _ = self._provider_with_tool_call(tc)
        resp = p(LlmRequest(model="openai/gpt-4o", user="hi"))
        assert resp.tool_calls == ()

    def test_skips_tool_call_with_empty_name(self):
        """Empty function.name would fail LlmResponse validation
        downstream; skip the call instead of crashing the response."""
        tc = MagicMock()
        tc.function = MagicMock(arguments='{"x":1}')
        tc.function.name = ""
        tc.id = "call_2"
        p, _ = self._provider_with_tool_call(tc)
        resp = p(LlmRequest(model="openai/gpt-4o", user="hi"))
        assert resp.tool_calls == ()

    def test_coerces_non_dict_arguments_to_empty(self):
        """`function.arguments == "null"` is valid JSON but decodes to
        None, which the LlmResponse validator rejects. Coerce to {}
        so the rest of the response still flows through."""
        tc = MagicMock()
        tc.function = MagicMock(arguments="null")
        tc.function.name = "search"
        tc.id = "call_3"
        p, _ = self._provider_with_tool_call(tc)
        resp = p(LlmRequest(model="openai/gpt-4o", user="hi"))
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["input"] == {}

    def test_coerces_list_arguments_to_empty(self):
        """Same coercion when arguments is a valid JSON array."""
        tc = MagicMock()
        tc.function = MagicMock(arguments="[1,2,3]")
        tc.function.name = "search"
        tc.id = "call_4"
        p, _ = self._provider_with_tool_call(tc)
        resp = p(LlmRequest(model="openai/gpt-4o", user="hi"))
        assert resp.tool_calls[0]["input"] == {}

    def test_assigns_fallback_id_when_missing(self):
        """LlmResponse requires tool_calls[*].id to be non-empty; if
        the provider didn't supply one, fall back to call_<index>."""
        tc = MagicMock()
        tc.function = MagicMock(arguments='{"q":"x"}')
        tc.function.name = "search"
        tc.id = None  # provider didn't send one
        p, _ = self._provider_with_tool_call(tc)
        resp = p(LlmRequest(model="openai/gpt-4o", user="hi"))
        assert resp.tool_calls[0]["id"].startswith("call_")


class TestStreamingFallback:
    def test_cost_unknown_set_when_chunk_builder_fails(self):
        """When stream_chunk_builder is unavailable / fails, the
        fallback response must mark cost_unknown=True so the ledger
        doesn't mistake the 0/0 token output for a real free call."""
        chunk = MagicMock(
            choices=[MagicMock(
                delta=MagicMock(content="hi", tool_calls=None),
                finish_reason="stop",
            )],
        )
        chunk.usage = None
        p, fake = _make_provider_with_fake_litellm(stream_chunks=[chunk])
        # Force the builder to return None (simulating a very old
        # litellm or a malformed chunk list).
        fake.stream_chunk_builder.side_effect = lambda *a, **kw: None
        # Also blow it up — both shapes should hit the fallback.
        resp = p.stream(
            LlmRequest(model="openai/gpt-4o", user="hi"),
            lambda d: None,
        )
        assert resp.metadata.get("cost_unknown") is True
        assert resp.text == "hi"


_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestRegistration:
    def test_litellm_is_optional_dependency(self):
        """litellm must live under [project.optional-dependencies]
        (extra 'litellm'), NOT in core deps — installing cheetahclaws
        should not force-pull litellm's transitive chain."""
        toml = (_REPO_ROOT / "pyproject.toml").read_text()
        core_block = toml.split("[project.optional-dependencies]")[0]
        assert "litellm" not in core_block, (
            "litellm leaked into [project.dependencies]; it must stay "
            "under [project.optional-dependencies] so installs stay slim"
        )
        # And the extra itself must exist so `pip install
        # cheetahclaws[litellm]` resolves.
        assert 'litellm     = ["litellm' in toml or 'litellm = ["litellm' in toml
        # And it must be reachable via the `all` extra too.
        all_block = toml.split('all         = [')[1].split("]")[0]
        assert "litellm" in all_block, (
            "litellm missing from [project.optional-dependencies].all"
        )

    def test_litellm_registered_in_runner(self):
        """CC_LLM_PROVIDER=litellm must be routable through the
        subprocess runner, otherwise the provider isn't actually
        usable end-to-end."""
        from cheetahclaws.kernel.runner.llm import __main__ as runner_main

        import inspect

        src = inspect.getsource(runner_main._select_provider)
        assert 'name == "litellm"' in src
        assert "LiteLLMProvider" in src

    def test_litellm_in_top_level_providers_registry(self):
        """The CLI / Web UI consult providers.PROVIDERS when resolving
        --model X. Without a litellm entry there, no end-to-end caller
        can reach the new adapter — only direct Python use works."""
        from cheetahclaws import providers

        assert "litellm" in providers.PROVIDERS
        entry = providers.PROVIDERS["litellm"]
        assert entry["type"] == "litellm"
        # The dispatcher in providers.stream() branches on this type.
        assert callable(providers.stream_litellm)

    def test_model_string_routes_to_litellm(self):
        """`litellm/openai/gpt-4o` must resolve as provider=litellm
        with bare_model=openai/gpt-4o (the form litellm.completion
        wants), so the slash-prefix routing actually works."""
        from cheetahclaws import providers

        assert providers.detect_provider("litellm/openai/gpt-4o") == "litellm"
        assert providers.bare_model("litellm/openai/gpt-4o") == "openai/gpt-4o"
        assert (
            providers.detect_provider("litellm/bedrock/anthropic.claude-3-5-sonnet")
            == "litellm"
        )
