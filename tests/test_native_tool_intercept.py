"""Tests for the native tool-call interceptor in providers.stream_openai_compat.

Targets Gemma 4 + vLLM `hermes` parser mismatch where Gemma's native
`<|tool_call>call:Foo{...}<tool_call|>` format leaks into the streamed
text. The interceptor catches it and converts to a proper tool_calls entry.
"""
from __future__ import annotations

import pytest

from cheetahclaws.providers import (
    _find_native_tool_marker,
    _extract_native_tool_calls,
    TextChunk, AssistantTurn,
)


# ── Marker detection ─────────────────────────────────────────────────────

def test_find_native_marker_gemma_official():
    text = "Some preamble <|tool_call|>{\"name\":\"x\"}"
    idx = _find_native_tool_marker(text)
    assert idx == len("Some preamble ")


def test_find_native_marker_gemma_4_variant():
    text = "Let me search. <|tool_call>call:Research{\"q\":\"v\"}<tool_call|>"
    idx = _find_native_tool_marker(text)
    assert idx == text.index("<|tool_call>")


def test_find_native_marker_returns_none_when_absent():
    assert _find_native_tool_marker("Just regular text.") is None


def test_find_native_marker_picks_earliest():
    text = "<tool_call>foo</tool_call> and later <|tool_call>bar<tool_call|>"
    idx = _find_native_tool_marker(text)
    assert idx == 0  # <tool_call> wins (earliest)


def test_find_native_marker_mistral():
    text = "Sure. [TOOL_CALLS] [{\"name\": \"x\"}]"
    idx = _find_native_tool_marker(text)
    assert idx == text.index("[TOOL_CALLS]")


# ── Format 2 (Gemma's call:NAME format — what the user actually saw) ─────

def test_extract_gemma_call_name_format():
    buf = '<|tool_call>call:Research{"topic":"NVDA"}<tool_call|>'
    calls = _extract_native_tool_calls(buf)
    assert len(calls) == 1
    assert calls[0]["name"] == "Research"
    assert calls[0]["input"] == {"topic": "NVDA"}


def test_extract_gemma_call_handles_quote_escapes():
    """Gemma sometimes escapes quotes as <|"|> inside its native format."""
    buf = '<|tool_call>call:Research{"topic":<|"|>NVDA<|"|>}<tool_call|>'
    calls = _extract_native_tool_calls(buf)
    assert len(calls) == 1
    assert calls[0]["name"] == "Research"
    assert calls[0]["input"] == {"topic": "NVDA"}


def test_extract_gemma_call_with_official_closer():
    buf = '<|tool_call|>call:Foo{"a":1}<|end_tool_call|>'
    calls = _extract_native_tool_calls(buf)
    assert len(calls) == 1
    assert calls[0]["name"] == "Foo"


# ── Format 1 (JSON envelope) ─────────────────────────────────────────────

def test_extract_json_envelope_format():
    buf = '<|tool_call|>{"name": "Search", "arguments": {"q": "NVDA"}}<|end_tool_call|>'
    calls = _extract_native_tool_calls(buf)
    assert len(calls) == 1
    assert calls[0]["name"] == "Search"
    assert calls[0]["input"] == {"q": "NVDA"}


def test_extract_json_envelope_with_args_alias():
    buf = '<|tool_call|>{"function": "X", "args": {"k": 1}}<|end_tool_call|>'
    calls = _extract_native_tool_calls(buf)
    assert len(calls) == 1
    assert calls[0]["name"] == "X"
    assert calls[0]["input"] == {"k": 1}


# ── Mistral [TOOL_CALLS] format ──────────────────────────────────────────

def test_extract_mistral_format():
    buf = '[TOOL_CALLS] [{"name": "Web", "arguments": {"q": "NVDA"}}]'
    calls = _extract_native_tool_calls(buf)
    assert len(calls) == 1
    assert calls[0]["name"] == "Web"
    assert calls[0]["input"] == {"q": "NVDA"}


# ── Robustness ──────────────────────────────────────────────────────────

def test_extract_returns_empty_on_empty_buf():
    assert _extract_native_tool_calls("") == []


def test_extract_returns_empty_on_unparsable_garbage():
    buf = '<|tool_call>not a valid call format at all<tool_call|>'
    calls = _extract_native_tool_calls(buf)
    assert calls == []


def test_extract_handles_multiple_calls_in_one_buffer():
    buf = (
        '<|tool_call>call:A{"x":1}<tool_call|>'
        '<|tool_call>call:B{"y":2}<tool_call|>'
    )
    calls = _extract_native_tool_calls(buf)
    assert len(calls) == 2
    assert calls[0]["name"] == "A" and calls[1]["name"] == "B"


# ── End-to-end: stream_openai_compat with mocked client ──────────────────

class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, delta, usage=None):
        self.choices = [_FakeChoice(delta)]
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
    def __iter__(self):
        return iter(self.chunks)


class _FakeChatCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
    def create(self, **kwargs):
        return _FakeStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks):
        self.completions = _FakeChatCompletions(chunks)


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None):
        self._chunks = []
    def _set_chunks(self, chunks):
        self.chat = _FakeChat(chunks)


def test_stream_buffers_gemma_output_and_emits_tool_call(monkeypatch):
    """End-to-end: streaming Gemma's native format → no garbage to user,
    proper tool_calls in AssistantTurn."""
    from cheetahclaws.providers import stream_openai_compat

    chunks = [
        _FakeChunk(_FakeDelta(content="Sure, let me check. ")),
        _FakeChunk(_FakeDelta(content="<|tool_call>")),
        _FakeChunk(_FakeDelta(content='call:Research{"topic":"NVDA"}')),
        _FakeChunk(_FakeDelta(content="<tool_call|>")),
    ]

    fake = _FakeOpenAI()
    fake._set_chunks(chunks)
    # providers.py does `from openai import OpenAI` *inside* the function,
    # so we patch the imported name on the openai module itself.
    import openai as _real
    monkeypatch.setattr(_real, "OpenAI", lambda **k: fake)

    events = list(stream_openai_compat(
        api_key="dummy", base_url="http://localhost:8000/v1",
        model="custom/gemma-4-31B-it",
        system="sys",
        messages=[{"role": "user", "content": "Find NVDA stocks"}],
        tool_schemas=[{"name": "Research",
                       "description": "search",
                       "input_schema": {"type": "object", "properties": {}}}],
        config={},
    ))

    text_chunks = [e for e in events if isinstance(e, TextChunk)]
    turns = [e for e in events if isinstance(e, AssistantTurn)]

    # Pre-marker text was yielded as a TextChunk
    pre_marker = "".join(c.text for c in text_chunks)
    assert "Sure, let me check." in pre_marker
    # No <|tool_call> in any TextChunk — interceptor caught it
    assert "<|tool_call>" not in pre_marker
    assert "<tool_call|>" not in pre_marker
    assert "call:Research" not in pre_marker

    # AssistantTurn carries the parsed tool call
    assert len(turns) == 1
    assert len(turns[0].tool_calls) == 1
    assert turns[0].tool_calls[0]["name"] == "Research"
    assert turns[0].tool_calls[0]["input"] == {"topic": "NVDA"}


def test_stream_falls_back_to_text_when_native_call_unparsable(monkeypatch):
    """If buffering started but the format is unrecognisable, emit the raw
    buffer as text so the user sees something."""
    from cheetahclaws.providers import stream_openai_compat

    chunks = [
        _FakeChunk(_FakeDelta(content="Looking up. ")),
        _FakeChunk(_FakeDelta(content="<|tool_call>completely-malformed-no-close")),
    ]

    fake = _FakeOpenAI()
    fake._set_chunks(chunks)
    import openai as _real
    monkeypatch.setattr(_real, "OpenAI", lambda **k: fake)

    events = list(stream_openai_compat(
        api_key="dummy", base_url="http://localhost:8000/v1",
        model="custom/gemma-4-31B-it",
        system="sys",
        messages=[{"role": "user", "content": "test"}],
        tool_schemas=[],
        config={},
    ))

    turns = [e for e in events if isinstance(e, AssistantTurn)]
    assert len(turns) == 1
    # No tool calls extracted
    assert turns[0].tool_calls == []
    # But the raw buffer is in the final text so user sees something
    assert "completely-malformed" in turns[0].text
