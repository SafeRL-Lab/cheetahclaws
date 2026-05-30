"""End-to-end: LLM calls GetFolderDescription on a real tmp_path layout.

Files with an inline `# [desc] ... [/desc]` tag return that tag verbatim
without any LLM call. Files without a tag would normally trigger a
describer LLM call -- we put the tag on every fixture file so the test
stays provider-independent and fast. Only `providers.stream` is mocked.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import tools as _tools_init  # noqa: F401 - registers GetFolderDescription
import folder_desc.cache as cache_mod
from agent import AgentState, run
from providers import AssistantTurn


def _scripted_stream(turns):
    cursor = iter(turns)

    def fake_stream(**_kwargs):
        spec = next(cursor)
        yield AssistantTurn(
            text=spec.get("text", ""),
            tool_calls=spec.get("tool_calls") or [],
            in_tokens=1, out_tokens=1,
        )

    return fake_stream


@pytest.fixture
def codebase(tmp_path, monkeypatch):
    """Build a small code tree with inline [desc] tags and redirect the cache."""
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "_cache")

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "# [desc] public API surface [/desc]\n\ndef hello(): ...\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "b.py").write_text(
        "# [desc] internal helpers [/desc]\n\n_x = 1\n",
        encoding="utf-8",
    )
    return tmp_path


def test_llm_sees_folder_tree_with_descriptions(monkeypatch, codebase):
    """Drive agent.run: the LLM calls GetFolderDescription and the tool_result
    carries a tree containing both files' inline descriptions."""
    turns = [
        {"tool_calls": [{
            "id": "fd1",
            "name": "GetFolderDescription",
            "input": {"folder_path": str(codebase)},
        }]},
        {"text": "got it"},
    ]
    monkeypatch.setattr("agent.stream", _scripted_stream(turns))

    state = AgentState()
    config = {"model": "test", "permission_mode": "accept-all",
              "_session_id": "fd_e2e", "disabled_tools": ["Agent"]}
    list(run("describe the folder", state, config, "sys"))

    tool_result = next(m for m in state.messages
                       if m.get("role") == "tool" and m.get("tool_call_id") == "fd1")
    content = tool_result["content"]
    assert "a.py" in content and "b.py" in content
    assert "public API surface" in content
    assert "internal helpers" in content
