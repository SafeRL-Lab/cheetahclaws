"""Tests for graceful handling of empty / malformed tool-call arguments.

Weak models on OpenAI-compat endpoints (qwen2.5 + vLLM, etc.) sometimes
fire a tool_call with the function name set but the arguments object
empty or partial. The agent loop must surface a helpful "missing
required parameter" string the model can self-correct from, NOT raise
a bare KeyError that bubbles up as `Error executing Write: KeyError:
'file_path'`.

The regression these tests guard:
    [cheetahclaws] /ssj brainstorm → main agent fires Write({}) → wrapper
    in tools/__init__.py used inputs['file_path'] for the permission
    description → KeyError before the registered ToolDef's friendly
    lambda ever ran.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws.tools import execute_tool


@pytest.mark.parametrize("name,empty_inputs,must_contain", [
    ("Write",        {},                              "missing required parameter"),
    ("Write",        {"file_path": ""},               "missing required parameter"),
    ("Write",        {"file_path": "/tmp/x", "content": ""},  None),  # permitted shape — runs
    ("Edit",         {},                              "missing required parameter"),
    ("Edit",         {"file_path": ""},               "missing required parameter"),
    ("Read",         {},                              "missing required parameter"),
    ("Read",         {"file_path": ""},               "missing required parameter"),
])
def test_missing_required_args_returns_friendly_error(name, empty_inputs, must_contain):
    """A tool call with empty args must return a friendly error string,
    NEVER raise KeyError up to the agent loop."""
    out = execute_tool(name, empty_inputs, permission_mode="accept-all", config={})
    assert isinstance(out, str), f"{name} returned non-string: {type(out).__name__}"
    if must_contain is not None:
        assert must_contain.lower() in out.lower(), (
            f"{name}({empty_inputs}) → {out!r} (expected '{must_contain}' substring)"
        )
    # The regression: this string was the KeyError fingerprint.
    assert "KeyError" not in out, (
        f"{name}({empty_inputs}) leaked KeyError to caller: {out!r}"
    )


def test_bash_empty_command_returns_friendly_error():
    """Bash with empty command — the wrapper's _is_safe_bash gate must not
    crash on empty input, and the inner ToolDef must surface the friendly
    'requires a non-empty command' message."""
    out = execute_tool("Bash", {}, permission_mode="accept-all", config={})
    assert isinstance(out, str)
    assert "KeyError" not in out
    # The inner lambda's exact phrasing.
    assert "non-empty" in out.lower() or "command" in out.lower()


def test_notebookedit_empty_args_does_not_keyerror():
    """NotebookEdit's permission description previously did
    inputs['notebook_path'] — the .get() fix must let empty args through
    to the registered tool's own missing-arg path."""
    out = execute_tool("NotebookEdit", {}, permission_mode="accept-all", config={})
    assert isinstance(out, str)
    assert "KeyError" not in out
