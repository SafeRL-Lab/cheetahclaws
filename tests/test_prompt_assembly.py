"""Tests for :func:`context.build_system_prompt` — dynamic block insertion.

The selection of ``prompts/base/*.md`` is covered in
``tests/test_prompt_selection.py``.  Here we verify that
``build_system_prompt`` correctly assembles base + env + conditional
fragments (memory / tmux / plan) and preserves the overall order.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws import context as _context


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test in an empty tmp cwd so real CLAUDE.md / .git don't leak in."""
    monkeypatch.chdir(tmp_path)
    yield


def _base_config(**overrides) -> dict:
    cfg = {"model": "claude-opus-4-7", "_session_id": "test-session"}
    cfg.update(overrides)
    return cfg


def test_assembled_prompt_contains_identity_and_env():
    prompt = _context.build_system_prompt(_base_config())
    assert "CheetahClaws" in prompt
    assert "# Environment" in prompt
    assert "Current date:" in prompt
    assert "Working directory:" in prompt


def test_plan_mode_appends_fragment_with_plan_file_filled(monkeypatch, tmp_path):
    plan_path = str(tmp_path / "plan.md")
    # Seed the runtime context so _render_plan_fragment can find the path.
    from cheetahclaws import runtime
    sctx = runtime.get_session_ctx("test-session")
    sctx.plan_file = plan_path
    try:
        prompt = _context.build_system_prompt(_base_config(permission_mode="plan"))
        assert "# Plan Mode (ACTIVE)" in prompt
        assert plan_path in prompt, "plan_file path must be interpolated into the fragment"
        # Raw placeholder must not leak through.
        assert "{plan_file}" not in prompt
    finally:
        sctx.plan_file = None
        runtime.release_session_ctx("test-session")


def test_plan_mode_absent_when_permission_mode_not_plan():
    prompt = _context.build_system_prompt(_base_config(permission_mode="auto"))
    assert "# Plan Mode (ACTIVE)" not in prompt


def test_tmux_fragment_absent_when_tmux_unavailable(monkeypatch):
    """If tmux isn't installed we must NOT inject the tmux block."""
    monkeypatch.setattr(_context, "_tmux_available", lambda: False)
    prompt = _context.build_system_prompt(_base_config())
    assert "TmuxNewSession" not in prompt


def test_tmux_fragment_present_when_tmux_available(monkeypatch):
    monkeypatch.setattr(_context, "_tmux_available", lambda: True)
    prompt = _context.build_system_prompt(_base_config(
        tool_profile="full", _active_tool_names=frozenset({"TmuxNewSession"}),
    ))
    assert "TmuxNewSession" in prompt
    assert "## Tmux (Terminal Multiplexer)" in prompt


def test_memory_block_injected_when_context_non_empty(monkeypatch):
    monkeypatch.setattr(_context, "get_memory_context", lambda: "- note one\n- note two")
    prompt = _context.build_system_prompt(_base_config())
    assert "# Memory" in prompt
    assert "note one" in prompt


def test_memory_block_omitted_when_context_empty(monkeypatch):
    monkeypatch.setattr(_context, "get_memory_context", lambda: "")
    prompt = _context.build_system_prompt(_base_config())
    # The base prompt mentions "MemorySave" etc. in the tool catalog, so
    # assert on the *section header* we only emit when memories exist.
    assert "Your persistent memories:" not in prompt


def test_assembly_order_is_base_then_env_then_memory_then_plan(monkeypatch):
    """Verify left-to-right ordering: base → env → memory → tmux → plan."""
    monkeypatch.setattr(_context, "get_memory_context", lambda: "- a memory")
    monkeypatch.setattr(_context, "_tmux_available", lambda: True)

    from cheetahclaws import runtime
    runtime.get_session_ctx("test-session").plan_file = "/tmp/plan.md"
    try:
        prompt = _context.build_system_prompt(
            _base_config(
                permission_mode="plan", tool_profile="full",
                _active_tool_names=frozenset({"TmuxNewSession"}),
            )
        )
    finally:
        runtime.get_session_ctx("test-session").plan_file = None
        runtime.release_session_ctx("test-session")

    idx_identity = prompt.index("CheetahClaws")
    idx_env = prompt.index("# Environment")
    idx_memory = prompt.index("Your persistent memories:")
    # The active-surface block may name TmuxNewSession earlier; use the
    # fragment header to assert the assembly position of the actual guidance.
    idx_tmux = prompt.index("## Tmux (Terminal Multiplexer)")
    idx_plan = prompt.index("# Plan Mode (ACTIVE)")

    assert idx_identity < idx_env < idx_memory < idx_tmux < idx_plan


def test_missing_config_falls_back_to_default():
    """build_system_prompt(None) must produce a usable prompt that uses the
    neutral default.md baseline — NOT a family-specific file like
    anthropic.md.  Falling back to a Claude-styled prompt would silently
    apply XML-tag structuring etc. to whatever model picked it up later.
    """
    from cheetahclaws.prompts import select as _select
    prompt = _context.build_system_prompt(None)
    assert "CheetahClaws" in prompt
    assert "# Environment" in prompt
    default_body = (_select._BASE_DIR / "default.md").read_text(encoding="utf-8")
    # The base portion of the prompt must match default.md verbatim, so
    # we can assert by checking the prompt starts with default's opening line.
    assert prompt.lstrip().startswith(default_body.splitlines()[0])


def test_active_tool_surface_matches_the_selected_profile(monkeypatch):
    monkeypatch.setattr(_context, "_tmux_available", lambda: False)

    standard = _context.build_system_prompt(_base_config(tool_profile="standard"))
    research = _context.build_system_prompt(_base_config(tool_profile="research"))

    assert "# Active Tool Surface" in standard
    assert "`WebFetch`" not in standard
    assert "`WebFetch`" in research


def test_planning_hint_is_only_shown_when_plan_tools_are_active(monkeypatch):
    """The compact default must not rely on slash-command import side effects."""
    monkeypatch.setattr(_context, "_tmux_available", lambda: False)
    monkeypatch.setattr(_context, "_render_commands_block", lambda: "")

    standard = _context.build_system_prompt(_base_config(tool_profile="standard"))
    orchestration = _context.build_system_prompt(
        _base_config(tool_profile="orchestration")
    )

    assert "For complex or multi-file work" not in standard
    assert "For complex or multi-file work" in orchestration


def test_standard_surface_omits_tmux_fragment_even_when_available(monkeypatch):
    monkeypatch.setattr(_context, "_tmux_available", lambda: True)

    prompt = _context.build_system_prompt(_base_config(tool_profile="standard"))

    assert "TmuxNewSession" not in prompt


def test_full_surface_omits_tmux_fragment_when_tool_is_not_registered(monkeypatch):
    monkeypatch.setattr(_context, "_tmux_available", lambda: True)

    prompt = _context.build_system_prompt(_base_config(tool_profile="full"))

    assert "TmuxNewSession" not in prompt
