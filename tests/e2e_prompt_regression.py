"""End-to-end regression check that build_system_prompt still produces
the expected shape after the prompts/ refactor.

Compares the generated prompt — with known-dynamic lines masked — to a
golden fixture on disk.  If this test fails after an intentional prompt
change, regenerate the fixture by running:

    python -m tests.e2e_prompt_regression --regenerate

(that flag is implemented below).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path (mirrors other tests in this suite)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import context as _context

_FIXTURE = Path(__file__).parent / "fixtures" / "golden_default_prompt.txt"

# Lines we expect to vary per-run.  Masked before comparison.
_MASKS = [
    (re.compile(r"^- Current date: .+$", re.M),         "- Current date: <MASKED>"),
    (re.compile(r"^- Working directory: .+$", re.M),    "- Working directory: <MASKED>"),
    (re.compile(r"^- Platform: .+$", re.M),             "- Platform: <MASKED>"),
]


def _mask(prompt: str) -> str:
    for pattern, replacement in _MASKS:
        prompt = pattern.sub(replacement, prompt)
    return prompt.rstrip() + "\n"


def _generate_masked_prompt(tmp_path, monkeypatch) -> str:
    """Build a prompt with all optional blocks forced off, then mask dynamics."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_context, "get_memory_context", lambda: "")
    monkeypatch.setattr(_context, "get_git_info",       lambda: "")
    monkeypatch.setattr(_context, "get_claude_md",      lambda: "")
    monkeypatch.setattr(_context, "get_platform_hints", lambda: "")
    monkeypatch.setattr(_context, "_tmux_available",    lambda: False)

    # Use a model whose family has NO overlay yet so this test pins to
    # the default.md baseline only — family-specific overlay edits don't
    # invalidate the fixture. As of this writing kimi has no overlay
    # (qwen used to but does now). If kimi ever gets one, switch to any
    # other no-overlay family (llama, mistral, gemma, phi, glm, minimax,
    # deepseek) per prompts/select.py:_OVERLAY_RULES.
    cfg = {"model": "kimi/moonshot-v1-128k", "_session_id": "regression-test"}
    return _mask(_context.build_system_prompt(cfg))


def test_default_prompt_matches_golden(tmp_path, monkeypatch):
    if not _FIXTURE.exists():
        pytest.skip(
            f"golden fixture missing: {_FIXTURE}. "
            f"Regenerate with: python {__file__} --regenerate"
        )
    actual = _generate_masked_prompt(tmp_path, monkeypatch)
    expected = _FIXTURE.read_text(encoding="utf-8")
    assert actual == expected, (
        "Default prompt drifted from golden fixture.\n"
        f"If this change is intentional, regenerate the fixture:\n"
        f"  python {__file__} --regenerate\n\n"
        f"First 500 chars of actual:\n{actual[:500]}\n"
        f"First 500 chars of expected:\n{expected[:500]}"
    )


def _regenerate() -> None:
    """Write the current output to the fixture.  Invoked by --regenerate."""
    import tempfile
    from unittest import mock

    # Manually replicate the monkeypatches the pytest fixture applies.
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(_context, "get_memory_context", return_value=""), \
             mock.patch.object(_context, "get_git_info",       return_value=""), \
             mock.patch.object(_context, "get_claude_md",      return_value=""), \
             mock.patch.object(_context, "get_platform_hints", return_value=""), \
             mock.patch.object(_context, "_tmux_available",    return_value=False):
            import os
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                cfg = {"model": "kimi/moonshot-v1-128k", "_session_id": "regression-test"}
                prompt = _mask(_context.build_system_prompt(cfg))
            finally:
                os.chdir(prev_cwd)

    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE.write_text(prompt, encoding="utf-8")
    print(f"Wrote {len(prompt)} chars to {_FIXTURE}")


def test_env_block_separates_platform_from_git_info(tmp_path, monkeypatch):
    """Regression: the Platform line must end in \\n so a non-empty git_info
    (which itself starts with "- Git branch:" without a leading newline)
    doesn't get glued onto the Platform line.

    Catches the bug where ``- Platform: Linux`` and ``- Git branch: main``
    rendered as ``- Platform: Linux- Git branch: main`` on the same line.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_context, "get_memory_context", lambda: "")
    monkeypatch.setattr(_context, "get_git_info",
                         lambda: "- Git branch: main\n- Recent commits:\n  abc123 first\n")
    monkeypatch.setattr(_context, "get_claude_md",      lambda: "")
    monkeypatch.setattr(_context, "get_platform_hints", lambda: "")
    monkeypatch.setattr(_context, "_tmux_available",    lambda: False)

    cfg = {"model": "kimi/moonshot-v1-128k", "_session_id": "regression-test"}
    prompt = _context.build_system_prompt(cfg)

    # The Platform line must terminate before "- Git branch:" begins.
    assert "- Platform: " in prompt
    assert "- Git branch: main" in prompt
    # Crucially: no line should contain BOTH Platform and Git branch.
    glued = [ln for ln in prompt.splitlines()
             if "- Platform:" in ln and "- Git branch:" in ln]
    assert not glued, (
        "Platform line is glued to git info on the same line — "
        "_render_env_block must keep the trailing \\n on the Platform line.\n"
        f"Offending line(s): {glued}"
    )


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        print("Usage: python e2e_prompt_regression.py --regenerate")
        sys.exit(1)
