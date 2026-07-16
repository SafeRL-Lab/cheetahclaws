"""Enforce line-count caps on base prompts and family overlays.

Rationale (from the Gemini 3 prompting guide):

    "Once a system instruction becomes a 300-line constitution, you can
    no longer tell what's working and what's superstition."

CheetahClaws splits the budget two ways:

* ``prompts/base/default.md`` — the **shared baseline** for every model.
  Cap: 150 lines.  Long-lived conditional content goes to
  ``prompts/fragments/*.md`` and is appended only when the runtime
  condition fires (tmux available, plan mode active, …).

* ``prompts/overlays/*.md`` — **family-specific quirks** appended on top
  of the baseline.  Cap: 20 lines.  An overlay should hold one quirk
  with an authoritative source link, not a re-implementation of general
  prompt-engineering wisdom (that belongs in default.md so all models
  benefit).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BASE_DIR     = Path(__file__).parent.parent / "cheetahclaws" / "prompts" / "base"
_OVERLAYS_DIR = Path(__file__).parent.parent / "cheetahclaws" / "prompts" / "overlays"

# Keep these in sync with prompts/README.md.  Bump deliberately, not by accident.
MAX_BASE_PROMPT_LINES = 150
MAX_OVERLAY_LINES     = 20


def _base_files() -> list[Path]:
    return sorted(_BASE_DIR.glob("*.md"))


def _overlay_files() -> list[Path]:
    return sorted(_OVERLAYS_DIR.glob("*.md"))


def test_base_prompt_directory_exists():
    assert _BASE_DIR.is_dir(), f"missing directory: {_BASE_DIR}"
    assert _base_files(), "expected at least one base prompt file"


def test_overlays_directory_exists():
    assert _OVERLAYS_DIR.is_dir(), f"missing directory: {_OVERLAYS_DIR}"


@pytest.mark.parametrize("path", _base_files(), ids=lambda p: p.name)
def test_base_prompt_under_line_cap(path: Path):
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert line_count <= MAX_BASE_PROMPT_LINES, (
        f"{path.name} has {line_count} lines, cap is {MAX_BASE_PROMPT_LINES}. "
        f"Extract conditional content into prompts/fragments/*.md or split "
        f"family-specific quirks into prompts/overlays/*.md."
    )


@pytest.mark.parametrize("path", _overlay_files(), ids=lambda p: p.name)
def test_overlay_under_line_cap(path: Path):
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert line_count <= MAX_OVERLAY_LINES, (
        f"{path.name} has {line_count} lines, cap is {MAX_OVERLAY_LINES}. "
        f"Overlays must be short and quirk-focused — anything broader "
        f"belongs in default.md so it benefits every model."
    )


@pytest.mark.parametrize("path", _overlay_files(), ids=lambda p: p.name)
def test_overlay_cites_source(path: Path):
    """Every overlay must point to an official prompting guide in a top comment.

    This is the gating discipline that keeps overlays grounded — no
    folklore, no re-derived wisdom.  If a maintainer can't link to an
    official source for a quirk, that quirk doesn't belong in an overlay.
    """
    head = path.read_text(encoding="utf-8")[:600].lower()
    assert "<!-- source:" in head, (
        f"{path.name}: missing top-of-file '<!-- Source: ... -->' citation. "
        f"Overlays must reference an authoritative prompting guide."
    )
