"""Tests for the /theme command and ui.render.apply_theme.

Regression coverage for the original PR #92 issues:
  - info() and ok() must not collapse to the same color
  - render_diff additions (green) and removals (red) must remain distinguishable
  - 'none' theme must produce truly plain text (no escape codes)
  - CODE_THEME must follow the active theme
"""
from __future__ import annotations

import importlib

import pytest

import ui.render as render
from ui.render import THEMES, apply_theme, C, _rgb, clr


@pytest.fixture(autouse=True)
def _restore_default_theme():
    """Ensure each test starts and ends on the default theme."""
    apply_theme("default")
    yield
    apply_theme("default")


def test_themes_dict_schema():
    """Every theme must have a 'code' key, plus either color hexes or disable_color."""
    for name, p in THEMES.items():
        assert isinstance(p, dict), f"theme {name} not a dict"
        assert "code" in p, f"theme {name} missing 'code'"
        if p.get("disable_color"):
            continue
        for key in ("accent", "warn"):
            assert key in p, f"theme {name} missing '{key}'"
            v = p[key]
            assert isinstance(v, str) and v.startswith("#") and len(v) == 7, (
                f"theme {name}.{key} is not a 7-char hex: {v!r}"
            )


def test_apply_theme_unknown_returns_false():
    assert apply_theme("does-not-exist") is False


def test_apply_theme_changes_color_map():
    apply_theme("default")
    before = C["cyan"]
    apply_theme("dracula")
    assert C["cyan"] != before
    assert C["cyan"] == _rgb("#BD93F9")


def test_info_and_ok_distinguishable_in_every_theme():
    """Regression: PR #92 originally collapsed cyan/green/blue to accent,
    making info() and ok() visually identical. They must stay distinct
    (or both empty for the 'none' theme)."""
    for name, p in THEMES.items():
        apply_theme(name)
        if p.get("disable_color"):
            assert C["cyan"] == "" and C["green"] == ""
            continue
        assert C["cyan"] != C["green"], (
            f"theme {name!r}: info (cyan) and ok (green) collapsed to {C['cyan']!r}"
        )


def test_diff_additions_and_removals_distinguishable():
    """render_diff colors '+' green and '-' red. They must stay different."""
    for name, p in THEMES.items():
        apply_theme(name)
        if p.get("disable_color"):
            continue
        assert C["green"] != C["red"], (
            f"theme {name!r}: diff add (green) and remove (red) collapsed"
        )


def test_none_theme_produces_plain_text():
    apply_theme("none")
    for k in ("cyan", "green", "yellow", "red", "blue", "magenta",
              "white", "bold", "dim", "reset"):
        assert C[k] == "", f"none theme should clear C[{k!r}], got {C[k]!r}"
    assert clr("hello", "cyan") == "hello"
    assert clr("warn-text", "yellow", "bold") == "warn-text"


def test_code_theme_tracks_active_theme():
    apply_theme("dracula")
    assert render.CODE_THEME == "dracula"
    apply_theme("nord")
    assert render.CODE_THEME == "nord"
    apply_theme("default")
    assert render.CODE_THEME == "monokai"


def test_apply_theme_idempotent_across_state():
    """Applying theme A after any prior theme must yield the same C dict."""
    apply_theme("synthwave")
    apply_theme("default")
    snapshot1 = dict(C)

    apply_theme("matrix")
    apply_theme("dracula")
    apply_theme("default")
    snapshot2 = dict(C)

    assert snapshot1 == snapshot2, (
        "apply_theme leaks state between invocations — same theme should "
        "produce the same C regardless of prior theme"
    )


def test_make_renderable_passes_code_theme():
    """The Rich Markdown renderable must use the active CODE_THEME."""
    pytest.importorskip("rich")
    apply_theme("dracula")
    md = render._make_renderable("# heading\n```py\nx=1\n```")
    # Rich Markdown stores the code theme on the instance
    assert getattr(md, "code_theme", None) == "dracula"
