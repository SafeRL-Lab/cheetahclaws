"""Tests for local-OCR enrichment of the ``/image`` command.

``ocr_image_bytes`` (tools/files.py) is a best-effort helper: it must
NEVER raise — missing deps, corrupt bytes, and empty results all
collapse to ''.  ``cmd_image`` (commands/core.py) appends the OCR text
to the prompt so non-vision models can act on clipboard screenshots.
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws.tools import files as files_mod
from cheetahclaws.tools.files import ocr_image_bytes


# ── ocr_image_bytes: never raises ─────────────────────────────────────────


def test_ocr_corrupt_bytes_returns_empty():
    assert ocr_image_bytes(b"this is not an image") == ""


def test_ocr_empty_bytes_returns_empty():
    assert ocr_image_bytes(b"") == ""


def test_ocr_missing_deps_returns_empty(monkeypatch):
    """If pytesseract/Pillow are not installed, return '' (no crash)."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("pytesseract", "PIL"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ocr_image_bytes(b"\x89PNG\r\n\x1a\n") == ""


def test_ocr_engine_error_returns_empty(monkeypatch):
    """Tesseract binary missing / engine blowing up → '' (no crash)."""
    pytest.importorskip("PIL")
    pytest.importorskip("pytesseract")
    import pytesseract

    def boom(*_a, **_k):
        raise RuntimeError("tesseract is not installed or it's not in your PATH")

    monkeypatch.setattr(pytesseract, "image_to_string", boom)
    png = _make_png_bytes()
    assert ocr_image_bytes(png) == ""


def test_ocr_strips_whitespace(monkeypatch):
    pytest.importorskip("PIL")
    pytest.importorskip("pytesseract")
    import pytesseract

    monkeypatch.setattr(
        pytesseract, "image_to_string", lambda *_a, **_k: "  hello world \n\n"
    )
    png = _make_png_bytes()
    assert ocr_image_bytes(png) == "hello world"


# ── cmd_image: prompt enrichment ──────────────────────────────────────────


def _make_png_bytes(size=(60, 20)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def _clipboard_image(monkeypatch):
    """Fake a clipboard image so cmd_image runs headless in CI."""
    pil = pytest.importorskip("PIL")
    from PIL import Image, ImageGrab

    img = Image.new("RGB", (60, 20), "white")
    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda: img)
    return img


def test_cmd_image_appends_ocr_text(monkeypatch, _clipboard_image):
    from cheetahclaws.commands import core as core_mod

    monkeypatch.setattr(
        files_mod, "ocr_image_bytes", lambda *_a, **_k: "Traceback: KeyError 'foo'"
    )
    config = {"_session_id": "test-ocr-1"}
    result = core_mod.cmd_image("what broke?", state=None, config=config)

    assert isinstance(result, tuple) and result[0] == "__image__"
    prompt = result[1]
    assert prompt.startswith("what broke?")
    assert "Traceback: KeyError 'foo'" in prompt
    assert "local OCR" in prompt


def test_cmd_image_no_ocr_text_leaves_prompt_untouched(monkeypatch, _clipboard_image):
    from cheetahclaws.commands import core as core_mod

    monkeypatch.setattr(files_mod, "ocr_image_bytes", lambda *_a, **_k: "")
    config = {"_session_id": "test-ocr-2"}
    result = core_mod.cmd_image("describe", state=None, config=config)

    assert isinstance(result, tuple) and result[0] == "__image__"
    assert result[1] == "describe"


def test_cmd_image_truncates_huge_ocr(monkeypatch, _clipboard_image):
    from cheetahclaws.commands import core as core_mod

    monkeypatch.setattr(files_mod, "ocr_image_bytes", lambda *_a, **_k: "x" * 50000)
    config = {"_session_id": "test-ocr-3"}
    result = core_mod.cmd_image("", state=None, config=config)

    prompt = result[1]
    # 8000-char cap on the OCR block keeps small context windows safe
    assert len(prompt) < 10000


def test_cmd_image_still_sets_pending_image(monkeypatch, _clipboard_image):
    """OCR enrichment must not break the vision path (pending_image b64)."""
    from cheetahclaws import runtime
    from cheetahclaws.commands import core as core_mod

    monkeypatch.setattr(files_mod, "ocr_image_bytes", lambda *_a, **_k: "some text")
    config = {"_session_id": "test-ocr-4"}
    core_mod.cmd_image("", state=None, config=config)

    sctx = runtime.get_ctx(config)
    assert sctx.pending_image, "pending_image should still carry the base64 PNG"
