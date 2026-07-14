"""Regression tests for bounded input work in high-volume tools."""
from __future__ import annotations

import sys
import types

from cheetahclaws.tools.files import (
    _extract_page_prefix,
    _parse_page_range,
    _parse_page_range_capped,
    _read_pdf,
)
from cheetahclaws.tools.fs import _read
from cheetahclaws.tools.web import _webfetch


def test_read_streams_requested_line_range(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("zero\none\ntwo\nthree\n", encoding="utf-8")

    result = _read(str(path), offset=1, limit=2, max_bytes=1_000)

    assert "     2\tone" in result
    assert "     3\ttwo" in result
    assert "zero" not in result
    assert "three" not in result


def test_read_caps_a_single_huge_line_without_loading_it(tmp_path):
    path = tmp_path / "minified.js"
    path.write_bytes(b"x" * 200_000 + b"\nnext\n")

    result = _read(str(path), max_bytes=128)

    assert "Read stopped after 128 source bytes" in result
    assert len(result) < 1_000


def test_read_never_returns_more_than_the_source_byte_cap_for_utf8(tmp_path):
    path = tmp_path / "emoji.txt"
    path.write_text("😀" * 10 + "\n", encoding="utf-8")

    result = _read(str(path), max_bytes=5, scan_max_bytes=100)
    visible = result.split("\n[...", 1)[0].split("\t", 1)[1]

    assert len(visible.encode("utf-8")) <= 5
    assert "stopped after 5 source bytes" in result


def test_read_does_not_claim_truncation_at_exact_source_byte_eof(tmp_path):
    path = tmp_path / "exact.txt"
    path.write_bytes(b"exact")

    result = _read(str(path), max_bytes=5, scan_max_bytes=100)

    assert "exact" in result
    assert "stopped after" not in result


def test_webfetch_streams_and_caps_response(monkeypatch):
    class FakeResponse:
        headers = {"content-type": "text/html", "content-length": "1000"}
        encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self, **_kwargs):
            yield b"<p>" + b"a" * 100
            yield b"b" * 900 + b"</p>"

    fake_httpx = types.SimpleNamespace(stream=lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = _webfetch("https://example.test", max_bytes=128)

    assert "WebFetch stopped after 128 response bytes" in result
    assert len(result) < 400


def test_pdf_extract_stops_after_character_cap(monkeypatch, tmp_path):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def get_text(self):
            return self._text

    class FakeDoc:
        def __init__(self):
            self.pages = [FakePage("a" * 900), FakePage("b" * 900)]
            self.closed = False

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

        def close(self):
            self.closed = True

    doc = FakeDoc()
    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda _p: doc))
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-fake")

    result = _read_pdf(
        {"file_path": str(path)},
        {"pdf_extract_max_chars": 1_000, "pdf_extract_max_pages": 50},
    )

    assert doc.closed is True
    assert "ReadPDF stopped at 1,000 extracted characters" in result
    assert len(result) < 1_500


def test_pdf_page_range_is_bounded_before_building_a_large_list():
    assert _parse_page_range("1-1000000", 1_000_000, max_pages=3) == [0, 1, 2]


def test_pdf_page_range_reports_when_explicit_request_is_capped():
    pages, truncated = _parse_page_range_capped("1-3", 3, max_pages=2)
    assert pages == [0, 1]
    assert truncated is True


def test_pdf_page_range_ignores_out_of_range_singletons_before_capping():
    pages, truncated = _parse_page_range_capped("999,1,2", 3, max_pages=2)
    assert pages == [0, 1]
    assert truncated is False


def test_pdf_extract_uses_bounded_clipped_bands():
    class Rect:
        def __init__(self, x0, y0, x1, y1):
            self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
            self.height = y1 - y0

    class Page:
        rect = Rect(0, 0, 100, 1_000_000_000)

        def __init__(self):
            self.calls = []

        def get_text(self, mode, *, clip):
            self.calls.append((mode, clip))
            return ""

    page = Page()
    fake_fitz = types.SimpleNamespace(Rect=Rect)
    text, truncated = _extract_page_prefix(page, fake_fitz, 1_000)

    assert text == ""
    assert truncated is True
    assert len(page.calls) == 256


def test_read_stops_seeking_when_offset_exceeds_scan_budget(tmp_path):
    path = tmp_path / "many-lines.txt"
    path.write_text("row\n" * 100, encoding="utf-8")

    result = _read(
        str(path), offset=50, limit=1,
        max_bytes=1_000, scan_max_bytes=16,
    )

    assert "stopped after scanning 16 bytes" in result


def test_read_keeps_legacy_cr_newline_offsets(tmp_path):
    path = tmp_path / "classic-mac.txt"
    path.write_bytes(b"first\rsecond\rthird\r")

    result = _read(str(path), offset=1, limit=1, max_bytes=1_000)

    assert "     2\tsecond\r" in result


def test_read_caps_rendered_output_for_many_short_lines(tmp_path):
    path = tmp_path / "empty-lines.txt"
    path.write_bytes(b"\n" * 10_000)

    result = _read(str(path), max_bytes=10_000, max_output_chars=200)

    assert "Read output capped at 200 characters" in result
    assert len(result) < 400


def test_read_cache_cannot_bypass_allowed_root(tmp_path):
    from cheetahclaws.tool_registry import clear_tool_cache
    from cheetahclaws.tools import execute_tool

    path = tmp_path / "visible.txt"
    path.write_text("private content\n", encoding="utf-8")
    clear_tool_cache()
    first = execute_tool(
        "Read", {"file_path": str(path)}, permission_mode="accept-all",
        config={"allowed_root": str(tmp_path), "_session_id": "root-change"},
    )
    second = execute_tool(
        "Read", {"file_path": str(path)}, permission_mode="accept-all",
        config={"allowed_root": str(tmp_path / "other"), "_session_id": "root-change"},
    )

    assert "private content" in first
    assert "Error:" in second
    assert "private content" not in second
