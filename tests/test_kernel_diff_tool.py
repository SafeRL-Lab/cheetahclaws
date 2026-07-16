"""Tests for the Diff built-in (RFC 0030)."""
from __future__ import annotations

import pytest

from cheetahclaws.kernel.tools.diff_tool import (
    DEFAULT_DIFF_CAP_BYTES,
    DIFF_TOOL,
    diff_handler,
)
from cheetahclaws.kernel.tools.registry import (
    ToolContext,
    ToolFsDenied,
    ToolInvalidArgs,
    ToolRegistry,
)


# ── Stub kernels ──────────────────────────────────────────────────


class _AllowAll:
    class _Cap:
        def check_fs(self, pid, path, mode): return True
        def check_tool(self, pid, t):        return True
    cap = _Cap()


class _DenyFs:
    class _Cap:
        def check_fs(self, pid, path, mode): return False
        def check_tool(self, pid, t):        return True
    cap = _Cap()


# ── Args validation ───────────────────────────────────────────────


def test_diff_neither_args():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        diff_handler({}, ctx)


def test_diff_mixed_args():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        diff_handler({"path_a": "/x", "text_a": "y"}, ctx)


def test_diff_context_lines_out_of_range():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        diff_handler({"text_a": "a", "text_b": "b",
                       "context_lines": 99}, ctx)


def test_diff_context_lines_non_int():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        diff_handler({"text_a": "a", "text_b": "b",
                       "context_lines": "3"}, ctx)


# ── Text mode ────────────────────────────────────────────────────


def test_text_identical():
    ctx = ToolContext(pid=1, kernel=None)
    r = diff_handler({"text_a": "x\n", "text_b": "x\n"}, ctx)
    assert r["identical"] is True
    assert r["diff"] == ""
    assert r["truncated"] is False
    assert r["lines_a"] == r["lines_b"] == 1


def test_text_one_line_change():
    ctx = ToolContext(pid=1, kernel=None)
    r = diff_handler({"text_a": "a\nb\nc\n",
                       "text_b": "a\nB\nc\n"}, ctx)
    assert r["identical"] is False
    assert "-b" in r["diff"]
    assert "+B" in r["diff"]
    assert r["label_a"] == "a"
    assert r["label_b"] == "b"


def test_text_label_override():
    ctx = ToolContext(pid=1, kernel=None)
    r = diff_handler({
        "text_a": "x\n", "text_b": "y\n",
        "label_a": "before.txt", "label_b": "after.txt",
    }, ctx)
    assert "before.txt" in r["diff"]
    assert "after.txt" in r["diff"]


def test_text_label_with_newline_rejected():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        diff_handler({"text_a": "x", "text_b": "y",
                       "label_a": "bad\nlabel"}, ctx)


def test_text_a_must_be_string():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        diff_handler({"text_a": 1, "text_b": "x"}, ctx)


def test_text_context_lines_zero():
    ctx = ToolContext(pid=1, kernel=None)
    r = diff_handler({"text_a": "a\nb\nc\n",
                       "text_b": "a\nB\nc\n",
                       "context_lines": 0}, ctx)
    assert r["context_lines"] == 0
    # n=0 → no surrounding context lines.
    assert "a" not in r["diff"].split("\n")[3:]   # rough check


# ── Path mode ────────────────────────────────────────────────────


def test_path_identical(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hi\n")
    f2.write_text("hi\n")
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = diff_handler({"path_a": str(f1), "path_b": str(f2)}, ctx)
    assert r["identical"] is True
    assert r["diff"] == ""


def test_path_different(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("apple\nbanana\ncherry\n")
    f2.write_text("apple\nBanana\ncherry\n")
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = diff_handler({"path_a": str(f1), "path_b": str(f2)}, ctx)
    assert r["identical"] is False
    assert "-banana" in r["diff"]
    assert "+Banana" in r["diff"]
    assert r["label_a"] == str(f1)
    assert r["label_b"] == str(f2)


def test_path_fs_denied(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("x")
    f2.write_text("y")
    ctx = ToolContext(pid=1, kernel=_DenyFs())
    with pytest.raises(ToolFsDenied):
        diff_handler({"path_a": str(f1), "path_b": str(f2)}, ctx)


def test_path_directory_rejected(tmp_path):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    with pytest.raises(Exception):    # ToolFailed
        diff_handler({"path_a": str(tmp_path),
                       "path_b": str(tmp_path)}, ctx)


def test_path_not_found(tmp_path):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    with pytest.raises(Exception):
        diff_handler({"path_a": str(tmp_path / "nope.txt"),
                       "path_b": str(tmp_path / "nope2.txt")}, ctx)


# ── Truncation ───────────────────────────────────────────────────


def test_truncation(monkeypatch):
    """Force a tiny cap to verify truncation marker."""
    ctx = ToolContext(pid=1, kernel=None)
    monkeypatch.setattr(
        "cheetahclaws.kernel.tools.diff_tool.DEFAULT_DIFF_CAP_BYTES",
        200,
    )
    big_a = "\n".join(f"line{i}" for i in range(500)) + "\n"
    big_b = "\n".join(f"LINE{i}" for i in range(500)) + "\n"
    r = diff_handler({"text_a": big_a, "text_b": big_b}, ctx)
    assert r["truncated"] is True
    assert "[diff truncated at 200 bytes]" in r["diff"]


# ── Registry shape ───────────────────────────────────────────────


def test_diff_tool_registered_by_register_builtin_tools():
    from cheetahclaws.kernel.tools.builtin import register_builtin_tools
    reg = ToolRegistry()
    names = register_builtin_tools(reg)
    assert "Diff" in names
    assert reg.has("Diff")


def test_diff_tool_metadata():
    assert DIFF_TOOL.name == "Diff"
    assert DIFF_TOOL.requires_capability is True
    # Path mode does its own fs check; declare requires_fs empty.
    assert DIFF_TOOL.requires_fs == ()
