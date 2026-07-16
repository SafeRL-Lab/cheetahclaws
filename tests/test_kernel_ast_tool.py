"""Tests for the AST built-in (RFC 0031)."""
from __future__ import annotations

import pytest

from cheetahclaws.kernel.tools.ast_tool import (
    ALLOWED_KINDS,
    AST_TOOL,
    ast_handler,
)
from cheetahclaws.kernel.tools.registry import (
    ToolContext,
    ToolFailed,
    ToolFsDenied,
    ToolInvalidArgs,
    ToolRegistry,
)


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


def test_neither_provided():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        ast_handler({}, ctx)


def test_both_provided():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        ast_handler({"path": "/x", "text": "y"}, ctx)


def test_include_unknown_kind():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        ast_handler({"text": "x = 1\n",
                      "include": ["function", "BOGUS"]}, ctx)


def test_max_depth_too_deep():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        ast_handler({"text": "x = 1\n", "max_depth": 999}, ctx)


def test_path_must_be_py(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("def foo(): pass\n")
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    with pytest.raises(ToolInvalidArgs):
        ast_handler({"path": str(f)}, ctx)


# ── Text mode ─────────────────────────────────────────────────────


def test_text_function_top_level():
    ctx = ToolContext(pid=1, kernel=None)
    r = ast_handler({"text": "def foo(x, y):\n    return x + y\n"}, ctx)
    assert r["syntax_error"] is None
    fns = [n for n in r["nodes"] if n["kind"] == "function"]
    assert len(fns) == 1
    assert fns[0]["name"] == "foo"
    assert fns[0]["args"] == ["x", "y"]
    assert fns[0]["scope"] == []


def test_text_class_with_method():
    src = (
        "class Bar(Base):\n"
        "    def m(self, n):\n"
        "        return n\n"
    )
    ctx = ToolContext(pid=1, kernel=None)
    r = ast_handler({"text": src}, ctx)
    classes = [n for n in r["nodes"] if n["kind"] == "class"]
    assert classes[0]["name"] == "Bar"
    assert classes[0]["bases"] == ["Base"]
    methods = [n for n in r["nodes"]
               if n["kind"] == "function" and n["scope"] == ["Bar"]]
    assert methods[0]["name"] == "m"
    assert methods[0]["args"] == ["self", "n"]


def test_text_imports():
    src = (
        "import os\n"
        "import sys, json\n"
        "from os.path import join, dirname\n"
        "from . import sibling\n"
    )
    ctx = ToolContext(pid=1, kernel=None)
    r = ast_handler({"text": src}, ctx)
    imps = [n for n in r["nodes"] if n["kind"] == "import"]
    assert imps[0]["names"] == ["os"]
    assert imps[1]["names"] == ["sys", "json"]
    fims = [n for n in r["nodes"] if n["kind"] == "import_from"]
    assert fims[0]["module"] == "os.path"
    assert fims[0]["names"] == ["join", "dirname"]
    assert fims[1]["level"] == 1
    assert fims[1]["module"] == ""


def test_text_async_function():
    src = "async def foo():\n    pass\n"
    ctx = ToolContext(pid=1, kernel=None)
    r = ast_handler({
        "text": src,
        "include": ["async_function"],
    }, ctx)
    assert len(r["nodes"]) == 1
    assert r["nodes"][0]["kind"] == "async_function"


def test_text_assign_and_annotation():
    src = "X: int = 1\nY = 2\n"
    ctx = ToolContext(pid=1, kernel=None)
    r = ast_handler({
        "text": src,
        "include": ["assign", "annotation"],
    }, ctx)
    kinds = [n["kind"] for n in r["nodes"]]
    assert "annotation" in kinds
    assert "assign" in kinds


def test_text_include_filter():
    src = "def f(): pass\nimport os\n"
    ctx = ToolContext(pid=1, kernel=None)
    r = ast_handler({"text": src, "include": ["function"]}, ctx)
    kinds = [n["kind"] for n in r["nodes"]]
    assert kinds == ["function"]


def test_text_decorators():
    src = (
        "@my.dec\n"
        "@other\n"
        "def f():\n"
        "    pass\n"
    )
    ctx = ToolContext(pid=1, kernel=None)
    r = ast_handler({"text": src}, ctx)
    fn = r["nodes"][0]
    assert "my.dec" in fn["decorators"]
    assert "other" in fn["decorators"]


def test_text_syntax_error_does_not_raise():
    src = "def f(\n"     # truncated
    ctx = ToolContext(pid=1, kernel=None)
    r = ast_handler({"text": src}, ctx)
    assert r["syntax_error"] is not None
    assert r["syntax_error"]["lineno"] >= 1
    assert r["nodes"] == []
    assert r["line_count"] >= 1


def test_max_depth_caps_nesting():
    src = (
        "class A:\n"
        "    class B:\n"
        "        class C:\n"
        "            def f(self): pass\n"
    )
    ctx = ToolContext(pid=1, kernel=None)
    r = ast_handler({"text": src, "max_depth": 1}, ctx)
    # Only top-level A should be reported.
    classes = [n for n in r["nodes"] if n["kind"] == "class"]
    assert len(classes) == 1
    assert classes[0]["name"] == "A"


# ── Path mode ─────────────────────────────────────────────────────


def test_path_mode_real_file(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "import os\n"
        "def hello():\n"
        "    return 'hi'\n"
    )
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = ast_handler({"path": str(f)}, ctx)
    assert r["path"] == str(f)
    names = sorted(n["name"] for n in r["nodes"]
                    if n["kind"] == "function")
    assert names == ["hello"]
    imps = [n for n in r["nodes"] if n["kind"] == "import"]
    assert imps[0]["names"] == ["os"]


def test_path_fs_denied(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    ctx = ToolContext(pid=1, kernel=_DenyFs())
    with pytest.raises(ToolFsDenied):
        ast_handler({"path": str(f)}, ctx)


def test_path_too_large(tmp_path, monkeypatch):
    f = tmp_path / "big.py"
    f.write_text("x = 1\n")
    monkeypatch.setattr(
        "cheetahclaws.kernel.tools.ast_tool.DEFAULT_AST_MAX_FILE_BYTES", 1,
    )
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    with pytest.raises(ToolFailed):
        ast_handler({"path": str(f)}, ctx)


# ── Registry ─────────────────────────────────────────────────────


def test_ast_tool_in_register_builtin_tools():
    from cheetahclaws.kernel.tools.builtin import register_builtin_tools
    reg = ToolRegistry()
    names = register_builtin_tools(reg)
    assert "AST" in names
    assert reg.has("AST")


def test_ast_tool_metadata():
    assert AST_TOOL.name == "AST"
    assert AST_TOOL.requires_capability is True
    assert AST_TOOL.requires_fs == ()
    assert "function" in ALLOWED_KINDS
