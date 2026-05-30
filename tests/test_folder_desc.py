"""Tests for the folder_desc package."""
import json
import os
import tempfile

import pytest

from folder_desc.cache import (
    CACHE_DIR, _cache_key, _cache_path,
    get_cached_desc, set_cached_desc, clear_cache,
)
from folder_desc.describer import extract_inline_desc, describe_files_parallel
from folder_desc.tree import (
    _is_code_file, _collect_files, _build_tree_string, get_folder_description,
    SKIP_DIRS,
)
from pathlib import Path


class TestCache:
    def test_cache_key_deterministic(self):
        assert _cache_key("/a/b.py") == _cache_key("/a/b.py")

    def test_cache_key_different_paths(self):
        assert _cache_key("/a/b.py") != _cache_key("/a/c.py")

    def test_set_and_get(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("print('hello')")
        import folder_desc.cache as mod
        old_dir = mod.CACHE_DIR
        mod.CACHE_DIR = tmp_path / "cache"
        try:
            set_cached_desc(str(f), "prints hello")
            assert get_cached_desc(str(f)) == "prints hello"
        finally:
            mod.CACHE_DIR = old_dir

    def test_cache_invalidates_on_change(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("v1")
        import folder_desc.cache as mod
        old_dir = mod.CACHE_DIR
        mod.CACHE_DIR = tmp_path / "cache"
        try:
            set_cached_desc(str(f), "version 1")
            f.write_text("v2")
            assert get_cached_desc(str(f)) is None
        finally:
            mod.CACHE_DIR = old_dir

    def test_get_nonexistent(self):
        assert get_cached_desc("/nonexistent/file.py") is None

    def test_clear_cache(self, tmp_path):
        import folder_desc.cache as mod
        old_dir = mod.CACHE_DIR
        mod.CACHE_DIR = tmp_path / "cache"
        try:
            (tmp_path / "cache").mkdir()
            (tmp_path / "cache" / "a.json").write_text("{}")
            (tmp_path / "cache" / "b.json").write_text("{}")
            assert clear_cache() == 2
        finally:
            mod.CACHE_DIR = old_dir


class TestDescriber:
    def test_extract_inline_desc(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("# [desc] Handles user authentication [/desc]\nimport os\n")
        assert extract_inline_desc(str(f)) == "Handles user authentication"

    def test_extract_inline_desc_missing(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("import os\n")
        assert extract_inline_desc(str(f)) is None

    def test_extract_inline_desc_nonexistent(self):
        assert extract_inline_desc("/nonexistent.py") is None

    def test_describe_files_parallel_inline(self, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_text("# [desc] Module A [/desc]\n")
        f2 = tmp_path / "b.py"
        f2.write_text("# [desc] Module B [/desc]\n")
        results = describe_files_parallel([str(f1), str(f2)])
        assert results[str(f1)] == "Module A"
        assert results[str(f2)] == "Module B"


class TestTree:
    def test_is_code_file(self):
        assert _is_code_file(Path("foo.py"))
        assert _is_code_file(Path("Makefile"))
        assert not _is_code_file(Path("image.png"))
        assert not _is_code_file(Path("data.bin"))

    def test_collect_files_skips_dirs(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.pyc").write_text("")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# [desc] Main entry [/desc]\n")
        (tmp_path / "readme.md").write_text("# Readme")
        files = _collect_files(tmp_path)
        names = [f.name for f in files]
        assert "main.py" in names
        assert "readme.md" in names
        assert "mod.pyc" not in names

    def test_build_tree_string(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        descs = {str(tmp_path / "a.py"): "Module A"}
        tree = _build_tree_string(tmp_path, descs)
        assert "a.py" in tree
        assert "[desc] Module A [/desc]" in tree

    def test_get_folder_description_not_dir(self):
        result = get_folder_description("/nonexistent/path")
        assert "Error" in result or "not a directory" in result

    def test_get_folder_description_with_inline(self, tmp_path):
        (tmp_path / "main.py").write_text("# [desc] Entry point [/desc]\nprint('hi')\n")
        (tmp_path / "utils.py").write_text("# [desc] Utility helpers [/desc]\n")
        result = get_folder_description(str(tmp_path))
        assert "2 code files found" in result
        assert "Entry point" in result
        assert "Utility helpers" in result

    def test_get_folder_description_empty(self, tmp_path):
        result = get_folder_description(str(tmp_path))
        assert "empty" in result.lower() or "no code files" in result.lower()
