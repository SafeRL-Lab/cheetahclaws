"""Tests for nested skill directory layout (skill-name/skill.md)."""
from __future__ import annotations

import pytest

from cheetahclaws.skill.loader import _iter_skill_files


class TestIterSkillFiles:
    def test_flat_md_files(self, tmp_path):
        (tmp_path / "alpha.md").write_text("# Alpha")
        (tmp_path / "beta.md").write_text("# Beta")
        result = _iter_skill_files(tmp_path)
        names = [p.name for p in result]
        assert names == ["alpha.md", "beta.md"]

    def test_nested_skill_md(self, tmp_path):
        d = tmp_path / "my-skill"
        d.mkdir()
        (d / "skill.md").write_text("# My Skill")
        result = list(_iter_skill_files(tmp_path))
        assert len(result) == 1
        assert result[0].name == "skill.md"

    def test_nested_SKILL_uppercase(self, tmp_path):
        import platform
        d = tmp_path / "my-skill"
        d.mkdir()
        (d / "SKILL.md").write_text("# My Skill")
        result = list(_iter_skill_files(tmp_path))
        assert len(result) == 1
        if platform.system() == "Windows":
            assert result[0].name.lower() == "skill.md"
        else:
            assert result[0].name == "SKILL.md"

    def test_mixed_flat_and_nested(self, tmp_path):
        (tmp_path / "flat.md").write_text("# Flat")
        d = tmp_path / "nested"
        d.mkdir()
        (d / "skill.md").write_text("# Nested")
        result = list(_iter_skill_files(tmp_path))
        assert len(result) == 2

    def test_empty_directory(self, tmp_path):
        assert list(_iter_skill_files(tmp_path)) == []

    def test_subdir_without_skill_md_ignored(self, tmp_path):
        d = tmp_path / "not-a-skill"
        d.mkdir()
        (d / "readme.md").write_text("# Just a readme")
        result = list(_iter_skill_files(tmp_path))
        assert result == []

    def test_skill_md_preferred_over_SKILL(self, tmp_path):
        d = tmp_path / "both"
        d.mkdir()
        (d / "skill.md").write_text("# lower")
        (d / "SKILL.md").write_text("# UPPER")
        result = list(_iter_skill_files(tmp_path))
        assert len(result) == 1
        assert result[0].name == "skill.md"
