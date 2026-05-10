"""Tests for agent output-path resolution.

Behavior under test: when a user launches an autonomous agent via the /agent
wizard and supplies a *relative* output filename (e.g. `research_notes.md`),
the path is rewritten to live under `~/.cheetahclaws/agents/<name>/output/`
instead of the cheetahclaws CWD. Absolute paths pass through unchanged.

Also validates AgentRunner exposes `output_dir` and creates it on init.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent_runner
from commands.agent_cmd import _resolve_output_path


# ── _resolve_output_path ─────────────────────────────────────────────────

class TestResolveOutputPath:
    def test_relative_filename_lands_under_dot_cheetahclaws(self, tmp_path, monkeypatch):
        # Force HOME to a tmp dir so the test doesn't pollute the real
        # ~/.cheetahclaws/agents/.
        monkeypatch.setenv("HOME", str(tmp_path))
        p = _resolve_output_path("research_notes.md", "research")
        assert p == tmp_path / ".cheetahclaws" / "agents" / "research" / "output" / "research_notes.md"

    def test_absolute_path_pass_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        abs_target = tmp_path / "elsewhere" / "out.md"
        p = _resolve_output_path(str(abs_target), "research")
        assert p == abs_target

    def test_tilde_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        p = _resolve_output_path("~/notes.md", "research")
        # ~/notes.md is technically absolute after expansion, so it lands
        # at HOME/notes.md (NOT under ~/.cheetahclaws/agents/.../output).
        assert p == tmp_path / "notes.md"

    def test_subdirectory_relative_path(self, tmp_path, monkeypatch):
        # Relative paths with subdirs must still be rewritten to live under
        # the agent output dir; subdir is preserved.
        monkeypatch.setenv("HOME", str(tmp_path))
        p = _resolve_output_path("subdir/notes.md", "research")
        assert p == (
            tmp_path / ".cheetahclaws" / "agents" / "research" / "output" /
            "subdir" / "notes.md"
        )

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        p = _resolve_output_path("nested/deep/notes.md", "research")
        # Parent dir must exist after resolution so the model's first Write
        # call succeeds without a separate mkdir step.
        assert p.parent.exists()
        assert p.parent.is_dir()

    def test_agent_name_isolates_outputs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        p1 = _resolve_output_path("notes.md", "research")
        p2 = _resolve_output_path("notes.md", "paper")
        assert p1 != p2
        assert "research" in str(p1)
        assert "paper" in str(p2)


# ── AgentRunner.output_dir ───────────────────────────────────────────────

class TestRunnerOutputDir:
    def test_output_dir_is_under_log_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            agent_runner, "_LOG_DIR", tmp_path / "agents"
        )
        runner = agent_runner.AgentRunner(
            name="test-out-dir",
            template_content="(test)",
            template_path="/tmp/dummy.md",
            args="",
            config={"model": "test"},
            send_fn=None,
            interval=0.0,
            auto_approve=True,
        )
        assert runner.output_dir == tmp_path / "agents" / "test-out-dir" / "output"

    def test_output_dir_created_on_init(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            agent_runner, "_LOG_DIR", tmp_path / "agents"
        )
        runner = agent_runner.AgentRunner(
            name="test-mkdir",
            template_content="(test)",
            template_path="/tmp/dummy.md",
            args="",
            config={"model": "test"},
            send_fn=None,
            interval=0.0,
            auto_approve=True,
        )
        # Eagerly created so model's first Write succeeds without mkdir
        assert runner.output_dir.exists()
        assert runner.output_dir.is_dir()
