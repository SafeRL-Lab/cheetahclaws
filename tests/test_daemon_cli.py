"""Tests for daemon/cli.py — top-level dispatch behaviour.

The actual `serve` loop is exercised by tests/e2e_daemon_skeleton.py;
this file covers the small dispatch branches.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws.daemon import cli


# ── --help / -h ─────────────────────────────────────────────────────────────

def test_main_help_long_form_exits_zero(capsys):
    rc = cli.main(["--help"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "serve" in captured.out
    assert "status" in captured.out


def test_main_help_short_form_exits_zero(capsys):
    rc = cli.main(["-h"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "usage" in captured.out.lower()


def test_main_no_args_prints_usage_to_stderr(capsys):
    rc = cli.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "usage" in captured.err.lower()


def test_main_unknown_subcommand_includes_usage(capsys):
    rc = cli.main(["banana"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "unknown subcommand" in captured.err.lower()
    # Usage banner follows the error so users see how to recover.
    assert "serve" in captured.err
