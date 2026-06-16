"""Tests for commands/daemon_cmd.py — `cheetahclaws daemon ...` subcommands.

End-to-end behavior against a live daemon is covered in
``tests/e2e_daemon_skeleton.py`` (F-1 task #10).  This file exercises the
small behaviours that don't need a real daemon: dispatch routing, the
"not running" branches, format helpers.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws.commands import daemon_cmd


# ── dispatch ───────────────────────────────────────────────────────────────

def test_dispatch_empty_returns_usage(capsys):
    rc = daemon_cmd.dispatch([])
    captured = capsys.readouterr()
    assert rc != 0
    assert "usage" in captured.err.lower()


def test_dispatch_unknown_action(capsys):
    rc = daemon_cmd.dispatch(["banana"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "unknown" in captured.err.lower()


# ── status when no daemon is running ───────────────────────────────────────

def test_status_when_not_running(monkeypatch, capsys, tmp_path):
    # Point discovery at an empty tmp file so locate() returns None.
    p = tmp_path / "daemon.json"
    monkeypatch.setattr(daemon_cmd._discovery, "get_default_path", lambda: p)
    rc = daemon_cmd.dispatch(["status"])
    assert rc == 1
    assert "not running" in capsys.readouterr().err.lower()


# ── stop when no daemon is running (idempotent) ────────────────────────────

def test_stop_when_not_running_returns_zero(monkeypatch, tmp_path, capsys):
    p = tmp_path / "daemon.json"
    monkeypatch.setattr(daemon_cmd._discovery, "get_default_path", lambda: p)
    rc = daemon_cmd.dispatch(["stop"])
    assert rc == 0  # already in desired state
    assert "not running" in capsys.readouterr().err.lower()


# ── logs ───────────────────────────────────────────────────────────────────

def test_logs_when_no_log_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(daemon_cmd, "_log_path", lambda: tmp_path / "absent.log")
    rc = daemon_cmd.dispatch(["logs"])
    assert rc == 0
    assert "no log file" in capsys.readouterr().err.lower()


def test_logs_tails_existing_file(monkeypatch, tmp_path, capsys):
    log = tmp_path / "daemon.log"
    log.write_text("\n".join(f"line{i}" for i in range(100)) + "\n")
    monkeypatch.setattr(daemon_cmd, "_log_path", lambda: log)
    rc = daemon_cmd.dispatch(["logs", "-n", "10"])
    out = capsys.readouterr().out
    assert rc == 0
    printed = [l for l in out.splitlines() if l.startswith("line")]
    assert printed == [f"line{i}" for i in range(90, 100)]


def test_logs_default_tail_is_50(monkeypatch, tmp_path, capsys):
    log = tmp_path / "daemon.log"
    log.write_text("\n".join(f"line{i}" for i in range(200)) + "\n")
    monkeypatch.setattr(daemon_cmd, "_log_path", lambda: log)
    daemon_cmd.dispatch(["logs"])
    out = capsys.readouterr().out
    printed = [l for l in out.splitlines() if l.startswith("line")]
    assert len(printed) == 50
    assert printed[0] == "line150"
    assert printed[-1] == "line199"


# ── rotate-token ───────────────────────────────────────────────────────────

def test_rotate_token_writes_new_token(monkeypatch, tmp_path, capsys):
    token_path = tmp_path / "token"
    monkeypatch.setattr(daemon_cmd, "_default_token_path",
                        lambda: token_path)
    rc = daemon_cmd.dispatch(["rotate-token"])
    out = capsys.readouterr().out
    assert rc == 0
    assert token_path.exists()
    assert token_path.read_text().strip() != ""
    assert "rotated" in out.lower()


def test_rotate_token_changes_value(monkeypatch, tmp_path):
    token_path = tmp_path / "token"
    monkeypatch.setattr(daemon_cmd, "_default_token_path",
                        lambda: token_path)
    daemon_cmd.dispatch(["rotate-token"])
    first = token_path.read_text()
    daemon_cmd.dispatch(["rotate-token"])
    second = token_path.read_text()
    assert first != second


# ── format helpers ─────────────────────────────────────────────────────────

def test_format_duration_seconds():
    assert daemon_cmd._format_duration(5) == "5s"
    assert daemon_cmd._format_duration(59.9) == "59s"


def test_format_duration_minutes():
    assert daemon_cmd._format_duration(60) == "1m 0s"
    assert daemon_cmd._format_duration(125) == "2m 5s"


def test_format_duration_hours():
    assert daemon_cmd._format_duration(3700) == "1h 1m 40s"


def test_seconds_since_handles_recent_iso():
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = daemon_cmd._seconds_since(iso)
    assert out is not None
    assert 0.0 <= out < 5.0


def test_seconds_since_returns_none_for_garbage():
    assert daemon_cmd._seconds_since("not a date") is None


# ── _resolve_token_path ────────────────────────────────────────────────────

def test_resolve_token_path_falls_back_to_default_when_info_is_none():
    out = daemon_cmd._resolve_token_path(None)
    assert out == daemon_cmd._default_token_path()


def test_resolve_token_path_falls_back_when_info_lacks_field():
    out = daemon_cmd._resolve_token_path({"transport": "tcp"})
    assert out == daemon_cmd._default_token_path()


def test_resolve_token_path_uses_recorded_path_when_present(tmp_path):
    custom = tmp_path / "custom-token"
    info = {"transport": "tcp", "token_path": str(custom)}
    out = daemon_cmd._resolve_token_path(info)
    assert out == custom


def test_resolve_token_path_ignores_empty_string():
    out = daemon_cmd._resolve_token_path({"token_path": ""})
    assert out == daemon_cmd._default_token_path()
