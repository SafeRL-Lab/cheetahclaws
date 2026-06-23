"""Tests for daemon/discovery.py — daemon.json read/write/locate."""
from __future__ import annotations

import json
import os
import sys
import stat
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws.daemon import discovery


# ── make_info ──────────────────────────────────────────────────────────────

def test_make_info_has_required_fields():
    info = discovery.make_info(pid=1234, transport="unix",
                                address="/tmp/x.sock", version="3.5.72")
    for key in ("pid", "started_at", "transport", "address", "version", "schema"):
        assert key in info
    assert info["pid"] == 1234
    assert info["transport"] == "unix"
    assert info["address"] == "/tmp/x.sock"
    assert info["version"] == "3.5.72"
    assert info["schema"] == discovery.SCHEMA_VERSION


def test_make_info_started_at_is_iso_utc():
    info = discovery.make_info(pid=1, transport="tcp",
                                address="127.0.0.1:8765", version="x")
    # ISO 8601 UTC with Z suffix: "YYYY-MM-DDTHH:MM:SSZ"
    assert info["started_at"].endswith("Z")
    assert "T" in info["started_at"]


def test_make_info_omits_token_path_by_default():
    info = discovery.make_info(pid=1, transport="tcp",
                                address="127.0.0.1:8765", version="x")
    assert "token_path" not in info


def test_make_info_records_token_path_when_overridden():
    info = discovery.make_info(pid=1, transport="tcp",
                                address="127.0.0.1:8765", version="x",
                                token_path="/tmp/custom-token")
    assert info["token_path"] == "/tmp/custom-token"
    # Schema does not bump — token_path is a strictly additive optional field.
    assert info["schema"] == discovery.SCHEMA_VERSION


# ── write / read ────────────────────────────────────────────────────────────

def test_write_then_read_roundtrip(tmp_path: Path):
    p = tmp_path / "daemon.json"
    info = discovery.make_info(pid=1234, transport="unix",
                                address="/x.sock", version="v")
    discovery.write(info, path=p)
    assert p.exists()
    got = discovery.read(path=p)
    assert got == info


def test_read_missing_returns_none(tmp_path: Path):
    p = tmp_path / "absent.json"
    assert discovery.read(path=p) is None


def test_read_corrupt_returns_none(tmp_path: Path):
    p = tmp_path / "daemon.json"
    p.write_text("{ this is not valid json")
    assert discovery.read(path=p) is None


def test_write_atomic_does_not_leave_partial_on_error(tmp_path: Path,
                                                      monkeypatch):
    p = tmp_path / "daemon.json"
    p.write_text(json.dumps({"old": True}))

    # Force os.replace to fail; the original file must remain intact.
    real_replace = os.replace

    def boom(_src, _dst):
        raise OSError("simulated failure")

    monkeypatch.setattr(os, "replace", boom)
    info = discovery.make_info(pid=1, transport="tcp",
                                address="127.0.0.1:1", version="v")
    with pytest.raises(OSError):
        discovery.write(info, path=p)

    # Original survives, no .tmp leakage in stable name.
    monkeypatch.setattr(os, "replace", real_replace)
    assert json.loads(p.read_text()) == {"old": True}


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_write_sets_mode_0600_on_posix(tmp_path: Path):
    p = tmp_path / "daemon.json"
    info = discovery.make_info(pid=1, transport="unix", address="/x", version="v")
    discovery.write(info, path=p)
    mode = stat.S_IMODE(os.stat(p).st_mode)
    assert mode == 0o600


# ── clear ──────────────────────────────────────────────────────────────────

def test_clear_removes_file(tmp_path: Path):
    p = tmp_path / "daemon.json"
    p.write_text("{}")
    discovery.clear(path=p)
    assert not p.exists()


def test_clear_idempotent_when_missing(tmp_path: Path):
    p = tmp_path / "absent.json"
    discovery.clear(path=p)  # must not raise


# ── pid_alive ──────────────────────────────────────────────────────────────

def test_pid_alive_self_returns_true():
    assert discovery.pid_alive(os.getpid()) is True


def test_pid_alive_zero_or_negative_returns_false():
    assert discovery.pid_alive(0) is False
    assert discovery.pid_alive(-1) is False


def test_pid_alive_unlikely_pid_returns_false():
    # Pick a pid extremely unlikely to be running.  Loop down from 2**31
    # in case any candidate happens to exist.
    for candidate in (2**31 - 1, 2**31 - 2, 2**31 - 3):
        if not discovery.pid_alive(candidate):
            return
    pytest.skip("no obviously-dead pid available on this system")


# ── locate ─────────────────────────────────────────────────────────────────

def test_locate_returns_none_when_no_file(tmp_path: Path):
    p = tmp_path / "daemon.json"
    assert discovery.locate(path=p) is None


def test_locate_returns_info_when_pid_alive(tmp_path: Path):
    p = tmp_path / "daemon.json"
    info = discovery.make_info(pid=os.getpid(), transport="unix",
                                address="/x", version="v")
    discovery.write(info, path=p)
    got = discovery.locate(path=p)
    assert got is not None
    assert got["pid"] == os.getpid()


def test_locate_clears_stale_file_when_pid_dead(tmp_path: Path, monkeypatch):
    p = tmp_path / "daemon.json"
    info = discovery.make_info(pid=999999999, transport="tcp",
                                address="127.0.0.1:1", version="v")
    discovery.write(info, path=p)
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: False)

    assert discovery.locate(path=p) is None
    assert not p.exists()  # stale file auto-cleared


# ── default path ───────────────────────────────────────────────────────────

def test_get_default_path_lives_under_config_dir():
    from cheetahclaws.config import CONFIG_DIR
    p = discovery.get_default_path()
    assert p == CONFIG_DIR / "daemon.json"
