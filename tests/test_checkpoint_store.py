"""Integration tests for checkpoint store: stderr capture + large file skip."""
from __future__ import annotations

import pytest

import checkpoint.store as store


@pytest.fixture(autouse=True)
def isolate_store(tmp_path, monkeypatch):
    """Redirect checkpoint root to tmp_path and reset global state."""
    monkeypatch.setattr(store, "_checkpoints_root", lambda: tmp_path / "checkpoints")
    store.reset_file_versions()


def test_large_file_skipped_and_logged_to_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "_MAX_FILE_SIZE", 50)
    big_file = tmp_path / "big.txt"
    big_file.write_bytes(b"x" * 100)

    result = store.track_file_edit("test-session", str(big_file))

    assert result is None
    captured = capsys.readouterr()
    assert "[checkpoint] skipping large file" in captured.err
    assert "100 bytes" in captured.err
    assert captured.out == ""


def test_normal_file_backed_up(tmp_path, capsys):
    small_file = tmp_path / "small.txt"
    content = b"hello world"
    small_file.write_bytes(content)

    result = store.track_file_edit("test-session", str(small_file))

    assert result is not None
    backup_dir = tmp_path / "checkpoints" / "test-session" / "backups"
    backup_path = backup_dir / result
    assert backup_path.exists()
    assert backup_path.read_bytes() == content
    captured = capsys.readouterr()
    assert captured.err == ""


def test_backup_failure_logged_to_stderr(tmp_path, monkeypatch, capsys):
    normal_file = tmp_path / "normal.txt"
    normal_file.write_bytes(b"some data")

    def failing_copy(*args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(store.shutil, "copy2", failing_copy)

    result = store.track_file_edit("test-session", str(normal_file))

    assert result is None
    captured = capsys.readouterr()
    assert "[checkpoint] backup failed" in captured.err
    assert "access denied" in captured.err
    assert captured.out == ""
