"""F-2 tests for jobs.py — SQLite backing + JSON-file migration."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jobs
from cc_daemon import schema


@pytest.fixture(autouse=True)
def _isolated_db_and_jobs_path(tmp_path: Path, monkeypatch):
    """Each test gets a private sessions.db AND its own jobs.json path
    so the legacy migration logic doesn't touch the real user file."""
    monkeypatch.setattr(jobs, "_JOBS_PATH", tmp_path / "jobs.json")
    schema.set_db_path(tmp_path / "sessions.db")
    schema._local.conn = None
    # Reset migration sentinel so each test re-runs migration
    jobs._migration_done_in_process = False
    yield
    schema._local.conn = None
    schema.set_db_path(schema.get_default_db_path())


# ── Roundtrip ──────────────────────────────────────────────────────────────

def test_create_persists_to_sqlite():
    j = jobs.create("hello world", source="test")
    conn = schema.get_conn()
    row = conn.execute(
        "SELECT id, prompt, status FROM jobs WHERE id=?", (j.id,)
    ).fetchone()
    assert row is not None
    assert row["prompt"] == "hello world"
    assert row["status"] == "queued"


def test_get_returns_persisted_job():
    j = jobs.create("foo", source="test")
    fetched = jobs.get(j.id)
    assert fetched is not None
    assert fetched.id == j.id
    assert fetched.prompt == "foo"


def test_lifecycle_updates_status_in_sqlite():
    j = jobs.create("lifecycle", source="test")
    jobs.start(j.id)
    assert jobs.get(j.id).status == "running"
    jobs.complete(j.id, result_preview="done")
    assert jobs.get(j.id).status == "done"


def test_steps_are_round_tripped_as_json():
    j = jobs.create("with steps", source="test")
    jobs.add_step(j.id, "Bash", "ls -la")
    jobs.finish_step(j.id, "Bash", "5 files")
    fetched = jobs.get(j.id)
    assert any(s.get("name") == "Bash" for s in fetched.steps)


def test_list_recent_orders_newest_first():
    j1 = jobs.create("one", source="t")
    j2 = jobs.create("two", source="t")
    j3 = jobs.create("three", source="t")
    recent = jobs.list_recent(10)
    assert [r.id for r in recent[:3]] == [j3.id, j2.id, j1.id]


def test_list_running_only_returns_running():
    j1 = jobs.create("a", source="t"); jobs.start(j1.id)
    j2 = jobs.create("b", source="t")  # stays queued
    j3 = jobs.create("c", source="t"); jobs.start(j3.id); jobs.complete(j3.id)
    running = jobs.list_running()
    assert {r.id for r in running} == {j1.id}


def test_max_jobs_pruning_keeps_recent():
    """The legacy JSON storage capped at 100; SQLite mirror should too."""
    original = jobs._MAX_JOBS
    try:
        jobs._MAX_JOBS = 5
        for i in range(10):
            jobs.create(f"job-{i}", source="t")
        recent = jobs.list_recent(20)
        assert len(recent) == 5
        # Newest 5 survived
        assert [r.prompt for r in recent] == [f"job-{i}" for i in (9, 8, 7, 6, 5)]
    finally:
        jobs._MAX_JOBS = original


# ── JSON migration ────────────────────────────────────────────────────────

def test_migration_imports_legacy_json_jobs(tmp_path):
    legacy = [
        {"id": "abc123", "title": "old job", "prompt": "old prompt",
         "status": "done", "source": "telegram", "steps": [],
         "step_count": 0, "current_step": "", "result": "old result",
         "error": "", "created_at": "2026-04-01T10:00:00",
         "started_at": "2026-04-01T10:00:01",
         "done_at": "2026-04-01T10:00:05",
         "duration_s": 4.0, "retry_of": ""},
        {"id": "def456", "title": "another", "prompt": "another prompt",
         "status": "failed", "source": "console", "steps": [],
         "step_count": 0, "current_step": "", "result": "",
         "error": "boom", "created_at": "2026-04-02T11:00:00",
         "started_at": "", "done_at": "2026-04-02T11:00:01",
         "duration_s": 1.0, "retry_of": ""},
    ]
    jobs._JOBS_PATH.write_text(json.dumps(legacy), encoding="utf-8")

    # First call triggers migration.
    out = jobs.list_recent(10)
    ids = {j.id for j in out}
    assert {"abc123", "def456"} <= ids


def test_migration_is_idempotent_across_calls():
    legacy = [{"id": "x", "title": "t", "prompt": "p", "status": "done",
               "source": "console", "steps": [], "step_count": 0,
               "current_step": "", "result": "", "error": "",
               "created_at": "2026-04-01", "started_at": "",
               "done_at": "", "duration_s": 0.0, "retry_of": ""}]
    jobs._JOBS_PATH.write_text(json.dumps(legacy), encoding="utf-8")

    jobs.list_recent(10)
    jobs.list_recent(10)
    jobs.list_recent(10)

    conn = schema.get_conn()
    count = conn.execute("SELECT COUNT(*) FROM jobs WHERE id='x'").fetchone()[0]
    assert count == 1


def test_migration_marks_schema_meta_after_run():
    jobs._JOBS_PATH.write_text("[]", encoding="utf-8")
    jobs.list_recent(10)
    conn = schema.get_conn()
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='jobs_migrated_from_json'"
    ).fetchone()
    assert row is not None
    assert row[0] == "1"


def test_migration_skips_when_no_legacy_file():
    # Without a JSON file the migration mark is still set so we never
    # re-scan on every call.
    assert not jobs._JOBS_PATH.exists()
    jobs.list_recent(10)
    conn = schema.get_conn()
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='jobs_migrated_from_json'"
    ).fetchone()
    assert row is not None


def test_migration_keeps_legacy_json_in_place():
    """One-release fallback: the JSON file is left readable post-migration."""
    legacy = [{"id": "x", "title": "t", "prompt": "p", "status": "done",
               "source": "c", "steps": [], "step_count": 0,
               "current_step": "", "result": "", "error": "",
               "created_at": "2026-04-01", "started_at": "",
               "done_at": "", "duration_s": 0.0, "retry_of": ""}]
    jobs._JOBS_PATH.write_text(json.dumps(legacy), encoding="utf-8")
    jobs.list_recent(10)
    assert jobs._JOBS_PATH.exists()  # not deleted


def test_migration_tolerates_corrupt_json():
    jobs._JOBS_PATH.write_text("{ this is not json", encoding="utf-8")
    # Should not raise — corrupt file is treated as empty.
    out = jobs.list_recent(10)
    assert out == []


def test_new_jobs_after_migration_persist_to_sqlite():
    legacy = [{"id": "old", "title": "t", "prompt": "p", "status": "done",
               "source": "c", "steps": [], "step_count": 0,
               "current_step": "", "result": "", "error": "",
               "created_at": "2026-04-01", "started_at": "",
               "done_at": "", "duration_s": 0.0, "retry_of": ""}]
    jobs._JOBS_PATH.write_text(json.dumps(legacy), encoding="utf-8")
    new_job = jobs.create("brand new", source="t")
    fetched = jobs.get(new_job.id)
    assert fetched is not None
    assert fetched.prompt == "brand new"
