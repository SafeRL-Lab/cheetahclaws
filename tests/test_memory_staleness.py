"""Regression tests for the "retrieval resets staleness" memory bug.

Before the fix, both the retrieval recency score and the staleness warning
were computed from a memory file's filesystem mtime, while ``touch_last_used``
rewrote the file on every MemorySearch hit — bumping that mtime. The effect:
a single *read* of a stale, never-re-verified memory reset its recency to ~1.0
and suppressed its "verify against current code" warning, exactly the
"stale-but-confident" failure the project's own design warns against.

The fix anchors staleness to a ``last_verified`` date (falling back to
``created``, then mtime for legacy files), refreshes it only via
``mark_verified`` / the MemoryVerify tool, and makes ``touch_last_used``
preserve the file mtime. These tests would FAIL against the pre-fix code and
PASS after it.
"""
import os
import time
from datetime import date, timedelta

import pytest

import cheetahclaws.memory.store as _store
from cheetahclaws.memory.store import (
    parse_frontmatter,
    touch_last_used,
    mark_verified,
)
from cheetahclaws.memory.context import find_relevant_memories
from cheetahclaws.memory.scan import (
    verified_epoch,
    trust_recency,
    memory_freshness_text,
    parse_date_epoch,
)


# ── Fixtures / helpers ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def redirect_memory_dirs(tmp_path, monkeypatch):
    """Point user + project memory dirs at a temp location for every test."""
    user_mem = tmp_path / "user_memory"
    user_mem.mkdir()
    proj_mem = tmp_path / "project_memory"
    proj_mem.mkdir()
    monkeypatch.setattr(_store, "USER_MEMORY_DIR", user_mem)
    monkeypatch.setattr(_store, "get_project_memory_dir", lambda: proj_mem)
    return user_mem, proj_mem


def _write_memory(mem_dir, slug, *, content, created, last_verified,
                  mtime_days_ago=None, confidence=1.0):
    """Create a memory .md file with explicit dates and (optionally) a
    backdated filesystem mtime. Returns the file path."""
    fp = mem_dir / f"{slug}.md"
    fm = ["---", f"name: {slug}", f"description: {slug} desc", "type: project",
          f"created: {created}"]
    if last_verified:
        fm.append(f"last_verified: {last_verified}")
    if confidence != 1.0:
        fm.append(f"confidence: {confidence:.2f}")
    fm.append("---")
    fp.write_text("\n".join(fm) + "\n" + content + "\n")
    if mtime_days_ago is not None:
        epoch = time.time() - mtime_days_ago * 86_400
        os.utime(fp, (epoch, epoch))
    return fp


def _find_one(query, name):
    """Run keyword retrieval and return the single result dict for ``name``."""
    results = find_relevant_memories(query, max_results=5, use_ai=False)
    matches = [r for r in results if r["name"] == name]
    assert matches, f"expected to retrieve {name!r}, got {[r['name'] for r in results]}"
    return matches[0]


# ── The bug: a read must not reset staleness ────────────────────────────────

class TestRetrievalDoesNotResetStaleness:
    def test_stale_memory_stays_stale_after_retrieval(self, redirect_memory_dirs):
        user_mem, _ = redirect_memory_dirs
        old = (date.today() - timedelta(days=60)).isoformat()
        fp = _write_memory(
            user_mem, "loader_location",
            content="The data loader is defined in utils/loader.py",
            created=old, last_verified=old, mtime_days_ago=60,
        )

        # Before any retrieval: clearly stale.
        before = _find_one("loader", "loader_location")
        assert before["freshness_text"], "a 60-day-old memory should warn as stale"
        assert trust_recency(before["verified_s"]) < 0.2

        # Simulate a MemorySearch hit writing last_used_at bookkeeping.
        touch_last_used(str(fp))

        # After retrieval: STILL stale. (Pre-fix, mtime was bumped to now and
        # both signals were derived from mtime, so this assertion failed.)
        after = _find_one("loader", "loader_location")
        assert after["freshness_text"], "retrieval must not suppress the stale warning"
        assert trust_recency(after["verified_s"]) < 0.2
        assert after["verified_s"] == before["verified_s"]

    def test_touch_last_used_preserves_file_mtime(self, redirect_memory_dirs):
        user_mem, _ = redirect_memory_dirs
        old = (date.today() - timedelta(days=45)).isoformat()
        fp = _write_memory(user_mem, "note", content="loader note",
                           created=old, last_verified=old, mtime_days_ago=45)
        mtime_before = fp.stat().st_mtime

        touch_last_used(str(fp))  # a read-side write

        # last_used_at was recorded...
        meta, _ = parse_frontmatter(fp.read_text())
        assert meta.get("last_used_at") == date.today().isoformat()
        # ...but the mtime was restored, and last_verified untouched.
        assert abs(fp.stat().st_mtime - mtime_before) < 1.0
        assert meta.get("last_verified") == old


# ── The correct refresh path: explicit verification ─────────────────────────

class TestExplicitVerificationRefreshes:
    def test_mark_verified_clears_staleness(self, redirect_memory_dirs):
        user_mem, _ = redirect_memory_dirs
        old = (date.today() - timedelta(days=90)).isoformat()
        fp = _write_memory(user_mem, "loader_location",
                           content="loader lives in utils/loader.py",
                           created=old, last_verified=old, mtime_days_ago=90)

        assert _find_one("loader", "loader_location")["freshness_text"]

        assert mark_verified(str(fp)) is True
        meta, _ = parse_frontmatter(fp.read_text())
        assert meta.get("last_verified") == date.today().isoformat()

        after = _find_one("loader", "loader_location")
        assert after["freshness_text"] == "", "a just-verified memory is fresh"
        assert trust_recency(after["verified_s"]) > 0.95


# ── Ranking: fresh-verified beats stale, even after the stale one is read ────

class TestVerificationAnchoredRanking:
    def test_recently_verified_outranks_stale(self, redirect_memory_dirs):
        user_mem, _ = redirect_memory_dirs
        old = (date.today() - timedelta(days=120)).isoformat()
        recent = date.today().isoformat()
        stale_fp = _write_memory(user_mem, "stale_loader",
                                 content="loader info (old)",
                                 created=old, last_verified=old, mtime_days_ago=120)
        _write_memory(user_mem, "fresh_loader",
                      content="loader info (verified today)",
                      created=old, last_verified=recent, mtime_days_ago=120)

        # Read the stale one (which, pre-fix, would have made it look fresh).
        touch_last_used(str(stale_fp))

        results = find_relevant_memories("loader", max_results=5, use_ai=False)
        ranked = sorted(
            results,
            key=lambda r: r.get("confidence", 1.0) * trust_recency(r["verified_s"]),
            reverse=True,
        )
        assert ranked[0]["name"] == "fresh_loader"
        assert ranked[-1]["name"] == "stale_loader"


# ── Backward compatibility: legacy files without date fields ────────────────

class TestLegacyFallback:
    def test_missing_dates_fall_back_to_mtime(self, redirect_memory_dirs):
        user_mem, _ = redirect_memory_dirs
        fp = user_mem / "legacy.md"
        # No created / last_verified frontmatter at all.
        fp.write_text("---\nname: legacy\ndescription: loader legacy\ntype: project\n---\nloader legacy body\n")
        os.utime(fp, ((time.time() - 50 * 86_400,) * 2))

        r = _find_one("loader", "legacy")
        # verified_epoch falls back to mtime → recognised as ~50 days stale,
        # and nothing crashes on the empty date fields.
        assert r["freshness_text"]
        assert 0.1 < trust_recency(r["verified_s"]) < 0.3

    def test_parse_date_epoch_handles_garbage(self):
        assert parse_date_epoch("") == 0.0
        assert parse_date_epoch("not-a-date") == 0.0
        assert parse_date_epoch("2026-04-02") > 0.0

    def test_verified_epoch_preference_order(self):
        lv = parse_date_epoch("2026-06-01")
        cr = parse_date_epoch("2026-01-01")
        # last_verified wins over created
        assert verified_epoch("2026-06-01", "2026-01-01", 123.0) == lv
        # created used when last_verified missing
        assert verified_epoch("", "2026-01-01", 123.0) == cr
        # mtime only as last resort
        assert verified_epoch("", "", 123.0) == 123.0
