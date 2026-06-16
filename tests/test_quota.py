"""Tests for quota.py."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws import quota
from cheetahclaws.quota import QuotaExceeded, check_quota, record_usage, get_usage, reset_session


# ── Helpers ───────────────────────────────────────────────────────────────

def _reset_session(sid):
    with quota._lock:
        quota._sess_tokens.pop(sid, None)
        quota._sess_cost.pop(sid, None)


# ── QuotaExceeded ─────────────────────────────────────────────────────────

class TestQuotaExceeded:
    def test_is_exception(self):
        assert issubclass(QuotaExceeded, Exception)

    def test_reason_attribute(self):
        e = QuotaExceeded("limit hit")
        assert e.reason == "limit hit"
        assert "limit hit" in str(e)


# ── check_quota ───────────────────────────────────────────────────────────

class TestCheckQuota:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sid    = "test_sess"
        _reset_session(self.sid)
        self._patcher = patch("cheetahclaws.quota._quota_dir", return_value=Path(self.tmpdir))
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()
        _reset_session(self.sid)

    def test_no_limits_no_exception(self):
        check_quota(self.sid, {})  # should not raise

    def test_zero_limits_treated_as_no_limit(self):
        check_quota(self.sid, {
            "session_token_budget": 0,
            "session_cost_budget":  0.0,
            "daily_token_budget":   0,
            "daily_cost_budget":    0.0,
        })

    def test_session_token_budget_exceeded(self):
        with quota._lock:
            quota._sess_tokens[self.sid] = 1000
        import pytest
        with pytest.raises(QuotaExceeded, match="Session token"):
            check_quota(self.sid, {"session_token_budget": 500})

    def test_session_token_budget_not_yet_exceeded(self):
        with quota._lock:
            quota._sess_tokens[self.sid] = 499
        check_quota(self.sid, {"session_token_budget": 500})

    def test_session_token_budget_at_exact_limit_raises(self):
        with quota._lock:
            quota._sess_tokens[self.sid] = 500
        import pytest
        with pytest.raises(QuotaExceeded):
            check_quota(self.sid, {"session_token_budget": 500})

    def test_session_cost_budget_exceeded(self):
        with quota._lock:
            quota._sess_cost[self.sid] = 1.0
        import pytest
        with pytest.raises(QuotaExceeded, match="Session cost"):
            check_quota(self.sid, {"session_cost_budget": 0.5})

    def test_daily_token_budget_exceeded(self):
        # Write a daily file that exceeds limit
        today = quota._today_key()
        p = Path(self.tmpdir) / f"{today}.json"
        p.write_text('{"tokens": 10000, "cost": 0.0}')
        import pytest
        with pytest.raises(QuotaExceeded, match="Daily token"):
            check_quota(self.sid, {"daily_token_budget": 5000})

    def test_daily_cost_budget_exceeded(self):
        today = quota._today_key()
        p = Path(self.tmpdir) / f"{today}.json"
        p.write_text('{"tokens": 0, "cost": 5.0}')
        import pytest
        with pytest.raises(QuotaExceeded, match="Daily cost"):
            check_quota(self.sid, {"daily_cost_budget": 2.0})

    def test_missing_daily_file_treated_as_zero(self):
        check_quota(self.sid, {"daily_token_budget": 100000})


# ── record_usage ──────────────────────────────────────────────────────────

class TestRecordUsage:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sid    = "rec_sess"
        _reset_session(self.sid)
        self._patcher = patch("cheetahclaws.quota._quota_dir", return_value=Path(self.tmpdir))
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()
        _reset_session(self.sid)

    def test_updates_session_token_counter(self):
        record_usage(self.sid, "claude-sonnet-4-6", 100, 200)
        with quota._lock:
            assert quota._sess_tokens[self.sid] == 300

    def test_updates_session_token_counter_accumulated(self):
        record_usage(self.sid, "claude-sonnet-4-6", 100, 200)
        record_usage(self.sid, "claude-sonnet-4-6", 50, 50)
        with quota._lock:
            assert quota._sess_tokens[self.sid] == 400

    def test_updates_daily_file(self):
        record_usage(self.sid, "claude-sonnet-4-6", 100, 200)
        today = quota._today_key()
        import json
        data = json.loads((Path(self.tmpdir) / f"{today}.json").read_text())
        assert data["tokens"] == 300

    def test_daily_file_accumulates_across_calls(self):
        record_usage(self.sid, "claude-sonnet-4-6", 100, 200)
        record_usage(self.sid, "claude-sonnet-4-6", 400, 600)
        today = quota._today_key()
        import json
        data = json.loads((Path(self.tmpdir) / f"{today}.json").read_text())
        assert data["tokens"] == 1300

    def test_cost_is_recorded(self):
        record_usage(self.sid, "claude-sonnet-4-6", 1_000_000, 0)
        # sonnet input is $3/M tokens → $3.00
        with quota._lock:
            assert quota._sess_cost[self.sid] == pytest_approx(3.0, rel=0.01)


def pytest_approx(value, rel=None):
    """Inline helper so we don't need to import pytest in helpers."""
    import pytest
    return pytest.approx(value, rel=rel)


# ── get_usage ─────────────────────────────────────────────────────────────

class TestGetUsage:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sid    = "get_sess"
        _reset_session(self.sid)
        self._patcher = patch("cheetahclaws.quota._quota_dir", return_value=Path(self.tmpdir))
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()
        _reset_session(self.sid)

    def test_returns_zeroes_for_fresh_session(self):
        u = get_usage(self.sid)
        assert u["session_tokens"] == 0
        assert u["session_cost"]   == 0.0
        assert u["daily_tokens"]   == 0
        assert u["daily_cost"]     == 0.0

    def test_reflects_recorded_usage(self):
        record_usage(self.sid, "claude-sonnet-4-6", 100, 200)
        u = get_usage(self.sid)
        assert u["session_tokens"] == 300
        assert u["daily_tokens"]   == 300


# ── reset_session ─────────────────────────────────────────────────────────

class TestResetSession:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sid    = "rst_sess"
        _reset_session(self.sid)
        self._patcher = patch("cheetahclaws.quota._quota_dir", return_value=Path(self.tmpdir))
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()
        _reset_session(self.sid)

    def test_clears_session_counters(self):
        record_usage(self.sid, "claude-sonnet-4-6", 100, 200)
        reset_session(self.sid)
        with quota._lock:
            assert self.sid not in quota._sess_tokens
            assert self.sid not in quota._sess_cost

    def test_daily_counters_unaffected(self):
        record_usage(self.sid, "claude-sonnet-4-6", 100, 200)
        reset_session(self.sid)
        u = get_usage(self.sid)
        # Daily file still has the recorded usage
        assert u["daily_tokens"] == 300

    def test_reset_nonexistent_session_is_noop(self):
        reset_session("nonexistent_session_xyz")  # should not raise


# ── Thread safety ─────────────────────────────────────────────────────────

class TestThreadSafety:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = patch("cheetahclaws.quota._quota_dir", return_value=Path(self.tmpdir))
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def test_concurrent_record_usage_thread_safe(self):
        sid = "thread_test"
        _reset_session(sid)
        try:
            errors = []

            def worker():
                try:
                    record_usage(sid, "claude-sonnet-4-6", 10, 10)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            with quota._lock:
                assert quota._sess_tokens[sid] == 20 * 20  # 20 tokens each
        finally:
            _reset_session(sid)
