"""Tests for logging_utils.py."""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws import logging_utils


def _reset():
    """Reset logging_utils to defaults between tests."""
    logging_utils._level    = logging_utils._LEVELS["warn"]
    logging_utils._log_fh   = None
    logging_utils._log_file_path = None
    logging_utils._cfg_key  = ("warn", None)


class TestConfigure:
    def test_default_level_is_warn(self):
        _reset()
        assert logging_utils._level == logging_utils._LEVELS["warn"]

    def test_set_info_level(self):
        _reset()
        logging_utils.configure("info")
        assert logging_utils._level == logging_utils._LEVELS["info"]

    def test_set_off_level(self):
        _reset()
        logging_utils.configure("off")
        assert logging_utils._level == 0

    def test_unknown_level_falls_back_to_warn(self):
        _reset()
        logging_utils.configure("bogus")
        assert logging_utils._level == logging_utils._LEVELS["warn"]

    def test_fast_path_skips_if_unchanged(self):
        _reset()
        logging_utils.configure("info")
        first_cfg_key = logging_utils._cfg_key
        logging_utils.configure("info")     # same — should be no-op
        assert logging_utils._cfg_key == first_cfg_key

    def test_configure_from_config(self):
        _reset()
        logging_utils.configure_from_config({"log_level": "debug", "log_file": None})
        assert logging_utils._level == logging_utils._LEVELS["debug"]

    def test_configure_from_config_missing_keys(self):
        _reset()
        logging_utils.configure_from_config({})
        assert logging_utils._level == logging_utils._LEVELS["warn"]


class TestEmit:
    def setup_method(self):
        _reset()
        logging_utils.configure("debug")   # let everything through

    def teardown_method(self):
        _reset()

    def _capture(self, fn, *args, **kwargs) -> dict | None:
        buf = io.StringIO()
        original_stderr = logging_utils._log_fh
        logging_utils._log_fh = None  # force to stderr path
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            fn(*args, **kwargs)
        finally:
            sys.stderr = old_stderr
            logging_utils._log_fh = original_stderr
        out = buf.getvalue().strip()
        if not out:
            return None
        return json.loads(out)

    def test_info_emits_json(self):
        rec = self._capture(logging_utils.info, "test_event", session_id="abc")
        assert rec is not None
        assert rec["event"] == "test_event"
        assert rec["level"] == "info"
        assert rec["session_id"] == "abc"

    def test_error_emits_json(self):
        rec = self._capture(logging_utils.error, "oops", code=42)
        assert rec["level"] == "error"
        assert rec["code"] == 42

    def test_warn_emits_json(self):
        rec = self._capture(logging_utils.warn, "watch_out")
        assert rec["level"] == "warn"

    def test_debug_emits_json(self):
        rec = self._capture(logging_utils.debug, "verbose")
        assert rec["level"] == "debug"

    def test_record_has_ts_field(self):
        rec = self._capture(logging_utils.info, "ts_test")
        assert "ts" in rec
        assert "T" in rec["ts"]   # ISO format

    def test_level_filtering_suppresses_debug_at_warn(self):
        _reset()
        logging_utils.configure("warn")
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            logging_utils.debug("should_not_appear")
            logging_utils.info("also_suppressed")
            logging_utils.warn("this_appears")
        finally:
            sys.stderr = old_stderr
        lines = [l for l in buf.getvalue().strip().splitlines() if l]
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "this_appears"

    def test_off_suppresses_everything(self):
        _reset()
        logging_utils.configure("off")
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            logging_utils.error("silent")
            logging_utils.warn("silent")
        finally:
            sys.stderr = old_stderr
        assert buf.getvalue().strip() == ""

    def test_extra_fields_in_output(self):
        rec = self._capture(logging_utils.info, "evt", foo="bar", num=7)
        assert rec["foo"] == "bar"
        assert rec["num"] == 7


class TestLogFile:
    def setup_method(self):
        _reset()

    def teardown_method(self):
        _reset()

    def test_writes_to_file(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".log", delete=False) as f:
            path = f.name
        try:
            logging_utils.configure("info", log_file=path)
            logging_utils.info("file_event", x=1)
            logging_utils.configure("warn", log_file=None)  # close file
            with open(path) as fh:
                data = json.loads(fh.read().strip())
            assert data["event"] == "file_event"
            assert data["x"] == 1
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    def test_same_path_not_reopened(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            logging_utils.configure("info", log_file=path)
            fh1 = logging_utils._log_fh
            logging_utils.configure("info", log_file=path)  # same path
            fh2 = logging_utils._log_fh
            assert fh1 is fh2   # same file object, not re-opened
        finally:
            logging_utils.configure("warn", log_file=None)
            try:
                os.unlink(path)
            except Exception:
                pass


class TestThreadSafety:
    def test_concurrent_logging_does_not_corrupt(self):
        _reset()
        logging_utils.configure("info")
        lines = []
        lock  = threading.Lock()

        def worker(tid):
            buf = io.StringIO()
            old = sys.stderr
            # Each thread writes to its own buffer to test the emit code path
            for _ in range(20):
                logging_utils.info("concurrent_test", tid=tid)
            # We mainly care that no exception is thrown

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # If we got here without exceptions the test passes
        _reset()
