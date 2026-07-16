"""Tests for circuit_breaker.py."""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws import circuit_breaker as cb
from cheetahclaws.circuit_breaker import CircuitBreaker, CircuitOpenError, State, get_breaker, reset_breaker

_PROV = "test_provider"
_CFG  = {"circuit_failure_threshold": 3,
          "circuit_window_seconds":    30,
          "circuit_cooldown_seconds":  60}


def setup_function():
    reset_breaker(_PROV)


def teardown_function():
    reset_breaker(_PROV)


# ── State machine ─────────────────────────────────────────────────────────

class TestStateMachine:
    def setup_method(self):
        reset_breaker(_PROV)

    def test_initial_state_is_closed(self):
        b = get_breaker(_PROV, _CFG)
        assert b.state == State.CLOSED

    def test_single_failure_stays_closed(self):
        b = get_breaker(_PROV, _CFG)
        b.record_failure()
        assert b.state == State.CLOSED

    def test_opens_after_threshold(self):
        b = get_breaker(_PROV, _CFG)
        for _ in range(3):
            b.record_failure()
        assert b.state == State.OPEN

    def test_allow_request_false_when_open(self):
        b = get_breaker(_PROV, _CFG)
        for _ in range(3):
            b.record_failure()
        assert not b.allow_request()

    def test_allow_request_true_when_closed(self):
        b = get_breaker(_PROV, _CFG)
        assert b.allow_request()

    def test_success_resets_to_closed(self):
        b = get_breaker(_PROV, _CFG)
        b.record_failure()
        b.record_failure()
        b.record_success()
        assert b.state == State.CLOSED
        assert b.allow_request()

    def test_success_clears_failure_times(self):
        b = get_breaker(_PROV, _CFG)
        b.record_failure()
        b.record_failure()
        b.record_success()
        # Two more failures should NOT open (counter reset)
        b.record_failure()
        b.record_failure()
        assert b.state == State.CLOSED

    def test_half_open_after_cooldown(self):
        now = [100.0]
        with patch("time.monotonic", side_effect=lambda: now[0]):
            reset_breaker(_PROV)
            b = get_breaker(_PROV, _CFG)
            b.record_failure(); b.record_failure(); b.record_failure()
            assert b.state == State.OPEN

            now[0] = 161.0   # 61 s later — past cooldown (60 s)
            assert b.state == State.HALF_OPEN

    def test_allow_request_true_in_half_open(self):
        now = [0.0]
        with patch("time.monotonic", side_effect=lambda: now[0]):
            reset_breaker(_PROV)
            b = get_breaker(_PROV, _CFG)
            for _ in range(3):
                b.record_failure()
            now[0] = 61.0
            assert b.allow_request()

    def test_closes_on_success_in_half_open(self):
        now = [0.0]
        with patch("time.monotonic", side_effect=lambda: now[0]):
            reset_breaker(_PROV)
            b = get_breaker(_PROV, _CFG)
            for _ in range(3):
                b.record_failure()
            now[0] = 61.0
            assert b.state == State.HALF_OPEN
            b.record_success()
            assert b.state == State.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        now = [0.0]
        with patch("time.monotonic", side_effect=lambda: now[0]):
            reset_breaker(_PROV)
            b = get_breaker(_PROV, _CFG)
            for _ in range(3):
                b.record_failure()
            now[0] = 61.0
            assert b.state == State.HALF_OPEN
            b.record_failure()          # probe failed → reopen
            now[0] = 62.0               # still within new cooldown
            assert b.state == State.OPEN

    def test_window_expiry_evicts_old_failures(self):
        now = [0.0]
        with patch("time.monotonic", side_effect=lambda: now[0]):
            reset_breaker(_PROV)
            b = get_breaker(_PROV, _CFG)
            b.record_failure(); b.record_failure()  # 2 at t=0
            assert b.state == State.CLOSED

            now[0] = 31.0   # past the 30-s window
            b.record_failure()   # 1 in window — threshold is 3 → stays closed
            assert b.state == State.CLOSED

            b.record_failure(); b.record_failure()  # 3 in window → open
            assert b.state == State.OPEN


# ── Registry ──────────────────────────────────────────────────────────────

class TestRegistry:
    def setup_method(self):
        reset_breaker(_PROV)
        reset_breaker("other")

    def teardown_method(self):
        reset_breaker(_PROV)
        reset_breaker("other")

    def test_get_breaker_returns_same_instance(self):
        b1 = get_breaker(_PROV, _CFG)
        b2 = get_breaker(_PROV, _CFG)
        assert b1 is b2

    def test_different_providers_are_independent(self):
        b1 = get_breaker(_PROV, _CFG)
        b2 = get_breaker("other", _CFG)
        for _ in range(3):
            b1.record_failure()
        assert b1.state == State.OPEN
        assert b2.state == State.CLOSED

    def test_reset_removes_from_registry(self):
        b1 = get_breaker(_PROV, _CFG)
        for _ in range(3):
            b1.record_failure()
        assert b1.state == State.OPEN
        reset_breaker(_PROV)
        b2 = get_breaker(_PROV, _CFG)
        assert b2 is not b1
        assert b2.state == State.CLOSED

    def test_config_applied_on_creation(self):
        reset_breaker("cfg_test")
        try:
            b = get_breaker("cfg_test", {
                "circuit_failure_threshold": 2,
                "circuit_window_seconds":    10,
                "circuit_cooldown_seconds":  5,
            })
            assert b.threshold == 2
            assert b.window    == 10.0
            assert b.cooldown  == 5.0
        finally:
            reset_breaker("cfg_test")


# ── CircuitOpenError ───────────────────────────────────────────────────────

class TestCircuitOpenError:
    def test_is_exception(self):
        assert issubclass(CircuitOpenError, Exception)

    def test_raise_and_catch(self):
        raised = False
        try:
            raise CircuitOpenError("test message")
        except CircuitOpenError as e:
            raised = True
            assert "test message" in str(e)
        assert raised

    def test_caught_by_exception_superclass(self):
        raised = False
        try:
            raise CircuitOpenError("test")
        except Exception:
            raised = True
        assert raised
