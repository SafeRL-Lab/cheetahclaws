"""Tests for health.py module-level payload helpers.

These are exercised by both health.py's own HTTP server and by the daemon
listener (daemon/server.py).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws import health


def test_healthz_payload_basic_shape():
    body = health.healthz_payload({"model": "claude-x"})
    assert body["status"] == "ok"
    assert body["model"] == "claude-x"
    assert "uptime_s" in body
    assert isinstance(body["active_sessions"], int)


def test_healthz_payload_handles_missing_config():
    body = health.healthz_payload(None)
    assert body["status"] == "ok"
    assert body["model"] == ""


def test_readyz_ok_when_no_open_circuits(monkeypatch):
    monkeypatch.setattr(health, "_circuit_states", lambda: {})
    body = health.readyz_payload({"model": "m"})
    assert body["status"] == "ok"
    assert body["circuits"] == {}
    assert "open_circuits" not in body


def test_readyz_degraded_when_a_circuit_is_open(monkeypatch):
    monkeypatch.setattr(health, "_circuit_states",
                        lambda: {"anthropic": "open", "openai": "closed"})
    body = health.readyz_payload({})
    assert body["status"] == "degraded"
    assert body["open_circuits"] == ["anthropic"]


def test_metrics_payload_includes_uptime_and_circuits(monkeypatch):
    monkeypatch.setattr(health, "_circuit_states",
                        lambda: {"x": "closed"})
    body = health.metrics_payload({"model": "m"})
    for key in ("uptime_s", "model", "active_sessions",
                 "circuits", "daily_tokens", "daily_cost_usd"):
        assert key in body
    assert body["circuits"] == {"x": "closed"}


def test_payload_for_dispatches_correct_keys(monkeypatch):
    monkeypatch.setattr(health, "_circuit_states", lambda: {})
    assert health.payload_for("/healthz", {"model": "x"})["status"] == "ok"
    assert health.payload_for("/readyz", {"model": "x"})["status"] == "ok"
    assert "daily_tokens" in health.payload_for("/metrics", {"model": "x"})


def test_payload_for_unknown_path_returns_empty():
    assert health.payload_for("/nope") == {}


def test_install_config_pins_module_default():
    health.install_config({"model": "pinned"})
    body = health.healthz_payload(None)
    assert body["model"] == "pinned"
