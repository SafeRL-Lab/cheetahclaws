"""Tests for RFC 0002 §F-9 cost-guardrail defaults.

Three layers:

  1. ``_apply_serve_defaults`` — pure function: None → conservative
     defaults; pre-set values left alone.
  2. ``system.status`` RPC — returns the four budget keys + live
     runner / bridge counts.
  3. ``agent.resume`` RPC — merges ``budget_overrides`` into
     ``daemon_state.config`` and rejects bad keys / non-numeric values.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── 1. _apply_serve_defaults ──────────────────────────────────────────────


class TestApplyServeDefaults(unittest.TestCase):

    def test_all_none_flipped_to_conservative_defaults(self):
        from cc_daemon.cli import _apply_serve_defaults, F9_SERVE_BUDGET_DEFAULTS
        cfg = {
            "session_token_budget": None,
            "session_cost_budget":  None,
            "daily_token_budget":   None,
            "daily_cost_budget":    None,
        }
        out = _apply_serve_defaults(cfg)
        # Same dict (no copy) — tests rely on this.
        self.assertIs(out, cfg)
        for k, v in F9_SERVE_BUDGET_DEFAULTS.items():
            self.assertEqual(cfg[k], v, f"{k} not defaulted")

    def test_existing_values_are_preserved(self):
        """An operator who already configured a budget keeps it. F-9
        defaults only fill in the gaps."""
        from cc_daemon.cli import _apply_serve_defaults
        cfg = {
            "session_token_budget": 50,    # user-chosen
            "session_cost_budget":  None,  # default applies
            "daily_token_budget":   1,     # explicit zero would also stick
            "daily_cost_budget":    0.0,   # ditto
        }
        _apply_serve_defaults(cfg)
        self.assertEqual(cfg["session_token_budget"], 50)
        self.assertEqual(cfg["daily_token_budget"], 1)
        self.assertEqual(cfg["daily_cost_budget"], 0.0)
        # Only the None slot got the default.
        self.assertEqual(cfg["session_cost_budget"], 2.0)

    def test_unrelated_keys_untouched(self):
        from cc_daemon.cli import _apply_serve_defaults
        cfg = {"log_level": "info", "model": "claude-opus-4-7"}
        _apply_serve_defaults(cfg)
        self.assertEqual(cfg["log_level"], "info")
        self.assertEqual(cfg["model"], "claude-opus-4-7")
        # Plus the four budget defaults landed.
        self.assertEqual(cfg["session_token_budget"], 200_000)


# ── 2. system.status RPC ──────────────────────────────────────────────────


class _FakeDaemonState:
    def __init__(self, config=None):
        self.config = config or {}


def _build_system_registry(state):
    from cc_daemon.rpc import RpcRegistry
    from cc_daemon import system_methods
    reg = RpcRegistry()
    system_methods.register(reg, state)
    return reg


def _ctx():
    from cc_daemon.rpc import CallContext
    return CallContext(client_id="t", transport="unix", api_version="0")


def _call(reg, method, params=None):
    envelope = {"jsonrpc": "2.0", "id": 1, "method": method,
                "params": params or {}}
    response, _ = reg.dispatch(envelope, _ctx())
    return response.get("result"), response.get("error")


class TestSystemStatus(unittest.TestCase):

    def test_status_returns_budgets(self):
        state = _FakeDaemonState({
            "session_token_budget": 200_000,
            "session_cost_budget":  2.0,
            "daily_token_budget":   2_000_000,
            "daily_cost_budget":    20.0,
        })
        reg = _build_system_registry(state)
        result, err = _call(reg, "system.status", {})
        self.assertIsNone(err)
        self.assertEqual(result["budgets"]["session_token_budget"], 200_000)
        self.assertEqual(result["budgets"]["session_cost_budget"], 2.0)
        self.assertEqual(result["budgets"]["daily_token_budget"], 2_000_000)
        self.assertEqual(result["budgets"]["daily_cost_budget"], 20.0)

    def test_status_returns_none_when_unlimited(self):
        state = _FakeDaemonState({})
        reg = _build_system_registry(state)
        result, err = _call(reg, "system.status", {})
        self.assertIsNone(err)
        for k in ("session_token_budget", "session_cost_budget",
                  "daily_token_budget",   "daily_cost_budget"):
            self.assertIsNone(result["budgets"][k])

    def test_status_includes_runner_and_bridge_counts(self):
        state = _FakeDaemonState({})
        reg = _build_system_registry(state)
        result, err = _call(reg, "system.status", {})
        self.assertIsNone(err)
        self.assertIn("runners", result)
        self.assertIn("bridges", result)


# ── 3. agent.resume RPC ───────────────────────────────────────────────────


def _build_agent_registry(state):
    from cc_daemon.rpc import RpcRegistry
    from cc_daemon import agent_methods
    reg = RpcRegistry()
    agent_methods.register(reg, state)
    return reg


class TestAgentResume(unittest.TestCase):

    def test_resume_merges_overrides_into_config(self):
        state = _FakeDaemonState({"session_token_budget": 10})
        reg = _build_agent_registry(state)
        result, err = _call(reg, "agent.resume", {
            "budget_overrides": {
                "session_token_budget": 500_000,
                "daily_cost_budget":    50.0,
            },
        })
        self.assertIsNone(err)
        # daemon_state.config mutated in place.
        self.assertEqual(state.config["session_token_budget"], 500_000)
        self.assertEqual(state.config["daily_cost_budget"], 50.0)
        # Response reports the merged result and resumed=null (no name).
        self.assertEqual(result["budgets"]["session_token_budget"], 500_000)
        self.assertIsNone(result["resumed"])

    def test_resume_null_value_resets_to_unlimited(self):
        state = _FakeDaemonState({"daily_token_budget": 100})
        reg = _build_agent_registry(state)
        _, err = _call(reg, "agent.resume", {
            "budget_overrides": {"daily_token_budget": None},
        })
        self.assertIsNone(err)
        self.assertIsNone(state.config["daily_token_budget"])

    def test_resume_rejects_unknown_key(self):
        state = _FakeDaemonState({})
        reg = _build_agent_registry(state)
        _, err = _call(reg, "agent.resume", {
            "budget_overrides": {"frob_budget": 1},
        })
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)
        self.assertIn("frob_budget", err["message"])

    def test_resume_rejects_non_numeric(self):
        state = _FakeDaemonState({})
        reg = _build_agent_registry(state)
        _, err = _call(reg, "agent.resume", {
            "budget_overrides": {"session_token_budget": "soon"},
        })
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)

    def test_resume_rejects_non_dict_overrides(self):
        state = _FakeDaemonState({})
        reg = _build_agent_registry(state)
        _, err = _call(reg, "agent.resume", {"budget_overrides": [1, 2]})
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)

    def test_resume_with_empty_overrides_is_noop(self):
        state = _FakeDaemonState({"session_token_budget": 100})
        reg = _build_agent_registry(state)
        result, err = _call(reg, "agent.resume", {"budget_overrides": {}})
        self.assertIsNone(err)
        self.assertEqual(state.config["session_token_budget"], 100)
        # Response still lists all four keys (some None, some preserved).
        self.assertEqual(result["budgets"]["session_token_budget"], 100)
        self.assertIsNone(result["resumed"])

    def test_resume_with_name_calls_supervisor(self):
        """When ``name`` is supplied, agent.resume should call
        runner_supervisor.resume(name) and surface the bool result."""
        from unittest.mock import patch
        state = _FakeDaemonState({})
        reg = _build_agent_registry(state)
        from cc_daemon import runner_supervisor as rs
        with patch.object(rs, "resume", return_value=True) as mock_resume:
            result, err = _call(reg, "agent.resume", {
                "name": "agent-x",
                "budget_overrides": {"session_token_budget": 1_000_000},
            })
            self.assertIsNone(err)
            mock_resume.assert_called_once_with("agent-x")
            self.assertTrue(result["resumed"])
            self.assertEqual(result["budgets"]["session_token_budget"], 1_000_000)

    def test_resume_rejects_empty_name(self):
        state = _FakeDaemonState({})
        reg = _build_agent_registry(state)
        _, err = _call(reg, "agent.resume", {"name": "", "budget_overrides": {}})
        self.assertIsNotNone(err)
        self.assertEqual(err["code"], -32602)
        self.assertIn("name", err["message"])


if __name__ == "__main__":
    unittest.main()
