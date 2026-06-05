"""Tests for the user-facing token/cost budget feature.

Covers the quota helpers (parse_budget / fmt_amount / usage_vs_limits / warnings),
the /budget command (view / set / clear), and the QuotaPause event the agent
yields when a budget is reached.
"""
import pytest

import quota


@pytest.fixture(autouse=True)
def _clean_session(tmp_path, monkeypatch):
    # Isolate the on-disk daily counter so tests never read or pollute the real
    # ~/.cheetahclaws/quota/ file (matches test_quota.py's approach).
    monkeypatch.setattr(quota, "_quota_dir", lambda: tmp_path)
    quota.reset_session("t")
    yield
    quota.reset_session("t")


# ── parse_budget ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("$5",      ("cost", 5.0)),
    ("$5.50",   ("cost", 5.5)),
    ("5usd",    ("cost", 5.0)),
    ("10$",     ("cost", 10.0)),
    ("200k",    ("tokens", 200_000)),
    ("1.5m",    ("tokens", 1_500_000)),
    ("200000",  ("tokens", 200_000)),
    ("2,000",   ("tokens", 2_000)),
    (" 200K ",  ("tokens", 200_000)),
])
def test_parse_budget_ok(text, expected):
    assert quota.parse_budget(text) == expected


@pytest.mark.parametrize("bad", ["", "abc", "-3", "$0", "0", "$-1", "k"])
def test_parse_budget_rejects(bad):
    with pytest.raises(ValueError):
        quota.parse_budget(bad)


def test_budget_keys_mapping():
    assert quota.BUDGET_KEYS[("cost", "session")] == "session_cost_budget"
    assert quota.BUDGET_KEYS[("tokens", "session")] == "session_token_budget"
    assert quota.BUDGET_KEYS[("cost", "daily")] == "daily_cost_budget"
    assert quota.BUDGET_KEYS[("tokens", "daily")] == "daily_token_budget"


# ── fmt_amount ──────────────────────────────────────────────────────────────

def test_fmt_amount():
    assert quota.fmt_amount(5, "usd") == "$5.00"
    assert quota.fmt_amount(1.834, "usd") == "$1.83"
    assert quota.fmt_amount(124_000, "tok") == "124k"
    assert quota.fmt_amount(2_000_000, "tok") == "2m"
    assert quota.fmt_amount(540, "tok") == "540"


# ── usage_vs_limits ─────────────────────────────────────────────────────────

def test_usage_vs_limits_unlimited_by_default():
    rows = quota.usage_vs_limits("t", {})
    assert {r["key"] for r in rows} == set(quota.BUDGET_KEYS.values())
    assert all(r["limit"] is None and r["pct"] is None for r in rows)


def test_usage_vs_limits_computes_pct():
    with quota._lock:
        quota._sess_tokens["t"] = 50_000
    rows = quota.usage_vs_limits("t", {"session_token_budget": 200_000})
    row = next(r for r in rows if r["key"] == "session_token_budget")
    assert row["used"] == 50_000
    assert row["limit"] == 200_000
    assert row["pct"] == pytest.approx(25.0)


# ── warnings (80% warn / 95% crit / 100% hard-stop, no warn) ─────────────────

@pytest.mark.parametrize("cost,level", [
    (3.0, None),    # 60% — no warning
    (4.3, "warn"),  # 86%
    (4.8, "crit"),  # 96%
    (6.0, None),    # 120% — exhausted; hard stop handles it, not a warning
])
def test_warnings_thresholds(cost, level):
    with quota._lock:
        quota._sess_cost["t"] = cost
    out = quota.warnings("t", {"session_cost_budget": 5.0})
    if level is None:
        assert out == []
    else:
        assert len(out) == 1 and out[0][0] == level


# ── /budget command ─────────────────────────────────────────────────────────

@pytest.fixture
def cmd(monkeypatch):
    import cc_config
    monkeypatch.setattr(cc_config, "save_config", lambda cfg: None)
    from commands.core import cmd_budget
    return cmd_budget


def test_cmd_budget_set_cost(cmd):
    cfg = {"_session_id": "t"}
    assert cmd("$5", None, cfg) is True
    assert cfg["session_cost_budget"] == 5.0


def test_cmd_budget_set_daily_tokens(cmd):
    cfg = {"_session_id": "t"}
    cmd("daily 2m", None, cfg)
    assert cfg["daily_token_budget"] == 2_000_000


def test_cmd_budget_explicit_session_scope(cmd):
    cfg = {"_session_id": "t"}
    cmd("session 200k", None, cfg)
    assert cfg["session_token_budget"] == 200_000


def test_cmd_budget_clear(cmd):
    cfg = {"_session_id": "t", "session_cost_budget": 5.0,
           "daily_token_budget": 2_000_000}
    cmd("clear", None, cfg)
    assert all(cfg[k] is None for k in quota.BUDGET_KEYS.values())


def test_cmd_budget_set_replaces_other_unit_in_scope(cmd):
    # A leftover token cap must not keep blocking after switching to a $ cap.
    cfg = {"_session_id": "t", "session_token_budget": 20_000}
    cmd("$2", None, cfg)
    assert cfg["session_cost_budget"] == 2.0
    assert cfg["session_token_budget"] is None      # replaced, not coexisting
    # Daily caps are a different scope — untouched.
    cfg2 = {"_session_id": "t", "daily_cost_budget": 20.0}
    cmd("session 200k", None, cfg2)
    assert cfg2["session_token_budget"] == 200_000
    assert cfg2["daily_cost_budget"] == 20.0


def test_cmd_budget_bad_value_does_not_set(cmd, capsys):
    cfg = {"_session_id": "t"}
    cmd("banana", None, cfg)
    assert "session_token_budget" not in cfg or cfg.get("session_token_budget") is None
    assert "session_cost_budget" not in cfg


def test_cmd_budget_view_runs_with_no_budgets(cmd, capsys):
    assert cmd("", None, {"_session_id": "t"}) is True
    assert "unlimited" in capsys.readouterr().out


# ── QuotaPause event ────────────────────────────────────────────────────────

def test_quota_pause_event_shape():
    from agent import QuotaPause
    ev = QuotaPause("Session cost budget reached", {"session_cost": 5.0})
    assert ev.reason == "Session cost budget reached"
    assert ev.usage["session_cost"] == 5.0


def test_check_quota_raises_when_over_budget():
    with quota._lock:
        quota._sess_cost["t"] = 5.0
    with pytest.raises(quota.QuotaExceeded):
        quota.check_quota("t", {"session_cost_budget": 5.0})


# ── pre-call projection (tight cap) ─────────────────────────────────────────

def test_check_quota_projection_stops_before_overshoot():
    # 30k spent, 20k cap... already over → "reached". Use under-cap spend instead.
    with quota._lock:
        quota._sess_tokens["t"] = 30_000
    cfg = {"session_token_budget": 40_000}
    # Without projection: 30k < 40k → allowed.
    quota.check_quota("t", cfg)
    # With a projected 15k next request: 30k+15k ≥ 40k → stop BEFORE the call.
    with pytest.raises(quota.QuotaExceeded) as ei:
        quota.check_quota("t", cfg, projected_tokens=15_000)
    assert "would be exceeded" in str(ei.value)


def test_check_quota_projection_allows_when_fits():
    with quota._lock:
        quota._sess_tokens["t"] = 10_000
    # 10k + 5k = 15k < 40k → fine.
    quota.check_quota("t", {"session_token_budget": 40_000}, projected_tokens=5_000)


def test_quota_exceeded_carries_breached_cap_fields():
    with quota._lock:
        quota._sess_tokens["t"] = 25_000
    with pytest.raises(quota.QuotaExceeded) as ei:
        quota.check_quota("t", {"session_token_budget": 20_000})
    e = ei.value
    assert e.key == "session_token_budget"
    assert e.scope == "session"
    assert e.unit == "tok"
    assert e.limit == 20_000


def test_quota_exceeded_cost_cap_fields():
    with quota._lock:
        quota._sess_cost["t"] = 3.0
    with pytest.raises(quota.QuotaExceeded) as ei:
        quota.check_quota("t", {"session_cost_budget": 2.0})
    assert ei.value.unit == "usd"
    assert ei.value.scope == "session"
    assert ei.value.key == "session_cost_budget"


# ── output_room (clamp) ─────────────────────────────────────────────────────

def test_output_room_none_without_budget():
    assert quota.output_room("t", {}) is None


def test_output_room_token_budget_headroom():
    with quota._lock:
        quota._sess_tokens["t"] = 30_000
    # 40k cap, 30k spent, 5k projected input → 5k left for output.
    room = quota.output_room("t", {"session_token_budget": 40_000}, projected_tokens=5_000)
    assert room == 5_000


def test_output_room_never_negative():
    with quota._lock:
        quota._sess_tokens["t"] = 50_000
    assert quota.output_room("t", {"session_token_budget": 40_000}) == 0


def test_output_room_takes_tightest_of_multiple():
    # record_usage advances BOTH the session and (isolated) daily counters.
    quota.record_usage("t", "claude-sonnet-4-6", 10_000, 0)
    cfg = {"session_token_budget": 40_000, "daily_token_budget": 12_000}
    # session leaves 30k, daily leaves 2k → min is 2k.
    assert quota.output_room("t", cfg) == 2_000


def test_output_room_cost_budget_uses_output_price(monkeypatch):
    # Model with a known $/Mtok output price → cost cap converts to token room.
    import providers
    monkeypatch.setitem(providers.COSTS, "budgetmodel", (1.0, 2.0))  # $2 /Mtok out
    with quota._lock:
        quota._sess_cost["t"] = 0.0
    cfg = {"model": "budgetmodel", "session_cost_budget": 0.10}  # $0.10 → 50k out tokens
    room = quota.output_room("t", cfg)
    assert room == 50_000
