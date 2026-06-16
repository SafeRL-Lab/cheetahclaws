"""Tests for kernel.ledger (RFC 0006)."""
from __future__ import annotations

import threading

import pytest

from cheetahclaws.kernel import (
    KernelStore,
    LedgerEntry,
    LedgerExists,
    LedgerInvalidAmount,
    LedgerInvalidRefund,
    LedgerInvalidWarnAt,
    LedgerStore,
    LedgerUnknownDim,
    UnknownPid,
)


@pytest.fixture
def stores(tmp_path):
    """Open kernel + ledger sharing connection + lock."""
    ks = KernelStore.open(tmp_path / "kernel.db")
    ls = LedgerStore(ks.connection, write_lock=ks.write_lock)
    yield ks, ls
    ks.close()


# ── create ────────────────────────────────────────────────────────────────


def test_create_round_trip(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    dims = ls.create(pid=a.pid, grants={"tokens": 200_000, "cost_micro": 2_000_000})
    assert sorted(dims) == ["cost_micro", "tokens"]
    led = ls.get(a.pid)
    assert led.pid == a.pid
    assert len(led.entries) == 2
    used_by_dim = {e.dim: e.used for e in led.entries}
    assert used_by_dim == {"tokens": 0, "cost_micro": 0}


def test_create_with_unknown_pid(stores):
    _, ls = stores
    with pytest.raises(UnknownPid):
        ls.create(pid=9999, grants={"tokens": 1000})


def test_create_rejects_zero_or_negative_grant(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    with pytest.raises(LedgerInvalidAmount):
        ls.create(pid=a.pid, grants={"tokens": 0})
    with pytest.raises(LedgerInvalidAmount):
        ls.create(pid=a.pid, grants={"tokens": -5})


def test_create_rejects_bad_warn_at(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    with pytest.raises(LedgerInvalidWarnAt):
        ls.create(pid=a.pid, grants={"tokens": 100}, warn_at=1.5)
    with pytest.raises(LedgerInvalidWarnAt):
        ls.create(pid=a.pid, grants={"tokens": 100}, warn_at=-0.1)


def test_create_duplicate_dim_raises(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    with pytest.raises(LedgerExists):
        ls.create(pid=a.pid, grants={"tokens": 200})


def test_create_additive_over_separate_dims(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    ls.create(pid=a.pid, grants={"cost_micro": 5_000})
    led = ls.get(a.pid)
    assert {e.dim for e in led.entries} == {"tokens", "cost_micro"}


# ── charge: standard transitions ──────────────────────────────────────────


def test_charge_under_budget(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 1000}, warn_at=0.8)
    r = ls.charge(pid=a.pid, dim="tokens", amount=100)
    assert r.used == 100
    assert r.over_limit is False
    assert r.warned is False
    assert r.first_breach is False


def test_charge_at_warn_threshold(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 1000}, warn_at=0.5)
    r = ls.charge(pid=a.pid, dim="tokens", amount=600)
    assert r.warned is True       # crossed 500 (= 0.5 * 1000)
    assert r.over_limit is False
    assert r.first_breach is False


def test_charge_warn_only_fires_once(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 1000}, warn_at=0.5)
    r1 = ls.charge(pid=a.pid, dim="tokens", amount=600)
    r2 = ls.charge(pid=a.pid, dim="tokens", amount=100)
    assert r1.warned is True
    assert r2.warned is False     # already crossed before this charge


def test_charge_first_breach(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    r = ls.charge(pid=a.pid, dim="tokens", amount=150)
    assert r.over_limit   is True
    assert r.first_breach is True
    assert r.used == 150


def test_charge_over_again_not_first_breach(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    ls.charge(pid=a.pid, dim="tokens", amount=150)         # first_breach=True
    r2 = ls.charge(pid=a.pid, dim="tokens", amount=50)
    assert r2.over_limit   is True
    assert r2.first_breach is False


def test_charge_unknown_dim_raises(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    with pytest.raises(LedgerUnknownDim) as e:
        ls.charge(pid=a.pid, dim="tokens", amount=10)
    assert e.value.pid == a.pid
    assert e.value.dim == "tokens"


def test_charge_invalid_amount_raises(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    with pytest.raises(LedgerInvalidAmount):
        ls.charge(pid=a.pid, dim="tokens", amount=-1)


def test_charge_zero_amount_is_a_noop_charge(stores):
    """Charging 0 is allowed (idempotent ping)."""
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    r = ls.charge(pid=a.pid, dim="tokens", amount=0)
    assert r.used == 0
    assert r.over_limit is False


# ── check (read-only) ─────────────────────────────────────────────────────


def test_check_does_not_mutate(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    ls.charge(pid=a.pid, dim="tokens", amount=30)
    chk = ls.check(pid=a.pid, dim="tokens", amount=50)
    assert chk.used == 30
    assert chk.would_use == 80
    assert chk.would_exceed is False
    # Confirm the actual used didn't change.
    assert ls.get(a.pid).entries[0].used == 30


def test_check_predicts_breach(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    chk = ls.check(pid=a.pid, dim="tokens", amount=200)
    assert chk.would_exceed is True


def test_check_unknown_dim(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    with pytest.raises(LedgerUnknownDim):
        ls.check(pid=a.pid, dim="cpu_s", amount=10)


# ── refund ────────────────────────────────────────────────────────────────


def test_refund_decrements(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    ls.charge(pid=a.pid, dim="tokens", amount=50)
    e = ls.refund(pid=a.pid, dim="tokens", amount=20)
    assert e.used == 30


def test_refund_over_used_rejected(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    ls.charge(pid=a.pid, dim="tokens", amount=10)
    with pytest.raises(LedgerInvalidRefund):
        ls.refund(pid=a.pid, dim="tokens", amount=20)


# ── update_grant ──────────────────────────────────────────────────────────


def test_update_grant_extends_budget(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    ls.charge(pid=a.pid, dim="tokens", amount=150)        # over_limit
    ls.update_grant(pid=a.pid, dim="tokens", new_grant=200)
    led = ls.get(a.pid)
    e = led.entries[0]
    assert e.granted == 200
    assert e.used == 150


def test_update_grant_unknown_dim(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    with pytest.raises(LedgerUnknownDim):
        ls.update_grant(pid=a.pid, dim="tokens", new_grant=100)


# ── list_breached ─────────────────────────────────────────────────────────


def test_list_breached_only_returns_over(stores):
    ks, ls = stores
    a1 = ks.create(name="a1", template="t")
    a2 = ks.create(name="a2", template="t")
    ls.create(pid=a1.pid, grants={"tokens": 100})
    ls.create(pid=a2.pid, grants={"tokens": 100})
    ls.charge(pid=a1.pid, dim="tokens", amount=200)        # over
    ls.charge(pid=a2.pid, dim="tokens", amount=50)         # under
    breached = ls.list_breached()
    assert {(e.pid, e.dim) for e in breached} == {(a1.pid, "tokens")}


def test_list_breached_after_grant_update_clears(stores):
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 100})
    ls.charge(pid=a.pid, dim="tokens", amount=200)
    assert len(ls.list_breached()) == 1
    ls.update_grant(pid=a.pid, dim="tokens", new_grant=500)
    assert ls.list_breached() == []


# ── concurrent charge atomicity ───────────────────────────────────────────


def test_concurrent_charges_no_lost_update(stores):
    """100 charges of 1 across 4 threads must result in used == 100."""
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 1_000_000})
    errors: list[Exception] = []

    def worker(n: int):
        try:
            for _ in range(n):
                ls.charge(pid=a.pid, dim="tokens", amount=1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(25,)) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    led = ls.get(a.pid)
    assert led.entries[0].used == 100


# ── Full lifecycle integration ────────────────────────────────────────────


def test_full_lifecycle(stores):
    """Create → charge → warn → breach → refund → update → check."""
    ks, ls = stores
    a = ks.create(name="x", template="t")
    ls.create(pid=a.pid, grants={"tokens": 1000}, warn_at=0.8)

    r1 = ls.charge(pid=a.pid, dim="tokens", amount=300)
    assert (r1.used, r1.warned, r1.over_limit) == (300, False, False)

    r2 = ls.charge(pid=a.pid, dim="tokens", amount=600)
    # 900 / 1000 = 90% — crosses 80% warn threshold
    assert (r2.used, r2.warned, r2.over_limit) == (900, True, False)

    r3 = ls.charge(pid=a.pid, dim="tokens", amount=200)
    # 1100 > 1000 — first breach
    assert (r3.used, r3.over_limit, r3.first_breach) == (1100, True, True)

    ls.refund(pid=a.pid, dim="tokens", amount=300)         # 800
    ls.update_grant(pid=a.pid, dim="tokens", new_grant=2000)
    chk = ls.check(pid=a.pid, dim="tokens", amount=500)
    assert chk.used == 800
    assert chk.would_use == 1300
    assert chk.would_exceed is False
