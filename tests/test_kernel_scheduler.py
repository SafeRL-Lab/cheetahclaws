"""Tests for kernel.scheduler (RFC 0007)."""
from __future__ import annotations

import threading
import time

import pytest

from cheetahclaws.kernel import (
    AgentState,
    KernelStore,
    LedgerStore,
    ReadyEntry,
    SCHED_EXIT_KINDS,
    SchedIllegalTransition,
    SchedInvalidPayload,
    SchedUnknownId,
    SchedulerStore,
    ScheduleSpec,
    UnknownPid,
)


@pytest.fixture
def stores(tmp_path):
    """Open kernel + scheduler (and ledger for admission tests) sharing
    one connection + lock."""
    ks = KernelStore.open(tmp_path / "kernel.db")
    sched = SchedulerStore(ks.connection, write_lock=ks.write_lock)
    led   = LedgerStore(ks.connection,   write_lock=ks.write_lock)
    yield ks, sched, led
    ks.close()


def _spec(pid: int, **kw) -> ScheduleSpec:
    return ScheduleSpec(pid=pid, **kw)


# ── enqueue validation ────────────────────────────────────────────────────


def test_enqueue_basic_round_trip(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sid = sched.enqueue(_spec(a.pid, priority=3, trigger="manual",
                              payload={"step": 1}))
    e = sched.get(sid)
    assert isinstance(e, ReadyEntry)
    assert e.pid == a.pid
    assert e.priority == 3
    assert e.state == "queued"
    assert e.payload == {"step": 1}


def test_enqueue_unknown_pid(stores):
    _, sched, _ = stores
    with pytest.raises(UnknownPid):
        sched.enqueue(_spec(9999))


def test_enqueue_dead_agent_rejected(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    ks.terminate(a.pid, exit_kind="completed")
    with pytest.raises(SchedInvalidPayload):
        sched.enqueue(_spec(a.pid))


def test_enqueue_negative_runnable_at_rejected(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    with pytest.raises(SchedInvalidPayload):
        sched.enqueue(_spec(a.pid, runnable_at=-1))


def test_enqueue_deadline_before_runnable_rejected(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    with pytest.raises(SchedInvalidPayload):
        sched.enqueue(_spec(a.pid, runnable_at=100, deadline=50))


def test_enqueue_empty_trigger_rejected(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    with pytest.raises(SchedInvalidPayload):
        sched.enqueue(_spec(a.pid, trigger=""))


# ── claim ─────────────────────────────────────────────────────────────────


def test_claim_returns_oldest_when_priorities_equal(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    s1 = sched.enqueue(_spec(a.pid))
    s2 = sched.enqueue(_spec(a.pid))
    s3 = sched.enqueue(_spec(a.pid))
    claimed = sched.claim(worker_id="w-1", max_n=10)
    assert [e.sched_id for e in claimed] == [s1, s2, s3]


def test_claim_respects_priority(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    low  = sched.enqueue(_spec(a.pid, priority=1))
    hi   = sched.enqueue(_spec(a.pid, priority=10))
    mid  = sched.enqueue(_spec(a.pid, priority=5))
    claimed = sched.claim(worker_id="w-1", max_n=3)
    assert [e.sched_id for e in claimed] == [hi, mid, low]


def test_claim_marks_dispatched_and_records_worker(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sid = sched.enqueue(_spec(a.pid))
    [entry] = sched.claim(worker_id="w-7")
    assert entry.state == "dispatched"
    assert entry.worker_id == "w-7"
    assert entry.dispatched_at is not None


def test_claim_skips_future_runnable_at(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    future = time.time() + 3600
    sched.enqueue(_spec(a.pid, runnable_at=future))
    claimed = sched.claim(worker_id="w-1", max_n=10)
    assert claimed == []


def test_claim_at_runnable_at_succeeds(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sched.enqueue(_spec(a.pid, runnable_at=100.0))
    # Pretend "now" is after runnable_at.
    [_] = sched.claim(worker_id="w-1", max_n=1, now=200.0)


def test_claim_skips_past_deadline(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sched.enqueue(_spec(a.pid, runnable_at=0, deadline=50))
    # "now" is past the deadline — entry must not be claimable.
    assert sched.claim(worker_id="w-1", now=100.0) == []


def test_claim_max_n_limits(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    for _ in range(5):
        sched.enqueue(_spec(a.pid))
    claimed = sched.claim(worker_id="w-1", max_n=2)
    assert len(claimed) == 2


def test_claim_invalid_worker_id(stores):
    _, sched, _ = stores
    with pytest.raises(SchedInvalidPayload):
        sched.claim(worker_id="", max_n=1)


def test_claim_invalid_max_n(stores):
    _, sched, _ = stores
    with pytest.raises(SchedInvalidPayload):
        sched.claim(worker_id="w-1", max_n=0)


# ── concurrent claim atomicity ───────────────────────────────────────────


def test_concurrent_claims_no_duplicates(stores):
    """N entries, M workers each claiming in a loop until empty.
    Collectively they must claim each entry exactly once."""
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    ENQ = 200
    expected = {sched.enqueue(_spec(a.pid)) for _ in range(ENQ)}

    seen: set = set()
    seen_lock = threading.Lock()
    errors: list = []

    def worker(wid: str):
        try:
            while True:
                got = sched.claim(worker_id=wid, max_n=3)
                if not got:
                    return
                with seen_lock:
                    for e in got:
                        if e.sched_id in seen:
                            errors.append(f"duplicate {e.sched_id} from {wid}")
                        seen.add(e.sched_id)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"w-{i}",))
               for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == [], errors
    assert seen == expected


# ── complete / cancel / state machine ────────────────────────────────────


def test_complete_round_trip(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sid = sched.enqueue(_spec(a.pid))
    sched.claim(worker_id="w-1")
    prev, new = sched.complete(sid, exit_kind="completed")
    assert (prev, new) == ("dispatched", "completed")
    e = sched.get(sid)
    assert e.state == "completed"
    assert e.exit_kind == "completed"


def test_complete_on_queued_rejected(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sid = sched.enqueue(_spec(a.pid))
    with pytest.raises(SchedIllegalTransition) as e:
        sched.complete(sid, exit_kind="completed")
    assert e.value.prev_state == "queued"
    assert e.value.op == "complete"


def test_complete_invalid_exit_kind(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sid = sched.enqueue(_spec(a.pid))
    sched.claim(worker_id="w-1")
    with pytest.raises(SchedInvalidPayload):
        sched.complete(sid, exit_kind="bogus")


def test_complete_unknown_id(stores):
    _, sched, _ = stores
    with pytest.raises(SchedUnknownId):
        sched.complete(9999, exit_kind="completed")


def test_cancel_only_from_queued(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sid = sched.enqueue(_spec(a.pid))
    sched.cancel(sid)
    assert sched.get(sid).state == "cancelled"


def test_cancel_dispatched_rejected(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sid = sched.enqueue(_spec(a.pid))
    sched.claim(worker_id="w-1")
    with pytest.raises(SchedIllegalTransition):
        sched.cancel(sid)


def test_cancelled_invisible_to_claim(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    s1 = sched.enqueue(_spec(a.pid))
    s2 = sched.enqueue(_spec(a.pid))
    sched.cancel(s1)
    [e] = sched.claim(worker_id="w-1", max_n=10)
    assert e.sched_id == s2


# ── gc_expired ────────────────────────────────────────────────────────────


def test_gc_expired_sweeps_past_deadline(stores):
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    s_pass = sched.enqueue(_spec(a.pid))                                # no deadline
    s_now  = sched.enqueue(_spec(a.pid, runnable_at=0, deadline=100))   # past
    s_fut  = sched.enqueue(_spec(a.pid, runnable_at=0, deadline=10**12))# future
    swept = sched.gc_expired(now=200.0)
    assert swept == 1
    assert sched.get(s_pass).state == "queued"
    assert sched.get(s_now).state  == "expired"
    assert sched.get(s_fut).state  == "queued"


def test_gc_expired_only_sweeps_queued(stores):
    """A dispatched entry whose deadline passes is the supervisor's
    problem; gc_expired only touches queued."""
    ks, sched, _ = stores
    a = ks.create(name="x", template="t")
    sid = sched.enqueue(_spec(a.pid, runnable_at=0, deadline=100))
    sched.claim(worker_id="w-1", now=50.0)  # claim while still alive
    swept = sched.gc_expired(now=200.0)
    assert swept == 0
    assert sched.get(sid).state == "dispatched"


# ── ledger admission filter ───────────────────────────────────────────────


def test_admission_skips_over_limit_agents(stores):
    ks, sched, led = stores
    a1 = ks.create(name="ok",  template="t")
    a2 = ks.create(name="bad", template="t")
    led.create(pid=a1.pid, grants={"tokens": 1000})
    led.create(pid=a2.pid, grants={"tokens": 1000})
    led.charge(pid=a2.pid, dim="tokens", amount=2000)   # over
    s_ok  = sched.enqueue(_spec(a1.pid))
    s_bad = sched.enqueue(_spec(a2.pid))
    [entry] = sched.claim(worker_id="w-1", max_n=10)
    assert entry.sched_id == s_ok                         # bad skipped
    # The bad entry stays queued, untouched.
    assert sched.get(s_bad).state == "queued"


def test_admission_disabled_picks_up_over_limit(stores):
    ks, sched, led = stores
    a = ks.create(name="x", template="t")
    led.create(pid=a.pid, grants={"tokens": 100})
    led.charge(pid=a.pid, dim="tokens", amount=200)   # over
    sid = sched.enqueue(_spec(a.pid))
    claimed = sched.claim(worker_id="w-1", admission_check=False)
    assert len(claimed) == 1
    assert claimed[0].sched_id == sid


def test_admission_recovers_after_grant_update(stores):
    """Operator update_grant after the agent breached should let the
    next claim pick the entry up."""
    ks, sched, led = stores
    a = ks.create(name="x", template="t")
    led.create(pid=a.pid, grants={"tokens": 100})
    led.charge(pid=a.pid, dim="tokens", amount=200)
    sid = sched.enqueue(_spec(a.pid))
    assert sched.claim(worker_id="w-1") == []
    led.update_grant(pid=a.pid, dim="tokens", new_grant=1000)
    [e] = sched.claim(worker_id="w-1")
    assert e.sched_id == sid


# ── list ─────────────────────────────────────────────────────────────────


def test_list_filters_by_state_and_pid(stores):
    ks, sched, _ = stores
    a1 = ks.create(name="a1", template="t")
    a2 = ks.create(name="a2", template="t")
    sched.enqueue(_spec(a1.pid))
    sched.enqueue(_spec(a2.pid))
    sid = sched.enqueue(_spec(a1.pid))
    sched.cancel(sid)
    queued, total = sched.list(state="queued")
    assert total == 2
    a1_only, _ = sched.list(pid=a1.pid)
    assert all(e.pid == a1.pid for e in a1_only)


def test_list_invalid_state(stores):
    _, sched, _ = stores
    with pytest.raises(SchedInvalidPayload):
        sched.list(state="bogus")
