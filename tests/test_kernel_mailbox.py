"""Tests for kernel.mailbox (RFC 0009)."""
from __future__ import annotations

import threading
import time

import pytest

from cheetahclaws.kernel import (
    KernelStore,
    Mailbox,
    MailboxAlreadyExists,
    MailboxFull,
    MailboxInvalidPayload,
    MailboxNotFound,
    MailboxStore,
    MailboxSubscriptionMissing,
    Message,
    UnknownPid,
)


@pytest.fixture
def stores(tmp_path):
    ks = KernelStore.open(tmp_path / "kernel.db")
    mb = MailboxStore(ks.connection, write_lock=ks.write_lock)
    yield ks, mb
    ks.close()


# ── create / get / delete ────────────────────────────────────────────────


def test_create_round_trip(stores):
    ks, mb = stores
    a = ks.create(name="x", template="t")
    box = mb.create(pid=a.pid, queue_size=4, retention_s=60)
    assert isinstance(box, Mailbox)
    assert box.pid == a.pid
    assert box.queue_size == 4
    again = mb.get(a.pid)
    assert again.queue_size == 4


def test_create_unknown_pid(stores):
    _, mb = stores
    with pytest.raises(UnknownPid):
        mb.create(pid=9999)


def test_create_duplicate_raises(stores):
    ks, mb = stores
    a = ks.create(name="x", template="t")
    mb.create(pid=a.pid)
    with pytest.raises(MailboxAlreadyExists):
        mb.create(pid=a.pid)


def test_get_unknown_raises(stores):
    _, mb = stores
    with pytest.raises(MailboxNotFound):
        mb.get(9999)


def test_delete_purges_messages_and_subs(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    mb.create(pid=a.pid)
    mb.create(pid=b.pid)
    mb.subscribe(a.pid, "topic.x")
    mb.send(sender_pid=b.pid, recipient_pid=a.pid, kind="k", payload={})
    mb.send(sender_pid=b.pid, recipient_pid=a.pid, kind="k", payload={})
    purged = mb.delete(a.pid)
    assert purged == 2
    with pytest.raises(MailboxNotFound):
        mb.get(a.pid)
    assert mb.list_subscriptions(a.pid) == []


# ── subscribe / unsubscribe ──────────────────────────────────────────────


def test_subscribe_idempotent(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid)
    mb.subscribe(a.pid, "topic.x")
    mb.subscribe(a.pid, "topic.x")  # no-op
    assert mb.list_subscriptions(a.pid) == ["topic.x"]


def test_subscribe_requires_mailbox(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    with pytest.raises(MailboxNotFound):
        mb.subscribe(a.pid, "t")


def test_unsubscribe_missing_raises(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid)
    with pytest.raises(MailboxSubscriptionMissing):
        mb.unsubscribe(a.pid, "never.subscribed")


# ── send (direct) ─────────────────────────────────────────────────────────


def test_send_round_trip(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    mb.create(pid=b.pid)
    mid = mb.send(sender_pid=a.pid, recipient_pid=b.pid,
                  kind="hello", payload={"msg": "hi"})
    msgs = mb.recv(pid=b.pid)
    assert len(msgs) == 1
    m = msgs[0]
    assert m.msg_id == mid
    assert m.sender_pid == a.pid
    assert m.recipient_pid == b.pid
    assert m.kind == "hello"
    assert m.payload == {"msg": "hi"}
    assert m.delivered_at is not None  # mark_delivered default


def test_send_to_unknown_mailbox(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    with pytest.raises(MailboxNotFound):
        mb.send(sender_pid=a.pid, recipient_pid=9999,
                kind="k", payload={})


def test_send_full_mailbox(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid, queue_size=2)
    mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})
    mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})
    with pytest.raises(MailboxFull):
        mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})


def test_send_after_recv_makes_room(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid, queue_size=1)
    mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})
    # Recipient drains.
    mb.recv(pid=a.pid)
    # Now there's room.
    mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})


def test_send_invalid_payload(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid)
    with pytest.raises(MailboxInvalidPayload):
        mb.send(sender_pid=None, recipient_pid=a.pid, kind="", payload={})
    with pytest.raises(MailboxInvalidPayload):
        mb.send(sender_pid=None, recipient_pid=a.pid, kind="k",
                payload="not a dict")  # type: ignore


# ── publish / fan-out ─────────────────────────────────────────────────────


def test_publish_fan_out(stores):
    ks, mb = stores
    a, b, c = (ks.create(name=n, template="t") for n in ("a", "b", "c"))
    for x in (a, b, c):
        mb.create(pid=x.pid)
    mb.subscribe(a.pid, "alerts")
    mb.subscribe(b.pid, "alerts")
    # c not subscribed
    pub = mb.publish(sender_pid=None, topic="alerts", kind="ping",
                     payload={"n": 1})
    assert pub["delivered"] == 2
    assert pub["rejected"]  == 0
    assert len(mb.recv(pid=a.pid)) == 1
    assert len(mb.recv(pid=b.pid)) == 1
    assert mb.recv(pid=c.pid) == []


def test_publish_partial_when_full(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    mb.create(pid=a.pid, queue_size=1)
    mb.create(pid=b.pid, queue_size=10)
    mb.subscribe(a.pid, "topic")
    mb.subscribe(b.pid, "topic")
    # Fill a's mailbox.
    mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})
    pub = mb.publish(sender_pid=None, topic="topic", kind="k",
                     payload={"x": 1})
    assert pub["delivered"] == 1   # only b received
    assert pub["rejected"]  == 1   # a was full
    assert len(pub["msg_ids"]) == 1


def test_publish_fail_on_full_aborts_all(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    mb.create(pid=a.pid, queue_size=1)
    mb.create(pid=b.pid, queue_size=10)
    mb.subscribe(a.pid, "topic")
    mb.subscribe(b.pid, "topic")
    mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})
    with pytest.raises(MailboxFull):
        mb.publish(sender_pid=None, topic="topic", kind="k",
                   payload={}, fail_on_full=True)
    # b's box must not have received anything (atomic abort).
    assert mb.recv(pid=b.pid) == []


def test_publish_unknown_topic_yields_zero_delivered(stores):
    ks, mb = stores
    pub = mb.publish(sender_pid=None, topic="nobody-listens",
                     kind="k", payload={})
    assert pub == {"delivered": 0, "rejected": 0, "msg_ids": []}


# ── recv / peek / cursor ─────────────────────────────────────────────────


def test_recv_advances_cursor(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid)
    for i in range(3):
        mb.send(sender_pid=None, recipient_pid=a.pid,
                kind="k", payload={"i": i})
    first = mb.recv(pid=a.pid, limit=2)
    assert len(first) == 2
    cursor = first[-1].msg_id
    rest = mb.recv(pid=a.pid, since_msg_id=cursor)
    assert len(rest) == 1


def test_peek_does_not_mark_delivered(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid)
    mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})
    msgs = mb.peek(pid=a.pid)
    assert len(msgs) == 1
    assert msgs[0].delivered_at is None
    # Still pending — recv next time will see it.
    msgs2 = mb.recv(pid=a.pid)
    assert len(msgs2) == 1


def test_recv_no_mark_keeps_pending(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid)
    mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})
    mb.recv(pid=a.pid, mark_delivered=False)
    again = mb.recv(pid=a.pid, mark_delivered=False)
    assert len(again) == 1


# ── TTL + retention gc ───────────────────────────────────────────────────


def test_recv_skips_expired(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid)
    past = time.time() - 1
    future = time.time() + 1000
    mb.send(sender_pid=None, recipient_pid=a.pid,
            kind="dead", payload={}, expires_at=past)
    mb.send(sender_pid=None, recipient_pid=a.pid,
            kind="alive", payload={}, expires_at=future)
    msgs = mb.recv(pid=a.pid)
    kinds = [m.kind for m in msgs]
    assert kinds == ["alive"]


def test_gc_expired_purges_past_ttl(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid)
    mb.send(sender_pid=None, recipient_pid=a.pid,
            kind="k", payload={}, expires_at=time.time() - 1)
    mb.send(sender_pid=None, recipient_pid=a.pid,
            kind="k", payload={}, expires_at=time.time() + 1000)
    purged = mb.gc_expired()
    assert purged == 1


def test_gc_purges_delivered_past_retention(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid, retention_s=0)   # immediate expiry on delivery
    mb.send(sender_pid=None, recipient_pid=a.pid, kind="k", payload={})
    mb.recv(pid=a.pid)  # delivers
    purged = mb.gc_expired(now=time.time() + 1)
    assert purged == 1


# ── concurrent send atomicity ────────────────────────────────────────────


def test_concurrent_sends_unique_msg_ids(stores):
    ks, mb = stores
    a = ks.create(name="a", template="t")
    mb.create(pid=a.pid, queue_size=1000)
    sent: list[int] = []
    sent_lock = threading.Lock()
    errors: list = []

    def worker(n):
        try:
            for _ in range(n):
                mid = mb.send(sender_pid=None, recipient_pid=a.pid,
                              kind="k", payload={})
                with sent_lock:
                    sent.append(mid)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(25,))
               for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    assert len(sent) == 100
    assert len(set(sent)) == 100
