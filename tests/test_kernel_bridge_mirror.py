"""Tests for kernel.bridge_mirror (RFC 0018)."""
from __future__ import annotations

import threading
import time

import pytest

from cheetahclaws.kernel import (
    BridgeKind,
    BridgeMessage,
    BridgeMirror,
    Kernel,
    OutboundReceiver,
    inbound_topic,
    outbound_topic,
)
from cheetahclaws.kernel.bridge_mirror import MESSAGE_KIND


@pytest.fixture
def kernel(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        yield k


def _make_subscriber(kernel: Kernel, *, name: str, kind: str,
                      direction: str = "inbound") -> int:
    """Create an agent + mailbox, subscribe to the bridge topic, return pid."""
    a = kernel.create_agent(name=name, template="bridge-test")
    kernel.mailbox.create(pid=a.pid, queue_size=100)
    topic = (inbound_topic(kind) if direction == "inbound"
             else outbound_topic(kind))
    kernel.mailbox.subscribe(a.pid, topic)
    return a.pid


# ── topic helpers ───────────────────────────────────────────────────────


def test_inbound_topic_format():
    assert inbound_topic("telegram") == "bridge.telegram.inbound"
    assert inbound_topic("custom-kind") == "bridge.custom-kind.inbound"


def test_outbound_topic_format():
    assert outbound_topic("slack") == "bridge.slack.outbound"


@pytest.mark.parametrize("bad_kind", [
    "", "Telegram", "TELEGRAM", "tele.gram", "tele gram", "tele/gram",
    "1telegram",
])
def test_invalid_kind_rejected(bad_kind):
    with pytest.raises(ValueError):
        inbound_topic(bad_kind)
    with pytest.raises(ValueError):
        outbound_topic(bad_kind)


def test_known_kinds_constants():
    assert BridgeKind.TELEGRAM == "telegram"
    assert BridgeKind.WECHAT   == "wechat"
    assert BridgeKind.SLACK    == "slack"
    assert BridgeKind.DISCORD  == "discord"
    assert BridgeKind.TELEGRAM in BridgeKind.KNOWN


# ── BridgeMessage round-trip ────────────────────────────────────────────


def test_bridge_message_to_dict_round_trip():
    msg = BridgeMessage(
        kind="telegram", sender="123", text="hi",
        direction="inbound", metadata={"msg_id": 99}, ts=1234.5,
    )
    d = msg.to_dict()
    assert d == {
        "kind": "telegram", "sender": "123", "text": "hi",
        "direction": "inbound", "metadata": {"msg_id": 99},
        "ts": 1234.5,
    }
    msg2 = BridgeMessage.from_payload(d)
    assert msg2.kind == "telegram"
    assert msg2.text == "hi"
    assert msg2.metadata == {"msg_id": 99}


def test_bridge_message_from_payload_handles_missing():
    """Defensive: bad payload doesn't crash."""
    msg = BridgeMessage.from_payload({})
    assert msg.kind == ""
    assert msg.metadata == {}


# ── mirror_inbound ──────────────────────────────────────────────────────


def test_mirror_inbound_publishes_to_inbound_topic(kernel):
    pid = _make_subscriber(kernel, name="rx", kind="telegram",
                            direction="inbound")
    mirror = BridgeMirror(kernel)
    result = mirror.mirror_inbound(
        kind="telegram", sender="chat:1", text="hello",
        metadata={"is_bot": False},
    )
    assert result["delivered"] == 1
    msgs = kernel.mailbox.recv(pid=pid)
    assert len(msgs) == 1
    assert msgs[0].kind == MESSAGE_KIND
    payload = msgs[0].payload
    assert payload["kind"] == "telegram"
    assert payload["sender"] == "chat:1"
    assert payload["text"] == "hello"
    assert payload["direction"] == "inbound"
    assert payload["metadata"] == {"is_bot": False}
    assert payload["ts"] > 0


def test_mirror_inbound_fan_out_three_subscribers(kernel):
    pids = [
        _make_subscriber(kernel, name=f"s{i}", kind="telegram")
        for i in range(3)
    ]
    nobody = kernel.create_agent(name="nobody", template="t")
    kernel.mailbox.create(pid=nobody.pid)
    # nobody doesn't subscribe.

    mirror = BridgeMirror(kernel)
    result = mirror.mirror_inbound(
        kind="telegram", sender="x", text="ping",
    )
    assert result["delivered"] == 3
    for pid in pids:
        msgs = kernel.mailbox.recv(pid=pid)
        assert len(msgs) == 1
        assert msgs[0].payload["text"] == "ping"
    assert kernel.mailbox.recv(pid=nobody.pid) == []


def test_mirror_inbound_no_subscribers_yields_zero(kernel):
    mirror = BridgeMirror(kernel)
    result = mirror.mirror_inbound(
        kind="discord", sender="x", text="lonely",
    )
    assert result["delivered"] == 0
    assert result["rejected"]  == 0
    assert result["msg_ids"]   == []


def test_mirror_inbound_records_sender_pid(kernel):
    """When BridgeMirror is constructed with sender_pid, every
    publish carries it."""
    sender = kernel.create_agent(name="s", template="t")
    receiver = _make_subscriber(kernel, name="r", kind="telegram")
    mirror = BridgeMirror(kernel, sender_pid=sender.pid)
    mirror.mirror_inbound(kind="telegram", sender="x", text="hi")
    msgs = kernel.mailbox.recv(pid=receiver)
    assert msgs[0].sender_pid == sender.pid


def test_mirror_inbound_custom_kind_works(kernel):
    pid = _make_subscriber(kernel, name="r", kind="matrix")
    mirror = BridgeMirror(kernel)
    result = mirror.mirror_inbound(
        kind="matrix", sender="@user:example.org", text="hi",
    )
    assert result["delivered"] == 1
    assert kernel.mailbox.recv(pid=pid)[0].payload["kind"] == "matrix"


def test_mirror_inbound_invalid_kind_raises(kernel):
    mirror = BridgeMirror(kernel)
    with pytest.raises(ValueError):
        mirror.mirror_inbound(kind="BadCase", sender="x", text="x")


# ── queue_outbound ──────────────────────────────────────────────────────


def test_queue_outbound_publishes_to_outbound_topic(kernel):
    pid = _make_subscriber(kernel, name="bridge_worker",
                            kind="telegram", direction="outbound")
    mirror = BridgeMirror(kernel)
    result = mirror.queue_outbound(
        kind="telegram", recipient="chat:7",
        text="bye", metadata={"reply_to": 99},
    )
    assert result["delivered"] == 1
    msgs = kernel.mailbox.recv(pid=pid)
    p = msgs[0].payload
    assert p["direction"] == "outbound"
    assert p["sender"] == "chat:7"        # recipient stored in 'sender' field
    assert p["metadata"] == {"reply_to": 99}


# ── subscribe_* sugar ──────────────────────────────────────────────────


def test_subscribe_inbound_sugar(kernel):
    a = kernel.create_agent(name="x", template="t")
    kernel.mailbox.create(pid=a.pid)
    mirror = BridgeMirror(kernel)
    mirror.subscribe_inbound(a.pid, "wechat")
    topics = kernel.mailbox.list_subscriptions(a.pid)
    assert "bridge.wechat.inbound" in topics


def test_subscribe_outbound_sugar(kernel):
    a = kernel.create_agent(name="x", template="t")
    kernel.mailbox.create(pid=a.pid)
    mirror = BridgeMirror(kernel)
    mirror.subscribe_outbound(a.pid, "slack")
    topics = kernel.mailbox.list_subscriptions(a.pid)
    assert "bridge.slack.outbound" in topics


# ── OutboundReceiver: drain_once ───────────────────────────────────────


def test_receiver_drain_once_calls_send_fn(kernel):
    pid = _make_subscriber(kernel, name="bridge_worker",
                            kind="telegram", direction="outbound")
    mirror = BridgeMirror(kernel)
    mirror.queue_outbound(
        kind="telegram", recipient="alice", text="hi",
    )
    mirror.queue_outbound(
        kind="telegram", recipient="bob", text="bye",
    )

    sent: list[BridgeMessage] = []
    rx = OutboundReceiver(
        kernel, agent_pid=pid, kind="telegram",
        send_fn=lambda m: sent.append(m),
    )
    drained = rx.drain_once()
    assert drained == 2
    recipients = sorted(m.sender for m in sent)
    assert recipients == ["alice", "bob"]
    # Cursor advanced — second drain returns 0.
    assert rx.drain_once() == 0


def test_receiver_drain_skips_non_bridge_messages(kernel):
    """If something publishes a non-bridge.message kind to the same
    inbox (shouldn't happen but…), drain_once skips it."""
    pid = _make_subscriber(kernel, name="r",
                            kind="telegram", direction="outbound")
    # Bypass the mirror and send a bogus-kind message directly.
    kernel.mailbox.publish(
        sender_pid=None, topic="bridge.telegram.outbound",
        kind="not.bridge.message", payload={"hi": True},
    )
    sent: list[BridgeMessage] = []
    rx = OutboundReceiver(
        kernel, agent_pid=pid, kind="telegram",
        send_fn=lambda m: sent.append(m),
    )
    rx.drain_once()
    assert sent == []


def test_receiver_drain_skips_inbound_direction(kernel):
    """An inbound bridge message shouldn't be drained as outbound
    even if the same agent happens to be subscribed to both."""
    pid = _make_subscriber(kernel, name="r",
                            kind="telegram", direction="outbound")
    BridgeMirror(kernel).subscribe_inbound(pid, "telegram")
    BridgeMirror(kernel).mirror_inbound(
        kind="telegram", sender="x", text="not for you",
    )
    sent: list[BridgeMessage] = []
    rx = OutboundReceiver(
        kernel, agent_pid=pid, kind="telegram",
        send_fn=lambda m: sent.append(m),
    )
    rx.drain_once()
    assert sent == []


def test_receiver_send_fn_error_is_isolated(kernel):
    """One failing send_fn shouldn't kill the drainer."""
    pid = _make_subscriber(kernel, name="r",
                            kind="telegram", direction="outbound")
    mirror = BridgeMirror(kernel)
    mirror.queue_outbound(
        kind="telegram", recipient="a", text="will-fail",
    )
    mirror.queue_outbound(
        kind="telegram", recipient="b", text="will-succeed",
    )

    successes: list[str] = []
    errors: list[tuple] = []

    def flaky_send(m: BridgeMessage):
        if m.text == "will-fail":
            raise RuntimeError("boom")
        successes.append(m.sender)

    rx = OutboundReceiver(
        kernel, agent_pid=pid, kind="telegram",
        send_fn=flaky_send,
        on_error=lambda m, e: errors.append((m.sender, type(e).__name__)),
    )
    rx.drain_once()
    assert successes == ["b"]
    assert errors == [("a", "RuntimeError")]


# ── OutboundReceiver: start/stop ───────────────────────────────────────


def test_receiver_start_drains_in_background(kernel):
    pid = _make_subscriber(kernel, name="r",
                            kind="telegram", direction="outbound")
    sent: list[BridgeMessage] = []
    rx = OutboundReceiver(
        kernel, agent_pid=pid, kind="telegram",
        send_fn=lambda m: sent.append(m),
        poll_interval_s=0.05,
    )
    rx.start()
    try:
        BridgeMirror(kernel).queue_outbound(
            kind="telegram", recipient="a", text="hi",
        )
        # Wait for the background drain.
        deadline = time.monotonic() + 5
        while not sent and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(sent) == 1
    finally:
        rx.stop()


def test_receiver_stop_is_graceful(kernel):
    pid = _make_subscriber(kernel, name="r",
                            kind="telegram", direction="outbound")
    rx = OutboundReceiver(
        kernel, agent_pid=pid, kind="telegram",
        send_fn=lambda m: None, poll_interval_s=0.1,
    )
    rx.start()
    rx.stop()
    # No assertion needed — just that stop returns within a few seconds.


def test_receiver_start_idempotent(kernel):
    pid = _make_subscriber(kernel, name="r",
                            kind="telegram", direction="outbound")
    rx = OutboundReceiver(
        kernel, agent_pid=pid, kind="telegram",
        send_fn=lambda m: None,
    )
    rx.start()
    rx.start()  # No-op
    rx.stop()


def test_receiver_stop_without_start_is_safe(kernel):
    pid = _make_subscriber(kernel, name="r",
                            kind="telegram", direction="outbound")
    rx = OutboundReceiver(
        kernel, agent_pid=pid, kind="telegram",
        send_fn=lambda m: None,
    )
    rx.stop()  # No error.


# ── Validation ─────────────────────────────────────────────────────────


def test_receiver_rejects_invalid_kind(kernel):
    pid = kernel.create_agent(name="r", template="t").pid
    with pytest.raises(ValueError):
        OutboundReceiver(
            kernel, agent_pid=pid, kind="BAD",
            send_fn=lambda m: None,
        )


def test_receiver_rejects_non_callable_send_fn(kernel):
    pid = kernel.create_agent(name="r", template="t").pid
    with pytest.raises(ValueError):
        OutboundReceiver(
            kernel, agent_pid=pid, kind="telegram",
            send_fn="not callable",  # type: ignore[arg-type]
        )


# ── End-to-end: inbound publish + outbound drain ──────────────────────


def test_full_round_trip(kernel):
    """Simulate: bridge receives a message, mirror publishes inbound;
    a researcher agent reads it; researcher emits a reply via
    queue_outbound; bridge worker drains and 'sends'."""
    # Researcher subscribes to inbound.
    researcher = kernel.create_agent(name="researcher", template="t")
    kernel.mailbox.create(pid=researcher.pid)
    BridgeMirror(kernel).subscribe_inbound(researcher.pid, "telegram")

    # Bridge worker subscribes to outbound.
    bridge_worker = kernel.create_agent(name="bridge_worker", template="t")
    kernel.mailbox.create(pid=bridge_worker.pid)
    BridgeMirror(kernel).subscribe_outbound(bridge_worker.pid, "telegram")

    # 1. World sends to bridge → mirror publishes inbound.
    BridgeMirror(kernel).mirror_inbound(
        kind="telegram", sender="alice", text="what's the weather",
    )

    # 2. Researcher reads inbound.
    inbound = kernel.mailbox.recv(pid=researcher.pid)
    assert len(inbound) == 1
    bridge_msg = BridgeMessage.from_payload(inbound[0].payload)
    assert bridge_msg.text == "what's the weather"
    assert bridge_msg.direction == "inbound"

    # 3. Researcher decides to reply.
    BridgeMirror(kernel).queue_outbound(
        kind="telegram", recipient=bridge_msg.sender,
        text="cloudy with a chance of LLM",
    )

    # 4. Bridge worker drains.
    sent: list[BridgeMessage] = []
    rx = OutboundReceiver(
        kernel, agent_pid=bridge_worker.pid, kind="telegram",
        send_fn=lambda m: sent.append(m),
    )
    rx.drain_once()
    assert len(sent) == 1
    assert sent[0].sender == "alice"
    assert sent[0].text == "cloudy with a chance of LLM"
