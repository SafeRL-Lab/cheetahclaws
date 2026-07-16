# Design Note: Bridge Mirror — bridges ↔ kernel.mbox glue

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0009-agent-mailbox.md`](./0009-agent-mailbox.md), [`0010-agent-registry.md`](./0010-agent-registry.md)

This RFC defines how existing bridges (Telegram, WeChat, Slack, …)
expose their inbound / outbound message streams through the kernel
mailbox primitive. It does **not** touch any bridge code under
`bridges/`. The deliverable is a small helper module
(`kernel/bridge_mirror.py`) that:

1. Defines a **canonical topic naming scheme** so any agent in the
   system can subscribe to bridge traffic without knowing which
   bridge wrote which row.
2. Defines a **canonical message shape** (`BridgeMessage`) so
   subscribers don't need to parse N different formats.
3. Provides **publish + drain helpers** so a future, optional patch
   to a bridge module can mirror its messages with two function
   calls — without rewriting the bridge.

The existing `RuntimeContext` callbacks in
`bridges/{telegram,wechat,slack}.py` keep working unchanged. A bridge
can adopt the mirror at its leisure; an agent can subscribe to the
canonical topic with no awareness of whether the mirror is active.

## 1. Goals & non-goals

**Goals:**

1. **One agent, all bridges.** A research agent that wants every
   inbound chat message subscribes to `bridge.*.inbound` (well —
   subscribes per-bridge in v1; topic-glob matching is a future
   RFC) and gets messages from all bridges in one inbox.
2. **One bridge, many subscribers.** Telegram inbound goes to one
   topic; the REPL, the research agent, and the audit logger all
   subscribe; each gets a copy.
3. **Outbound queue.** A bridge worker subscribes to
   `bridge.<kind>.outbound`; any agent that wants to send a message
   publishes to the same topic. The bridge worker drains and emits.
4. **No required bridge changes.** The mirror is a library used
   from outside the bridge or via a small voluntary patch inside
   the bridge.
5. **Backwards compatible message shape.** The canonical
   `BridgeMessage` fields are the lowest-common-denominator across
   the existing bridges; specific bridges' richer fields go in
   `metadata`.

**Non-goals (v1):**

- **Routing rules.** No filtering ("only forward if the chat_id
  matches X"); subscribers do that themselves.
- **Acknowledgement / retries.** Bridges' delivery semantics vary
  too much. The mirror just enqueues; the bridge worker's send
  callback owns retry policy.
- **Encryption.** Bridge payloads ride in the kernel mailbox
  cleartext (same trust boundary as the kernel itself).
- **Cross-bridge orchestration.** A user "send the same message to
  all my bridges" is a layer on top — write a small fan-in helper
  using publish-to-each-topic.

## 2. Topic naming

```
bridge.<kind>.inbound       — messages received from the world
bridge.<kind>.outbound      — messages queued for the world
```

Where `<kind>` is a lowercase bridge identifier. The four shipped
helpers cover:

| Constant | Value |
|---|---|
| `BridgeKind.TELEGRAM` | `"telegram"` |
| `BridgeKind.WECHAT` | `"wechat"` |
| `BridgeKind.SLACK` | `"slack"` |
| `BridgeKind.DISCORD` | `"discord"` |

Custom kinds (`"matrix"`, `"signal"`, …) are valid as long as they
match `^[a-z][a-z0-9_-]*$`. Validation rejects upper-case, dots,
and other reserved characters.

A future RFC may add `bridge.<kind>.<chat_id>.inbound` for
per-channel topics. For v1, all messages from a bridge land on one
topic; subscribers filter by `payload.sender` if needed.

## 3. Message shape

```python
@dataclass(frozen=True)
class BridgeMessage:
    kind:      str          # "telegram" | "wechat" | "slack" | …
    sender:    str          # opaque sender id (chat_id, user_id, …)
    text:      str
    direction: str          # "inbound" | "outbound"
    metadata:  dict         # bridge-specific extras
    ts:        float        # epoch seconds
```

When sent on `kernel.mbox`:
- `kind` is `"bridge.message"` (the kernel mailbox `kind` field).
- `payload` is the BridgeMessage as a dict (`.to_dict()`).
- `topic` is `bridge.<kind>.<direction>`.

This keeps `kernel.mbox.recv` consumers uniform: they peek at
`m.kind` to know it's a bridge message, and pull the bridge-specific
fields from `m.payload`.

## 4. API

```python
class BridgeMirror:
    def __init__(self, kernel: Kernel, *,
                 sender_pid: int | None = None): ...

    def mirror_inbound(self, *,
                       kind: str, sender: str, text: str,
                       metadata: dict | None = None,
                       ts: float | None = None) -> dict:
        """Publish to bridge.<kind>.inbound. Returns the dict
        returned by kernel.mbox.publish (delivered/rejected/msg_ids)."""

    def queue_outbound(self, *,
                       kind: str, recipient: str, text: str,
                       metadata: dict | None = None,
                       ts: float | None = None) -> dict:
        """Publish to bridge.<kind>.outbound."""

    def subscribe_inbound(self, agent_pid: int, kind: str) -> None:
        """Sugar for kernel.mbox.subscribe(pid, inbound_topic(kind))."""

    def subscribe_outbound(self, agent_pid: int, kind: str) -> None: ...


class OutboundReceiver:
    """Background drainer for outbound bridge messages.

    The agent at ``agent_pid`` must have a mailbox and be subscribed
    to ``bridge.<kind>.outbound`` (use mirror.subscribe_outbound).
    The receiver thread polls the mailbox, decodes the BridgeMessage,
    and calls ``send_fn(message)`` for each.
    """
    def __init__(
        self, kernel: Kernel, *,
        agent_pid: int, kind: str,
        send_fn: Callable[[BridgeMessage], None],
        poll_interval_s: float = 1.0,
        batch_size: int = 32,
    ): ...

    def drain_once(self) -> int: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

## 5. Topology example

```
                    ┌─────────────┐
                    │  Telegram   │
                    │  bridge     │   bridges/telegram.py
                    │  (existing) │
                    └─┬─────────┬─┘
                      │         │
                inbound│         │outbound
                      ▼         ▲
                ┌─────────────────────────┐
        publish │                         │  drain
        ───────►│  bridge.telegram.       │◄───────
                │  inbound                │
                │  outbound               │
                └─┬─────────────────────┬─┘
                  │                     │
        subscribers                     publishers
                  │                     │
        ┌─────────┴────┐         ┌──────┴────┐
        │  research    │         │   any     │
        │  agent       │         │   agent   │
        └──────────────┘         └───────────┘
```

A bridge wrapper for Telegram would:

1. Create a "bridge gateway" agent (or reuse one), give it a
   mailbox.
2. Subscribe to `bridge.telegram.outbound`.
3. Start an `OutboundReceiver` whose `send_fn` calls the existing
   bridge's actual send routine.
4. On receiving a message from Telegram, call
   `mirror.mirror_inbound(kind="telegram", sender=chat_id,
   text=msg)`.

## 6. Backwards compatibility

- Bridge code in `bridges/` is **not** touched by this RFC.
- The `kernel/bridge_mirror.py` module is purely additive.
- Existing tests for bridges keep passing (they don't use the
  mirror).
- The mirror requires only the existing `kernel.mbox` primitives;
  no schema bump, no new RPC method.

## 7. Open questions

1. **Glob subscriptions.** A subscriber that wants every bridge's
   inbound has to subscribe four times today
   (`bridge.telegram.inbound`, `bridge.wechat.inbound`, …).
   Adding a topic glob like `bridge.*.inbound` is a kernel.mbox
   feature, not a mirror feature. Out of scope here.
2. **Sender identity.** v1 stores `sender` as an opaque string. A
   future RFC may map sender → registered agent (via the
   AgentRegistry) so messages have proper attribution.
3. **Inbound message ordering across bridges.** Each bridge writes
   its own topic. If a research agent subscribes to all four, the
   relative order of messages from different bridges is FIFO within
   the recipient's mailbox (since the kernel uses
   `INTEGER PRIMARY KEY AUTOINCREMENT`), which is good enough.

## 8. Acceptance criteria

A PR claiming this RFC must:

1. `mirror_inbound(kind, sender, text)` produces one mailbox
   message per subscriber to `bridge.<kind>.inbound` with the
   correct `BridgeMessage` payload.
2. Multi-subscriber fan-out: 3 agents subscribed → 3 message rows
   on a single publish.
3. `queue_outbound` round-trips through `kernel.mbox.publish` to
   any subscriber on `bridge.<kind>.outbound`.
4. `OutboundReceiver`: starts a background thread, drains messages
   into the supplied `send_fn`, stops cleanly via `stop()`, doesn't
   leak if `start()` is never called.
5. Custom (non-`BridgeKind.KNOWN`) kinds work, e.g. `"matrix"`.
6. Validation rejects upper-case kinds and reserved characters.
7. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
