# Design Note: AgentMailbox — direct messaging + topic pub/sub

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 3 — IPC)
- **Builds on:** [`0003-agent-process-and-event-log.md`](./0003-agent-process-and-event-log.md), [`0007-agent-scheduler.md`](./0007-agent-scheduler.md)
- **Sibling RFC:** [`0010-agent-registry.md`](./0010-agent-registry.md) (ships in same schema bump)

This RFC introduces the **inter-agent IPC primitive**. Today an agent
that wants to talk to another must go through its parent (fork-join via
`subagent.py`), or pass through a bridge (Telegram/WeChat/etc). The
mailbox is the missing direct path: an agent's pid has an inbox; other
agents enqueue typed messages; the recipient consumes on its own
schedule.

The kernel **stores and routes**; **execution** of messages is the
recipient's job (the supervisor or runtime drains the inbox during a
turn). The mailbox does not drive scheduling itself — but a message
arriving for a SUSPENDED agent is a signal the orchestrator can
listen for via the existing event bus.

This RFC ships **purely additive** code. Schema version bumps v3 → v4
together with RFC 0010 (Registry); the two land in one PR.

## 1. Goals & non-goals

**Goals:**

1. **Direct messages.** Agent A sends one typed message to agent B's
   pid; B receives it on next `recv`.
2. **Topic pub/sub.** Agents subscribe to topic strings; senders
   publish to a topic; the kernel fans out one row per subscriber. v1
   uses the **push model** (one row per subscriber per publish) for
   simplicity; future RFC may add pull/cursor model for high-fanout
   topics.
3. **Persistence.** Messages survive daemon restart. `recv` works
   correctly across reboots — the recipient's last-read offset is the
   client's responsibility (it tracks `since_msg_id`), the kernel
   stores everything in `agent_messages`.
4. **Bounded inbox.** Each mailbox has a `queue_size`. A `send` to a
   full mailbox is rejected (`MailboxFull`); the sender chooses retry
   policy.
5. **Retention + TTL.** Delivered messages purge after `retention_s`.
   Optional `expires_at` lets senders cap message lifetime regardless
   of retention.

**Non-goals (v1):**

- **Causal ordering across topics.** Messages within one
  recipient×topic are FIFO by `msg_id`; cross-topic ordering is not
  guaranteed.
- **Acknowledgments / at-least-once with retries.** A `recv` with
  `mark_delivered=True` records delivery; the kernel does not retry
  if the recipient crashes mid-handling. RFC 0007's scheduler can
  re-enqueue work if the supervisor records that it needed to.
- **Streaming.** Each message is one row. For streams, use RFC 0007's
  scheduler payload + chunked messages.
- **Cross-host.** Single daemon. RFC 0015 cluster layer would route
  cross-node mailboxes; out of scope.
- **Encryption.** Sender + recipient + topic + payload are stored
  plaintext in kernel.db. Same trust boundary as the kernel itself.

## 2. Data model

### `Mailbox`

```python
@dataclass(frozen=True)
class Mailbox:
    pid:          int           # owning agent
    queue_size:   int           # max pending (delivered_at IS NULL) messages
    retention_s:  float         # purge delivered messages after this
    created_at:   float
```

### `Message`

```python
@dataclass(frozen=True)
class Message:
    msg_id:        int
    sender_pid:    int | None        # NULL = system / external
    recipient_pid: int               # always set; for pubsub the kernel
                                      # writes one row per subscriber
    topic:         str | None        # NULL = direct send; non-NULL = pubsub
    kind:          str               # opaque application-level kind
    payload:       dict
    posted_at:     float
    delivered_at:  float | None      # set on first read with mark_delivered
    expires_at:    float | None      # absolute epoch seconds
```

`kind` is application-level (e.g. `"task_done"`, `"approval_request"`).
The kernel does not interpret it.

### `Subscription`

```python
@dataclass(frozen=True)
class Subscription:
    pid:        int
    topic:      str
    created_at: float
```

`UNIQUE(pid, topic)` — duplicate subscribe is idempotent.

## 3. Operations

### `mailbox.create(pid, queue_size=1024, retention_s=3600)`

Idempotent: a second create for the same pid raises
`MailboxAlreadyExists`. The supervisor calls this once when an agent
is registered; subsequent agent restarts reuse the existing mailbox
row.

### `mailbox.delete(pid)`

Drops the mailbox row, all subscriptions, and all messages where
`recipient_pid = pid`. Used on agent termination cleanup. Optional —
the supervisor may instead leave the mailbox in place for audit.

### `subscribe(pid, topic)` / `unsubscribe(pid, topic)`

Adds/removes a row in `agent_subscriptions`. Topic strings are opaque;
naming convention recommendation: `"<service>/<event>"` (e.g.
`"bridge/telegram_inbound"`).

### `send(sender_pid, recipient_pid, kind, payload, expires_at=None)`

```
BEGIN IMMEDIATE
  count = SELECT COUNT(*) WHERE recipient_pid=? AND delivered_at IS NULL
  if count >= mailbox.queue_size: raise MailboxFull
  INSERT one row → return msg_id
COMMIT
```

`sender_pid=None` is allowed (system messages).

### `publish(sender_pid, topic, kind, payload, expires_at=None)`

For each `pid` in `agent_subscriptions` with `topic=?`:
1. Check that pid's mailbox isn't full.
2. INSERT one row with `recipient_pid=pid, topic=topic, …`.

If any subscriber's mailbox is full, the publish is **partial-success**
by default: the kernel inserts to subscribers with capacity, returns
the count delivered AND the count rejected. Strictness mode
`fail_on_full=True` aborts the whole publish if any subscriber is
full.

### `recv(pid, since_msg_id=0, limit=100, mark_delivered=True)`

Return up to `limit` messages with `recipient_pid=pid AND msg_id >
since_msg_id AND (expires_at IS NULL OR expires_at > now)`. If
`mark_delivered=True`, the kernel updates `delivered_at` in the same
transaction.

Returns a list ordered by `msg_id ASC`. The recipient tracks the
high-water mark and passes it as `since_msg_id` next call — same
pattern as `kernel.events.tail`.

### `peek(pid, since_msg_id=0, limit=100)`

Same as recv but no `delivered_at` write. Useful for inspection /
debugging.

### `gc_expired(now=time.time())`

Two passes in one transaction:
1. Delete rows where `expires_at IS NOT NULL AND expires_at < now`.
2. Delete rows where `delivered_at IS NOT NULL AND delivered_at +
   mailbox.retention_s < now`.

Returns count purged. Operator runs this periodically; a future
`monitor.scheduler` job will call it on a cron.

## 4. Storage

Schema v4 (additive on v3). Three tables.

```sql
CREATE TABLE IF NOT EXISTS agent_mailboxes (
    pid          INTEGER PRIMARY KEY,
    queue_size   INTEGER NOT NULL DEFAULT 1024,
    retention_s  REAL    NOT NULL DEFAULT 3600,
    created_at   REAL    NOT NULL,
    FOREIGN KEY (pid) REFERENCES agent_processes(pid)
);

CREATE TABLE IF NOT EXISTS agent_subscriptions (
    pid          INTEGER NOT NULL,
    topic        TEXT    NOT NULL,
    created_at   REAL    NOT NULL,
    PRIMARY KEY (pid, topic),
    FOREIGN KEY (pid) REFERENCES agent_processes(pid)
);

CREATE INDEX IF NOT EXISTS idx_agent_subscriptions_topic
    ON agent_subscriptions(topic);

CREATE TABLE IF NOT EXISTS agent_messages (
    msg_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_pid    INTEGER,
    recipient_pid INTEGER NOT NULL,
    topic         TEXT,
    kind          TEXT    NOT NULL,
    payload       TEXT    NOT NULL,
    posted_at     REAL    NOT NULL,
    delivered_at  REAL,
    expires_at    REAL,
    FOREIGN KEY (recipient_pid) REFERENCES agent_processes(pid)
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_pending
    ON agent_messages (recipient_pid, msg_id)
    WHERE delivered_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_agent_messages_expires
    ON agent_messages (expires_at)
    WHERE expires_at IS NOT NULL;
```

## 5. RPC surface

```
kernel.mbox.create
  params: { pid, queue_size?, retention_s? }
  result: { pid }

kernel.mbox.delete
  params: { pid }
  result: { pid, removed_messages: int }

kernel.mbox.subscribe / unsubscribe
  params: { pid, topic }
  result: { pid, topic }

kernel.mbox.list_subscriptions
  params: { pid }
  result: { topics: [str] }

kernel.mbox.send
  params: { sender_pid?, recipient_pid, kind, payload, expires_at? }
  result: { msg_id }

kernel.mbox.publish
  params: { sender_pid?, topic, kind, payload, expires_at?, fail_on_full? }
  result: { delivered: int, rejected: int, msg_ids: [int] }

kernel.mbox.recv
  params: { pid, since_msg_id?, limit?, mark_delivered? }
  result: { messages: [Message], next_cursor: int }

kernel.mbox.peek
  params: { pid, since_msg_id?, limit? }
  result: { messages: [Message] }

kernel.mbox.gc_expired
  params: { now? }
  result: { purged: int }
```

### Error codes

| Code | Name | Meaning |
|---|---|---|
| -32141 | `kernel_mbox_not_found` | No mailbox row for the pid |
| -32142 | `kernel_mbox_already_exists` | Second create |
| -32143 | `kernel_mbox_full` | queue_size hit |
| -32144 | `kernel_mbox_invalid_payload` | bad pid/topic/payload |
| -32145 | `kernel_mbox_subscription_missing` | unsubscribe on absent sub |

## 6. Backwards compatibility

- Schema v3 → v4 forward migration is additive (new tables only).
- No existing module modified.
- No existing test changes required (unlike when SCHEMA_VERSION
  literal was hardcoded — past slices already swapped to
  `kernel.SCHEMA_VERSION`).

## 7. Open questions

1. **Should publish be all-or-nothing on full mailboxes?** Current
   default is partial-success with `rejected` count. `fail_on_full=True`
   gives strict-publish for cases where the publisher would rather
   retry the whole batch. **Lean: keep partial as default.**
2. **TTL precision.** `expires_at` is filtered at recv time and
   purged at gc time. A message with `expires_at` in the past is
   visible to `peek` but invisible to `recv`. Consistent with what
   most users want; if a user wants strict immediate purge, they call
   `gc_expired` themselves.
3. **Message ordering across publishes.** Within one recipient,
   `msg_id` is monotonic across all sources (direct + pubsub). That's
   what AUTOINCREMENT gives us. **Sufficient.**

## 8. Acceptance criteria

A PR claiming this RFC must:

1. Schema migrates v3 → v4 forward; new tables present.
2. Direct send + recv round-trip with `mark_delivered=True` correctly
   advances `delivered_at` and the `next_cursor`.
3. Pub/sub fan-out: 3 subscribers, 1 publish → 3 message rows; only
   subscribed pids see them.
4. queue_size enforced: send to a full mailbox raises `MailboxFull`;
   publish with `fail_on_full=False` reports `rejected` count
   without raising.
5. Retention: a delivered message older than `retention_s` is purged
   by `gc_expired`.
6. TTL: a message whose `expires_at < now` is invisible to `recv`
   and purged by `gc_expired`.
7. Concurrent send: 4 threads × 25 sends → 100 distinct `msg_id`s
   with no losses.
8. RPC surface works end-to-end through the daemon.
9. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
