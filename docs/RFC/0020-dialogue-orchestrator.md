# Design Note: DialogueOrchestrator — multi-turn LLM conversation

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0011-agent-fs.md`](./0011-agent-fs.md), [`0016-subprocess-agent-runner.md`](./0016-subprocess-agent-runner.md), [`0019-llm-runner.md`](./0019-llm-runner.md)

The LLM runner from RFC 0019 makes one call. A real conversation
makes many. This RFC ties N single-turn calls together into a
multi-turn dialogue, with **conversation state persisted in
AgentFS** so a daemon restart, a runner crash, or an operator
intervention doesn't lose the thread.

The orchestrator runs in the daemon's Python process, not in a
subprocess — it's a thin loop that:

1. Loads conversation history from AgentFS.
2. Appends a new user message.
3. Spawns the LLM runner with the full message list.
4. Receives the response via `RunnerExitInfo.text`.
5. Appends the assistant response.
6. Saves history back to AgentFS.

Each turn is **its own subprocess**. A long conversation is
N subprocess spawns, each charged separately to the ledger, each
sandboxed independently.

This RFC ships **purely additive** code in `kernel/orchestrator/`,
plus two **backwards-compatible** extensions:

- `LlmRequest` gets an optional `messages: list[dict]` field
  (canonical multi-turn format). Existing single-turn callers
  using `user` still work.
- `RunnerExitInfo` gets optional `text: str` and `metadata: dict`
  fields, populated from the runner's exit message. Existing
  callers that don't read these fields are unaffected.

## 1. Goals & non-goals

**Goals:**

1. **Real multi-turn conversation.** The standard chat shape
   `[{role: 'user' | 'assistant' | 'system', content: ...}]` is
   the source of truth.
2. **Persistent state.** History stored in AgentFS at
   `/conversations/<pid>/history.json` (path configurable). Survives
   daemon restart.
3. **Per-turn isolation.** Each LLM call is its own subprocess
   under its own sandbox + ledger charges. A bad turn (timeout,
   crash) doesn't poison the next turn — the orchestrator retries
   or stops based on the exit_kind.
4. **No special privilege.** The orchestrator is a Python class
   that uses the same Kernel facade everyone else does. Operators
   can subclass / extend without forking.

**Non-goals (this RFC):**

- **Tool dispatch.** Out of scope. Tools require permission routing
  + capability checks + multi-step "tool call → tool response →
  next LLM turn" flow. Separate, much bigger RFC.
- **Streaming.** Each turn buffers full response. Token-by-token
  streaming back to a caller is a future RFC.
- **Memory / context compaction.** When the conversation grows
  past the model's context window, this RFC just lets the LLM
  return an error. Truncation / summarization policies live
  above.
- **Branching / undo.** History is append-only via
  ``turn()``; ``reset()`` wipes. Branching dialogue (multiple
  hypotheticals) is a future feature.
- **Cross-agent dialogue.** Each orchestrator instance owns one
  agent + one history. Multi-agent conversation is a separate
  pattern (mailbox).

## 2. Data model

### Message

```python
{"role": "system" | "user" | "assistant", "content": str}
```

The orchestrator accepts and emits this exact shape. The kernel
LLM runner (RFC 0019, with the messages extension below) passes
it through to the provider.

### History storage

AgentFS path (default): `/conversations/<pid>/history.json`

File contents:

```jsonc
{
  "version":  1,
  "model":    "claude-opus-4-7",
  "system":   "You are helpful.",
  "messages": [
    {"role": "user",      "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user",      "content": "what's 2+2"}
  ],
  "stats": {
    "turns":         2,
    "total_tokens":  85,
    "total_cost_micro": 1750,
    "started_at":    1714867123.0,
    "last_turn_at":  1714867205.0
  }
}
```

`messages[*].role == "system"` is allowed but discouraged; the
orchestrator prefers the dedicated `system` field at the top
level (matches Anthropic's API shape).

## 3. API

```python
class DialogueOrchestrator:
    def __init__(
        self,
        kernel: Kernel,
        *,
        agent_pid: int,
        model:        str    = "claude-opus-4-7",
        system:       str    = "",
        max_tokens:   int    = 1024,
        temperature: float   = 0.7,
        runner_argv:  Sequence[str] | None = None,   # default: kernel.runner.llm
        runner_policy: SandboxPolicy | None = None,
        runner_env:   Mapping[str, str] | None = None,
        history_path: str | None = None,             # default: /conversations/<pid>/history.json
        wait_timeout_s: float = 300.0,
    ): ...

    def turn(self, user_message: str) -> str:
        """Append user_message, call LLM, return assistant text.
        Persists updated history to AgentFS atomically (one fs.write)."""

    def reset(self, *, keep_system: bool = True) -> None:
        """Clear conversation. Optionally keep the system prompt."""

    def history(self) -> list[dict]:
        """Snapshot of current messages list."""

    def stats(self) -> dict:
        """Aggregate turns / tokens / cost."""
```

`turn()` errors:

| Cause | Behaviour |
|---|---|
| LLM runner exits failed | Raise `DialogueTurnFailed`; user message is NOT persisted (orchestrator rolls history back) |
| Wall-clock timeout | Raise `DialogueTurnTimeout`; same rollback |
| Ledger first_breach during turn | Raise `DialogueQuotaBreached`; user message persisted, no assistant — caller can decide whether to bump the budget and retry |
| Successful turn | Persist updated history, return assistant text |

## 4. Per-turn flow

```
turn(user_message):
  1. history = load_history()  (or empty if first turn)
  2. tentative_history = history + [{role: user, content: user_message}]
  3. spawn LLM runner with init_payload = {
       model:    self.model,
       system:   self.system,
       messages: tentative_history,
       max_tokens: ...
     }
  4. info = supervisor.wait(pid)
  5. if info.exit_kind != "completed":
        raise DialogueTurnFailed(info)
     (no rollback needed — history not yet persisted)
  6. assistant_text = info.text  (full response, not truncated)
  7. final_history = tentative_history + [{role: assistant, content: assistant_text}]
  8. stats.turns += 1; stats.total_tokens += info.ledger_charged.tokens; ...
  9. save_history(final_history, stats)  via kernel.fs.write
 10. return assistant_text
```

Steps 4 and 5 are where ledger first_breach kicks in. The
supervisor records the breach event; the orchestrator detects
breached state via `info.ledger_charged` against the last
charge result. If first_breach is observed in this turn, the
orchestrator's policy is: persist user + assistant if both
arrived, then raise `DialogueQuotaBreached` so the caller knows
not to start another turn without a budget bump.

## 5. AgentFS as state plane

The orchestrator uses `kernel.fs` for ALL persistent state. No new
schema, no in-memory state that survives the Python instance.
This means:

- Daemon restart: the next `DialogueOrchestrator(kernel,
  agent_pid=42)` instance picks up exactly where the last one
  left off.
- Multi-orchestrator on same pid: undefined behaviour — last
  writer wins. Operator's responsibility to avoid.
- AgentFS quota (`fs_w_bytes`): writing the history file counts.
  A 500-turn conversation can be sizeable; operators with tight
  fs quotas need to call `reset()` periodically or dial budget.

`kernel.fs.write` writes the whole history file each turn (no
incremental append). Cost: O(history size) per turn. For
typical chat sizes (100s of KB), fine. A future RFC may switch
to append-only logging if needed.

## 6. LlmRequest extension (additive)

Add a `messages` field to `LlmRequest`:

```python
@dataclass(frozen=True)
class LlmRequest:
    model:       str
    user:        str = ""              # CHANGED: was required
    system:      str = ""
    messages:    list = ()             # NEW
    max_tokens:  int = 1024
    temperature: float = 0.7
    metadata:    dict = field(...)

    def __post_init__(self):
        if not self.messages and not self.user:
            raise ProviderInvalidRequest(
                "either 'messages' or 'user' must be set",
            )
        # … existing validation continues
```

Single-turn callers (`user="..."`) work unchanged.
Multi-turn callers pass `messages=[...]` and leave `user=""`.

`AnthropicProvider.__call__`:

```python
if request.messages:
    payload_messages = list(request.messages)
else:
    payload_messages = [{"role": "user", "content": request.user}]
client.messages.create(messages=payload_messages, ...)
```

`MockProvider`: messages is opaque; mock returns its fixed
response regardless.

## 7. RunnerExitInfo extension (additive)

```python
@dataclass(frozen=True)
class RunnerExitInfo:
    pid:          int
    exit_kind:    str
    exit_code:    int
    stdout_tail:  bytes
    stderr_tail:  bytes
    duration_s:   float
    ledger_charged: dict
    text:         str = ""        # NEW: full response from exit msg
    metadata:     dict = field(default_factory=dict)  # NEW
```

`Supervisor.wait` reads `text` and `metadata` from the runner's
`exit` message and populates the new fields. Defaults preserve
existing test assertions.

The LLM runner emits:

```jsonc
{
  "op": "exit",
  "exit_kind": "completed",
  "summary":   "<truncated, ≤500 chars>",
  "text":      "<full response, no truncation>",
  "metadata":  { "finish_reason": "stop", "tokens_total": 25, … }
}
```

Existing runners (echo) don't emit `text` — supervisor sets it
to "".

## 8. Backwards compatibility

- `LlmRequest`: `user` becomes optional with default `""`. The
  `__post_init__` requires *either* `messages` or `user`, so
  single-turn callers passing `user="hi"` still work; multi-turn
  passes `messages=[...]`.
- `RunnerExitInfo`: two new fields with defaults; existing tests
  that don't reference them keep passing.
- `LlmResponse`: unchanged.
- AgentFS: no schema change; orchestrator writes through the
  existing `kernel.fs.write` API.

The new `kernel/orchestrator/` package is purely additive.

## 9. Open questions

1. **Auto-truncation when context window overflows.** v1 lets the
   provider raise; orchestrator surfaces the error. A future RFC
   may add policy: "drop oldest N user/assistant pairs".
2. **Concurrent turns.** Two `turn()` calls on the same
   orchestrator instance are not synchronised in v1 — caller
   serialises. A `threading.Lock` would be easy to add but the
   pattern of "one orchestrator per agent per chat" makes it
   unnecessary in practice.
3. **System prompt mutation between turns.** v1 freezes
   `system` at construction. Changing it mid-conversation
   requires `reset(keep_system=False)` then re-construct. RFC
   could add `update_system(new)` later.

## 10. Acceptance criteria

A PR claiming this RFC must:

1. `LlmRequest(messages=[...])` works; the runner forwards
   messages to the provider; mock + (with SDK) anthropic both
   accept.
2. `RunnerExitInfo.text` is populated from the runner's exit
   `text` field; defaults to "".
3. Two-turn conversation via `DialogueOrchestrator`:
   - turn 1: user "hi" → assistant "hello"
   - turn 2: user "what's 2+2" → assistant gets a `messages` list
     with the full prior exchange in the prompt.
4. History persists to AgentFS; reading it back via
   `kernel.fs.read` returns the documented JSON shape.
5. Ledger charges accumulate across turns when the agent has
   `tokens` / `cost_micro` budgets.
6. Daemon restart simulation: re-instantiate the orchestrator with
   the same `agent_pid`; the next `turn()` includes prior history.
7. `reset()` clears history (file is overwritten).
8. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
