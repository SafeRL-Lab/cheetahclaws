# Agent OS — `kernel/`

CheetahClaws ships a **single-node agent operating system** under
`kernel/`. It's the substrate that turns the legacy REPL/bridge
into a long-running, multi-agent kernel: process table, capability
model, quota ledger, scheduler, mailbox/registry, virtual filesystem,
observability, and a stable JSON-RPC contract — backed by a single
SQLite WAL-mode database.

The kernel layer is **strictly additive**. Every change lands behind
the `--enable-kernel` activation flag; default behaviour is
byte-for-byte unchanged from prior releases. Existing REPL, bridges,
and CLI users see no difference.

## Why this exists

Before `kernel/`, cheetahclaws was an *agent runtime/middleware*:
single-user REPL → tool dispatch → LLM. There was no place to:

- Run multiple agents concurrently with isolation between them.
- Cap CPU / memory / file-size / network on a per-agent basis with
  observable rollups.
- Route tool calls through capability checks.
- Wire one agent's output into another's mailbox.
- Persist agent state across restarts via a versioned schema.
- Surface live metrics over Prometheus.

The kernel layer adds all of that as opt-in surface, while keeping
the legacy single-process REPL path intact.

## Layout

```
kernel/
  api.py             # `Kernel` facade — open(...), make_supervisor(), …
  store.py           # SQLite WAL store, single shared connection
  schema.py          # Forward-only migrations v1 → v7
  capability.py      # tool_grants / fs_grants / net_grants / model_grants
  ledger.py          # Per-agent ResourceLedger + first_breach signal
  scheduler.py       # Priority queue + admission filter
  mailbox.py         # Direct + topic pub-sub (RFC 0009)
  registry.py        # name → pid (RFC 0010)
  agent_fs.py        # VFS unifying memory/checkpoint/skill/task
  sandbox.py         # RLIMIT + bubblewrap + wall-clock killer
  contract.py        # Frozen v1.0 method registry, drift CI guard
  cli.py             # `cheetahclaws kernel <action>` subcommand
  tools/             # Built-in tools (Echo, Read, Write, Glob, List,
                     # Diff, AST) and opt-in (Exec, Fetch, Git)
  runner/
    supervisor.py    # Subprocess agents w/ IPC + chunk streaming
    ipc.py           # Line-delimited JSON channel
    llm/             # LLM runner (Anthropic + scripted mock providers)
    bridge_mirror/   # bridges ↔ kernel.mbox without touching bridges/
```

## Activation

Operators turn the kernel on via:

```bash
cheetahclaws serve --enable-kernel
```

…then introspect it via the kernel CLI:

```bash
cheetahclaws kernel summary             # uptime, agents, queue rollup
cheetahclaws kernel info                # version, schema, API surface
cheetahclaws kernel agents [--state S]
cheetahclaws kernel proc <pid>          # combined per-agent view
cheetahclaws kernel events [--pid P]
cheetahclaws kernel queue [--state S]
cheetahclaws kernel registry [--prefix P] [--tag T]
cheetahclaws kernel methods [--tier T]  # documented kernel.* RPCs
cheetahclaws kernel prometheus          # Prometheus exposition text
```

Without `--enable-kernel`, the daemon serves the same surface as
before and `kernel/` code is dormant.

## RFC roadmap

The kernel was built one RFC at a time. Every behaviour change is
documented under [`docs/RFC/`](RFC/); all RFCs in this table are
shipped.

| RFC | Theme |
|---|---|
| [0001](RFC/0001-daemon-design-note.md) | Daemon design note (IPC, auth, originator) |
| [0002](RFC/0002-daemon-foundation-roadmap.md) | Foundation roadmap (F-1..F-9) |
| [0003](RFC/0003-agent-process-and-event-log.md) | AgentProcess + EventLog |
| [0005](RFC/0005-capability-model.md) | Capability model |
| [0006](RFC/0006-resource-ledger.md) | Per-agent quota ledger |
| [0007](RFC/0007-agent-scheduler.md) | Priority scheduler + admission filter |
| [0008](RFC/0008-agent-sandbox.md) | RLIMIT + bubblewrap sandbox |
| [0009](RFC/0009-agent-mailbox.md) | Mailbox + pub-sub IPC |
| [0010](RFC/0010-agent-registry.md) | Agent registry / service discovery |
| [0011](RFC/0011-agent-fs.md) | AgentFS — unified VFS |
| [0012](RFC/0012-observability.md) | Observability + chaos suite |
| [0013](RFC/0013-api-stability.md) | API stability + deprecation policy |
| [0016](RFC/0016-subprocess-agent-runner.md) | Subprocess agent runner |
| [0017](RFC/0017-worker-loop.md) | WorkerLoop (scheduler↔supervisor glue) |
| [0018](RFC/0018-bridge-mirror.md) | Bridge ↔ kernel.mbox glue |
| [0019](RFC/0019-llm-runner.md) | LLM runner MVP |
| [0020](RFC/0020-dialogue-orchestrator.md) | Multi-turn dialogue orchestrator |
| [0021](RFC/0021-tool-dispatch.md) | Tool dispatch + permission routing |
| [0022](RFC/0022-llm-tool-calling.md) | LLM tool calling integration |
| [0023](RFC/0023-shell-exec-tool.md) | Exec tool (argv-only, RLIMITed) |
| [0024](RFC/0024-glob-list-tools.md) | Glob + List built-in tools |
| [0025](RFC/0025-fetch-tool.md) | Fetch tool (SSRF/DNS-rebind defended) |
| [0026](RFC/0026-ipc-streaming.md) | IPC streaming chunks |
| [0027](RFC/0027-llm-streaming.md) | LLM streaming (provider opt-in) |
| [0028](RFC/0028-exec-streaming.md) | Exec stdout/stderr line streaming |
| [0029](RFC/0029-fetch-streaming.md) | Fetch terminal-hop body streaming |
| [0030](RFC/0030-diff-tool.md) | Diff tool (path + text mode) |
| [0031](RFC/0031-ast-tool.md) | AST tool (Python source inspector) |
| [0032](RFC/0032-git-tool.md) | Git tool (read-only, op+flag allowlist) |
| 0014 | Multi-tenant — **parked** |
| 0015 | Cluster — **parked** |

Phasing: **Phase 1** fault domain (0003+0008) → **Phase 2** quota +
capability (0005+0006) → **Phase 3** scheduler + IPC (0007+0009+0010)
→ **Phase 4** AgentFS (0011) → **Phase 5** ops (0012+0013) → tools +
streaming (0019-0032).

## Tool inventory

### Auto-registered (`register_builtin_tools`)

| Tool | Purpose | fs grant |
|---|---|---|
| `Echo` | Smoke-test the dispatch path | — |
| `Read` | Read a file, 4 MB cap | `r` |
| `Write` | Write a file, 4 MB cap | `rw` |
| `Glob` | Pattern match (≤ 10k results) | `r` |
| `List` | Directory listing | `r` |
| `Diff` | Unified diff (path or text mode) | `r` |
| `AST` | Python AST inspector | `r` |

### Opt-in (operator must call `register_<tool>(registry)`)

| Tool | Purpose | Sandbox |
|---|---|---|
| `Exec` | argv-only subprocess, no shell | RLIMIT + wall-clock + scrubbed env |
| `Fetch` | Bounded HTTP, SSRF/DNS-rebind defended | per-hop cap check + IP block |
| `Git` | Read-only git inspector | RLIMIT + op+flag allowlist + gitconfig disabled |

The opt-in tools are **NOT** in `register_builtin_tools` because
their threat surface is materially larger. Operators must explicitly
opt in.

## Streaming

Three layers stream incrementally to a single `on_chunk(payload)`
sink:

- **LLM** — provider-side `stream(req, on_delta)` emits per-token
  text deltas (RFC 0027).
- **Exec** — Popen + queue-serialized reader threads emit per-line
  stdout/stderr (RFC 0028).
- **Fetch** — terminal-hop body chunks per 8 KB read (RFC 0029).

Plumbed end-to-end through:

```python
sup.wait(pid, on_chunk=lambda c: my_ui.append(c))
```

…where each chunk is a dict `{op:"chunk", kind, content,
metadata:{...}}`. `RunnerExitInfo.chunks` accumulates the full
sequence post-exit.

## Backwards compatibility

- All kernel code is gated behind `--enable-kernel`. Default
  CheetahClaws CLI / REPL / bridges / web UI unchanged.
- Kernel SQLite schema is forward-only (versioned migrations
  `v1 → v7`). Old kernel.db files upgrade in place.
- The v1.0 RPC contract (58 stable methods) has CI drift guard
  via `kernel/contract.py` — accidental method removal fails
  the build.
- Tests: 1771 passing, zero regressions on the legacy code paths.

## Where to next

The kernel is at v1.0 production-grade for **single-node** use.
Two RFCs remain explicitly parked:

- **RFC 0014 multi-tenant** — only worth doing if cheetahclaws is
  deployed as team SaaS / shared infrastructure.
- **RFC 0015 cluster** — only worth doing once a single host
  saturates real workload (distributed scheduler + cross-host
  mailbox + partition tolerance).

Higher-ROI follow-ups: tag a v1.x release + CHANGELOG, integration
performance tests under real LLM workload, operator documentation
for `--enable-kernel` deployment.
