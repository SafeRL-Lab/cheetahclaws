# Daemon Foundation Roadmap

- **Status:** Tracking
- **Refs:** [#68](https://github.com/SafeRL-Lab/cheetahclaws/issues/68), [RFC 0001 design note](./0001-daemon-design-note.md)
- **Last updated:** 2026-04-30

The "foundation PR" described at the end of [RFC 0001](./0001-daemon-design-note.md) is too big for one reviewable change (~5 KLoC including stdlib HTTP server, auth, JSON-RPC + SSE, SQLite schema, `daemon` CLI, bridges-into-daemon, subprocess-per-agent, and conservative cost defaults). This document splits it into nine stackable PRs and pins the acceptance criteria for each. Implementation follows this index in order; later items can land in parallel once F-1 and F-2 are merged.

## Index

| ID  | Scope                                               | Depends on | Est LoC | Status |
|-----|-----------------------------------------------------|------------|---------|--------|
| F-1 | `daemon/` package skeleton; `serve` + `daemon` CLI  | —          | ~1500   | MERGED #80 |
| F-2 | SQLite schema + events persistence + jobs migration | F-1        | ~700    | OPEN   |
| F-3 | `monitor/scheduler` runs in daemon                  | F-2        | ~700    | OPEN   |
| F-4 | `agent_runner` becomes subprocess-per-agent         | F-2        | ~1000   | TODO   |
| F-5 | `proactive` watcher runs in daemon                  | F-2        | ~200    | TODO   |
| F-6 | Telegram bridge in daemon                           | F-2        | ~500    | TODO   |
| F-7 | Slack bridge in daemon                              | F-6        | ~500    | TODO   |
| F-8 | WeChat bridge in daemon                             | F-6        | ~500    | TODO   |
| F-9 | Conservative cost-guardrail defaults under `serve`  | F-1        | ~150    | TODO   |

## F-1 — daemon skeleton

**Scope.** Adopt the `cc_daemon/` reference scaffolding from
[`feature/daemon-spike`](https://github.com/SafeRL-Lab/cheetahclaws/tree/feature/daemon-spike)
(`server`, `auth`, `originator`, `rpc`, `events`, `permission`, `methods`)
**as-is** — those modules encode the contract the maintainer reviewed in
PR #74.  Layer the foundation glue on top:

- `cc_daemon/discovery.py` — atomic `~/.cheetahclaws/daemon.json` so
  REPL / Web / bridge clients can locate the running daemon (transport,
  address, version).  Spike's pid file stays for "is anything running?"
  liveness; discovery answers "where is it?".
- `cc_daemon/system_methods.py` — registers `system.ping` (returns
  `"pong"`) and `system.shutdown` (sets `DaemonState.shutdown_event`,
  giving us cross-platform graceful exit since Windows can't deliver
  SIGTERM cleanly to another Python process).
- `cc_daemon/cli.py` — rewritten `serve_main(argv)` that calls
  `bootstrap()`, pins `log_file` to `<data_dir>/logs/daemon.log`, threads
  the loaded `config` and the `--unauthenticated-metrics` flag through
  `DaemonState`, writes the discovery file on bind, watches the shutdown
  event, and clears discovery on exit.
- `cc_daemon/server.py` — minimal patch: route `/healthz` `/readyz`
  `/metrics` through `health.payload_for(path, config)` instead of
  the spike's stub `{"status": "ok"}`.  Auth-gated by default; opt out
  via `--unauthenticated-metrics`.  Adds Windows guard around
  `socketserver.UnixStreamServer` (unavailable on Windows).
- `commands/daemon_cmd.py` — `cheetahclaws daemon {status, stop, logs,
  rotate-token}` subcommand handlers.  `status` reads discovery + pings
  `system.ping`; `stop` calls `system.shutdown` RPC then falls back to
  SIGTERM / TerminateProcess; `logs` tails `~/.cheetahclaws/logs/daemon.log`;
  `rotate-token` regenerates the token (notes that existing TCP clients
  receive 401 until they re-read the file).
- `health.py` — refactor: extract module-level `healthz_payload(config)`
  / `readyz_payload(config)` / `metrics_payload(config)` /
  `payload_for(path, config)` so both the existing standalone health
  HTTP server and `cc_daemon/server.py` reuse the same
  circuit-breaker / quota / runtime-registry probes.  No behaviour
  change for existing `health_check_port` users.
- `cheetahclaws.py` — main() short-circuit: `cheetahclaws serve`
  dispatches to `cc_daemon.cli.serve_main`; `cheetahclaws daemon
  <action>` dispatches to `commands.daemon_cmd.dispatch`.  Replaces the
  spike's `spike-daemon` shim.

**Acceptance.**
- `cheetahclaws serve` starts; `cheetahclaws daemon status` reports pid,
  transport, address, uptime, ping outcome.
- Unix socket (POSIX): `curl --unix-socket <path> -X POST /rpc
  -H "Cheetahclaws-Api-Version: 0" -d '{"jsonrpc":"2.0","id":1,"method":"system.ping"}'`
  returns `{"jsonrpc":"2.0","id":1,"result":"pong"}`.
- TCP: same call without `Authorization: Bearer <token>` returns 401;
  with valid token returns 200; sustained bad-token attempts trip the
  spike's brute-force throttle (429).
- `curl … GET /events` keeps the stream open; heartbeats arrive at
  spike's 15 s cadence.
- `cheetahclaws daemon stop` → `system.shutdown` RPC → discovery file
  cleared and process exits 0.
- `cheetahclaws daemon rotate-token` regenerates the token; existing TCP
  clients receive 401 on next request until they re-read the file.
- pytest green on Linux, macOS, Windows (TCP-only on Windows; Unix
  socket tests skip on Windows).

## F-2 — SQLite schema + events persistence + jobs migration

**Scope.** Seven additive tables in `~/.cheetahclaws/sessions.db`; swap
the F-1 in-memory event ring for a SQLite-backed channel; migrate
`jobs.py` JSON storage to SQLite.  **Originator-tracked permission flow
is already provided by spike's `cc_daemon/originator.py` +
`cc_daemon/permission.py`** (see PR #80) — this PR doesn't re-do it.

**Tables (additive — `sessions` from `session_store.py` untouched).**
`schema_meta`, `daemon_events`, `agent_runs`, `agent_iterations`,
`jobs`, `monitor_subscriptions`, `monitor_reports`, `bridges`.

**Deliverables.**
- `cc_daemon/schema.py` — DDL + `init_schema(db_path)` (idempotent,
  internally locked) + `get_conn()` (thread-local, mirrors
  `session_store` pattern) + `get_schema_version()` accessor; future
  migrations land in `_apply_migrations()`.
- `cc_daemon/cli.py:cmd_serve` calls `init_schema()` right after
  `bootstrap()` so tables exist before the first publish.
- `cc_daemon/events.py` — rewritten: `EventBus.publish` does an INSERT
  into `daemon_events` (id from `AUTOINCREMENT`, monotonic across
  restarts and prunes), still fans out to in-process subscribers for
  live tail; `replay_since(N)` reads from SQLite and emits a synthetic
  `gap` event when `N` is older than the oldest surviving row.
  Default retention: 24 h / 100 K rows; opportunistic prune every 100
  publishes.
- `jobs.py` — `_persist`/`_row_to_job` hit SQLite; `_ensure_migrated()`
  imports legacy `~/.cheetahclaws/jobs.json` once (tracked via
  `schema_meta.jobs_migrated_from_json`).  JSON file is **left readable
  in place** for one release as fallback.  Public API unchanged.

**Acceptance.**
- `init_schema()` is idempotent across daemon restarts and concurrent
  callers (verified by 12 unit tests in `tests/test_cc_daemon_schema.py`).
- Spike's 13 contract tests in `tests/test_daemon_spike.py` keep
  passing on the SQLite-backed bus (only the two ring-buffer tests
  needed an in-place rewrite to test retention-based eviction instead
  of the deleted in-memory cap).
- New `tests/test_cc_daemon_events_sqlite.py` (15 tests) covers
  persistence, retention by row count + age, gap-on-old-since,
  cross-instance replay (simulated daemon restart), and the
  `reset_bus_for_tests()` truncate path.
- New `tests/test_jobs_sqlite.py` (14 tests) covers create / start /
  add_step / lifecycle / list_recent / list_running / `_MAX_JOBS`
  pruning + JSON-file migration (idempotency, corrupt-file tolerance,
  legacy-file kept readable).
- New e2e `tests/e2e_daemon_skeleton.py::test_events_persist_in_sqlite_across_daemon_restart`
  publishes events on daemon A via `echo.ping`, stops A, starts B
  against the same data dir, and verifies `GET /events?since=0`
  replays the events from SQLite.

## F-3 — monitor in daemon

**Scope.** `monitor/scheduler.py` runs daemon-side; subscription store
moves from JSON to the F-2 `monitor_subscriptions` table; reports
persist + emit SSE events; REPL skips its local scheduler when a
daemon is detected.

**Deliverables.**

- `monitor/store.py` — SQLite-backed (`monitor_subscriptions` and
  `monitor_reports` tables).  One-shot import of legacy
  `~/.cheetahclaws/monitor_subscriptions.json` on first call (tracked
  in `schema_meta.monitor_migrated_from_json`); JSON kept readable for
  one release.  New helpers: `save_report`, `list_reports`.  Public
  API of the legacy store unchanged.
- `monitor/scheduler.py` — `run_one()` persists the full report body
  via `save_report` and publishes a `monitor_report` event on
  `cc_daemon.events.get_bus()` with `{topic, report_id, body, sent_to,
  errors}`.  Loop's idle wait switched from `time.sleep(30)` ×60 to a
  single `Event.wait(60)` so daemon shutdown isn't stalled by the
  scheduler thread napping.
- `cc_daemon/monitor_methods.py` — registers `monitor.subscribe`,
  `monitor.unsubscribe`, `monitor.list`, `monitor.run` for external
  clients (Web UI / third-party tools).  `DaemonState.__init__` calls
  `monitor_methods.register` next to `system_methods`.
- `cc_daemon/cli.py:cmd_serve` — starts the scheduler with
  `monitor.scheduler.start(config)` after schema init; the existing
  shutdown watcher calls `monitor.scheduler.stop()` before triggering
  HTTP-server shutdown.
- `commands/monitor_cmd.py` — `/monitor start` and `/monitor stop`
  detect a live daemon via `cc_daemon.discovery.locate()` and no-op
  with a friendly message.  `/monitor subscribe` / `unsubscribe` /
  `list` continue to work in REPL because they hit SQLite directly.

**Acceptance.**

- `cheetahclaws serve` running → `monitor.subscribe` over RPC persists
  to SQLite; daemon scheduler fires on cadence; reports show up in
  `monitor_reports` and on the SSE channel as `monitor_report` events.
- Daemon stop → start with same data dir → `monitor.list` over RPC
  returns the previously-subscribed topics.  (Verified by
  `tests/e2e_daemon_skeleton.py::test_monitor_subscribe_via_rpc_survives_daemon_restart`.)
- REPL `/monitor subscribe` while daemon is running: subscription
  visible via `monitor.list` from outside.  Daemon picks up the new
  row on its next 60 s poll.
- Without daemon: today's REPL-only behaviour unchanged
  (in-process scheduler thread).
- Telegram / Slack / WeChat delivery from daemon: out of scope for F-3
  (waits for F-6/F-7/F-8).  Reports + `monitor_report` events still
  fire so the digest isn't lost; bridges deliver only when REPL is
  running with the channel connected.

**Tests.** `tests/test_monitor_store_sqlite.py` (18), 
`tests/test_monitor_scheduler_events.py` (7),
`tests/test_cc_daemon_monitor_methods.py` (12), plus 1 new e2e in
`tests/e2e_daemon_skeleton.py` for the survive-restart case.

## F-4 — agent_runner subprocess

**Scope.** Each `AgentRunner` is its own subprocess. From #68: *"subprocess-per-agent rather than threads — one leaking/crashing runner shouldn't take down the scheduler and bridges."*

**Deliverables.**
- `daemon/runner_supervisor.py` — spawn / monitor / restart agent-runner subprocesses.
- `daemon/runner_ipc.py` — line-delimited JSON over stdin/stdout between supervisor and runner.
- `agent_runner.py` — main entry point usable as `python -m agent_runner --pipe …`; iteration-log writes flow back to the daemon and land in `agent_iterations`.
- Permission requests from runners routed through supervisor → `daemon/permissions.py`.

**Acceptance.**
- Runner crash (`kill -9 <runner_pid>`) does not kill the daemon; supervisor logs the crash and emits `agent_runner_crash` event.
- Runner OOM does not affect monitor or bridges.
- Runner subprocess stops within 5 s of `agent.stop` RPC.
- Iteration-log entries match in-process behavior (status, duration, summary, token counts).

## F-5 — proactive watcher in daemon

**Scope.** `_proactive_watcher_loop` from `cheetahclaws.py` becomes a daemon-owned task.

**Acceptance.**
- `/proactive 5m` while daemon is running: setting persists, sentinel runs in daemon, survives REPL exit.
- Without daemon: unchanged.

## F-6 / F-7 / F-8 — bridges in daemon (one PR per bridge)

**Scope per PR.** The named bridge (`telegram`, then `slack`, then `wechat`) runs inside daemon; incoming messages enter via `POST /rpc {"method":"session.send", …}`; outgoing replies come from an SSE subscription to that session's events.

**Per-bridge deliverables.**
- Move `bridges/<kind>.py` poll loop into a daemon-owned worker.
- Drop `RuntimeContext.<kind>_send` / `<kind>_input_event` and friends; replace with the API-mediated path.
- `bridge.start` / `bridge.stop` / `bridge.list` RPC methods.
- Persist bridge state to `bridges` table.

**Acceptance per bridge.**
- Phone message → daemon `session.send` → REPL/Web/another bridge can subscribe to the same session and see events.
- Bridge survives REPL exit; user can keep texting.
- Permission requests originating from a bridge-driven turn route only to that bridge for answer (per RFC 0001 §2).

F-7 depends on F-6 (shared scaffolding); F-8 the same.

## F-9 — cost guardrail defaults under `serve`

**Scope.** When running under `cheetahclaws serve`, the four budget keys default to non-`None`:

```jsonc
{
  "session_token_budget": 200000,
  "session_cost_budget":   2.0,
  "daily_token_budget":   2000000,
  "daily_cost_budget":     20.0
}
```

REPL `--in-process` mode keeps `None` defaults (no surprise for existing users).

**Acceptance.**
- `cheetahclaws serve` started without overrides → `cheetahclaws daemon status` reports the four defaults.
- Agent runner exceeds per-session budget → status moves to `paused_budget`, `quota_warn` event emitted, runner pauses.
- `agent.resume` RPC with a new budget argument unpauses the runner.
- REPL without daemon: budgets still default to `None`.

## Cross-cutting conventions

- **Tests.** Every PR ships unit tests; F-1, F-3, F-4, F-6/7/8 also ship `tests/e2e_daemon_<area>.py`.
- **Docs.** Every PR updates the relevant section in `docs/architecture.md`. The "Daemon" header is created by F-1; subsequent PRs append.
- **Config keys.** New keys go in `cc_config.DEFAULTS`; documented in `docs/architecture.md`.
- **Backwards compatibility.** Users who never run `cheetahclaws serve` see no behavior change until the eventual default flip — that flip is out of scope here and tracked in [#68](https://github.com/SafeRL-Lab/cheetahclaws/issues/68) as the "Phase D" item.

## Updating this document

When a PR lands, change its **Status** in the index from `TODO` to `MERGED #<pr>`. If acceptance criteria evolve during a PR, update the per-PR section in the same PR — do not let this doc drift from the implementation.
