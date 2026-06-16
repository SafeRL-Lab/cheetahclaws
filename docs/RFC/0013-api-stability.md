# Design Note: API Stability — tiers, deprecation, contract test

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 5 — contract)
- **Sibling RFC:** [`0012-observability.md`](./0012-observability.md) (ships in same Phase 5 PR)

This RFC commits the kernel to **stable APIs**. v1.0 ships 58
`kernel.*` RPC methods across 11 namespaces. From this point on, every
breaking change to those methods has to follow a deprecation cycle,
and every PR that adds or changes one has to update the contract
file. The kernel is no longer a moving target for downstream code.

This RFC ships **purely additive** code: a frozen method registry, a
contract-verification helper, and two new RPCs (`kernel.api.list_methods`
and `kernel.api.version_info`) that let clients introspect what they're
talking to.

## 1. Goals & non-goals

**Goals:**

1. **Stable API surface.** Every method in `kernel.*` is classified
   as **stable**, **experimental**, or **deprecated**. v1.0 ships
   everything as **stable**; future PRs must explicitly downgrade
   before removing.
2. **Contract test.** A test in `tests/test_kernel_api_contract.py`
   asserts that the live RPC registry contents (after
   `register_with_daemon`) match the frozen list in `contract.py`.
   New methods that aren't in the list, or removed methods that
   still are, fail the test.
3. **Version handshake.** `kernel.api.version_info` returns:
   `kernel_version`, `schema_version`, daemon `api_version`,
   per-namespace stability tier counts. Clients can refuse to talk
   to a kernel whose tier doesn't include their needed methods.
4. **Deprecation cycle.** Documented in this RFC. Code-level
   enforcement is voluntary at v1; it becomes a hard test at v1.1.

**Non-goals (v1):**

- **Backward-compat shims.** When a method is deprecated, it stays
  in the registry until the next minor version; the shim is just
  the unchanged method.
- **Multi-version routing.** A single daemon serves a single
  contract version. Cluster-level multi-version coexistence is RFC
  0015's problem.
- **Auto-generated client SDKs.** The contract is the source of
  truth; SDK generation is an optional follow-up.

## 2. Stability tiers

| Tier | Compatibility commitment | Removal cycle |
|---|---|---|
| **stable** | Method name, params, result shape, error codes are frozen across minor versions. Bug fixes and additive changes (new optional params, new optional result fields) allowed. | One minor version of `deprecated` status before removal. |
| **experimental** | May change between minor versions. Clients are warned in changelog. | One minor version of `deprecated` status before removal. |
| **deprecated** | Will be removed in the next minor version. Method works but emits a warning event. | Removed in next minor version. |

For v1.0, **every kernel.* method is stable**. The
`EXPERIMENTAL_METHODS` and `DEPRECATED_METHODS` sets are empty.

## 3. Frozen method list (v1.0)

These methods, registered by `kernel.register_with_daemon`, are
the v1.0 stable kernel API surface.

```
# Process & event log (RFC 0003)
kernel.agent.create
kernel.agent.get
kernel.agent.list
kernel.agent.transition
kernel.agent.terminate
kernel.events.append
kernel.events.tail
kernel.info

# Capability (RFC 0005)
kernel.cap.create
kernel.cap.derive
kernel.cap.get
kernel.cap.check_tool
kernel.cap.check_model
kernel.cap.check_net
kernel.cap.check_fs

# ResourceLedger (RFC 0006)
kernel.ledger.create
kernel.ledger.charge
kernel.ledger.check
kernel.ledger.get
kernel.ledger.list_breached
kernel.ledger.refund
kernel.ledger.update_grant

# Scheduler (RFC 0007)
kernel.sched.enqueue
kernel.sched.claim
kernel.sched.complete
kernel.sched.cancel
kernel.sched.get
kernel.sched.list
kernel.sched.gc_expired

# Mailbox (RFC 0009)
kernel.mbox.create
kernel.mbox.delete
kernel.mbox.subscribe
kernel.mbox.unsubscribe
kernel.mbox.list_subscriptions
kernel.mbox.send
kernel.mbox.publish
kernel.mbox.recv
kernel.mbox.peek
kernel.mbox.gc_expired

# Registry (RFC 0010)
kernel.registry.register
kernel.registry.unregister
kernel.registry.unregister_pid
kernel.registry.lookup
kernel.registry.list

# AgentFS (RFC 0011)
kernel.fs.write
kernel.fs.read
kernel.fs.stat
kernel.fs.exists
kernel.fs.list
kernel.fs.delete
kernel.fs.set_mode
kernel.fs.gc_orphaned

# Observability (RFC 0012)
kernel.observe.proc
kernel.observe.summary
kernel.observe.trace
kernel.observe.prometheus

# API stability (this RFC)
kernel.api.list_methods
kernel.api.version_info
```

(58 methods at v1.0.)

## 4. Contract test

`tests/test_kernel_api_contract.py`:

```python
from kernel.contract import (
    STABLE_METHODS, EXPERIMENTAL_METHODS, DEPRECATED_METHODS,
    verify_contract,
)

def test_no_undocumented_methods():
    """Every method registered after register_with_daemon must appear
    in one of the three tier sets. New methods MUST update contract.py
    before merging."""
    # Spin up a kernel-enabled daemon, get the registry,
    # call verify_contract.
    result = verify_contract(daemon.rpc)
    assert result["extra"] == [], (
        "These methods exist on the daemon but aren't classified in "
        f"contract.py — add them to STABLE/EXPERIMENTAL/DEPRECATED: "
        f"{result['extra']}"
    )
    assert result["missing"] == [], (
        "These methods are documented in contract.py but aren't "
        f"registered — implementation drift: {result['missing']}"
    )

def test_v1_no_deprecated_methods():
    """v1.0 ships nothing as deprecated."""
    assert DEPRECATED_METHODS == frozenset()
```

## 5. RPC surface

```
kernel.api.list_methods
  params: { tier?=null }      # null = all; 'stable' | 'experimental' | 'deprecated'
  result: { methods: [str], tier_counts: { stable, experimental, deprecated } }

kernel.api.version_info
  params: {}
  result: {
    kernel_version:     str,
    schema_version:     int,
    api_version:        str,    # daemon's API_VERSION (RFC 0001)
    method_count:       int,
    tier_counts:        { stable, experimental, deprecated },
    rfcs_implemented:   [int],  # [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  }
```

## 6. Deprecation cycle

When a method becomes obsolete:

1. PR moves it from `STABLE_METHODS` to `DEPRECATED_METHODS` in
   `contract.py` (one-line move). The method's implementation stays.
   The contract test is updated to allow it.
2. Changelog entry calls out the deprecation, the replacement
   (if any), and the removal version.
3. The kernel emits a `kernel.api.deprecated_call` event (kind in
   `agent_events`) on every call to a deprecated method, with
   pid=originator pid and `payload={"method": "..."}`. The
   supervisor / web UI can surface this to the user.
4. **Next minor version**: PR removes the method registration from
   `kernel/<module>.py::register()` and removes the entry from
   `DEPRECATED_METHODS`.
5. Clients that were ignoring the deprecation now see
   `METHOD_NOT_FOUND` (-32601) on call.

For v1.0 → v1.1 (planned ~6 months out), the cycle gives users one
minor version's worth of warning. In v1.x → v2.0 (much later), a
breaking change might use longer deprecation; the policy here covers
the minimum.

## 7. Backwards compatibility

- No schema change.
- Adds two new RPC methods to the registry.
- The contract test is the only enforcement; nothing prevents a
  developer from skipping it locally, but CI will catch them.

## 8. Open questions

1. **Should the test fail on `extra` or just warn?** Failing is the
   safer default — it catches genuine drift. Pushback only if the
   policy turns out to be too rigid for typical PR flows.
2. **Should `version_info` include the kernel.db file path?** Lean:
   no — clients shouldn't depend on filesystem layout. The daemon's
   discovery file already exposes the data dir for tools that
   need it.
3. **Schema version vs API version.** They're decoupled: schema can
   bump (additive) without changing API; API can deprecate methods
   without changing schema. Both are exposed in `version_info`.

## 9. Acceptance criteria

1. `STABLE_METHODS` contains all 58 v1.0 methods.
2. `EXPERIMENTAL_METHODS` and `DEPRECATED_METHODS` are empty.
3. `verify_contract` returns `{missing: [], extra: []}` against a
   live kernel-enabled daemon.
4. `kernel.api.list_methods` returns all methods (or filtered by
   tier).
5. `kernel.api.version_info` returns the documented shape.
6. The contract test fails if a developer adds a method to a kernel
   module's `register()` without updating `contract.py`.
7. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
