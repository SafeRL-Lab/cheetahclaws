"""contract.py — frozen API surface + version_info / list_methods RPC (RFC 0013).

This file is the **single source of truth** for the kernel's stable
RPC contract at v1.0. Every method registered by
``register_with_daemon`` MUST appear in exactly one of the three
tier sets below. The contract test (tests/test_kernel_api_contract.py)
fails CI when a developer adds a method without classifying it here,
or removes one without going through deprecation.

Deprecation cycle is documented in RFC 0013 §6.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import InvalidPayload

if TYPE_CHECKING:
    from cheetahclaws.daemon.rpc import CallContext, RpcRegistry


# ── Frozen v1.0 method registry ────────────────────────────────────────────


STABLE_METHODS: frozenset = frozenset({
    # Process & event log (RFC 0003)
    "kernel.agent.create",
    "kernel.agent.get",
    "kernel.agent.list",
    "kernel.agent.transition",
    "kernel.agent.terminate",
    "kernel.events.append",
    "kernel.events.tail",
    "kernel.info",
    # Capability (RFC 0005)
    "kernel.cap.create",
    "kernel.cap.derive",
    "kernel.cap.get",
    "kernel.cap.check_tool",
    "kernel.cap.check_model",
    "kernel.cap.check_net",
    "kernel.cap.check_fs",
    # ResourceLedger (RFC 0006)
    "kernel.ledger.create",
    "kernel.ledger.charge",
    "kernel.ledger.check",
    "kernel.ledger.get",
    "kernel.ledger.list_breached",
    "kernel.ledger.refund",
    "kernel.ledger.update_grant",
    # Scheduler (RFC 0007)
    "kernel.sched.enqueue",
    "kernel.sched.claim",
    "kernel.sched.complete",
    "kernel.sched.cancel",
    "kernel.sched.get",
    "kernel.sched.list",
    "kernel.sched.gc_expired",
    # Mailbox (RFC 0009)
    "kernel.mbox.create",
    "kernel.mbox.delete",
    "kernel.mbox.subscribe",
    "kernel.mbox.unsubscribe",
    "kernel.mbox.list_subscriptions",
    "kernel.mbox.send",
    "kernel.mbox.publish",
    "kernel.mbox.recv",
    "kernel.mbox.peek",
    "kernel.mbox.gc_expired",
    # Registry (RFC 0010)
    "kernel.registry.register",
    "kernel.registry.unregister",
    "kernel.registry.unregister_pid",
    "kernel.registry.lookup",
    "kernel.registry.list",
    # AgentFS (RFC 0011)
    "kernel.fs.write",
    "kernel.fs.read",
    "kernel.fs.stat",
    "kernel.fs.exists",
    "kernel.fs.list",
    "kernel.fs.delete",
    "kernel.fs.set_mode",
    "kernel.fs.gc_orphaned",
    # Observability (RFC 0012)
    "kernel.observe.proc",
    "kernel.observe.summary",
    "kernel.observe.trace",
    "kernel.observe.prometheus",
    # API stability (RFC 0013)
    "kernel.api.list_methods",
    "kernel.api.version_info",
})


EXPERIMENTAL_METHODS: frozenset = frozenset()
DEPRECATED_METHODS:   frozenset = frozenset()


# Convenience: the union, used by verify_contract for membership checks.
ALL_KNOWN_METHODS: frozenset = (
    STABLE_METHODS | EXPERIMENTAL_METHODS | DEPRECATED_METHODS
)


# RFC numbers implemented at v1.0 (in order). Used by version_info.
# RFC 0016 (subprocess agent runner) is a Python-API-only addition; it
# adds no kernel.* RPC methods, so STABLE_METHODS is unchanged. The
# RFC is still listed here so version_info reports it.
RFCS_IMPLEMENTED: tuple = (3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)


# ── Verification helper ────────────────────────────────────────────────────


def verify_contract(rpc_registry) -> dict:
    """Compare a live ``RpcRegistry`` against the frozen sets.

    Returns ``{"missing": [...], "extra": [...], "by_tier": {...}}``.

    - ``missing``: methods documented in ALL_KNOWN_METHODS but absent
      from the registry. Implementation drifted.
    - ``extra``: methods registered but not classified. Developer
      forgot to update contract.py.
    - ``by_tier``: classification of the registered kernel.* methods.

    Methods outside the ``kernel.*`` namespace are ignored (e.g. the
    daemon's existing ``system.*``, ``echo.*``, ``permission.*``
    surfaces are not part of the kernel contract).
    """
    live = set(rpc_registry.methods())
    live_kernel = {m for m in live if m.startswith("kernel.")}
    missing = sorted(ALL_KNOWN_METHODS - live_kernel)
    extra   = sorted(live_kernel - ALL_KNOWN_METHODS)
    by_tier = {
        "stable":       sorted(live_kernel & STABLE_METHODS),
        "experimental": sorted(live_kernel & EXPERIMENTAL_METHODS),
        "deprecated":   sorted(live_kernel & DEPRECATED_METHODS),
        "unclassified": sorted(extra),
    }
    return {
        "missing": missing,
        "extra":   extra,
        "by_tier": by_tier,
    }


# ── RPC registration ───────────────────────────────────────────────────────


def register(registry: "RpcRegistry") -> None:
    from .errors import KernelError
    from . import KERNEL_VERSION, SCHEMA_VERSION

    def _translate(fn):
        def wrapper(params, ctx):
            try:
                return fn(params, ctx)
            except InvalidPayload as e:
                raise TypeError(str(e))
            except KernelError as e:
                raise RuntimeError(f"{type(e).__name__}: {e}")
        wrapper.__name__ = fn.__name__
        return wrapper

    @_translate
    def api_list_methods(params, ctx):
        tier = params.get("tier")
        if tier is None:
            methods = sorted(ALL_KNOWN_METHODS)
        elif tier == "stable":
            methods = sorted(STABLE_METHODS)
        elif tier == "experimental":
            methods = sorted(EXPERIMENTAL_METHODS)
        elif tier == "deprecated":
            methods = sorted(DEPRECATED_METHODS)
        else:
            raise InvalidPayload(
                "tier must be 'stable' | 'experimental' | 'deprecated' or null",
                field="tier",
            )
        return {
            "methods": methods,
            "tier_counts": {
                "stable":       len(STABLE_METHODS),
                "experimental": len(EXPERIMENTAL_METHODS),
                "deprecated":   len(DEPRECATED_METHODS),
            },
        }

    @_translate
    def api_version_info(params, ctx):
        # Best-effort daemon API version pull-in. The daemon module
        # may not be importable in pure-kernel test setups, so we
        # fall back to None.
        try:
            from cheetahclaws.daemon import API_VERSION as _api
            api_version = _api
        except Exception:
            api_version = None
        return {
            "kernel_version":   KERNEL_VERSION,
            "schema_version":   SCHEMA_VERSION,
            "api_version":      api_version,
            "method_count":     len(ALL_KNOWN_METHODS),
            "tier_counts": {
                "stable":       len(STABLE_METHODS),
                "experimental": len(EXPERIMENTAL_METHODS),
                "deprecated":   len(DEPRECATED_METHODS),
            },
            "rfcs_implemented": list(RFCS_IMPLEMENTED),
        }

    registry.register("kernel.api.list_methods", api_list_methods)
    registry.register("kernel.api.version_info", api_version_info)
