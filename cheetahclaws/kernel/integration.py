"""integration.py — Glue that activates the kernel inside daemon.

Called from ``daemon/cli.py`` only when ``cheetahclaws serve
--enable-kernel`` is passed. The function is small by design: open the
DB, run recovery, register methods. Anything more lives in the modules
this glue calls into.

The only existing module touched by this RFC is ``daemon/cli.py``,
which gains one argparse flag and one conditional call.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .store import KernelStore, RECOVERY_SUSPEND, _RECOVERY_POLICIES

if TYPE_CHECKING:
    from cheetahclaws.daemon.server import DaemonState

log = logging.getLogger(__name__)


def register_with_daemon(
    daemon_state: "DaemonState",
    db_path: str | Path,
    *,
    recovery: str = RECOVERY_SUSPEND,
    publish_to_bus: bool = True,
) -> KernelStore:
    """Bring the kernel up inside an already-built ``DaemonState``.

    1. Open / init kernel.db and run startup recovery.
    2. Register kernel.* methods on the daemon's RPC registry.
    3. Stash the store on ``daemon_state.kernel_store`` for inspection
       and clean shutdown.

    Returns the live ``KernelStore`` so callers can hold a reference
    (e.g. for tests).
    """
    if recovery not in _RECOVERY_POLICIES:
        raise ValueError(
            f"unknown recovery policy: {recovery!r} "
            f"(use one of {_RECOVERY_POLICIES})"
        )

    bus = None
    if publish_to_bus:
        # Lazy import: only the kernel-enabled path pulls in the bus.
        from cheetahclaws.daemon import events as _events
        bus = _events.get_bus()

    store = KernelStore.open(db_path, bus=bus)
    recovered = store.recover(policy=recovery)
    if recovered:
        log.info("kernel: recovered %d stale agent(s) at startup "
                 "(policy=%s)", recovered, recovery)

    # Register the methods on the daemon's existing registry. This is
    # additive: existing methods (system.*, echo.*, permission.*) keep
    # working unchanged.
    from . import methods as _methods
    _methods.register(daemon_state.rpc, store)

    # Phase 2: capability + ledger. Phase 3: scheduler + mailbox +
    # registry. Phase 4: agentfs. All share the kernel store's
    # connection AND write lock — see CapabilityStore docstring for the
    # implicit-transaction reason.
    from . import capability as _cap
    from . import ledger as _ledger
    from . import scheduler as _sched
    from . import mailbox as _mbox
    from . import registry as _reg
    from . import agentfs as _fs
    cap_store = _cap.CapabilityStore(
        store.connection, write_lock=store.write_lock,
    )
    ledger_store = _ledger.LedgerStore(
        store.connection, write_lock=store.write_lock,
    )
    sched_store = _sched.SchedulerStore(
        store.connection, write_lock=store.write_lock,
    )
    mbox_store = _mbox.MailboxStore(
        store.connection, write_lock=store.write_lock,
    )
    reg_store = _reg.RegistryStore(
        store.connection, write_lock=store.write_lock,
    )
    fs_store = _fs.AgentFSStore(
        store.connection, write_lock=store.write_lock,
        ledger=ledger_store,    # enables fs_w_bytes quota when configured
    )
    _cap.register(daemon_state.rpc, cap_store)
    _ledger.register(daemon_state.rpc, ledger_store)
    _sched.register(daemon_state.rpc, sched_store)
    _mbox.register(daemon_state.rpc, mbox_store)
    _reg.register(daemon_state.rpc, reg_store)
    _fs.register(daemon_state.rpc, fs_store)

    # Phase 5: observability + API contract. Observability needs every
    # other store wired in to compute combined views. Contract is
    # stateless.
    from . import observability as _obs
    from . import contract as _contract
    obs_store = _obs.ObservabilityStore(
        kernel_store=store,
        capability_store=cap_store,
        ledger_store=ledger_store,
        scheduler_store=sched_store,
        mailbox_store=mbox_store,
        registry_store=reg_store,
        agentfs_store=fs_store,
    )
    _obs.register(daemon_state.rpc, obs_store)
    _contract.register(daemon_state.rpc)

    # Expose the stores on the DaemonState. We do this dynamically
    # rather than adding fields to ``DaemonState.__init__`` so the
    # existing constructor signature is unchanged. Tests and tooling
    # that expect ``daemon_state.kernel_store`` /
    # ``daemon_state.capability_store`` / ``daemon_state.ledger_store``
    # / ``daemon_state.scheduler_store`` only see them when the kernel
    # is enabled; otherwise a normal AttributeError fires (intentional).
    setattr(daemon_state, "kernel_store",     store)
    setattr(daemon_state, "capability_store", cap_store)
    setattr(daemon_state, "ledger_store",     ledger_store)
    setattr(daemon_state, "scheduler_store",  sched_store)
    setattr(daemon_state, "mailbox_store",    mbox_store)
    setattr(daemon_state, "registry_store",   reg_store)
    setattr(daemon_state, "agentfs_store",    fs_store)
    setattr(daemon_state, "observability_store", obs_store)

    log.info("kernel %s registered with daemon (db=%s, recovery=%s)",
             _kernel_version(), db_path, recovery)
    return store


def _kernel_version() -> str:
    from . import KERNEL_VERSION
    return KERNEL_VERSION


def detach(daemon_state: "DaemonState") -> Optional[KernelStore]:
    """Inverse of register_with_daemon — used by tests on teardown.

    Closes the SQLite connection and removes the attribute. Methods
    registered on the RPC registry are not removed (the registry has no
    unregister API and the kernel.* names won't collide with anything
    else). Capability and ledger stores share the kernel store's
    connection so closing the kernel closes them too; we just drop the
    attribute references.
    """
    store: Optional[KernelStore] = getattr(daemon_state, "kernel_store", None)
    if store is None:
        return None
    try:
        store.close()
    finally:
        for attr in ("kernel_store", "capability_store",
                     "ledger_store", "scheduler_store",
                     "mailbox_store", "registry_store",
                     "agentfs_store", "observability_store"):
            try:
                delattr(daemon_state, attr)
            except AttributeError:
                pass
    return store
