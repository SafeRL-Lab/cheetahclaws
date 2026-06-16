"""api.py — Kernel facade: one class instead of seven.

Through Phase 5 + RFC 0016/0017, the kernel grew to nine independent
stores plus a supervisor and worker loop. Every test fixture and the
``kernel.integration.register_with_daemon`` helper builds them up
in the same shape:

    KernelStore.open(db) → CapabilityStore + LedgerStore +
    SchedulerStore + MailboxStore + RegistryStore + AgentFSStore →
    ObservabilityStore → contract.register → optional supervisor +
    worker.

This module collapses that boilerplate into one class. It is **purely
additive**: existing direct-store usage continues to work, and every
piece of behaviour is owned by the underlying RFC modules. The facade
just wires.

Usage:

    >>> kernel = Kernel.open("~/.cheetahclaws/kernel.db")
    >>> agent = kernel.create_agent(name="alice", template="research")
    >>> kernel.cap.create(pid=agent.pid, tool_grants=["Read"])
    >>> kernel.ledger.create(pid=agent.pid, grants={"tokens": 100_000})
    >>> kernel.close()

Use ``Kernel.attach_to_daemon(daemon_state)`` from inside a daemon
process to register all RPC methods + stash references on the
DaemonState — same end result as ``register_with_daemon`` but
called via the facade.

Supervisor + WorkerLoop are not started by default — they require
factories (argv / policy / env) the application owns.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .agentfs import AgentFSStore
from .capability import CapabilityStore
from .ledger import LedgerStore
from .mailbox import MailboxStore
from .observability import ObservabilityStore
from .process import AgentState
from .registry import RegistryStore
from .scheduler import SchedulerStore
from .store import KernelStore

if TYPE_CHECKING:
    from .runner import RunnerSupervisor
    from .worker import WorkerLoop


class Kernel:
    """One-stop facade over the v1 kernel substrate.

    Construction is via ``Kernel.open(db_path)``; passing a pre-built
    KernelStore works too via ``Kernel.from_kernel_store(...)`` for
    advanced setups (e.g. tests sharing fixtures).

    The facade does NOT auto-start the supervisor or worker loop.
    Those require application-supplied factories (argv, policy, env)
    that decide how to spawn each agent.
    """

    # ── construction ──────────────────────────────────────────────────

    def __init__(
        self,
        kernel_store: KernelStore,
        *,
        publish_to_bus: bool = False,
    ) -> None:
        self.process    = kernel_store
        self.cap        = CapabilityStore(
            kernel_store.connection, write_lock=kernel_store.write_lock,
        )
        self.ledger     = LedgerStore(
            kernel_store.connection, write_lock=kernel_store.write_lock,
        )
        self.scheduler  = SchedulerStore(
            kernel_store.connection, write_lock=kernel_store.write_lock,
        )
        self.mailbox    = MailboxStore(
            kernel_store.connection, write_lock=kernel_store.write_lock,
        )
        self.registry   = RegistryStore(
            kernel_store.connection, write_lock=kernel_store.write_lock,
        )
        self.fs         = AgentFSStore(
            kernel_store.connection, write_lock=kernel_store.write_lock,
            ledger=self.ledger,
        )
        self.observability = ObservabilityStore(
            kernel_store=self.process,
            capability_store=self.cap,
            ledger_store=self.ledger,
            scheduler_store=self.scheduler,
            mailbox_store=self.mailbox,
            registry_store=self.registry,
            agentfs_store=self.fs,
        )
        # Optional pieces — created on demand via factories.
        self._supervisor: Optional["RunnerSupervisor"] = None
        self._worker:     Optional["WorkerLoop"]      = None
        self._closed = False

    @classmethod
    def open(
        cls,
        db_path: str | Path,
        *,
        publish_to_bus: bool = False,
    ) -> "Kernel":
        """Open or create kernel.db at ``db_path``, run schema
        migration, return a fully-wired Kernel.

        ``publish_to_bus`` only takes effect when ``attach_to_daemon``
        is called later; standalone callers don't have a bus.
        """
        bus = None
        if publish_to_bus:
            try:
                from cheetahclaws.daemon import events as _events
                bus = _events.get_bus()
            except Exception:
                bus = None
        ks = KernelStore.open(db_path, bus=bus)
        return cls(ks, publish_to_bus=publish_to_bus)

    @classmethod
    def from_kernel_store(cls, kernel_store: KernelStore) -> "Kernel":
        """Wrap an existing KernelStore. Useful for tests sharing a
        fixture or for advanced setups that pre-customise the
        connection / bus."""
        return cls(kernel_store)

    # ── supervisor + worker (lazy) ────────────────────────────────────

    def make_supervisor(
        self, **kwargs,
    ) -> "RunnerSupervisor":
        """Construct (or return cached) RunnerSupervisor.

        kwargs are passed through to ``RunnerSupervisor.__init__``;
        ``ledger_store`` is auto-supplied if not given. Pass
        ``tool_registry=...`` (and the facade is auto-wired as
        ``tool_kernel``) to enable RFC 0021 tool dispatch.
        """
        if self._supervisor is not None:
            return self._supervisor
        from .runner import RunnerSupervisor
        kwargs.setdefault("ledger_store", self.ledger)
        if "tool_registry" in kwargs and "tool_kernel" not in kwargs:
            kwargs["tool_kernel"] = self
        self._supervisor = RunnerSupervisor(self.process, **kwargs)
        return self._supervisor

    def make_worker(
        self,
        *,
        argv_factory,
        policy_factory=None,
        env_factory=None,
        **kwargs,
    ) -> "WorkerLoop":
        """Construct (or return cached) WorkerLoop. Auto-creates the
        supervisor if needed."""
        if self._worker is not None:
            return self._worker
        from .worker import WorkerLoop
        sup = self.make_supervisor()
        self._worker = WorkerLoop(
            kernel_store=self.process,
            scheduler_store=self.scheduler,
            supervisor=sup,
            argv_factory=argv_factory,
            policy_factory=policy_factory,
            env_factory=env_factory,
            **kwargs,
        )
        return self._worker

    # ── daemon attachment ────────────────────────────────────────────

    def attach_to_daemon(self, daemon_state) -> None:
        """Register every kernel.* RPC method on the daemon's RPC
        registry and stash store references on ``daemon_state``.

        After this call the daemon serves the full kernel surface.
        Equivalent in spirit to ``kernel.register_with_daemon`` but
        uses the facade's already-built stores.
        """
        from . import (
            capability as _cap,
            ledger as _ledger,
            scheduler as _sched,
            mailbox as _mbox,
            registry as _reg,
            agentfs as _fs,
            observability as _obs,
            contract as _contract,
            methods as _methods,
        )
        _methods.register(daemon_state.rpc, self.process)
        _cap.register(daemon_state.rpc, self.cap)
        _ledger.register(daemon_state.rpc, self.ledger)
        _sched.register(daemon_state.rpc, self.scheduler)
        _mbox.register(daemon_state.rpc, self.mailbox)
        _reg.register(daemon_state.rpc, self.registry)
        _fs.register(daemon_state.rpc, self.fs)
        _obs.register(daemon_state.rpc, self.observability)
        _contract.register(daemon_state.rpc)
        # Stash references for tooling that pokes at the daemon state.
        setattr(daemon_state, "kernel",            self)
        setattr(daemon_state, "kernel_store",      self.process)
        setattr(daemon_state, "capability_store",  self.cap)
        setattr(daemon_state, "ledger_store",      self.ledger)
        setattr(daemon_state, "scheduler_store",   self.scheduler)
        setattr(daemon_state, "mailbox_store",     self.mailbox)
        setattr(daemon_state, "registry_store",    self.registry)
        setattr(daemon_state, "agentfs_store",     self.fs)
        setattr(daemon_state, "observability_store", self.observability)

    # ── convenience helpers ──────────────────────────────────────────

    def create_agent(
        self,
        *,
        name: str,
        template: str,
        parent_pid: Optional[int] = None,
        metadata: Optional[dict] = None,
    ):
        """Sugar for ``self.process.create(...)``."""
        return self.process.create(
            name=name, template=template,
            parent_pid=parent_pid, metadata=metadata,
        )

    def info(self) -> dict:
        """Combined system snapshot. Equivalent to
        ``observability.summary()`` plus a few facade-specific facts."""
        s = self.observability.summary()
        s["facade"] = {
            "supervisor_active": self._supervisor is not None,
            "worker_active":     self._worker is not None,
        }
        return s

    # ── shutdown ─────────────────────────────────────────────────────

    def close(
        self, *,
        worker_drain: bool = True,
        worker_drain_timeout_s: float = 30.0,
    ) -> None:
        """Stop the worker (if running), close the connection. Safe to
        call repeatedly."""
        if self._closed:
            return
        if self._worker is not None:
            try:
                self._worker.stop(
                    drain=worker_drain,
                    drain_timeout_s=worker_drain_timeout_s,
                )
            except Exception:
                pass
            self._worker = None
        # Supervisor doesn't need explicit shutdown — its handles are
        # subprocesses. If the worker drained, there shouldn't be any.
        # Existing handles will be reaped on process exit.
        self._supervisor = None
        try:
            self.process.close()
        except Exception:
            pass
        self._closed = True

    # ── context manager ──────────────────────────────────────────────

    def __enter__(self) -> "Kernel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["Kernel"]
