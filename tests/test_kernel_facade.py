"""Tests for kernel.api.Kernel facade."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from cheetahclaws.kernel import (
    AgentState,
    Kernel,
    KernelStore,
    SCHEMA_VERSION,
)


# ── Lifecycle ────────────────────────────────────────────────────────────


def test_open_creates_db_with_schema(tmp_path):
    db = tmp_path / "kernel.db"
    assert not db.exists()
    k = Kernel.open(db)
    try:
        assert db.exists()
        assert k.info()["schema_version"] == SCHEMA_VERSION
    finally:
        k.close()


def test_context_manager(tmp_path):
    db = tmp_path / "kernel.db"
    with Kernel.open(db) as k:
        a = k.create_agent(name="x", template="t")
        assert a.state == AgentState.READY


def test_close_idempotent(tmp_path):
    k = Kernel.open(tmp_path / "kernel.db")
    k.close()
    k.close()  # no error


def test_from_kernel_store(tmp_path):
    """Wrap an existing KernelStore — useful when tests share the
    underlying connection across fixtures."""
    ks = KernelStore.open(tmp_path / "kernel.db")
    try:
        k = Kernel.from_kernel_store(ks)
        a = k.create_agent(name="x", template="t")
        assert k.process.get(a.pid).pid == a.pid
    finally:
        ks.close()


# ── Store accessors ──────────────────────────────────────────────────────


def test_all_stores_accessible(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        # Just verify each store exists and isn't None.
        assert k.process is not None
        assert k.cap is not None
        assert k.ledger is not None
        assert k.scheduler is not None
        assert k.mailbox is not None
        assert k.registry is not None
        assert k.fs is not None
        assert k.observability is not None


def test_stores_share_connection(tmp_path):
    """All stores must use the same underlying connection — required
    for cross-store atomic transactions and the shared write lock."""
    with Kernel.open(tmp_path / "kernel.db") as k:
        conn = k.process.connection
        # Capability and ledger stores expose ._conn (private but
        # checkable at the test layer).
        assert k.cap._conn is conn
        assert k.ledger._conn is conn
        assert k.scheduler._conn is conn
        assert k.mailbox._conn is conn
        assert k.registry._conn is conn
        assert k.fs._conn is conn


def test_stores_share_write_lock(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        wl = k.process.write_lock
        assert k.cap._lock is wl
        assert k.ledger._lock is wl
        assert k.scheduler._lock is wl
        assert k.mailbox._lock is wl
        assert k.registry._lock is wl
        assert k.fs._lock is wl


# ── Convenience helpers ─────────────────────────────────────────────────


def test_create_agent_sugar(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        a = k.create_agent(name="alice", template="t",
                            metadata={"role": "researcher"})
        fetched = k.process.get(a.pid)
        assert fetched.name == "alice"
        assert fetched.metadata == {"role": "researcher"}


def test_info_combines_summary_and_facade(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        info = k.info()
        # Summary fields.
        assert "schema_version" in info
        assert "agents" in info
        assert "scheduler" in info
        # Facade-specific.
        assert info["facade"] == {
            "supervisor_active": False,
            "worker_active":     False,
        }


# ── Lazy supervisor + worker ────────────────────────────────────────────


def test_make_supervisor_caches(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        s1 = k.make_supervisor()
        s2 = k.make_supervisor()
        assert s1 is s2


@pytest.mark.skipif(os.name != "posix",
                    reason="worker spawns POSIX subprocesses")
def test_make_worker_after_make_supervisor(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        sup = k.make_supervisor()
        worker = k.make_worker(
            argv_factory=lambda e: [
                sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main",
            ],
            max_concurrent=1,
        )
        assert worker is not None
        # Worker re-uses the cached supervisor.
        assert k.make_supervisor() is sup


@pytest.mark.skipif(os.name != "posix",
                    reason="worker spawns POSIX subprocesses")
def test_close_stops_worker(tmp_path):
    """Kernel.close() halts the worker if started."""
    k = Kernel.open(tmp_path / "kernel.db")
    worker = k.make_worker(
        argv_factory=lambda e: [
            sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main",
        ],
    )
    worker.start()
    k.close()
    # No assertions on worker state — just that close() didn't hang.


# ── Daemon attachment ──────────────────────────────────────────────────


def test_attach_to_daemon_registers_methods(tmp_path):
    """attach_to_daemon registers all kernel.* methods on the
    daemon's RPC registry."""
    import socket
    import uuid
    from cheetahclaws.daemon import events as _events
    from cheetahclaws.daemon.server import make_tcp_server
    from cheetahclaws.kernel import verify_contract

    _events.reset_bus_for_tests()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = make_tcp_server(
        "127.0.0.1", port, data_dir=tmp_path,
        token="t-" + uuid.uuid4().hex, audit_enabled=True,
    )
    k = Kernel.open(tmp_path / "kernel.db")
    try:
        k.attach_to_daemon(server.daemon_state)
        # Contract verification: every documented method registered.
        result = verify_contract(server.daemon_state.rpc)
        assert result["missing"] == []
        assert result["extra"] == []
        # Stash references match.
        assert server.daemon_state.kernel is k
        assert server.daemon_state.kernel_store is k.process
    finally:
        # Don't call server.shutdown() — serve_forever was never started
        # (see test_kernel_api_contract.py for the rationale).
        server.server_close()
        k.close()


# ── End-to-end example as a test ───────────────────────────────────────


@pytest.mark.skipif(os.name != "posix",
                    reason="example spawns POSIX subprocesses")
def test_e2e_smoke_example_runs(tmp_path):
    """Run examples/kernel_e2e_smoke.py via subprocess, expect exit 0."""
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "examples.kernel_e2e_smoke"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"example failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    # Spot-check key milestones in the output.
    assert "All assertions passed" in result.stdout
    assert "agents.DEAD:    2" in result.stdout
    assert "scheduler.completed: 2" in result.stdout
