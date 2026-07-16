"""Tests for kernel.registry (RFC 0010)."""
from __future__ import annotations

import pytest

from cheetahclaws.kernel import (
    KernelStore,
    RegistryEntry,
    RegistryInvalidName,
    RegistryNameExists,
    RegistryNotFound,
    RegistryStore,
    UnknownPid,
)


@pytest.fixture
def stores(tmp_path):
    ks = KernelStore.open(tmp_path / "kernel.db")
    rg = RegistryStore(ks.connection, write_lock=ks.write_lock)
    yield ks, rg
    ks.close()


# ── register / lookup ────────────────────────────────────────────────────


def test_register_lookup_round_trip(stores):
    ks, rg = stores
    a = ks.create(name="alice", template="t")
    rg.register(name="/agents/alice", pid=a.pid,
                tags=["research", "v2"],
                metadata={"role": "researcher"})
    e = rg.lookup("/agents/alice")
    assert isinstance(e, RegistryEntry)
    assert e.name == "/agents/alice"
    assert e.pid == a.pid
    assert set(e.tags) == {"research", "v2"}
    assert e.metadata == {"role": "researcher"}


def test_resolve_pid(stores):
    ks, rg = stores
    a = ks.create(name="alice", template="t")
    rg.register(name="/agents/alice", pid=a.pid)
    assert rg.resolve_pid("/agents/alice") == a.pid


def test_register_duplicate_name_rejected(stores):
    ks, rg = stores
    a = ks.create(name="x", template="t")
    b = ks.create(name="y", template="t")
    rg.register(name="/x", pid=a.pid)
    with pytest.raises(RegistryNameExists):
        rg.register(name="/x", pid=b.pid)


def test_register_unknown_pid(stores):
    _, rg = stores
    with pytest.raises(UnknownPid):
        rg.register(name="/x", pid=9999)


def test_lookup_unknown_raises(stores):
    _, rg = stores
    with pytest.raises(RegistryNotFound):
        rg.lookup("/missing")


# ── name validation ──────────────────────────────────────────────────────


def test_empty_name_rejected(stores):
    _, rg = stores
    with pytest.raises(RegistryInvalidName):
        rg.register(name="", pid=1)


def test_nul_in_name_rejected(stores):
    _, rg = stores
    with pytest.raises(RegistryInvalidName):
        rg.register(name="bad\x00name", pid=1)


def test_control_char_in_name_rejected(stores):
    _, rg = stores
    with pytest.raises(RegistryInvalidName):
        rg.register(name="bad\nname", pid=1)


def test_oversize_name_rejected(stores):
    _, rg = stores
    big = "/" + "a" * 1000
    with pytest.raises(RegistryInvalidName):
        rg.register(name=big, pid=1)


def test_unicode_name_accepted(stores):
    """Non-ASCII printable is fine; only NUL/control are blocked."""
    ks, rg = stores
    a = ks.create(name="x", template="t")
    rg.register(name="/agents/中文", pid=a.pid)
    assert rg.lookup("/agents/中文").pid == a.pid


# ── tags ─────────────────────────────────────────────────────────────────


def test_tags_dedup(stores):
    ks, rg = stores
    a = ks.create(name="x", template="t")
    rg.register(name="/x", pid=a.pid, tags=["a", "a", "b"])
    e = rg.lookup("/x")
    assert sorted(e.tags) == ["a", "b"]


def test_tag_filter_in_list(stores):
    ks, rg = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    c = ks.create(name="c", template="t")
    rg.register(name="/agents/a", pid=a.pid, tags=["research"])
    rg.register(name="/agents/b", pid=b.pid, tags=["bridge"])
    rg.register(name="/agents/c", pid=c.pid, tags=["research", "v2"])
    entries, total = rg.list(tag="research")
    assert total == 2
    names = {e.name for e in entries}
    assert names == {"/agents/a", "/agents/c"}


def test_too_many_tags_rejected(stores):
    ks, rg = stores
    a = ks.create(name="x", template="t")
    too_many = [f"t{i}" for i in range(40)]
    with pytest.raises(RegistryInvalidName):
        rg.register(name="/x", pid=a.pid, tags=too_many)


# ── prefix list ──────────────────────────────────────────────────────────


def test_list_prefix_filter(stores):
    ks, rg = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    rg.register(name="/agents/research/a", pid=a.pid)
    rg.register(name="/agents/research/b", pid=b.pid)
    rg.register(name="/services/x",        pid=a.pid)
    entries, total = rg.list(prefix="/agents/research/")
    assert total == 2
    assert all(e.name.startswith("/agents/research/") for e in entries)


def test_list_prefix_with_special_chars(stores):
    """LIKE wildcards in the prefix must be escaped."""
    ks, rg = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    rg.register(name="/agents/100%match", pid=a.pid)
    rg.register(name="/agents/100xmatch", pid=b.pid)
    entries, total = rg.list(prefix="/agents/100%")
    # Without escape, "%" would match anything ending in match.
    # With escape, only /agents/100%match matches the literal "100%".
    assert total == 1
    assert entries[0].name == "/agents/100%match"


def test_list_pagination(stores):
    ks, rg = stores
    a = ks.create(name="x", template="t")
    for i in range(5):
        rg.register(name=f"/n/{i}", pid=a.pid)
    page1, total = rg.list(limit=2, offset=0)
    page2, _ = rg.list(limit=2, offset=2)
    page3, _ = rg.list(limit=2, offset=4)
    assert total == 5
    assert len(page1) == 2 and len(page2) == 2 and len(page3) == 1


# ── unregister ───────────────────────────────────────────────────────────


def test_unregister_removes(stores):
    ks, rg = stores
    a = ks.create(name="x", template="t")
    rg.register(name="/x", pid=a.pid)
    assert rg.unregister("/x") == 1
    with pytest.raises(RegistryNotFound):
        rg.lookup("/x")


def test_unregister_idempotent(stores):
    _, rg = stores
    assert rg.unregister("/never") == 0


def test_unregister_pid_clears_all(stores):
    ks, rg = stores
    a = ks.create(name="x", template="t")
    b = ks.create(name="y", template="t")
    rg.register(name="/a/1", pid=a.pid)
    rg.register(name="/a/2", pid=a.pid)
    rg.register(name="/b/1", pid=b.pid)
    n = rg.unregister_pid(a.pid)
    assert n == 2
    # b's row remains.
    assert rg.lookup("/b/1").pid == b.pid


# ── multiple names per pid ───────────────────────────────────────────────


def test_multiple_names_per_pid_allowed(stores):
    ks, rg = stores
    a = ks.create(name="x", template="t")
    rg.register(name="/agents/x", pid=a.pid)
    rg.register(name="/services/research", pid=a.pid)
    assert rg.resolve_pid("/agents/x") == a.pid
    assert rg.resolve_pid("/services/research") == a.pid
