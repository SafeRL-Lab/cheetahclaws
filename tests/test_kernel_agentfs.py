"""Tests for kernel.agentfs (RFC 0011)."""
from __future__ import annotations

import threading

import pytest

from cheetahclaws.kernel import (
    AgentFSStore,
    DEFAULT_MAX_OBJECT_BYTES,
    FsAlreadyExists,
    FsInvalidPath,
    FsNotFound,
    FsObject,
    FsQuotaExceeded,
    FsReadOnly,
    KernelStore,
    LedgerStore,
    UnknownPid,
)


@pytest.fixture
def stores(tmp_path):
    ks = KernelStore.open(tmp_path / "kernel.db")
    led = LedgerStore(ks.connection, write_lock=ks.write_lock)
    fs = AgentFSStore(ks.connection, write_lock=ks.write_lock,
                      ledger=led, max_object_bytes=4096)
    yield ks, fs, led
    ks.close()


# ── write / read round-trip ──────────────────────────────────────────────


def test_write_read_round_trip(stores):
    ks, fs, _ = stores
    a = ks.create(name="alice", template="t")
    obj = fs.write(pid=a.pid, path="/memory/alice/note",
                   content=b"hello world")
    assert isinstance(obj, FsObject)
    assert obj.path == "/memory/alice/note"
    assert obj.size == 11
    assert obj.owner_pid == a.pid
    content, meta = fs.read("/memory/alice/note")
    assert content == b"hello world"
    assert meta.size == 11
    assert meta.accessed_at is not None


def test_write_string_encodes_utf8(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/x", content="héllo")  # type: ignore[arg-type]
    content, _ = fs.read("/x")
    assert content == "héllo".encode("utf-8")


def test_write_binary_payload(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    payload = bytes(range(256))
    fs.write(pid=a.pid, path="/x", content=payload)
    content, _ = fs.read("/x")
    assert content == payload


def test_write_unknown_pid(stores):
    _, fs, _ = stores
    with pytest.raises(UnknownPid):
        fs.write(pid=9999, path="/x", content=b"")


def test_write_replaces(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/x", content=b"v1")
    fs.write(pid=a.pid, path="/x", content=b"v2-longer")
    content, meta = fs.read("/x")
    assert content == b"v2-longer"
    assert meta.size == len(b"v2-longer")


def test_write_owner_unchanged_on_replace(stores):
    ks, fs, _ = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    fs.write(pid=a.pid, path="/shared/x", content=b"a")
    fs.write(pid=b.pid, path="/shared/x", content=b"b")
    obj = fs.stat("/shared/x")
    assert obj.owner_pid == a.pid   # original creator


# ── path validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_path", [
    "",
    "no-leading-slash",
    "/has\x00nul",
    "/has\nnewline",
    "/../escape",
    "/foo/../bar",
    "/foo/..",
    "/" + "x" * 2000,
])
def test_path_validation_rejects(stores, bad_path):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    with pytest.raises(FsInvalidPath):
        fs.write(pid=a.pid, path=bad_path, content=b"x")


def test_path_unicode_accepted(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/agents/中文/笔记", content=b"hi")
    assert fs.exists("/agents/中文/笔记")


# ── content size cap ─────────────────────────────────────────────────────


def test_oversize_rejected(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    too_big = b"a" * (DEFAULT_MAX_OBJECT_BYTES + 1) if False else b"x" * 5000
    # The fixture caps at 4096 bytes for tests.
    with pytest.raises(FsInvalidPath):
        fs.write(pid=a.pid, path="/x", content=too_big)


# ── if_absent ────────────────────────────────────────────────────────────


def test_if_absent_blocks_overwrite(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/x", content=b"v1")
    with pytest.raises(FsAlreadyExists):
        fs.write(pid=a.pid, path="/x", content=b"v2", if_absent=True)
    # Original survives.
    content, _ = fs.read("/x")
    assert content == b"v1"


def test_if_absent_creates_when_missing(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/x", content=b"v1", if_absent=True)
    assert fs.exists("/x")


# ── ro mode ──────────────────────────────────────────────────────────────


def test_ro_mode_blocks_subsequent_writes(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/x", content=b"v1", mode="ro")
    with pytest.raises(FsReadOnly):
        fs.write(pid=a.pid, path="/x", content=b"v2")


def test_set_mode_toggles(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/x", content=b"v1", mode="ro")
    fs.set_mode("/x", "rw")
    fs.write(pid=a.pid, path="/x", content=b"v2")
    content, meta = fs.read("/x")
    assert content == b"v2"
    assert meta.mode == "rw"


def test_set_mode_unknown_path(stores):
    _, fs, _ = stores
    with pytest.raises(FsNotFound):
        fs.set_mode("/nope", "ro")


def test_set_mode_invalid(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/x", content=b"")
    with pytest.raises(FsInvalidPath):
        fs.set_mode("/x", "x")


# ── stat / exists / read of missing ──────────────────────────────────────


def test_stat_unknown(stores):
    _, fs, _ = stores
    with pytest.raises(FsNotFound):
        fs.stat("/nope")


def test_read_unknown(stores):
    _, fs, _ = stores
    with pytest.raises(FsNotFound):
        fs.read("/nope")


def test_exists(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    assert fs.exists("/nope") is False
    fs.write(pid=a.pid, path="/x", content=b"")
    assert fs.exists("/x") is True


# ── list with prefix ─────────────────────────────────────────────────────


def test_list_prefix_filter(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    for p in ["/memory/a/1", "/memory/a/2", "/memory/b/1", "/skills/x"]:
        fs.write(pid=a.pid, path=p, content=b"x")
    entries, total = fs.list(prefix="/memory/a/")
    assert total == 2
    paths = sorted(e.path for e in entries)
    assert paths == ["/memory/a/1", "/memory/a/2"]


def test_list_prefix_escapes_wildcards(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/agents/100%match", content=b"")
    fs.write(pid=a.pid, path="/agents/100Xmatch", content=b"")
    # Without LIKE escape, "100%" would match both.
    entries, total = fs.list(prefix="/agents/100%")
    assert total == 1
    assert entries[0].path == "/agents/100%match"


def test_list_owner_filter(stores):
    ks, fs, _ = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    fs.write(pid=a.pid, path="/x/1", content=b"")
    fs.write(pid=b.pid, path="/x/2", content=b"")
    entries, total = fs.list(owner_pid=a.pid)
    assert total == 1
    assert entries[0].owner_pid == a.pid


def test_list_pagination(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    for i in range(5):
        fs.write(pid=a.pid, path=f"/x/{i}", content=b"x")
    page1, total = fs.list(limit=2, offset=0)
    page2, _ = fs.list(limit=2, offset=2)
    assert total == 5
    assert len(page1) == 2 and len(page2) == 2


# ── delete + gc_orphaned ─────────────────────────────────────────────────


def test_delete_removes(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/x", content=b"")
    assert fs.delete("/x") is True
    assert fs.exists("/x") is False


def test_delete_idempotent(stores):
    _, fs, _ = stores
    assert fs.delete("/never") is False


def test_gc_orphaned_clears_owner(stores):
    ks, fs, _ = stores
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    fs.write(pid=a.pid, path="/x/1", content=b"")
    fs.write(pid=a.pid, path="/x/2", content=b"")
    fs.write(pid=b.pid, path="/y/1", content=b"")
    n = fs.gc_orphaned(a.pid)
    assert n == 2
    assert fs.exists("/x/1") is False
    assert fs.exists("/y/1") is True


# ── ledger integration ──────────────────────────────────────────────────


def test_ledger_charges_fs_w_bytes(stores):
    ks, fs, led = stores
    a = ks.create(name="x", template="t")
    led.create(pid=a.pid, grants={"fs_w_bytes": 1000})
    fs.write(pid=a.pid, path="/x", content=b"x" * 100)
    fs.write(pid=a.pid, path="/y", content=b"y" * 200)
    # used should be 300.
    entry = led.get(a.pid).entries[0]
    assert entry.used == 300


def test_ledger_quota_exceeded_rolls_back(stores):
    ks, fs, led = stores
    a = ks.create(name="x", template="t")
    led.create(pid=a.pid, grants={"fs_w_bytes": 100})
    with pytest.raises(FsQuotaExceeded):
        fs.write(pid=a.pid, path="/x", content=b"x" * 200)
    # Write should have been rolled back.
    assert fs.exists("/x") is False
    # Used count must NOT have been incremented.
    assert led.get(a.pid).entries[0].used == 0


def test_ledger_no_charge_when_dim_missing(stores):
    """With no ledger row for fs_w_bytes, writes succeed un-tracked."""
    ks, fs, led = stores
    a = ks.create(name="x", template="t")
    fs.write(pid=a.pid, path="/x", content=b"x" * 100)
    led_obj = led.get(a.pid)
    assert led_obj.entries == ()


# ── concurrent writes ───────────────────────────────────────────────────


def test_concurrent_writes_to_distinct_paths_safe(stores):
    ks, fs, _ = stores
    a = ks.create(name="x", template="t")
    errors: list = []

    def worker(start: int, n: int):
        try:
            for i in range(n):
                fs.write(pid=a.pid, path=f"/x/{start + i}",
                         content=f"content-{start+i}".encode())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i * 25, 25))
               for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    _, total = fs.list(prefix="/x/")
    assert total == 100
