"""Tests for kernel.capability (RFC 0005)."""
from __future__ import annotations

import pytest

from cheetahclaws.kernel import (
    AgentState,
    Capability,
    CapabilityDerivationError,
    CapabilityExists,
    CapabilityInvalidGrant,
    CapabilityStore,
    CapabilityUnknownPid,
    FsGrant,
    KernelStore,
    fs_grant_matches,
    host_matches_glob,
)


@pytest.fixture
def stores(tmp_path):
    """Open kernel + capability stores sharing one connection + lock."""
    ks = KernelStore.open(tmp_path / "kernel.db")
    cs = CapabilityStore(ks.connection, write_lock=ks.write_lock)
    yield ks, cs
    ks.close()


# ── Glob matching primitive ────────────────────────────────────────────────


@pytest.mark.parametrize("host,glob,expected", [
    # Exact
    ("example.com",     "example.com",       True),
    ("api.example.com", "example.com",       False),
    # Single-level wildcard
    ("api.example.com", "*.example.com",     True),
    ("a.b.example.com", "*.example.com",     False),
    ("example.com",     "*.example.com",     False),
    # Multi-level wildcard
    ("api.example.com", "**.example.com",    True),
    ("a.b.example.com", "**.example.com",    True),
    ("example.com",     "**.example.com",    True),
    ("evil.com",        "**.example.com",    False),
    # Universal
    ("anything.tld",    "*",                 True),
    # Empties
    ("",                "example.com",       False),
    ("example.com",     "",                  False),
])
def test_host_matches_glob(host, glob, expected):
    assert host_matches_glob(host, glob) is expected


# ── Path matching primitive ────────────────────────────────────────────────


@pytest.mark.parametrize("path,prefix,grant_mode,req_mode,expected", [
    # Exact prefix matches
    ("/agents/alice/x.txt", "/agents/alice/", "rw", "rw", True),
    ("/agents/alice/x.txt", "/agents/alice/", "rw", "r",  True),
    ("/agents/alice/x.txt", "/agents/alice/", "r",  "rw", False),
    ("/agents/alice/x.txt", "/agents/alice/", "r",  "r",  True),
    # Prefix without trailing slash
    ("/agents/alice/x",     "/agents/alice",  "r",  "r",  True),
    # Path equals prefix (the directory itself)
    ("/agents/alice",       "/agents/alice/", "r",  "r",  True),
    # Prefix boundary respected — alicia ≠ alice
    ("/agents/alicia/x",    "/agents/alice/", "rw", "r",  False),
    # Root prefix matches everything
    ("/etc/passwd",         "/",              "r",  "r",  True),
    # Relative paths rejected
    ("agents/alice",        "/agents/alice/", "rw", "r",  False),
])
def test_fs_grant_matches(path, prefix, grant_mode, req_mode, expected):
    g = FsGrant(prefix=prefix, mode=grant_mode)
    assert fs_grant_matches(g, path, req_mode) is expected


# ── FsGrant validation ─────────────────────────────────────────────────────


def test_fs_grant_rejects_relative_prefix():
    with pytest.raises(CapabilityInvalidGrant):
        FsGrant(prefix="rel/path", mode="r")


def test_fs_grant_rejects_bad_mode():
    with pytest.raises(CapabilityInvalidGrant):
        FsGrant(prefix="/x", mode="x")
    with pytest.raises(CapabilityInvalidGrant):
        FsGrant(prefix="/x", mode="w")  # no write-only mode


# ── CapabilityStore.create ─────────────────────────────────────────────────


def test_create_round_trip(stores):
    ks, cs = stores
    a = ks.create(name="alice", template="t")
    cap = cs.create(
        pid=a.pid,
        tool_grants=["Read", "Bash"],
        fs_grants=[{"prefix": "/agents/alice/", "mode": "rw"}],
        net_grants=["*.example.com"],
        model_grants=["claude-opus-4"],
        sub_agent=True,
    )
    assert isinstance(cap, Capability)
    assert cap.pid == a.pid
    assert cap.tool_grants == frozenset({"Read", "Bash"})
    assert len(cap.fs_grants) == 1
    assert cap.fs_grants[0].prefix == "/agents/alice/"
    assert cap.net_grants == frozenset({"*.example.com"})
    assert cap.sub_agent is True
    assert cap.parent_cap_id is None


def test_create_unknown_pid(stores):
    _, cs = stores
    from cheetahclaws.kernel import UnknownPid
    with pytest.raises(UnknownPid):
        cs.create(pid=9999, tool_grants=["Read"])


def test_create_duplicate_raises(stores):
    ks, cs = stores
    a = ks.create(name="x", template="t")
    cs.create(pid=a.pid, tool_grants=["Read"])
    with pytest.raises(CapabilityExists) as e:
        cs.create(pid=a.pid, tool_grants=["Read"])
    assert e.value.pid == a.pid


def test_create_with_no_grants_is_inert(stores):
    """Per RFC 0005 §8 OQ#1, empty grants are allowed (default-deny)."""
    ks, cs = stores
    a = ks.create(name="x", template="t")
    cap = cs.create(pid=a.pid)
    assert cap.tool_grants == frozenset()
    assert cap.fs_grants == ()
    assert cap.net_grants == frozenset()
    assert cap.model_grants == frozenset()
    assert cap.sub_agent is False


def test_get_unknown_pid(stores):
    _, cs = stores
    with pytest.raises(CapabilityUnknownPid):
        cs.get(9999)


# ── Derivation rules ───────────────────────────────────────────────────────


def _setup_parent(ks, cs, **grants):
    p = ks.create(name="parent", template="t")
    cs.create(pid=p.pid, **grants)
    return p


def test_derive_strict_subset_succeeds(stores):
    ks, cs = stores
    p = _setup_parent(ks, cs,
                      tool_grants=["Read", "Bash", "Write"],
                      fs_grants=[{"prefix": "/work/", "mode": "rw"}],
                      net_grants=["*.example.com", "api.x.com"],
                      model_grants=["claude-opus-4"],
                      sub_agent=True)
    c = ks.create(name="child", template="t", parent_pid=p.pid)
    derived = cs.derive(
        parent_pid=p.pid, child_pid=c.pid,
        tool_grants=["Read"],
        fs_grants=[{"prefix": "/work/sub/", "mode": "r"}],
        net_grants=["api.x.com"],
        model_grants=[],
        sub_agent=False,
    )
    assert derived.parent_cap_id is not None
    assert derived.tool_grants == frozenset({"Read"})


def test_derive_extra_tool_rejected(stores):
    ks, cs = stores
    p = _setup_parent(ks, cs, tool_grants=["Read"])
    c = ks.create(name="child", template="t", parent_pid=p.pid)
    with pytest.raises(CapabilityDerivationError) as e:
        cs.derive(parent_pid=p.pid, child_pid=c.pid,
                  tool_grants=["Read", "Bash"])
    assert e.value.field == "tool_grants"


def test_derive_extra_model_rejected(stores):
    ks, cs = stores
    p = _setup_parent(ks, cs, model_grants=["claude-opus-4"])
    c = ks.create(name="child", template="t", parent_pid=p.pid)
    with pytest.raises(CapabilityDerivationError):
        cs.derive(parent_pid=p.pid, child_pid=c.pid,
                  model_grants=["gpt-4"])


def test_derive_broader_net_rejected(stores):
    ks, cs = stores
    p = _setup_parent(ks, cs, net_grants=["*.example.com"])
    c = ks.create(name="child", template="t", parent_pid=p.pid)
    # Conservative subset: even though api.example.com is more specific,
    # string subset rules — child must inherit the exact glob.
    with pytest.raises(CapabilityDerivationError):
        cs.derive(parent_pid=p.pid, child_pid=c.pid,
                  net_grants=["api.example.com"])


def test_derive_path_outside_parent_rejected(stores):
    ks, cs = stores
    p = _setup_parent(ks, cs,
                      fs_grants=[{"prefix": "/work/", "mode": "rw"}])
    c = ks.create(name="child", template="t", parent_pid=p.pid)
    with pytest.raises(CapabilityDerivationError) as e:
        cs.derive(parent_pid=p.pid, child_pid=c.pid,
                  fs_grants=[{"prefix": "/etc/", "mode": "r"}])
    assert e.value.field == "fs_grants"


def test_derive_mode_upgrade_rejected(stores):
    ks, cs = stores
    p = _setup_parent(ks, cs,
                      fs_grants=[{"prefix": "/work/", "mode": "r"}])
    c = ks.create(name="child", template="t", parent_pid=p.pid)
    with pytest.raises(CapabilityDerivationError):
        # Parent has /work/ "r"; child wants /work/sub/ "rw" — mode upgrade.
        cs.derive(parent_pid=p.pid, child_pid=c.pid,
                  fs_grants=[{"prefix": "/work/sub/", "mode": "rw"}])


def test_derive_sub_agent_without_parent_rejected(stores):
    ks, cs = stores
    p = _setup_parent(ks, cs, sub_agent=False)
    c = ks.create(name="child", template="t", parent_pid=p.pid)
    with pytest.raises(CapabilityDerivationError) as e:
        cs.derive(parent_pid=p.pid, child_pid=c.pid, sub_agent=True)
    assert e.value.field == "sub_agent"


def test_derive_universal_parent_allows_anything(stores):
    """A parent with '*' tool_grants permits any child tool_grants."""
    ks, cs = stores
    p = _setup_parent(ks, cs,
                      tool_grants=["*"], model_grants=["*"],
                      net_grants=["*"], sub_agent=True,
                      fs_grants=[{"prefix": "/", "mode": "rw"}])
    c = ks.create(name="child", template="t", parent_pid=p.pid)
    cs.derive(parent_pid=p.pid, child_pid=c.pid,
              tool_grants=["Random_Tool"],
              model_grants=["future-model"],
              net_grants=["api.x.com"],
              fs_grants=[{"prefix": "/agents/x/", "mode": "rw"}])


def test_derive_self_rejected(stores):
    ks, cs = stores
    p = _setup_parent(ks, cs, tool_grants=["Read"])
    with pytest.raises(CapabilityDerivationError):
        cs.derive(parent_pid=p.pid, child_pid=p.pid)


# ── Default-deny on missing rows ──────────────────────────────────────────


def test_check_tool_no_row_default_deny(stores):
    ks, cs = stores
    a = ks.create(name="x", template="t")
    # No cap.create called — checks must default deny.
    assert cs.check_tool(a.pid, "Read") is False
    assert cs.check_fs(a.pid, "/etc/passwd", "r") is False
    assert cs.check_net(a.pid, "example.com") is False
    assert cs.check_model(a.pid, "claude-opus-4") is False


def test_check_tool_grant_then_revoke_via_terminate(stores):
    """DEAD agent's cap row remains — checks still return per the row."""
    ks, cs = stores
    a = ks.create(name="x", template="t")
    cs.create(pid=a.pid, tool_grants=["Read"])
    assert cs.check_tool(a.pid, "Read") is True
    ks.terminate(a.pid, exit_kind="completed")
    # Cap row not auto-deleted; check still returns True per RFC.
    assert cs.check_tool(a.pid, "Read") is True


def test_check_tool_with_universal(stores):
    ks, cs = stores
    a = ks.create(name="x", template="t")
    cs.create(pid=a.pid, tool_grants=["*"])
    assert cs.check_tool(a.pid, "Anything") is True


def test_check_fs_path_under_grant(stores):
    ks, cs = stores
    a = ks.create(name="x", template="t")
    cs.create(pid=a.pid,
              fs_grants=[{"prefix": "/work/", "mode": "rw"}])
    assert cs.check_fs(a.pid, "/work/file.txt", "r") is True
    assert cs.check_fs(a.pid, "/work/file.txt", "rw") is True
    assert cs.check_fs(a.pid, "/etc/passwd", "r") is False


def test_check_fs_mode_subset(stores):
    ks, cs = stores
    a = ks.create(name="x", template="t")
    cs.create(pid=a.pid,
              fs_grants=[{"prefix": "/ro/", "mode": "r"}])
    assert cs.check_fs(a.pid, "/ro/x", "r") is True
    assert cs.check_fs(a.pid, "/ro/x", "rw") is False


def test_check_net_glob(stores):
    ks, cs = stores
    a = ks.create(name="x", template="t")
    cs.create(pid=a.pid,
              net_grants=["*.example.com", "api.x.com"])
    assert cs.check_net(a.pid, "api.example.com") is True
    assert cs.check_net(a.pid, "api.x.com")       is True
    assert cs.check_net(a.pid, "evil.com")        is False


def test_check_handles_invalid_input_safely(stores):
    ks, cs = stores
    a = ks.create(name="x", template="t")
    cs.create(pid=a.pid, tool_grants=["Read"])
    assert cs.check_tool(a.pid, "")     is False
    assert cs.check_fs  (a.pid, "rel",  "r")  is False
    assert cs.check_fs  (a.pid, "/x",   "wx") is False
    assert cs.check_net (a.pid, "")           is False
