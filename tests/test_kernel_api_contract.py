"""Contract tests for the kernel.* RPC surface (RFC 0013).

These tests guard against API drift. When a developer adds a new
kernel.* method to a module's register() function, they MUST also
add it to one of the three tier sets in kernel/contract.py — or
this test fails.

Conversely, when a method is removed from register() without going
through the deprecation cycle, this test catches the drift.
"""
from __future__ import annotations

import socket
import threading
import time
import uuid

import pytest

from cheetahclaws.daemon import events
from cheetahclaws.daemon.server import make_tcp_server

from cheetahclaws.kernel import (
    ALL_KNOWN_METHODS,
    DEPRECATED_METHODS,
    EXPERIMENTAL_METHODS,
    KERNEL_VERSION,
    RFCS_IMPLEMENTED,
    SCHEMA_VERSION,
    STABLE_METHODS,
    register_with_daemon,
    verify_contract,
)
from cheetahclaws.kernel.integration import detach


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def daemon_with_kernel(tmp_path):
    """Build a daemon WITHOUT starting serve_forever — these tests only
    poke at the in-memory RpcRegistry, no HTTP needed.

    Important: do NOT call server.shutdown() on a server whose
    serve_forever was never started — the BaseServer's
    __is_shut_down Event blocks forever in that case. Only close the
    listening socket and detach the kernel.
    """
    events.reset_bus_for_tests()
    server = make_tcp_server(
        "127.0.0.1", _free_port(), data_dir=tmp_path,
        token="t-" + uuid.uuid4().hex, audit_enabled=True,
    )
    register_with_daemon(server.daemon_state, tmp_path / "kernel.db")
    yield server
    detach(server.daemon_state)
    server.server_close()


# ── Headline contract test ───────────────────────────────────────────────


def test_no_undocumented_methods(daemon_with_kernel):
    """Every kernel.* method registered after register_with_daemon must
    appear in ALL_KNOWN_METHODS. New methods MUST update contract.py
    before merging."""
    result = verify_contract(daemon_with_kernel.daemon_state.rpc)
    assert result["extra"] == [], (
        "These methods are registered but not classified in "
        f"contract.py — add them to STABLE/EXPERIMENTAL/DEPRECATED: "
        f"{result['extra']}"
    )
    assert result["missing"] == [], (
        "These methods are documented in contract.py but aren't "
        f"registered — implementation drift: {result['missing']}"
    )


def test_v1_no_deprecated_methods():
    """v1.0 ships with empty deprecated set."""
    assert DEPRECATED_METHODS == frozenset()


def test_v1_no_experimental_methods():
    """v1.0 ships everything as stable. Future versions may add
    experimental methods; this test will fail when that happens,
    forcing an update + review."""
    assert EXPERIMENTAL_METHODS == frozenset()


def test_all_known_is_union():
    assert ALL_KNOWN_METHODS == (
        STABLE_METHODS | EXPERIMENTAL_METHODS | DEPRECATED_METHODS
    )


def test_method_count_matches_documentation():
    """contract.py advertises 58 stable methods at v1.0. If you add a
    method, this count test will fail until you bump the literal —
    forcing a deliberate decision."""
    assert len(STABLE_METHODS) == 58
    assert len(ALL_KNOWN_METHODS) == 58


# ── kernel.api.* RPCs ────────────────────────────────────────────────────


def test_list_methods_all_tiers(daemon_with_kernel):
    state = daemon_with_kernel.daemon_state
    list_fn = state.rpc._methods["kernel.api.list_methods"]  # type: ignore[attr-defined]

    class _Ctx:
        client_id = "test"
        transport = "tcp"
        api_version = "0"

    result = list_fn({}, _Ctx())
    assert set(result["methods"]) == ALL_KNOWN_METHODS
    assert result["tier_counts"]["stable"]       == len(STABLE_METHODS)
    assert result["tier_counts"]["experimental"] == 0
    assert result["tier_counts"]["deprecated"]   == 0


def test_list_methods_filtered_by_tier(daemon_with_kernel):
    state = daemon_with_kernel.daemon_state
    list_fn = state.rpc._methods["kernel.api.list_methods"]

    class _Ctx:
        client_id = "test"
        transport = "tcp"
        api_version = "0"

    only_stable = list_fn({"tier": "stable"}, _Ctx())
    assert set(only_stable["methods"]) == STABLE_METHODS

    only_exp = list_fn({"tier": "experimental"}, _Ctx())
    assert only_exp["methods"] == []


def test_list_methods_invalid_tier(daemon_with_kernel):
    state = daemon_with_kernel.daemon_state
    list_fn = state.rpc._methods["kernel.api.list_methods"]

    class _Ctx:
        client_id = "test"
        transport = "tcp"
        api_version = "0"

    with pytest.raises(TypeError):  # InvalidPayload → TypeError → INVALID_PARAMS
        list_fn({"tier": "bogus"}, _Ctx())


def test_version_info_shape(daemon_with_kernel):
    state = daemon_with_kernel.daemon_state
    info_fn = state.rpc._methods["kernel.api.version_info"]

    class _Ctx:
        client_id = "test"
        transport = "tcp"
        api_version = "0"

    info = info_fn({}, _Ctx())
    assert info["kernel_version"]   == KERNEL_VERSION
    assert info["schema_version"]   == SCHEMA_VERSION
    assert info["api_version"]      == "0"   # daemon's API_VERSION
    assert info["method_count"]     == len(ALL_KNOWN_METHODS)
    assert info["rfcs_implemented"] == list(RFCS_IMPLEMENTED)
    assert info["tier_counts"]["stable"] == len(STABLE_METHODS)


# ── Drift detection (negative test) ──────────────────────────────────────


def test_verify_contract_detects_extra(daemon_with_kernel, monkeypatch):
    """Simulate a developer registering a new method without updating
    contract.py — verify_contract() must flag it as 'extra'."""
    state = daemon_with_kernel.daemon_state
    # Inject a stray method.
    state.rpc.register("kernel.bogus.notRegistered",
                        lambda p, c: {"ok": True})
    result = verify_contract(state.rpc)
    assert "kernel.bogus.notRegistered" in result["extra"]


def test_verify_contract_does_not_flag_non_kernel_methods(daemon_with_kernel):
    """The daemon's existing system.*, echo.*, permission.* methods
    are not part of the kernel contract."""
    state = daemon_with_kernel.daemon_state
    result = verify_contract(state.rpc)
    # The daemon has system.ping etc. registered. They shouldn't show
    # up as 'extra' because verify_contract filters to kernel.*.
    for non_kernel in ("system.ping", "system.shutdown", "echo.ping"):
        assert non_kernel not in result["extra"]
