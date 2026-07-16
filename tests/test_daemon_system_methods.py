"""Tests for daemon/system_methods.py — system.ping + system.shutdown."""
from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws.daemon import system_methods
from cheetahclaws.daemon.rpc import CallContext, RpcRegistry


# ── Fake DaemonState that exposes just the surface system_methods touches ──

class _FakeState:
    def __init__(self) -> None:
        self.shutdown_event = threading.Event()
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.shutdown_event.set()


def _ctx(client_id: str = "test-client") -> CallContext:
    return CallContext(client_id=client_id, transport="tcp", api_version="0")


def _call(registry: RpcRegistry, method: str, params=None,
           ctx: CallContext = None) -> dict:
    envelope = {"jsonrpc": "2.0", "id": 1, "method": method,
                "params": params or {}}
    response, _status = registry.dispatch(envelope, ctx or _ctx())
    return response


# ── system.ping ─────────────────────────────────────────────────────────────

def test_system_ping_registers_and_returns_pong():
    registry = RpcRegistry()
    system_methods.register(registry, _FakeState())
    response = _call(registry, "system.ping")
    assert response["result"] == "pong"


def test_system_ping_ignores_params():
    registry = RpcRegistry()
    system_methods.register(registry, _FakeState())
    response = _call(registry, "system.ping", {"any": "thing"})
    assert response["result"] == "pong"


def test_system_ping_appears_in_method_list():
    registry = RpcRegistry()
    system_methods.register(registry, _FakeState())
    methods = registry.methods()
    assert "system.ping" in methods
    assert "system.shutdown" in methods


# ── system.shutdown ─────────────────────────────────────────────────────────

def test_system_shutdown_triggers_state_shutdown():
    state = _FakeState()
    registry = RpcRegistry()
    system_methods.register(registry, state)
    response = _call(registry, "system.shutdown")
    assert response["result"] == "shutdown_initiated"
    assert state.shutdown_calls == 1
    assert state.shutdown_event.is_set()


def test_system_shutdown_returned_from_dispatch_with_correct_envelope():
    state = _FakeState()
    registry = RpcRegistry()
    system_methods.register(registry, state)
    envelope = {"jsonrpc": "2.0", "id": "abc", "method": "system.shutdown"}
    response, status = registry.dispatch(envelope, _ctx())
    assert status == 200
    assert response == {"jsonrpc": "2.0", "id": "abc",
                        "result": "shutdown_initiated"}


# ── Coexistence with spike methods ──────────────────────────────────────────

def test_register_does_not_clash_with_spike_methods():
    """Spike's `register_methods` is invoked first by DaemonState.__init__;
    system_methods.register must not collide with any spike-defined names."""
    from cheetahclaws.daemon.methods import register as register_spike
    from cheetahclaws.daemon.permission import PermissionStore

    registry = RpcRegistry()
    store = PermissionStore()
    register_spike(registry, store)
    state = _FakeState()
    system_methods.register(registry, state)

    methods = registry.methods()
    assert "system.ping" in methods
    assert "system.shutdown" in methods
    assert "echo.ping" in methods  # spike survives
