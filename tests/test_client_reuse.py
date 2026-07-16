"""SDK client-reuse coverage for the Anthropic path.

stream_anthropic used to construct a fresh anthropic.Anthropic() per request,
paying a new TCP+TLS handshake on every one of the 5-50 stream() calls a
single tool loop issues. Clients are now leased from a module-level cache:

1. Same (api_key, endpoint) reuses one client; different key or endpoint
   gets a distinct client; a racing get-or-create creates exactly one.
2. Streams hold a lease for their lifetime (released in a finally, even when
   the generator is abandoned), so eviction can tell idle from in-flight.
3. Past the soft cap the least-recently-used IDLE client is closed and
   evicted; a leased client is never closed underneath a live stream, and
   an all-busy overflow contracts back to the cap on release.
4. _client_cache_clear() closes every pooled client (shutdown/fork/tests).

All tests fake the anthropic SDK via sys.modules — no network, no API key.
"""
from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

import pytest


# ── Fake Anthropic SDK ─────────────────────────────────────────────────────

class _FakeUsage:
    input_tokens = 10
    output_tokens = 2
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _FakeFinal:
    content: list = []
    usage = _FakeUsage()


class _FakeStreamCtx:
    def __init__(self, kwargs: dict):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return _FakeFinal()


class _FakeMessages:
    def stream(self, **kwargs):
        return _FakeStreamCtx(kwargs)


class _FakeAnthropicClient:
    instantiations = 0

    def __init__(self, api_key: str = "", base_url: str = ""):
        type(self).instantiations += 1
        self.api_key = api_key
        self.base_url = base_url
        self.closed = False
        self.messages = _FakeMessages()

    def close(self):
        self.closed = True


_EP = "https://api.anthropic.com"


@pytest.fixture()
def fake_anthropic(monkeypatch):
    """Install a fake `anthropic` module and reset the client cache."""
    from cheetahclaws import providers
    _FakeAnthropicClient.instantiations = 0
    monkeypatch.setitem(sys.modules, "anthropic",
                        SimpleNamespace(Anthropic=_FakeAnthropicClient))
    providers._client_cache_clear()
    yield
    providers._client_cache_clear()


def _lease_release(providers, key, endpoint=_EP):
    """Lease then immediately release — an 'idle' cache entry."""
    c = providers._lease_anthropic_client(key, endpoint)
    providers._release_anthropic_client(key, endpoint)
    return c


def _call_stream(config: dict, messages: list):
    from cheetahclaws.providers import stream_anthropic
    return list(stream_anthropic(
        api_key="k1",
        model="claude-sonnet-4-5",
        system="SYSTEM PROMPT",
        messages=messages,
        tool_schemas=[],
        config=config,
    ))


# ── get-or-create semantics ────────────────────────────────────────────────

def test_client_reused_for_same_key_and_endpoint(fake_anthropic):
    from cheetahclaws import providers
    c1 = _lease_release(providers, "k1")
    c2 = _lease_release(providers, "k1")
    c3 = _lease_release(providers, "k2")
    c4 = _lease_release(providers, "k1", "https://proxy.local")
    assert c1 is c2
    assert c1 is not c3
    assert c1 is not c4
    assert _FakeAnthropicClient.instantiations == 3


def test_client_cache_thread_race_single_instantiation(fake_anthropic):
    from cheetahclaws import providers

    barrier = threading.Barrier(8)
    results = []

    def _get():
        barrier.wait()
        results.append(_lease_release(providers, "k1"))

    threads = [threading.Thread(target=_get) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _FakeAnthropicClient.instantiations == 1
    assert all(r is results[0] for r in results)


def test_stream_reuses_cached_client(fake_anthropic):
    cfg = {"model": "claude-sonnet-4-5"}
    _call_stream(cfg, [{"role": "user", "content": "a"}])
    _call_stream(cfg, [{"role": "user", "content": "b"}])
    assert _FakeAnthropicClient.instantiations == 1


def test_stream_releases_lease_when_done(fake_anthropic):
    from cheetahclaws import providers
    _call_stream({"model": "claude-sonnet-4-5"},
                 [{"role": "user", "content": "a"}])
    with providers._CLIENT_CACHE_LOCK:
        assert not providers._CLIENT_LEASES   # generator finally ran


# ── lifecycle: close on clear, idle-only eviction, LRU ─────────────────────

def test_client_cache_clear_closes_clients(fake_anthropic):
    from cheetahclaws import providers
    c1 = _lease_release(providers, "k1")
    providers._client_cache_clear()
    assert c1.closed


def test_client_cache_cap_evicts_and_closes_oldest_idle(fake_anthropic):
    from cheetahclaws import providers
    clients = [_lease_release(providers, f"k{i}")
               for i in range(providers._CLIENT_CACHE_MAX + 2)]
    with providers._CLIENT_CACHE_LOCK:
        assert len(providers._CLIENT_CACHE) <= providers._CLIENT_CACHE_MAX
    assert clients[0].closed and clients[1].closed
    assert not clients[-1].closed


def test_eviction_never_closes_leased_client(fake_anthropic):
    """A client mid-stream in another thread must never be evicted/closed
    even when it is the oldest entry — the cap is soft while it is busy."""
    from cheetahclaws import providers
    busy = providers._lease_anthropic_client("busy", _EP)   # held, not released
    for i in range(providers._CLIENT_CACHE_MAX + 3):
        _lease_release(providers, f"k{i}")
    assert not busy.closed
    with providers._CLIENT_CACHE_LOCK:
        assert ("anthropic", _EP, "busy", os.getpid()) in providers._CLIENT_CACHE
    providers._release_anthropic_client("busy", _EP)


def test_soft_cap_contracts_when_leased_clients_finish(fake_anthropic):
    """An all-busy overflow must shrink after streams release their leases.

    This exercises the only path where the cap is intentionally exceeded:
    every cached client is in-flight at creation time, so none may be closed.
    Once one finishes, the newly idle least-recently-used entry is safe to
    evict and its connection pool must be closed.
    """
    from cheetahclaws import providers
    clients = [providers._lease_anthropic_client(f"busy-{i}", _EP)
               for i in range(providers._CLIENT_CACHE_MAX + 1)]
    with providers._CLIENT_CACHE_LOCK:
        assert len(providers._CLIENT_CACHE) == providers._CLIENT_CACHE_MAX + 1

    for i in range(providers._CLIENT_CACHE_MAX + 1):
        providers._release_anthropic_client(f"busy-{i}", _EP)

    with providers._CLIENT_CACHE_LOCK:
        assert len(providers._CLIENT_CACHE) == providers._CLIENT_CACHE_MAX
        assert not providers._CLIENT_LEASES
    assert clients[0].closed


def test_lru_hit_protects_recently_used_from_eviction(fake_anthropic):
    """A cache hit moves the entry to the back of the eviction order
    (true LRU, not FIFO)."""
    from cheetahclaws import providers
    n = providers._CLIENT_CACHE_MAX
    clients = [_lease_release(providers, f"k{i}") for i in range(n)]  # full
    _lease_release(providers, "k0")          # hit → k0 becomes most recent
    _lease_release(providers, "k_new")       # overflow → evicts k1, not k0
    assert not clients[0].closed
    assert clients[1].closed


# ── lease hygiene on failure paths ──────────────────────────────────────────

def test_request_build_failure_does_not_wedge_lease(fake_anthropic):
    """A request that fails while being BUILT (messages_to_anthropic raising
    on a tool message with no tool_call_id) must leave no lease behind: a
    wedged lease makes eviction treat the client as in-flight forever."""
    from cheetahclaws import providers
    bad = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t1", "name": "Read", "input": {}}]},
        {"role": "tool", "name": "Read", "content": "x"},  # no tool_call_id
    ]
    for _ in range(4):   # one agent turn's full retry budget
        with pytest.raises(KeyError):
            _call_stream({"model": "claude-sonnet-4-5"}, bad)
    with providers._CLIENT_CACHE_LOCK:
        assert not providers._CLIENT_LEASES


def test_stream_failure_after_lease_releases_lease(fake_anthropic):
    """A failure inside the streaming block itself still releases the lease
    through the finally, keeping the lease/release pairing intact."""
    from cheetahclaws import providers

    def _boom(**kwargs):
        raise RuntimeError("connection reset")

    client = _lease_release(providers, "k1")   # seed the cached client
    client.messages.stream = _boom
    with pytest.raises(RuntimeError):
        _call_stream({"model": "claude-sonnet-4-5"},
                     [{"role": "user", "content": "a"}])
    with providers._CLIENT_CACHE_LOCK:
        assert not providers._CLIENT_LEASES
