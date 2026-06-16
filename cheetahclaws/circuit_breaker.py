"""
circuit_breaker.py — Per-provider circuit breaker for CheetahClaws API calls.

States:
  CLOSED    — normal, tracking failures in a rolling window
  OPEN      — failing fast; no calls until cooldown expires
  HALF_OPEN — one probe call allowed; success → CLOSED, failure → OPEN

Config keys (all optional):
  circuit_failure_threshold  int   failures in window to trip open  (default 5)
  circuit_window_seconds     int   rolling failure-count window (s)  (default 60)
  circuit_cooldown_seconds   int   open → half-open wait time   (s)  (default 120)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitOpenError(Exception):
    """Raised by providers.stream() when the circuit is open."""


class State(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    provider:   str
    threshold:  int   = 5
    window:     float = 60.0
    cooldown:   float = 120.0

    # Internal mutable state — not init params
    _state:         State        = field(default=State.CLOSED, init=False, repr=False)
    _failure_times: list[float]  = field(default_factory=list,  init=False, repr=False)
    _opened_at:     float | None = field(default=None,           init=False, repr=False)
    _lock:          threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ── Read-only properties ──────────────────────────────────────────────

    @property
    def state(self) -> State:
        with self._lock:
            return self._resolve_state()

    def _resolve_state(self) -> State:
        """Promote OPEN → HALF_OPEN if cooldown has elapsed. Lock must be held."""
        if self._state is State.OPEN:
            if self._opened_at is not None and \
               (time.monotonic() - self._opened_at) >= self.cooldown:
                self._state = State.HALF_OPEN
        return self._state

    # ── Control ───────────────────────────────────────────────────────────

    def allow_request(self) -> bool:
        """Return True if a request should be attempted."""
        with self._lock:
            return self._resolve_state() in (State.CLOSED, State.HALF_OPEN)

    def record_success(self) -> None:
        """Call after a successful API response (AssistantTurn received)."""
        with self._lock:
            was_open = self._state in (State.OPEN, State.HALF_OPEN)
            self._state = State.CLOSED
            self._failure_times.clear()
            self._opened_at = None
        if was_open:
            from cheetahclaws import logging_utils as _log
            _log.info("circuit_closed", provider=self.provider)

    def record_failure(self) -> None:
        """Call after a provider exception."""
        from cheetahclaws import logging_utils as _log
        with self._lock:
            now = time.monotonic()
            self._failure_times.append(now)
            # Evict failures older than the rolling window
            cutoff = now - self.window
            self._failure_times = [t for t in self._failure_times if t >= cutoff]

            if self._state is State.HALF_OPEN:
                # Probe call failed — reopen immediately
                self._state    = State.OPEN
                self._opened_at = now
                _log.error("circuit_reopened", provider=self.provider)

            elif self._state is State.CLOSED and len(self._failure_times) >= self.threshold:
                self._state    = State.OPEN
                self._opened_at = now
                _log.error("circuit_opened",
                           provider=self.provider,
                           failures=len(self._failure_times),
                           window_s=self.window,
                           cooldown_s=self.cooldown)


# ── Per-provider registry ──────────────────────────────────────────────────

_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(provider: str, config: dict | None = None) -> CircuitBreaker:
    """Return (creating if needed) the CircuitBreaker for a provider."""
    cfg = config or {}
    with _registry_lock:
        if provider not in _registry:
            _registry[provider] = CircuitBreaker(
                provider   = provider,
                threshold  = int(cfg.get("circuit_failure_threshold", 5)),
                window     = float(cfg.get("circuit_window_seconds",    60)),
                cooldown   = float(cfg.get("circuit_cooldown_seconds", 120)),
            )
        return _registry[provider]


def reset_breaker(provider: str) -> None:
    """Remove a circuit breaker from the registry. Used by tests and /circuit reset."""
    with _registry_lock:
        _registry.pop(provider, None)
