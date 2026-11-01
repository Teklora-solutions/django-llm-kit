"""Per-provider circuit breaker, fed by recorded calls and read by touchpoint resolution.

A provider's circuit opens after ``failures`` retryable errors (rate limit,
overload, timeout, connection) within ``window`` seconds, or at once on an
exhausted quota (retrying soon won't help). While open, resolution skips the
provider's models in favour of fallbacks. After the cooldown the circuit is
half-open: the next call is let through, and its outcome closes or re-opens it.

State lives in a ``Store`` shared by all processes (the Django adapter uses the
cache), so one worker's failures protect the others.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from llm_kit.core.observability.classify import RETRYABLE


class Store(Protocol):
    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: Any, ttl: float) -> None: ...

    def delete(self, key: str) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ttl):
        self.data[key] = value

    def delete(self, key):
        self.data.pop(key, None)


@dataclass(frozen=True)
class Policy:
    failures: int = 5
    window: float = 60.0
    cooldown: float = 60.0
    quota_cooldown: float = 900.0


_PROVIDERS = "llm_kit:breaker:providers"


class CircuitBreaker:
    def __init__(self, store: Store | None = None, policy: Policy | None = None, clock=time.time) -> None:
        self.store: Store = store or MemoryStore()
        self.policy = policy or Policy()
        self.clock = clock
        self._lock = threading.Lock()

    def _key(self, provider: str) -> str:
        return f"llm_kit:breaker:{provider}"

    def _load(self, provider: str) -> dict[str, Any]:
        return self.store.get(self._key(provider)) or {"failures": [], "open_until": None, "reason": ""}

    def _save(self, provider: str, state: dict[str, Any]) -> None:
        ttl = max(self.policy.window, self.policy.quota_cooldown) * 2
        self.store.set(self._key(provider), state, ttl)
        known = set(self.store.get(_PROVIDERS) or [])
        if provider not in known:
            self.store.set(_PROVIDERS, sorted(known | {provider}), 7 * 86400)

    def record(self, provider: str, ok: bool, error_class: str = "") -> None:
        """Feed one call outcome. Non-retryable errors (bad request, auth…) don't trip the circuit."""
        if not provider:
            return
        now = self.clock()
        with self._lock:
            state = self._load(provider)
            if ok:
                if state["failures"] or state["open_until"]:
                    self._save(provider, {"failures": [], "open_until": None, "reason": ""})
                return
            if error_class == "quota_exhausted":
                state.update(open_until=now + self.policy.quota_cooldown, reason="quota exhausted")
                self._save(provider, state)
                return
            if error_class not in RETRYABLE:
                return
            failures = [t for t in state["failures"] if now - t <= self.policy.window] + [now]
            state["failures"] = failures
            half_open = state["open_until"] is not None and now >= state["open_until"]
            if half_open or len(failures) >= self.policy.failures:
                state.update(open_until=now + self.policy.cooldown, reason=f"{len(failures)} {error_class} errors")
            self._save(provider, state)

    def is_open(self, provider: str) -> bool:
        state = self.store.get(self._key(provider))
        return bool(state and state.get("open_until") and self.clock() < state["open_until"])

    def state(self, provider: str) -> dict[str, Any]:
        s = self._load(provider)
        now = self.clock()
        if s["open_until"] and now < s["open_until"]:
            name = "open"
        elif s["open_until"]:
            name = "half_open"
        else:
            name = "closed"
        return {
            "state": name,
            "reason": s["reason"],
            "recent_failures": len([t for t in s["failures"] if now - t <= self.policy.window]),
            "retry_in_s": round(s["open_until"] - now) if name == "open" else 0,
        }

    def states(self) -> dict[str, dict[str, Any]]:
        """Providers whose circuit isn't plainly closed."""
        out = {}
        for provider in self.store.get(_PROVIDERS) or []:
            st = self.state(provider)
            if st["state"] != "closed" or st["recent_failures"]:
                out[provider] = st
        return out

    def reset(self, provider: str | None = None) -> None:
        for p in [provider] if provider else list(self.store.get(_PROVIDERS) or []):
            self.store.delete(self._key(p))
