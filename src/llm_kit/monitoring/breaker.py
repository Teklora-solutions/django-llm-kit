"""The process-wide circuit breaker, with state in Django's cache (shared across workers)."""

from __future__ import annotations

from django.core.cache import cache

from llm_kit import conf
from llm_kit.core.observability.breaker import CircuitBreaker, Policy


class CacheStore:
    def get(self, key):
        return cache.get(key)

    def set(self, key, value, ttl):
        cache.set(key, value, int(ttl) + 1)

    def delete(self, key):
        cache.delete(key)


def _policy() -> Policy:
    return Policy(**(conf.get("CIRCUIT_BREAKER") or {}))


breaker = CircuitBreaker(CacheStore(), _policy())


def observe(call) -> None:
    """Recorder observer: feed every call outcome to the breaker."""
    if call.provider and call.status in ("ok", "error"):
        breaker.record(call.provider, call.status == "ok", call.error_class)
