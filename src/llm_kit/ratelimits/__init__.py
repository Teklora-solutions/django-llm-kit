"""Per-model rate limits of your provider accounts (requests/tokens per minute, requests per day).

Set them in the admin (*LLM Kit → Rate limits*) or in settings::

    LLM_KIT = {
        "RATE_LIMITS": {
            "gemini/gemini-3.8-flash": {"rpm": 10, "tpm": 250_000, "rpd": 500},
            "anthropic/*": {"rpm": 50},           # each matching model gets these
        },
    }

Before each call a slot is taken in a count shared by every process (Django's cache; with Redis,
a sliding window). At a limit, the call waits up to the touchpoint's retry wait, then goes to the
next fallback; when no model has a slot it fails with ``RateLimited``. Calls on keys a tenant or
user brought aren't counted: those are their account's limits. See
``llm_kit.core.touchpoints.ratelimit``.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
from typing import Any

from llm_kit.core.touchpoints.ratelimit import Limiter, Limits, RateLimited
from llm_kit.runtime.versioned_cache import VersionedCache

logger = logging.getLogger("llm_kit")

__all__ = ["Limits", "RateLimited", "invalidate", "limiter", "limits_for", "usage"]

_rows = VersionedCache("rate_limits")
_lock = threading.Lock()
_limiter: Limiter | None = None
_store_key: Any = None


def _load_rows() -> dict[str, Limits]:
    from llm_kit.models import RateLimit

    return {r.model: Limits(r.rpm, r.tpm, r.rpd) for r in RateLimit.objects.all()}


def configured() -> dict[str, Limits]:
    """Every pattern with limits: ``RATE_LIMITS``, then rows (a row wins for the same pattern)."""
    from llm_kit import conf

    out = {k: Limits.coerce(v) for k, v in (conf.get("RATE_LIMITS") or {}).items()}
    try:
        out.update(_rows.get(_load_rows))
    except Exception:  # no table yet (before migrate)
        logger.debug("Rate limit rows unavailable", exc_info=True)
    return out


def _specificity(pattern: str) -> tuple[int, int]:
    wild = any(c in pattern for c in "*?[")
    return (0 if wild else 1, len(pattern.replace("*", "")))


def limits_for(model: str) -> Limits | None:
    """The most specific limits for ``model``: its own, else the longest pattern matching it."""
    patterns = configured()
    if model in patterns:
        return patterns[model]
    matching = [p for p in patterns if fnmatch.fnmatchcase(model, p)]
    return patterns[max(matching, key=_specificity)] if matching else None


def _store():
    from django.core.cache import cache

    from llm_kit import conf
    from llm_kit.ratelimits.stores import CacheStore, RedisStore, redis_client

    backend = conf.get("RATE_LIMIT_BACKEND")
    if backend not in ("auto", "redis", "cache"):
        from llm_kit.core.errors import ConfigError

        raise ConfigError(f'RATE_LIMIT_BACKEND must be "auto", "redis", "cache" or "off", not {backend!r}.')
    if backend in ("auto", "redis"):
        client = redis_client(cache)
        if client is not None:
            return RedisStore(client)
        if backend == "redis":
            from llm_kit.core.errors import ConfigError

            raise ConfigError('RATE_LIMIT_BACKEND="redis" needs the default cache to be Redis.')
    return CacheStore(cache)


def limiter() -> Limiter | None:
    """The shared limiter, or None when no limits are set (or ``RATE_LIMIT_BACKEND="off"``)."""
    global _limiter, _store_key
    from django.conf import settings

    from llm_kit import conf

    if conf.get("RATE_LIMIT_BACKEND") == "off" or not configured():
        return None
    key = (conf.get("RATE_LIMIT_BACKEND"), repr(getattr(settings, "CACHES", None)))
    with _lock:
        if _limiter is None or _store_key != key:
            _limiter, _store_key = Limiter(_store(), limits_for), key
        return _limiter


def usage() -> list[dict[str, Any]]:
    """Use against limits for every model that has limits or has been counted recently.

    Patterns are reported for the models of your touchpoints that match them."""
    current = limiter()
    if current is None:
        return []
    from llm_kit.core.touchpoints import all_touchpoints

    models = {m for m in configured() if not any(c in m for c in "*?[")}
    for tp in all_touchpoints():
        for ref in [tp.default, *tp.fallbacks]:
            if ref and limits_for(ref):
                models.add(ref)
    return [current.usage(m) for m in sorted(models)]


def invalidate() -> None:
    _rows.bump()
