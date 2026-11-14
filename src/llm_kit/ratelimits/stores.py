"""Shared counts for rate limits, in Django's cache.

``CacheStore`` works with any cache backend that has atomic ``incr`` (Redis, Memcached; the
local-memory cache counts per process). Its per-minute windows are fixed (the clock minute), so
across a minute boundary up to twice the per-minute limit can go out in a short burst.

``RedisStore`` (Django's ``RedisCache`` or django-redis) checks and counts in one Lua script with
a sliding one-minute window, so it never goes over.
"""

from __future__ import annotations

import hashlib
import uuid

from llm_kit.core.touchpoints.ratelimit import (
    DAY,
    MINUTE,
    Limits,
    day_bucket,
    minute_bucket,
    until_next_day,
    until_next_minute,
)

PREFIX = "llm_kit:rl:"


def _id(model: str) -> str:
    return hashlib.sha1(model.encode()).hexdigest()[:20]  # cache-safe keys for any model ref


class CacheStore:
    name = "cache"

    def __init__(self, cache=None) -> None:
        if cache is None:
            from django.core.cache import cache
        self.cache = cache

    def _key(self, model: str, kind: str, bucket: int) -> str:
        return f"{PREFIX}{_id(model)}:{kind}:{bucket}"

    def _incr(self, key: str, ttl: float) -> int:
        self.cache.add(key, 0, int(ttl))
        try:
            return self.cache.incr(key)
        except ValueError:  # evicted in between
            self.cache.set(key, 1, int(ttl))
            return 1

    def _undo(self, key: str) -> None:
        try:
            self.cache.decr(key)
        except ValueError:
            pass

    def take(self, model: str, limits: Limits, now: float) -> tuple[bool, str, float]:
        tokens = self._key(model, "t", minute_bucket(now))
        if limits.tpm and (self.cache.get(tokens) or 0) >= limits.tpm:
            return False, "tpm", until_next_minute(now)
        day = self._key(model, "d", day_bucket(now))
        if self._incr(day, DAY + 3600) > limits.rpd > 0:
            self._undo(day)  # a refused request takes no slot
            return False, "rpd", until_next_day(now)
        minute = self._key(model, "r", minute_bucket(now))
        if self._incr(minute, 2 * MINUTE) > limits.rpm > 0:
            self._undo(minute)
            self._undo(day)
            return False, "rpm", until_next_minute(now)
        return True, "ok", 0.0

    def add_tokens(self, model: str, tokens: int, now: float) -> None:
        key = self._key(model, "t", minute_bucket(now))
        self.cache.add(key, 0, int(2 * MINUTE))
        try:
            self.cache.incr(key, int(tokens))
        except ValueError:
            self.cache.set(key, int(tokens), int(2 * MINUTE))

    def usage(self, model: str, now: float) -> dict[str, int]:
        get = self.cache.get
        return {
            "rpm": int(get(self._key(model, "r", minute_bucket(now))) or 0),
            "tpm": int(get(self._key(model, "t", minute_bucket(now))) or 0),
            "rpd": int(get(self._key(model, "d", day_bucket(now))) or 0),
        }


# KEYS: requests zset, tokens this minute, requests today
# ARGV: now, rpm, tpm, rpd, member
_TAKE = """
local now = tonumber(ARGV[1])
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, now - 60)
local rpm = tonumber(ARGV[2])
if rpm > 0 and redis.call('ZCARD', KEYS[1]) >= rpm then
    local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
    return {0, 'rpm', tostring(tonumber(oldest[2]) + 60 - now)}
end
local tpm = tonumber(ARGV[3])
if tpm > 0 and tonumber(redis.call('GET', KEYS[2]) or '0') >= tpm then
    return {0, 'tpm', tostring(60 - (now % 60))}
end
local rpd = tonumber(ARGV[4])
if rpd > 0 and tonumber(redis.call('GET', KEYS[3]) or '0') >= rpd then
    return {0, 'rpd', tostring(86400 - (now % 86400))}
end
redis.call('ZADD', KEYS[1], now, ARGV[5])
redis.call('EXPIRE', KEYS[1], 120)
redis.call('INCR', KEYS[3])
redis.call('EXPIRE', KEYS[3], 90000)
return {1, 'ok', '0'}
"""


def redis_client(cache):
    """The raw client behind a Django Redis cache (Django's own, or django-redis), or None."""
    client = getattr(cache, "client", None)
    if client is not None and hasattr(client, "get_client"):  # django-redis
        try:
            return client.get_client(write=True)
        except TypeError:
            pass
    inner = getattr(cache, "_cache", None)  # django.core.cache.backends.redis.RedisCache
    if inner is not None and hasattr(inner, "get_client"):
        return inner.get_client(None, write=True)
    return None


class RedisStore:
    name = "redis"

    def __init__(self, client) -> None:
        self.client = client
        self._take = client.register_script(_TAKE)

    def _keys(self, model: str, now: float) -> list[str]:
        base = f"{PREFIX}{_id(model)}"
        return [f"{base}:z", f"{base}:t:{minute_bucket(now)}", f"{base}:d:{day_bucket(now)}"]

    def take(self, model: str, limits: Limits, now: float) -> tuple[bool, str, float]:
        member = f"{now}:{uuid.uuid4().hex}"  # equal timestamps must not collapse into one entry
        granted, reason, wait = self._take(
            keys=self._keys(model, now), args=[repr(now), limits.rpm, limits.tpm, limits.rpd, member]
        )
        reason = reason.decode() if isinstance(reason, bytes) else reason
        wait = wait.decode() if isinstance(wait, bytes) else wait
        return bool(granted), reason, max(0.0, float(wait))

    def add_tokens(self, model: str, tokens: int, now: float) -> None:
        key = self._keys(model, now)[1]
        pipe = self.client.pipeline()
        pipe.incrby(key, int(tokens))
        pipe.expire(key, int(2 * MINUTE))
        pipe.execute()

    def usage(self, model: str, now: float) -> dict[str, int]:
        zset, tokens, day = self._keys(model, now)
        pipe = self.client.pipeline()
        pipe.zcount(zset, now - MINUTE, now)
        pipe.get(tokens)
        pipe.get(day)
        rpm, tpm, rpd = pipe.execute()
        return {"rpm": int(rpm or 0), "tpm": int(tpm or 0), "rpd": int(rpd or 0)}
