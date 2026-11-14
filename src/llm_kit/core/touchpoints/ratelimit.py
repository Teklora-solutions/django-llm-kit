"""Per-model rate limits: requests per minute, tokens per minute, requests per day.

Limits are facts about your provider account (its tier), set by people: they are never synced
from a public catalogue. The Django side reads them from ``RateLimit`` rows and
``LLM_KIT["RATE_LIMITS"]`` and keeps the counts in Django's cache (Redis, Memcached), so every
process shares one budget per model.

Before each model call a slot is acquired: when the model is at a limit, the call waits up to
the touchpoint's retry wait (``RetryPolicy.max_wait``) for a slot, then moves to the next
fallback; when no candidate gets a slot, the call fails with ``RateLimited`` (classified
``rate_limit``, or ``quota_exhausted`` for the daily limit). Tokens are counted after each call,
so a token limit binds from the next call: one call can overshoot it.

A refused request takes no slot. All three are counted for a model even when it has no limit
of that kind, so usage can be reported.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from llm_kit.core.errors import LLMKitError

MINUTE = 60.0
DAY = 86_400.0

_REASONS = {"rpm": "requests per minute", "tpm": "tokens per minute", "rpd": "requests per day"}


@dataclass(frozen=True)
class Limits:
    """0 = no limit of that kind."""

    rpm: int = 0
    tpm: int = 0
    rpd: int = 0

    @classmethod
    def coerce(cls, value: Any) -> Limits | None:
        if value is None or isinstance(value, Limits):
            return value
        if isinstance(value, dict):
            unknown = set(value) - {"rpm", "tpm", "rpd"}
            if unknown:
                from llm_kit.core.errors import ConfigError

                raise ConfigError(f"Rate limits take rpm, tpm and rpd, not {sorted(unknown)}.")
            return cls(**{k: int(v or 0) for k, v in value.items()})
        raise TypeError(f"Rate limits must be a dict like {{'rpm': 60}}, not {value!r}")

    def __bool__(self) -> bool:
        return bool(self.rpm or self.tpm or self.rpd)


class RateLimited(LLMKitError):
    """No slot for ``model`` within the wait allowed. ``reason`` is ``rpm``, ``tpm`` or ``rpd``."""

    def __init__(self, model: str, reason: str, retry_in: float | None = None) -> None:
        self.model, self.reason, self.retry_in = model, reason, retry_in
        when = f"; next slot in {retry_in:.0f}s" if retry_in is not None else ""
        super().__init__(f"Rate limit reached for {model}: {_REASONS.get(reason, reason)} limit{when}.")


class Store(Protocol):
    """Shared counts per model. ``take`` checks and, if allowed, counts one request atomically."""

    def take(self, model: str, limits: Limits, now: float) -> tuple[bool, str, float]:
        """``(granted, reason, seconds until a slot may free up)``."""

    def add_tokens(self, model: str, tokens: int, now: float) -> None: ...

    def usage(self, model: str, now: float) -> dict[str, int]:
        """``{"rpm": requests in the last minute, "tpm": tokens this minute, "rpd": requests today}``."""


def minute_bucket(now: float) -> int:
    return int(now // MINUTE)


def day_bucket(now: float) -> int:
    return int(now // DAY)  # UTC days


def until_next_minute(now: float) -> float:
    return MINUTE - (now % MINUTE)


def until_next_day(now: float) -> float:
    return DAY - (now % DAY)


class MemoryStore:
    """Counts in this process only: for running without a framework, and for tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[str, deque[float]] = {}
        self._tokens: dict[tuple[str, int], int] = {}
        self._days: dict[tuple[str, int], int] = {}

    def _window(self, model: str, now: float) -> deque[float]:
        window = self._requests.setdefault(model, deque())
        while window and window[0] <= now - MINUTE:
            window.popleft()
        return window

    def take(self, model: str, limits: Limits, now: float) -> tuple[bool, str, float]:
        with self._lock:
            window = self._window(model, now)
            if limits.rpm and len(window) >= limits.rpm:
                return False, "rpm", max(0.0, window[0] + MINUTE - now)
            if limits.tpm and self._tokens.get((model, minute_bucket(now)), 0) >= limits.tpm:
                return False, "tpm", until_next_minute(now)
            day = (model, day_bucket(now))
            if limits.rpd and self._days.get(day, 0) >= limits.rpd:
                return False, "rpd", until_next_day(now)
            window.append(now)
            self._days[day] = self._days.get(day, 0) + 1
            return True, "ok", 0.0

    def add_tokens(self, model: str, tokens: int, now: float) -> None:
        with self._lock:
            key = (model, minute_bucket(now))
            self._tokens[key] = self._tokens.get(key, 0) + int(tokens)

    def usage(self, model: str, now: float) -> dict[str, int]:
        with self._lock:
            return {
                "rpm": len(self._window(model, now)),
                "tpm": self._tokens.get((model, minute_bucket(now)), 0),
                "rpd": self._days.get((model, day_bucket(now)), 0),
            }


class Limiter:
    """Acquires slots from a ``Store`` for the limits ``limits_for(model)`` returns."""

    def __init__(self, store: Store, limits_for: Callable[[str], Limits | None], clock=time.time) -> None:
        self.store = store
        self.limits_for = limits_for
        self.clock = clock

    def _try(self, model: str) -> tuple[bool, str, float]:
        return self.store.take(model, self.limits_for(model) or Limits(), self.clock())

    def acquire(self, model: str, max_wait: float = 0.0) -> None:
        """Take a slot for one call to ``model``, waiting up to ``max_wait`` seconds; else ``RateLimited``."""
        deadline = time.monotonic() + max(0.0, max_wait)
        while True:
            granted, reason, wait = self._try(model)
            if granted:
                return
            left = deadline - time.monotonic()
            if wait > left:  # no slot frees up in time: don't wait for nothing
                raise RateLimited(model, reason, wait)
            time.sleep(max(wait, 0.01))

    async def aacquire(self, model: str, max_wait: float = 0.0) -> None:
        deadline = time.monotonic() + max(0.0, max_wait)
        while True:
            granted, reason, wait = await asyncio.to_thread(self._try, model)  # the store may do I/O
            if granted:
                return
            left = deadline - time.monotonic()
            if wait > left:
                raise RateLimited(model, reason, wait)
            await asyncio.sleep(max(wait, 0.01))

    def record(self, model: str, tokens: int) -> None:
        if tokens:
            self.store.add_tokens(model, int(tokens), self.clock())

    def usage(self, model: str) -> dict[str, Any]:
        limits = self.limits_for(model) or Limits()
        used = self.store.usage(model, self.clock())
        return {
            "model": model,
            **{f"{k}_used": used.get(k, 0) for k in ("rpm", "tpm", "rpd")},
            **{f"{k}_limit": getattr(limits, k) for k in ("rpm", "tpm", "rpd")},
        }


def tokens_of(output: Any) -> int:
    """Tokens a chat model's reply reports using (0 when it doesn't say)."""
    usage = getattr(output, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if total:
        return int(total)
    return int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
