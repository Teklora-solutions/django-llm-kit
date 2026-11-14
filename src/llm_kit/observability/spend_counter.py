"""A spend counter shared by every process, in Django's cache, so budgets see all workers' spend.

Each priced call adds its cost to the counters of the budgets it counts towards (the caller's
tenant, member and user budgets, and its touchpoint's daily budget). Totals are kept in
micro-dollars under ``llm_kit:spend:<key>:<period start>`` with ``cache.incr``, which is atomic
on Redis and Memcached.

``LLM_KIT["SPEND_COUNTER"]``: ``"auto"`` (default) uses the default cache unless it is
per-process (``LocMemCache``, ``DummyCache``), ``"cache"`` always uses it, ``None`` turns it off.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from typing import Any

from llm_kit import conf
from llm_kit.core.observability.spend import SpendKey

logger = logging.getLogger("llm_kit.observability")

MICRO = Decimal(1_000_000)
TIMEOUT = 40 * 24 * 3600  # longer than a month; a new period uses new keys anyway
_PER_PROCESS = ("LocMemCache", "DummyCache")


def enabled() -> bool:
    setting = conf.get("SPEND_COUNTER")
    if not setting:
        return False
    if setting == "cache":
        return True
    from django.conf import settings

    backend = (getattr(settings, "CACHES", {}).get("default") or {}).get(
        "BACKEND", "django.core.cache.backends.locmem.LocMemCache"
    )
    return not backend.rsplit(".", 1)[-1].endswith(_PER_PROCESS)


class CacheCounter:
    def _key(self, key: SpendKey, start: datetime) -> str:
        digest = hashlib.sha1(repr(key).encode()).hexdigest()[:20]
        return f"llm_kit:spend:{digest}:{start.isoformat()}"

    def get(self, key: SpendKey, start: datetime) -> Decimal:
        from django.core.cache import cache

        return Decimal(int(cache.get(self._key(key, start), 0))) / MICRO

    def add(self, key: SpendKey, start: datetime, amount: Decimal) -> None:
        from django.core.cache import cache

        micro = int((Decimal(amount) * MICRO).to_integral_value(rounding=ROUND_CEILING))
        if micro <= 0:
            return
        name = self._key(key, start)
        try:
            cache.incr(name, micro)
        except ValueError:  # not there yet
            if not cache.add(name, micro, TIMEOUT):
                cache.incr(name, micro)  # another process created it meanwhile


def keys_for(call: Any) -> list[SpendKey]:
    """Every budget total this call counts towards, whichever process checks it."""
    from llm_kit.core.errors import UnknownTouchpoint
    from llm_kit.core.observability.spend import daily_budget_of
    from llm_kit.core.touchpoints import get_touchpoint
    from llm_kit.scoped.budgets import stored

    tenant, user, name = str(call.tenant or ""), str(call.user or ""), call.touchpoint or ""
    keys: list[SpendKey] = []
    if name:
        try:
            if daily_budget_of(get_touchpoint(name)) is not None:
                keys.append(SpendKey(touchpoint=name))
        except UnknownTouchpoint:
            pass
    budgets: list[tuple[str, str, dict]] = []
    if tenant:
        budgets += [("tenant", tenant, b) for b in stored("tenant", tenant)]
    if tenant and user:
        member = f"{tenant}:{user}"
        budgets += [("member", member, b) for b in (*stored("member", tenant), *stored("member", member))]
    if user:
        budgets += [("user", user, b) for b in stored("user", user)]
    for scope, scope_id, b in budgets:
        keys.append(SpendKey(scope, scope_id, b["touchpoint"], b["period"], b["counts"]))
    return [k for k in dict.fromkeys(keys) if k.matches(call)]


class SharedObserver:
    """Recorder observer adding each priced call to the shared counters of its budgets."""

    def __init__(self, counter: CacheCounter, period_start) -> None:
        self.counter, self.period_start = counter, period_start

    def __call__(self, call: Any) -> None:
        cost = getattr(call, "cost", None)
        if getattr(call, "status", "") != "ok" or not cost:
            return
        try:
            for key in keys_for(call):
                self.counter.add(key, self.period_start(key.period), cost)
        except Exception:
            logger.warning("Could not add a call to the shared spend counters.", exc_info=True)
