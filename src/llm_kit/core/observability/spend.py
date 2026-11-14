"""Spend so far in a period, kept in memory so budgets can be checked before every call.

A total is identified by a ``SpendKey``: whose calls (everyone's, a tenant's, a user's, or one
user's within a tenant), which touchpoint ("" = all), which period (``day``, ``week``,
``month`` or one you define), and whether calls made with keys the caller brought count.

Each total is the period's recorded spend (read from the database by the Django adapter, again
every ``REFRESH`` seconds) plus what was spent since that read. With a shared ``counter`` (the
Django adapter uses a Redis or Memcached cache), "since" covers every process's priced calls, so
a budget sees other workers' spend at once. Without one it covers this process's calls only, and
other processes' spend shows up at the next read. Either way a total can go over its budget by
about one call's cost: a call already under way finishes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

logger = logging.getLogger("llm_kit.observability")

REFRESH = 30.0  # seconds between reads of the recorded total


@dataclass(frozen=True)
class SpendKey:
    scope: str = ""  # "" everyone | "tenant" | "user" | "member" (one user within one tenant)
    scope_id: str = ""  # the tenant's or user's id; for "member": "<tenant id>:<user id>"
    touchpoint: str = ""  # "" = every touchpoint
    period: str = "day"
    counts: str = "all"  # "all" | "platform_keys" (calls on keys the caller brought don't count)

    def matches(self, call: Any) -> bool:
        if self.touchpoint and getattr(call, "touchpoint", "") != self.touchpoint:
            return False
        if self.counts == "platform_keys" and getattr(call, "key_owner", "") not in ("", "platform"):
            return False
        if self.scope == "tenant":
            return str(getattr(call, "tenant", "")) == self.scope_id
        if self.scope == "user":
            return str(getattr(call, "user", "")) == self.scope_id
        if self.scope == "member":
            return f"{getattr(call, 'tenant', '')}:{getattr(call, 'user', '')}" == self.scope_id
        return True


Seed = Callable[[SpendKey, datetime], Decimal]  # recorded spend for key since start
PeriodStart = Callable[[str], datetime]  # period name -> when the current one started


class Counter(Protocol):
    """Spend shared by every process, per ``(key, period start)``."""

    def get(self, key: SpendKey, start: datetime) -> Decimal: ...

    def add(self, key: SpendKey, start: datetime, amount: Decimal) -> None: ...


def default_period_start(period: str, now: datetime | None = None) -> datetime:
    now = now or datetime.now()
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "day":
        return day
    if period == "week":
        return day - timedelta(days=day.weekday())
    if period == "month":
        return day.replace(day=1)
    raise ValueError(f"Unknown budget period {period!r}: use day, week, month or BUDGET_PERIODS.")


@dataclass
class _Total:
    start: datetime
    recorded: Decimal  # from the last read
    local: Decimal  # this process's calls priced since then
    read_at: float  # time.monotonic() of the last read; 0 = never read
    reading: bool = False
    shared_at_read: Decimal | None = None  # the shared counter when the database was read


class SpendTracker:
    def __init__(self) -> None:
        self.seed: Seed | None = None
        self.period_start: PeriodStart = default_period_start
        self.counter: Counter | None = None
        self._totals: dict[SpendKey, _Total] = {}
        self._lock = threading.Lock()

    def configure(
        self, seed: Seed | None, period_start: PeriodStart | None = None, counter: Counter | None = None
    ) -> None:
        self.seed = seed
        self.period_start = period_start or default_period_start
        self.counter = counter
        with self._lock:
            self._totals.clear()

    def _total(self, key: SpendKey) -> _Total:
        """The entry for the current period (a new period starts at zero). Call with the lock held."""
        start = self.period_start(key.period)
        entry = self._totals.get(key)
        if entry is None or entry.start != start:
            if entry is None and len(self._totals) > 50_000:
                self._totals.clear()
            entry = self._totals[key] = _Total(start, Decimal(0), Decimal(0), 0.0)
        return entry

    def spent(self, key: SpendKey) -> Decimal:
        """Spend for ``key`` this period, reading the recorded total when it is stale.

        In a thread with a running event loop the read happens in a background thread (a
        database query must not block the loop); until it lands, the last value is used."""
        with self._lock:
            entry = self._total(key)
            stale = self.seed is not None and not entry.reading and time.monotonic() - entry.read_at > REFRESH
            if stale:
                entry.reading = True
            value = self._value(key, entry)
        if not stale:
            return value
        if _in_event_loop():
            threading.Thread(target=self._read, args=(key,), daemon=True, name="llm-kit-spend").start()
            return value
        self._read(key)
        with self._lock:
            return self._value(key, self._total(key))

    def _value(self, key: SpendKey, entry: _Total) -> Decimal:
        """Recorded spend plus what was spent since: every process's (shared counter), else this one's."""
        if self.counter is not None and entry.shared_at_read is not None:
            try:
                since = self.counter.get(key, entry.start) - entry.shared_at_read
            except Exception:
                logger.warning("Shared spend counter unavailable; using this process's spend.", exc_info=True)
            else:
                return entry.recorded + max(since, Decimal(0))
        return entry.recorded + entry.local

    def _read(self, key: SpendKey) -> None:
        started = time.monotonic()
        with self._lock:
            start = self._total(key).start
        shared = None
        if self.counter is not None:  # first: a call counted here but not yet written is missed, not doubled
            try:
                shared = self.counter.get(key, start)
            except Exception:
                shared = None
        try:
            recorded = self.seed(key, start) if self.seed is not None else Decimal(0)
        except Exception:
            logger.warning("Could not read recorded spend for %s; using this process's.", key, exc_info=True)
            with self._lock:
                entry = self._total(key)
                entry.reading = False
                entry.read_at = started  # don't retry on every call
            return
        with self._lock:
            entry = self._total(key)
            # The read includes everything already written; this process's calls before it are in it.
            entry.recorded, entry.local, entry.read_at, entry.reading = Decimal(recorded), Decimal(0), started, False
            entry.shared_at_read = shared

    def observe(self, call: Any) -> None:
        """Recorder observer: count each priced call towards every total it belongs to."""
        cost = getattr(call, "cost", None)
        if getattr(call, "status", "") != "ok" or not cost:
            return
        with self._lock:
            keys = [k for k in self._totals if k.matches(call)]
            for key in keys:
                self._total(key).local += cost


def _in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


spend = SpendTracker()


def check(key: SpendKey, limit: Decimal, what: str) -> None:
    """Raise ``BudgetExceeded`` when ``key``'s spend this period has reached ``limit``."""
    spent = spend.spent(key)
    if spent >= limit:
        from llm_kit.core.errors import BudgetExceeded

        raise BudgetExceeded(what, spent.quantize(Decimal("0.01")), limit, key.period)


def check_daily_budget(tp) -> None:
    """Raise ``BudgetExceeded`` when ``tp`` is over its daily budget (``daily_budget`` or
    ``DAILY_BUDGETS["touchpoints"][name]``)."""
    limit = daily_budget_of(tp)
    if limit is not None:
        check(SpendKey(touchpoint=tp.name), limit, f"{tp.name!r} touchpoint")


def daily_budget_of(tp) -> Decimal | None:
    from llm_kit.core import config

    configured = ((config.get("DAILY_BUDGETS") or {}).get("touchpoints") or {}).get(tp.name)
    value = configured if configured is not None else tp.daily_budget
    return None if value is None else Decimal(str(value))
