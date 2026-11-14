"""Today's spend per touchpoint, kept in memory so a daily budget can be checked before every call.

The total is today's recorded spend (read from the database by the Django adapter, again
every ``REFRESH`` seconds, which also picks up other processes' calls), plus the cost of
this process's calls priced since that read. It is approximate: calls priced in another
process show up at the next read, and a call can go over by about one call's cost.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

logger = logging.getLogger("llm_kit.observability")

REFRESH = 30.0  # seconds between reads of the recorded total

Seed = Callable[[str], Decimal]  # touchpoint -> USD recorded since local midnight
Today = Callable[[], date]


@dataclass
class _Total:
    day: date
    recorded: Decimal  # from the last read
    local: Decimal  # this process's calls priced since then
    read_at: float  # time.monotonic() of the last read; 0 = never read
    reading: bool = False


class SpendTracker:
    def __init__(self) -> None:
        self.seed: Seed | None = None
        self.today: Today = date.today
        self._totals: dict[str, _Total] = {}
        self._lock = threading.Lock()

    def configure(self, seed: Seed | None, today: Today | None = None) -> None:
        self.seed = seed
        self.today = today or date.today
        with self._lock:
            self._totals.clear()

    def _total(self, touchpoint: str) -> _Total:
        """The entry for today (a new day starts at zero). Call with the lock held."""
        today = self.today()
        entry = self._totals.get(touchpoint)
        if entry is None or entry.day != today:
            entry = self._totals[touchpoint] = _Total(today, Decimal(0), Decimal(0), 0.0)
        return entry

    def add(self, touchpoint: str, cost: Decimal | None) -> None:
        """A call of ``touchpoint`` was priced at ``cost``."""
        if not touchpoint or not cost:
            return
        with self._lock:
            self._total(touchpoint).local += cost

    def spent(self, touchpoint: str) -> Decimal:
        """Today's spend for ``touchpoint``, reading the recorded total when it is stale.

        In a thread with a running event loop the read happens in a background thread (a
        database query must not block the loop); until it lands, the last value is used."""
        with self._lock:
            entry = self._total(touchpoint)
            stale = self.seed is not None and not entry.reading and time.monotonic() - entry.read_at > REFRESH
            if stale:
                entry.reading = True
            value = entry.recorded + entry.local
        if not stale:
            return value
        if _in_event_loop():
            threading.Thread(target=self._read, args=(touchpoint,), daemon=True, name="llm-kit-spend").start()
            return value
        self._read(touchpoint)
        with self._lock:
            entry = self._total(touchpoint)
            return entry.recorded + entry.local

    def _read(self, touchpoint: str) -> None:
        started = time.monotonic()
        try:
            recorded = self.seed(touchpoint) if self.seed is not None else Decimal(0)
        except Exception:
            logger.warning("Could not read today's spend for %s; using this process's.", touchpoint, exc_info=True)
            with self._lock:
                entry = self._total(touchpoint)
                entry.reading = False
                entry.read_at = started  # don't retry on every call
            return
        with self._lock:
            entry = self._total(touchpoint)
            # The read includes everything already written; this process's calls before it are in it.
            entry.recorded, entry.local, entry.read_at, entry.reading = Decimal(recorded), Decimal(0), started, False

    def observe(self, call: Any) -> None:
        """Recorder observer: count each priced call towards its touchpoint."""
        if getattr(call, "status", "") == "ok":
            self.add(getattr(call, "touchpoint", ""), getattr(call, "cost", None))


def _in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


spend = SpendTracker()


def check_daily_budget(tp) -> None:
    """Raise ``BudgetExceeded`` when ``tp`` is over its daily budget (``daily_budget`` or
    ``DAILY_BUDGETS["touchpoints"][name]``)."""
    limit = daily_budget_of(tp)
    if limit is None:
        return
    spent = spend.spent(tp.name)
    if spent >= limit:
        from llm_kit.core.errors import BudgetExceeded

        raise BudgetExceeded(f"{tp.name!r} touchpoint", spent.quantize(Decimal("0.01")), limit)


def daily_budget_of(tp) -> Decimal | None:
    from llm_kit.core import config

    configured = ((config.get("DAILY_BUDGETS") or {}).get("touchpoints") or {}).get(tp.name)
    value = configured if configured is not None else tp.daily_budget
    return None if value is None else Decimal(str(value))
