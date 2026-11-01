"""The filters every monitoring query accepts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from django.utils import timezone
from django.utils.dateparse import parse_datetime

_PERIOD = re.compile(r"^(\d+)\s*([mhdw])$")
_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
FILTER_FIELDS = ("touchpoint", "model", "provider", "user", "tenant", "graph", "agent", "status", "chosen_by")


def parse_period(value: str | timedelta | None) -> timedelta:
    """``"15m"``, ``"24h"``, ``"7d"``, ``"4w"`` or a timedelta. Default 24h."""
    if value is None or value == "":
        return timedelta(hours=24)
    if isinstance(value, timedelta):
        return value
    m = _PERIOD.match(str(value).strip().lower())
    if not m:
        raise ValueError(f"Invalid period {value!r}; use e.g. 15m, 24h, 7d, 4w.")
    return timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})


def _when(value: str | datetime | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    parsed = parse_datetime(str(value))
    if parsed is None:
        raise ValueError(f"Invalid datetime {value!r}; use ISO 8601.")
    return parsed if timezone.is_aware(parsed) else timezone.make_aware(parsed)


@dataclass(frozen=True)
class Window:
    since: datetime
    until: datetime
    filters: dict[str, str]

    @classmethod
    def build(cls, period=None, since=None, until=None, **filters) -> Window:
        unknown = set(filters) - set(FILTER_FIELDS)
        if unknown:
            raise ValueError(f"Unknown filter(s) {sorted(unknown)}; use {', '.join(FILTER_FIELDS)}.")
        end = _when(until) or timezone.now()
        start = _when(since) or end - parse_period(period)
        if start >= end:
            raise ValueError("'since' must be before 'until'.")
        return cls(start, end, {k: v for k, v in filters.items() if v not in (None, "")})

    def calls(self):
        from llm_kit.models import LLMCall

        return LLMCall.objects.filter(started_at__gte=self.since, started_at__lt=self.until, **self.filters)

    def runs(self):
        from llm_kit.models import LLMRun

        allowed = {k: v for k, v in self.filters.items() if k in ("user", "tenant", "graph", "agent", "status")}
        qs = LLMRun.objects.filter(started_at__gte=self.since, started_at__lt=self.until, **allowed)
        # Requests, not their subagents' runs (already inside them), unless asking about one agent.
        return qs if "agent" in allowed else qs.filter(parent_run_id__isnull=True)

    def as_dict(self) -> dict:
        return {"since": self.since.isoformat(), "until": self.until.isoformat(), **self.filters}
