"""Aggregations over recorded calls. Every function takes the same filters:

``period`` ("24h", "7d"…) or ``since``/``until`` (ISO datetimes), plus any of
touchpoint, model, provider, user, tenant, graph, status, chosen_by.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from django.db.models import Count, F, Q, Sum
from django.db.models.functions import TruncDay, TruncHour, TruncWeek

from llm_kit.monitoring.filters import Window

METRICS = ("calls", "errors", "cost", "input_tokens", "output_tokens", "fallbacks")
BREAKDOWN_KEYS = ("touchpoint", "model", "provider", "user", "tenant", "graph", "chosen_by", "error_class", "node")
BUCKETS = {"hour": TruncHour, "day": TruncDay, "week": TruncWeek}
PERCENTILE_SAMPLE = 50_000  # latencies fetched to compute percentiles in Python (portable across databases)

_AGG = {
    "calls": Count("id"),
    "errors": Count("id", filter=Q(status="error")),
    "cost": Sum("cost"),
    "input_tokens": Sum("input_tokens"),
    "output_tokens": Sum("output_tokens"),
    "fallbacks": Count("id", filter=Q(fallback_index__gt=0)),
    "unpriced": Count("id", filter=Q(unpriced=True)),
    "estimated": Count("id", filter=Q(usage_estimated=True)),
}


def _percentiles(values: list[int], points=(50, 95, 99)) -> dict[str, int | None]:
    if not values:
        return {f"p{p}": None for p in points}
    values = sorted(values)
    out = {}
    for p in points:
        k = max(0, min(len(values) - 1, round(p / 100 * len(values) + 0.5) - 1))  # nearest-rank
        out[f"p{p}"] = values[k]
    return out


def _latency(qs) -> dict[str, Any]:
    lat = list(qs.filter(status="ok").values_list("latency_ms", flat=True)[:PERCENTILE_SAMPLE])
    ttft = list(
        qs.filter(status="ok", time_to_first_token_ms__isnull=False).values_list("time_to_first_token_ms", flat=True)[
            :PERCENTILE_SAMPLE
        ]
    )
    return {"latency_ms": _percentiles(lat), "time_to_first_token_ms": _percentiles(ttft)}


def _rates(row: dict[str, Any]) -> dict[str, Any]:
    calls = row.get("calls") or 0
    row["cost"] = row.get("cost") or Decimal(0)
    for key in ("input_tokens", "output_tokens"):
        row[key] = row.get(key) or 0
    row["error_rate"] = round((row.get("errors") or 0) / calls, 4) if calls else 0.0
    row["fallback_rate"] = round((row.get("fallbacks") or 0) / calls, 4) if calls else 0.0
    return row


def summary(**filters) -> dict[str, Any]:
    """Totals for the window: calls, errors, tokens, cost, rates, latency percentiles, runs and tools."""
    w = Window.build(**filters)
    qs = w.calls()
    row = _rates(qs.aggregate(**_AGG))
    runs = w.runs().aggregate(
        runs=Count("id"), tool_calls=Sum("tool_calls"), interrupted=Count("id", filter=Q(status="interrupted"))
    )
    return {
        "window": w.as_dict(),
        **row,
        **_latency(qs),
        "runs": runs["runs"],
        "interrupted_runs": runs["interrupted"],
        "tool_calls": runs["tool_calls"] or 0,
    }


def timeseries(metric: str = "cost", *, bucket: str = "day", group_by: str | None = None, **filters) -> dict[str, Any]:
    """``[{"bucket": datetime, "group": str | None, "value": number}]``, oldest first."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; use {', '.join(METRICS)}.")
    if bucket not in BUCKETS:
        raise ValueError(f"Unknown bucket {bucket!r}; use {', '.join(BUCKETS)}.")
    if group_by is not None and group_by not in BREAKDOWN_KEYS:
        raise ValueError(f"Unknown group_by {group_by!r}; use {', '.join(BREAKDOWN_KEYS)}.")
    w = Window.build(**filters)
    keys = ["bucket", *([group_by] if group_by else [])]
    rows = (
        w.calls()
        .annotate(bucket=BUCKETS[bucket]("started_at"))
        .values(*keys)
        .annotate(value=_AGG[metric])
        .order_by(*keys)
    )
    points = [
        {"bucket": r["bucket"], "group": r.get(group_by) if group_by else None, "value": r["value"] or 0} for r in rows
    ]
    return {"window": w.as_dict(), "metric": metric, "bucket": bucket, "group_by": group_by, "points": points}


def breakdown(by: str = "touchpoint", *, limit: int = 50, **filters) -> dict[str, Any]:
    """Per ``by`` value: calls, errors, cost, tokens, rates and p95 latency, most expensive first."""
    if by not in BREAKDOWN_KEYS:
        raise ValueError(f"Unknown breakdown {by!r}; use {', '.join(BREAKDOWN_KEYS)}.")
    w = Window.build(**filters)
    qs = w.calls()
    # nulls_last: Postgres sorts NULL (unpriced) first in descending order; SQLite doesn't.
    rows = list(qs.values(by).annotate(**_AGG).order_by(F("cost").desc(nulls_last=True), "-calls")[:limit])
    for r in rows:
        r["key"] = r.pop(by)
        _rates(r)
        r["latency_ms"] = _percentiles(
            list(qs.filter(**{by: r["key"]}, status="ok").values_list("latency_ms", flat=True)[:PERCENTILE_SAMPLE]),
            points=(50, 95),
        )
    return {"window": w.as_dict(), "by": by, "rows": rows}


def errors(**filters) -> dict[str, Any]:
    """Failed calls by error class, and by model within each class, plus the latest messages."""
    w = Window.build(**{**filters, "status": "error"})
    qs = w.calls()
    by_class = list(qs.values("error_class").annotate(count=Count("id")).order_by("-count"))
    by_model = list(qs.values("error_class", "model").annotate(count=Count("id")).order_by("-count")[:50])
    recent = list(
        qs.order_by("-started_at").values("started_at", "touchpoint", "model", "error_class", "error_message")[:20]
    )
    return {"window": w.as_dict(), "by_class": by_class, "by_model": by_model, "recent": recent}


def top(by: str = "user", *, metric: str = "cost", n: int = 10, **filters) -> dict[str, Any]:
    """Biggest consumers (users, tenants, threads…) by a metric."""
    if by not in (*BREAKDOWN_KEYS, "thread_id"):
        raise ValueError(f"Unknown key {by!r}.")
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; use {', '.join(METRICS)}.")
    w = Window.build(**filters)
    rows = list(
        w.calls()
        .exclude(**{by: ""})
        .values(by)
        .annotate(value=_AGG[metric])
        .order_by(F("value").desc(nulls_last=True))[:n]
    )
    return {
        "window": w.as_dict(),
        "by": by,
        "metric": metric,
        "rows": [{"key": r[by], "value": r["value"] or 0} for r in rows],
    }


def provider_health(period: str = "15m", **filters) -> dict[str, Any]:
    """Per provider over a short window: calls, error rate, retryable errors, p95 latency, circuit state."""
    from llm_kit.core.observability.classify import RETRYABLE
    from llm_kit.monitoring.breaker import breaker

    w = Window.build(period=period, **filters)
    qs = w.calls().exclude(provider="")
    rows = list(
        qs.values("provider")
        .annotate(
            calls=Count("id"),
            errors=Count("id", filter=Q(status="error")),
            retryable=Count("id", filter=Q(error_class__in=RETRYABLE)),
            quota=Count("id", filter=Q(error_class="quota_exhausted")),
        )
        .order_by("provider")
    )
    states = breaker.states()
    seen = set()
    for r in rows:
        seen.add(r["provider"])
        r["error_rate"] = round(r["errors"] / r["calls"], 4) if r["calls"] else 0.0
        r["latency_ms"] = _percentiles(
            list(
                qs.filter(provider=r["provider"], status="ok").values_list("latency_ms", flat=True)[:PERCENTILE_SAMPLE]
            ),
            points=(50, 95),
        )
        r["circuit"] = states.get(r["provider"], {"state": "closed"})
    for provider, state in states.items():  # open circuits with no calls in the window
        if provider not in seen:
            rows.append(
                {
                    "provider": provider,
                    "calls": 0,
                    "errors": 0,
                    "retryable": 0,
                    "quota": 0,
                    "error_rate": 0.0,
                    "latency_ms": {},
                    "circuit": state,
                }
            )
    return {"window": w.as_dict(), "providers": rows}
