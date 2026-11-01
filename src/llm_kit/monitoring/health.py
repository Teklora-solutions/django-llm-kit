"""One health report, used by ``llm_kit_status`` and the admin dashboard."""

from __future__ import annotations

from datetime import timedelta
from typing import Any


def report() -> dict[str, Any]:
    from llm_kit import catalogue, monitoring
    from llm_kit.core.observability import recorder
    from llm_kit.core.touchpoints import all_touchpoints, problems
    from llm_kit.models import AlertEvent
    from llm_kit.monitoring.breaker import breaker
    from llm_kit.touchpoints.env import DjangoEnv

    env = DjangoEnv()
    tp_problems = {tp.name: p for tp in all_touchpoints() if (p := problems(tp, env))}
    open_circuits = {p: s for p, s in breaker.states().items() if s["state"] == "open"}
    last = catalogue.last_synced()
    day = monitoring.summary(period="24h")
    from django.utils import timezone

    alerts = list(
        AlertEvent.objects.filter(fired_at__gte=timezone.now() - timedelta(hours=24)).values_list("message", flat=True)[
            :10
        ]
    )
    issues = []
    if catalogue.is_stale():
        issues.append("catalogue is stale" if last else "catalogue was never synced")
    issues += [f"circuit open: {p} ({s['reason']})" for p, s in open_circuits.items()]
    issues += [f"touchpoint {n}: {x}" for n, ps in tp_problems.items() for x in ps]
    if recorder.dropped:
        issues.append(f"recorder dropped {recorder.dropped} records (queue full)")
    return {
        "healthy": not issues,
        "issues": issues,
        "catalogue_last_synced": last,
        "open_circuits": open_circuits,
        "touchpoint_problems": tp_problems,
        "last_24h": {k: day[k] for k in ("calls", "errors", "error_rate", "cost", "fallback_rate", "unpriced")},
        "latency_ms_24h": day["latency_ms"],
        "alerts_24h": alerts,
    }
