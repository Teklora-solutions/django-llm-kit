"""``/llm/metrics/``: Prometheus text format. Gauges over the last hour, plus current state.

Auth: a staff session, or ``Authorization: Bearer <LLM_KIT["METRICS_TOKEN"]>``.
"""

from __future__ import annotations

import hmac

from django.http import HttpResponse, HttpResponseForbidden
from django.views.decorators.http import require_GET

from llm_kit import conf


def _esc(value) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _line(name: str, labels: dict, value) -> str:
    lbl = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
    return f"{name}{{{lbl}}} {value}" if lbl else f"{name} {value}"


def _authorized(request) -> bool:
    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated and user.is_staff:
        return True
    token = conf.get("METRICS_TOKEN")
    header = request.headers.get("Authorization", "")
    return bool(token) and hmac.compare_digest(header, f"Bearer {token}")


@require_GET
def metrics(request):
    if not _authorized(request):
        return HttpResponseForbidden("staff session or bearer token required\n")
    from django.db.models import Count, Q, Sum
    from django.utils import timezone

    from llm_kit import catalogue, monitoring
    from llm_kit.core.observability import recorder
    from llm_kit.monitoring.breaker import breaker
    from llm_kit.monitoring.filters import Window

    out: list[str] = []

    def metric(name, kind, help_text, samples):
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {kind}")
        out.extend(_line(name, labels, value) for labels, value in samples)

    qs = Window.build(period="1h").calls()
    rows = list(
        qs.values("touchpoint", "model", "provider").annotate(
            calls=Count("id"),
            errors=Count("id", filter=Q(status="error")),
            cost=Sum("cost"),
            tin=Sum("input_tokens"),
            tout=Sum("output_tokens"),
        )
    )
    key = lambda r: {"touchpoint": r["touchpoint"], "model": r["model"], "provider": r["provider"]}  # noqa: E731
    metric("llm_kit_calls_1h", "gauge", "Model calls in the last hour.", [(key(r), r["calls"]) for r in rows])
    metric("llm_kit_errors_1h", "gauge", "Failed model calls in the last hour.", [(key(r), r["errors"]) for r in rows])
    metric(
        "llm_kit_cost_usd_1h", "gauge", "Spend in the last hour (USD).", [(key(r), float(r["cost"] or 0)) for r in rows]
    )
    metric(
        "llm_kit_tokens_1h",
        "gauge",
        "Tokens in the last hour.",
        [({**key(r), "direction": "input"}, r["tin"] or 0) for r in rows]
        + [({**key(r), "direction": "output"}, r["tout"] or 0) for r in rows],
    )
    lat = monitoring.summary(period="1h")["latency_ms"]
    metric(
        "llm_kit_latency_ms_1h",
        "gauge",
        "Latency percentiles of successful calls, last hour.",
        [({"quantile": q[1:]}, v) for q, v in lat.items() if v is not None],
    )
    metric(
        "llm_kit_circuit_open",
        "gauge",
        "1 when a provider's circuit is open.",
        [({"provider": p}, 1 if s["state"] == "open" else 0) for p, s in breaker.states().items()],
    )
    last = catalogue.last_synced()
    if last:
        metric(
            "llm_kit_catalogue_age_seconds",
            "gauge",
            "Seconds since the catalogue last synced.",
            [({}, int((timezone.now() - last).total_seconds()))],
        )
    metric(
        "llm_kit_recorder_dropped_total",
        "counter",
        "Records dropped because the queue was full.",
        [({}, recorder.dropped)],
    )
    return HttpResponse("\n".join(out) + "\n", content_type="text/plain; version=0.0.4")
