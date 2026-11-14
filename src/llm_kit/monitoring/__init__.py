"""Monitoring: the numbers behind any dashboard, as plain Python data.

    from llm_kit import monitoring

    monitoring.summary(period="24h", touchpoint="chat.assistant")
    monitoring.timeseries("cost", bucket="day", period="30d", group_by="model")
    monitoring.breakdown("touchpoint", period="7d")
    monitoring.errors(period="24h")
    monitoring.provider_health()
    monitoring.readiness()      # configured providers' credentials, before any call is made

The JSON API (``llm_kit.transport.api``) and the admin dashboard use these same functions.
"""

from llm_kit.monitoring.queries import (
    BREAKDOWN_KEYS,
    METRICS,
    breakdown,
    errors,
    provider_health,
    summary,
    timeseries,
    top,
)
from llm_kit.monitoring.readiness import readiness

__all__ = [
    "BREAKDOWN_KEYS",
    "METRICS",
    "breakdown",
    "errors",
    "provider_health",
    "readiness",
    "summary",
    "timeseries",
    "top",
]
