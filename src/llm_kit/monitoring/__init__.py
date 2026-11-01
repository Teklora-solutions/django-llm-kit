"""Monitoring: the numbers behind any dashboard, as plain Python data.

    from llm_kit import monitoring

    monitoring.summary(period="24h", touchpoint="chat.assistant")
    monitoring.timeseries("cost", bucket="day", period="30d", group_by="model")
    monitoring.breakdown("touchpoint", period="7d")
    monitoring.errors(period="24h")
    monitoring.provider_health()

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

__all__ = ["BREAKDOWN_KEYS", "METRICS", "breakdown", "errors", "provider_health", "summary", "timeseries", "top"]
