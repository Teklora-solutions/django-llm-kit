---
description: Monitoring functions, JSON endpoints, an admin dashboard, a per-provider circuit breaker, alert rules, health checks, Prometheus metrics and data retention.
---

# Monitoring

## Query functions

Build your own dashboard on the same functions the admin uses. Each returns plain data and takes the same filters: `period` (`"24h"`, `"7d"`…) or `since`/`until`, plus `touchpoint`, `model`, `provider`, `user`, `tenant`, `graph`, `agent`, `status`, `chosen_by`.

```python
from llm_kit import monitoring

monitoring.summary(period="24h")                       # calls, errors, error/fallback rate, tokens, cost, p50/p95/p99 latency, TTFT
monitoring.timeseries("cost", bucket="day", period="30d", group_by="model")
monitoring.breakdown("touchpoint", period="7d")        # per key: calls, error rate, cost, p95
monitoring.errors(period="24h")                        # by class and model, latest messages
monitoring.top("user", metric="cost", n=10)
monitoring.provider_health()                           # last 15 min per provider + circuit state
```

## JSON API

The same data over HTTP: `GET /llm/api/monitoring/{summary,timeseries,breakdown,errors,top,providers,alerts}/?period=7d&touchpoint=…`.

Staff only, unless `MONITORING_PERMISSION` names a DRF permission class.

## Admin dashboard

*LLM Kit → Monitoring dashboard*: headline numbers, spend per day, breakdowns by touchpoint and model, provider health, errors, alerts, and catalogue changes to models your touchpoints use. Light and dark theme.

## Circuit breaker

Each provider's circuit opens after repeated retryable errors (5 in 60 s by default), or at once on an exhausted quota (for 15 minutes). While it's open, touchpoint resolution moves that provider's models behind healthy fallbacks. After the cooldown, one call is let through to test it.

State is shared through Django's cache, so use Redis or Memcached when you run several workers; with the local-memory cache each process has its own circuits.

```python
LLM_KIT = {"CIRCUIT_BREAKER": {"failures": 5, "window": 60, "cooldown": 60, "quota_cooldown": 900}}
```

## Alerts

Create rules in the admin (*Alert rules*). Kinds:

| Kind | Fires when |
|---|---|
| `spend` | cost in the window is over the threshold |
| `error_rate` | the share of failed calls is over the threshold |
| `latency_p95` | p95 latency is over the threshold |
| `unpriced_share` | too many calls have no known price |
| `provider_down` | a provider's circuit is open |
| `catalogue_change` | a touchpoint's model was removed, deprecated or changed price |

Each rule takes a window, threshold, filters and a cooldown, and is delivered by email, a webhook (Slack/Discord-compatible `text` payload), and the `llm_kit.monitoring.alerts.alert_fired` signal.

Run the checks every few minutes: `python manage.py llm_kit_check_alerts`, or the Celery task `llm_kit.check_alerts`.

## Health checks

`python manage.py llm_kit_status` prints catalogue age, open circuits, touchpoint problems and the last 24 hours. It exits 1 when unhealthy, for cron, CI or a container health check (`--no-fail` always exits 0).

## Prometheus

`/llm/metrics/` (part of `llm_kit.transport.sse.urls`) serves gauges for the last hour, plus circuit state and catalogue age. It needs a staff session or `Authorization: Bearer <METRICS_TOKEN>`.

## Retention

```
python manage.py llm_kit_prune [--dry-run] [--calls DAYS] [--content DAYS] [--changes DAYS] [--threads DAYS]
```

Deletes old calls, runs, tool calls (`CALL_RETENTION_DAYS`, 90), catalogue changes and alerts (`--changes`, 180), and clears captured content (`CONTENT_RETENTION_DAYS`, 7). Threads, and their graph state, are deleted only when you pass `--threads`.
