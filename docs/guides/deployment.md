---
description: Running django-llm-kit in production - ASGI, caches, scheduled jobs, keys, and retention.
---

# Deployment

## Serve with ASGI

Streaming needs an ASGI server: Daphne, Uvicorn or Hypercorn. Under WSGI the SSE views still work, but each reply is buffered until the run ends.

Behind nginx, the SSE responses already send `X-Accel-Buffering: no`. Make sure proxy timeouts are longer than your longest run, or rely on the keep-alive comments (`SSE_KEEPALIVE`, 15 s).

## Use a shared cache

The [circuit breaker](monitoring.md#circuit-breaker) and the versioned caches for touchpoint overrides, agent overrides and stored keys live in Django's cache. With several processes, use Redis or Memcached so they agree; the local-memory cache gives each process its own view.

## Schedule the jobs

| Job | Every | Command | Celery task |
|---|---|---|---|
| catalogue sync | few hours | `llm_kit_sync_catalogue` | `llm_kit.sync_catalogue` |
| alert checks | few minutes | `llm_kit_check_alerts` | `llm_kit.check_alerts` |
| retention | daily | `llm_kit_prune` | — |
| health | as you like | `llm_kit_status` (exit 1 when unhealthy) | — |

## Keys

Keep provider keys in environment variables, a secret store behind `CREDENTIALS`, or encrypted in the database with `ENCRYPTION_KEYS`. To rotate encryption keys, put the new key first in the list; old keys still decrypt.

## Checks before release

```
python manage.py check --deploy
python manage.py llm_kit_touchpoints --check   # every touchpoint resolves, is priced and has a key
python manage.py llm_kit_agents
```

## Postgres

On Postgres you can keep `DjangoSaver`, or switch to LangGraph's own saver with `CHECKPOINTER = "postgres"` (see [Checkpointers](checkpointer.md)). Close pooled savers on shutdown with `get_checkpointer().close()`.
