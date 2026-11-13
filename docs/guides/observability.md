---
description: Every model call, tool call and run is recorded with tokens, cost, latency, fallbacks and classified errors, to the database, JSON logs, OpenTelemetry or your own sink.
---

# Observability

Every call through a touchpoint, and every graph or agent run started through llm-kit, is recorded. There is nothing to add to your code.

## What is recorded

| Record | What it holds |
|---|---|
| **LLM call** (one per attempt) | touchpoint, graph/node, agent, thread, user, tenant; requested → resolved → used → served model; who chose it; fallback index; input/output/cached/reasoning tokens (estimated and flagged when a provider sends none); cost; latency and time to first token; status and error class |
| **LLM run** (graph run, agent run or `complete()`) | totals: calls, tool calls, tokens, cost, unpriced calls, duration; status (`ok`, `error`, `interrupted`, `cancelled`); parent run and depth for subagents |
| **Tool call** | name, duration, status, input/output hashes |

- **Cost** comes from the [catalogue](catalogue.md), including cached-token and long-context pricing. A model without a known price is flagged `unpriced`; it is never counted as $0.
- **A failed primary and the fallback that answered are two rows**, so you can see how often each provider fails.
- **Content is not stored** by default, only hashes. Opt in per touchpoint with `touchpoint(..., capture=True)` or globally with `CAPTURE_CONTENT`. Captured content is cleared after `CONTENT_RETENTION_DAYS` by `llm_kit_prune`.

## Errors

Errors are classified the same way for every provider and client library:

`rate_limit`, `quota_exhausted`, `timeout`, `auth`, `overloaded`, `context_length`, `content_filter`, `bad_request`, `connection`.

`rate_limit` and `quota_exhausted` are kept apart because they need different handling: one passes in seconds, the other needs a person. Use the classifier in your own code too:

```python
from llm_kit.observability import classify_error, is_retryable

kind = classify_error(exc)
```

## Context outside requests

Requests carry the user automatically. In Celery tasks, scripts or management commands, set it yourself:

```python
from llm_kit import observability

with observability.context(user=user, tenant=org, tags={"job": "nightly"}):
    summarise.complete(text)
```

## It never slows a call down

Callbacks only enqueue records; a background thread prices and writes them in batches. A failing sink never breaks a call. `observability.flush()` waits for pending writes, for tests and scripts that exit right after.

## Sinks

`RECORD_SINKS` (default `["database", "log"]`):

| Sink | Writes |
|---|---|
| `"database"` | the *LLM calls*, *LLM runs* and *Tool calls* tables (admin and [monitoring](monitoring.md) read these) |
| `"log"` | one JSON line per record on the `llm_kit.calls` logger, never including content |
| `"otel"` | OpenTelemetry spans with the GenAI semantic conventions (`[otel]` extra) |
| `"myapp.sinks:Sink"` | your own class with `write(calls, runs, tools)` |

`OBSERVABILITY: False` turns recording off.

## What isn't recorded

Every llm-kit call goes through LangChain, so the LangChain callback sees them all. Direct `litellm.completion()` calls in your own code bypass touchpoints and are not recorded.
