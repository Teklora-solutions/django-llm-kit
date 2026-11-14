---
description: Every key of the LLM_KIT settings dict, with its default and meaning.
---

# Settings

All settings live in one dict in `settings.py`:

```python
LLM_KIT = {
    "GRAPHS": {...},
    "REQUIRE_AUTH": True,
    ...
}
```

## Graphs and transports

| Key | Default | Meaning |
|---|---|---|
| `GRAPHS` | `{}` | name → dotted path, or `{"graph": path, "touchpoint": name}`. See [Graphs](../guides/graphs.md) |
| `DEFAULT_GRAPH` | `None` | graph used when a request doesn't name one; falls back to the only graph |
| `CHECKPOINTER` | `None` | `None` (`DjangoSaver`), `"postgres"`, a dotted path, or `{"sync": …, "async": …}`. See [Checkpointers](../guides/checkpointer.md) |
| `STREAM_MODES` | `["messages", "updates", "custom"]` | LangGraph stream modes for SSE/WebSocket runs (`custom` carries subagents' messages) |
| `SSE_KEEPALIVE` | `15` | seconds between keep-alive comments; `0` turns them off |
| `EXPOSE_ERRORS` | `DEBUG` | include exception details in `error` events |

## Access

| Key | Default | Meaning |
|---|---|---|
| `REQUIRE_AUTH` | `True` | every endpoint needs a signed-in user; `False` lets visitors chat on their own threads |
| `ADOPT_SESSION_THREADS` | `True` | move a visitor's threads to their account when they sign in |
| `SCOPE_PERMISSIONS` | `{}` | who may change each scope's saved settings: `{"platform" \| "tenant" \| "user": "dotted.path:fn(user, tenant) -> bool"}` ([details](../guides/settings.md#who-may-change-what)) |
| `BYOK_SCOPES` | `()` | scopes that may store their own provider keys: `"tenant"`, `"user"` ([bring your own key](../guides/settings.md#bring-your-own-key)) |
| `KEY_POLICY` | `"platform_only"` | whose keys touchpoints use unless `Settable(keys=)` says: `platform_only`, `own_only`, `own_then_platform`, `tenant_chooses` |
| `BUDGET_PERIODS` | `("day", "month")` | periods tenant, member and user budgets may use; names, or `{"name": "dotted.path:fn(now) -> start"}` |
| `BUDGET_EDITORS` | `{}` | who may set each kind of budget: `{"tenant" \| "member" \| "user": ["platform" \| "tenant" \| "user"]}` ([budgets](../guides/settings.md#budgets)) |
| `SPEND_COUNTER` | `"auto"` | budgets see every process's spend at once through Django's cache: `"auto"` (unless the cache is per-process), `"cache"`, or `None` |
| `TENANT_MEMBERS` | `None` | dotted path to `fn(tenant_id) -> users or ids`: lets tenant managers set one member's budget, and lists members for pickers |
| `TENANT_RESOLVER` | `None` | dotted path to `fn(user) -> tenant`, or `fn(user, request=None, scope=None)` to pick a tenant per request ([users in several tenants](../guides/access.md#users-in-several-tenants)); scopes threads, model tools and call records |

## Models and keys

| Key | Default | Meaning |
|---|---|---|
| `MODEL_BACKEND` | `"auto"` | `litellm`, `langchain`, or a dotted path to `fn(ref, params, api_keys) -> BaseChatModel` |
| `EMBEDDING_BACKEND` | `"auto"` | `litellm`, `langchain`, or a dotted path to `fn(ref, params, api_keys)` returning an embedder ([embeddings](../guides/touchpoints.md#embeddings)) |
| `LITELLM_ROUTER` | `{"routing_strategy": "simple-shuffle", "num_retries": 0}` | kwargs for the LiteLLM Router used with key pools |
| `CREDENTIALS` | `None` | dotted path to `fn(provider)` returning a secret, a list of secrets, or `None`; asked before stored keys and env vars |
| `ENCRYPTION_KEYS` | `None` | Fernet keys or passphrases for provider keys stored in the database, newest first |
| `RETRY` | `{"attempts": 1, "max_wait": 6.0}` | waiting out a short rate limit on a model before falling back: waits per model and the longest wait in seconds. Per touchpoint or call: `retry=` ([details](../guides/touchpoints.md#rate-limits-and-fallbacks)) |
| `CATALOGUE_SOURCES` | all | source names or dotted paths |
| `CATALOGUE_MAX_AGE_HOURS` | `24` | when the catalogue counts as stale |

## Agents

| Key | Default | Meaning |
|---|---|---|
| `AGENT_MAX_DEPTH` | `2` | how deep subagents may nest (`0`: no delegation) |
| `AGENT_MAX_DELEGATIONS` | `8` | hand-offs per request, across all its agents |
| `AGENT_TIMEOUT` | `300` | seconds after which a request's agents start no new model call |
| `DEFER_TOOLS_ABOVE` | `20` | agents with more tools than this load them on demand through `find_tools`, unless they declare `defer_tools` ([many tools](../guides/agents.md#many-tools)) |
| `MAX_LOADED_TOOLS` | `15` | how many found tools stay bound, most recently found first |
| `TOOL_SEARCH_TOUCHPOINT` | `None` | an embedding touchpoint; `find_tools` then searches by meaning as well as by words |
| `AGENT_BUDGET` | `None` | USD a request may spend; then its agents stop with `stopped="budget"` (per run: `agents.run(budget=)`; [budgets](../guides/agents.md#budgets)) |
| `DAILY_BUDGETS` | `{}` | USD per day for `"user"` and `"tenant"` (a request over it is refused, 429), and `"touchpoints": {name: USD}` (calls over it raise `BudgetExceeded`); from recorded calls |

## Recording and monitoring

| Key | Default | Meaning |
|---|---|---|
| `OBSERVABILITY` | `True` | record calls, runs and tool calls |
| `RECORD_SINKS` | `["database", "log"]` | `database`, `log`, `otel`, or dotted paths to sink classes |
| `CAPTURE_CONTENT` | `False` | store prompts and responses (per touchpoint: `capture=`) |
| `CONTENT_RETENTION_DAYS` | `7` | captured content kept this long by `llm_kit_prune` |
| `CALL_RETENTION_DAYS` | `90` | calls, runs and tool calls kept this long by `llm_kit_prune` |
| `CIRCUIT_BREAKER` | `{}` | overrides of `{"failures": 5, "window": 60, "cooldown": 60, "quota_cooldown": 900}` |
| `MONITORING_PERMISSION` | staff only | dotted path to a DRF permission class for `/llm/api/monitoring/` |
| `METRICS_TOKEN` | `None` | bearer token for `/llm/metrics/` |
