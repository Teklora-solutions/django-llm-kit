---
description: The public Python entry points of django-llm-kit, by module.
---

# Python API

The names below are the supported public API. Anything else, in particular `llm_kit.core.*` internals not listed here, may change between minor versions before 1.0.

## `llm_kit`

| Name | What it is |
|---|---|
| `touchpoint(name, *, default, fallbacks=(), params=None, choice=None, requires=None, description="", backend=None, capture=None, retry=None)` | declare a [touchpoint](../guides/touchpoints.md); returns a `Touchpoint` |
| `RetryPolicy(attempts=0, max_wait=0.0)`, `RetryPolicy.BATCH`, `.INTERACTIVE`, `.NEVER` | how long a call waits out a [short rate limit](../guides/touchpoints.md#rate-limits-and-fallbacks) before falling back |
| `AllModelsFailed` | raised when every model of a touchpoint failed; `errors` is `[(model, exception), …]` |
| `get_touchpoint(name)` | a declared touchpoint |
| `ServerChoice(on_requested="ignore")`, `ClientChoice(allow=("*",), requires={}, max_price_input=None, max_price_output=None, on_disallowed="reject", use_fallbacks=True)` | model choice policies |
| `agent(name, *, description, touchpoint, prompt, tools, subagents, output, max_steps, fork, defer_tools, essential, history, tool_result_chars)` | declare an [agent](../guides/agents.md); `@agent.code(name, …)` for Python agents |
| `get_graph(name=None)` | the compiled graph or agent, cached per process |
| `DjangoSaver(using="default")` | the ORM [checkpointer](../guides/checkpointer.md) |

### `Touchpoint`

| Method | Returns |
|---|---|
| `model(requested=None, *, config=None, retry=None, fallbacks=True, **params)` | a LangChain chat model with fallbacks and retry waits; `params` override the touchpoint's for this call; `fallbacks=False` uses only the resolved model |
| `complete(prompt, *, schema=None, requested=None, config=None, retry=None, fallbacks=True, **params)` | one-shot reply; a validated `schema` instance when given (native structured output, or JSON parsed from the reply) |
| `resolve(requested=None, *, config=None)` | the resolution: chosen model, fallbacks, who chose and why. Raises `ModelNotAllowed` or `TouchpointDisabled` |

## `llm_kit.agents`

`run(name, task, *, user=None, tenant=None, thread_id=None, thread=None, join=True, retry=None, **config) -> AgentResult`, `arun(...)`, `stream(...)` (async iterator of events). Inside a running agent, `run`/`arun` [join its request](../guides/agents.md#running-agents) unless `join=False`. `AgentResult` has `text`, `output`, `steps`, `stopped` and token counts.

## `llm_kit.tools`

`expose(model, **options)`, `register(tool, *, permission=…, category=…)`, `select(*, names, categories, kinds)`, `for_user(user, tenant=…)`, `for_config(config)`, `all_tools()`, `guidance(tools)`, `pending_for(user)`, `approve(proposal, by)`, `reject(proposal, by, reason="")`, `PUBLIC`, `Caller`. See [Model tools](../guides/tools.md).

## `llm_kit.catalogue`

`get(ref)`, `search(requires=…, max_price_input=…, usable_only=…)`, `has_credentials(provider)`, `is_stale()`. See [Catalogue](../guides/catalogue.md).

## `llm_kit.observability`

`context(*, user=None, tenant=None, tags=None)`, `flush(timeout=10.0)`, `classify_error(exc)`, `is_retryable(exc)`. See [Observability](../guides/observability.md).

## `llm_kit.monitoring`

`summary`, `timeseries`, `breakdown`, `errors`, `top`, `provider_health`. See [Monitoring](../guides/monitoring.md).

## `llm_kit.runtime`

| Name | What it is |
|---|---|
| `threads.open_thread(graph, *, key, user=None, tenant=None, title="")`, `threads.aopen_thread(...)` | get or create a [keyed thread](../guides/access.md#threads-your-code-keys) |
| `access.adopt_session_threads(request, user=None)` | move a visitor's threads to a user |
| `graphs.get_checkpointer()` | the configured checkpointer |

## Models

`llm_kit.models`: `Thread`, `LLMCall`, `LLMRun`, `ToolCall`, `ToolProposal`, `TouchpointOverride`, `AgentOverride`, `ProviderKey`, `AlertRule`, `AlertEvent`, the `Catalogue*` tables and the `Checkpoint*` tables. All are registered in the admin.
