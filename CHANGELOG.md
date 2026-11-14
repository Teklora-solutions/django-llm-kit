# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/) (before 1.0, minor versions may break things; each break is listed).

## [Unreleased]

### Added

- `caller=` on `model()`, `complete()`, `acomplete()`, `embed()` and `aembed()`: who a call outside a request or agent run is for. Their saved settings, keys and budgets apply, and the call records their user and tenant. A caller given in `config` is now recorded too (#24).

- Testing provider keys: one tiny call with the key alone, with the result stored on the key (migration 0015). Available as `POST llm/api/keys/<id>/test/`, `verify` when adding a key (a refused key isn't saved), a Test button in the keys template, and an admin action.

- The example project has a `/settings/` page built from the settings templates.
- Settings templates: `{% llm_kit_settings %}`, `{% llm_kit_keys %}` and `{% llm_kit_budgets %}` render settings, brought keys and budgets as forms. They post to new views under `llm/settings/` and can be overridden template by template.

- Budgets per tenant, per member of a tenant, and per user, for any period (`BUDGET_PERIODS`: day, week, month or your own), optionally for one touchpoint and optionally counting only calls on platform keys. `BUDGET_EDITORS` decides who sets them. Checked before every call. `GET/PUT/DELETE llm/api/budgets/` returns spent, remaining and when the period resets. New `ScopedBudget` table (migration 0014). The spend tracker now keeps a total per scope, touchpoint and period.

- Bring your own key: tenants and users store provider keys (`BYOK_SCOPES`, `GET/POST/DELETE llm/api/keys/`). Each touchpoint's `KeyPolicy` (`Settable(keys=)` or `KEY_POLICY`) decides whose keys a call uses: `platform_only`, `own_only` (raises `NoKey` without one), `own_then_platform` (theirs, then yours as the next attempt), or `tenant_chooses`. Calls record `key_owner` (migration 0013).

- Scoped settings: `touchpoint(..., settable=Settable(model=..., params=...))` lets tenants and users save a model choice and parameters, kept within the touchpoint's choice. `llm_kit.scoped.register(Setting(...))` and `scoped.value()` add your own settings. `SCOPE_PERMISSIONS` decides who may change each scope. `GET/PATCH/DELETE llm/api/settings/` describes and saves them for UIs. New `ScopedSetting` table (migration 0012).

- Daily budgets per touchpoint: `touchpoint(..., daily_budget=)` or `DAILY_BUDGETS["touchpoints"]`. Checked before every call against a running total kept in memory and re-read from recorded calls every 30 s. Calls over the limit raise `BudgetExceeded`, and agents stop with `stopped="budget"`.

- Many tools: agents with more than `DEFER_TOOLS_ABOVE` tools load them on demand. The prompt gets a category overview instead of every schema. `find_tools` ranks by words, with rarer words counting more, takes a `category`, and can search by meaning through `TOOL_SEARCH_TOUCHPOINT`. Only the `MAX_LOADED_TOOLS` most recently found tools stay bound.

- Embeddings: `touchpoint.embed()` / `aembed()` with the same resolution, keys, fallbacks, retry waits, circuit breaker and recording as chat calls. Inputs may be media parts for backends that take them. `EMBEDDING_BACKEND` setting; `LLMCall.kind` (`chat` or `embedding`, migration 0011).

- A touchpoint called inside a running agent without a `config` (`complete()`, `acomplete()`, `model()` with `run_config()`) joins that agent's request: recorded in its run tree, counted in its totals, and using its `retry` policy.
- `TENANT_RESOLVER` may take `request` and `scope`: the HTTP views pass the request and the WebSocket consumer the Channels scope, so a user in several tenants can pick one per request (a header, say). Raising `PermissionDenied` refuses the request (403, or WebSocket close 4403).
- Budgets: `AGENT_BUDGET` / `agents.run(budget=)` cap what a request may spend (its agents stop with `stopped="budget"`), and `DAILY_BUDGETS` refuses requests from a user or tenant over its daily spend (`BudgetExceeded`, 429).

### Changed

- `defer_tools` defaults to `None` (defer when there are more than `DEFER_TOOLS_ABOVE` tools) instead of `False`. Pass `defer_tools=False` to keep every tool bound.

### Fixed

- `embed()` ignored a touchpoint's dotted-path or callable `backend=` and fell back to `EMBEDDING_BACKEND`. It now uses `touchpoint(embedding_backend=)`, then `backend=`, then the setting. A factory that returns a chat model raises an error that names it (#17).

## [0.2.0] - 2026-11-14

### Added

- **Touchpoints:** each place your code calls a model declares its default, fallbacks, required capabilities and whether clients may pick the model. Admin overrides can narrow it without a deploy. Backends: LiteLLM (with key pools), LangChain `init_chat_model`, or your own factory. Provider keys from env vars, a `CREDENTIALS` hook, or encrypted in the database.
- **Model catalogue:** prices, limits and capabilities for about 12,600 models, merged from models.dev, OpenRouter and LiteLLM, with admin overrides, a change log and an offline snapshot.
- **Observability:** every model call, tool call and run recorded with tokens, cost, latency, fallbacks and classified errors. Sinks: database, JSON log, OpenTelemetry, or your own.
- **Monitoring:** query functions, JSON API, admin dashboard, per-provider circuit breaker, alert rules, `llm_kit_status`, Prometheus metrics and `llm_kit_prune`.
- **Agent tools from Django models:** `AgentAccess` declarations generate find/save tools scoped to the caller's rows, with Django permissions and approval proposals.
- **Agents and subagents:** one harness for every agent; subagents are the same harness with another model, prompt and tools. Delegation, depth/hand-off/time limits, history windows, deferred tools, admin overrides and a recorded run tree.
- **Thread ownership:** threads belong to a user, a visitor's session, a key your code chose, or nobody; visitor threads move to the account at sign-in; tenant isolation.
- **Checkpointer choice:** `CHECKPOINTER` can be `DjangoSaver`, LangGraph's `PostgresSaver`, or your own saver or factories, with one async saver per event loop.
- **Waiting out rate limits:** a short rate limit or overload is waited out on the same model before falling back, as long as the touchpoint's `RetryPolicy` allows (`RETRY` setting, `touchpoint(retry=)`, `model(retry=)`, `agents.run(retry=)`). Daily quotas are never waited out.
- `AllModelsFailed` when every model of a touchpoint fails, listing each model's error.
- `model(fallbacks=False)` / `complete(fallbacks=False)`: only the resolved model, for one call.
- `agents.run` inside a running agent joins its request: same caller and limits, recorded in its run tree, streamed with it (`join=False` for a separate request).
- Documentation site with Markdown versions of every page and `llms.txt`.

### Changed

- The package is split into a Django-free core (`llm_kit.core`) and Django adapters grouped by feature. URL includes moved to `llm_kit.transport.sse.urls`, `llm_kit.transport.api.urls` and `llm_kit.transport.ws.routing`.
- With `REQUIRE_AUTH=False`, threads with no owner are staff-only; visitors use only their own threads.
- `EXPOSE_ERRORS` controls error details in events (defaults to `DEBUG`).
- `Touchpoint.model()` returns the chat model itself only when there is one candidate and nothing to wait for (`retry="never"`); otherwise a `TouchpointChatModel`.
- Gemini's per-minute 429 ("exceeded your current quota … PerMinute") is classified as `rate_limit`, not `quota_exhausted`, so the circuit breaker no longer gives it the long quota cooldown.

### Fixed

- Provider errors that carry their HTTP status only in the message (LangChain's Google client: "503 UNAVAILABLE", "429 RESOURCE_EXHAUSTED") are classified, so they are waited out and counted by the circuit breaker.
- The Postgres checkpointer's per-loop connection pool could make `asyncio.run` hang at exit when a pool worker was cancelled mid-job.

### Removed

- The v0.1 `LLMModel` table, the `MODELS` and `DEFAULT_MODEL` settings, and `GET models/`. Use touchpoints and the catalogue instead.

## [0.1.0] - 2026-11-01

First version: `DjangoSaver` checkpointer, SSE streaming views, DRF API, Channels WebSocket consumer, and a model table.
