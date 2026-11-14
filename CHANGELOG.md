# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/) (before 1.0, minor versions may break things; each break is listed).

## [Unreleased]

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
