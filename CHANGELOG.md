# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/) (before 1.0, minor versions may break things; each break is listed).

## [Unreleased]

### Added

- Generated media through touchpoints (#48): `tp.generate(kind, **request)` / `agenerate` make images, video, speech, transcripts, document text, and kinds you register (`register_kind(MediaKind(...))`). They get the same keys, rate limits, retry waits, fallbacks, records and budgets as chat and embeddings ([guide](docs/guides/media.md)).
  - **Backends:** LiteLLM by default (`MEDIA_BACKEND`), or `touchpoint(..., media_backend=)` with your own `fn(ref, params, api_keys)`.
  - **Long jobs** (video) are submitted, then checked every `MEDIA_POLL_EVERY` seconds for at most `MEDIA_JOB_WAIT`. A job that runs out of time raises `MediaJobTimeout` with its id and is never sent to the next model.
  - **Refusals:** a provider's content-policy refusal raises `ContentRefused` and doesn't try the next model unless the touchpoint has `fallback_on_refusal=True`.
  - **`Media`** takes bytes, paths, URLs, `data:` URIs and files, and streams into Django storage (`save()`, `MEDIA_STORAGE`, `MEDIA_PATH`). Fetching is limited to public http(s) URLs, checked as each connection opens (`MEDIA_FETCH_PRIVATE`), and to `MEDIA_MAX_BYTES`. A string is never read as a file path.
  - **Records and prices:** `LLMCall.units` records what a media call was billed by (images, seconds, characters...) (migration 0021). The cost is what the backend reported, else the units times the catalogue's per-unit prices.
- Images (#48): `tp.image(prompt, n=, size=, quality=, style=)` and `tp.edit_image(prompt, images, mask=)`, with async forms, through LiteLLM's `image_generation` and `image_edit` for every provider it supports. Images returned as URLs are downloaded before the call returns (`fetch=False` keeps the URLs). `revised_prompt`, image tokens and LiteLLM's reported cost are kept.
- Speech and transcription (#48). `tp.speak(text, voice=, format=, speed=, instructions=)` reads text aloud. `tp.transcribe(audio, language=, prompt=, format=, timestamps=)` writes speech down, with `language`, `duration`, `segments` and `words` when the provider gives them. Both have async forms and go through LiteLLM's `speech` and `transcription`. Units are characters and seconds, plus audio tokens for models billed by tokens. WAV clips are measured by the kit when the provider doesn't say.
- Video and background media jobs (#48).
  - **Video:** `tp.video(prompt, image=, seconds=, size=, resolution=)` waits for the video, through LiteLLM's `video_generation`, `video_status` and `video_content` (OpenAI, Google, Runway, fal, Replicate, xAI, MiniMax...).
  - **Background jobs:** `tp.start_video(...)` and `tp.start(kind, ...)` run media as a `BackgroundRun` with `kind="media"` and no thread, on the background executors. They come with watch events (`progress`, `media`, `status`), cancelling (at the provider too), lost-run detection and the `run_finished` signal, plus the new `media_job_finished` signal. Their media are saved to storage, and `run.result` has the names, URLs and the provider's job id. The job's owner may watch, list (`GET llm/api/runs/?kind=media`) and cancel it (migration 0022).
  - **Provider webhooks:** `POST llm/media/webhooks/<provider>/` (`MEDIA_WEBHOOKS`, `MEDIA_WEBHOOK_URL`) wakes a job so it checks the provider at once. Checks: Standard Webhooks (Replicate, OpenAI), HMAC, fal's ED25519, or your own function. Status checks stay on as a safety net (`MEDIA_WEBHOOK_POLL_EVERY`). System checks `llm_kit.E003` and `llm_kit.W004` report settings that can't work.
- OCR (#48): `tp.ocr(document, pages=)` and `aocr` read a PDF or image through LiteLLM's `ocr` (Mistral, Azure Document Intelligence, Cohere...). The text comes back as Markdown, with each page's text, and calls are priced per page (`ocr_cost_per_page`, `annotation_cost_per_page`). The media guide shows how to add a kind of your own (music, say), from its backend to the catalogue entry.
- Checked against the providers' own documentation and LiteLLM's code:
  - **fal public keys:** webhook signatures are checked with keys from `https://rest.fal.ai/.well-known/jwks.json`, fal's published address.
  - **fal starting image:** fal video models get it as a URL, as fal requires. Bytes raise `BadRequest`.
  - **Which providers can send webhooks** is now documented: OpenAI through its dashboard. fal and Replicate need a backend of your own, because LiteLLM's video calls don't send fal's `fal_webhook` query parameter and LiteLLM has no Replicate video.
- `touchpoint(..., model_params={ref or pattern: params})`: parameters for one model only, over `params`, for chat, embeddings and media.
- The catalogue keeps media models' prices by unit (`price_units`: per image, second, character, pixel, resolution...).

- Tool calls have a time limit, so a tool that hangs no longer holds an agent until the request gives up.
  - **The limit:** each call may take `TOOL_TIMEOUT` seconds (90), or per tool `tools.register(..., timeout=)` / `metadata={"llm_kit_timeout": s}`.
  - **Past it:** the call ends as an error result (`status="error"`) that tells the model the tool didn't finish and its result is unknown, and the run goes on. LangChain's callbacks record the call as failed.
  - **How a call ends:** async tools are cancelled. Sync tools run on a pool of `TOOL_THREADS` threads and are abandoned at the limit; their database connections are checked before and after each call. Inside a database transaction a sync tool still runs where it was called.
- The request's deadline leaves time to answer. `AGENT_ANSWER_RESERVE` (25 s) of `AGENT_TIMEOUT` is kept back.
  - **Tool calls:** each one gets at most the time left less the reserve.
  - **Near the end:** the model is offered no tools and is told to answer now with what it has, saying what it couldn't check. Tools can read `agents.time_left(config)`.
- Background runs keep their events, so a watcher can start from any point. Every event has `seq` (1, 2, 3, … with no gaps: the SSE `id:`, `"seq"` over WebSocket), and `watch(run, after=n)`, `?after=n`, a `Last-Event-ID` header or WebSocket `"after"` replay what came after `n`, then go on live. The starter, a late watcher and a reconnecting client miss nothing, and a watcher whose queue overflowed fills the gap from the log. If events asked for are gone, a `snapshot` with `"resync": true` comes first.
  - The log is kept in the Django cache when it is shared, otherwise in memory (`RUN_EVENT_LOG`, `RUN_EVENT_LOG_CACHE`). It is capped at `RUN_EVENT_LOG_SIZE` (2,000) events per run and dropped after `RUN_EVENT_LOG_TTL` (300 s). Tokens are written in batches every 100 ms.
  - The new system check `llm_kit.W003` warns about a worker executor with a log in memory.

- `background.publish(target, event)` / `apublish`: send your own events (named with a dot, like `myapp.outcome`) to a background run's watchers, before its final `status`. `target` is a hook's `RunSetup`, a run's config, a `BackgroundRun` or an id; `background.run_id_of(target)` gives the id. Outside a background run nothing is sent.

- `background.active_run(thread)` is now exported from `llm_kit.background`: the thread's queued or running background run, or `None`.

- `touchpoint(..., adapt_input=fn)` and `model(adapt_input=)`: `fn(input, ref, info)` is called for each model just before it is tried, fallbacks included, so each gets input it can read (audio or a transcript; a file reference only the provider that holds it can open). It works for invoke, ainvoke, stream and astream, after `bind_tools` and `with_structured_output`, and before the rate-limit slot. It may be async. A model it raises for is skipped.

- Per-model rate limits: requests per minute, tokens per minute and requests per day for each model, from `RATE_LIMITS` or `RateLimit` rows (patterns like `gemini/*`; set by people, never synced). Counts are shared across processes in Django's cache: with Redis, one atomic script with a sliding window. A slot is taken before every chat and embedding call. At a limit the call waits up to the retry policy's `max_wait`, then moves to the next fallback, then raises `RateLimited` (classified `rate_limit` or `quota_exhausted`). Tokens are counted after each call, and refused requests take no slot. Calls on brought keys aren't counted. `GET llm/api/monitoring/rate-limits/` shows use against limits (migration 0020).

- Run hooks: `RUN_HOOKS` and `GRAPHS[name]["prepare"]` name functions called before every run, in the process that runs it: SSE, REST, WebSocket, background runs (in the worker, from the saved payload and owner) and `agents.run`/`arun`/`stream`. A hook sets server-only state (merged over the client's input), `configurable` keys and callbacks, and returns context managers entered around the whole run. `RunRefused` refuses a run (403); any other error fails it cleanly before it starts. Server-set `context=` (on `background.start`, `agents.run`/`arun`/`stream`/`start` and `runtime.runs.prepare`) reaches hooks as `run.context` and tools through `context_of(config)`. It is never set from a request, and is saved with background runs (migration 0020).

- Provider readiness: `monitoring.readiness()` and `GET llm/api/monitoring/readiness/` list every provider your touchpoints are configured with (defaults and fallbacks, after overrides), including ones that have never had a call. Each says whether credentials are present, where they come from (`CREDENTIALS`, database, environment, or none needed) and how stored keys did in their last test, without returning secrets. Touchpoints get `ready` and `failover_ready` (#33).

- Background subagents: `agent(..., background_subagents=True | [names])` gives `delegate` a `background` argument. The subagent starts as a `BackgroundRun` linked to the parent (`parent_thread`, `parent_run_id`, recorded under the parent's run) with the request's remaining hand-offs, depth and budget. The parent gets a task id at once. When the task finishes, its answer is queued on the parent's thread as a message, so the parent agent tells the user. `check_task` and `cancel_task` tools follow or stop the thread's tasks (migration 0018).

- `llm_kit.background.run_finished` signal (`run`, `status`, `result`, `error`): sent once per background run when it is done, interrupted, errors, is cancelled (also before it started) or is lost. It comes from whichever process notices: the worker, the canceller, or whoever marks it lost. Hook messages and push notifications to it without keeping watchers open.

- Queued follow-ups: a background run started on a busy thread with `"queue": true` (`background.start(..., queue=True)`, `agents.start(..., queue=True)`, or `RUN_QUEUE`) waits as `waiting` instead of getting 409, and starts when the runs before it finish, in order. Waiting runs are listed in `llm/api/runs/` and in the watch snapshot's `queue`, and can be cancelled. Lost and cancelled runs let the queue move on. An unanswered interrupt holds it, and a `resume` never waits (migration 0017).

- Background runs: start a run detached from the request (`"background": true` on SSE, REST and WebSocket runs; `agents.start()`, `background.start()`), so it keeps going when the client leaves. Anyone who may use the thread can watch it (`GET llm/runs/<id>/watch/`, WebSocket `watch`, `background.watch()`): a snapshot of where it is, then its events live. They can also cancel it from any process. `GET llm/api/runs/` lists them. Runs execute where `RUN_EXECUTOR` says: a thread (default), Celery, Django-Q2 or your own function. Their events cross processes through the Channels layer (`RUN_EVENTS`). New `BackgroundRun` table (migration 0016). A thread runs one at a time: a second run while one is going gets 409.

- Budgets across processes: with Redis or Memcached as Django's cache, priced calls are added to shared counters for the budgets they count towards, so every worker sees all spend at once instead of on the next 30-second read (`SPEND_COUNTER`, default `"auto"`).

- Managing other scopes: platform admins pass `scope_id` to the settings, keys and budgets APIs and template tags to manage any tenant or user, and may change every scope. Tenant managers can set one member's budget (`member=`), which replaces the all-members budget for that member. Membership is checked with the new `TENANT_MEMBERS` hook, which also lists members for pickers.

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

- The catalogue labels media models with what they do: `image`, `image_edit`, `video`, `speech`, `transcription`, `ocr`, `realtime`, `music`, `rerank`, `moderation`. These replace the broad `audio` and `image`, and models are re-labelled on the next catalogue sync. `requires={"kind": "audio"}` matches nothing after that: use `speech` or `transcription`. LiteLLM decides a model's kind over the other sources.
- Tool errors shown to the model: a `ToolException`'s message (how a tool reports a failure) and invalid arguments are shown as before. For any other exception only its class name is shown, unless `EXPOSE_ERRORS`; the text used to include the exception's message.
- Sync tools in an agent run now run on llm-kit's tool threads (outside a database transaction), so they can be stopped at their time limit. ContextVars carry over; `threading.local` values don't.
- Starting a background run over SSE or WebSocket streams it from its first event (`thread`, `run`, then the run's events with `seq`), instead of `run` then a `snapshot`. Watching a run (`watch/`, WebSocket `watch` without `after`) still begins with a `snapshot`.
- A touchpoint's model is no longer changed by llm-kit. The touchpoint's metadata, tags and recording handler used to be set on the object the backend factory returned. A factory that returns one shared model (and llm-kit's own cached LiteLLM routers) gained another handler on every build, so calls were recorded more and more times. `model()` now returns the factory's model wrapped in a `TouchpointModel` (a `RunnableBinding`: the model is `.bound`, its attributes are forwarded, and `bind_tools`/`with_structured_output` keep the config). Code that checked `isinstance(tp.model(), SomeChatModel)` should check `.bound`.
- `defer_tools` defaults to `None` (defer when there are more than `DEFER_TOOLS_ABOVE` tools) instead of `False`. Pass `defer_tools=False` to keep every tool bound.

### Fixed

- Embeddings always come back one vector per input.
  - **The bug:** on Vertex AI, Gemini embedding models other than `gemini-embedding-001` embed a list as one input. Through LiteLLM, `embed([t1, …, tN])` silently returned a single vector, and a caller pairing vectors with passages stored the wrong ones.
  - **Splitting:** lists are now split into calls of at most the model's batch size, in order: `EMBEDDING_BATCH_SIZE`, a backend's `max_batch`, LiteLLM's model info, then llm-kit's list (one input for those Vertex models). Each call takes its own rate-limit slot and is recorded, and `aembed()` runs up to `EMBEDDING_CONCURRENCY` at once.
  - **A wrong count is an error:** a reply with a different number of vectors raises `EmbeddingCountMismatch`, naming the model, and the next fallback embeds the whole list again.
- The Postgres checkpointer no longer races to create its tables. LangGraph's `setup()` reads the highest applied migration, then inserts the missing ones. On a fresh database, or after an upgrade that adds a migration, two event loops or processes starting together inserted the same version, and a run failed with `UniqueViolation` on `checkpoint_migrations_pkey`. Setup now runs once per process and database, under a Postgres advisory lock.
- Minimum versions that actually work: `langgraph-checkpoint>=4.3.0` (`DjangoSaver` needs `writes_sort_key`; with 4.1 Django failed at start-up), `langchain-core>=1.2.5` (what `langgraph-checkpoint` 4.3 needs, though it declares less) and `langgraph>=1.1.5` (earlier 1.x break with the `langgraph-prebuilt` they allow). CI now also runs on the lowest allowed versions (#35).
- The SSE views read each event in a new task, so a ContextVar set during a run didn't reliably reach its later steps. They now read the whole run in one task.
- Budget totals started at $0 recorded spend for up to 30 s on a freshly booted host (monotonic time under the refresh interval made a never-read total look fresh).
- A tenant member could read another member's own budget and spend (`GET llm/api/budgets/?scope=member&member=<id>`). Seeing someone else's member budget now needs the same permission as setting it.
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
