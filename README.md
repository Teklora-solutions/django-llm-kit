# django-llm-kit

Use LangGraph and LangChain in Django the Django way. Register a graph in `settings.py` and you get:

- **`DjangoSaver`**: a LangGraph checkpointer that stores state in your Django database (SQLite, Postgres, MySQL). Conversations survive restarts, show up in the admin, and are deleted when their `Thread` is deleted.
- **Streaming views**: async Server-Sent-Events endpoints that stream tokens, node updates and interrupts.
- **REST API** (DRF): threads, runs, state, history, touchpoints, model pickers, graphs.
- **WebSocket consumer** (Channels): two-way chat, human-in-the-loop resume, and cancel.
- **Touchpoints**: each place your code calls an LLM declares its model, fallbacks and **who may choose the model** (server or client, within limits) in code. Admins can override this without a deploy.
- **Observability**: every model call, tool call and run is recorded: tokens, cost (from the catalogue), latency and time to first token, which model actually answered after fallbacks, and classified errors. Stored in your database and logged as JSON; OpenTelemetry is optional.
- **Monitoring**: spend, errors, latency and provider health as plain Python functions for your own dashboards, the same data as JSON endpoints, an admin dashboard, a per-provider circuit breaker that steers fallbacks, alert rules (email/webhook/signal), `llm_kit_status`, and a Prometheus endpoint.
- **Model catalogue**: prices, limits and capabilities for ~12,600 models, synced from models.dev, OpenRouter and LiteLLM. Touchpoints are checked against it.

```
pip install "django-llm-kit[all]"   # drf + channels + litellm + crypto; or pick extras
```

## Quick start

```python
# settings.py
INSTALLED_APPS = [
    "daphne",            # optional: makes `runserver` ASGI so SSE/WebSockets stream
    ...,
    "rest_framework",    # optional
    "channels",          # optional
    "llm_kit",
]

LLM_KIT = {
    # "touchpoint": the policy a client's "model" choice is checked against
    "GRAPHS": {"assistant": {"graph": "myapp.graphs:builder", "touchpoint": "myapp.chat"}},
}
```

```python
# myapp/touchpoints.py  (imported automatically at startup)
from llm_kit import ClientChoice, touchpoint

chat = touchpoint(
    "myapp.chat",
    default="anthropic/claude-sonnet-5-5",
    fallbacks=["gemini/gemini-3.8-flash"],
    requires={"tool_call": True},
    choice=ClientChoice(allow=("anthropic/*", "gemini/*"), max_price_input=5),
)
```

```python
# myapp/graphs.py
from langgraph.graph import START, END, MessagesState, StateGraph
from myapp.touchpoints import chat

def call_model(state: MessagesState, config):
    llm = chat.model(config=config)          # the client's pick if allowed, else the default; with fallbacks
    return {"messages": [llm.invoke(state["messages"], config)]}

builder = StateGraph(MessagesState)          # uncompiled: llm-kit compiles it with DjangoSaver
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
```

```python
# urls.py
urlpatterns = [
    path("llm/", include("llm_kit.transport.sse.urls")),          # SSE streaming
    path("llm/api/", include("llm_kit.transport.api.urls")),  # REST API (needs DRF)
]
```

```
python manage.py migrate
```

Then stream a reply:

```
curl -N -X POST localhost:8000/llm/stream/ -H 'Content-Type: application/json' \
     -b cookies.txt -H "X-CSRFToken: $CSRF" -d '{"message": "Hello!", "model": "gemini/gemini-3.8-flash"}'

event: thread
data: {"id": "6f1c…", "graph": "assistant"}

event: token
data: {"node": "model", "message": {"id": "run-…", "type": "AIMessageChunk", "content": "Hi"}}
…
event: end
data: {"next": [], "interrupts": []}
```

## Registering graphs

Each value in `LLM_KIT["GRAPHS"]` is a dotted path (`pkg.mod:attr` or `pkg.mod.attr`) to one of:

| Target | What llm-kit does |
|---|---|
| uncompiled `StateGraph` | `builder.compile(checkpointer=DjangoSaver())` |
| compiled graph | attaches `DjangoSaver` unless it already has a checkpointer |
| `fn(checkpointer)` factory | calls it; use this for `create_agent(..., checkpointer=checkpointer)` or custom compile options |

Or a dict `{"graph": <path>, "touchpoint": "<name>"}` to accept a client's model choice for that graph. Without a touchpoint, a request with `"model"` is rejected (400). Graph modules are imported at startup so touchpoints declared in them are registered.

In your own code: `llm_kit.get_graph("assistant")` returns the compiled graph (cached per process). `DEFAULT_GRAPH` names the default; if only one graph is registered, that one is the default.

## Touchpoints: who chooses the model

A **touchpoint** is one place your code calls an LLM. You declare it once; everything about which model runs there follows from the declaration.

```python
from llm_kit import touchpoint, ClientChoice, ServerChoice

summarise = touchpoint(
    "crm.summarise_lead",
    default="anthropic/claude-haiku-5-5",     # LiteLLM-style provider/model_id
    fallbacks=["gemini/gemini-3.8-flash"],    # tried in order if the call fails
    params={"temperature": 0},
    requires={"structured_output": True},     # checked against the catalogue
    choice=ServerChoice(),                    # the default: clients can't pick
)

lead = summarise.complete(text, schema=Lead)  # one-shot structured call, no graph needed
llm = summarise.model()                       # a LangChain chat model (bind_tools etc. keep fallbacks)
summarise.resolve()                           # which model(s) and why, without calling anything
```

**Choice policy, declared in code:**

| | Client sends `"model"` | Then |
|---|---|---|
| `ServerChoice()` | ignored, with the reason recorded | `on_requested="reject"` returns 400 instead |
| `ClientChoice(allow=…, requires=…, max_price_input=…, max_price_output=…)` | accepted if it matches the patterns, has the capabilities and is under the price caps (checked in the catalogue) | `on_disallowed="ignore"` uses the default instead of returning 400; `use_fallbacks=False` drops fallbacks behind a client's pick |

**Resolution order:** a client pick the policy allows → an admin override (*Touchpoint overrides* in the admin: default, fallbacks, params, a kill switch, and an `allow` list that can **narrow but never widen** what the code allows) → the code default. Fallbacks without an API key, or failing `requires`, are skipped. If the default has no key, the first usable fallback is promoted. Every result records who chose and why.

`python manage.py llm_kit_touchpoints [--check]` lists every touchpoint, what it resolves to, and problems: models missing from the catalogue, deprecated, unpriced, or without a key.

**Backends** (`LLM_KIT["MODEL_BACKEND"]`, or `touchpoint(..., backend=…)`):
- `"litellm"`: the default when installed (`[litellm]` extra). One dependency covers every provider. With several keys for a provider, calls are spread across them through a LiteLLM Router.
- `"langchain"`: `init_chat_model` with the provider's `langchain-*` package.
- A dotted path to your own `fn(ref, params, api_keys) -> BaseChatModel`.

**API keys:** env vars first (`ANTHROPIC_API_KEY`, the names models.dev lists, or a comma-separated pool in `ANTHROPIC_API_KEYS`), plus optional **encrypted keys in the DB** (*Provider keys* in the admin; needs `[crypto]` and `LLM_KIT["ENCRYPTION_KEYS"]`, a list for rotation). Keys are never shown again after saving.

## Observability

Every call through a touchpoint, and every graph run started through llm-kit, is recorded. Nothing to add to your code.

| Record | What it holds |
|---|---|
| **LLM call** (one per attempt) | touchpoint, graph/node, thread, user, tenant; requested → resolved → used → served model; who chose it; fallback index; input/output/cached/reasoning tokens (estimated and flagged when a provider sends none); cost; latency and time to first token; status and error class |
| **LLM run** (graph run or `complete()`) | totals: calls, tool calls, tokens, cost, unpriced calls, duration; status (`ok`, `error`, `interrupted`, `cancelled`) |
| **Tool call** | name, duration, status, input/output hashes |

- **Cost** comes from the catalogue, including cached-token and long-context pricing. A model without a known price is flagged `unpriced`; it is never counted as $0.
- **A failed primary and the fallback that answered are two rows**, so you can see how often each provider fails.
- **Errors** are classified the same way for every provider: `rate_limit` vs `quota_exhausted` (they need different handling), `timeout`, `auth`, `overloaded`, `context_length`, `content_filter`, `bad_request`, `connection`. Use `llm_kit.observability.classify_error(exc)` yourself too.
- **Content is not stored** by default, only hashes. Opt in per touchpoint with `touchpoint(..., capture=True)` or globally with `CAPTURE_CONTENT`.
- **Context** outside requests: `with llm_kit.observability.context(user=…, tenant=…, tags={…}):`.
- **Never slows a call down:** callbacks only enqueue records; a background thread prices and writes them. A failing sink never breaks a call. `llm_kit.observability.flush()` waits for pending writes (tests, scripts).
- **Sinks** (`RECORD_SINKS`): `"database"` (admin: *LLM calls*, *LLM runs*, *Tool calls*), `"log"` (one JSON line per record on the `llm_kit.calls` logger, never including content), `"otel"` (OpenTelemetry spans with GenAI semantic conventions), or a dotted path to your own class with `write(calls, runs, tools)`.

## Monitoring

Build your own dashboard on the same functions the admin uses. Each returns plain data and takes the same filters: `period` ("24h", "7d"…) or `since`/`until`, plus `touchpoint`, `model`, `provider`, `user`, `tenant`, `graph`, `status`, `chosen_by`.

```python
from llm_kit import monitoring

monitoring.summary(period="24h")                       # calls, errors, error/fallback rate, tokens, cost, p50/p95/p99 latency, TTFT
monitoring.timeseries("cost", bucket="day", period="30d", group_by="model")
monitoring.breakdown("touchpoint", period="7d")        # per key: calls, error rate, cost, p95
monitoring.errors(period="24h")                        # by class and model, latest messages
monitoring.top("user", metric="cost", n=10)
monitoring.provider_health()                           # last 15 min per provider + circuit state
```

The same data as JSON (staff only unless `MONITORING_PERMISSION` names a DRF permission class): `GET /llm/api/monitoring/{summary,timeseries,breakdown,errors,top,providers,alerts}/?period=7d&touchpoint=…`.

**Admin dashboard** (*LLM Kit → Monitoring dashboard*): headline numbers, spend per day, breakdowns by touchpoint and model, provider health, errors, alerts, and catalogue changes to models your touchpoints use. Light and dark theme.

**Circuit breaker:** each provider's circuit opens after repeated retryable errors (5 in 60 s by default), or at once on an exhausted quota (15 min). While it's open, touchpoint resolution moves that provider's models behind healthy fallbacks. After the cooldown, one call is let through to test it. State is shared through Django's cache, so use Redis/Memcached with several workers. Tune with `CIRCUIT_BREAKER`.

**Alerts** (*Alert rules* in the admin): `spend`, `error_rate`, `latency_p95`, `unpriced_share`, `provider_down`, `catalogue_change` (a touchpoint's model was removed, deprecated or changed price). Each rule takes a window, threshold, filters and cooldown, and delivers by email, a webhook (Slack/Discord-compatible `text`), and the `llm_kit.monitoring.alerts.alert_fired` signal. Run `manage.py llm_kit_check_alerts` every few minutes (Celery: `llm_kit.check_alerts`).

**Operations:**
- `manage.py llm_kit_status`: catalogue age, open circuits, touchpoint problems, last 24 h. Exits 1 when unhealthy, for cron, CI or health checks.
- `/llm/metrics/`: Prometheus gauges for the last hour plus circuit state and catalogue age. Needs a staff session or `Authorization: Bearer <METRICS_TOKEN>`.
- `manage.py llm_kit_prune [--dry-run] [--threads DAYS]`: deletes old calls, runs, tool calls, catalogue changes and alerts, and clears captured content after `CONTENT_RETENTION_DAYS`. Threads are only deleted when you pass `--threads`.

## Agent tools from your models

Declare what agents may see on a model, and llm-kit generates the tools: `find_<plural>` (search, filter, page, or look up by id) and, if the model allows writes, `save_<name>` (create, or update by id). There are no delete tools.

```python
class Invoice(models.Model):
    ...
    class AgentAccess:
        description = "invoices we sent to customers"
        fields = ["number", "customer", "total", "status", "issued_on"]   # nothing else is shown
        search_fields = ["number", "customer__name"]
        filter_fields = ["status", "issued_on", "customer"]              # dates get _from / _to
        tenant_field = "organisation"     # whose rows: owner_field, tenant_field, agent_queryset, or unscoped = "why"
        writes = "direct"                 # "none" | "direct" | "approval"
        writable_fields = ["total", "customer", "issued_on"]
        approval_fields = ["status"]      # proposed by the agent, applied when a person approves
```

For models you can't edit, put the same options in `<app>/agent_tools.py` (imported at startup):

```python
tools.expose(User, fields=["username", "first_name"], owner_field="pk", name="account", plural="accounts")
tools.register(my_langchain_tool, permission="crm.view_lead", category="crm")   # hand-written tools join the catalogue
```

**Using them:**

```python
from llm_kit import tools

llm.bind_tools(tools.for_user(request.user))       # what this user may use, bound to them
ToolNode(tools.all_tools())                        # in a graph built once: each call reads the run's caller
llm.bind_tools(tools.for_config(config))           # inside a node: the caller the server put in config
system_prompt += tools.guidance(bound_tools)       # one usage line per tool
```

- **The caller is set by the server.** Runs started through llm-kit's views carry `Caller(user, tenant)` in the config. The tenant comes from `TENANT_RESOLVER(user)`. Clients can't set it, it is never written to checkpoints, and a tool with no caller refuses to run.
- **Every query and write is limited to the caller's rows.**
  - On create, the owner and tenant are filled in by the server.
  - A writable link only accepts rows the caller can see.
  - A write that would move a row out of the caller's scope is rolled back.
- **Permissions:**
  - Django's view/add/change permissions by default; set `permission` / `write_permission` to override.
  - Checked when tools are listed or bound, and again on every call.
  - Anonymous users get no tools.
- **Approvals:**
  - `writes = "approval"`, or a change to any `approval_fields`, creates a `ToolProposal` instead of writing.
  - `tools.pending_for(user)`, `tools.approve(p, by=user)` and `tools.reject(p, by=user, reason=...)`, plus admin actions and the JSON API. Approving refuses if the record changed since it was proposed.
  - Signals: `proposal_created`, `proposal_decided`.
- **Refused at startup:**
  - Fields that look like secrets (password, token, api_key…), binary or vector columns.
  - Writable links to users or file fields.
  - Fields like `status`/`is_active` as direct writes.
  - Creates that would miss a required field.
  - A model with no declared scope.
  - Duplicate tool names.
  - `paused = "why"` keeps a declaration checked but hidden.
- **Catalogue:**
  - `GET /llm/api/tools/` lists the current user's tools with their JSON schemas.
  - `GET /llm/api/tools/proposals/` and `POST .../<id>/approve/` or `.../<id>/reject/`.
  - `manage.py llm_kit_tools [--user name]`.
  - Tool calls are recorded like everything else (*Tool calls* in the admin).

## Model catalogue

Metadata for ~12,600 models (prices, context limits, capabilities, modalities, release and deprecation dates), synced from **models.dev**, **OpenRouter** and **LiteLLM** into your database:

```
python manage.py llm_kit_sync_catalogue            # conditional requests; safe to run often
python manage.py llm_kit_sync_catalogue --offline  # bundled snapshot, no network
```

Schedule it every few hours with Celery Beat (`llm_kit.sync_catalogue`), Django-Q2 (`llm_kit.tasks.run_catalogue_sync`) or cron. It never runs during a request.

```python
from llm_kit import catalogue

m = catalogue.get("anthropic/claude-sonnet-5-5")   # also "anthropic:…" or a bare id
m.price_input, m.context_window, m.tool_call        # USD per 1M tokens, ints, bools
catalogue.search(requires={"tool_call": True, "input": ["image"]}, max_price_input=3, usable_only=True)
catalogue.has_credentials("gemini")                 # key env var names come from models.dev
```

- Models are keyed by LiteLLM provider names, so `m.ref` is a valid LiteLLM model string.
- **Merging:** each field comes from the best source for it. Prices prefer LiteLLM; capabilities prefer models.dev; a provider's own API (OpenRouter) wins for its own models. The admin shows which source supplied each field, and where sources disagree.
- **Overrides** (admin: *Catalogue overrides*): correct any field, hide a model, or add a private/self-hosted one. They apply immediately and survive syncs.
- **Change log** (admin: *Catalogue changes*): added, removed, restored, price changes, capability changes, deprecations. The first sync is a silent baseline.
- **Failures:** a source that fails keeps its previous data; nothing is marked removed because of a network error.

## Run payload and events

Same body for SSE, the REST `runs` action and WebSocket `run` messages:

| Key | Meaning |
|---|---|
| `message` | shorthand for `{"input": {"messages": [{"role": "user", "content": …}]}}` |
| `input` | raw graph input |
| `resume` | answer an `interrupt()`: sent as `Command(resume=value)` |
| `model` | e.g. `"anthropic/claude-sonnet-5-5"`; checked against the graph's touchpoint, and reaches nodes as `configurable.model` |
| `config` | `{"recursion_limit": …, "configurable": {…}}`; `thread_id` is always forced to the thread's own id |
| `graph`, `title` | when creating a new thread |

Events: `thread` (new thread created), `token` (AI message chunk), `message` (complete message), `update` (node output), `interrupt`, `end` (`next` + pending `interrupts`), `error`. WebSockets also send `busy` and `cancelled`. Error details are only included when `DEBUG=True`; the exception is always logged to the `llm_kit` logger.

## Endpoints

**SSE** (`llm_kit.transport.sse.urls`, async views, serve with ASGI):

- `POST stream/`: create a thread and stream the first run
- `POST threads/<uuid>/stream/`: stream a run on an existing thread

These are plain Django views, so they use session auth and need the `X-CSRFToken` header. Under WSGI they still work, but the response is buffered instead of streamed.

**REST** (`llm_kit.transport.api.urls`, DRF, uses your `DEFAULT_AUTHENTICATION_CLASSES`):

| | |
|---|---|
| `GET/POST threads/` | list / create (`{"graph": "assistant", "title": "…"}`) |
| `GET/PATCH/DELETE threads/<id>/` | delete also removes the checkpoints |
| `POST threads/<id>/runs/` | run to completion or the next interrupt, returns state |
| `GET threads/<id>/state/` | `values`, `next`, `interrupts`, `checkpoint_id` |
| `GET threads/<id>/history/?limit=20` | checkpoints, newest first |
| `GET touchpoints/` | touchpoints and who chooses their model |
| `GET touchpoints/<name>/models/` | models a client may pick there, with prices and limits (for a picker) |
| `GET graphs/` | registered graphs and their touchpoint |

**WebSocket** (`llm_kit.transport.ws.routing.websocket_urlpatterns`):

```python
# asgi.py
application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AllowedHostsOriginValidator(AuthMiddlewareStack(URLRouter(websocket_urlpatterns))),
})
```

- `ws/llm/threads/<uuid>/`: existing thread
- `ws/llm/threads/`: new thread, created on the first `run`

Send `{"type": "run", …payload}` or `{"type": "cancel"}`. Close codes: 4401 not logged in, 4403 not your thread, 4404 no such thread.

## Access control

With `REQUIRE_AUTH: True` (the default), every endpoint needs a logged-in user, and users only see threads they own. Ownerless threads (made in the admin or by scripts) are open only to staff. With `REQUIRE_AUTH: False`, anonymous users can create threads and use them by id (ids are random UUIDs), but listing never shows ownerless threads.

## Using the checkpointer directly

```python
from llm_kit import DjangoSaver

graph = builder.compile(checkpointer=DjangoSaver())          # or DjangoSaver(using="replica")
graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "abc"}})
```

It implements the sync and async LangGraph checkpointer API: `get_tuple`, `list` (with `filter`, `before`, `limit`), `put`, `put_writes`, `delete_thread`, `copy_thread`. That covers interrupts, time travel and forking. Channel values are stored once per version in `CheckpointBlob` and shared across checkpoints, so long chats don't duplicate their history at every step. `prune` is not implemented yet.

**SQLite:** LangGraph writes checkpoints from worker threads, so set `"OPTIONS": {"transaction_mode": "IMMEDIATE", "timeout": 20}` (Django 5.1+). The saver works without it, but this avoids "database is locked" under concurrent load.

## Settings

| Key | Default | |
|---|---|---|
| `GRAPHS` | `{}` | name → dotted path, or `{"graph": path, "touchpoint": name}` |
| `DEFAULT_GRAPH` | `None` | falls back to the only graph |
| `MODEL_BACKEND` | `"auto"` | `litellm`, `langchain`, or a dotted path to a factory |
| `LITELLM_ROUTER` | `{"routing_strategy": "simple-shuffle", "num_retries": 0}` | kwargs for the Router used with key pools |
| `ENCRYPTION_KEYS` | `None` | Fernet keys or passphrases for DB-stored provider keys, newest first |
| `CATALOGUE_SOURCES` | all | source names or dotted paths |
| `CATALOGUE_MAX_AGE_HOURS` | `24` | when the catalogue counts as stale |
| `EXPOSE_ERRORS` | `DEBUG` | include exception details in error events |
| `OBSERVABILITY` | `True` | record calls, runs and tool calls |
| `RECORD_SINKS` | `["database", "log"]` | `database`, `log`, `otel`, or dotted paths |
| `CAPTURE_CONTENT` | `False` | store prompts/responses (per touchpoint: `capture=`) |
| `CONTENT_RETENTION_DAYS` / `CALL_RETENTION_DAYS` | `7` / `90` | used by `llm_kit_prune` |
| `CIRCUIT_BREAKER` | `{}` | `{"failures": 5, "window": 60, "cooldown": 60, "quota_cooldown": 900}` |
| `MONITORING_PERMISSION` | staff only | dotted DRF permission class for `/llm/api/monitoring/` |
| `METRICS_TOKEN` | `None` | bearer token for `/llm/metrics/` |
| `TENANT_RESOLVER` | `None` | dotted path to `fn(user) -> tenant` for model tools (and call records) |
| `REQUIRE_AUTH` | `True` | |
| `STREAM_MODES` | `["messages", "updates"]` | LangGraph stream modes used for SSE/WebSocket runs |
| `SSE_KEEPALIVE` | `15` | seconds between keep-alive comments; `0` turns them off |

## Example project

```
cd example
python manage.py migrate
ANTHROPIC_API_KEY=… python manage.py runserver        # or LLM_KIT_DEMO_OFFLINE=1 for an echo model
```

Open http://localhost:8000 for a streaming chat page with a model picker and a tool-calling graph.

## Development

```
uv venv && uv pip install -e ".[dev]"
pytest                                                             # SQLite
LLM_KIT_TEST_PG=postgres://postgres:pw@localhost:5432/postgres pytest   # Postgres (needs psycopg)
```
