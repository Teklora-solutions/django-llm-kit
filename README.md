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
tools.register(site_search, permission=tools.PUBLIC, category="site")            # anyone: visitors, and runs with no user
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

A thread belongs to a **user**, an anonymous **visitor**, a **key your code chose**, or **nobody** (made by code).

- With `REQUIRE_AUTH: True` (the default), every endpoint needs a signed-in user, and users see only their own threads.
- With `REQUIRE_AUTH: False`, visitors may chat too. Their threads are tied to a random id kept in their session data, so other visitors can't use them even if they know the id. Over WebSocket this needs `SessionMiddlewareStack` and a session started by an earlier page load.
- When a visitor signs in, their threads move to their account (`ADOPT_SESSION_THREADS`, on by default). Django rotates the session key at login but keeps the session data, which is why the id lives in the data. Call `access.adopt_session_threads(request, user)` yourself if you sign users in some other way.
- Staff may open threads that have no user (visitor, keyed or code threads), to inspect them; never another user's.
- With `TENANT_RESOLVER`, new threads are stamped with the user's tenant, and a thread from another tenant is refused.

**Threads your code keys** (a WhatsApp number, a ticket, a Celery job). Clients can never pick or look up a key; only your code can:

```python
from llm_kit.runtime.threads import open_thread

thread = open_thread("support.assistant", key=f"whatsapp:{jid}", tenant=business.pk)  # get or create
result = agents.run("support.assistant", text, thread=thread)  # user and tenant come from the thread
```

Keys are unique per tenant. `aopen_thread` is the async form.

## Using the checkpointer directly

```python
from llm_kit import DjangoSaver

graph = builder.compile(checkpointer=DjangoSaver())          # or DjangoSaver(using="replica")
graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "abc"}})
```

It implements the sync and async LangGraph checkpointer API: `get_tuple`, `list` (with `filter`, `before`, `limit`), `put`, `put_writes`, `delete_thread`, `copy_thread`. That covers interrupts, time travel and forking. Channel values are stored once per version in `CheckpointBlob` and shared across checkpoints, so long chats don't duplicate their history at every step. `prune` is not implemented yet.

**SQLite:** LangGraph writes checkpoints from worker threads, so set `"OPTIONS": {"transaction_mode": "IMMEDIATE", "timeout": 20}` (Django 5.1+). The saver works without it, but this avoids "database is locked" under concurrent load.

**SQLite in tests:** Django's default SQLite test database is in memory, and another thread's write locks whole tables there, so tests that run graphs fail with "database table is locked". Give the test database a file: `DATABASES["default"]["TEST"] = {"NAME": BASE_DIR / "test.sqlite3"}`. The system check `llm_kit.W001` warns when it is in memory. Tests that run graphs also need `TransactionTestCase` (or `django_db(transaction=True)`), because the checkpoint thread can't see a test transaction's uncommitted rows. Recording has the same shape on SQLite: the recorder writes from a background thread, and its writes can collide with a test's open transaction ("database is locked", on either side). Set `RECORD_SINKS` to `[]` in test settings unless a test reads the recorded rows. Postgres has neither problem.

## Choosing the checkpointer

Every graph and agent uses one checkpointer, set by `LLM_KIT["CHECKPOINTER"]`:

| Value | Meaning |
|---|---|
| `None` (default) | `DjangoSaver`: llm-kit's tables, on any database Django supports |
| `"postgres"` | LangGraph's own `PostgresSaver`/`AsyncPostgresSaver`, on Django's Postgres database (`pip install django-llm-kit[postgres]`). Tables are created on first use |
| `"myapp.ai:saver"` | a saver instance, a saver class, or a zero-argument factory |
| `{"sync": "myapp.ai:saver", "async": "myapp.ai:asaver"}` | separate factories for sync and async code; the async one may be a coroutine |

A project that already has an `AsyncPostgresSaver` (as some projects already do) can point at its factory, and its existing threads keep working.

- **One async saver per event loop.** An async connection pool belongs to the loop that opened it. A Celery task that calls `asyncio.run` gets a new loop each time, and a shared saver would fail on the second task. llm-kit makes one async saver per running loop and closes the savers of loops that have finished.
- **Missing halves are filled in.** A sync-only saver (`PostgresSaver`) is run in a worker thread for async code. An async-only factory serves sync code from one background loop.
- **Shut down cleanly.** With pooled savers (`"postgres"`, or a sync/async dict), `get_checkpointer().close()` frees the connections.
- Deleting a `Thread` deletes its state from whichever checkpointer is configured. With `DjangoSaver` its subagents' threads go too; other savers have no way to find those by prefix, so they stay in that checkpointer.

## Agents and subagents

One harness runs every agent. An agent is a declaration: its model (a touchpoint), prompt and tools. A subagent is the same harness with a different declaration, the way Claude Code's subagents are Claude Code with another model, prompt and tool set.

```python
# myapp/agents.py  (imported automatically at startup)
from llm_kit import agent, tools

agent(
    "crm.assistant",
    description="Answers staff questions about customers and invoices.",
    touchpoint="crm.assistant",                       # model policy, client choice, fallbacks, overrides
    prompt="You help {user}. Today is {date}.",       # template ({date} {now} {user} {tenant} {agent}) or fn(ctx)
    tools=[current_time, tools.select(categories=["crm"])],
    subagents=["crm.researcher"],
    max_steps=12,
)

agent(
    "crm.researcher",
    description="Looks through invoices to answer one question; reports facts with ids. Changes nothing.",
    touchpoint="crm.cheap",                           # another model
    prompt="Answer the task from the records. Quote ids.",
    tools=tools.select(kinds=["find"]),               # read-only
    output=ResearchReport,                            # optional pydantic shape for the answer
)

@agent.code("crm.weekly_report", description="Builds the weekly report.", subagents=["crm.researcher"])
def weekly_report(task, ctx):                         # Python instead of a model loop
    facts = ctx.run("crm.researcher", "What was invoiced this week?")
    return facts.output
```

- **Every agent is a graph by its name.** `/llm/stream/` with `{"graph": "crm.assistant"}`, threads, the WebSocket and the model picker work without listing it in `GRAPHS`.
- **From code:** `llm_kit.agents.run("crm.researcher", "Which invoices are overdue?", user=request.user)` returns an `AgentResult` (`text`, `output`, `steps`, `stopped`, tokens). `arun` and `stream` too. Pass `thread=` or `thread_id=` to continue a conversation. Without either, the run is throwaway: it and its subagents keep their state in memory, never in the database, and it is freed when the run ends. The recorded calls and runs are kept. This also means a throwaway run works inside a test transaction (Django's `TestCase`).
- **The loop:** model → tools → model, until the model answers or reaches `max_steps`; on its last step it is told to answer. Each step reads the declaration and admin overrides again.
- **Tools** are always limited to what the caller may use. The tool node refuses any call outside the agent's own set, so a parent's tools never reach a subagent.
- **Handing work over:**
  - An agent with `subagents` gets one `delegate(agent, task)` tool.
  - The subagent runs as the same user, on its own checkpointed thread `<thread>/<agent>/<call id>`. It sees only the task, or the parent's conversation with `fork=True`.
  - Its answer (or `output` as JSON) is the tool result.
  - Its messages stream to the client with `agent` and `depth` set on `token`/`message` events.
- **Limits per request:**
  - `AGENT_MAX_DEPTH` (2) for how deep subagents nest.
  - `AGENT_MAX_DELEGATIONS` (8) hand-offs in total.
  - `AGENT_TIMEOUT` (300 s) before no new model call starts.
  - No agent can hand work to itself or to an agent above it.
- **History window:** `history=2000` sends each model call only the newest ~2000 tokens of the conversation, starting at a user message. The checkpoint still keeps everything.
- **Long tool results:** `tool_result_chars=12000` cuts each tool result to that many characters, with a note saying how much was left out, so one fetched page can't fill the context window.
- **Public tools:** a tool registered with `permission=tools.PUBLIC` is offered to anonymous visitors and to runs with no user (a Celery task). Use it only for tools that read public data. Tools generated from your models can't be public.
- **Loading tools on demand:** `defer_tools=True, essential=[...]` binds only the essential tools plus `find_tools`. Tools it finds are bound from the next step and stay loaded in the thread.
- **Admin overrides** (*Agent overrides*): append to or replace the prompt, lower `max_steps`, narrow the tools, or pause an agent. They narrow the declaration in code, never widen it. Change an agent's model through its touchpoint override.
- **Recording:** every call and run carries its agent. A subagent's run links to its parent (`parent_run_id`, `depth`). The request's run counts every call under it, so its cost includes its subagents. The admin shows the run tree, and monitoring filters and breaks down by `agent`.
- **Checked at startup:** unknown subagents, touchpoints or tool names, agents that delegate in a circle, and names that clash with `GRAPHS`. Run `manage.py llm_kit_agents [--user name]` to list agents and the tools a user gets.

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
| `AGENT_MAX_DEPTH` | `2` | how deep subagents may nest (`0`: no delegation) |
| `AGENT_MAX_DELEGATIONS` | `8` | hand-offs per request, across all its agents |
| `AGENT_TIMEOUT` | `300` | seconds after which a request's agents start no new model call |
| `REQUIRE_AUTH` | `True` | |
| `CREDENTIALS` | `None` | Dotted path to `fn(provider)` returning a secret, a list of secrets, or None; asked before stored keys and env vars (a settings object, a vault, a service-account path, OAuth tokens). |
| `CHECKPOINTER` | `None` | `DjangoSaver`; see [Choosing the checkpointer](#choosing-the-checkpointer). |
| `ADOPT_SESSION_THREADS` | `True` | Move a visitor's threads to their account when they sign in. |
| `STREAM_MODES` | `["messages", "updates", "custom"]` | LangGraph stream modes used for SSE/WebSocket runs (`custom` carries subagents' messages) |
| `SSE_KEEPALIVE` | `15` | seconds between keep-alive comments; `0` turns them off |

## Example project

```
cd example
python manage.py migrate
ANTHROPIC_API_KEY=… python manage.py runserver        # or LLM_KIT_DEMO_OFFLINE=1 for an echo model
```

Open http://localhost:8000 for a streaming chat page with a model picker. It runs the `chat.assistant` agent (`example/chat/agents.py`), which hands questions about your notes to a read-only `chat.researcher` on another model. Offline, a message containing "research" shows the hand-off. The hand-built graph in `chat/graphs.py` is still registered as `assistant`.

## Development

```
uv venv && uv pip install -e ".[dev]"
pytest                                                             # SQLite
LLM_KIT_TEST_PG=postgres://postgres:pw@localhost:5432/postgres pytest   # Postgres (needs psycopg)
```
