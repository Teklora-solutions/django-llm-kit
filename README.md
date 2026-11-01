# django-llm-kit

Use LangGraph and LangChain in Django the Django way. Register a graph in `settings.py` and you get:

- **`DjangoSaver`**: a LangGraph checkpointer that stores state in your Django database (SQLite, Postgres, MySQL). Conversations survive restarts, show up in the admin, and are deleted when their `Thread` is deleted.
- **Streaming views**: async Server-Sent-Events endpoints that stream tokens, node updates and interrupts.
- **REST API** (DRF): threads, runs, state, history, models, graphs.
- **WebSocket consumer** (Channels): two-way chat, human-in-the-loop resume, and cancel.
- **Model catalogue**: named chat models you can edit in the admin, resolved with LangChain's `init_chat_model`, so any provider works and each request can pick its model.

```
pip install "django-llm-kit[all]"        # or [drf] / [channels] / no extras
pip install langchain-anthropic          # plus the provider packages you use
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
    "GRAPHS": {"assistant": "myapp.graphs:builder"},
    "DEFAULT_MODEL": "anthropic:claude-sonnet-5-5",
    "MODELS": {"fast": {"model": "anthropic:claude-haiku-5-5", "temperature": 0}},
}
```

```python
# myapp/graphs.py
from langgraph.graph import START, END, MessagesState, StateGraph
from llm_kit import model_from_config

def call_model(state: MessagesState, config):
    llm = model_from_config(config)          # the model the request asked for, or the default
    return {"messages": [llm.invoke(state["messages"], config)]}

builder = StateGraph(MessagesState)          # uncompiled: llm-kit compiles it with DjangoSaver
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
```

```python
# urls.py
urlpatterns = [
    path("llm/", include("llm_kit.urls")),          # SSE streaming
    path("llm/api/", include("llm_kit.api.urls")),  # REST API (needs DRF)
]
```

```
python manage.py migrate
```

Then stream a reply:

```
curl -N -X POST localhost:8000/llm/stream/ -H 'Content-Type: application/json' \
     -b cookies.txt -H "X-CSRFToken: $CSRF" -d '{"message": "Hello!", "model": "fast"}'

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

In your own code: `llm_kit.get_graph("assistant")` returns the compiled graph (cached per process). `DEFAULT_GRAPH` names the default; if only one graph is registered, that one is the default.

## Model catalogue

`get_chat_model(name)` (and `model_from_config(config)` inside nodes) looks up `name` in this order:

1. an active **`LLMModel`** row (admin: *LLM Kit → LLM models*): `model`, optional `provider`, `params` JSON (temperature, max_tokens, …), `api_key_env`
2. an alias in `LLM_KIT["MODELS"]`
3. the name itself as an `init_chat_model` id, e.g. `"openai:gpt-5"`

With no name it uses the row marked **default**, then `LLM_KIT["DEFAULT_MODEL"]`. API keys are never stored in the database. Provider packages read their usual env vars (`ANTHROPIC_API_KEY`, …), or set `api_key_env` on a row to name a different variable.

Clients pick a model per run with `"model": "fast"`; it reaches the graph as `config["configurable"]["model"]`. Unknown names are rejected with HTTP 400 before the run starts. `GET /llm/api/models/` lists the choices.

## Run payload and events

Same body for SSE, the REST `runs` action and WebSocket `run` messages:

| Key | Meaning |
|---|---|
| `message` | shorthand for `{"input": {"messages": [{"role": "user", "content": …}]}}` |
| `input` | raw graph input |
| `resume` | answer an `interrupt()`: sent as `Command(resume=value)` |
| `model` | catalogue name |
| `config` | `{"recursion_limit": …, "configurable": {…}}`; `thread_id` is always forced to the thread's own id |
| `graph`, `title` | when creating a new thread |

Events: `thread` (new thread created), `token` (AI message chunk), `message` (complete message), `update` (node output), `interrupt`, `end` (`next` + pending `interrupts`), `error`. WebSockets also send `busy` and `cancelled`. Error details are only included when `DEBUG=True`; the exception is always logged to the `llm_kit` logger.

## Endpoints

**SSE** (`llm_kit.urls`, async views, serve with ASGI):

- `POST stream/`: create a thread and stream the first run
- `POST threads/<uuid>/stream/`: stream a run on an existing thread

These are plain Django views, so they use session auth and need the `X-CSRFToken` header. Under WSGI they still work, but the response is buffered instead of streamed.

**REST** (`llm_kit.api.urls`, DRF, uses your `DEFAULT_AUTHENTICATION_CLASSES`):

| | |
|---|---|
| `GET/POST threads/` | list / create (`{"graph": "assistant", "title": "…"}`) |
| `GET/PATCH/DELETE threads/<id>/` | delete also removes the checkpoints |
| `POST threads/<id>/runs/` | run to completion or the next interrupt, returns state |
| `GET threads/<id>/state/` | `values`, `next`, `interrupts`, `checkpoint_id` |
| `GET threads/<id>/history/?limit=20` | checkpoints, newest first |
| `GET models/`, `GET graphs/` | catalogue and registered graphs |

**WebSocket** (`llm_kit.routing.websocket_urlpatterns`):

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
| `GRAPHS` | `{}` | name → dotted path |
| `DEFAULT_GRAPH` | `None` | falls back to the only graph |
| `DEFAULT_MODEL` | `None` | `init_chat_model` id or alias |
| `MODELS` | `{}` | alias → `"provider:model"` or `{"model": …, "provider": …, "api_key_env": …, **params}` |
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
