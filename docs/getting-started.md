---
description: Install django-llm-kit, register a graph and a touchpoint, add the URLs, and stream your first reply.
---

# Getting started

## Install

```
pip install "django-llm-kit[all]"
```

`[all]` installs the optional integrations. Pick extras instead if you prefer:

| Extra | Adds | Needed for |
|---|---|---|
| `drf` | Django REST Framework | the REST API, monitoring and tools endpoints |
| `channels` | Django Channels | the WebSocket consumer |
| `litellm` | `langchain-litellm` | the default model backend (every provider through one dependency) |
| `crypto` | `cryptography` | provider keys stored encrypted in the database |
| `otel` | OpenTelemetry API | the `otel` record sink |
| `postgres` | `langgraph-checkpoint-postgres`, `psycopg` | `CHECKPOINTER = "postgres"` |

Without `litellm`, touchpoints use LangChain's `init_chat_model`, which needs the provider's own `langchain-*` package (for example `langchain-anthropic`).

## Configure

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

## Declare a touchpoint

A touchpoint is one place your code calls a model. Put touchpoints in `<app>/touchpoints.py`; that module is imported at startup.

```python
# myapp/touchpoints.py
from llm_kit import ClientChoice, touchpoint

chat = touchpoint(
    "myapp.chat",
    default="anthropic/claude-sonnet-5-5",
    fallbacks=["gemini/gemini-3.8-flash"],
    requires={"tool_call": True},
    choice=ClientChoice(allow=("anthropic/*", "gemini/*"), max_price_input=5),
)
```

Set the provider keys as environment variables (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …). See [Touchpoints](guides/touchpoints.md) for other key sources.

## Write a graph

```python
# myapp/graphs.py
from langgraph.graph import START, END, MessagesState, StateGraph
from myapp.touchpoints import chat

def call_model(state: MessagesState, config):
    llm = chat.model(config=config)   # the client's pick if allowed, else the default; with fallbacks
    return {"messages": [llm.invoke(state["messages"], config)]}

builder = StateGraph(MessagesState)   # uncompiled: llm-kit compiles it with the checkpointer
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
```

You don't have to write graphs by hand: an [agent](guides/agents.md) declared with `agent(...)` is a graph too.

## Add the URLs

```python
# urls.py
urlpatterns = [
    path("llm/", include("llm_kit.transport.sse.urls")),      # SSE streaming (+ /llm/metrics/)
    path("llm/api/", include("llm_kit.transport.api.urls")),  # REST API (needs DRF)
]
```

```
python manage.py migrate
python manage.py llm_kit_sync_catalogue     # or --offline to load the bundled snapshot
```

## Stream a reply

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

The SSE views use Django sessions, so the request needs a session cookie and a CSRF token. See [Streaming and APIs](guides/streaming.md) for the payload, the events, the REST API and WebSockets.

## Check your setup

```
python manage.py check                   # includes llm-kit's system checks
python manage.py llm_kit_touchpoints     # what each touchpoint resolves to, and problems
python manage.py llm_kit_status          # catalogue age, open circuits, last 24 h
```

## Try the example project

The repository has a runnable demo: a streaming chat page with a model picker, an agent that hands questions to a read-only subagent, and a `Note` model exposed as agent tools.

```
git clone https://github.com/Teklora-solutions/django-llm-kit
cd django-llm-kit && uv venv && uv pip install -e ".[dev]"
cd example
python manage.py migrate
ANTHROPIC_API_KEY=… python manage.py runserver        # or LLM_KIT_DEMO_OFFLINE=1 for an echo model
```

Open <http://localhost:8000>. Offline, a message containing "research" shows the hand-off to the subagent.
