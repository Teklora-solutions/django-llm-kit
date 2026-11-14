# django-llm-kit

[![CI](https://github.com/Teklora-solutions/django-llm-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/Teklora-solutions/django-llm-kit/actions/workflows/ci.yml)
[![Docs](https://github.com/Teklora-solutions/django-llm-kit/actions/workflows/docs.yml/badge.svg)](https://teklora-solutions.github.io/django-llm-kit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Teklora-solutions/django-llm-kit/blob/master/LICENSE)

Use LangGraph and LangChain in Django the Django way. Declare your models, agents and tools in code; llm-kit gives you the database storage, the HTTP and WebSocket endpoints, the admin screens, and a record of every call and what it cost.

**Documentation: https://teklora-solutions.github.io/django-llm-kit/** · for AI assistants: [`llms.txt`](https://teklora-solutions.github.io/django-llm-kit/llms.txt), [`llms-full.txt`](https://teklora-solutions.github.io/django-llm-kit/llms-full.txt), and every page as Markdown.

- **[Touchpoints](https://teklora-solutions.github.io/django-llm-kit/guides/touchpoints/)**: each place your code calls a model declares its model, fallbacks, required capabilities and **who may choose the model** (server, or client within limits). Admins can narrow it without a deploy.
- **[Agents and subagents](https://teklora-solutions.github.io/django-llm-kit/guides/agents/)**: one harness runs every agent; a subagent is the same harness with another model, prompt and tools. Delegation, limits, history windows, admin overrides, and a recorded run tree.
- **[Tools from your models](https://teklora-solutions.github.io/django-llm-kit/guides/tools/)**: declare what agents may see on a Django model and get find/save tools, limited to the caller's rows, with Django permissions and approval before changes.
- **[Model catalogue](https://teklora-solutions.github.io/django-llm-kit/guides/catalogue/)**: prices, limits and capabilities for ~12,600 models, synced from models.dev, OpenRouter and LiteLLM.
- **[Observability](https://teklora-solutions.github.io/django-llm-kit/guides/observability/)**: every model call, tool call and run recorded: tokens, cost, latency, time to first token, which model answered after fallbacks, and classified errors. Database, JSON logs, OpenTelemetry.
- **[Monitoring](https://teklora-solutions.github.io/django-llm-kit/guides/monitoring/)**: query functions, JSON API, an admin dashboard, a per-provider circuit breaker, alerts (email/webhook/signal), `llm_kit_status`, Prometheus.
- **[Checkpointers](https://teklora-solutions.github.io/django-llm-kit/guides/checkpointer/)**: `DjangoSaver` keeps LangGraph state in your Django database (SQLite, Postgres, MySQL); or use LangGraph's `PostgresSaver` or your own.
- **[Streaming and APIs](https://teklora-solutions.github.io/django-llm-kit/guides/streaming/)**: async SSE views, a DRF API and a Channels WebSocket consumer, with one payload and one set of events.
- **[Access control](https://teklora-solutions.github.io/django-llm-kit/guides/access/)**: threads belong to a user, a visitor, a key your code chose, or nobody; tenants are isolated.

```
pip install "django-llm-kit[all]"   # drf + channels + litellm + crypto + otel; or pick extras
```

Requires Python 3.11+, Django 5.0+, LangGraph 1.0+ and LangChain 1.0+.

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

See [Getting started](https://teklora-solutions.github.io/django-llm-kit/getting-started/) for the rest, and the [guides](https://teklora-solutions.github.io/django-llm-kit/) for each feature.

## Example project

```
git clone https://github.com/Teklora-solutions/django-llm-kit && cd django-llm-kit
uv venv && uv pip install -e ".[dev]"
cd example && python manage.py migrate
ANTHROPIC_API_KEY=… python manage.py runserver        # or LLM_KIT_DEMO_OFFLINE=1 for an echo model
```

Open http://localhost:8000 for a streaming chat page with a model picker. It runs the `chat.assistant` agent (`example/chat/agents.py`), which hands questions about your notes to a read-only `chat.researcher` on another model. Offline, a message containing "research" shows the hand-off. Sign in (`python manage.py createsuperuser`) and open http://localhost:8000/settings/ for a user's AI settings page (model, temperature, their own API keys, a spending limit), built from llm-kit's [settings templates](https://teklora-solutions.github.io/django-llm-kit/guides/settings/).

## Status

Beta. The API may still change before 1.0; every breaking change is listed in the [changelog](https://github.com/Teklora-solutions/django-llm-kit/blob/master/CHANGELOG.md).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/Teklora-solutions/django-llm-kit/blob/master/CONTRIBUTING.md); report security issues privately as described in [SECURITY.md](https://github.com/Teklora-solutions/django-llm-kit/blob/master/SECURITY.md).

## License

[MIT](https://github.com/Teklora-solutions/django-llm-kit/blob/master/LICENSE)
