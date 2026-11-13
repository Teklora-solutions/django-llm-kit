---
description: The SSE views, REST API and WebSocket consumer share one run payload and one set of events; this page covers all three.
---

# Streaming and APIs

Three transports, one payload, one set of events. Pick what suits your frontend:

| Transport | Include | Auth |
|---|---|---|
| **SSE** (async Django views) | `path("llm/", include("llm_kit.transport.sse.urls"))` | Django session + `X-CSRFToken` |
| **REST** (DRF) | `path("llm/api/", include("llm_kit.transport.api.urls"))` | your `DEFAULT_AUTHENTICATION_CLASSES` |
| **WebSocket** (Channels) | `llm_kit.transport.ws.routing.websocket_urlpatterns` | Channels' `AuthMiddlewareStack` |

## Run payload

The same body for SSE, the REST `runs` action and WebSocket `run` messages:

| Key | Meaning |
|---|---|
| `message` | shorthand for `{"input": {"messages": [{"role": "user", "content": …}]}}` |
| `input` | raw graph input |
| `resume` | answer an `interrupt()`: sent as `Command(resume=value)` |
| `model` | e.g. `"anthropic/claude-sonnet-5-5"`; checked against the graph's [touchpoint](touchpoints.md), and reaches nodes as `configurable.model` |
| `config` | `{"recursion_limit": …, "configurable": {…}}`; `thread_id` is always forced to the thread's own id, and keys starting with `__` are dropped |
| `graph`, `title` | when creating a new thread |

## Events

| Event | Data |
|---|---|
| `thread` | a new thread was created: `id`, `graph` |
| `token` | an AI message chunk, with `node`; `agent` and `depth` when it comes from a subagent |
| `message` | a complete message |
| `update` | a node's output |
| `interrupt` | the graph is waiting for a `resume` |
| `end` | `next` and pending `interrupts` |
| `error` | the run failed; details only when `EXPOSE_ERRORS` (defaults to `DEBUG`) |
| `busy`, `cancelled` | WebSocket only |

The exception behind an `error` event is always logged to the `llm_kit` logger.

## SSE

- `POST stream/`: create a thread and stream the first run.
- `POST threads/<uuid>/stream/`: stream a run on an existing thread.

These are plain async Django views, so they use session auth and need the `X-CSRFToken` header. A keep-alive comment is sent every `SSE_KEEPALIVE` seconds (15) of silence, and `X-Accel-Buffering: no` stops nginx from buffering the stream. Under WSGI they still work, but the response is buffered instead of streamed.

Status codes: 401 not signed in, 403 not your thread, 404 no such thread, 400 bad JSON, unknown graph or a model the touchpoint refuses.

```js
const res = await fetch("/llm/stream/", {
  method: "POST",
  headers: {"Content-Type": "application/json", "X-CSRFToken": csrftoken},
  body: JSON.stringify({message: "Hello!"}),
});
const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
// parse "event: …\ndata: …\n\n" blocks as they arrive
```

## REST

| Route | Action |
|---|---|
| `GET/POST threads/` | list your threads / create one (`{"graph": "assistant", "title": "…"}`) |
| `GET/PATCH/DELETE threads/<id>/` | delete also removes the graph state |
| `POST threads/<id>/runs/` | run to completion or the next interrupt; returns the state |
| `GET threads/<id>/state/` | `values`, `next`, `interrupts`, `checkpoint_id` |
| `GET threads/<id>/history/?limit=20` | checkpoints, newest first |
| `GET touchpoints/` | touchpoints and who chooses their model |
| `GET touchpoints/<name>/models/` | models a client may pick there, with prices and limits (for a picker) |
| `GET graphs/` | registered graphs and agents, and their touchpoint |
| `GET tools/`, `tools/proposals/` | see [Model tools](tools.md) |
| `GET monitoring/…` | see [Monitoring](monitoring.md) |

## WebSocket

```python
# asgi.py
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from llm_kit.transport.ws.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AllowedHostsOriginValidator(AuthMiddlewareStack(URLRouter(websocket_urlpatterns))),
})
```

- `ws/llm/threads/<uuid>/`: an existing thread.
- `ws/llm/threads/`: a new thread, created on the first `run`.

Send `{"type": "run", …payload}` or `{"type": "cancel"}`. One run per connection at a time; a second gets `busy`. Disconnecting cancels the run.

Close codes: 4401 not signed in, 4403 not your thread, 4404 no such thread.

## Human in the loop

When a node calls LangGraph's `interrupt(value)`, the run ends with an `interrupt` event and an `end` event whose `interrupts` list is non-empty. Send the answer as a new run on the same thread with `{"resume": …}`.
