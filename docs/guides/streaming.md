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
| `background` | `true` starts the run [in the background](#background-runs) |
| `queue` | `true`: a background run that [waits its turn](#queued-follow-ups) on a busy thread instead of being refused |

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
| `run`, `snapshot`, `status` | [background runs](#background-runs): started, where it is now, each change of status |

The exception behind an `error` event is always logged to the `llm_kit` logger.

## SSE

- `POST stream/`: create a thread and stream the first run.
- `POST threads/<uuid>/stream/`: stream a run on an existing thread.
- `GET runs/<uuid>/watch/`: watch a [background run](#background-runs).

These are plain async Django views, so they use session auth and need the `X-CSRFToken` header. A keep-alive comment is sent every `SSE_KEEPALIVE` seconds (15) of silence, and `X-Accel-Buffering: no` stops nginx from buffering the stream. Under WSGI they still work, but the response is buffered instead of streamed.

Status codes: 401 not signed in, 403 not your thread, 404 no such thread, 400 bad JSON, unknown graph or a model the touchpoint refuses, 409 the thread has a background run in progress, 429 over a daily budget.

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
| `POST threads/<id>/runs/` | run to completion or the next interrupt; returns the state. With `"background": true`: 202 and the run |
| `GET runs/?status=&thread=` | your [background runs](#background-runs), newest first |
| `GET runs/<id>/`, `POST runs/<id>/cancel/` | one run; ask it to stop |
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

Send `{"type": "run", …payload}` or `{"type": "cancel"}`. One run per connection at a time; a second gets `busy`. Disconnecting cancels the run, unless it is a background run: `{"type": "run", "background": true, …}` starts one, `{"type": "watch", "run": "<id>"}` watches one, and `cancel` stops the one watched.

Close codes: 4401 not signed in, 4403 not your thread, 4404 no such thread.

## Background runs

A background run belongs to its thread, not to the request that started it. Closing the tab, a dropped connection or a deploy of the web tier (with a worker executor) doesn't stop it. Anyone who may use the thread can watch it or cancel it.

Start one with `"background": true` in any run payload (SSE, REST, WebSocket), or from code:

```python
from llm_kit import agents, background

run = agents.start("crm.researcher", "Audit last month's invoices", user=request.user)
run = background.start(thread, {"message": "…"}, user=request.user)   # any graph
run.status   # (waiting,) queued, running, then interrupted, done, error, cancelled or lost
run.cancel()
```

Bad payloads and callers over a daily budget are refused when you start it (400, 429). A thread runs one at a time: starting another, or a normal run, while one is going is refused (409, `RunBusy`, WebSocket `busy`) with the run in progress, unless you [queue it](#queued-follow-ups).

`background.active_run(thread)` returns the thread's queued or running run, or `None` (a run whose worker stopped is marked `lost` first). Use it to tell someone their message will be answered after the current one, or for a status command. It queries the database: from async code, wrap it in `sync_to_async`.

**Watching** (`GET llm/runs/<id>/watch/`, WebSocket `watch`, or `async for event in background.watch(run)`) gives:

1. `snapshot`: the run (`id`, `status`, timestamps, `result`, `error`), the thread's saved state (`values`, `next`, `interrupts`), so a late watcher sees what has been done so far, the runs waiting on the thread (`queue`), and `seq`, the last event it covers;
2. the run's events from then on, the same as a normal run's (`token`, `message`, `update`, `interrupt`, `end`, `error`);
3. a final `status` event once it has finished; the stream then closes.

**From any point.** Every event of a background run has `seq`: 1, 2, 3, … with no gaps (the SSE `id:`, and `"seq"` in WebSocket messages). The run's events are kept for a while, so you can watch from where you were instead of from a snapshot:

- `background.watch(run, after=n)`, `GET llm/runs/<id>/watch/?after=n`, or WebSocket `{"type": "watch", "run": "<id>", "after": n}` give the events after `n`, then live ones. `after=0` is every event from the first.
- A browser's `EventSource` reconnects with a `Last-Event-ID` header, and the watch view picks up after it: a dropped connection loses nothing.
- Starting a background run over SSE or WebSocket streams it from its first event, with no snapshot. Events sent before the stream subscribed aren't lost.
- A slow watcher doesn't slow the run. If its queue fills up, it reads what it missed from the log.
- If events it asked for are no longer kept, it gets a `snapshot` with `"resync": true` first. That snapshot replaces what the watcher has, and the events after it follow.

The log is capped at `RUN_EVENT_LOG_SIZE` events per run (2,000) and dropped `RUN_EVENT_LOG_TTL` seconds after the run's last event (300). Size it like this: a token event is roughly 150 bytes, so 1,000 runs finishing within the TTL, each streaming 1,000 tokens, keep about 150 MB. Writes are batched: the process running the run writes tokens at most every 100 ms, and other events at once, so a token never costs its own cache write. Watchers read the log only when they start and when they find a gap.

Where it's kept is `RUN_EVENT_LOG`:

- `"auto"` (the default) uses the Django cache `RUN_EVENT_LOG_CACHE` (`"default"`) when it is shared (Redis, Memcached), and this process's memory otherwise.
- `"cache"` or `"memory"` picks one.
- `"off"` keeps nothing: live events only, with no `seq`.

With workers, use a shared cache. The system check `llm_kit.W003` warns about a worker executor with a log in memory. Events your code [publishes](#background-runs) from a process that isn't running the run reach watchers live, without `seq`, and aren't kept.

**Your own events.** A run hook's context manager, a tool or a callback can send watchers something only your code knows, such as how a turn ended: `background.publish(target, {"event": "myapp.outcome", "data": {...}})` (or `await background.apublish(...)`). `target` is the hook's `RunSetup`, the run's `config`, a `BackgroundRun` or its id; `background.run_id_of(target)` gives the id, or `None`. Watchers get it like the kit's events, before the final `status`. The name needs a dot (the kit's own never have one), `data` must be JSON, and outside a background run nothing is sent and it returns `False`.

A run that ends at an `interrupt()` has status `interrupted` with the question in `result.interrupts`. Answer with a new run on the thread: `{"resume": …}`, in the background or not.

### Queued follow-ups

People send a second message before the first answer arrives. With `"queue": true` (or `background.start(..., queue=True)`, `agents.start(..., queue=True)`, or `RUN_QUEUE = True` for every background run), a run on a busy thread isn't refused. It is saved as `waiting` and starts when the runs before it have finished, in the order they were sent. `"queue": true` implies `"background": true`.

- Waiting runs show in `GET llm/api/runs/?status=waiting` and in every snapshot's `queue`. Watching one gives its snapshot, then its events once it starts.
- Cancel a waiting run like any other; it never starts. The runs behind it move up.
- A run that ends `done`, `error`, `cancelled` or `lost` lets the next one start. A run that ends `interrupted` holds the queue until the interrupt is answered. A `resume` never waits in the queue: it answers the interrupt, and the queue moves on when it finishes.
- A normal (not background) run may still go when nothing is running, and the queue moves on after it. If you run a thread's graph yourself, call `background.advance(thread)` afterwards.
- Each queued message is its own turn. They aren't merged: each keeps its own id, can be cancelled on its own, and its answer follows it in the thread.

### When a run finishes

To message someone when their long job is done (a chat message, a push, an email) without keeping a watcher open, connect to `run_finished`:

```python
from django.dispatch import receiver
from llm_kit.background import run_finished

@receiver(run_finished)
def tell_them(sender, run, status, result, error, **kwargs):
    if status in ("done", "interrupted"):
        notify(run.user, "Your report is ready.")      # your delivery
```

It is sent **once** for every background run that reaches a final status: `done`, `interrupted`, `error`, `cancelled` (also a waiting or queued run cancelled before it started) or `lost`. Whichever process notices sends it: the worker that ran it, the request that cancelled it, or the one that found it lost. When two processes notice the same lost run, only the one whose change took effect sends it, and a run that finishes after being marked lost isn't announced again. `result` is the `end` event's data (`next`, `interrupts`) for done and interrupted runs. It is sent after the change is committed. A receiver that raises is logged and doesn't stop the other receivers or the queue. Receivers run in the process that sends it, so hand slow delivery to a task queue.

**Where runs execute** is `RUN_EXECUTOR`:

| Value | Runs in | Notes |
|---|---|---|
| `"thread"` (default) | a thread in the web process | nothing to set up; a restart loses the run |
| `"celery"` | your Celery workers | task `llm_kit.run_in_background` |
| `"django_q"` | Django-Q2 | `llm_kit.background.tasks.run_in_background` |
| `"pkg.module:fn"` | your choice | called with the run id once the run is saved; have it call `llm_kit.background.execute(run_id)` |

A run whose process died is marked `lost` once it has sent no heartbeat for `RUN_LOST_AFTER` seconds (60), and then no longer holds the thread. Cancelling works from any process: the run notices within a second.

**How events reach watchers** is `RUN_EVENTS`. `"auto"` (the default) uses the Channels layer when `CHANNEL_LAYERS` is set to a shared one such as `channels_redis`; otherwise events only reach watchers in the same process. That is fine with the thread executor and one web process. With workers, or several web processes behind a load balancer, set up a Redis channel layer. The system check `llm_kit.W002` warns about a worker executor without one.

## Human in the loop

When a node calls LangGraph's `interrupt(value)`, the run ends with an `interrupt` event and an `end` event whose `interrupts` list is non-empty. Send the answer as a new run on the same thread with `{"resume": …}`.
