---
description: Register LangGraph graphs by dotted path; llm-kit compiles them with the configured checkpointer and serves them by name.
---

# Graphs

Each entry in `LLM_KIT["GRAPHS"]` names a graph that clients can run through the [streaming views, REST API and WebSocket](streaming.md).

```python
LLM_KIT = {
    "GRAPHS": {
        "assistant": {"graph": "myapp.graphs:builder", "touchpoint": "myapp.chat"},
        "summariser": "myapp.graphs.summariser",
    },
    "DEFAULT_GRAPH": "assistant",
}
```

## What a graph entry can point at

The value is a dotted path (`pkg.mod:attr` or `pkg.mod.attr`) to one of:

| Target | What llm-kit does |
|---|---|
| uncompiled `StateGraph` | `builder.compile(checkpointer=<configured checkpointer>)` |
| compiled graph | attaches the checkpointer unless it already has one |
| `fn(checkpointer)` factory | calls it; use this for `create_agent(..., checkpointer=checkpointer)` or custom compile options |

Or a dict `{"graph": <path>, "touchpoint": "<name>", "prepare": "<hook>"}`. The touchpoint is the policy a client's `"model"` choice is checked against. Without a touchpoint, a request that sends `"model"` is rejected with 400. `prepare` is a [run hook](#run-hooks) for this graph.

Graph modules are imported at startup, so touchpoints declared in them are registered before the first request.

## Agents are graphs too

Every [agent](agents.md) declared with `agent(...)` is available as a graph under its own name, without an entry in `GRAPHS`. Agent names must not clash with `GRAPHS` keys; this is checked at startup.

## Using a graph in your own code

```python
import llm_kit

graph = llm_kit.get_graph("assistant")    # compiled, cached per process
graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "abc"}})
```

`DEFAULT_GRAPH` names the graph used when a request doesn't say; if only one graph is registered, that one is the default.

## Choosing the model inside a node

Inside a node, ask the touchpoint for a model with the run's config. The config carries the client's pick (if the policy allows it) and the caller:

```python
def call_model(state, config):
    llm = chat.model(config=config)
    return {"messages": [llm.invoke(state["messages"], config)]}
```

Passing `config` on to `invoke` keeps the call inside the run, so tokens stream to the client and the call is recorded against the run.

## Run hooks

Some of a run's input isn't the client's to decide: what the caller may do, who the answer is for, which tools are allowed, where to record activity. A run hook sets it on the server, every time a graph runs: the SSE, REST and WebSocket views, background runs (in whichever process or worker runs them) and `agents.run`/`arun`/`stream` from your code.

```python
# settings.py
LLM_KIT = {
    "RUN_HOOKS": ["myapp.llm:prepare"],           # every graph and agent
    "GRAPHS": {"support": {"graph": "myapp.graphs:support", "prepare": "myapp.llm:support_only"}},
}

# myapp/llm.py
def prepare(run):
    user = run.owner.user
    run.set_state(
        audience="staff" if user and user.is_staff else "customer",
        can_refund=bool(user and user.has_perm("shop.refund")),
    )
    run.set_configurable(__corpus=f"kb-{run.owner.tenant_id}")   # server-only
    run.add_callback(ActivityRecorder(run.thread))
    return [use_channel("web"), telemetry_row(run)]              # context managers, sync or async
```

A hook is a plain function (it may query the database) called with a `RunSetup` just before the run, in the process that runs it. A hook may:

- **set state** with `run.set_state(...)`. The keys are merged over the graph's input after the client's, so a client can't set or replace them; when resuming an interrupt they become the `Command`'s `update`. Set every key your graph relies on, every time. `run.input` can also be changed in place, for example to turn attachments into content blocks.
- **set configuration** with `run.set_configurable(...)`, which overrides the client's. Keys starting with `__` never come from a client.
- **add callbacks** with `run.add_callback(cb)`.
- **wrap the run** by returning context managers (or calling `run.enter(cm)`). They are entered around the whole run, in order, in the task that runs it, and exited when it ends however it ends. This is the place to set ContextVars your tools read, or write a row in `finally`.

It sees `run.graph`, `run.owner` (`user`, `tenant`), `run.thread`, `run.payload` (a read-only copy of what the client sent), `run.background` and `run.context`.

### Server-set context

The payload is shaped by the client. Facts only your server knows go in `context` instead: the channel a message came in on, an API key's narrowed permissions, attachment ids you've already checked against the caller. Your own views and tasks pass it:

```python
background.start(thread, {"message": text}, user=user, context={"channel": "chat", "attachments": [42]})
agents.run("support.assistant", text, user=user, context={"channel": "api", "scopes": ["orders:read"]})
agents.start(..., context={...})
setup = runtime.runs.prepare(thread, payload, owner=owner, context={...})   # if you run graphs yourself
```

Hooks get it read-only as `run.context`. Tools can read it with `llm_kit.core.hooks.context_of(config)`. No transport ever sets it from a request: a `"context"` key in a client's body is ignored, and `__llm_kit_context` in its `configurable` is dropped like every `__` key. So it is empty for runs the built-in views start, unless your own view passes it.

It must be a small JSON object (16 KB at most), because it is saved with a background run for the worker. A background subagent inherits its parent run's context, and so does the report of its answer.

For a background run, the hooks run in the worker, from the saved payload and owner, not when the run is queued. A permission change between queueing and running is seen. Nothing the hooks return is stored.

**Errors fail the run before anything runs.** Raise `llm_kit.core.hooks.RunRefused("…")` to refuse it: a 403 with your message, a WebSocket `error`, or a background run in status `error` with that message. Anything else a hook raises, or a context manager raises on entering, is an `error` event (details only with `EXPOSE_ERRORS`), a 500 from the REST API, or status `error`, and is logged. Hooks run in order: `RUN_HOOKS`, then the graph's `prepare`. A run started inside another agent's request (`agents.run` from a tool) doesn't run them again: it is part of that request.

