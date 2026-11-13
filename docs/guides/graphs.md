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

Or a dict `{"graph": <path>, "touchpoint": "<name>"}`. The touchpoint is the policy a client's `"model"` choice is checked against. Without a touchpoint, a request that sends `"model"` is rejected with 400.

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
