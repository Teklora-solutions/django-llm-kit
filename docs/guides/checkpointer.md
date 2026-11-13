---
description: Choose where LangGraph state is stored - DjangoSaver on any Django database, LangGraph's PostgresSaver, or your own saver - and use DjangoSaver directly.
---

# Checkpointers

Every graph and agent uses one checkpointer, set by `LLM_KIT["CHECKPOINTER"]`:

| Value | Meaning |
|---|---|
| `None` (default) | `DjangoSaver`: llm-kit's tables, on any database Django supports (SQLite, Postgres, MySQL) |
| `"postgres"` | LangGraph's own `PostgresSaver`/`AsyncPostgresSaver`, on Django's Postgres database (`[postgres]` extra). Tables are created on first use |
| `"myapp.ai:saver"` | a saver instance, a saver class, or a zero-argument factory |
| `{"sync": "myapp.ai:saver", "async": "myapp.ai:asaver"}` | separate factories for sync and async code; the async one may be a coroutine |

A project that already has an `AsyncPostgresSaver` can point at its factory, and its existing threads keep working.

- **One async saver per event loop.** An async connection pool belongs to the loop that opened it. A Celery task that calls `asyncio.run` gets a new loop each time, and a shared saver would fail on the second task. llm-kit makes one async saver per running loop and closes the savers of loops that have finished.
- **Missing halves are filled in.** A sync-only saver (`PostgresSaver`) is run in a worker thread for async code. An async-only factory serves sync code from one background loop.
- **Shut down cleanly.** With pooled savers (`"postgres"`, or a sync/async dict), `llm_kit.runtime.graphs.get_checkpointer().close()` frees the connections.
- **Deleting a `Thread`** deletes its state from whichever checkpointer is configured. With `DjangoSaver` its subagents' threads go too; other savers have no way to find those by prefix, so they stay in that checkpointer.

## Using DjangoSaver directly

```python
from llm_kit import DjangoSaver

graph = builder.compile(checkpointer=DjangoSaver())          # or DjangoSaver(using="replica")
graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "abc"}})
```

It implements the sync and async LangGraph checkpointer API: `get_tuple`, `list` (with `filter`, `before`, `limit`), `put`, `put_writes`, `delete_thread`, `copy_thread`. That covers interrupts, time travel and forking.

Channel values are stored once per version in `CheckpointBlob` and shared across checkpoints, so long chats don't duplicate their history at every step. The checkpoint tables use plain string thread ids, so `DjangoSaver` works without llm-kit's `Thread` model. Checkpoints are visible read-only in the admin.

`prune` (deleting old checkpoints inside a thread) is not implemented, because LangGraph's delta channels make a naive version break state. `llm_kit_prune --threads DAYS` deletes whole idle threads instead.

## SQLite

LangGraph writes checkpoints from worker threads. Set this to avoid "database is locked" under concurrent load (Django 5.1+):

```python
DATABASES["default"]["OPTIONS"] = {"transaction_mode": "IMMEDIATE", "timeout": 20}
```

For tests on SQLite, see [Testing](testing.md).
