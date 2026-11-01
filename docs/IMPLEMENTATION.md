# django-llm-kit: Implementation Record

This file records what was built in v0.1.0 (commit `7f169ed`, 2026-11-01), why it was built that way, and what is still missing. It was written after the work was done, because no plan was shared before it started. Section 2 lists the decisions made without asking you, so you can review them.

---

## 1. Goal and the choices you made

**Goal:** make it easy to use LangGraph and LangChain inside Django projects.

You chose:

| Question | Your answer |
|---|---|
| Shape | Reusable pip package (an installable Django app), plus an example project |
| v0.1 features | ORM checkpointer, streaming views, DRF endpoints, Channels / WebSocket, **model catalogue** (your addition) |
| LLM provider | Any provider, set in settings (via LangChain `init_chat_model`) |

Versions it was built and tested against: Python 3.11, Django 5.2.18, DRF 3.18.3, Channels 4.3.2, LangGraph 1.2.19, langgraph-checkpoint 4.3.1, LangChain 1.4.7, langchain-core 1.7.3.

---

## 2. Decisions made without asking you (please review)

| # | Decision | Why | Easy to change? |
|---|---|---|---|
| D1 | Package name **`django-llm-kit`**, import name **`llm_kit`** | `llm-kit` is probably taken on PyPI; the `django-` prefix is the usual convention | Yes (pyproject + imports) |
| D2 | **"Model catalogue"** means a DB table `LLMModel` (editable in the admin), plus aliases in settings, plus raw `provider:model` strings, all resolved with `init_chat_model` | You named the feature but didn't describe it; this was my reading | Moderately |
| D3 | API keys are **never stored in the DB**. A catalogue row can name an env var (`api_key_env`) instead | Keeps secrets out of the database, backups and the admin | Yes |
| D4 | The model is chosen **per request** and reaches the graph as `config["configurable"]["model"]` | Lets a client pick "fast" or "smart" per message | Yes |
| D5 | A **`Thread` model** owns conversations (user FK). Its UUID is the LangGraph `thread_id` | Needed for permissions, listing, titles and admin | Hard (core design) |
| D6 | **Access rule:** logged-in users only see their own threads (`REQUIRE_AUTH=True` by default). Ownerless threads are staff-only | Safe default | Yes (`access.py`) |
| D7 | Streaming uses **plain async Django views with SSE**, not DRF | DRF views are sync; streaming needs async | Moderately |
| D8 | SSE endpoints use **session auth + CSRF only** (no token/JWT yet) | Simplest secure default | Known gap (see §9) |
| D9 | DRF and Channels are **optional extras** (`[drf]`, `[channels]`, `[all]`) | Not every project uses them | Yes |
| D10 | Checkpoints stored in a **blob-per-channel-version** layout (like the official Postgres saver), not one whole state per checkpoint | Long chats don't copy the full history into every step | Hard (schema) |
| D11 | Error details are sent to clients **only when `DEBUG=True`**; otherwise "The run failed." (always logged) | Avoid leaking internals | Yes |
| D12 | Deleting a `Thread` **also deletes its checkpoints**, via a `post_delete` signal | Works for admin, API and user cascade deletes | Yes |
| D13 | `prune` (deleting old checkpoints) was **not implemented** | LangGraph's DeltaChannel rules make a naive version break state | Planned |
| D14 | Initialised a **git repo** and made one commit, authored as you | Project had no VCS | — |
| D15 | Build tool **hatchling**, `src/` layout, MIT license, ruff (line length 120) | Common defaults | Yes |

---

## 3. Architecture

```
          HTTP (SSE)              HTTP (REST)                WebSocket
        llm_kit/views.py       llm_kit/api/views.py      llm_kit/consumers.py
               │                       │                         │
               └───────────┬───────────┴────────────┬────────────┘
                           ▼                        ▼
                   llm_kit/access.py         llm_kit/runs.py
                (who can use a thread)   (payload → input/config,
                                          streaming events, JSON)
                                                    │
                                                    ▼
                                           llm_kit/graphs.py
                                (registry: name → compiled graph,
                                 compiled with DjangoSaver, cached)
                                     │                    │
                                     ▼                    ▼
                        llm_kit/checkpoint.py     llm_kit/catalogue.py
                         DjangoSaver (ORM)       get_chat_model / model_from_config
                                     │                    │
                                     ▼                    ▼
                         Checkpoint / Blob / Write     LLMModel table
                               tables                + LLM_KIT["MODELS"]
                                                      → init_chat_model
```

The three transports (SSE, REST, WebSocket) all go through the same `runs.py` code, so they accept the same payload and send the same events.

---

## 4. Files

```
pyproject.toml                 package metadata, extras, pytest + ruff config
README.md                      user guide
LICENSE                        MIT
src/llm_kit/
  __init__.py                  lazy exports: DjangoSaver, get_graph, get_chat_model, model_from_config
  apps.py                      AppConfig; connects signals in ready()
  conf.py                      reads settings.LLM_KIT and applies defaults
  models.py                    Thread, LLMModel, Checkpoint, CheckpointBlob, CheckpointWrite
  migrations/0001_initial.py   generated
  checkpoint.py                DjangoSaver: the LangGraph checkpointer
  catalogue.py                 model catalogue resolution
  graphs.py                    graph registry + shared checkpointer
  runs.py                      payload parsing, config building, streaming, serialization
  access.py                    thread permission rules
  views.py                     async SSE endpoints + keep-alive
  urls.py                      SSE routes
  signals.py                   thread delete → checkpoint cleanup
  admin.py                     admin for LLMModel, Thread, Checkpoint (read-only)
  consumers.py                 Channels WebSocket consumer
  routing.py                   WebSocket routes
  api/
    serializers.py             Thread, Run, ModelSpec serializers
    permissions.py             ThreadAccess (wraps access.py)
    views.py                   ThreadViewSet (+ state/history/runs), ModelListView, GraphListView
    urls.py                    DRF router
tests/
  settings.py, urls.py         test project (SQLite file DB; Postgres via env var)
  graphs.py                    fake echo model + test graphs (chat, approval, boom, catalogue, slow)
  conftest.py                  fixtures (fresh graph cache, users, fake init_chat_model, async DB cleanup)
  test_checkpoint.py           8 tests
  test_catalogue.py            7 tests
  test_views.py                7 tests
  test_api.py                  5 tests
  test_consumers.py            4 tests
example/
  manage.py, project/          runnable demo project (daphne, DRF, Channels)
  chat/graphs.py               tool-calling assistant (current_time tool), model chosen per request
  chat/offline.py              echo model for LLM_KIT_DEMO_OFFLINE=1
  chat/templates/chat/index.html   streaming chat page with model picker
```

---

## 5. Data model

### `Thread`: a conversation
| Field | Type | Notes |
|---|---|---|
| `id` | UUID PK | used as the LangGraph `thread_id` |
| `user` | FK → AUTH_USER_MODEL, nullable | owner; cascade delete |
| `graph` | char(100) | key in `LLM_KIT["GRAPHS"]`; can't be changed through the API |
| `title` | char(255) | defaults to the first message (SSE) |
| `metadata` | JSON | free-form |
| `created_at` / `updated_at` | datetime | `updated_at` is bumped after each run |

Index: `(user, -updated_at)`.

### `LLMModel`: the catalogue
| Field | Notes |
|---|---|
| `name` (slug, unique) | the alias clients send, e.g. `fast` |
| `model` | `init_chat_model` id, e.g. `anthropic:claude-sonnet-5-5` |
| `provider` | optional `model_provider` |
| `params` (JSON) | extra kwargs: temperature, max_tokens, … |
| `api_key_env` | name of the env var holding the key |
| `description`, `is_active`, `is_default` | saving one default unsets the others (inside a transaction) |

### Checkpoint storage
| Table | Unique key | Holds |
|---|---|---|
| `Checkpoint` | (thread_id, checkpoint_ns, checkpoint_id) | serialized checkpoint (without channel values), parent id, metadata as serde bytes **and** as JSON (for filtering/admin) |
| `CheckpointBlob` | (thread_id, checkpoint_ns, channel, version) | one channel value at one version; `type="empty"` marks a deleted value |
| `CheckpointWrite` | (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) | pending writes from tasks (incl. interrupts/errors with negative idx) |

The checkpoint tables use plain string `thread_id`s, not FKs to `Thread`, so `DjangoSaver` also works on its own without the `Thread` model.

---

## 6. Components in detail

### 6.1 `DjangoSaver` (`checkpoint.py`)
Implements `BaseCheckpointSaver`:

| Method | Behaviour |
|---|---|
| `get_tuple` | the checkpoint with the given id, or the latest by `checkpoint_id` (uuid6 ids sort by time) |
| `list` | filters by thread, namespace, id, `before`, metadata `filter` and `limit`; newest first. Scalar filters run in SQL (`metadata__key=value`) and are checked again in Python; other filters run only in Python |
| `put` | stores a blob for each channel in `new_versions`, then the checkpoint row |
| `put_writes` | normal writes are kept the first time (`ignore_conflicts`); special writes (error/interrupt/resume/scheduled) overwrite |
| `delete_thread`, `copy_thread` | across all three tables, in one transaction |
| `get_next_version` | `"{n:032}.{random}"` string versions (same scheme as InMemorySaver) |
| async versions | wrap the sync methods with `sync_to_async` (thread-sensitive, which is Django's recommendation for the ORM) |

Loading is batched: building any number of tuples takes 1 query for checkpoints, 1 for blobs and 1 for writes.

**Two problems found and fixed during testing:**
1. **Leaked connections.** LangGraph's sync loop calls the checkpointer from short-lived worker threads. The `@_db` decorator closes the connection afterwards if the call opened it (and isn't inside a transaction).
2. **SQLite "database is locked".** `update_or_create` reads before it writes, and SQLite can't upgrade that lock while other threads are writing. `put` and `put_writes` now only write (delete + insert), which works on every backend.

### 6.2 Graph registry (`graphs.py`)
`LLM_KIT["GRAPHS"]` maps a name to a dotted path (`pkg.mod:attr` or `pkg.mod.attr`) pointing at one of:
- an uncompiled `StateGraph` → `compile(checkpointer=DjangoSaver())`
- a compiled graph → `copy(update={"checkpointer": saver})`, unless it already has a checkpointer
- a factory `fn(checkpointer)` → must return a compiled graph (for `create_agent` and custom compile options)

Compiled graphs are cached per process (with a lock). `DEFAULT_GRAPH`, or the only registered graph, is the default. `clear_cache()` is for tests.

### 6.3 Model catalogue (`catalogue.py`)
How `resolve(name)` looks a name up:
1. active `LLMModel` row with that name
2. `LLM_KIT["MODELS"][name]` (a string, or a dict whose leftover keys become params)
3. `name` itself if it contains `:` (raw `init_chat_model` id)
4. otherwise `UnknownModel`

With no name: the row with `is_default=True`, then `LLM_KIT["DEFAULT_MODEL"]`, then `UnknownModel`.

Public functions: `get_chat_model`, `aget_chat_model`, `model_from_config`, `amodel_from_config`, `resolve`, `available()` (DB rows + settings aliases not overridden by a row). A missing `api_key_env` variable raises `ImproperlyConfigured`.

### 6.4 Run core (`runs.py`)
- `build_input(payload)`: `resume` → `Command(resume=…)`; `message` → user message; otherwise `input`.
- `build_config(thread, payload, user)`: merges the client's `configurable`, adds `model`, **always overwrites `thread_id`** (a client can't reach another thread), passes `recursion_limit`, and adds metadata `llm_kit_thread` and `user_id`.
- `astream_run(graph, input, config)`: streams with `STREAM_MODES` and turns the output into events (below). Exceptions become an `error` event.
- `invoke_run`: sync, for REST; returns the state as JSON.
- `to_json` / `message_to_json` / `state_to_json`: turn messages, interrupts, pydantic models, dataclasses and state snapshots into JSON.

**Events:** `thread`, `token`, `message`, `update`, `interrupt`, `end`, `error` (+ WebSocket only: `busy`, `cancelled`).

### 6.5 Access (`access.py`)
- `can_use_api(user)`: logged in, or `REQUIRE_AUTH=False`.
- `check_thread`: owner must match. Ownerless threads are staff-only when auth is required, open to everyone when it isn't.
- `threads_for(user)`: listing only shows the user's own threads (anonymous users see nothing).

### 6.6 SSE views (`views.py`)
- `POST /llm/stream/`: creates a thread (`graph`, `title` optional) and streams the first run.
- `POST /llm/threads/<uuid>/stream/`: streams a run on an existing thread.
- Status codes: 401 not logged in, 403 not your thread, 404 thread not found, 400 bad JSON / unknown graph / unknown model, 405 for GET.
- `with_keepalive` sends a `: keepalive` comment every `SSE_KEEPALIVE` seconds of silence, without cancelling the run's pending step, and cleans up when the client disconnects.
- Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no` (stops nginx from buffering the stream).

### 6.7 REST API (`api/`)
| Route | Action |
|---|---|
| `GET/POST threads/` | list your threads / create one |
| `GET/PATCH/PUT/DELETE threads/<id>/` | delete also removes the checkpoints |
| `POST threads/<id>/runs/` | run synchronously; returns the state |
| `GET threads/<id>/state/` | current state |
| `GET threads/<id>/history/?limit=` | newest first, capped at 200 |
| `GET models/` | catalogue (no `api_key_env` in the output) |
| `GET graphs/` | registered graph names |

### 6.8 WebSocket (`consumers.py`, `routing.py`)
- `ws/llm/threads/<uuid>/` (existing thread) and `ws/llm/threads/` (thread created on the first run).
- Client messages: `{"type":"run", …payload}` and `{"type":"cancel"}`.
- One run per connection at a time (a second run gets `busy`). Each run is an asyncio task, so cancelling stops the graph. Disconnecting cancels the run.
- Close codes: 4401, 4403, 4404.

### 6.9 Admin
- `LLMModel`: list, filter, `is_active` editable in the list.
- `Thread`: shows the checkpoint count.
- `Checkpoint`: read-only, shows step/source from the metadata; the binary fields are hidden.

---

## 7. Settings reference (`LLM_KIT`)

| Key | Default | Meaning |
|---|---|---|
| `GRAPHS` | `{}` | name → dotted path |
| `DEFAULT_GRAPH` | `None` | falls back to the only graph |
| `DEFAULT_MODEL` | `None` | id or alias used when nothing else matches |
| `MODELS` | `{}` | static aliases |
| `REQUIRE_AUTH` | `True` | see §6.5 |
| `STREAM_MODES` | `["messages", "updates", "custom"]` | LangGraph stream modes (`custom` carries subagents' messages) |
| `SSE_KEEPALIVE` | `15` | seconds; `0` turns keep-alives off |

---

## 8. Testing and verification

| Check | Result |
|---|---|
| `pytest` on SQLite (file DB) | **31/31 passed** (~2.3 s), stable across 3 repeat runs of the checkpointer tests |
| `pytest` on **Postgres 16** (Docker, `LLM_KIT_TEST_PG=…`) | **31/31 passed** |
| `ruff check` + `ruff format --check` | clean |
| `uv build` | wheel + sdist build; migrations included |
| `manage.py check` on the example | no issues |
| Browser (Playwright) on the example, offline echo model | 2 messages streamed over SSE into one thread, model picker filled from `/llm/api/models/`; DB showed 1 thread with 6 checkpoints |
| CSRF | a POST without a token was rejected (403) |

What the tests cover: matching results with LangGraph's `InMemorySaver`, a fresh saver reading existing state (like after a restart), blobs being shared between checkpoints, time travel and forking, interrupt and resume (sync + async), list filter/before/limit/across threads, `put_writes` rules, copy and delete thread, catalogue lookup order and defaults and API-key env vars, SSE tokens/updates/interrupt/resume/model choice/errors/anonymous mode/keep-alive, REST CRUD/runs/state/history/user isolation/validation, WebSocket new-thread/reuse/interrupt/rejections/cancel/busy.

**Not verified:** real LLM providers (everything ran on a fake echo model), MySQL, behaviour under concurrent load, and `prune`.

Two test-only fixes worth knowing about:
- The test DB is a **file**, not `:memory:`, because LangGraph's worker threads need to share it.
- A conftest fixture closes the DB connections that async ORM calls in test bodies leave open (a pytest-django/asgiref quirk, not a library bug).

---

## 9. Known gaps / suggested next steps

1. **Token/JWT auth for SSE.** Today SSE only accepts session + CSRF; separate SPA or mobile frontends need DRF authenticators or a pluggable auth hook. *(Most important if you have a separate frontend.)*
2. **`prune` / retention.** Must respect DeltaChannel snapshot chains (see the LangGraph docs in `BaseCheckpointSaver.prune`).
3. **Faster `get_delta_channel_history`.** The default implementation walks the chain with one query per ancestor.
4. **Background runs** (Celery/RQ) with results streamed through the channel layer, for long jobs.
5. **Rate limiting / quotas** per user or per model.
6. **Real-provider smoke tests** (marked, opt-in) and **MySQL CI**.
7. **CI pipeline** (GitHub Actions: SQLite + Postgres matrix).
8. **`StreamingHttpResponse` under WSGI** works but buffers the whole reply; document ASGI as required or add a warning.
9. **Publish:** check whether `django-llm-kit` is free on PyPI.
