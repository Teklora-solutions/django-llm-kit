---
description: Who may use a thread - users, anonymous visitors, keys your code chose, staff - and how tenants are isolated.
---

# Access control

A thread belongs to a **user**, an anonymous **visitor**, a **key your code chose**, or **nobody** (made by code).

- With `REQUIRE_AUTH: True` (the default), every endpoint needs a signed-in user, and users see only their own threads.
- With `REQUIRE_AUTH: False`, visitors may chat too. Their threads are tied to a random id kept in their session data, so other visitors can't use them even if they know the thread id. Over WebSocket this needs `SessionMiddlewareStack` and a session started by an earlier page load.
- When a visitor signs in, their threads move to their account (`ADOPT_SESSION_THREADS`, on by default). Django rotates the session key at login but keeps the session data, which is why the id lives in the data. Call `llm_kit.runtime.access.adopt_session_threads(request, user)` yourself if you sign users in some other way.
- Staff may open threads that have no user (visitor, keyed or code threads), to inspect them; never another user's.
- With `TENANT_RESOLVER` (a dotted path to `fn(user) -> tenant`), new threads are stamped with the user's tenant, and a thread from another tenant is refused. The same tenant scopes [model tools](tools.md) and is recorded on every call.

## Threads your code keys

For conversations that belong to something other than a signed-in user (a WhatsApp number, a ticket, a Celery job). Clients can never pick or look up a key; only your code can:

```python
from llm_kit import agents
from llm_kit.runtime.threads import open_thread

thread = open_thread("support.assistant", key=f"whatsapp:{jid}", tenant=business.pk)  # get or create
result = agents.run("support.assistant", text, thread=thread)  # user and tenant come from the thread
```

Keys are unique per tenant. `aopen_thread` is the async form.

## Model choice is access control too

A client's `"model"` is checked against the graph's [touchpoint](touchpoints.md) policy. With `ServerChoice` (the default) it is ignored or rejected; with `ClientChoice` it must match the allowed patterns, capabilities and price caps, and have an API key.
