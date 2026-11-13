---
description: How to test code that uses django-llm-kit - fake models, SQLite test databases, transactions, and recording.
---

# Testing

## Don't call real providers

Point your touchpoints at a fake model in tests. The simplest way is a custom backend in test settings:

```python
# tests/fakes.py
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class EchoModel(GenericFakeChatModel):
    """Replies 'echo: <last message>' and streams it word by word."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.messages = iter([AIMessage(content=f"echo: {messages[-1].content}")])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):   # agents bind tools; the echo model ignores them
        return self


def fake_backend(ref, params, api_keys):     # MODEL_BACKEND: every touchpoint gets the echo model
    return EchoModel(messages=iter([]))


def any_key(provider):                       # CREDENTIALS: every provider counts as configured
    return "test-key"
```

```python
# test settings
LLM_KIT = {**LLM_KIT, "MODEL_BACKEND": "tests.fakes:fake_backend", "CREDENTIALS": "tests.fakes:any_key"}
```

To script tool calls, write a `BaseChatModel` whose `_generate` returns `AIMessage(tool_calls=[...])` from a queue; llm-kit's own suite does this in `tests/graphs.py` (`ScriptModel`).

Without `CREDENTIALS`, touchpoint resolution would skip models that have no API key. Load the catalogue from the bundled snapshot rather than the network: `call_command("llm_kit_sync_catalogue", offline=True)`.

## SQLite test databases

Django's default SQLite test database is in memory, and another thread's write locks whole tables there, so tests that run graphs fail with "database table is locked". Give the test database a file:

```python
DATABASES["default"]["TEST"] = {"NAME": BASE_DIR / "test.sqlite3"}
```

The system check `llm_kit.W001` warns when it is in memory.

## Transactions

Tests that run graphs with a database checkpointer need `TransactionTestCase` (or `pytest.mark.django_db(transaction=True)`), because the checkpoint thread can't see a test transaction's uncommitted rows.

`agents.run(...)` without a thread is the exception: throwaway runs keep their state in memory, so they work inside `TestCase`.

## Recording

The recorder writes from a background thread, and on SQLite its writes can collide with a test's open transaction ("database is locked", on either side). Set `RECORD_SINKS` to `[]` in test settings unless a test reads the recorded rows; in those tests, call `llm_kit.observability.flush()` before asserting.

Postgres has none of these problems.
