import pytest

from llm_kit.runtime import graphs
from tests.testapp.fixtures import world  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def _fresh_graphs():
    graphs.clear_cache()
    yield
    graphs.clear_cache()


@pytest.fixture(autouse=True, scope="session")
def _close_checkpointer(django_db_setup):
    """Pooled checkpointers (CHECKPOINTER="postgres") hold connections; free them before the test DB is dropped."""
    yield
    saver = graphs._saver
    if hasattr(saver, "close"):
        saver.close()


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Rolled-back test transactions fire no signals, so cached overrides/keys must be reset."""
    from django.core.cache import cache

    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def saver():
    from llm_kit.persistence.saver import DjangoSaver

    return DjangoSaver()


@pytest.fixture
def built():
    """Models built by the fake backend during the test."""
    from tests.graphs import BUILT

    BUILT.clear()
    return BUILT


@pytest.fixture
def keys(monkeypatch):
    """API keys for the providers used in tests."""
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(var, f"key-{var.lower()}")


@pytest.fixture
def user(django_user_model):
    return django_user_model.objects.create_user("alice", password="pw")


@pytest.fixture
def other_user(django_user_model):
    return django_user_model.objects.create_user("bob", password="pw")


@pytest.fixture(autouse=True)
async def _close_async_db_connections():
    """Async ORM calls in test bodies open a connection in asgiref's sync thread
    that nothing closes; close it so Postgres can drop the test database."""
    yield
    from asgiref.sync import sync_to_async
    from django.db import connections

    await sync_to_async(connections.close_all)()
