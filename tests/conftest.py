import pytest

from llm_kit.runtime import graphs


@pytest.fixture(autouse=True)
def _fresh_graphs():
    graphs.clear_cache()
    yield
    graphs.clear_cache()


@pytest.fixture
def saver():
    from llm_kit.persistence.saver import DjangoSaver

    return DjangoSaver()


@pytest.fixture
def fake_init(monkeypatch):
    """Replace init_chat_model with one that records calls and returns an echo model."""
    from tests.graphs import EchoModel

    calls = []

    def fake(model, **kwargs):
        calls.append((model, kwargs))
        return EchoModel(messages=iter([]))

    monkeypatch.setattr("llm_kit.runtime.legacy_catalogue.init_chat_model", fake)
    return calls


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
