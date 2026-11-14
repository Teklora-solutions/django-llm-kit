"""Calls made for someone outside a request or agent run: caller= (or the caller in config)
applies their settings and records their user and tenant."""

import pytest

from llm_kit.core.observability import recorder
from llm_kit.core.scoped import Settable
from llm_kit.core.touchpoints import touchpoint
from llm_kit.core.touchpoints.touchpoint import _unregister
from llm_kit.tools.caller import CONFIG_KEY, caller_for
from tests.graphs import SCRIPTS

pytestmark = pytest.mark.django_db

H = "ollama/script-helper"


@pytest.fixture
def seen(monkeypatch, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "EMBEDDING_BACKEND": "tests.test_embeddings:backend"}
    records = []
    monkeypatch.setattr(recorder, "emit", records.append)
    SCRIPTS.clear()
    yield records
    SCRIPTS.clear()
    _unregister("t.direct")


def calls(records):
    return [r for r in records if hasattr(r, "key_owner")]


def test_complete_records_the_caller(seen, world):
    tp = touchpoint("t.direct", default=H)
    SCRIPTS[H] = ["one", "two"]
    tp.complete("hi", caller=caller_for(world["carol"]))
    tp.complete("hi", config={"configurable": {CONFIG_KEY: caller_for(world["bob"])}})
    carol, bob = calls(seen)
    assert (carol.user, carol.tenant) == (str(world["carol"].pk), str(world["acme"].pk))
    assert (bob.user, bob.tenant) == (str(world["bob"].pk), str(world["globex"].pk))


def test_a_plain_user_works_as_a_caller(seen, world):
    tp = touchpoint("t.direct", default=H)
    SCRIPTS[H] = ["one"]
    tp.complete("hi", caller=world["carol"])  # a user: no tenant known
    [rec] = calls(seen)
    assert (rec.user, rec.tenant) == (str(world["carol"].pk), "")


def test_embed_records_the_caller(seen, world):
    from tests.test_touchpoints_core import FakeEnv

    tp = touchpoint("t.direct", default="openai/text-embedding-3-small")
    tp.embed("x", env=FakeEnv(), caller=caller_for(world["carol"]))
    import asyncio

    asyncio.run(tp.aembed("y", env=FakeEnv(), config={"configurable": {CONFIG_KEY: caller_for(world["bob"])}}))
    first, asyncio_rec = calls(seen)
    assert (first.user, first.tenant, first.kind) == (str(world["carol"].pk), str(world["acme"].pk), "embedding")
    assert (asyncio_rec.user, asyncio_rec.tenant) == (str(world["bob"].pk), str(world["globex"].pk))


def test_caller_applies_their_saved_settings(seen, world, settings):
    from llm_kit.core.touchpoints.policy import ClientChoice
    from llm_kit.models import ScopedSetting

    tp = touchpoint("t.direct", default=H, choice=ClientChoice(allow=["ollama/*"]), settable=Settable(model=("user",)))
    # Written directly: the scripted model isn't in the catalogue, so the API wouldn't offer it.
    carol = str(world["carol"].pk)
    ScopedSetting.objects.create(
        scope="user", scope_id=carol, touchpoint="t.direct", key="model", value="ollama/script-assistant"
    )
    SCRIPTS["ollama/script-assistant"] = ["from her model"]
    assert tp.complete("hi", caller=caller_for(world["carol"])).content == "from her model"
    assert tp.model(caller=caller_for(world["carol"])) is not None
