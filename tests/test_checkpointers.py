"""Pluggable checkpointers: any saver from sync and async code, one async saver per event loop."""

import asyncio
import operator
import os
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph

from llm_kit.core.checkpointers import CheckpointRouter, build_checkpointer, has_async, has_sync


class State(TypedDict):
    seen: Annotated[list, operator.add]


def graph(checkpointer):
    g = StateGraph(State)
    g.add_node("step", lambda s: {"seen": [len(s["seen"]) + 1]})
    g.add_edge(START, "step")
    return g.compile(checkpointer=checkpointer)


CFG = {"configurable": {"thread_id": "t1"}}


class SyncOnly(BaseCheckpointSaver):
    """Like PostgresSaver: no async methods."""

    def __init__(self):
        super().__init__()
        self.inner = InMemorySaver()

    def get_tuple(self, config):
        return self.inner.get_tuple(config)

    def list(self, config, **kw):
        return self.inner.list(config, **kw)

    def put(self, *args):
        return self.inner.put(*args)

    def put_writes(self, *args, **kw):
        return self.inner.put_writes(*args, **kw)

    def delete_thread(self, thread_id):
        return self.inner.delete_thread(thread_id)

    def get_next_version(self, current, channel=None):
        return self.inner.get_next_version(current, channel)


SHARED = InMemorySaver()  # stands in for the database every loop's saver talks to
MADE: list["PerLoop"] = []


class PerLoop(BaseCheckpointSaver):
    """Like AsyncPostgresSaver over a pool: async only, and only on the loop that made it."""

    def __init__(self):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.closed = False
        self.llm_kit_close = lambda: setattr(self, "closed", True)
        MADE.append(self)

    def _check(self):
        assert asyncio.get_running_loop() is self.loop, "used on another event loop"
        assert not self.closed

    async def aget_tuple(self, config):
        self._check()
        return await SHARED.aget_tuple(config)

    async def alist(self, config, **kw):
        self._check()
        async for item in SHARED.alist(config, **kw):
            yield item

    async def aput(self, *args):
        self._check()
        return await SHARED.aput(*args)

    async def aput_writes(self, *args, **kw):
        self._check()
        return await SHARED.aput_writes(*args, **kw)

    async def adelete_thread(self, thread_id):
        self._check()
        return await SHARED.adelete_thread(thread_id)

    def get_next_version(self, current, channel=None):
        return SHARED.get_next_version(current, channel)


async def make_per_loop():
    await asyncio.sleep(0)  # factories may be coroutines (open a pool, run setup)
    return PerLoop()


@pytest.fixture(autouse=True)
def _reset():
    SHARED.storage.clear()
    SHARED.writes.clear()
    MADE.clear()


# --------------------------------------------------------------------- build


def test_build_accepts_savers_classes_factories_and_paths():
    mem = InMemorySaver()
    assert build_checkpointer(mem) is mem  # sync and async already: used as is
    assert isinstance(build_checkpointer("langgraph.checkpoint.memory:InMemorySaver"), InMemorySaver)
    assert isinstance(build_checkpointer(InMemorySaver), InMemorySaver)
    wrapped = build_checkpointer(SyncOnly)
    assert isinstance(wrapped, CheckpointRouter) and has_sync(wrapped) and has_async(wrapped)


def test_build_refuses_what_cannot_work():
    with pytest.raises(TypeError, match="one event loop"):
        build_checkpointer(PerLoop.__new__(PerLoop))  # an async-only instance can't serve every loop
    with pytest.raises(ValueError, match="'sync' and/or 'async'"):
        build_checkpointer({"sink": "x"})
    with pytest.raises(TypeError, match="Not a LangGraph checkpointer"):
        build_checkpointer(lambda: 42)
    with pytest.raises(TypeError, match="async factory"):
        build_checkpointer(make_per_loop)


# ---------------------------------------------------------------- sync-only


def test_a_sync_only_saver_works_from_async_code():
    g = graph(build_checkpointer(SyncOnly))
    assert g.invoke({"seen": []}, CFG)["seen"] == [1]
    assert asyncio.run(g.ainvoke({"seen": []}, CFG))["seen"] == [1, 2]  # async methods run it in a thread
    assert [s.values["seen"] for s in g.get_state_history(CFG)][0] == [1, 2]


# ------------------------------------------------------------- async per loop


def test_each_event_loop_gets_its_own_async_saver():
    router = build_checkpointer({"async": make_per_loop})
    g = graph(router)
    assert asyncio.run(g.ainvoke({"seen": []}, CFG))["seen"] == [1]
    assert asyncio.run(g.ainvoke({"seen": []}, CFG))["seen"] == [1, 2]  # a second asyncio.run, as in Celery
    first, second = MADE
    assert first.loop is not second.loop
    assert first.closed and not second.closed  # the dead loop's saver was closed when it was replaced


def test_sync_calls_to_an_async_only_saver_run_on_a_background_loop():
    router = build_checkpointer({"async": make_per_loop})
    g = graph(router)
    assert g.invoke({"seen": []}, CFG)["seen"] == [1]
    assert asyncio.run(g.ainvoke({"seen": []}, CFG))["seen"] == [1, 2]
    assert len(list(router.list(CFG))) >= 2
    router.delete_thread("t1")
    assert router.get_tuple(CFG) is None
    router.close()
    assert all(s.closed for s in MADE)


def test_sync_and_async_factories_each_serve_their_side():
    sync = SyncOnly()
    sync.inner = SHARED
    router = build_checkpointer({"sync": lambda: sync, "async": make_per_loop})
    g = graph(router)
    g.invoke({"seen": []}, CFG)
    assert MADE == []  # sync code never made an async saver
    assert asyncio.run(g.ainvoke({"seen": []}, CFG))["seen"] == [1, 2]
    assert len(MADE) == 1


# ------------------------------------------------------------------ Django


@pytest.mark.django_db(transaction=True)
def test_the_setting_picks_the_checkpointer(settings):
    from llm_kit.persistence.saver import DjangoSaver
    from llm_kit.runtime.graphs import get_checkpointer

    assert isinstance(get_checkpointer(), DjangoSaver) or os.environ.get("LLM_KIT_TEST_CHECKPOINTER")
    settings.LLM_KIT = {**settings.LLM_KIT, "CHECKPOINTER": "langgraph.checkpoint.memory:InMemorySaver"}
    saver = get_checkpointer()
    assert isinstance(saver, InMemorySaver) and get_checkpointer() is saver  # built once per setting


@pytest.mark.skipif(not os.environ.get("LLM_KIT_TEST_PG"), reason="needs LLM_KIT_TEST_PG")
@pytest.mark.django_db(transaction=True)
def test_postgres_saver_end_to_end(settings):
    from llm_kit.models import Thread
    from llm_kit.runtime.graphs import get_checkpointer, get_graph

    settings.LLM_KIT = {**settings.LLM_KIT, "CHECKPOINTER": "postgres"}
    saver = get_checkpointer()
    try:
        thread = Thread.objects.create(graph="chat")
        g = get_graph("chat")
        cfg = thread.config
        g.invoke({"messages": [("user", "one")]}, cfg)
        asyncio.run(g.ainvoke({"messages": [("user", "two")]}, cfg))
        asyncio.run(g.ainvoke({"messages": [("user", "three")]}, cfg))  # a new loop: a new pool
        humans = [m.content for m in g.get_state(cfg).values["messages"] if m.type == "human"]
        assert humans == ["one", "two", "three"]
        thread.delete()  # the post_delete signal deletes its state from Postgres
        assert saver.get_tuple({"configurable": {"thread_id": thread.thread_id, "checkpoint_ns": ""}}) is None
    finally:
        saver.close()


def test_in_memory_sqlite_test_database_is_flagged(settings):
    from django.db import connections

    from llm_kit.persistence.checks import sqlite_test_database_is_shared

    assert sqlite_test_database_is_shared() == [] or os.environ.get("LLM_KIT_TEST_PG")  # ours is a file
    original = dict(connections.settings["default"])
    try:
        connections.settings["default"] = {**original, "ENGINE": "django.db.backends.sqlite3", "TEST": {}}
        assert [w.id for w in sqlite_test_database_is_shared()] == ["llm_kit.W001"]
    finally:
        connections.settings["default"] = original
