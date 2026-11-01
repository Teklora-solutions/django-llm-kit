import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from llm_kit.models import Checkpoint, CheckpointBlob, CheckpointWrite
from tests.graphs import chat_builder, make_approval_graph

pytestmark = pytest.mark.django_db(transaction=True)


def cfg(tid=None):
    return {"configurable": {"thread_id": tid or str(uuid.uuid4())}}


def _messages(state):
    return [(m.type, m.content) for m in state.values["messages"]]


def test_conversation_persists_and_matches_in_memory_saver(saver):
    django_graph = chat_builder.compile(checkpointer=saver)
    memory_graph = chat_builder.compile(checkpointer=InMemorySaver())
    c1, c2 = cfg(), cfg()
    for text in ["hello", "again"]:
        django_graph.invoke({"messages": [("user", text)]}, c1)
        memory_graph.invoke({"messages": [("user", text)]}, c2)

    assert (
        _messages(django_graph.get_state(c1))
        == _messages(memory_graph.get_state(c2))
        == [
            ("human", "hello"),
            ("ai", "echo: hello"),
            ("human", "again"),
            ("ai", "echo: again"),
        ]
    )
    django_hist = [s.metadata["step"] for s in django_graph.get_state_history(c1)]
    memory_hist = [s.metadata["step"] for s in memory_graph.get_state_history(c2)]
    assert django_hist == memory_hist

    # A fresh saver instance (e.g. after a restart) sees the same state.
    from llm_kit.checkpoint import DjangoSaver

    reloaded = chat_builder.compile(checkpointer=DjangoSaver())
    assert _messages(reloaded.get_state(c1))[-1] == ("ai", "echo: again")


def test_blobs_are_shared_between_checkpoints(saver):
    graph = chat_builder.compile(checkpointer=saver)
    c = cfg()
    graph.invoke({"messages": [("user", "hi")]}, c)
    n_checkpoints = Checkpoint.objects.filter(thread_id=c["configurable"]["thread_id"]).count()
    n_blobs = CheckpointBlob.objects.filter(thread_id=c["configurable"]["thread_id"]).count()
    assert n_checkpoints >= 3
    assert n_blobs < n_checkpoints * 3  # only changed channels get a new blob


def test_time_travel_to_older_checkpoint(saver):
    graph = chat_builder.compile(checkpointer=saver)
    c = cfg()
    graph.invoke({"messages": [("user", "one")]}, c)
    first_final = graph.get_state(c).config
    graph.invoke({"messages": [("user", "two")]}, c)

    old = graph.get_state(first_final)
    assert _messages(old)[-1] == ("ai", "echo: one")
    # Fork from the old checkpoint.
    graph.invoke({"messages": [("user", "branch")]}, first_final)
    assert [m for _, m in _messages(graph.get_state(c))] == ["one", "echo: one", "branch", "echo: branch"]


def test_interrupt_and_resume(saver):
    graph = make_approval_graph(saver)
    c = cfg()
    result = graph.invoke({"request": "deploy", "log": []}, c)
    assert result["__interrupt__"][0].value == {"question": "Approve deploy?"}
    state = graph.get_state(c)
    assert state.next == ("ask",)
    assert CheckpointWrite.objects.filter(thread_id=c["configurable"]["thread_id"], idx__lt=0).exists()

    result = graph.invoke(Command(resume=True), c)
    assert result["log"] == ["approved=True"]
    assert graph.get_state(c).next == ()


async def test_async_stream_and_resume(saver):
    graph = make_approval_graph(saver)
    c = cfg()
    chunks = [ch async for ch in graph.astream({"request": "x", "log": []}, c, stream_mode="updates")]
    assert "__interrupt__" in chunks[-1]
    final = await graph.ainvoke(Command(resume="yes"), c)
    assert final["log"] == ["approved=yes"]
    assert len([t async for t in saver.alist(c)]) >= 3


def test_list_filter_before_limit(saver):
    graph = chat_builder.compile(checkpointer=saver)
    c = cfg()
    graph.invoke({"messages": [("user", "a")]}, c)
    graph.invoke({"messages": [("user", "b")]}, c)

    all_ = list(saver.list(c))
    ids = [t.config["configurable"]["checkpoint_id"] for t in all_]
    assert ids == sorted(ids, reverse=True)
    assert [t.checkpoint["id"] for t in saver.list(c, limit=2)] == ids[:2]
    assert [t.checkpoint["id"] for t in saver.list(c, before=all_[1].config)] == ids[2:]

    inputs = list(saver.list(c, filter={"source": "input"}))
    assert inputs and all(t.metadata["source"] == "input" for t in inputs)
    assert len(list(saver.list(c, filter={"source": "input"}, limit=1))) == 1
    # Non-scalar filters are evaluated in Python.
    assert list(saver.list(c, filter={"parents": {}})) == list(saver.list(c))
    # No config: across threads.
    other = cfg()
    graph.invoke({"messages": [("user", "z")]}, other)
    threads = {t.config["configurable"]["thread_id"] for t in saver.list(None)}
    assert {c["configurable"]["thread_id"], other["configurable"]["thread_id"]} <= threads


def test_put_writes_regular_idempotent_special_overwrites(saver):
    c = cfg()
    tid = c["configurable"]["thread_id"]
    wc = {"configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": "cp1"}}
    saver.put_writes(wc, [("a", 1)], task_id="t1")
    saver.put_writes(wc, [("a", 2)], task_id="t1")
    saver.put_writes(wc, [("__error__", "first")], task_id="t1")
    saver.put_writes(wc, [("__error__", "second")], task_id="t1")
    rows = {
        (w.channel, w.idx): saver.serde.loads_typed((w.type, bytes(w.blob)))
        for w in CheckpointWrite.objects.filter(thread_id=tid)
    }
    assert rows == {("a", 0): 1, ("__error__", -1): "second"}


def test_delete_and_copy_thread(saver):
    graph = chat_builder.compile(checkpointer=saver)
    src, dst = cfg(), cfg()
    graph.invoke({"messages": [("user", "copy me")]}, src)
    saver.copy_thread(src["configurable"]["thread_id"], dst["configurable"]["thread_id"])
    assert _messages(graph.get_state(dst)) == _messages(graph.get_state(src))
    graph.invoke({"messages": [("user", "more")]}, dst)  # the copy continues independently
    assert len(graph.get_state(src).values["messages"]) == 2

    saver.delete_thread(src["configurable"]["thread_id"])
    for model in (Checkpoint, CheckpointBlob, CheckpointWrite):
        assert not model.objects.filter(thread_id=src["configurable"]["thread_id"]).exists()
    assert graph.get_state(src).values == {}
    assert len(graph.get_state(dst).values["messages"]) == 4
