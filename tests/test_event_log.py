"""Background runs' event log: every event numbered, kept, and replayed from any point."""

import asyncio

import pytest
from asgiref.sync import sync_to_async

from llm_kit import background
from llm_kit.background import log as event_log
from llm_kit.models import Thread
from tests.graphs import GATE
from tests.test_background import finish, wait_for
from tests.test_consumers import collect, communicator
from tests.test_views import read_events

pytestmark = pytest.mark.django_db(transaction=True)

WORDS = " ".join(f"w{i}" for i in range(40))


@pytest.fixture(autouse=True)
def gate():
    GATE.clear()
    yield GATE
    GATE.set()


async def _watch(run, after=None, slow=0.0):
    events = []
    async for event in background.watch(run, after=after):
        events.append(event)
        if slow:
            await asyncio.sleep(slow)
    return events


def _text(events):
    return "".join(e["data"]["message"]["content"] for e in events if e["event"] == "token")


def _gapless(events, first=1):
    seqs = [e["seq"] for e in events if "seq" in e]
    return seqs == list(range(first, first + len(seqs)))


async def test_watching_from_the_first_event_misses_nothing_even_after_the_run(user):
    thread = await Thread.objects.acreate(graph="chat", user=user)
    run = await background.astart(thread, {"message": "hi there"}, user=user)
    await sync_to_async(finish)(run)  # nobody watched while it ran

    events = await _watch(run, after=0)
    assert events[0]["event"] == "status" and events[0]["data"]["status"] == "running"
    assert events[-1]["event"] == "status" and events[-1]["data"]["status"] == "done"
    assert _gapless(events) and _text(events) == "echo: hi there"

    later = await _watch(run, after=3)  # a client that had seen 1-3
    assert later == events[3:]


async def test_a_snapshot_says_which_event_it_covers(user):
    thread = await Thread.objects.acreate(graph="chat", user=user)
    run = await background.astart(thread, {"message": "hi"}, user=user)
    await sync_to_async(finish)(run)
    events = await _watch(run)
    assert [e["event"] for e in events] == ["snapshot"]
    assert events[0]["data"]["seq"] == (await _watch(run, after=0))[-1]["seq"]


async def test_events_no_longer_kept_give_a_resync_snapshot(user):
    thread = await Thread.objects.acreate(graph="chat", user=user)
    run = await background.astart(thread, {"message": "hi"}, user=user)
    await sync_to_async(finish)(run)
    event_log.get_log().store.delete_many([f"llm_kit:runlog:{run.run_id}:0"])  # expired

    events = await _watch(run, after=0)
    assert events[0]["event"] == "snapshot" and events[0]["data"]["resync"] is True
    assert events[0]["data"]["state"]["values"]["messages"][-1]["content"] == "echo: hi"
    assert events[-1]["event"] == "status" and events[-1]["data"]["status"] == "done"


async def test_a_slow_watcher_gets_every_token_from_the_log(user, gate, monkeypatch):
    from llm_kit.background import bus, service

    monkeypatch.setattr(bus, "QUEUE_SIZE", 2)  # its queue overflows at once
    fills = []
    real_fill = service._fill

    async def fill(*args):
        fills.append(args[2:])
        return await real_fill(*args)

    monkeypatch.setattr(service, "_fill", fill)
    thread = await Thread.objects.acreate(graph="gated", user=user)
    run = await background.astart(thread, {"message": WORDS}, user=user)
    await sync_to_async(wait_for)(run, "running")
    watching = asyncio.create_task(_watch(run, after=0, slow=0.01))
    await asyncio.sleep(0.2)
    gate.set()
    events = await asyncio.wait_for(watching, 20)
    assert _gapless(events) and _text(events) == f"echo: {WORDS}"
    assert events[-1]["data"]["status"] == "done" and fills  # gaps were filled from the log


async def test_a_watcher_joining_mid_run_gets_the_tokens_it_missed(user, gate):
    thread = await Thread.objects.acreate(graph="gated", user=user)
    run = await background.astart(thread, {"message": WORDS}, user=user)
    await sync_to_async(wait_for)(run, "running")
    gate.set()
    await asyncio.sleep(0.05)  # tokens on their way, some not yet written to the log
    events = await asyncio.wait_for(_watch(run, after=0), 20)
    assert _gapless(events) and _text(events) == f"echo: {WORDS}"


async def test_the_cache_store_works_the_same(user, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EVENT_LOG": "cache"}
    assert isinstance(event_log.get_log().store, event_log.CacheStore)
    thread = await Thread.objects.acreate(graph="chat", user=user)
    run = await background.astart(thread, {"message": "hi"}, user=user)
    await sync_to_async(finish)(run)
    events = await _watch(run, after=0)
    assert _gapless(events) and _text(events) == "echo: hi" and events[-1]["data"]["status"] == "done"


async def test_with_the_log_off_watchers_get_live_events_without_seq(user, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EVENT_LOG": "off"}
    thread = await Thread.objects.acreate(graph="chat", user=user)
    run = await background.astart(thread, {"message": "hi"}, user=user)
    await sync_to_async(finish)(run)
    events = await _watch(run, after=0)
    assert [e["event"] for e in events] == ["snapshot"] and "seq" not in events[0]["data"]


def test_tokens_are_written_in_batches_and_the_log_is_capped():
    store = event_log.MemoryStore()
    log = event_log.EventLog(store, size=64, ttl=60)

    writes = []
    original = store.set_many
    store.set_many = lambda mapping, timeout: (writes.append(len(mapping)), original(mapping, timeout))
    writer = event_log.Writer(log, "r")
    for i in range(200):
        writer.add({"event": "token", "data": i})
    assert writes == [] and log.head("r") == 0  # nothing written yet

    asyncio.run(writer.flush())
    assert len(writes) == 1 and log.head("r") == 200
    events, complete = log.read("r", 150)
    assert complete and [e["seq"] for e in events] == list(range(151, 201))
    events, complete = log.read("r", 0)  # the first chunks are past the cap: gone
    assert not complete and events == []


def test_a_log_kept_in_memory_with_a_worker_executor_is_warned_about(settings):
    from llm_kit.background.checks import worker_events_are_kept_where_watchers_read as check

    assert check() == []  # the thread executor
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "celery"}
    assert [m.id for m in check()] == ["llm_kit.W003"]
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EVENT_LOG": "off"}
    assert check() == []
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EVENT_LOG": "disk"}
    assert [m.id for m in check()] == ["llm_kit.E002"]


# ------------------------------------------------------------------ transports


async def test_sse_watch_resumes_after_last_event_id(async_client, user):
    await async_client.aforce_login(user)
    thread = await Thread.objects.acreate(graph="chat", user=user)
    run = await background.astart(thread, {"message": "hi"}, user=user)
    await sync_to_async(finish)(run)

    response = await async_client.get(f"/llm/runs/{run.id}/watch/", headers={"Last-Event-ID": "2"})
    text = "".join([c.decode() async for c in response.streaming_content])
    assert text.startswith("id: 3\n")
    response = await async_client.get(f"/llm/runs/{run.id}/watch/?after=2")
    events = await read_events(response)
    assert events[-1][0] == "status" and events[-1][1]["status"] == "done"
    assert (await async_client.get(f"/llm/runs/{run.id}/watch/?after=x")).status_code == 400


async def test_websocket_watch_from_a_seq(user):
    thread = await Thread.objects.acreate(graph="chat", user=user)
    run = await background.astart(thread, {"message": "hi"}, user=user)
    await sync_to_async(finish)(run)
    c = communicator(f"/ws/llm/threads/{thread.id}/", user)
    assert (await c.connect())[0]
    await c.send_json_to({"type": "watch", "run": run.run_id, "after": 1})
    events = await collect(c, until=("status",))
    while events[-1]["data"]["status"] == "running":
        events += await collect(c, until=("status",))
    assert events[0]["seq"] == 2 and events[-1]["data"]["status"] == "done"
    await c.send_json_to({"type": "watch", "run": run.run_id, "after": -1})
    assert (await c.receive_json_from(timeout=5))["event"] == "error"
    await c.disconnect()


def test_an_event_sent_from_another_loop_goes_through_the_runs_loop():
    import threading

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    async def open_():
        event_log.open_writer("r-loops")

    try:
        asyncio.run_coroutine_threadsafe(open_(), loop).result(5)
        sent = asyncio.run(event_log.send("r-loops", {"event": "test.x", "data": 1}))  # a loop of its own
        assert sent["seq"] == 1
        asyncio.run_coroutine_threadsafe(event_log.close_writer("r-loops"), loop).result(5)
        events, complete = event_log.get_log().read("r-loops", 0)
        assert complete and [e["event"] for e in events] == ["test.x"]
    finally:
        loop.call_soon_threadsafe(loop.stop)
