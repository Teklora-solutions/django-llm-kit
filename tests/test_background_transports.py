"""Background runs over REST, SSE and WebSocket."""

import asyncio
import json

import pytest
from asgiref.sync import sync_to_async
from django.test import Client

from llm_kit.background import executors
from llm_kit.models import BackgroundRun, Thread
from tests.graphs import GATE
from tests.test_background import finish, wait_for
from tests.test_consumers import collect, communicator
from tests.test_views import post, read_events

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def gate():
    GATE.clear()
    yield GATE
    GATE.set()


@pytest.fixture
def mallory(django_user_model):
    return django_user_model.objects.create_user("mallory")


# ------------------------------------------------------------------ REST


def test_rest_starts_lists_shows_and_cancels(user, mallory):
    client = Client()
    client.force_login(user)
    thread = Thread.objects.create(graph="slow", user=user)
    r = client.post(f"/llm/api/threads/{thread.id}/runs/", {"message": "x", "background": True}, "application/json")
    assert r.status_code == 202 and r.json()["status"] == "queued"
    run_id = r.json()["id"]
    run = wait_for(BackgroundRun.objects.get(pk=run_id), "running")

    r = client.post(f"/llm/api/threads/{thread.id}/runs/", {"message": "y"}, "application/json")
    assert r.status_code == 409 and r.json()["run"]["id"] == run_id  # not while it runs
    r = client.post(f"/llm/api/threads/{thread.id}/runs/", {"message": "y", "background": True}, "application/json")
    assert r.status_code == 409

    assert [x["id"] for x in client.get("/llm/api/runs/?status=running").json()] == [run_id]
    assert client.get(f"/llm/api/runs/?thread={thread.id}").json()[0]["id"] == run_id
    assert client.get("/llm/api/runs/?status=done").json() == []
    assert client.get("/llm/api/runs/?thread=nope").status_code == 400
    assert client.get(f"/llm/api/runs/{run_id}/").json()["status"] == "running"

    other = Client()
    other.force_login(mallory)
    assert other.get(f"/llm/api/runs/{run_id}/").status_code == 404
    assert other.post(f"/llm/api/runs/{run_id}/cancel/").status_code == 404
    assert other.get("/llm/api/runs/").json() == []

    r = client.post(f"/llm/api/runs/{run_id}/cancel/")
    assert r.status_code == 202 and r.json()["cancelled"] is True
    assert finish(run).status == "cancelled"
    assert client.post(f"/llm/api/runs/{run_id}/cancel/").status_code == 409


def test_rest_refuses_a_caller_over_budget(user, settings):
    from decimal import Decimal
    from uuid import uuid4

    from django.utils import timezone

    from llm_kit.models import LLMCall

    settings.LLM_KIT = {**settings.LLM_KIT, "DAILY_BUDGETS": {"user": 1}}
    LLMCall.objects.create(call_id=uuid4(), started_at=timezone.now(), cost=Decimal("2"), user=str(user.pk))
    client = Client()
    client.force_login(user)
    thread = Thread.objects.create(graph="chat", user=user)
    r = client.post(f"/llm/api/threads/{thread.id}/runs/", {"message": "x", "background": True}, "application/json")
    assert r.status_code == 429 and not BackgroundRun.objects.exists()


# ------------------------------------------------------------------ SSE


async def test_sse_background_run_carries_on_after_the_stream_closes(async_client, user, gate):
    await async_client.aforce_login(user)
    response = await post(async_client, "/llm/stream/", {"message": "hey", "graph": "gated", "background": True})
    stream = response.streaming_content.__aiter__()
    first = [await stream.__anext__() for _ in range(3)]  # thread, run, snapshot
    names = [chunk.decode().split("\n")[0] for chunk in first]
    assert names == ["event: thread", "event: run", "event: snapshot"]
    await stream.aclose()  # the visitor leaves

    run = await BackgroundRun.objects.aget()
    gate.set()
    run = await sync_to_async(finish)(run)
    assert run.status == "done"

    events = await read_events(await async_client.get(f"/llm/runs/{run.id}/watch/"))
    assert [e for e, _ in events] == ["snapshot"]
    assert events[0][1]["state"]["values"]["messages"][-1]["content"] == "echo: hey"


async def test_sse_watch_follows_a_running_run(async_client, user, gate):
    await async_client.aforce_login(user)
    thread = await Thread.objects.acreate(graph="gated", user=user)
    from llm_kit import background

    run = await background.astart(thread, {"message": "hi"}, user=user)
    await sync_to_async(wait_for)(run, "running")

    async def release():
        await asyncio.sleep(0.3)
        gate.set()

    releasing = asyncio.create_task(release())
    events = await read_events(await async_client.get(f"/llm/runs/{run.id}/watch/"))
    await releasing
    names = [e for e, _ in events]
    assert names[0] == "snapshot" and names[-1] == "status" and events[-1][1]["status"] == "done"
    assert "".join(d["message"]["content"] for e, d in events if e == "token") == "echo: hi"


async def test_sse_watch_is_only_for_the_threads_owner(async_client, user, mallory, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    thread = await Thread.objects.acreate(graph="chat", user=user)
    from llm_kit import background

    run = await background.astart(thread, {"message": "hi"}, user=user)
    await async_client.aforce_login(mallory)
    assert (await async_client.get(f"/llm/runs/{run.id}/watch/")).status_code == 404
    r = await post(async_client, f"/llm/threads/{thread.id}/stream/", {"message": "x"})
    assert r.status_code == 403


async def test_sse_refuses_a_second_run_on_a_busy_thread(async_client, user, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    await async_client.aforce_login(user)
    thread = await Thread.objects.acreate(graph="chat", user=user)
    r = await post(async_client, f"/llm/threads/{thread.id}/stream/", {"message": "x", "background": True})
    await r.streaming_content.__aiter__().__anext__()
    r = await post(async_client, f"/llm/threads/{thread.id}/stream/", {"message": "y"})
    assert r.status_code == 409 and json.loads(r.content)["run"]["status"] == "queued"


# ------------------------------------------------------------------ WebSocket


async def test_websocket_background_run_survives_a_disconnect(user, gate):
    c = communicator("/ws/llm/threads/", user)
    assert (await c.connect())[0]
    await c.send_json_to({"type": "run", "message": "hey", "graph": "gated", "background": True})
    events = [await c.receive_json_from(timeout=10) for _ in range(3)]
    assert [e["event"] for e in events] == ["thread", "run", "snapshot"]
    run_id = events[1]["data"]["id"]
    await c.disconnect()

    run = await BackgroundRun.objects.aget(pk=run_id)
    assert run.status == "running"  # still going

    thread_id = events[0]["data"]["id"]
    c = communicator(f"/ws/llm/threads/{thread_id}/", user)
    assert (await c.connect())[0]
    await c.send_json_to({"type": "watch", "run": run_id})
    snapshot = await c.receive_json_from(timeout=10)
    assert snapshot["event"] == "snapshot" and snapshot["data"]["run"]["status"] == "running"
    gate.set()
    events = await collect(c, until=("status",))
    while events[-1]["data"]["status"] == "running":
        events += await collect(c, until=("status",))
    assert events[-1]["data"]["status"] == "done"
    assert "".join(e["data"]["message"]["content"] for e in events if e["event"] == "token") == "echo: hey"
    await c.disconnect()


async def test_websocket_cancel_stops_the_watched_run(user):
    thread = await Thread.objects.acreate(graph="slow", user=user)
    c = communicator(f"/ws/llm/threads/{thread.id}/", user)
    assert (await c.connect())[0]
    await c.send_json_to({"type": "run", "message": "x", "background": True})
    assert (await c.receive_json_from(timeout=10))["event"] == "run"
    assert (await c.receive_json_from(timeout=10))["event"] == "snapshot"
    run = await BackgroundRun.objects.aget()
    await sync_to_async(wait_for)(run, "running")
    await c.send_json_to({"type": "cancel"})
    events = await collect(c, until=("status",))
    while events[-1]["data"]["status"] not in ("cancelled",):
        events += await collect(c, until=("status",))
    assert await sync_to_async(executors.join)(run.run_id, 5)
    await c.disconnect()


async def test_websocket_watch_refuses_other_peoples_runs(user, mallory, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    thread = await Thread.objects.acreate(graph="chat", user=user)
    from llm_kit import background

    run = await background.astart(thread, {"message": "hi"}, user=user)
    c = communicator("/ws/llm/threads/", mallory)
    assert (await c.connect())[0]
    for bad in (str(run.id), "not-an-id"):
        await c.send_json_to({"type": "watch", "run": bad})
        assert await c.receive_json_from(timeout=10) == {"event": "error", "data": {"message": "Run not found."}}
    await c.disconnect()


async def test_a_client_leaving_mid_wait_closes_the_stream_cleanly():
    from llm_kit.transport.sse.views import with_keepalive

    closed = asyncio.Event()

    async def events():
        try:
            yield {"event": "run", "data": {}}
            await asyncio.sleep(30)  # a quiet run
            yield {"event": "never", "data": {}}
        finally:
            closed.set()

    stream = with_keepalive(events(), 0)

    async def read():
        async for _ in stream:
            pass

    reading = asyncio.create_task(read())
    await asyncio.sleep(0.05)
    reading.cancel()  # what the server does when the client disconnects
    with pytest.raises(asyncio.CancelledError):
        await reading
    assert closed.is_set()  # the inner generator was closed, not left "already running"
