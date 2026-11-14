"""Queued follow-ups: a run started on a busy thread with queue=True waits its turn."""

import asyncio
from datetime import timedelta

import pytest
from asgiref.sync import sync_to_async
from django.test import Client
from django.utils import timezone

from llm_kit import agents, background
from llm_kit.models import BackgroundRun, Thread
from llm_kit.runtime.graphs import get_graph
from tests.graphs import GATE, SCRIPTS
from tests.test_background import finish, wait_for

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def gate():
    GATE.clear()
    yield GATE
    GATE.set()


@pytest.fixture
def no_worker(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}


def _messages(thread):
    return [m.content for m in get_graph(thread.graph).get_state(thread.config).values["messages"]]


def _drain(thread, timeout=10):
    """Wait until nothing on the thread is left to run."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending = list(BackgroundRun.objects.filter(thread=thread, status__in=("waiting", "queued", "running")))
        if not pending:
            return
        for run in pending:
            finish(run, timeout=1) if run.status == "running" else time.sleep(0.02)
    raise AssertionError("runs still pending")


def test_follow_ups_wait_their_turn_and_run_in_order(user, gate):
    thread = Thread.objects.create(graph="gated", user=user)
    first = wait_for(background.start(thread, {"message": "one"}, user=user), "running")
    second = background.start(thread, {"message": "two"}, user=user, queue=True)
    third = background.start(thread, {"message": "three"}, user=user, queue=True)
    assert second.status == third.status == "waiting"
    assert list(background.waiting_runs(thread)) == [second, third]
    with pytest.raises(background.RunBusy):
        background.start(thread, {"message": "four"}, user=user)  # without queue: refused as before

    gate.set()
    _drain(thread)
    runs = {r.pk: r for r in BackgroundRun.objects.filter(thread=thread)}
    assert [runs[r.pk].status for r in (first, second, third)] == ["done"] * 3
    assert runs[first.pk].ended_at <= runs[second.pk].started_at
    assert runs[second.pk].ended_at <= runs[third.pk].started_at
    assert _messages(thread) == ["one", "echo: one", "two", "echo: two", "three", "echo: three"]


async def test_the_snapshot_shows_the_queue_and_a_waiting_run_can_be_watched(user, gate):
    thread = await Thread.objects.acreate(graph="gated", user=user)
    first = await background.astart(thread, {"message": "one"}, user=user)
    await sync_to_async(wait_for)(first, "running")
    second = await background.astart(thread, {"message": "two"}, user=user, queue=True)

    watching = asyncio.create_task(_collect(second))
    await asyncio.sleep(0.2)
    gate.set()
    events = await asyncio.wait_for(watching, 15)
    snapshot = events[0]["data"]
    assert snapshot["run"]["status"] == "waiting" and [r["id"] for r in snapshot["queue"]] == [second.run_id]
    assert "".join(e["data"]["message"]["content"] for e in events if e["event"] == "token") == "echo: two"
    assert events[-1]["event"] == "status" and events[-1]["data"]["status"] == "done"


async def _collect(run):
    return [event async for event in background.watch(run)]


def test_a_waiting_run_can_be_cancelled(user, gate):
    thread = Thread.objects.create(graph="gated", user=user)
    wait_for(background.start(thread, {"message": "one"}, user=user), "running")
    dropped = background.start(thread, {"message": "two"}, user=user, queue=True)
    kept = background.start(thread, {"message": "three"}, user=user, queue=True)
    assert dropped.cancel() and dropped.status == "cancelled"
    gate.set()
    _drain(thread)
    kept.refresh_from_db()
    assert kept.status == "done" and _messages(thread) == ["one", "echo: one", "three", "echo: three"]


def test_a_lost_run_does_not_strand_the_queue(user, no_worker):
    thread = Thread.objects.create(graph="chat", user=user)
    front = background.start(thread, {"message": "one"}, user=user)
    behind = background.start(thread, {"message": "two"}, user=user, queue=True)
    stale = timezone.now() - timedelta(minutes=5)
    BackgroundRun.objects.filter(pk=front.pk).update(status="running", heartbeat_at=stale)
    assert background.refresh(front).status == "lost"
    behind.refresh_from_db()
    assert behind.status == "queued"  # handed to the executor


def test_a_cancelled_run_does_not_strand_the_queue(user, no_worker):
    thread = Thread.objects.create(graph="chat", user=user)
    front = background.start(thread, {"message": "one"}, user=user)
    behind = background.start(thread, {"message": "two"}, user=user, queue=True)
    assert front.cancel()
    behind.refresh_from_db()
    assert behind.status == "queued"


def test_the_queue_waits_for_an_interrupt_to_be_answered(user, no_worker):
    thread = Thread.objects.create(graph="approval", user=user)
    asking = background.start(thread, {"input": {"request": "ship", "log": []}}, user=user)
    behind = background.start(thread, {"input": {"request": "next", "log": []}}, user=user, queue=True)
    background.execute(asking.run_id)
    asking.refresh_from_db()
    behind.refresh_from_db()
    assert asking.status == "interrupted" and behind.status == "waiting"

    answer = background.start(thread, {"resume": "y"}, user=user, queue=True)  # a resume never waits
    assert answer.status == "queued"
    background.execute(answer.run_id)
    behind.refresh_from_db()
    assert behind.status == "queued"


def test_the_setting_queues_by_default(user, no_worker, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_QUEUE": True}
    thread = Thread.objects.create(graph="chat", user=user)
    background.start(thread, {"message": "one"}, user=user)
    assert background.start(thread, {"message": "two"}, user=user).status == "waiting"
    with pytest.raises(background.RunBusy):
        background.start(thread, {"message": "three"}, user=user, queue=False)


def test_agents_start_can_queue(world, no_worker):
    thread = Thread.objects.create(graph="tests.assistant", user=world["alice"])
    agents.start("tests.assistant", "one", thread=thread)
    assert agents.start("tests.assistant", "two", thread=thread, queue=True).status == "waiting"
    SCRIPTS.clear()


def test_rest_queues_and_lists_waiting_runs(user, no_worker):
    client = Client()
    client.force_login(user)
    thread = Thread.objects.create(graph="chat", user=user)
    url = f"/llm/api/threads/{thread.id}/runs/"
    assert client.post(url, {"message": "one", "background": True}, "application/json").status_code == 202
    r = client.post(url, {"message": "two", "queue": True}, "application/json")
    assert r.status_code == 202 and r.json()["status"] == "waiting"
    assert [x["id"] for x in client.get("/llm/api/runs/?status=waiting").json()] == [r.json()["id"]]
    assert client.post(url, {"message": "three"}, "application/json").status_code == 409


def test_a_foreground_run_moves_the_queue_on(user, no_worker):
    """A run made outside the background (a plain REST run) also lets the next waiting one start."""
    thread = Thread.objects.create(graph="chat", user=user)
    front = background.start(thread, {"message": "one"}, user=user)
    behind = background.start(thread, {"message": "two"}, user=user, queue=True)
    BackgroundRun.objects.filter(pk=front.pk).update(status="done")  # finished without telling anyone
    client = Client()
    client.force_login(user)
    r = client.post(f"/llm/api/threads/{thread.id}/runs/", {"message": "now"}, "application/json")
    assert r.status_code == 200  # nothing was running, so it ran
    behind.refresh_from_db()
    assert behind.status == "queued"  # and the waiting run started after it


async def test_websocket_and_sse_queue(async_client, user, no_worker):
    from tests.test_consumers import communicator
    from tests.test_views import post

    thread = await Thread.objects.acreate(graph="chat", user=user)
    await background.astart(thread, {"message": "one"}, user=user)
    c = communicator(f"/ws/llm/threads/{thread.id}/", user)
    assert (await c.connect())[0]
    await c.send_json_to({"type": "run", "message": "two", "queue": True})
    run = await c.receive_json_from(timeout=10)
    assert run["event"] == "run" and run["data"]["status"] == "waiting"
    await c.disconnect()

    await async_client.aforce_login(user)
    r = await post(async_client, f"/llm/threads/{thread.id}/stream/", {"message": "three", "queue": True})
    first = (await r.streaming_content.__aiter__().__anext__()).decode()
    assert first.startswith("event: run") and '"status": "waiting"' in first
