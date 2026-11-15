"""Background runs: started detached from any request, watched (snapshot + live events), cancelled."""

import asyncio
import time
from datetime import timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from asgiref.sync import sync_to_async
from django.utils import timezone

from llm_kit import agents, background
from llm_kit.background import executors
from llm_kit.core.errors import BudgetExceeded
from llm_kit.models import BackgroundRun, Thread
from llm_kit.runtime import access
from tests.graphs import GATE, SCRIPTS

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def gate():
    GATE.clear()
    yield GATE
    GATE.set()  # let anything still waiting finish


def finish(run, timeout=10):
    assert executors.join(run.run_id, timeout)
    run.refresh_from_db()
    return run


def wait_for(run, status, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run.refresh_from_db()
        if run.status == status:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run is {run.status}, not {status}")


def noop(run_id):
    """An executor that never runs anything (a worker that is down)."""


def test_a_run_finishes_after_the_starter_has_moved_on(user):
    thread = Thread.objects.create(graph="chat", user=user)
    run = background.start(thread, {"message": "hi"}, user=user)
    assert run.status == "queued" and run.user == user and run.executor == "thread"
    run = finish(run)
    assert run.status == "done" and run.result == {"next": [], "interrupts": []}
    assert run.started_at and run.ended_at
    from llm_kit.runtime.graphs import get_graph

    state = get_graph("chat").get_state(thread.config)
    assert state.values["messages"][-1].content == "echo: hi"


async def _saved_input(thread, timeout=5):
    from llm_kit.runtime.graphs import get_graph

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = await get_graph(thread.graph).aget_state(thread.config)
        if state.values.get("messages"):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the run's input was never saved")


async def _collect(run, until_terminal=True):
    events = []
    async for event in background.watch(run):
        events.append(event)
    return events


async def test_watchers_get_a_snapshot_then_live_events(user, gate):
    thread = await Thread.objects.acreate(graph="gated", user=user)
    run = await background.astart(thread, {"message": "hey"}, user=user)
    await sync_to_async(wait_for)(run, "running")
    await _saved_input(thread)  # "running" is set just before the first checkpoint is written

    first, second = asyncio.create_task(_collect(run)), asyncio.create_task(_collect(run))
    await asyncio.sleep(0.2)  # both subscribed
    gate.set()
    a, b = await asyncio.wait_for(first, 10), await asyncio.wait_for(second, 10)

    assert a[0]["event"] == "snapshot" and a[0]["data"]["run"]["status"] == "running"
    assert [m["content"] for m in a[0]["data"]["state"]["values"]["messages"]] == ["hey"]
    text = "".join(e["data"]["message"]["content"] for e in a if e["event"] == "token")
    assert text == "echo: hey"
    assert a[-1]["event"] == "status" and a[-1]["data"]["status"] == "done"
    assert [e["event"] for e in a] == [e["event"] for e in b]


async def test_watching_a_finished_run_gives_its_snapshot(user):
    thread = await Thread.objects.acreate(graph="chat", user=user)
    run = await background.astart(thread, {"message": "hi"}, user=user)
    await sync_to_async(finish)(run)
    events = await _collect(run)
    assert [e["event"] for e in events] == ["snapshot"]
    assert events[0]["data"]["run"]["status"] == "done"
    assert events[0]["data"]["state"]["values"]["messages"][-1]["content"] == "echo: hi"


async def test_cancel_stops_a_running_run(user):
    thread = await Thread.objects.acreate(graph="slow", user=user)
    run = await background.astart(thread, {"message": "x"}, user=user)
    await sync_to_async(wait_for)(run, "running")
    watching = asyncio.create_task(_collect(run))
    await asyncio.sleep(0.2)
    assert await sync_to_async(run.cancel)()
    events = await asyncio.wait_for(watching, 10)
    assert events[-1]["event"] == "status" and events[-1]["data"]["status"] == "cancelled"
    assert not await sync_to_async(run.cancel)()  # already over


def test_a_cancel_from_another_process_is_noticed(user):
    thread = Thread.objects.create(graph="slow", user=user)
    run = wait_for(background.start(thread, {"message": "x"}, user=user), "running")
    BackgroundRun.objects.filter(pk=run.pk).update(cancel_requested=True)  # as another process would
    assert finish(run, timeout=5).status == "cancelled"


def test_a_run_no_worker_took_is_cancelled_at_once(user, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    thread = Thread.objects.create(graph="chat", user=user)
    run = background.start(thread, {"message": "x"}, user=user)
    assert run.status == "queued" and run.cancel() and run.status == "cancelled"
    background.execute(run.run_id)  # a worker taking it late does nothing
    run.refresh_from_db()
    assert run.status == "cancelled" and run.started_at is None


def test_one_run_at_a_time_per_thread(user, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    thread = Thread.objects.create(graph="chat", user=user)
    first = background.start(thread, {"message": "x"}, user=user)
    with pytest.raises(background.RunBusy) as e:
        background.start(thread, {"message": "y"}, user=user)
    assert e.value.run == first
    assert background.active_run(thread) == first
    first.cancel()
    assert background.active_run(thread) is None
    background.start(thread, {"message": "y"}, user=user)  # free again


def test_a_run_whose_heartbeat_stopped_is_lost(user, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    thread = Thread.objects.create(graph="chat", user=user)
    run = background.start(thread, {"message": "x"}, user=user)
    stale = timezone.now() - timedelta(minutes=5)
    BackgroundRun.objects.filter(pk=run.pk).update(status="running", heartbeat_at=stale)
    assert background.refresh(run).status == "lost"
    background.start(thread, {"message": "y"}, user=user)  # a lost run doesn't hold the thread


def test_an_interrupted_run_is_resumed_by_another(user):
    thread = Thread.objects.create(graph="approval", user=user)
    run = finish(background.start(thread, {"input": {"request": "ship", "log": []}}, user=user))
    assert run.status == "interrupted" and run.result["interrupts"][0]["value"] == {"question": "Approve ship?"}
    run = finish(background.start(thread, {"resume": "y"}, user=user))
    assert run.status == "done"


def test_a_failing_run_ends_in_error(user):
    thread = Thread.objects.create(graph="boom", user=user)
    run = finish(background.start(thread, {"message": "x"}, user=user))
    assert run.status == "error" and run.error


def test_a_caller_over_budget_is_refused_before_anything_starts(user, settings):
    from llm_kit.models import LLMCall

    settings.LLM_KIT = {**settings.LLM_KIT, "DAILY_BUDGETS": {"user": 1}}
    LLMCall.objects.create(call_id=uuid4(), started_at=timezone.now(), cost=Decimal("2"), user=str(user.pk))
    thread = Thread.objects.create(graph="chat", user=user)
    with pytest.raises(BudgetExceeded):
        background.start(thread, {"message": "x"}, user=user)
    assert not BackgroundRun.objects.exists()


def test_only_people_who_may_use_the_thread_see_its_runs(user, django_user_model, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    other = django_user_model.objects.create_user("mallory")
    thread = Thread.objects.create(graph="chat", user=user)
    run = background.start(thread, {"message": "x"}, user=user)
    assert background.get_run(run.pk, access.owner_for(user)) == run
    with pytest.raises(access.NotAllowed):
        background.get_run(run.pk, access.owner_for(other))
    assert list(background.runs_for(access.owner_for(user))) == [run]
    assert not background.runs_for(access.owner_for(other)).exists()


def test_the_payload_kept_has_nothing_the_server_sets(user, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    thread = Thread.objects.create(graph="chat", user=user)
    payload = {"message": "x", "config": {"configurable": {"__llm_kit_caller": "me", "k": 1}}, "junk": 1}
    run = background.start(thread, payload, user=user)
    assert run.payload == {"message": "x", "config": {"configurable": {"k": 1}}}


def test_an_agent_started_in_the_background(world):
    SCRIPTS["ollama/script-assistant"] = ["all done"]
    try:
        run = finish(agents.start("tests.assistant", "check A-1", user=world["alice"]))
    finally:
        SCRIPTS.clear()
    assert run.status == "done" and run.thread.graph == "tests.assistant"
    assert run.thread.user == world["alice"] and run.tenant == str(world["acme"].pk)
    from llm_kit.runtime.graphs import get_graph

    state = get_graph("tests.assistant").get_state(run.thread.config)
    assert state.values["messages"][-1].content == "all done"


async def test_the_channels_bus_carries_events():
    from channels.layers import InMemoryChannelLayer

    from llm_kit.background.bus import ChannelsBus

    bus = ChannelsBus(InMemoryChannelLayer())
    async with bus.subscribe("r-1") as queue:
        await bus.publish("r-1", {"event": "token", "data": {"x": 1}})
        await bus.publish("r-2", {"event": "token", "data": {"x": 2}})
        assert await asyncio.wait_for(queue.get(), 2) == {"event": "token", "data": {"x": 1}}
        assert queue.empty()


async def test_a_slow_watcher_misses_tokens_not_other_events(monkeypatch):
    from llm_kit.background import bus as bus_module

    monkeypatch.setattr(bus_module, "QUEUE_SIZE", 2)
    bus = bus_module.LocalBus()
    async with bus.subscribe("r") as queue:
        for i in range(3):
            await bus.publish("r", {"event": "token", "data": i})
        await bus.publish("r", {"event": "status", "data": "done"})
        await asyncio.sleep(0.05)
        got = [queue.get_nowait() for _ in range(queue.qsize())]
    assert got[-1] == {"event": "status", "data": "done"} and len(got) == 2


def test_a_worker_executor_without_a_shared_layer_is_warned_about(settings):
    from llm_kit.background.checks import worker_events_reach_watchers

    assert worker_events_reach_watchers() == []  # the thread executor
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "celery"}
    assert [m.id for m in worker_events_reach_watchers()] == ["llm_kit.W002"]  # in-memory layer: local
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EVENTS": "nope"}
    assert [m.id for m in worker_events_reach_watchers()] == ["llm_kit.E001"]


def test_prune_deletes_old_finished_runs_only(user, settings):
    from django.core.management import call_command

    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    thread = Thread.objects.create(graph="chat", user=user)
    queued = background.start(thread, {"message": "x"}, user=user)
    old = timezone.now() - timedelta(days=400)
    done = BackgroundRun.objects.create(thread=thread, graph="chat", status="done")
    BackgroundRun.objects.filter(pk__in=[queued.pk, done.pk]).update(created_at=old)
    call_command("llm_kit_prune")
    assert list(BackgroundRun.objects.all()) == [queued]


def test_a_celery_worker_runs_it(user, settings):
    pytest.importorskip("celery")
    from llm_kit.background.tasks import run_in_background_task

    calls = []
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "celery"}
    original = run_in_background_task.delay
    run_in_background_task.delay = lambda run_id: calls.append(run_id) or run_in_background_task(run_id)
    try:
        thread = Thread.objects.create(graph="chat", user=user)
        run = background.start(thread, {"message": "hi"}, user=user)
    finally:
        run_in_background_task.delay = original
    run.refresh_from_db()
    assert calls == [run.run_id] and run.status == "done" and run.executor == "celery"
