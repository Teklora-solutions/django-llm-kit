"""Start, run, cancel and watch background runs.

A background run is a run of a thread's graph that does not belong to the request that
started it. It keeps going when that request ends; anyone allowed to use the thread can
watch it (a snapshot of where it is, then its events live) or cancel it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from asgiref.sync import sync_to_async
from django.db import IntegrityError, transaction
from django.utils import timezone

from llm_kit.models import BackgroundRun, Thread
from llm_kit.models.runs import ACTIVE, FINISHED, WAITING
from llm_kit.runtime import access, runs

logger = logging.getLogger("llm_kit")

RUN_KEY = "__llm_kit_background_run"  # configurable: the background run a graph is running in
POLL = 1.0  # seconds between checks for a cancel asked for elsewhere
HEARTBEAT = 10.0  # seconds between heartbeats (RUN_LOST_AFTER must be well above it)
WATCH_CHECK = 5.0  # seconds a quiet watcher waits before checking the run is still going

_PAYLOAD_KEYS = ("message", "input", "resume", "model")


class RunBusy(Exception):
    """The thread already has a queued or running background run."""

    def __init__(self, run: BackgroundRun | None = None):
        self.run = run
        super().__init__("This thread already has a run in progress.")


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """What is kept to run later: the run payload without anything the server sets."""
    out = {k: payload[k] for k in _PAYLOAD_KEYS if k in payload}
    extra = payload.get("config")
    if isinstance(extra, dict):
        config: dict[str, Any] = {}
        configurable = {k: v for k, v in dict(extra.get("configurable") or {}).items() if not str(k).startswith("__")}
        if configurable:
            config["configurable"] = configurable
        if "recursion_limit" in extra:
            config["recursion_limit"] = extra["recursion_limit"]
        if config:
            out["config"] = config
    return out


def _owner(owner: Any = None, *, user=None, tenant: Any = None) -> access.Owner:
    if isinstance(owner, access.Owner):
        return owner
    return access.owner_for(user, tenant=tenant)


# ------------------------------------------------------------------ starting


def active_run(thread: Thread) -> BackgroundRun | None:
    """The thread's queued or running background run (a lost one is marked so first)."""
    for run in BackgroundRun.objects.filter(thread=thread, status__in=ACTIVE):
        if not _is_lost(run):
            return run
        _mark_lost(run)
    return None


def waiting_runs(thread: Thread):
    """The runs waiting their turn on ``thread``, in the order they will start."""
    return BackgroundRun.objects.filter(thread=thread, status=WAITING).order_by("created_at")


def start(
    thread: Thread,
    payload: dict[str, Any],
    owner: Any = None,
    *,
    user=None,
    tenant: Any = None,
    queue: bool | None = None,
    links: dict[str, Any] | None = None,
) -> BackgroundRun:
    """Start a run of ``thread``'s graph in the background and return it at once.

    ``payload`` is the usual run payload (``message``, ``input``, ``resume``, ``model``,
    ``config``). The run acts for ``owner`` (an ``access.Owner``) or ``user``/``tenant``.
    A bad payload (``runs.BadRequest``) or a caller over budget (``BudgetExceeded``) is
    refused here, before anything starts.

    A thread runs one run at a time. When it is busy, the run is refused (``RunBusy``), or
    with ``queue=True`` (default ``LLM_KIT["RUN_QUEUE"]``) saved as ``waiting`` and started
    when the runs before it have finished, in order. A ``resume`` never waits: it answers the
    interrupt the thread is stopped at, and a queue doesn't move past an unanswered interrupt.

    ``links``: ``BackgroundRun`` fields linking a background subagent to the run that started it
    (set by the harness; see ``llm_kit.background.subagents``).
    """
    from llm_kit import conf

    owner = _owner(owner, user=user, tenant=tenant)
    payload = dict(payload or {})
    if queue is None:
        queue = bool(conf.get("RUN_QUEUE"))
    if "resume" in payload:
        queue = False
    runs.validate_choice(thread.graph, payload.get("model"))
    runs.build_input(payload)
    runs.build_config(thread, payload, owner=owner)  # budgets, the config's shape
    executor = str(conf.get("RUN_EXECUTOR"))
    from llm_kit.background import executors

    submit = executors.get_executor(executor)  # an unknown executor fails before the row exists
    fields = {
        "thread": thread,
        "graph": thread.graph,
        "payload": _clean_payload(payload),
        "user": owner.user if owner.is_authenticated else None,
        "session_key": "" if owner.is_authenticated else owner.visitor_id,
        "tenant": owner.tenant_id,
        "executor": executor,
        **(links or {}),
    }
    busy = active_run(thread)
    if queue and (busy is not None or waiting_runs(thread).exists()):
        return _enqueue(fields)
    if busy is not None:
        raise RunBusy(busy)
    try:
        with transaction.atomic():
            run = BackgroundRun.objects.create(**fields)
    except IntegrityError:  # another run started in between
        if queue:
            return _enqueue(fields)
        raise RunBusy(active_run(thread)) from None
    transaction.on_commit(lambda: submit(run.run_id))
    return run


def _enqueue(fields: dict) -> BackgroundRun:
    run = BackgroundRun.objects.create(**fields, status=WAITING)
    thread_id = run.thread_id
    # The run in front may have finished while this was saved: then nothing would move the queue.
    transaction.on_commit(lambda: advance(thread_id))
    return run


def advance(thread) -> BackgroundRun | None:
    """Start the thread's next waiting run if nothing runs there (and it isn't stopped at an
    interrupt). Called whenever a run finishes; call it yourself after a run you made without
    ``llm_kit.background`` (the views do). Returns the run started, if any."""
    from llm_kit import conf
    from llm_kit.background import executors

    thread_id = getattr(thread, "pk", thread)
    nxt = BackgroundRun.objects.filter(thread_id=thread_id, status=WAITING).order_by("created_at").first()
    if nxt is None:
        return None
    thread = Thread.objects.get(pk=thread_id)
    if active_run(thread) is not None or _at_interrupt(thread):
        return None
    executor = str(conf.get("RUN_EXECUTOR"))
    try:
        with transaction.atomic():
            moved = BackgroundRun.objects.filter(pk=nxt.pk, status=WAITING).update(status="queued", executor=executor)
    except IntegrityError:  # something else started on the thread first
        return None
    if not moved:  # cancelled, or moved by someone else
        return None
    nxt.status, nxt.executor = "queued", executor
    submit = executors.get_executor(executor)
    transaction.on_commit(lambda: submit(nxt.run_id))
    return nxt


def _at_interrupt(thread: Thread) -> bool:
    try:
        from llm_kit.runtime.graphs import get_graph

        return bool(get_graph(thread.graph).get_state(thread.config).interrupts)
    except Exception:
        logger.warning("Could not read the state of thread %s", thread.pk, exc_info=True)
        return False


def _finished(run: BackgroundRun) -> None:
    """After this process moved ``run`` to a final status (only the one whose change took
    effect calls this): ``run_finished``, and the next waiting run on its thread may start."""
    transaction.on_commit(lambda: _announce(run.pk))
    try:
        advance(run.thread_id)
    except Exception:
        logger.exception("Could not start the next run on thread %s", run.thread_id)


def _announce(run_pk) -> None:
    from llm_kit.background.signals import run_finished

    run = BackgroundRun.objects.select_related("thread").get(pk=run_pk)
    for receiver, response in run_finished.send_robust(
        sender=BackgroundRun, run=run, status=run.status, result=run.result, error=run.error
    ):
        if isinstance(response, Exception):
            logger.error("run_finished receiver %r failed for run %s", receiver, run.run_id, exc_info=response)


async def astart(thread: Thread, payload: dict[str, Any], owner: Any = None, **kw) -> BackgroundRun:
    return await sync_to_async(start)(thread, payload, owner, **kw)


async def aadvance(thread) -> BackgroundRun | None:
    return await sync_to_async(advance)(thread)


# ------------------------------------------------------------------ running


def execute(run_id: str) -> None:
    """Run a queued background run to its end (what every executor calls)."""
    asyncio.run(_execute_in_own_context(run_id))


async def _execute_in_own_context(run_id: str) -> None:
    """The run's database work gets a thread of its own: outside a request, asgiref would queue
    it behind every other piece of sync code in the process on one shared thread."""
    from asgiref.sync import ThreadSensitiveContext
    from django.db import connections

    async with ThreadSensitiveContext():
        try:
            await aexecute(run_id)
        finally:
            await sync_to_async(connections.close_all)()  # that thread's connections


async def aexecute(run_id: str) -> None:
    from llm_kit.background import executors
    from llm_kit.background.bus import get_bus

    now = timezone.now()
    claimed = await BackgroundRun.objects.filter(pk=run_id, status="queued").aupdate(
        status="running", started_at=now, heartbeat_at=now
    )
    if not claimed:  # already taken, finished or cancelled (a task delivered twice)
        return
    run = await BackgroundRun.objects.select_related("thread", "user").aget(pk=run_id)
    bus = get_bus()
    await bus.publish(run.run_id, _status_event(run))
    task = asyncio.current_task()
    executors.register_local(run.run_id, asyncio.get_running_loop(), task)
    watchdog = asyncio.create_task(_watchdog(run.run_id, task))
    status, result, error = "done", None, ""
    try:
        if run.cancel_requested:
            raise asyncio.CancelledError
        from llm_kit.runtime.graphs import get_graph

        owner = access.Owner(user=run.user, tenant=run.tenant or None)
        config = await sync_to_async(runs.build_config)(run.thread, run.payload, owner=owner)
        config["run_id"] = run.id  # the recorded LLMRun has the same id
        config["configurable"][RUN_KEY] = run.run_id
        meta = config.setdefault("metadata", {})
        if "llm_kit_agent" in meta:
            meta["llm_kit_agent_run"] = run.run_id
        if run.agents:
            from llm_kit.background.subagents import inherit_limits

            inherit_limits(run, config)
        graph = get_graph(run.graph)
        async for event in runs.astream_run(graph, runs.build_input(run.payload), config):
            if event["event"] == "end":
                result = event["data"]
                status = "interrupted" if result.get("interrupts") else "done"
            elif event["event"] == "error":
                status, error = "error", str(event["data"].get("message", ""))[:500]
            await bus.publish(run.run_id, event)
    except asyncio.CancelledError:
        status = "cancelled"
        if task is not None:
            task.uncancel()
    except Exception as exc:
        logger.exception("llm_kit background run %s failed", run.run_id)
        status, error = "error", runs.error_message(exc)[:500]
        await bus.publish(run.run_id, {"event": "error", "data": {"message": error}})
    finally:
        watchdog.cancel()
        executors.unregister_local(run.run_id)
    # Only if still running: a run marked lost meanwhile (heartbeats delayed) stays lost, and was
    # announced as lost already.
    ended = await BackgroundRun.objects.filter(pk=run.pk, status="running").aupdate(
        status=status, ended_at=timezone.now(), result=result, error=error
    )
    await Thread.objects.filter(pk=run.thread_id).aupdate(updated_at=timezone.now())
    run = await BackgroundRun.objects.aget(pk=run.pk)
    await bus.publish(run.run_id, _status_event(run))
    if ended:
        await sync_to_async(_finished)(run)
    else:
        logger.warning("Background run %s finished %s after it was marked %s", run.run_id, status, run.status)


async def _watchdog(run_id: str, task: asyncio.Task | None) -> None:
    """Heartbeats, and a cancel asked for from another process."""
    beat = 0.0
    while True:
        await asyncio.sleep(POLL)
        beat += POLL
        wanted = await BackgroundRun.objects.filter(pk=run_id).values_list("cancel_requested", flat=True).afirst()
        if wanted is None or wanted:
            if task is not None:
                task.cancel()
            return
        if beat >= HEARTBEAT:
            beat = 0.0
            await BackgroundRun.objects.filter(pk=run_id).aupdate(heartbeat_at=timezone.now())


def _status_event(run: BackgroundRun) -> dict:
    return {"event": "status", "data": run.to_json()}


# ------------------------------------------------------------------ cancelling, status


def cancel(run: BackgroundRun) -> bool:
    """Ask ``run`` to stop (or, waiting, not to start). False when it had already finished."""
    from llm_kit.background import executors

    if BackgroundRun.objects.filter(pk=run.pk, status=WAITING).update(
        status="cancelled", cancel_requested=True, ended_at=timezone.now()
    ):
        run.status, run.cancel_requested = "cancelled", True
        _finished(run)
        return True
    # Not picked up yet: it ends here (a worker that takes it later finds it cancelled).
    if BackgroundRun.objects.filter(pk=run.pk, status="queued").update(
        status="cancelled", cancel_requested=True, ended_at=timezone.now()
    ):
        run.status, run.cancel_requested = "cancelled", True
        _finished(run)
        return True
    updated = BackgroundRun.objects.filter(pk=run.pk, status="running").update(cancel_requested=True)
    if updated:
        run.cancel_requested = True
        executors.cancel_local(run.run_id)
    return bool(updated)


def _is_lost(run: BackgroundRun) -> bool:
    from llm_kit import conf

    if run.status != "running" or run.heartbeat_at is None:
        return False
    return timezone.now() - run.heartbeat_at > timedelta(seconds=float(conf.get("RUN_LOST_AFTER")))


def _mark_lost(run: BackgroundRun) -> None:
    marked = BackgroundRun.objects.filter(pk=run.pk, status="running").update(status="lost", ended_at=timezone.now())
    run.status = "lost"
    if marked:
        _finished(run)


def refresh(run: BackgroundRun) -> BackgroundRun:
    """``run`` as it is now; a running run whose heartbeat stopped is marked ``lost``."""
    run.refresh_from_db()
    if _is_lost(run):
        _mark_lost(run)
    return run


def get_run(run_id, who) -> BackgroundRun:
    """A background run the requester may see (they may use its thread).

    Raises ``BackgroundRun.DoesNotExist`` or ``access.NotAllowed``."""
    run = BackgroundRun.objects.select_related("thread").get(pk=run_id)
    access.check_thread(run.thread, who)
    return refresh(run)


def runs_for(who):
    """Background runs on the threads the requester may list, newest first."""
    return BackgroundRun.objects.filter(thread__in=access.threads_for(who))


# ------------------------------------------------------------------ watching


async def _snapshot(run: BackgroundRun) -> dict:
    state = None
    try:
        from llm_kit.runtime.graphs import get_graph

        thread = await Thread.objects.aget(pk=run.thread_id)
        state = runs.state_to_json(await get_graph(run.graph).aget_state(thread.config))
    except Exception:
        logger.warning("Could not read the state of thread %s", run.thread_id, exc_info=True)
    queue = [
        r.to_json()
        async for r in BackgroundRun.objects.filter(thread_id=run.thread_id, status=WAITING).order_by("created_at")
    ]
    return {"event": "snapshot", "data": {"run": run.to_json(), "state": state, "queue": queue}}


async def watch(run: BackgroundRun) -> AsyncIterator[dict]:
    """Where ``run`` is (a ``snapshot``: the run and its thread's saved state), then its
    events as they happen, ending with a ``status`` event once it has finished.

    Nothing is replayed: a token streamed before you started watching is in the snapshot
    only once its message is complete. Check access first (``get_run``)."""
    from llm_kit.background.bus import get_bus

    async with get_bus().subscribe(run.run_id) as queue:
        run = await sync_to_async(refresh)(run)  # subscribed first: nothing falls in between
        yield await _snapshot(run)
        if not run.active:
            return
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=WATCH_CHECK)
            except TimeoutError:
                run = await sync_to_async(refresh)(run)
                if not run.active:  # its last events went elsewhere, or it was lost
                    yield _status_event(run)
                    return
                continue
            yield event
            if event.get("event") == "status" and event["data"].get("status") in FINISHED:
                return
