"""Async Server-Sent-Events endpoints (run under ASGI for real streaming).

    POST /stream/                       create a thread and stream the first run
    POST /threads/<uuid>/stream/        stream a run on an existing thread
    GET  /runs/<uuid>/watch/            watch a background run

With ``"background": true`` in the body the run is started detached: the response streams it
(``run``, ``snapshot``, then its events) but closing the connection leaves it running. With
``"queue": true`` (a background run too) a busy thread doesn't refuse it: it waits its turn.

Body: see ``llm_kit.runtime.runs``. With session auth, send the ``X-CSRFToken`` header.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator

from asgiref.sync import sync_to_async
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from llm_kit import conf
from llm_kit.core.errors import BudgetExceeded
from llm_kit.models import Thread
from llm_kit.runtime import access, runs
from llm_kit.runtime.graphs import UnknownGraph, default_graph_name, get_graph

logger = logging.getLogger("llm_kit")


def sse(event: str, data, seq: int | None = None) -> str:
    head = f"id: {seq}\n" if seq is not None else ""  # a reconnecting EventSource sends it back
    return f"{head}event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


_DONE = object()


async def with_keepalive(events: AsyncIterator[dict], interval: float) -> AsyncIterator[str]:
    """Format events as SSE, emitting a comment line when the run is quiet.

    The events are read by one task from start to end, so the run keeps one context: a
    ContextVar a run hook sets stays set for the whole run, and is reset where it was set."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)  # a slow client slows the run, not memory

    async def pump():
        it = events.__aiter__()
        try:
            async for item in it:
                await queue.put(item)
            await queue.put(_DONE)
        except asyncio.CancelledError:
            raise
        except BaseException as e:  # handed to the reader, which raises it
            await queue.put(e)
        finally:
            if hasattr(it, "aclose"):
                with contextlib.suppress(BaseException):
                    await it.aclose()  # closed in this task, the one that started it

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval or None)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield sse(item["event"], item["data"], item.get("seq"))
    finally:
        if not task.done():
            task.cancel()  # the client left: stop the run's stream (a background run carries on)
        with contextlib.suppress(BaseException):
            await task


def _error(status: int, message: str) -> JsonResponse:
    return JsonResponse({"detail": message}, status=status)


def _resolve_thread(thread_id, request, body):
    """Sync: load (or create) the thread and validate the payload."""
    owner = access.owner_from_request(request)
    if thread_id is None:
        graph = body.get("graph") or default_graph_name()
        get_graph(graph)  # raises UnknownGraph
        runs.validate_choice(graph, body.get("model"))  # before creating anything
        message = body.get("message")
        title = body.get("title") or (message if isinstance(message, str) else "")
        thread = Thread.objects.create(graph=graph, title=str(title)[:255], **access.owner_fields(owner))
        created = True
    else:
        thread = access.get_thread(thread_id, owner)
        runs.validate_choice(thread.graph, body.get("model"))
        created = False
    return thread, created, owner


@require_POST
async def stream(request, thread_id=None):
    user = await request.auser()
    if not access.can_use_api(user):
        return _error(401, "Authentication required.")
    try:
        body = json.loads(request.body or b"{}")
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        return _error(400, "Body must be a JSON object.")

    from llm_kit import background

    queue = body.pop("queue", None)
    detached = body.pop("background", False) is True or queue is True
    failed = None  # a run hook that failed: the stream says so instead of running
    try:
        thread, created, owner = await sync_to_async(_resolve_thread)(thread_id, request, body)
        if detached:
            run = await background.astart(thread, body, owner, queue=queue if isinstance(queue, bool) else None)
        else:
            if (busy := await sync_to_async(_active_run)(thread)) is not None:
                raise background.RunBusy(busy)
            graph = get_graph(thread.graph)
            try:
                setup = await sync_to_async(runs.prepare)(thread, body, owner=owner)
            except runs.HookFailed as e:
                logger.exception("Run hook failed for thread %s", thread.pk)
                failed = runs.error_message(e)
    except Thread.DoesNotExist:
        return _error(404, "Thread not found.")
    except (access.NotAllowed, runs.RunRefused) as e:
        return _error(403, str(e))
    except (UnknownGraph, runs.BadRequest) as e:
        return _error(400, str(e))
    except BudgetExceeded as e:
        return _error(429, str(e))
    except background.RunBusy as e:
        return _busy(e.run)

    if detached:

        async def watching():
            if created:
                yield {"event": "thread", "data": {"id": thread.thread_id, "graph": thread.graph}}
            yield {"event": "run", "data": {"id": run.run_id, "status": run.status}}
            # From its first event. Leaving stops the watching, not the run.
            async for item in background.watch(run, after=0):
                yield item

        return _event_stream(watching())

    async def events():
        if created:
            yield {"event": "thread", "data": {"id": thread.thread_id, "graph": thread.graph}}
        if failed is not None:
            yield {"event": "error", "data": {"message": failed}}
            return
        async for item in runs.astream_run(graph, setup.input, setup.config, setup.contexts):
            yield item
        await Thread.objects.filter(pk=thread.pk).aupdate(updated_at=timezone.now())
        await background.aadvance(thread)  # a run queued meanwhile may start now

    return _event_stream(events())


def _event_stream(events: AsyncIterator[dict]) -> StreamingHttpResponse:
    response = StreamingHttpResponse(
        with_keepalive(events, conf.get("SSE_KEEPALIVE")), content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # disable nginx buffering
    return response


def _active_run(thread):
    from llm_kit.background.service import active_run

    return active_run(thread)


def _busy(run) -> JsonResponse:
    body = {"detail": "This thread already has a run in progress."}
    if run is not None:
        body["run"] = run.to_json()
    return JsonResponse(body, status=409)


def _get_run(request, run_id):
    from llm_kit import background

    return background.get_run(run_id, access.owner_from_request(request))


@require_GET
async def watch(request, run_id):
    """``GET llm/runs/<id>/watch/``: a background run's snapshot, then its events live, ending
    with a ``status`` event once it has finished. With ``?after=<seq>`` or a ``Last-Event-ID``
    header (an ``EventSource`` reconnecting sends it), the events after that one instead."""
    from llm_kit import background
    from llm_kit.models import BackgroundRun

    user = await request.auser()
    if not access.can_use_api(user):
        return _error(401, "Authentication required.")
    try:
        run = await sync_to_async(_get_run)(request, run_id)
    except (BackgroundRun.DoesNotExist, access.NotAllowed):
        return _error(404, "Run not found.")
    after = request.GET.get("after", request.headers.get("Last-Event-ID"))
    if after is not None:
        try:
            after = int(after)
        except ValueError:
            return _error(400, "after must be an event's seq (a whole number).")
        if after < 0:
            return _error(400, "after must be an event's seq (a whole number).")
    return _event_stream(background.watch(run, after=after))
