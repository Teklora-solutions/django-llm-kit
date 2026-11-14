"""Async Server-Sent-Events endpoints (run under ASGI for real streaming).

    POST /stream/                       create a thread and stream the first run
    POST /threads/<uuid>/stream/        stream a run on an existing thread
    GET  /runs/<uuid>/watch/            watch a background run

With ``"background": true`` in the body the run is started detached: the response streams it
(``run``, ``snapshot``, then its events) but closing the connection leaves it running.

Body: see ``llm_kit.runtime.runs``. With session auth, send the ``X-CSRFToken`` header.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
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


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def with_keepalive(events: AsyncIterator[dict], interval: float) -> AsyncIterator[str]:
    """Format events as SSE, emitting a comment line when the run is quiet."""
    it = events.__aiter__()
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(it.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval or None)
            if not done:
                yield ": keepalive\n\n"
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                return
            finally:
                pending = None
            yield sse(item["event"], item["data"])
    finally:
        if pending is not None:
            pending.cancel()  # the client left mid-read: let that read finish before closing
            with contextlib.suppress(BaseException):
                await pending
        if hasattr(it, "aclose"):
            await it.aclose()


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

    detached = body.pop("background", False) is True
    try:
        thread, created, owner = await sync_to_async(_resolve_thread)(thread_id, request, body)
        if detached:
            run = await background.astart(thread, body, owner)
        else:
            if (busy := await sync_to_async(_active_run)(thread)) is not None:
                raise background.RunBusy(busy)
            graph = get_graph(thread.graph)
            payload_input = runs.build_input(body)
            config = await sync_to_async(runs.build_config)(thread, body, owner=owner)
    except Thread.DoesNotExist:
        return _error(404, "Thread not found.")
    except access.NotAllowed as e:
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
            async for item in background.watch(run):  # leaving stops the watching, not the run
                yield item

        return _event_stream(watching())

    async def events():
        if created:
            yield {"event": "thread", "data": {"id": thread.thread_id, "graph": thread.graph}}
        async for item in runs.astream_run(graph, payload_input, config):
            yield item
        await Thread.objects.filter(pk=thread.pk).aupdate(updated_at=timezone.now())

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
    with a ``status`` event once it has finished. Reconnecting gives a fresh snapshot."""
    from llm_kit import background
    from llm_kit.models import BackgroundRun

    user = await request.auser()
    if not access.can_use_api(user):
        return _error(401, "Authentication required.")
    try:
        run = await sync_to_async(_get_run)(request, run_id)
    except (BackgroundRun.DoesNotExist, access.NotAllowed):
        return _error(404, "Run not found.")
    return _event_stream(background.watch(run))
