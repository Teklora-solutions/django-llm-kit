"""Async Server-Sent-Events endpoints (run under ASGI for real streaming).

    POST /stream/                       create a thread and stream the first run
    POST /threads/<uuid>/stream/        stream a run on an existing thread

Body: see ``llm_kit.runtime.runs``. With session auth, send the ``X-CSRFToken`` header.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from asgiref.sync import sync_to_async
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.http import require_POST

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
            pending.cancel()
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

    try:
        thread, created, owner = await sync_to_async(_resolve_thread)(thread_id, request, body)
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

    async def events():
        if created:
            yield {"event": "thread", "data": {"id": thread.thread_id, "graph": thread.graph}}
        async for item in runs.astream_run(graph, payload_input, config):
            yield item
        await Thread.objects.filter(pk=thread.pk).aupdate(updated_at=timezone.now())

    response = StreamingHttpResponse(
        with_keepalive(events(), conf.get("SSE_KEEPALIVE")), content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # disable nginx buffering
    return response
