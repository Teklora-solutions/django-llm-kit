"""Media made in the background: a video takes minutes, longer than a request should wait.

``tp.start_video(...)`` (or ``tp.start(kind, ...)``) saves a ``BackgroundRun`` with
``kind="media"`` and returns it at once. It runs where background runs run (``RUN_EXECUTOR``:
a thread, Celery, Django-Q2 or your own), with everything background runs have:

* ``background.watch(run)`` (SSE ``llm/runs/<id>/watch/``, WebSocket ``watch``): ``progress``
  events (``submitted``, then the provider's status and progress), a ``media`` event with the
  result, then the final ``status``;
* ``run.cancel()`` from any process (the provider's job is cancelled too when it can be);
* lost-run detection, and the ``run_finished`` signal, plus ``media_job_finished`` (below).

The job's media are saved to storage (``MEDIA_STORAGE``, under ``MEDIA_PATH``/jobs/<run id>/)
before the run is ``done``; ``run.result`` has their names and URLs, the text, units, cost and
model. While it runs, ``run.result["job"]`` has the provider's job id.

A media job has no thread. Who may watch or cancel it is decided by who started it (the same
rules as threads: its user, its visitor, staff).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from asgiref.sync import sync_to_async
from django.db import transaction
from django.utils import timezone

from llm_kit import conf
from llm_kit.core.errors import BadRequest
from llm_kit.core.media import backends as mb
from llm_kit.core.media.calls import JobWatcher
from llm_kit.core.media.types import Media, MediaResult
from llm_kit.models import BackgroundRun
from llm_kit.models.runs import MEDIA

logger = logging.getLogger("llm_kit")

INPUT_LIMIT = 25 << 20  # bytes of media a job may be given inline (pass a URL for more)


# ------------------------------------------------------------------ the request, saved


def _pack(value: Any, budget: list[int]) -> Any:
    if isinstance(value, Media):
        if value.url is not None and value.data is None:
            return {"__media__": {"url": value.url, "mime_type": value.mime_type, "name": value.name}}
        data = value.read()
        budget[0] += len(data)
        if budget[0] > INPUT_LIMIT:
            raise BadRequest("Media given to a background job is too large to keep with it; pass a URL instead.")
        packed = {"data": base64.b64encode(data).decode(), "mime_type": value.mime_type, "name": value.name}
        return {"__media__": packed}
    if isinstance(value, (list, tuple)):
        return [_pack(v, budget) for v in value]
    if isinstance(value, dict):
        return {k: _pack(v, budget) for k, v in value.items()}
    return value


def pack(request: dict[str, Any]) -> dict[str, Any]:
    """``request`` as JSON to keep with the run: media inline (base64) or as their URLs."""
    import json

    packed = _pack(request, [0])
    try:
        json.dumps(packed)
    except TypeError as e:
        raise BadRequest(f"A background media job's request must be JSON (or media): {e}") from e
    return packed


def unpack(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__media__"}:
            return Media.of(value["__media__"])
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


# ------------------------------------------------------------------ starting


def start(
    tp: Any,
    kind: str,
    request: dict[str, Any],
    *,
    params: dict[str, Any] | None = None,
    requested: str | None = None,
    owner: Any = None,
    user=None,
    tenant: Any = None,
    wait: float | None = None,
) -> BackgroundRun:
    """Start a media job in the background and return its ``BackgroundRun`` at once.

    It acts for ``owner`` (an ``access.Owner``) or ``user``/``tenant``: their keys, saved
    settings and budgets. A caller over budget, a model they may not use or a missing key is
    refused here, before anything is saved."""
    import json

    from llm_kit.background import executors
    from llm_kit.background.service import _owner
    from llm_kit.core.media.kinds import get_kind
    from llm_kit.core.touchpoints.touchpoint import get_touchpoint

    tp = get_touchpoint(tp) if isinstance(tp, str) else tp
    kind = get_kind(kind)
    owner = _owner(owner, user=user, tenant=tenant)
    tp._media(kind.name, requested, _config(owner))  # budgets, the model, keys: refused now
    try:
        json.dumps(params or {})
    except TypeError as e:
        raise BadRequest(f"A background media job's params must be JSON: {e}") from e
    executor = str(conf.get("RUN_EXECUTOR"))
    submit = executors.get_executor(executor)
    run = BackgroundRun.objects.create(
        kind=MEDIA,
        payload={
            "touchpoint": tp.name,
            "kind": kind.name,
            "request": pack(request),
            "params": dict(params or {}),
            "model": requested,
            "wait": wait,
        },
        user=owner.user if owner.is_authenticated else None,
        session_key="" if owner.is_authenticated else (owner.visitor_id or ""),
        tenant=owner.tenant_id,
        executor=executor,
    )
    transaction.on_commit(lambda: submit(run.run_id))
    return run


async def astart(tp: Any, kind: str, request: dict[str, Any], **kw: Any) -> BackgroundRun:
    return await sync_to_async(start)(tp, kind, request, **kw)


def _config(owner: Any, run_id: str | None = None) -> dict[str, Any]:
    from llm_kit.core.touchpoints.touchpoint import with_caller

    config: dict[str, Any] = {"metadata": {"llm_kit_media_job": run_id}} if run_id else {}
    if owner is not None and (owner.is_authenticated or owner.tenant_id):
        return with_caller(config, owner)
    return config


# ------------------------------------------------------------------ running (in the worker)

_wakes: dict[str, asyncio.Event] = {}


def wake_local(run_id: str) -> None:
    """A webhook said the job changed: check it now (called by the run's watchdog)."""
    event = _wakes.get(run_id)
    if event is not None:
        event.set()


class _Watcher(JobWatcher):
    def __init__(self, run: BackgroundRun) -> None:
        self.run = run
        self.job: dict[str, Any] = {}
        self.backend: Any = None
        self.kind = run.payload.get("kind", "")
        self.last: tuple | None = None

    def extra_params(self, ref: str) -> dict[str, Any]:
        from llm_kit.media import webhooks

        return webhooks.submit_params(ref)

    async def submitted(self, ref: str, job_id: str, backend: Any) -> None:
        from llm_kit.background import log

        provider_id = job_id
        if callable(getattr(backend, "provider_job_id", None)):
            provider_id = backend.provider_job_id(job_id)
        self.backend = backend
        self.job = {"id": job_id, "provider_id": provider_id, "model": ref}
        await BackgroundRun.objects.filter(pk=self.run.pk).aupdate(result={"job": self.job})
        data = {"status": "submitted", "model": ref, "job": job_id}
        await log.send(self.run.run_id, {"event": "progress", "data": data})

    async def progress(self, ref: str, job_id: str, state: mb.JobState) -> None:
        from llm_kit.background import log

        now = (state.status, state.progress)
        if now == self.last:
            return
        self.last = now
        data = {"status": state.status, "progress": state.progress, "model": ref}
        await log.send(self.run.run_id, {"event": "progress", "data": data})

    async def pause(self, ref: str, seconds: float) -> None:
        from llm_kit.media import webhooks

        if webhooks.config_for(ref) is not None:  # the provider says when: check rarely meanwhile
            seconds = max(seconds, float(conf.get("MEDIA_WEBHOOK_POLL_EVERY") or 60))
        event = _wakes.setdefault(self.run.run_id, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout=seconds)
        except TimeoutError:
            pass
        event.clear()


async def run(run: BackgroundRun) -> dict[str, Any]:
    """Make a media job's media (what ``background`` runs for ``kind="media"``). Returns
    ``run.result``. Raises like a media call (``ContentRefused``, ``MediaJobTimeout``...)."""
    from llm_kit.background import log
    from llm_kit.core.media import calls
    from llm_kit.core.media.kinds import get_kind
    from llm_kit.core.touchpoints.touchpoint import get_touchpoint
    from llm_kit.runtime import access

    payload = run.payload or {}
    tp = get_touchpoint(payload["touchpoint"])
    kind = get_kind(payload["kind"])
    owner = access.Owner(user=run.user, tenant=run.tenant or None)
    config = _config(owner, run.run_id)
    call = await sync_to_async(tp._media)(kind.name, payload.get("model"), config)
    request = unpack(payload.get("request") or {})
    wait = payload.get("wait") or conf.get("MEDIA_BACKGROUND_WAIT")
    watcher = _Watcher(run)
    _wakes[run.run_id] = asyncio.Event()
    try:
        result = await calls.run_async(call, request, payload.get("params") or {}, wait, watcher)
    except asyncio.CancelledError:
        if watcher.job and watcher.backend is not None:
            try:
                await asyncio.shield(mb.acancel(watcher.backend, kind.name, watcher.job["id"]))
            except Exception:
                logger.warning("Could not cancel media job %s at the provider", watcher.job["id"], exc_info=True)
        raise
    finally:
        _wakes.pop(run.run_id, None)
    await calls.afetched(result)
    kept = await keep(result, run.run_id)
    if watcher.job:
        kept["job"] = watcher.job
    await log.send(run.run_id, {"event": "media", "data": kept})
    return kept


async def keep(result: MediaResult, run_id: str) -> dict[str, Any]:
    """Save the result's media to storage; the result as JSON with their names and URLs."""
    from llm_kit.core.media.types import default_storage

    storage = await sync_to_async(default_storage)()
    prefix = str(conf.get("MEDIA_PATH") or "llm_kit/media/").rstrip("/")
    out = result.to_json()
    for i, (media, entry) in enumerate(zip(result.media, out["media"], strict=True), start=1):
        saved = await media.asave(storage, f"{prefix}/jobs/{run_id}/{i}{media.extension}")
        entry["name"] = saved
        try:
            entry["url"] = await sync_to_async(storage.url, thread_sensitive=False)(saved)
        except Exception:  # a storage without URLs
            entry.pop("url", None)
    return out


def notify(job_id: str) -> int:
    """A provider said job ``job_id`` changed: its running media job checks it now. Returns how
    many runs were told (0 when no running job has that id)."""
    from django.db.models import Q

    from llm_kit.models.runs import ACTIVE

    return BackgroundRun.objects.filter(
        Q(result__job__provider_id=job_id) | Q(result__job__id=job_id), kind=MEDIA, status__in=ACTIVE
    ).update(notified_at=timezone.now())
