"""Running a media call through a touchpoint: the same resolution, keys (yours, a tenant's or a
user's), rate limits, retry waits, fallbacks, budgets and call records as chat and embeddings.

* Each model the touchpoint may use is tried in turn. A provider's content-policy refusal stops
  there (``ContentRefused``) unless the touchpoint has ``fallback_on_refusal=True``.
* A long kind (video) is submitted, then checked until done (``MEDIA_POLL_EVERY``), for at most
  ``wait`` seconds (``MEDIA_JOB_WAIT``). Past that, ``MediaJobTimeout`` carries the provider's job
  id, and the next model is not tried: the first may still finish, and bill.
* Each attempt is an ``LLMCall`` with ``kind`` set to the media kind and ``units`` (images,
  seconds, characters...). Its cost is what the backend reported, else the units times the
  catalogue's prices; it counts towards the touchpoint's, tenant's, user's and request's budgets.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from llm_kit.core import config
from llm_kit.core.media import backends as mb
from llm_kit.core.media.errors import ContentRefused, MediaJobFailed, MediaJobTimeout, NoMediaReturned, is_refusal
from llm_kit.core.media.kinds import MediaKind, units_for
from llm_kit.core.media.types import Media, MediaResult
from llm_kit.core.touchpoints.env import provider_of
from llm_kit.core.touchpoints.retry import RetryPolicy

logger = logging.getLogger("llm_kit.touchpoints")


@dataclass
class _Call:
    tp: Any
    kind: MediaKind
    refs: list[str]
    backends: list[Any]
    owners: list[str]
    retry: RetryPolicy
    resolution: Any
    run: dict | None = None
    limiter: Any = None
    params: dict[str, Any] = field(default_factory=dict)


def prepare(tp, kind: MediaKind, *, requested, config_, env, retry, fallbacks) -> _Call:
    from llm_kit.core.observability.spend import check_daily_budget
    from llm_kit.core.touchpoints.backends import key_sets, no_key
    from llm_kit.core.touchpoints.env import default_env
    from llm_kit.core.touchpoints.retry import policy_for

    check_daily_budget(tp)
    env = tp.env_for(config_, env or default_env())
    if (check_budgets := getattr(env, "check_budgets", None)) is not None:
        check_budgets(tp)
    resolution = tp.resolve(requested, config=config_, env=env)
    if not fallbacks:
        resolution.fallbacks = []
    factory = mb.get_media_factory(getattr(tp, "media_backend", None))
    refs, owners, made = [], [], []
    for ref in resolution.candidates:
        for owner, keys in key_sets(env, ref, tp):  # BYOK: the caller's keys, then the platform's
            refs.append(ref)
            owners.append(owner)
            made.append(factory(ref, tp.params_for(ref, resolution.params), keys))
    if not made:
        raise no_key(tp, resolution.candidates)
    run = config_ if config_ is not None else _running()
    call = _Call(tp, kind, refs, made, owners, policy_for(tp, run, retry), resolution, run)
    limiter_for = getattr(env, "rate_limiter", None)
    call.limiter = limiter_for() if callable(limiter_for) else None
    return call


def _running():
    from llm_kit.core.agents.limits import running_config

    return running_config()


def _setting(name: str, default: float) -> float:
    value = config.get(name)
    return float(default if value is None else value)


def _poll_every() -> float:
    return _setting("MEDIA_POLL_EVERY", 5)


def _wait_for(wait: float | None) -> float:
    return float(wait) if wait is not None else _setting("MEDIA_JOB_WAIT", 900)


def _usable(ref: str, result: Any) -> MediaResult:
    if not isinstance(result, MediaResult):
        raise TypeError(f"A media backend must return MediaResult, not {type(result).__name__}.")
    if not result.media and not result.text:
        raise NoMediaReturned(f"{ref} returned no media.")
    return result


def _job_failed(ref: str, job_id: str, state: mb.JobState) -> Exception:
    error = MediaJobFailed(f"{ref} job {job_id} failed: {state.error or 'no reason given'}", job_id)
    return ContentRefused(str(error)) if is_refusal(error) else error


def _make_sync(call: _Call, ref: str, backend: Any, request: dict, extra: dict, wait: float | None) -> MediaResult:
    if call.kind.long and mb.supports_jobs(backend):
        job_id = backend.submit(call.kind.name, request, **extra)
        deadline, every = time.monotonic() + _wait_for(wait), _poll_every()
        while True:
            state = backend.poll(call.kind.name, job_id)
            if state.status == mb.DONE and state.result is not None:
                state.result.job_id = job_id
                return state.result
            if state.finished:
                raise _job_failed(ref, job_id, state)
            if time.monotonic() + every > deadline:
                raise MediaJobTimeout(f"{ref} job {job_id} did not finish in time.", job_id, ref)
            time.sleep(every)
    return backend.generate(call.kind.name, request, **extra)


class JobWatcher:
    """Hooks into a long job run asynchronously (a background media job uses them): when it is
    submitted, each state it reports, and the wait between checks. Override what you need."""

    async def submitted(self, ref: str, job_id: str, backend: Any) -> None:
        """The provider took the job."""

    async def progress(self, ref: str, job_id: str, state: mb.JobState) -> None:
        """A check found it ``state``."""

    async def pause(self, ref: str, seconds: float) -> None:
        """Wait before the next check (return early to check sooner, after a webhook)."""
        await asyncio.sleep(seconds)

    def extra_params(self, ref: str, backend: Any = None) -> dict[str, Any]:
        """More provider parameters for the submission (a webhook URL), sent through ``backend``."""
        return {}


async def _make_async(
    call: _Call,
    ref: str,
    backend: Any,
    request: dict,
    extra: dict,
    wait: float | None,
    watcher: JobWatcher | None = None,
) -> MediaResult:
    if call.kind.long and mb.supports_jobs(backend):
        watcher = watcher or JobWatcher()
        extra = {**extra, **watcher.extra_params(ref, backend)}
        job_id = await mb.asubmit(backend, call.kind.name, request, **extra)
        await watcher.submitted(ref, job_id, backend)
        deadline, every = time.monotonic() + _wait_for(wait), _poll_every()
        while True:
            state = await mb.apoll(backend, call.kind.name, job_id)
            await watcher.progress(ref, job_id, state)
            if state.status == mb.DONE and state.result is not None:
                state.result.job_id = job_id
                return state.result
            if state.finished:
                raise _job_failed(ref, job_id, state)
            if time.monotonic() + every > deadline:
                raise MediaJobTimeout(f"{ref} job {job_id} did not finish in time.", job_id, ref)
            await watcher.pause(ref, every)
    return await mb.agenerate(backend, call.kind.name, request, **extra)


class _Stop(Exception):
    """Don't try another model."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


def _after_failure(call: _Call, error: BaseException, attempt: int) -> float | None:
    """Seconds to wait before trying the same model again; None to move on. Raises ``_Stop``."""
    if isinstance(error, MediaJobTimeout):
        raise _Stop(error)
    if is_refusal(error):
        if not getattr(call.tp, "fallback_on_refusal", False):
            refused = error if isinstance(error, ContentRefused) else ContentRefused(str(error))
            if refused is not error:
                refused.__cause__ = error
            raise _Stop(refused)
        return None
    if isinstance(error, (NoMediaReturned, TypeError)):
        return None
    return call.retry.delay(error, attempt)


def run_sync(call: _Call, request: dict[str, Any], extra: dict[str, Any], wait: float | None = None) -> MediaResult:
    from llm_kit.core.touchpoints import backends
    from llm_kit.core.touchpoints.embeddings import _failure, _slot

    errors: list[tuple[str, BaseException]] = []
    for index, (ref, backend) in enumerate(zip(call.refs, call.backends, strict=True)):
        attempt = 0
        while True:
            if (refused := _slot(call, index, ref)) is not None:
                errors.append((ref, refused))
                break
            rec, t0 = _start(call, ref, index, request)
            try:
                result = _usable(ref, _make_sync(call, ref, backend, request, extra, wait))
            except Exception as e:
                _end(call, rec, t0, request, ref, error=e)
                try:
                    delay = _after_failure(call, e, attempt)
                except _Stop as stop:
                    raise stop.error from (None if stop.error is e else e)
                if delay is None:
                    errors.append((ref, e))
                    break
                attempt += 1
                backends._sleep(delay)
                continue
            return _end(call, rec, t0, request, ref, result=result)
    raise _failure(call, errors)


async def run_async(
    call: _Call,
    request: dict[str, Any],
    extra: dict[str, Any],
    wait: float | None = None,
    watcher: JobWatcher | None = None,
) -> MediaResult:
    from llm_kit.core.touchpoints import backends
    from llm_kit.core.touchpoints.embeddings import _aslot, _failure

    errors: list[tuple[str, BaseException]] = []
    for index, (ref, backend) in enumerate(zip(call.refs, call.backends, strict=True)):
        attempt = 0
        while True:
            if (refused := await _aslot(call, index, ref)) is not None:
                errors.append((ref, refused))
                break
            rec, t0 = _start(call, ref, index, request)
            try:
                result = _usable(ref, await _make_async(call, ref, backend, request, extra, wait, watcher))
                await _aprice(call, ref, request, result)
            except Exception as e:
                _end(call, rec, t0, request, ref, error=e)
                try:
                    delay = _after_failure(call, e, attempt)
                except _Stop as stop:
                    raise stop.error from (None if stop.error is e else e)
                if delay is None:
                    errors.append((ref, e))
                    break
                attempt += 1
                await backends._asleep(delay)
                continue
            return _end(call, rec, t0, request, ref, result=result)
    raise _failure(call, errors)


def fetched(result: MediaResult) -> MediaResult:
    """Download media the provider gave as URLs (they often expire within the hour)."""
    for media in result.media:
        if media.url is not None and media.data is None:
            media.read()
    return result


async def afetched(result: MediaResult) -> MediaResult:
    for media in result.media:
        if media.url is not None and media.data is None:
            await media.aread()
    return result


# ------------------------------------------------------------------ recording


def describe(request: dict[str, Any]) -> dict[str, Any]:
    """The request as it can be stored: media as its description, never its bytes."""

    def one(value: Any) -> Any:
        if isinstance(value, Media):
            return value.to_json()
        if isinstance(value, (bytes, bytearray)):
            return f"<{len(value)} bytes>"
        if isinstance(value, (list, tuple)):
            return [one(v) for v in value]
        if isinstance(value, dict):
            return {k: one(v) for k, v in value.items()}
        return value

    return {k: one(v) for k, v in request.items()}


def _start(call: _Call, ref: str, index: int, request: dict[str, Any]):
    from llm_kit.core.observability.handler import get_handler

    t0 = time.perf_counter()
    if get_handler() is None:
        return None, t0
    from llm_kit.core.catalogue.fields import normalise_provider
    from llm_kit.core.observability.context import current
    from llm_kit.core.observability.handler import _hash
    from llm_kit.core.observability.records import CallRecord
    from llm_kit.core.touchpoints.embeddings import _root_run
    from llm_kit.core.touchpoints.touchpoint import CALLER_KEY, caller_ids

    parent = _running() or {}
    meta, ctx = dict(parent.get("metadata") or {}), current()
    caller = ((call.run or {}).get("configurable") or {}).get(CALLER_KEY)
    if caller is not None:
        user_id, tenant = caller_ids(caller)
        meta.setdefault("user_id", user_id)
        meta.setdefault("tenant", tenant)
    meta.update((call.run or {}).get("metadata") or {})
    rec = CallRecord(
        call_id=str(uuid.uuid4()),
        run_id=_root_run(parent) or meta.get("llm_kit_media_job"),
        started_at=datetime.now(UTC),
        kind=call.kind.name,
        key_owner=call.owners[index] if call.owners else "platform",
        touchpoint=call.tp.name,
        graph=str(meta.get("llm_kit_graph", "")),
        agent=str(meta.get("llm_kit_agent") or ""),
        thread_id=str(meta.get("llm_kit_thread") or meta.get("thread_id") or ""),
        user=str(meta.get("user_id") or ctx.user or ""),
        tenant=str(meta.get("tenant") or ctx.tenant or ""),
        tags=dict(ctx.tags),
        requested_model=str(call.resolution.requested or ""),
        resolved_model=call.resolution.model,
        model=ref,
        provider=normalise_provider(provider_of(ref)),
        chosen_by=call.resolution.chosen_by,
        fallback_index=index,
        prompt_hash=_hash(describe(request)),
    )
    return rec, t0


def _capture(call: _Call) -> bool:
    capture = getattr(call.tp, "capture", None)
    return bool(config.get("CAPTURE_CONTENT")) if capture is None else bool(capture)


def _price(ref: str, result: MediaResult) -> Decimal | None:
    if result.cost is not None:
        return Decimal(str(result.cost))
    from llm_kit.core.observability.cost import price_media
    from llm_kit.core.observability.recorder import recorder

    lookup = recorder.price_lookup
    if lookup is None:
        return None
    try:
        prices = lookup(ref) or (lookup(result.served_model) if result.served_model else None)
    except Exception:
        logger.warning("Could not look up the price of %s", ref, exc_info=True)
        return None
    return price_media(
        prices, result.units, input_tokens=result.input_tokens or 0, output_tokens=result.output_tokens or 0
    )


def _measure(call: _Call, ref: str, request: dict, result: MediaResult) -> None:
    result.kind, result.model = call.kind.name, ref
    result.units = units_for(call.kind, request, result)
    result.cost = _price(ref, result)
    result.__dict__["_priced"] = True


async def _aprice(call: _Call, ref: str, request: dict, result: MediaResult) -> None:
    """Price on a thread: the catalogue lookup may query the database, which an event loop can't."""
    from asgiref.sync import sync_to_async

    await sync_to_async(_measure, thread_sensitive=True)(call, ref, request, result)


def _end(call: _Call, rec, t0: float, request: dict, ref: str, *, result: MediaResult | None = None, error=None):
    if result is not None:
        if not result.__dict__.pop("_priced", False):
            _measure(call, ref, request, result)
        tree = _tree()
        if tree is not None:
            tree.add_cost(result.cost)
    if rec is None:
        return result
    from llm_kit.core.observability.handler import _status_for, handler
    from llm_kit.core.observability.recorder import recorder

    rec.latency_ms = int((time.perf_counter() - t0) * 1000)
    if _capture(call):
        rec.prompt = describe(request)
    if error is not None:
        rec.status, rec.error_class = _status_for(error)
        if is_refusal(error):
            rec.error_class = "refused"
        rec.error_message = f"{type(error).__name__}: {error}"[:500]
        job = getattr(error, "job_id", "")
        if job:
            rec.provider_request_id = str(job)[:255]
    elif result is not None:
        rec.units = result.units
        rec.input_tokens = int(result.input_tokens or 0)
        rec.output_tokens = int(result.output_tokens or 0)
        rec.served_model = result.served_model
        rec.provider_request_id = (result.job_id or "")[:255]
        rec.cost = result.cost
        if _capture(call):
            rec.completion = {"text": result.text, "media": [m.to_json() for m in result.media]}
        handler.count_call(rec)
    recorder.emit(rec)
    return result


def _tree():
    from llm_kit.core.agents.limits import tree_of

    return tree_of(_running())
