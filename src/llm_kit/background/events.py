"""Your own events, sent to a background run's watchers alongside the kit's.

A run hook's context manager, a tool or a callback can tell watchers something only your code
knows (how a turn ended, a step's progress)::

    from llm_kit import background

    @contextlib.contextmanager
    def outcome(run):
        try:
            yield
        except TimeoutError:
            background.publish(run, {"event": "myapp.outcome", "data": {"kind": "timeout"}})
            raise

The event reaches every watcher like the kit's own (SSE, WebSocket, ``background.watch``),
before the run's final ``status`` event. Its name must contain a dot (``myapp.outcome``): the
kit's own names never do, so yours can't be mistaken for them. ``data`` must be JSON. Outside a
background run, nothing is sent and ``publish`` returns ``False``.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

from llm_kit.background.service import RUN_KEY

__all__ = ["apublish", "publish", "run_id_of"]

_pending: dict[str, set[asyncio.Task]] = defaultdict(set)


def run_id_of(target: Any) -> str | None:
    """The background run id of ``target``: a run's ``config``, a hook's ``RunSetup``, a
    ``BackgroundRun`` or an id; ``None`` when it isn't a background run."""
    from llm_kit.core.hooks import RunSetup

    if isinstance(target, RunSetup):
        target = target.config
    if target is None:
        return None
    if isinstance(target, str):
        return target or None
    run_id = getattr(target, "run_id", None)
    if run_id is not None and not isinstance(target, dict):
        return str(run_id)
    if isinstance(target, dict):
        return ((target.get("configurable") or {}).get(RUN_KEY)) or None
    raise TypeError(f"Expected a run's config, a BackgroundRun or a run id, not {type(target).__name__}.")


def _checked(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict) or not isinstance(event.get("event"), str):
        raise ValueError('An event is a dict like {"event": "myapp.outcome", "data": {...}}.')
    if "." not in event["event"]:
        raise ValueError(f"Name your event with a dot ('myapp.{event['event']}'): the kit's own names have none.")
    try:
        return json.loads(json.dumps({"event": event["event"], "data": event.get("data")}))
    except (TypeError, ValueError) as e:
        raise ValueError(f"An event's data must be JSON-serialisable: {e}") from None


async def apublish(target: Any, event: dict[str, Any]) -> bool:
    """Send ``event`` to the watchers of ``target``'s background run; ``False`` if there's none."""
    from llm_kit.background.log import send

    event = _checked(event)
    run_id = run_id_of(target)
    if run_id is None:
        return False
    await send(run_id, event)
    return True


def publish(target: Any, event: dict[str, Any]) -> bool:
    """``apublish`` for sync code, including a sync context manager around an async run: then
    the event is sent from the run's loop, still before the run's final ``status`` event."""
    event = _checked(event)
    run_id = run_id_of(target)
    if run_id is None:
        return False
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        from asgiref.sync import async_to_sync

        return async_to_sync(apublish)(run_id, event)
    task = asyncio.ensure_future(apublish(run_id, event))
    _pending[run_id].add(task)
    task.add_done_callback(lambda t: _forget(run_id, t))
    return True


def _forget(run_id: str, task: asyncio.Task) -> None:
    tasks = _pending.get(run_id)
    if tasks is not None:
        tasks.discard(task)
        if not tasks:
            _pending.pop(run_id, None)


async def flush(run_id: str) -> None:
    """Wait for events ``publish`` sent from the run's loop (before its final ``status``)."""
    tasks = list(_pending.get(run_id, ()))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
