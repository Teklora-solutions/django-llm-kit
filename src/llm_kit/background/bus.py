"""How a background run's events reach the people watching it, as they happen (nothing is stored).

``LLM_KIT["RUN_EVENTS"]``:

* ``"local"``: watchers in this process only. Enough with the ``"thread"`` executor and one
  web process.
* ``"channels"``: the Channels layer (``CHANNEL_LAYERS``, e.g. Redis), so a run in a Celery
  worker reaches a watcher in any web process.
* ``"auto"`` (default): ``"channels"`` when ``CHANNEL_LAYERS`` is set, else ``"local"``.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

QUEUE_SIZE = 1000  # per watcher; past it a slow watcher misses tokens, never other events


def _put(queue: asyncio.Queue, event: dict) -> None:
    if queue.full():
        if event.get("event") == "token":
            return
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()  # make room: the oldest goes
    queue.put_nowait(event)


class LocalBus:
    name = "local"

    def __init__(self) -> None:
        self._watchers: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = defaultdict(set)
        self._lock = threading.Lock()

    async def publish(self, run_id: str, event: dict) -> None:
        with self._lock:
            watchers = list(self._watchers.get(run_id, ()))
        for loop, queue in watchers:
            with contextlib.suppress(RuntimeError):  # that watcher's loop has closed
                loop.call_soon_threadsafe(_put, queue, event)

    @contextlib.asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[asyncio.Queue]:
        entry = (asyncio.get_running_loop(), asyncio.Queue(QUEUE_SIZE))
        with self._lock:
            self._watchers[run_id].add(entry)
        try:
            yield entry[1]
        finally:
            with self._lock:
                self._watchers[run_id].discard(entry)
                if not self._watchers[run_id]:
                    del self._watchers[run_id]


def _group(run_id: str) -> str:
    return f"llm_kit.run.{run_id.replace('-', '')}"


class ChannelsBus:
    """Events go through a Channels group per run."""

    name = "channels"

    def __init__(self, layer: Any) -> None:
        self.layer = layer

    async def publish(self, run_id: str, event: dict) -> None:
        await self.layer.group_send(_group(run_id), {"type": "llm_kit.run.event", "event": event})

    @contextlib.asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[asyncio.Queue]:
        channel = await self.layer.new_channel()
        await self.layer.group_add(_group(run_id), channel)
        queue: asyncio.Queue = asyncio.Queue(QUEUE_SIZE)

        async def pump():
            while True:
                message = await self.layer.receive(channel)
                if isinstance(message, dict) and isinstance(message.get("event"), dict):
                    _put(queue, message["event"])

        task = asyncio.create_task(pump())
        try:
            yield queue
        finally:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            await self.layer.group_discard(_group(run_id), channel)


_local = LocalBus()


def get_bus() -> LocalBus | ChannelsBus:
    from llm_kit import conf

    mode = conf.get("RUN_EVENTS")
    if mode not in ("auto", "local", "channels"):
        from llm_kit.core.errors import ConfigError

        raise ConfigError(f'RUN_EVENTS must be "auto", "local" or "channels", not {mode!r}.')
    if mode == "local":
        return _local
    layer = _channel_layer()
    if mode == "auto" and type(layer).__name__ == "InMemoryChannelLayer":
        layer = None  # in-process anyway, and bound to one event loop; runs have their own
    if layer is None:
        if mode == "channels":
            from llm_kit.core.errors import ConfigError

            raise ConfigError('RUN_EVENTS="channels" needs django-channels and CHANNEL_LAYERS.')
        return _local
    return ChannelsBus(layer)


def _channel_layer():
    from django.conf import settings

    if not getattr(settings, "CHANNEL_LAYERS", None):
        return None
    try:
        from channels.layers import get_channel_layer
    except ImportError:
        return None
    return get_channel_layer()
