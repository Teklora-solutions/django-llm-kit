"""Use any LangGraph checkpointer, from sync and async code alike.

``build_checkpointer(value)`` accepts:

* a saver instance, a saver class, or a zero-argument factory (or a dotted path to one);
* ``{"sync": factory, "async": factory}``: a sync saver for sync code, and an async saver
  **per event loop** for async code. Async factories may be coroutines.

Why per loop: an async connection pool (``AsyncPostgresSaver`` over ``AsyncConnectionPool``)
belongs to the event loop that opened it. A Celery task that calls ``asyncio.run`` twice
gets two loops, and the second would reach for a pool whose loop is gone. The router makes
one async saver per running loop, and forgets savers of loops that have closed.

Gaps are filled in, so every graph works from both sides:

* a sync-only saver (e.g. ``PostgresSaver``) gets async methods that run it in a worker thread;
* an async-only factory gets sync methods that run on one private background loop.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from llm_kit.core.graphs import import_target

__all__ = ["CheckpointRouter", "build_checkpointer", "has_async", "has_sync"]

Factory = Callable[[], Any]


def has_sync(saver: BaseCheckpointSaver) -> bool:
    return type(saver).get_tuple is not BaseCheckpointSaver.get_tuple


def has_async(saver: BaseCheckpointSaver) -> bool:
    return type(saver).aget_tuple is not BaseCheckpointSaver.aget_tuple


def _resolve(value: Any) -> Any:
    return import_target(value) if isinstance(value, str) else value


def _instance(value: Any) -> BaseCheckpointSaver:
    value = _resolve(value)
    if isinstance(value, BaseCheckpointSaver):
        return value
    if callable(value):
        if inspect.iscoroutinefunction(value):
            raise TypeError("An async factory must be given as {'async': factory}.")
        saver = value()
        if inspect.isawaitable(saver):
            if inspect.iscoroutine(saver):
                saver.close()
            raise TypeError("An async factory must be given as {'async': factory}.")
        if isinstance(saver, BaseCheckpointSaver):
            return saver
    raise TypeError(f"Not a LangGraph checkpointer, or a factory for one: {value!r}")


def build_checkpointer(value: Any) -> BaseCheckpointSaver:
    """A checkpointer usable from sync and async code. See the module docstring."""
    if isinstance(value, dict):
        unknown = set(value) - {"sync", "async"}
        if unknown or not value:
            raise ValueError("A checkpointer dict takes 'sync' and/or 'async' factories.")
        sync, async_ = value.get("sync"), value.get("async")
        return CheckpointRouter(
            sync=(lambda: _instance(sync)) if sync else None,
            async_=_resolve(async_) if async_ else None,
        )
    saver = _instance(value)
    if has_sync(saver) and has_async(saver):
        return saver
    if has_sync(saver):
        return CheckpointRouter(sync=lambda: saver)
    raise TypeError(
        f"{type(saver).__name__} only works from async code on one event loop; "
        "give it as {'async': factory} so each loop gets its own."
    )


def _close(saver: BaseCheckpointSaver) -> None:
    closer = getattr(saver, "llm_kit_close", None)
    if callable(closer):
        try:
            closer()
        except Exception:  # closing is best effort
            pass


class _LoopThread:
    """One private event loop on a daemon thread, for sync calls to an async-only saver."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="llm-kit-checkpointer", daemon=True).start()

    def run(self, coro):
        try:
            if asyncio.get_running_loop() is self.loop:
                raise RuntimeError("Sync checkpointer call from its own background loop.")
        except RuntimeError as e:
            if "background loop" in str(e):
                raise
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


class CheckpointRouter(BaseCheckpointSaver):
    """Sends sync calls to a sync saver, and async calls to the saver for the running loop."""

    def __init__(self, *, sync: Factory | None = None, async_: Factory | None = None) -> None:
        if sync is None and async_ is None:
            raise ValueError("CheckpointRouter needs a sync or an async factory.")
        super().__init__()
        self._sync_factory, self._async_factory = sync, async_
        self._sync: BaseCheckpointSaver | None = None
        self._lock = threading.Lock()
        # Strong references on purpose: with weak keys, a finished asyncio.run()'s loop would vanish
        # from the dict before its saver's connections were closed.
        self._per_loop: dict[asyncio.AbstractEventLoop, BaseCheckpointSaver] = {}
        self._creating: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
        self._bg: _LoopThread | None = None

    # ------------------------------------------------------------ savers

    def sync_saver(self) -> BaseCheckpointSaver | None:
        if self._sync_factory is None:
            return None
        if self._sync is None:
            with self._lock:
                if self._sync is None:
                    self._sync = self._sync_factory()
        return self._sync

    async def async_saver(self) -> BaseCheckpointSaver | None:
        """The async saver for the running loop, created on first use (None if there's no async factory)."""
        if self._async_factory is None:
            return None
        loop = asyncio.get_running_loop()
        saver = self._per_loop.get(loop)
        if saver is not None:
            return saver
        with self._lock:
            creating = self._creating.setdefault(loop, asyncio.Lock())
        async with creating:
            saver = self._per_loop.get(loop)
            if saver is None:
                saver = self._async_factory()
                if inspect.isawaitable(saver):
                    saver = await saver
                if not isinstance(saver, BaseCheckpointSaver):
                    raise TypeError(f"The async checkpointer factory returned {saver!r}.")
                with self._lock:
                    for old in [lp for lp in self._per_loop if lp.is_closed()]:
                        _close(self._per_loop.pop(old))  # a finished asyncio.run(): free its connections
                        self._creating.pop(old, None)
                    self._per_loop[loop] = saver
        return saver

    def close(self) -> None:
        """Close every saver this router made (at shutdown, or at the end of a test session).

        A saver is closed through its ``llm_kit_close`` attribute, if it has one (the
        Postgres factories in ``llm_kit.persistence.postgres`` set it)."""
        with self._lock:
            savers = [self._sync, *self._per_loop.values()]
            self._sync = None
            self._per_loop.clear()
        for saver in savers:
            if saver is not None:
                _close(saver)

    def _background(self) -> _LoopThread:
        if self._bg is None:
            with self._lock:
                if self._bg is None:
                    self._bg = _LoopThread()
        return self._bg

    def _call(self, name: str, *args, **kwargs):
        """A sync call: the sync saver, or the async saver on the background loop."""
        saver = self.sync_saver()
        if saver is not None:
            return getattr(saver, name)(*args, **kwargs)

        async def call():
            return await getattr(await self.async_saver(), "a" + name)(*args, **kwargs)

        return self._background().run(call())

    async def _acall(self, name: str, *args, **kwargs):
        """An async call: the loop's async saver, or the sync saver in a worker thread."""
        saver = await self.async_saver()
        if saver is not None:
            return await getattr(saver, "a" + name)(*args, **kwargs)
        return await asyncio.to_thread(getattr(self.sync_saver(), name), *args, **kwargs)

    # -------------------------------------------------------------- sync

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._call("get_tuple", config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        saver = self.sync_saver()
        if saver is not None:
            yield from saver.list(config, filter=filter, before=before, limit=limit)
            return

        async def collect():
            s = await self.async_saver()
            return [t async for t in s.alist(config, filter=filter, before=before, limit=limit)]

        yield from self._background().run(collect())

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self._call("put", config, checkpoint, metadata, new_versions)

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        return self._call("put_writes", config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        return self._call("delete_thread", thread_id)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        return self._call("copy_thread", source_thread_id, target_thread_id)

    def delete_for_runs(self, run_ids) -> None:
        return self._call("delete_for_runs", run_ids)

    def prune(self, thread_ids, *, strategy: str = "keep_latest") -> None:
        return self._call("prune", thread_ids, strategy=strategy)

    def get_next_version(self, current, channel=None):
        saver = self.sync_saver() or next(iter(self._per_loop.values()), None)
        if saver is not None:
            return saver.get_next_version(current, channel)
        return super().get_next_version(current, channel)

    # ------------------------------------------------------------- async

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self._acall("get_tuple", config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        saver = await self.async_saver()
        if saver is not None:
            async for item in saver.alist(config, filter=filter, before=before, limit=limit):
                yield item
            return
        sync = self.sync_saver()
        items = await asyncio.to_thread(lambda: list(sync.list(config, filter=filter, before=before, limit=limit)))
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await self._acall("put", config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        return await self._acall("put_writes", config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return await self._acall("delete_thread", thread_id)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        return await self._acall("copy_thread", source_thread_id, target_thread_id)

    async def adelete_for_runs(self, run_ids) -> None:
        return await self._acall("delete_for_runs", run_ids)

    async def aprune(self, thread_ids, *, strategy: str = "keep_latest") -> None:
        return await self._acall("prune", thread_ids, strategy=strategy)
