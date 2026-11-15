"""Each background run's events, kept for a while so a watcher can start from any point.

Every event a run sends gets ``seq``: 1, 2, 3, ... with no gaps. ``watch(run, after=n)`` replays
the events after ``n`` from the log, then carries on live, so the starter of a run, a late
watcher and a reconnecting client miss nothing, and a watcher whose queue overflowed fills
the gap from the log.

Built for many runs at once:

* The process running the run owns its counter (a run runs in one place at a time) and writes
  the log in chunks of ``CHUNK`` events: tokens are batched and written at most every
  ``FLUSH_EVERY`` seconds, other events at once. A token never costs its own cache write.
* The log is capped (``RUN_EVENT_LOG_SIZE`` events per run, older ones are dropped) and expires
  ``RUN_EVENT_LOG_TTL`` seconds after the run's last event.
* A watcher reads it only when it starts and when it finds a gap: a few ``get_many`` calls.

Where it lives (``RUN_EVENT_LOG``): ``"cache"``, the Django cache ``RUN_EVENT_LOG_CACHE``
(Redis, Memcached: shared by every process); ``"memory"``, this process only (enough with the
thread executor and one web process); ``"auto"`` (the default), the cache unless it is a local
memory or dummy cache; ``"off"``, no log: watchers get live events only and no ``seq``.

Events sent from a process that isn't running the run (rare: ``background.publish`` from
elsewhere) go to watchers live, without a ``seq``, and aren't kept.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from llm_kit import conf

logger = logging.getLogger("llm_kit")

CHUNK = 64  # events per cache entry
FLUSH_EVERY = 0.1  # seconds between writes of a running run's tokens
FILL_TRIES = 5  # a watcher that finds a gap re-reads the log this many times, FLUSH_EVERY apart

_LOCAL_CACHES = ("django.core.cache.backends.locmem.LocMemCache", "django.core.cache.backends.dummy.DummyCache")


def _head(run_id: str) -> str:
    return f"llm_kit:runlog:{run_id}"


def _chunk(run_id: str, index: int) -> str:
    return f"llm_kit:runlog:{run_id}:{index}"


def _index(seq: int) -> int:
    return (seq - 1) // CHUNK


class MemoryStore:
    """A dict with expiry, for one process."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._writes = 0

    def get_many(self, keys) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            return {k: item[1] for k in keys if (item := self._data.get(k)) and item[0] > now}

    def set_many(self, mapping: dict[str, Any], timeout: float) -> None:
        expires = time.monotonic() + timeout
        with self._lock:
            for key, value in mapping.items():
                self._data[key] = (expires, value)
            self._writes += 1
            if self._writes % 1000 == 0:  # sweep what has expired now and then
                now = time.monotonic()
                for key in [k for k, (e, _) in self._data.items() if e <= now]:
                    del self._data[key]

    def delete_many(self, keys) -> None:
        with self._lock:
            for key in keys:
                self._data.pop(key, None)


_memory = MemoryStore()


class CacheStore:
    def __init__(self, cache) -> None:
        self.cache = cache

    def get_many(self, keys) -> dict[str, Any]:
        return self.cache.get_many(list(keys))

    def set_many(self, mapping: dict[str, Any], timeout: float) -> None:
        self.cache.set_many(mapping, timeout=timeout)

    def delete_many(self, keys) -> None:
        self.cache.delete_many(list(keys))


class EventLog:
    def __init__(self, store, size: int, ttl: float) -> None:
        self.store, self.size, self.ttl = store, max(int(size), CHUNK), float(ttl)

    def head(self, run_id: str) -> int:
        """The last ``seq`` written for ``run_id`` (0 when none is)."""
        return int(self.store.get_many([_head(run_id)]).get(_head(run_id)) or 0)

    def read(self, run_id: str, after: int) -> tuple[list[dict], bool]:
        """The events after ``after`` that the log holds, in order and without gaps, and whether
        they start right after ``after`` (``False``: older events are gone; resync)."""
        head = self.head(run_id)
        if head <= after:
            return [], True
        first, last = _index(after + 1), _index(head)
        keys = [_chunk(run_id, i) for i in range(first, last + 1)]
        found = self.store.get_many(keys)
        events: list[dict] = []
        for key in keys:
            for event in found.get(key) or ():
                if event["seq"] == after + 1 + len(events):
                    events.append(event)
        return events, bool(events)

    async def aread(self, run_id: str, after: int) -> tuple[list[dict], bool]:
        return await asyncio.to_thread(self.read, run_id, after)

    async def ahead(self, run_id: str) -> int:
        return await asyncio.to_thread(self.head, run_id)


class Writer:
    """The log of one run, in the process running it."""

    def __init__(self, log: EventLog, run_id: str) -> None:
        self.log, self.run_id = log, run_id
        self.seq = 0
        self._chunks: dict[int, list[dict]] = {}
        self._dirty: set[int] = set()
        self._dropped = -1  # chunks up to this index have been deleted (past the cap)
        self._lock = threading.Lock()
        self._writing = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None  # the run's: flushes happen there

    def add(self, event: dict) -> dict:
        with self._lock:
            self.seq += 1
            event = {**event, "seq": self.seq}
            index = _index(self.seq)
            self._chunks.setdefault(index, []).append(event)
            self._dirty.add(index)
        return event

    def _take(self) -> tuple[dict[str, Any], list[str]]:
        with self._lock:
            if not self._dirty:
                return {}, []
            mapping = {_chunk(self.run_id, i): list(self._chunks[i]) for i in self._dirty}
            mapping[_head(self.run_id)] = self.seq
            self._dirty.clear()
            current = _index(self.seq)
            for i in [i for i in self._chunks if i < current]:  # written in full: not needed here
                del self._chunks[i]
            stale = []
            keep_from = _index(max(1, self.seq - self.log.size + 1))
            if keep_from - 1 > self._dropped:
                stale = [_chunk(self.run_id, i) for i in range(self._dropped + 1, keep_from)]
                self._dropped = keep_from - 1
        return mapping, stale

    async def flush(self) -> None:
        async with self._writing:  # one write at a time, so an older chunk never lands last
            mapping, stale = self._take()
            if not mapping:
                return
            try:
                await asyncio.to_thread(self._write, mapping, stale)
            except Exception:
                logger.warning("Could not write the event log of background run %s", self.run_id, exc_info=True)

    def _write(self, mapping, stale) -> None:
        self.log.store.set_many(mapping, self.log.ttl)
        if stale:
            self.log.store.delete_many(stale)

    def start(self) -> None:
        async def flushing():
            while True:
                await asyncio.sleep(FLUSH_EVERY)
                await self.flush()

        self.loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(flushing())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.flush()


_writers: dict[str, Writer] = {}


def get_log() -> EventLog | None:
    """The configured log, or ``None`` when ``RUN_EVENT_LOG`` is ``"off"``."""
    from llm_kit.core.errors import ConfigError

    kind = conf.get("RUN_EVENT_LOG") or "off"
    size, ttl = conf.get("RUN_EVENT_LOG_SIZE"), conf.get("RUN_EVENT_LOG_TTL")
    if kind == "off":
        return None
    if kind == "memory":
        return EventLog(_memory, size, ttl)
    if kind in ("cache", "auto"):
        from django.conf import settings
        from django.core.cache import caches

        alias = conf.get("RUN_EVENT_LOG_CACHE")
        backend = settings.CACHES.get(alias, {}).get("BACKEND", "") if alias in settings.CACHES else ""
        if kind == "auto" and (not backend or backend in _LOCAL_CACHES):
            return EventLog(_memory, size, ttl)
        if alias not in settings.CACHES:
            raise ConfigError(f"RUN_EVENT_LOG_CACHE={alias!r} is not in CACHES.")
        return EventLog(CacheStore(caches[alias]), size, ttl)
    raise ConfigError(f'RUN_EVENT_LOG must be "auto", "cache", "memory" or "off", not {kind!r}.')


def open_writer(run_id: str) -> Writer | None:
    """Called by the process that runs ``run_id``, before its first event."""
    try:
        log = get_log()
    except Exception:  # a bad setting (the system check reports it) must not stop the run
        logger.exception("No event log for background run %s", run_id)
        return None
    if log is None:
        return None
    writer = _writers[run_id] = Writer(log, run_id)
    writer.start()
    return writer


async def close_writer(run_id: str) -> None:
    writer = _writers.pop(run_id, None)
    if writer is not None:
        await writer.close()


async def send(run_id: str, event: dict) -> dict:
    """Number ``event`` (when this process runs ``run_id``), keep it, and send it to watchers."""
    from llm_kit.background.bus import get_bus

    writer = _writers.get(run_id)
    if writer is not None and writer.loop is not None and writer.loop is not asyncio.get_running_loop():
        # Sent from another loop (a sync tool's publish(), say): hand it to the run's loop.
        future = asyncio.run_coroutine_threadsafe(send(run_id, event), writer.loop)
        return await asyncio.wrap_future(future)
    if writer is not None:
        event = writer.add(event)
        if event.get("event") != "token":  # tokens are written in batches; the rest at once
            await writer.flush()
    await get_bus().publish(run_id, event)
    return event
