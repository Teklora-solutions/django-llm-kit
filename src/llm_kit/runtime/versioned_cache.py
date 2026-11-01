"""Process-local caches invalidated across processes through Django's cache.

Each process keeps its own copy; a version token in the shared cache tells
every process when to reload (works across workers with Redis/Memcached; with
the default LocMemCache it is per-process, which is still correct for one worker).

The token is random, not a counter: a counter restarts when the cache is cleared or
evicts the key, and could come back to the number a process already loaded at,
leaving it on stale data.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import Any

from django.core.cache import cache


class VersionedCache:
    def __init__(self, name: str) -> None:
        self.key = f"llm_kit:version:{name}"
        self._version: Any = object()
        self._value: Any = None
        self._lock = threading.Lock()

    def get(self, load: Callable[[], Any]) -> Any:
        version = cache.get(self.key, 0)
        if version != self._version:
            with self._lock:
                if version != self._version:
                    self._value = load()
                    self._version = version
        return self._value

    def bump(self) -> None:
        """Invalidate now, and again after the surrounding transaction commits, so a
        reload that happened mid-transaction can't keep uncommitted data."""
        from django.db import transaction

        self._new_version()
        transaction.on_commit(self._new_version)

    def _new_version(self) -> None:
        cache.set(self.key, uuid.uuid4().hex, None)
