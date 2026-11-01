"""Process-local caches invalidated across processes through Django's cache.

Each process keeps its own copy; a version number in the shared cache tells
every process when to reload (works across workers with Redis/Memcached; with
the default LocMemCache it is per-process, which is still correct for one worker).
"""

from __future__ import annotations

import threading
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

        self._incr()
        transaction.on_commit(self._incr)

    def _incr(self) -> None:
        try:
            cache.incr(self.key)
        except ValueError:
            cache.set(self.key, 1, None)
