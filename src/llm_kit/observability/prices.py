"""Catalogue prices for the recorder, cached briefly (prices change hours apart, calls happen constantly)."""

from __future__ import annotations

import threading
import time
from typing import Any

_TTL = 300.0
_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
_lock = threading.Lock()


def lookup(ref: str) -> dict[str, Any] | None:
    if not ref:
        return None
    now = time.monotonic()
    hit = _cache.get(ref)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    from llm_kit import catalogue

    row = catalogue.get(ref, include_removed=True)
    prices = None
    if row is not None:
        prices = {k: v for k, v in (row.values or {}).items() if k.startswith("price_")}
    with _lock:
        if len(_cache) > 5000:
            _cache.clear()
        _cache[ref] = (now, prices)
    return prices


def clear() -> None:
    _cache.clear()
