"""Configuration, independent of any web framework.

Adapters call ``configure(source)`` with a mapping, or a zero-argument callable
returning one; it is read on every ``get`` so settings overrides in tests work.
The Django adapter wires this to ``settings.LLM_KIT``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

DEFAULTS: dict[str, Any] = {
    "GRAPHS": {},
    "DEFAULT_GRAPH": None,
    "DEFAULT_MODEL": None,
    "MODELS": {},
    "REQUIRE_AUTH": True,
    # Stream modes requested from LangGraph for streamed runs.
    "STREAM_MODES": ["messages", "updates"],
    # Seconds between SSE keep-alive comments (0 disables).
    "SSE_KEEPALIVE": 15,
    # Include exception type and message in client-facing error events.
    "EXPOSE_ERRORS": False,
}

_source: Callable[[], Mapping[str, Any]] = dict


def configure(source: Mapping[str, Any] | Callable[[], Mapping[str, Any]]) -> None:
    global _source
    _source = source if callable(source) else (lambda: source)


def get(key: str) -> Any:
    if key not in DEFAULTS:
        raise KeyError(f"Unknown llm-kit setting {key!r}")
    user = _source() or {}
    return user[key] if key in user else DEFAULTS[key]
