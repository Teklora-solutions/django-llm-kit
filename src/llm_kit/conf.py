"""Settings for llm_kit, read from ``settings.LLM_KIT``.

Example::

    LLM_KIT = {
        # name -> dotted path to a StateGraph builder, a compiled graph,
        # or a factory ``fn(checkpointer) -> compiled graph``
        "GRAPHS": {"chat": "myapp.graphs:builder"},
        "DEFAULT_GRAPH": "chat",
        # Used when no catalogue entry / alias is requested.
        "DEFAULT_MODEL": "anthropic:claude-sonnet-5-5",
        # Static aliases, merged with the DB-backed catalogue (DB wins).
        "MODELS": {"fast": {"model": "anthropic:claude-haiku-5-5", "temperature": 0}},
        # Require an authenticated user for the HTTP / WebSocket endpoints.
        "REQUIRE_AUTH": True,
    }
"""

from __future__ import annotations

from typing import Any

from django.conf import settings

DEFAULTS: dict[str, Any] = {
    "GRAPHS": {},
    "DEFAULT_GRAPH": None,
    "DEFAULT_MODEL": None,
    "MODELS": {},
    "REQUIRE_AUTH": True,
    # Stream modes requested from LangGraph for SSE / WebSocket runs.
    "STREAM_MODES": ["messages", "updates"],
    # Seconds between SSE keep-alive comments (0 disables).
    "SSE_KEEPALIVE": 15,
}


def get(key: str) -> Any:
    user = getattr(settings, "LLM_KIT", {}) or {}
    if key in user:
        return user[key]
    return DEFAULTS[key]
