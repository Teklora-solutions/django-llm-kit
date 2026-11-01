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
    "REQUIRE_AUTH": True,
    # Stream modes requested from LangGraph for streamed runs.
    "STREAM_MODES": ["messages", "updates"],
    # Seconds between SSE keep-alive comments (0 disables).
    "SSE_KEEPALIVE": 15,
    # Include exception type and message in client-facing error events.
    "EXPOSE_ERRORS": False,
    # Catalogue: source names / dotted paths (None = all built-ins), and the
    # age after which `llm_kit_status` reports the catalogue as stale.
    "CATALOGUE_SOURCES": None,
    "CATALOGUE_MAX_AGE_HOURS": 24,
    # Model construction: "auto" (LiteLLM if installed, else LangChain's
    # init_chat_model), "litellm", "langchain", or a dotted path to
    # fn(ref: str, params: dict, api_keys: list[str]) -> BaseChatModel.
    "MODEL_BACKEND": "auto",
    # Extra kwargs for litellm.Router when a provider has several keys.
    "LITELLM_ROUTER": {"routing_strategy": "simple-shuffle", "num_retries": 0},
    # Keys for encrypting ProviderKey rows: a list, newest first (rotation).
    # Each is a Fernet key or any passphrase (hashed to one).
    "ENCRYPTION_KEYS": None,
    # Observability: record every model call, tool call and run.
    "OBSERVABILITY": True,
    # Where records go: "database" (Django app), "log", "otel", or dotted paths to sink classes.
    "RECORD_SINKS": ["database", "log"],
    # Store prompts/responses (off: only hashes). Per touchpoint: touchpoint(..., capture=True).
    "CAPTURE_CONTENT": False,
    # Days to keep captured content and call rows (used by llm_kit_prune).
    "CONTENT_RETENTION_DAYS": 7,
    "CALL_RETENTION_DAYS": 90,
    # Circuit breaker per provider: {"failures": 5, "window": 60, "cooldown": 60, "quota_cooldown": 900}
    "CIRCUIT_BREAKER": {},
    # Monitoring API permission (dotted DRF permission class; default: staff only).
    "MONITORING_PERMISSION": None,
    # Bearer token accepted by /llm/metrics/ (Prometheus); staff sessions always work.
    "METRICS_TOKEN": None,
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
