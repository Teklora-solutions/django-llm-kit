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
    # Checkpointer for every graph: None (DjangoSaver), "postgres" (LangGraph's PostgresSaver on
    # Django's database), a saver/class/factory (dotted path), or {"sync": ..., "async": ...}.
    "CHECKPOINTER": None,
    # Move a visitor's threads to their account when they sign in (Django's user_logged_in).
    "ADOPT_SESSION_THREADS": True,
    # Stream modes requested from LangGraph for streamed runs ("custom" carries subagents' messages).
    "STREAM_MODES": ["messages", "updates", "custom"],
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
    # Embeddings (touchpoint.embed): "auto", "litellm", "langchain", or a dotted path to
    # fn(ref, params, api_keys) -> Embedder | LangChain Embeddings.
    "EMBEDDING_BACKEND": "auto",
    # Extra kwargs for litellm.Router when a provider has several keys.
    "LITELLM_ROUTER": {"routing_strategy": "simple-shuffle", "num_retries": 0},
    # Waiting out a short rate limit (or overload) on a model before falling back to the next:
    # {"attempts": waits per model, "max_wait": longest wait in seconds}. Per touchpoint:
    # touchpoint(..., retry=RetryPolicy.BATCH); per call: model(retry=...) or agents.run(retry=...).
    "RETRY": {"attempts": 1, "max_wait": 6.0},
    # Keys for encrypting ProviderKey rows: a list, newest first (rotation).
    # Each is a Fernet key or any passphrase (hashed to one).
    "ENCRYPTION_KEYS": None,
    # Dotted path to fn(provider) -> secret | [secrets] | None, consulted before stored keys and env vars.
    "CREDENTIALS": None,
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
    # Scoped settings (llm_kit.scoped): who may change each scope's values,
    # {"platform"|"tenant"|"user": "dotted.path:fn(user, tenant) -> bool"}.
    "SCOPE_PERMISSIONS": {},
    # Dotted path to fn(tenant_id) -> users (or ids) in that tenant: lets tenant managers set one
    # member's budget, and gives settings pages a member list.
    "TENANT_MEMBERS": None,
    # Bring your own key: scopes that may store provider keys ("tenant", "user"; empty = off),
    # and whose keys touchpoints use unless Settable(keys=...) says (a KeyPolicy value).
    "BYOK_SCOPES": (),
    "KEY_POLICY": "platform_only",
    # Budgets saved per tenant, member or user (llm_kit.scoped.budgets): the periods they may
    # use (names, or {"name": "dotted.path:fn(now) -> start"}), and who may set each kind:
    # {"tenant" | "member" | "user": ["platform" | "tenant" | "user", ...]}. Empty = nobody.
    "BUDGET_PERIODS": ("day", "month"),
    "BUDGET_EDITORS": {},
    # Spend shared across processes for budgets: "auto" (Django's cache unless it is per-process),
    # "cache" (always), or None (each process sees others' spend every 30 s).
    "SPEND_COUNTER": "auto",
    # Model tools: dotted path to fn(user) -> tenant (instance or id) for the run's caller.
    "TENANT_RESOLVER": None,
    # Agents: how deep subagents may nest (0 = no delegation), how many hand-offs one request
    # may make in total, and seconds before a run stops starting new model calls.
    "AGENT_MAX_DEPTH": 2,
    "AGENT_MAX_DELEGATIONS": 8,
    "AGENT_TIMEOUT": 300,
    # Tools loaded on demand (find_tools): agents with more tools than this defer them unless
    # they declare defer_tools; how many found tools stay bound (most recent first); and an
    # embedding touchpoint that lets find_tools search by meaning as well as by words.
    "DEFER_TOOLS_ABOVE": 20,
    "MAX_LOADED_TOOLS": 15,
    "TOOL_SEARCH_TOUCHPOINT": None,
    # USD one request (an agent and its subagents) may spend before its agents stop starting
    # model calls; None = no limit. Per run: agents.run(..., budget=0.5).
    "AGENT_BUDGET": None,
    # USD per day, from recorded calls: {"user": 5, "tenant": 50}. A request whose user or
    # tenant is over is refused before it starts.
    "DAILY_BUDGETS": {},
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
