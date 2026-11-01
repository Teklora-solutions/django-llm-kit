"""The normalised catalogue record: field names, merge precedence, id helpers.

A model is keyed by ``(provider, model_id)`` using LiteLLM's provider names,
so ``f"{provider}/{model_id}"`` is a valid LiteLLM model string.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

Key = tuple[str, str]

PRICE_FIELDS = ("price_input", "price_output", "price_cache_read", "price_cache_write", "price_reasoning")
"""USD per 1M tokens, stored as decimal strings."""

CAPABILITY_FIELDS = (
    "tool_call",
    "structured_output",
    "reasoning",
    "attachments",
    "temperature",
    "prompt_caching",
)

FIELDS = (
    "name",
    "family",
    "kind",  # chat | embedding | image | audio | rerank | other
    "canonical_id",
    "input_modalities",
    "output_modalities",
    "context_window",
    "max_input",
    "max_output",
    *CAPABILITY_FIELDS,
    "open_weights",
    *PRICE_FIELDS,
    "price_tiers",  # [{"above_tokens": int, "price_input": str, ...}]
    "release_date",
    "knowledge_cutoff",
    "status",  # "" | alpha | beta | deprecated
    "deprecation_date",
)

# Highest priority first. "override" is an admin override and always wins.
PRICE_PRECEDENCE = ("override", "litellm", "models.dev", "openrouter", "openrouter-embeddings")
DEFAULT_PRECEDENCE = ("override", "models.dev", "litellm", "openrouter", "openrouter-embeddings")

# A provider's own source beats aggregators for that provider's models
# (e.g. OpenRouter's API is the price OpenRouter actually bills).
AUTHORITATIVE_SOURCES = {
    "openrouter": ("openrouter", "openrouter-embeddings"),
}

# Fields where sources disagreeing is worth showing to an admin.
CONFLICT_FIELDS = (*PRICE_FIELDS, "context_window", "max_output")
CONFLICT_TOLERANCE = Decimal("0.01")  # 1% relative difference

# Other provider spellings -> LiteLLM provider names.
PROVIDER_ALIASES = {
    "google": "gemini",
    "google-vertex": "vertex_ai",
    "google-vertex-anthropic": "vertex_ai",
    "amazon-bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "togetherai": "together_ai",
    "together": "together_ai",
    "fireworks-ai": "fireworks_ai",
    "fireworks": "fireworks_ai",
    "nvidia": "nvidia_nim",
    "moonshotai": "moonshot",
    "alibaba": "dashscope",
    "vercel": "vercel_ai_gateway",
    "github-copilot": "github_copilot",
    "x-ai": "xai",
    "mistralai": "mistral",
}


def normalise_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p.startswith("vertex_ai-"):  # LiteLLM's vertex_ai-language-models, vertex_ai-anthropic_models, ...
        return "vertex_ai"
    return PROVIDER_ALIASES.get(p, p)


def per_mtok(value: Any, *, per_token: bool) -> str | None:
    """A price as a decimal string in USD per 1M tokens. Negative or missing -> None."""
    if value is None or value == "":
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if d < 0:  # OpenRouter uses -1 for "varies" on router models
        return None
    if per_token:
        d *= 1_000_000
    return format(d.normalize(), "f")  # plain notation, no trailing zeros: "0.15", "3"


def model_ref(key: Key) -> str:
    return f"{key[0]}/{key[1]}"


def clean(partial: dict[str, Any]) -> dict[str, Any]:
    """Drop None/empty values and unknown fields."""
    return {k: v for k, v in partial.items() if k in FIELDS and v is not None and v != "" and v != []}
