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
    "kind",  # one of KINDS
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
    "price_units",  # {"output_cost_per_image": "0.04", ...}: USD per unit (media models); see MEDIA_PRICE_KEYS
    "release_date",
    "knowledge_cutoff",
    "status",  # "" | alpha | beta | deprecated
    "deprecation_date",
)

KINDS = (
    "chat",
    "embedding",
    "image",  # makes images from a prompt
    "image_edit",  # changes an image it is given
    "video",
    "speech",  # text to speech
    "transcription",  # speech to text
    "ocr",
    "realtime",  # live two-way audio
    "music",
    "rerank",
    "moderation",
    "other",
)
"""What a model does (``requires={"kind": ...}``)."""


def is_media_price(key: str) -> bool:
    """A LiteLLM price key (``output_cost_per_image``, ``input_cost_per_second``,
    ``output_cost_per_second_720p``, ``ocr_cost_per_page``, ...) that prices media by a unit other
    than plain text tokens. Kept as given, in USD per unit, under ``price_units``."""
    if "_cost_per_" not in key or not key.startswith(("input_", "output_", "ocr_", "annotation_")):
        return False
    if any(word in key for word in ("_batches", "_priority", "_flex", "_above_", "dbu", "reasoning")):
        return False
    return key not in ("input_cost_per_token", "output_cost_per_token")


# Highest priority first. "override" is an admin override and always wins.
PRICE_PRECEDENCE = ("override", "litellm", "models.dev", "openrouter", "openrouter-embeddings")
DEFAULT_PRECEDENCE = ("override", "models.dev", "litellm", "openrouter", "openrouter-embeddings")
# Fields whose sources rank differently. LiteLLM says what a media model does (its "mode");
# the others list most of them as chat models.
FIELD_PRECEDENCE = {"kind": ("override", "litellm", "models.dev", "openrouter", "openrouter-embeddings")}

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
    return {k: v for k, v in partial.items() if k in FIELDS and v is not None and v != "" and v != [] and v != {}}
