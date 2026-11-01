"""Looking models up in the catalogue."""

from __future__ import annotations

import os
from datetime import timedelta
from decimal import Decimal
from typing import Any

from django.utils import timezone

from llm_kit.core.catalogue.fields import CAPABILITY_FIELDS, PRICE_FIELDS, normalise_provider
from llm_kit.core.catalogue.match import REQUIREMENT_KEYS
from llm_kit.models import CatalogueChange, CatalogueModel, CatalogueProvider

# Providers that make their own models; preferred when a bare model id is ambiguous.
FIRST_PARTY = ("anthropic", "openai", "gemini", "mistral", "xai", "deepseek", "cohere", "moonshot", "dashscope")


def _split(ref: str) -> tuple[str | None, str]:
    if "/" in ref:
        provider, model_id = ref.split("/", 1)
        return normalise_provider(provider), model_id
    if ":" in ref:  # init_chat_model style "anthropic:claude-..."
        provider, model_id = ref.split(":", 1)
        return normalise_provider(provider), model_id
    return None, ref


def active():
    return CatalogueModel.objects.filter(removed=False, hidden=False)


def get(ref: str, *, include_removed: bool = False) -> CatalogueModel | None:
    """Look up ``provider/model_id`` (LiteLLM style), ``provider:model_id``, or a bare model id.

    A prefix is only taken as a provider if that lookup finds something, so ids
    that contain ``:`` or ``/`` themselves (Bedrock ``...-v1:0``, OpenAI
    ``ft:gpt-5:acme``) still resolve. A bare id that several providers serve
    resolves to the first-party provider if there is one, otherwise ``None``.
    """
    qs = CatalogueModel.objects.all() if include_removed else CatalogueModel.objects.filter(removed=False)
    ref = ref.strip()
    provider, model_id = _split(ref)
    if provider:
        row = qs.filter(provider=provider, model_id=model_id).first()
        if row:
            return row
    candidates = list(qs.filter(model_id=ref)[:50])
    if len(candidates) == 1:
        return candidates[0]
    for p in FIRST_PARTY:
        for c in candidates:
            if c.provider == p:
                return c
    return None


def search(
    *,
    kind: str | None = "chat",
    provider: str | None = None,
    requires: dict[str, Any] | None = None,
    max_price_input: Decimal | float | str | None = None,
    max_price_output: Decimal | float | str | None = None,
    usable_only: bool = False,
    include_hidden: bool = False,
):
    """Active models matching filters, as a QuerySet.

    ``requires`` uses the same keys as touchpoints, e.g.
    ``{"tool_call": True, "input": ["image"], "min_context": 200000}``.
    Unknown capabilities (null) never match a requirement.
    """
    qs = CatalogueModel.objects.filter(removed=False)
    if not include_hidden:
        qs = qs.filter(hidden=False)
    if kind:
        qs = qs.filter(kind=kind)
    if provider:
        qs = qs.filter(provider=normalise_provider(provider))
    for key, want in (requires or {}).items():
        if key not in REQUIREMENT_KEYS:
            raise ValueError(f"Unknown requirement {key!r}. Use one of: {', '.join(REQUIREMENT_KEYS)}.")
        if key in CAPABILITY_FIELDS:
            qs = qs.filter(**{key: bool(want)})
        elif key == "kind":
            qs = qs.filter(kind=want)
        elif key in ("input", "output"):
            for m in [want] if isinstance(want, str) else want:
                qs = qs.filter(**{f"{key}_modalities__contains": f",{m},"})
        elif key == "min_context":
            qs = qs.filter(context_window__gte=int(want))
        elif key == "min_output":
            qs = qs.filter(max_output__gte=int(want))
    if max_price_input is not None:
        qs = qs.filter(price_input__lte=Decimal(str(max_price_input)))
    if max_price_output is not None:
        qs = qs.filter(price_output__lte=Decimal(str(max_price_output)))
    if usable_only:
        qs = qs.filter(provider__in=[p for p in qs.values_list("provider", flat=True).distinct() if has_credentials(p)])
    return qs


def has_credentials(provider: str) -> bool:
    """True if an API key for ``provider`` is in the environment (env names come from models.dev)."""
    provider = normalise_provider(provider)
    env = CatalogueProvider.objects.filter(provider=provider).values_list("env", flat=True).first() or []
    env = [*env, f"{provider.upper()}_API_KEY", f"{provider.upper()}_API_KEYS"]
    return any(os.environ.get(e) for e in env)


def prices(ref: str) -> dict[str, Decimal | None] | None:
    """USD per 1M tokens for a model, or None if unknown."""
    row = get(ref)
    if row is None:
        return None
    return {f.removeprefix("price_"): getattr(row, f) for f in PRICE_FIELDS}


def recent_changes(days: int = 7):
    return CatalogueChange.objects.filter(synced_at__gte=timezone.now() - timedelta(days=days))
