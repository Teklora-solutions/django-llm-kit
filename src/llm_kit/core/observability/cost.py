"""Price a call from catalogue prices (USD per 1M tokens)."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any

PriceLookup = Callable[[str], dict[str, Any] | None]
"""ref -> {"price_input": "3", "price_output": "15", "price_cache_read": ..., "price_tiers": [...]} or None."""

_MTOK = Decimal(1_000_000)


def _d(value: Any) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def price_call(
    prices: dict[str, Any] | None,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> Decimal | None:
    """USD cost, or None when the model has no known input/output price.

    ``input_tokens`` includes cached tokens (LangChain's convention); cached
    reads and writes are billed at their own rates when the catalogue has them.
    Context-size tiers (e.g. above 200k tokens) apply to the whole call.
    """
    if not prices:
        return None
    p = dict(prices)
    for tier in sorted(p.get("price_tiers") or [], key=lambda t: t.get("above_tokens", 0)):
        if input_tokens > tier.get("above_tokens", 0):
            p.update({k: v for k, v in tier.items() if k.startswith("price_") and v is not None})
    p_in, p_out = _d(p.get("price_input")), _d(p.get("price_output"))
    if p_in is None or p_out is None:
        return None
    p_read = _d(p.get("price_cache_read"))
    p_write = _d(p.get("price_cache_write"))
    p_reason = _d(p.get("price_reasoning"))

    cached_read = cache_read_tokens if p_read is not None else 0
    cached_write = cache_write_tokens if p_write is not None else 0
    plain_in = max(input_tokens - cached_read - cached_write, 0)
    # Reasoning tokens are part of output_tokens; bill them separately only if priced differently.
    reason = reasoning_tokens if p_reason is not None else 0
    plain_out = max(output_tokens - reason, 0)
    total = (
        plain_in * p_in
        + cached_read * (p_read or 0)
        + cached_write * (p_write or 0)
        + plain_out * p_out
        + reason * (p_reason or 0)
    )
    return (total / _MTOK).quantize(Decimal("0.0000000001"))


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token) when a provider sends no usage."""
    return max(1, len(text) // 4) if text else 0
