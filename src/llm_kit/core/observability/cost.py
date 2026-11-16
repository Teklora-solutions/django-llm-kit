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


# ------------------------------------------------------------------ media

UNITS = (
    "images",  # images made
    "size",  # "1024x1024"
    "input_images",  # images given (an edit, image-to-video)
    "input_seconds",  # audio heard (transcription)
    "output_seconds",  # audio or video made
    "resolution",  # "720p", "1080p", "4k"
    "characters",  # text read aloud
    "videos",  # clips made
    "pages",  # pages read (OCR)
    "annotated_pages",  # pages given structured annotations (OCR)
    "input_image_tokens",
    "output_image_tokens",
    "input_audio_tokens",
    "output_audio_tokens",
)
"""What a media call measures (``CallRecord.units``); a backend may add its own."""


def _size_labels(size: Any) -> list[str]:
    """``"1024x1792"`` -> the labels LiteLLM prices image sizes under: ``["1792", "2K"]``."""
    try:
        w, h = (int(x) for x in str(size).lower().split("x"))
    except (TypeError, ValueError):
        return [str(size)] if size else []
    side = max(w, h)
    k = "0.5K" if side <= 512 else "1K" if side <= 1024 else "2K" if side <= 2048 else "4K"
    return [str(side), k]


def _pixels(size: Any) -> int:
    try:
        w, h = (int(x) for x in str(size).lower().split("x"))
    except (TypeError, ValueError):
        return 0
    return w * h


def price_media(
    prices: dict[str, Any] | None, units: dict[str, Any], *, input_tokens: int = 0, output_tokens: int = 0
) -> Decimal | None:
    """USD for a media call from the catalogue's ``price_units`` (USD per unit, LiteLLM's names:
    ``output_cost_per_image``, ``input_cost_per_second``, ``output_cost_per_second_720p``,
    ``input_cost_per_character``, ...) and the text-token prices. None when nothing it used has
    a price."""
    if not prices:
        return None
    per = {k: _d(v) for k, v in (prices.get("price_units") or {}).items()}
    total, priced = Decimal(0), False

    def first(*keys: str) -> Decimal | None:
        for key in keys:
            if per.get(key) is not None:
                return per[key]
        return None

    def add(price: Decimal | None, quantity: Any) -> None:
        nonlocal total, priced
        if price is not None and quantity:
            total += price * Decimal(str(quantity))
            priced = True

    res = str(units.get("resolution") or "").lower()
    if units.get("output_seconds"):
        add(
            first(
                *([f"output_cost_per_second_{res}"] if res else []),
                "output_cost_per_video_per_second",
                "output_cost_per_second",
            ),
            units["output_seconds"],
        )
    add(per.get("input_cost_per_second"), units.get("input_seconds"))
    add(per.get("input_cost_per_character"), units.get("characters"))
    images = units.get("images")
    if images:
        labels = _size_labels(units.get("size"))
        per_image = first(*(f"output_cost_per_image_{label}" for label in labels), "output_cost_per_image")
        if per_image is not None:
            add(per_image, images)
        elif per.get("output_cost_per_pixel") is not None:
            add(per["output_cost_per_pixel"], _pixels(units.get("size")) * int(images))
    add(per.get("input_cost_per_image"), units.get("input_images"))
    add(first("output_cost_per_video", "output_cost_per_video_clip"), units.get("videos"))
    add(first("ocr_cost_per_page", "input_cost_per_page", "output_cost_per_page"), units.get("pages"))
    add(per.get("annotation_cost_per_page"), units.get("annotated_pages"))
    for direction in ("input", "output"):
        for medium in ("image", "audio"):
            add(per.get(f"{direction}_cost_per_{medium}_token"), units.get(f"{direction}_{medium}_tokens"))
    p_in, p_out = _d(prices.get("price_input")), _d(prices.get("price_output"))
    if input_tokens and p_in is not None:
        add(p_in / _MTOK, input_tokens)
    if output_tokens and p_out is not None:
        add(p_out / _MTOK, output_tokens)
    return total.quantize(Decimal("0.0000000001")) if priced else None
