"""Merge per-source partial records into one effective record, with provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from llm_kit.core.catalogue.fields import (
    AUTHORITATIVE_SOURCES,
    CONFLICT_FIELDS,
    CONFLICT_TOLERANCE,
    DEFAULT_PRECEDENCE,
    FIELDS,
    PRICE_FIELDS,
    PRICE_PRECEDENCE,
)


@dataclass
class Merged:
    values: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)  # field -> source that supplied it
    conflicts: dict[str, dict[str, Any]] = field(default_factory=dict)  # field -> {source: value}
    hidden: bool = False


def _order(sources: dict[str, dict], precedence: tuple[str, ...]) -> list[str]:
    known = [s for s in precedence if s in sources]
    return known + sorted(s for s in sources if s not in precedence)


def _differs(a: Any, b: Any) -> bool:
    try:
        da, db = Decimal(str(a)), Decimal(str(b))
    except Exception:
        return a != b
    if da == db:
        return False
    base = max(abs(da), abs(db))
    return base != 0 and abs(da - db) / base > CONFLICT_TOLERANCE


def merge(
    source_values: dict[str, dict[str, Any]],
    override: dict[str, Any] | None = None,
    *,
    price_precedence: tuple[str, ...] | None = None,
    default_precedence: tuple[str, ...] | None = None,
    provider: str | None = None,
) -> Merged:
    """Pick each field from the highest-priority source that has it.

    ``override`` (an admin override) beats every source. Its special key
    ``hidden`` hides the model from selection without deleting it. When
    ``provider`` has its own source (``AUTHORITATIVE_SOURCES``), that source
    ranks right after overrides.
    """
    price_precedence = price_precedence or PRICE_PRECEDENCE
    default_precedence = default_precedence or DEFAULT_PRECEDENCE
    if provider in AUTHORITATIVE_SOURCES:
        own = AUTHORITATIVE_SOURCES[provider]

        def promote(order: tuple[str, ...]) -> tuple[str, ...]:
            return ("override", *own, *(s for s in order if s != "override" and s not in own))

        price_precedence, default_precedence = promote(price_precedence), promote(default_precedence)
    sources = dict(source_values)
    override = dict(override or {})
    out = Merged(hidden=bool(override.pop("hidden", False)))
    if override:
        sources["override"] = {k: v for k, v in override.items() if k in FIELDS}

    price_order = _order(sources, price_precedence)
    default_order = _order(sources, default_precedence)
    for f in FIELDS:
        order = price_order if f in PRICE_FIELDS else default_order
        for src in order:
            v = sources[src].get(f)
            if v is not None:
                out.values[f] = v
                out.provenance[f] = src
                break
        if f in CONFLICT_FIELDS:
            seen = {s: sources[s][f] for s in order if s != "override" and sources[s].get(f) is not None}
            vals = list(seen.values())
            if len(vals) > 1 and any(_differs(vals[0], v) for v in vals[1:]):
                out.conflicts[f] = seen
    return out
