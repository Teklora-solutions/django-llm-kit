"""Check catalogue records against requirements, e.g. ``{"tool_call": True, "input": ["image"]}``."""

from __future__ import annotations

from typing import Any

from llm_kit.core.catalogue.fields import CAPABILITY_FIELDS

REQUIREMENT_KEYS = (*CAPABILITY_FIELDS, "kind", "input", "output", "min_context", "min_output")


def unmet(values: dict[str, Any], requires: dict[str, Any] | None) -> list[str]:
    """Return why ``values`` fails ``requires`` (empty list = satisfied).

    Unknown capability values (``None``) count as unmet: a model is only
    accepted when the catalogue positively says it can do the thing.
    """
    reasons = []
    for key, want in (requires or {}).items():
        if key not in REQUIREMENT_KEYS:
            raise ValueError(f"Unknown requirement {key!r}. Use one of: {', '.join(REQUIREMENT_KEYS)}.")
        if key in CAPABILITY_FIELDS:
            if bool(values.get(key)) != bool(want) or (want and values.get(key) is None):
                reasons.append(f"{key} is {values.get(key)!r}, need {want!r}")
        elif key == "kind":
            if values.get("kind") != want:
                reasons.append(f"kind is {values.get('kind')!r}, need {want!r}")
        elif key in ("input", "output"):
            have = set(values.get(f"{key}_modalities") or [])
            missing = set([want] if isinstance(want, str) else want) - have
            if missing:
                reasons.append(f"{key} modalities lack {sorted(missing)}")
        elif key in ("min_context", "min_output"):
            field = "context_window" if key == "min_context" else "max_output"
            if (values.get(field) or 0) < int(want):
                reasons.append(f"{field} is {values.get(field)!r}, need >= {want}")
    return reasons
