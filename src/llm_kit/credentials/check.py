"""Checking a provider key works: one tiny call (a few tokens) with that key alone.

Used by "Test" in the admin, ``POST llm/api/keys/<id>/test/``, the keys template, and
``add_key(..., verify=True)``. The model is the cheapest one your touchpoints use for that
provider, else the cheapest chat model the catalogue lists for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_kit.core.catalogue.fields import normalise_provider

# A refusal for these reasons still proves the provider accepted the key.
_KEY_ACCEPTED = ("rate_limit", "overloaded", "quota_exhausted", "context_length", "bad_request")


@dataclass
class KeyCheck:
    ok: bool
    model: str = ""
    error: str = ""  # classified reason when the key was refused: auth, permission, ...
    message: str = ""


def model_for(provider: str) -> str | None:
    """A cheap chat model of ``provider`` to try a key on."""
    from llm_kit import catalogue
    from llm_kit.core.touchpoints import all_touchpoints
    from llm_kit.core.touchpoints.env import provider_of

    provider = normalise_provider(provider)
    used = [ref for tp in all_touchpoints() for ref in (tp.default, *tp.fallbacks) if provider_of(ref) == provider]
    rows = catalogue.search(kind="chat", provider=provider).exclude(price_input=None).order_by("price_input")
    if used:
        cheapest = rows.filter(model_id__in=[r.split("/", 1)[1] for r in used]).first()
        return cheapest.ref if cheapest is not None else used[0]
    row = rows.first()
    return row.ref if row is not None else None


def check_key(provider: str, secret: str) -> KeyCheck:
    """Call a model of ``provider`` once with ``secret`` (and no other key)."""
    from langchain_core.messages import HumanMessage

    from llm_kit.core.observability.classify import classify_error
    from llm_kit.core.touchpoints.backends import get_factory

    ref = model_for(provider)
    if ref is None:
        return KeyCheck(ok=False, error="no_model", message=f"No model of {provider!r} is known to test the key on.")
    try:
        get_factory(None)(ref, {"max_tokens": 5, "temperature": 0}, [secret]).invoke([HumanMessage("Reply OK.")])
    except Exception as e:
        kind = classify_error(e)
        if kind in _KEY_ACCEPTED:
            return KeyCheck(ok=True, model=ref, message=f"The key works ({kind} on the test call).")
        return KeyCheck(ok=False, model=ref, error=kind or "error", message=f"{type(e).__name__}: {e}"[:300])
    return KeyCheck(ok=True, model=ref, message="The key works.")


def check_stored(row) -> KeyCheck:
    """Check a ``ProviderKey`` and record the result on it."""
    from django.utils import timezone

    result = check_key(row.provider, row.get_key())
    row.last_checked_at = timezone.now()
    row.last_check_ok = result.ok
    row.last_check_error = "" if result.ok else f"{result.error}: {result.message}"[:255]
    row.save(update_fields=["last_checked_at", "last_check_ok", "last_check_error", "updated_at"])
    return result
