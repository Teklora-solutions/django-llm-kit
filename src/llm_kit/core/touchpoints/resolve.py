"""Turn a touchpoint + an optional client request into the model(s) to call, with reasons."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from llm_kit.core.catalogue.fields import normalise_provider
from llm_kit.core.catalogue.match import unmet
from llm_kit.core.errors import ModelNotAllowed, TouchpointDisabled
from llm_kit.core.touchpoints.env import ModelInfo, ResolutionEnv, has_credentials, provider_of
from llm_kit.core.touchpoints.policy import ClientChoice, ServerChoice, matches_any

if TYPE_CHECKING:
    from llm_kit.core.touchpoints.touchpoint import Touchpoint


@dataclass
class Resolution:
    touchpoint: str
    model: str
    fallbacks: list[str]
    params: dict[str, Any]
    chosen_by: Literal["client", "admin", "code"]
    requested: str | None = None
    notes: list[str] = field(default_factory=list)  # decisions, in order
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (model, why)

    @property
    def candidates(self) -> list[str]:
        return [self.model, *self.fallbacks]


def canonical(ref: str, env: ResolutionEnv) -> tuple[str, ModelInfo | None]:
    """Normalise ``provider:model`` / ``google/...`` spellings and find catalogue info."""
    ref = ref.strip()
    info = env.lookup(ref)
    if info:
        return info.ref, info
    for sep in ("/", ":"):
        if sep in ref:
            provider, model_id = ref.split(sep, 1)
            return f"{normalise_provider(provider)}/{model_id}", None
    return ref, None


def _price_ok(info: ModelInfo | None, field_: str, cap) -> str | None:
    if cap is None:
        return None
    price = (info.values if info else {}).get(field_)
    if price is None:
        return f"{field_} unknown (cap {cap})"
    if Decimal(str(price)) > Decimal(str(cap)):
        return f"{field_} {price} exceeds cap {cap}"
    return None


def check_client_pick(
    tp: Touchpoint,
    ref: str,
    info: ModelInfo | None,
    choice: ClientChoice,
    allow_narrowed: list[str] | None,
    env: ResolutionEnv,
) -> list[str]:
    """Why a client's pick is not allowed (empty list = allowed). Same rules as the picker."""
    why = []
    if not matches_any(ref, choice.allow):
        why.append(f"not in allowed patterns {list(choice.allow)}")
    if allow_narrowed is not None and not matches_any(ref, allow_narrowed):
        why.append(f"not in admin-allowed patterns {allow_narrowed}")
    requires = {**tp.requires, **choice.requires}
    needs_catalogue = requires or choice.max_price_input is not None or choice.max_price_output is not None
    if info is None:
        if needs_catalogue:
            why.append("not in the model catalogue, so its capabilities and price can't be checked")
    else:
        if info.hidden:
            why.append("hidden by an admin")
        why += unmet(info.values, requires)
        why += [r for r in (_price_ok(info, "price_input", choice.max_price_input),) if r]
        why += [r for r in (_price_ok(info, "price_output", choice.max_price_output),) if r]
    if not has_credentials(env, ref):
        why.append("no API key configured for its provider")
    return why


def resolve(tp: Touchpoint, requested: str | None, env: ResolutionEnv) -> Resolution:
    ov = env.override(tp.name)
    if ov and ov.disabled:
        raise TouchpointDisabled(f"Touchpoint {tp.name!r} is disabled by an admin.")

    default = (ov.default if ov and ov.default else None) or tp.default
    fallbacks = list(ov.fallbacks) if ov and ov.fallbacks is not None else list(tp.fallbacks)
    params = {**tp.params, **(ov.params if ov else {})}
    chosen_by: Literal["client", "admin", "code"] = "admin" if ov and ov.default else "code"
    notes: list[str] = []
    if chosen_by == "admin":
        notes.append(f"admin override sets default to {default}")

    primary, _ = canonical(default, env)
    if requested:
        req_ref, info = canonical(requested, env)
        choice = tp.choice
        if isinstance(choice, ServerChoice):
            msg = f"client requested {req_ref}, but {tp.name!r} lets the server choose"
            if choice.on_requested == "reject":
                raise ModelNotAllowed(f"Touchpoint {tp.name!r} does not accept a model choice.", [msg])
            notes.append(msg + "; ignored")
        else:
            why = check_client_pick(tp, req_ref, info, choice, ov.allow if ov else None, env)
            if why:
                if choice.on_disallowed == "reject":
                    raise ModelNotAllowed(f"Model {req_ref!r} is not allowed for {tp.name!r}: {'; '.join(why)}.", why)
                notes.append(f"client requested {req_ref}, not allowed ({'; '.join(why)}); using default")
            else:
                primary, chosen_by = req_ref, "client"
                notes.append(f"client requested {req_ref}; allowed")
                if not choice.use_fallbacks:
                    fallbacks = []

    # Fallbacks: drop duplicates, models without credentials, and ones failing requires.
    skipped: list[tuple[str, str]] = []
    usable_fallbacks: list[str] = []
    for fb in fallbacks:
        ref, info = canonical(fb, env)
        if ref == primary or ref in usable_fallbacks:
            continue
        if not has_credentials(env, ref):
            skipped.append((ref, "no API key configured"))
        elif info is not None and (why := unmet(info.values, tp.requires)):
            skipped.append((ref, "; ".join(why)))
        else:
            usable_fallbacks.append(ref)

    if not has_credentials(env, primary) and chosen_by != "client" and usable_fallbacks:
        skipped.append((primary, "no API key configured"))
        notes.append(f"{primary} has no API key; promoted fallback {usable_fallbacks[0]}")
        primary = usable_fallbacks.pop(0)

    # Providers whose circuit is open (recent failures) go to the back of the line.
    circuit_open = getattr(env, "circuit_open", None)
    if circuit_open is not None:
        down = [ref for ref in [primary, *usable_fallbacks] if circuit_open(provider_of(ref))]
        if down and len(down) <= len(usable_fallbacks):  # at least one healthy candidate remains
            healthy = [ref for ref in [primary, *usable_fallbacks] if ref not in down]
            for ref in down:
                skipped.append((ref, "provider circuit open (recent failures)"))
            if primary in down:
                notes.append(f"{primary}: provider circuit open; using {healthy[0]}")
            primary, usable_fallbacks = healthy[0], healthy[1:]
        elif down:
            notes.append("all candidates' provider circuits are open; trying anyway")

    return Resolution(
        touchpoint=tp.name,
        model=primary,
        fallbacks=usable_fallbacks,
        params=params,
        chosen_by=chosen_by,
        requested=requested,
        notes=notes,
        skipped=skipped,
    )


def problems(tp: Touchpoint, env: ResolutionEnv) -> list[str]:
    """Configuration problems worth showing to a developer (``llm_kit_touchpoints``)."""
    out = []
    ov = env.override(tp.name)
    default = (ov.default if ov and ov.default else None) or tp.default
    fallbacks = list(ov.fallbacks) if ov and ov.fallbacks is not None else list(tp.fallbacks)
    for role, model in [("default", default), *(("fallback", f) for f in fallbacks)]:
        ref, info = canonical(model, env)
        if info is None:
            out.append(f"{role} {ref} is not in the catalogue")
            continue
        v = info.values
        if why := unmet(v, tp.requires):
            out.append(f"{role} {ref}: {'; '.join(why)}")
        if v.get("status") == "deprecated" or v.get("deprecation_date"):
            out.append(f"{role} {ref} is deprecated ({v.get('deprecation_date') or 'no date'})")
        if v.get("price_input") is None:
            out.append(f"{role} {ref} has no known price")
        if not has_credentials(env, ref):
            out.append(f"{role} {ref} has no API key configured")
    if ov and ov.disabled:
        out.append("disabled by an admin")
    return out
