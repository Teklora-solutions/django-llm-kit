"""Models a client may pick for a touchpoint, for UI pickers and the API."""

from __future__ import annotations

from llm_kit.core.touchpoints.env import has_credentials
from llm_kit.core.touchpoints.policy import ClientChoice, matches_any
from llm_kit.core.touchpoints.touchpoint import Touchpoint
from llm_kit.touchpoints.env import DjangoEnv


def allowed_models(tp: Touchpoint, *, limit: int = 200, choice: ClientChoice | None = None) -> list:
    """Catalogue rows a client may choose for ``tp`` (empty for ``ServerChoice``).

    Same checks as resolution: allow patterns (code and admin), capabilities,
    price caps, not hidden, and an API key configured. ``choice`` replaces the
    touchpoint's own (the bounds of a saved choice, ``llm_kit.scoped``).
    """
    from llm_kit import catalogue

    choice = choice or tp.choice
    if not isinstance(choice, ClientChoice):
        return []
    env = DjangoEnv()
    ov = env.override(tp.name)
    qs = catalogue.search(
        kind=tp.requires.get("kind", "chat"),
        requires={**tp.requires, **choice.requires},
        max_price_input=choice.max_price_input,
        max_price_output=choice.max_price_output,
    ).order_by("provider", "model_id")
    out = []
    for row in qs.iterator():
        if not matches_any(row.ref, choice.allow):
            continue
        if ov and ov.allow is not None and not matches_any(row.ref, ov.allow):
            continue
        if not has_credentials(env, row.ref):
            continue
        out.append(row)
        if len(out) >= limit:
            break
    return out
