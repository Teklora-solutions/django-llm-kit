"""Options a UI shows for a setting, e.g. the models a saved choice may name."""

from __future__ import annotations


def choice_for(tp):
    """The ``ClientChoice`` that bounds a saved model choice for ``tp``."""
    from llm_kit.core.touchpoints.policy import ClientChoice

    if isinstance(tp.choice, ClientChoice):
        return tp.choice
    return ClientChoice(allow=tp.settable.model_allow if tp.settable is not None else ("*",))


def model_options(tp, limit: int = 200) -> list[dict]:
    from llm_kit.touchpoints.picker import allowed_models

    return [
        {
            "value": row.ref,
            "label": row.name or row.ref,
            "price_input": None if row.price_input is None else str(row.price_input),
            "price_output": None if row.price_output is None else str(row.price_output),
            "context_window": row.context_window,
        }
        for row in allowed_models(tp, choice=choice_for(tp), limit=limit)
    ]
