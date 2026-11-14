"""Options a UI shows for a setting, e.g. the models a saved choice may name."""

from __future__ import annotations


def choice_for(tp):
    """The ``ClientChoice`` that bounds a saved model choice for ``tp``."""
    from llm_kit.core.touchpoints.policy import ClientChoice

    if isinstance(tp.choice, ClientChoice):
        return tp.choice
    return ClientChoice(allow=tp.settable.model_allow if tp.settable is not None else ("*",))


def model_options(tp, owner=None, limit: int = 200) -> list[dict]:
    from llm_kit.touchpoints.env import DjangoEnv
    from llm_kit.touchpoints.picker import allowed_models

    env = DjangoEnv()
    if owner is not None:  # keys the asker's tenant or user brought count
        env = env.for_caller(owner).for_touchpoint(tp)

    return [
        {
            "value": row.ref,
            "label": row.name or row.ref,
            "price_input": _usd(row.price_input),
            "price_output": _usd(row.price_output),
            "context_window": row.context_window,
        }
        for row in allowed_models(tp, choice=choice_for(tp), limit=limit, env=env)
    ]


def _usd(value) -> str | None:
    """``2.00000000`` -> ``"2"``, ``0.10000000`` -> ``"0.1"`` (USD per 1M tokens)."""
    from llm_kit.core.errors import _usd as fmt

    return None if value is None else fmt(value)
