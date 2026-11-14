"""Drop cached overrides (touchpoints, agents) and keys in every process when an admin edits them."""


def _connect():
    from django.db.models.signals import post_delete, post_save

    from llm_kit.agents.overrides import invalidate as invalidate_agent_overrides
    from llm_kit.credentials.keys import invalidate as invalidate_keys
    from llm_kit.models import AgentOverride, ProviderKey, ScopedBudget, ScopedSetting, TouchpointOverride
    from llm_kit.scoped.budgets import invalidate as invalidate_budgets
    from llm_kit.scoped.service import invalidate as invalidate_scoped
    from llm_kit.touchpoints.env import invalidate_overrides

    def overrides_changed(**kwargs):
        invalidate_overrides()

    def keys_changed(instance=None, **kwargs):
        scope = getattr(instance, "scope", "")
        invalidate_keys(scope, getattr(instance, "scope_id", "")) if scope else invalidate_keys()

    for signal in (post_save, post_delete):
        signal.connect(overrides_changed, sender=TouchpointOverride, weak=False, dispatch_uid=f"llm_kit_tpo_{signal}")
        signal.connect(keys_changed, sender=ProviderKey, weak=False, dispatch_uid=f"llm_kit_pk_{signal}")
        signal.connect(
            lambda instance, **kw: invalidate_scoped(instance.scope, instance.scope_id),
            sender=ScopedSetting,
            weak=False,
            dispatch_uid=f"llm_kit_ss_{signal}",
        )
        signal.connect(
            lambda instance, **kw: invalidate_budgets(instance.scope, instance.scope_id),
            sender=ScopedBudget,
            weak=False,
            dispatch_uid=f"llm_kit_sb_{signal}",
        )
        signal.connect(
            lambda **kw: invalidate_agent_overrides(),
            sender=AgentOverride,
            weak=False,
            dispatch_uid=f"llm_kit_ao_{signal}",
        )


_connect()
