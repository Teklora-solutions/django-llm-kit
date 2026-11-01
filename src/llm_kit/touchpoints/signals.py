"""Drop cached overrides and keys in every process when an admin edits them."""


def _connect():
    from django.db.models.signals import post_delete, post_save

    from llm_kit.credentials.keys import invalidate as invalidate_keys
    from llm_kit.models import ProviderKey, TouchpointOverride
    from llm_kit.touchpoints.env import invalidate_overrides

    def overrides_changed(**kwargs):
        invalidate_overrides()

    def keys_changed(**kwargs):
        invalidate_keys()

    for signal in (post_save, post_delete):
        signal.connect(overrides_changed, sender=TouchpointOverride, weak=False, dispatch_uid=f"llm_kit_tpo_{signal}")
        signal.connect(keys_changed, sender=ProviderKey, weak=False, dispatch_uid=f"llm_kit_pk_{signal}")


_connect()
