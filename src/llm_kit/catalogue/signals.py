"""Re-apply catalogue overrides as soon as an admin edits one."""


def _override_changed(sender, instance, **kwargs):
    from llm_kit.catalogue import apply_override

    apply_override(instance.provider, instance.model_id)


def _connect_override_signals():
    from django.db.models.signals import post_delete, post_save

    from llm_kit.models import CatalogueOverride

    post_save.connect(_override_changed, sender=CatalogueOverride, dispatch_uid="llm_kit_override_saved")
    post_delete.connect(_override_changed, sender=CatalogueOverride, dispatch_uid="llm_kit_override_deleted")


_connect_override_signals()
