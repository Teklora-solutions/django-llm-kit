from django.db.models.signals import post_delete
from django.dispatch import receiver

from llm_kit.models import Thread


@receiver(post_delete, sender=Thread, dispatch_uid="llm_kit_thread_cleanup")
def delete_thread_checkpoints(sender, instance, using, **kwargs):
    """Deleting a Thread (admin, API, user cascade) also deletes its LangGraph state."""
    from llm_kit.checkpoint import DjangoSaver

    DjangoSaver(using=using).delete_thread(instance.thread_id)
