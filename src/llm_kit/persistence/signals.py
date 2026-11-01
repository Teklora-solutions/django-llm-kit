from django.contrib.auth.signals import user_logged_in
from django.db.models.signals import post_delete
from django.dispatch import receiver

from llm_kit.models import Thread


@receiver(post_delete, sender=Thread, dispatch_uid="llm_kit_thread_cleanup")
def delete_thread_checkpoints(sender, instance, using, **kwargs):
    """Deleting a Thread (admin, API, user cascade) also deletes its LangGraph state."""
    from llm_kit.persistence.saver import DjangoSaver

    DjangoSaver(using=using).delete_thread(instance.thread_id)


@receiver(user_logged_in, dispatch_uid="llm_kit_adopt_session_threads")
def adopt_visitor_threads(sender, request, user, **kwargs):
    """A visitor who signs in keeps their conversations."""
    from llm_kit import conf
    from llm_kit.runtime.access import adopt_session_threads

    if request is not None and conf.get("ADOPT_SESSION_THREADS"):
        adopt_session_threads(request, user)
