"""Who may see which thread.

* ``REQUIRE_AUTH=True`` (default): users see only threads they own.
* ``REQUIRE_AUTH=False``: anonymous users may also use ownerless threads by
  id (ids are random UUIDs), but listing never shows them.
"""

from __future__ import annotations

from llm_kit import conf
from llm_kit.models import Thread


class NotAllowed(PermissionError):
    pass


def can_use_api(user) -> bool:
    return bool(user and user.is_authenticated) or not conf.get("REQUIRE_AUTH")


def threads_for(user):
    if user and user.is_authenticated:
        return Thread.objects.filter(user=user)
    return Thread.objects.none()


def owner_for_new_thread(user):
    return user if user and user.is_authenticated else None


def check_thread(thread: Thread, user) -> None:
    if not can_use_api(user):
        raise NotAllowed("Authentication required.")
    if thread.user_id is None:
        if conf.get("REQUIRE_AUTH") and not (user and user.is_staff):
            raise NotAllowed("Not your thread.")
        return
    if not (user and user.is_authenticated and thread.user_id == user.pk):
        raise NotAllowed("Not your thread.")


def get_thread(thread_id, user) -> Thread:
    """Fetch a thread the user may use. Raises Thread.DoesNotExist or NotAllowed."""
    thread = Thread.objects.get(pk=thread_id)
    check_thread(thread, user)
    return thread
