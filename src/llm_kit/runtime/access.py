"""Who owns a thread, and who may use it.

A thread belongs to one of:

* a **user**;
* an anonymous **visitor**: a random id kept in their session data. The id lives in the
  data, not the Django session key, because ``login()`` rotates the key but keeps the data;
* an **external key** your code chose, e.g. ``"whatsapp:254700000000"`` (see
  ``llm_kit.runtime.threads.open_thread``). Clients never pick these;
* **nobody**: created by code. Only staff and your own code can use it.

Rules for requests:

* ``REQUIRE_AUTH=True`` (default): only signed-in users, and only their own threads.
* ``REQUIRE_AUTH=False``: visitors may also create threads, and use and list only their own.
* Staff may open any thread that has no user (to inspect it); never another user's.
* With a tenant (``TENANT_RESOLVER``), a thread stamped with another tenant is refused.

Everything here is sync (sessions and ``TENANT_RESOLVER`` may query); call it through
``sync_to_async`` from async code.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from django.db.models import Q

from llm_kit import conf
from llm_kit.models import Thread

SESSION_KEY = "llm_kit_visitor"


class NotAllowed(PermissionError):
    pass


@dataclass
class Owner:
    """The party a request acts for."""

    user: Any = None
    session: Any = None  # a Django session (or any mapping), for anonymous visitors
    tenant: Any = None

    @property
    def is_authenticated(self) -> bool:
        return bool(self.user is not None and getattr(self.user, "is_authenticated", False))

    @property
    def is_staff(self) -> bool:
        return self.is_authenticated and bool(getattr(self.user, "is_staff", False))

    @property
    def tenant_id(self) -> str:
        if self.tenant is None:
            return ""
        return str(getattr(self.tenant, "pk", self.tenant))

    @property
    def visitor_id(self) -> str:
        """The visitor id in the session, or "" if there is none yet."""
        if self.session is None:
            return ""
        return self.session.get(SESSION_KEY) or ""

    def ensure_visitor_id(self) -> str:
        if self.session is None:
            raise NotAllowed("Anonymous use needs Django sessions (SessionMiddleware, or SessionMiddlewareStack).")
        if not self.visitor_id:
            self.session[SESSION_KEY] = secrets.token_hex(16)
        return self.session[SESSION_KEY]


def owner_for(user=None, session=None, tenant: Any = None) -> Owner:
    """An ``Owner``; the tenant comes from ``TENANT_RESOLVER`` unless given."""
    if isinstance(user, Owner):
        return user
    if tenant is None:
        from llm_kit.tools.caller import resolve_tenant

        tenant = resolve_tenant(user)
    return Owner(user=user, session=session, tenant=tenant)


def owner_from_request(request) -> Owner:
    return owner_for(getattr(request, "user", None), getattr(request, "session", None))


def owner_from_scope(scope) -> Owner:
    """For Channels: the user and session put in the scope by ``AuthMiddlewareStack``."""
    return owner_for(scope.get("user"), scope.get("session"))


# ------------------------------------------------------------------ checks


def can_use_api(who) -> bool:
    """``who``: an ``Owner`` or a user."""
    user = who.user if isinstance(who, Owner) else who
    return bool(user and getattr(user, "is_authenticated", False)) or not conf.get("REQUIRE_AUTH")


def threads_for(who):
    """Threads a requester may list: their own (a user's, or a visitor's)."""
    owner = who if isinstance(who, Owner) else owner_for(who)
    if owner.is_authenticated:
        qs = Thread.objects.filter(user=owner.user)
        if owner.tenant_id:
            qs = qs.filter(Q(tenant=owner.tenant_id) | Q(tenant=""))
        return qs
    if not conf.get("REQUIRE_AUTH") and owner.visitor_id:
        return Thread.objects.filter(user__isnull=True, session_key=owner.visitor_id)
    return Thread.objects.none()


def owner_fields(who) -> dict:
    """Fields for a thread the requester creates."""
    owner = who if isinstance(who, Owner) else owner_for(who)
    if owner.is_authenticated:
        return {"user": owner.user, "tenant": owner.tenant_id}
    if not conf.get("REQUIRE_AUTH"):
        return {"session_key": owner.ensure_visitor_id()}
    raise NotAllowed("Authentication required.")


def owner_for_new_thread(user):
    """Deprecated: use ``owner_fields``. The user to own a new thread, or None."""
    return user if user and user.is_authenticated else None


def check_thread(thread: Thread, who) -> None:
    owner = who if isinstance(who, Owner) else owner_for(who)
    if not can_use_api(owner):
        raise NotAllowed("Authentication required.")
    if thread.tenant and owner.tenant_id and thread.tenant != owner.tenant_id:
        raise NotAllowed("Not your thread.")
    if thread.user_id is not None:
        if owner.is_authenticated and thread.user_id == owner.user.pk:
            return
        raise NotAllowed("Not your thread.")
    if owner.is_staff:  # visitor, keyed and code threads: staff may inspect them
        return
    if thread.session_key and not conf.get("REQUIRE_AUTH") and not owner.is_authenticated:
        if owner.visitor_id and secrets.compare_digest(thread.session_key, owner.visitor_id):
            return
    raise NotAllowed("Not your thread.")


def get_thread(thread_id, who) -> Thread:
    """Fetch a thread the requester may use. Raises Thread.DoesNotExist or NotAllowed."""
    thread = Thread.objects.get(pk=thread_id)
    check_thread(thread, who)
    return thread


# ---------------------------------------------------------------- adoption


def adopt_session_threads(request, user=None) -> int:
    """Give a visitor's threads to ``user`` (default ``request.user``) after they sign in.

    Runs on ``user_logged_in`` unless ``LLM_KIT["ADOPT_SESSION_THREADS"]`` is False.
    Returns how many threads moved.
    """
    user = user or getattr(request, "user", None)
    session = getattr(request, "session", None)
    if session is None or not (user and getattr(user, "is_authenticated", False)):
        return 0
    visitor = session.get(SESSION_KEY)
    if not visitor:
        return 0
    from llm_kit.tools.caller import resolve_tenant

    tenant = resolve_tenant(user)
    moved = Thread.objects.filter(user__isnull=True, session_key=visitor).update(
        user=user, session_key="", tenant="" if tenant is None else str(getattr(tenant, "pk", tenant))
    )
    session.pop(SESSION_KEY, None)
    return moved
