"""Threads your code keeps by a key of its choosing (a WhatsApp number, a ticket, a hunt)::

    from llm_kit.runtime.threads import open_thread

    thread = open_thread("support.assistant", key=f"whatsapp:{jid}", tenant=business.pk)
    result = agents.run("support.assistant", text, thread=thread)

The key is unique per tenant. Clients can never choose or look up a key; only your code
can. A keyed thread with no user is usable only by your code (and staff, to inspect it).
"""

from __future__ import annotations

from typing import Any

from asgiref.sync import sync_to_async
from django.db import IntegrityError, transaction

from llm_kit.models import Thread

__all__ = ["aopen_thread", "open_thread"]


def _tenant_id(tenant: Any) -> str:
    if tenant is None:
        return ""
    return str(getattr(tenant, "pk", tenant))


def open_thread(graph: str, *, key: str, user=None, tenant: Any = None, title: str = "") -> Thread:
    """The thread for ``key`` (within ``tenant``), created on first use.

    An existing thread keeps its graph and owner; asking for a different graph raises
    ``ValueError`` rather than silently mixing two graphs' state in one thread.
    """
    if not key:
        raise ValueError("open_thread needs a non-empty key.")
    from llm_kit.runtime.graphs import get_graph

    get_graph(graph)  # raises UnknownGraph
    tenant_id = _tenant_id(tenant)
    lookup = {"external_key": key, "tenant": tenant_id}
    defaults = {"graph": graph, "user": user if getattr(user, "is_authenticated", False) else None, "title": title}
    try:
        with transaction.atomic():
            thread, _ = Thread.objects.get_or_create(**lookup, defaults=defaults)
    except IntegrityError:  # created by someone else in between
        thread = Thread.objects.get(**lookup)
    if thread.graph != graph:
        raise ValueError(f"Thread for key {key!r} runs graph {thread.graph!r}, not {graph!r}.")
    return thread


async def aopen_thread(graph: str, *, key: str, user=None, tenant: Any = None, title: str = "") -> Thread:
    return await sync_to_async(open_thread)(graph, key=key, user=user, tenant=tenant, title=title)
