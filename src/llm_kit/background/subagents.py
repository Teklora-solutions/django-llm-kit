"""Subagents started in the background: ``delegate(agent, task, background=true)``.

The parent agent gets a task id at once and carries on. The subagent runs as a
``BackgroundRun`` on a thread of its own, for the same caller, recorded under the parent's run.
It inherits the request's limits: the agents above it (depth), the hand-offs left, and the
budget left. When it finishes (``run_finished``), its answer is queued on the parent's thread
as a message, so the parent agent picks it up in a turn of its own and can tell the user.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Any

from django.dispatch import receiver

from llm_kit.background.signals import run_finished
from llm_kit.models import BackgroundRun, Thread

logger = logging.getLogger("llm_kit.agents")

REPORT_PREFIX = "[Automatic message, not from the user]"


def _thread_of(config) -> Thread | None:
    thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
    try:
        return Thread.objects.get(pk=uuid.UUID(str(thread_id)))
    except (ValueError, TypeError, Thread.DoesNotExist):
        return None


def can_detach(config) -> bool:
    """Only a run on a saved thread (a ``Thread``) can be told its tasks' results later."""
    from llm_kit.core.agents.limits import is_ephemeral

    return not is_ephemeral(config) and _thread_of(config) is not None


def start(agent: str, task: str, config, limits: dict[str, Any]) -> dict[str, Any]:
    from llm_kit import background
    from llm_kit.core.agents.delegate import Refused
    from llm_kit.core.errors import BudgetExceeded
    from llm_kit.runtime import access
    from llm_kit.tools.caller import caller_from

    parent = _thread_of(config)
    if parent is None:
        raise Refused("Background work needs a saved conversation; hand the task over without background.")
    caller = caller_from(config)
    user = caller.user if caller is not None and caller.is_authenticated else None
    tenant = caller.tenant if caller is not None else None
    child = Thread.objects.create(
        graph=agent,
        user=parent.user,
        session_key=parent.session_key,
        tenant=parent.tenant,
        title=task[:255],
        metadata={"parent_thread": str(parent.pk)},
    )
    parent_run = limits.get("parent_run") or ""
    try:
        run = background.start(
            child,
            {"message": task},
            access.Owner(user=user, tenant=tenant),
            links={
                "parent_thread": parent,
                "parent_run_id": uuid.UUID(parent_run) if parent_run else None,
                "agents": list(limits.get("agents") or []),
                "limits": {k: limits.get(k) for k in ("budget", "delegations", "max_depth")},
            },
        )
    except BudgetExceeded as e:
        child.delete()
        raise Refused(str(e)) from e
    return {"id": run.run_id, "status": run.status}


def inherit_limits(run: BackgroundRun, config: dict) -> None:
    """A background subagent's run tree: the agents above it, and the limits left when it started."""
    from llm_kit.core.agents.limits import STACK_KEY, TREE_KEY

    configurable, meta = config["configurable"], config.setdefault("metadata", {})
    configurable[STACK_KEY] = tuple(run.agents)
    meta["llm_kit_depth"] = len(run.agents) - 1
    if run.parent_run_id:
        meta["llm_kit_detached_from"] = str(run.parent_run_id)
    tree, limits = configurable.get(TREE_KEY), run.limits or {}
    if tree is None:
        return
    if limits.get("budget") is not None:
        tree.budget = Decimal(str(limits["budget"]))
    if limits.get("delegations") is not None:
        tree.max_delegations = int(limits["delegations"])
    if limits.get("max_depth") is not None:
        tree.max_depth = int(limits["max_depth"])


def _task(task_id, config) -> BackgroundRun | None:
    parent = _thread_of(config)
    if parent is None:
        return None
    try:
        return BackgroundRun.objects.get(pk=uuid.UUID(str(task_id).strip()), parent_thread=parent)
    except (ValueError, TypeError, BackgroundRun.DoesNotExist):
        return None


def answer(run: BackgroundRun) -> str:
    """What a finished background subagent came up with, as the model sees a hand-off's answer."""
    if run.status not in ("done", "interrupted"):
        return ""
    try:
        from llm_kit.core.agents.results import from_state
        from llm_kit.core.agents.spec import get_agent
        from llm_kit.runtime.graphs import get_graph

        thread = Thread.objects.get(pk=run.thread_id)
        state = get_graph(run.graph).get_state(thread.config)
        return from_state(get_agent(run.graph), state.values, thread.thread_id).for_model()
    except Exception:
        logger.warning("Could not read the answer of background task %s", run.run_id, exc_info=True)
        return ""


def status(task_id, config) -> dict[str, Any] | None:
    from llm_kit import background

    run = _task(task_id, config)
    if run is None:
        return None
    run = background.refresh(run)
    return {"status": run.status, "agent": run.graph, "answer": answer(run) or run.error}


def cancel(task_id, config) -> bool | None:
    run = _task(task_id, config)
    return None if run is None else run.cancel()


_OUTCOME = {
    "done": "is done",
    "interrupted": "stopped to ask a question",
    "error": "failed",
    "cancelled": "was cancelled",
    "lost": "stopped unexpectedly (its worker went away)",
}


def report(run: BackgroundRun, status: str, error: str) -> str:
    text = f"{REPORT_PREFIX} Background task {run.run_id} ({run.graph}) {_OUTCOME.get(status, status)}."
    if found := answer(run):
        text += f"\n\n{found}"
    elif error:
        text += f"\n\n{error}"
    return text + "\n\nTell the user what came of it."


@receiver(run_finished, dispatch_uid="llm_kit_report_background_task")
def report_to_parent(sender, run: BackgroundRun, status: str, error: str = "", **kwargs) -> None:
    """Queue the task's outcome on the parent's thread, so the parent agent tells the user."""
    if run.parent_thread_id is None:
        return
    from llm_kit import background
    from llm_kit.runtime import access

    parent = Thread.objects.filter(pk=run.parent_thread_id).first()
    if parent is None:
        return
    message = {"role": "user", "content": report(run, status, error)}
    try:
        background.start(
            parent,
            {"input": {"messages": [message]}},
            access.Owner(user=run.user, tenant=run.tenant or None),
            queue=True,
            links={"reports_on": run},
        )
    except Exception:
        logger.exception("Could not report background task %s to thread %s", run.run_id, parent.pk)
