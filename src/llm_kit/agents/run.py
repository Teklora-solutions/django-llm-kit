"""Running an agent from your code (views, tasks, management commands)::

    result = agents.run("crm.researcher", "Which invoices are overdue?", user=request.user)
    result.text, result.output, result.steps, result.stopped

The run acts for ``user`` (tools are limited to theirs) and is recorded like any other run.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from asgiref.sync import sync_to_async

from llm_kit.core.agents import runner
from llm_kit.core.agents.results import AgentResult
from llm_kit.core.agents.spec import get_agent


def _from_thread(name: str, thread, user, tenant, thread_id):
    """A ``Thread`` supplies the thread id, and the user and tenant when not given."""
    if thread is None:
        return user, tenant, thread_id
    if thread.graph != name:
        raise ValueError(f"Thread {thread.thread_id} runs {thread.graph!r}, not {name!r}.")
    if user is None and thread.user_id is not None:
        user = thread.user
    if tenant is None and thread.tenant:
        tenant = thread.tenant
    return user, tenant, thread.thread_id


def config_for(name: str, *, user=None, tenant: Any = None, thread_id: str | None = None, thread=None, **extra) -> dict:
    """Config for running agent ``name`` for ``user``: caller, recording, limits, thread.

    ``thread`` (a ``llm_kit.models.Thread``, e.g. from ``open_thread``) continues that thread.
    """
    throwaway = thread is None and thread_id is None
    user, tenant, thread_id = _from_thread(name, thread, user, tenant, thread_id)
    from llm_kit import conf  # noqa: F401  (wires settings into the core config)
    from llm_kit.core.observability.handler import get_handler
    from llm_kit.tools.caller import _UNSET, CONFIG_KEY, caller_for

    spec = get_agent(name)
    caller = caller_for(user, _UNSET if tenant is None else tenant)
    metadata = {"llm_kit_graph": spec.name}
    if caller.is_authenticated:
        metadata["user_id"] = str(caller.user_id)
    if caller.tenant_id is not None:
        metadata["tenant"] = str(caller.tenant_id)
    configurable = {CONFIG_KEY: caller, **(extra.pop("configurable", None) or {})}
    if throwaway:  # nothing will continue it: keep its state (and its subagents') in memory
        from llm_kit.core.agents.limits import EPHEMERAL_KEY

        configurable[EPHEMERAL_KEY] = True
    config: dict[str, Any] = {"configurable": configurable, "metadata": metadata, **extra}
    if (handler := get_handler()) is not None:  # recorded, alongside any callbacks of your own
        mine = config.get("callbacks") or []
        config["callbacks"] = [handler, *(mine if isinstance(mine, list) else [mine])]
    return runner.root_config(spec, config, thread_id or f"agent-{uuid.uuid4()}")


def _forget(cfg: dict) -> None:
    """Free a throwaway run's in-memory state: its thread and every subagent thread it started."""
    from llm_kit.core.agents.limits import TREE_KEY
    from llm_kit.runtime.graphs import memory_checkpointer

    saver = memory_checkpointer()
    tree = cfg["configurable"].get(TREE_KEY)
    for thread_id in [cfg["configurable"]["thread_id"], *(tree.threads if tree else [])]:
        saver.delete_thread(thread_id)


def run(
    name: str, task: str, *, user=None, tenant: Any = None, thread_id: str | None = None, thread=None, **config
) -> AgentResult:
    """Run agent ``name`` on ``task`` to the end. ``thread`` or ``thread_id`` continues a conversation.

    Without either, the run is throwaway: its saved state (and its subagents') is deleted when it
    ends. The result, and the recorded calls and runs, are kept.
    """
    cfg = config_for(name, user=user, tenant=tenant, thread_id=thread_id, thread=thread, **config)
    try:
        result = runner.run(name, task, config=cfg)
    finally:
        if thread is None and thread_id is None:
            _forget(cfg)
    if thread is not None:
        thread.save(update_fields=["updated_at"])
    return result


async def arun(
    name: str, task: str, *, user=None, tenant: Any = None, thread_id: str | None = None, thread=None, **config
) -> AgentResult:
    cfg = await sync_to_async(config_for, thread_sensitive=True)(
        name, user=user, tenant=tenant, thread_id=thread_id, thread=thread, **config
    )
    try:
        result = await runner.arun(name, task, config=cfg)
    finally:
        if thread is None and thread_id is None:
            await sync_to_async(_forget)(cfg)
    if thread is not None:
        await thread.asave(update_fields=["updated_at"])
    return result


async def stream(
    name: str, task: str, *, user=None, tenant: Any = None, thread_id: str | None = None, thread=None, **config
) -> AsyncIterator[dict]:
    """llm-kit events (``token``, ``message``, ``update``, ``end``, ``error``) as the agent works.
    Subagents' tokens arrive too, with ``agent`` and ``depth`` set."""
    from langchain_core.messages import HumanMessage

    from llm_kit.core.runs import astream_run
    from llm_kit.runtime.graphs import get_graph

    cfg = await sync_to_async(config_for, thread_sensitive=True)(
        name, user=user, tenant=tenant, thread_id=thread_id, thread=thread, **config
    )
    try:
        graph = get_graph(name, ephemeral=thread is None and thread_id is None)
        async for event in astream_run(graph, {"messages": [HumanMessage(content=task)]}, cfg):
            yield event
    finally:
        if thread is None and thread_id is None:
            await sync_to_async(_forget)(cfg)
