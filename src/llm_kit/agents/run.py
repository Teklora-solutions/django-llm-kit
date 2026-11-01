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


def config_for(name: str, *, user=None, tenant: Any = None, thread_id: str | None = None, **extra) -> dict:
    """Config for running agent ``name`` for ``user``: caller, recording, limits, thread."""
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
    config: dict[str, Any] = {"configurable": configurable, "metadata": metadata, **extra}
    if (handler := get_handler()) is not None:
        config["callbacks"] = [handler]
    return runner.root_config(spec, config, thread_id or f"agent-{uuid.uuid4()}")


def run(name: str, task: str, *, user=None, tenant: Any = None, thread_id: str | None = None, **config) -> AgentResult:
    """Run agent ``name`` on ``task`` to the end. ``thread_id`` continues a conversation."""
    return runner.run(name, task, config=config_for(name, user=user, tenant=tenant, thread_id=thread_id, **config))


async def arun(
    name: str, task: str, *, user=None, tenant: Any = None, thread_id: str | None = None, **config
) -> AgentResult:
    cfg = await sync_to_async(config_for, thread_sensitive=True)(
        name, user=user, tenant=tenant, thread_id=thread_id, **config
    )
    return await runner.arun(name, task, config=cfg)


async def stream(
    name: str, task: str, *, user=None, tenant: Any = None, thread_id: str | None = None, **config
) -> AsyncIterator[dict]:
    """llm-kit events (``token``, ``message``, ``update``, ``end``, ``error``) as the agent works.
    Subagents' tokens arrive too, with ``agent`` and ``depth`` set."""
    from langchain_core.messages import HumanMessage

    from llm_kit.core.runs import astream_run
    from llm_kit.runtime.graphs import get_graph

    cfg = await sync_to_async(config_for, thread_sensitive=True)(
        name, user=user, tenant=tenant, thread_id=thread_id, **config
    )
    async for event in astream_run(get_graph(name), {"messages": [HumanMessage(content=task)]}, cfg):
        yield event
