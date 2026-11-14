"""Running an agent from your code (views, tasks, management commands)::

    result = agents.run("crm.researcher", "Which invoices are overdue?", user=request.user)
    result.text, result.output, result.steps, result.stopped

The run acts for ``user`` (tools are limited to theirs) and is recorded like any other run.

Called while another agent is running (from a tool, a code agent, or code they call), the
run joins that agent's request: same caller unless you pass ``user``/``tenant``, same
limits (depth, timeout), its own thread under the parent's, recorded as a subagent run in
the request's tree, and its messages forwarded to the request's stream. Pass
``join=False`` to start a separate request instead.

``retry`` sets how long this run's model calls wait out a short rate limit before
falling back (a ``RetryPolicy``, a dict, or ``"batch"``/``"interactive"``/``"never"``).
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Iterator
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


def running_parent() -> dict | None:
    """The config of the llm-kit run this code is running inside, if any."""
    from langchain_core.runnables.config import var_child_runnable_config

    from llm_kit.core.agents.limits import tree_of

    parent = var_child_runnable_config.get()
    return parent if tree_of(parent) is not None else None


@contextlib.contextmanager
def _detached(nested: bool) -> Iterator[None]:
    """A separate request started inside a running graph: hide the parent's config, which
    LangChain and LangGraph would otherwise merge in (making it a hidden part of the parent)."""
    if nested:
        yield
        return
    from langchain_core.runnables.config import var_child_runnable_config

    token = var_child_runnable_config.set(None)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _with_callbacks(config: dict, mine) -> None:
    """Add your callbacks next to the ones already in ``config`` (a list or a callback manager)."""
    if not mine:
        return
    mine = mine if isinstance(mine, list) else [mine]
    current = config.get("callbacks")
    if current is None or isinstance(current, list):
        config["callbacks"] = [*(current or []), *mine]
        return
    manager = current.copy()
    for handler in mine:
        manager.add_handler(handler, inherit=True)
    config["callbacks"] = manager


def _nested_config(parent: dict, name: str, *, user, tenant, thread_id, throwaway, extra) -> dict:
    from llm_kit.core.agents.delegate import check_depth, child_config
    from llm_kit.core.agents.limits import EPHEMERAL_KEY
    from llm_kit.tools.caller import _UNSET, CONFIG_KEY, caller_for

    spec = get_agent(name)
    check_depth(parent)
    parent_thread = (parent.get("configurable") or {}).get("thread_id") or "run"
    thread_id = thread_id or f"{parent_thread}/{spec.name}/{uuid.uuid4().hex}"
    config = child_config(parent, spec, thread_id)
    configurable, metadata = config["configurable"], config["metadata"]
    if user is not None or tenant is not None:
        caller = caller_for(user, _UNSET if tenant is None else tenant)
        configurable[CONFIG_KEY] = caller
        metadata.pop("user_id", None)
        metadata.pop("tenant", None)
        if caller.is_authenticated:
            metadata["user_id"] = str(caller.user_id)
        if caller.tenant_id is not None:
            metadata["tenant"] = str(caller.tenant_id)
    if throwaway:
        configurable[EPHEMERAL_KEY] = True
    else:
        configurable.pop(EPHEMERAL_KEY, None)
    configurable.update(extra.pop("configurable", None) or {})
    metadata.update(extra.pop("metadata", None) or {})
    _with_callbacks(config, extra.pop("callbacks", None))
    config.update(extra)
    return runner.root_config(spec, config, thread_id)


def _config(
    name: str, *, user=None, tenant: Any = None, thread_id=None, thread=None, join=True, retry=None, **extra
) -> tuple[dict, bool]:
    throwaway = thread is None and thread_id is None
    user, tenant, thread_id = _from_thread(name, thread, user, tenant, thread_id)
    from llm_kit import conf  # noqa: F401  (wires settings into the core config)
    from llm_kit.core.touchpoints.retry import RETRY_KEY, RetryPolicy

    if retry is not None:
        extra["configurable"] = {**(extra.get("configurable") or {}), RETRY_KEY: RetryPolicy.coerce(retry)}
    parent = running_parent() if join else None
    if parent is not None:
        return _nested_config(
            parent, name, user=user, tenant=tenant, thread_id=thread_id, throwaway=throwaway, extra=extra
        ), True

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
    return runner.root_config(spec, config, thread_id or f"agent-{uuid.uuid4()}"), False


def config_for(name: str, *, user=None, tenant: Any = None, thread_id: str | None = None, thread=None, **extra) -> dict:
    """Config for running agent ``name`` for ``user``: caller, recording, limits, thread.

    ``thread`` (a ``llm_kit.models.Thread``, e.g. from ``open_thread``) continues that thread.
    Inside a running agent, the config joins its request (see the module docstring).
    """
    return _config(name, user=user, tenant=tenant, thread_id=thread_id, thread=thread, **extra)[0]


def _forget(cfg: dict, nested: bool) -> None:
    """Free a throwaway run's in-memory state: its thread and every subagent thread it started.

    A nested run shares its parent's tree, so it frees only its own thread and those below it."""
    from llm_kit.core.agents.limits import TREE_KEY
    from llm_kit.runtime.graphs import memory_checkpointer

    saver = memory_checkpointer()
    own = cfg["configurable"]["thread_id"]
    tree = cfg["configurable"].get(TREE_KEY)
    below = [t for t in (tree.threads if tree else []) if not nested or t.startswith(f"{own}/")]
    for thread_id in [own, *below]:
        saver.delete_thread(thread_id)


def run(
    name: str,
    task: str,
    *,
    user=None,
    tenant: Any = None,
    thread_id: str | None = None,
    thread=None,
    join: bool = True,
    retry: Any = None,
    **config,
) -> AgentResult:
    """Run agent ``name`` on ``task`` to the end. ``thread`` or ``thread_id`` continues a conversation.

    Without either, the run is throwaway: its saved state (and its subagents') is deleted when it
    ends. The result, and the recorded calls and runs, are kept.
    """
    cfg, nested = _config(
        name, user=user, tenant=tenant, thread_id=thread_id, thread=thread, join=join, retry=retry, **config
    )
    try:
        with _detached(nested):
            result = runner.run(name, task, config=cfg, nested=nested)
    finally:
        if thread is None and thread_id is None:
            _forget(cfg, nested)
    if thread is not None:
        thread.save(update_fields=["updated_at"])
    return result


async def arun(
    name: str,
    task: str,
    *,
    user=None,
    tenant: Any = None,
    thread_id: str | None = None,
    thread=None,
    join: bool = True,
    retry: Any = None,
    **config,
) -> AgentResult:
    cfg, nested = await sync_to_async(_config, thread_sensitive=True)(
        name, user=user, tenant=tenant, thread_id=thread_id, thread=thread, join=join, retry=retry, **config
    )
    try:
        with _detached(nested):
            result = await runner.arun(name, task, config=cfg, nested=nested)
    finally:
        if thread is None and thread_id is None:
            await sync_to_async(_forget)(cfg, nested)
    if thread is not None:
        await thread.asave(update_fields=["updated_at"])
    return result


async def stream(
    name: str,
    task: str,
    *,
    user=None,
    tenant: Any = None,
    thread_id: str | None = None,
    thread=None,
    retry: Any = None,
    **config,
) -> AsyncIterator[dict]:
    """llm-kit events (``token``, ``message``, ``update``, ``end``, ``error``) as the agent works.
    Subagents' tokens arrive too, with ``agent`` and ``depth`` set. A stream is always its own
    request (it never joins a running agent)."""
    from langchain_core.messages import HumanMessage

    from llm_kit.core.runs import astream_run
    from llm_kit.runtime.graphs import get_graph

    cfg, nested = await sync_to_async(_config, thread_sensitive=True)(
        name, user=user, tenant=tenant, thread_id=thread_id, thread=thread, join=False, retry=retry, **config
    )
    try:
        graph = get_graph(name, ephemeral=thread is None and thread_id is None)
        # Only touch the context when there is a parent to hide (the generator may be resumed elsewhere).
        with _detached(running_parent() is None):
            async for event in astream_run(graph, {"messages": [HumanMessage(content=task)]}, cfg):
                yield event
    finally:
        if thread is None and thread_id is None:
            await sync_to_async(_forget)(cfg, nested)
