"""Handing work to a subagent: the ``delegate`` tool, and ``ctx.run`` in code agents.

A subagent is the same harness with another declaration. It runs inside the tool call,
for the same caller, on its own checkpointed thread ``<parent thread>/<agent>/<call id>``.
It starts with only the task (or a copy of the parent's conversation, with ``fork=True``),
and its answer comes back as the tool result. Its model calls go through the parent's
callbacks, so they count towards the parent's run, and its messages are forwarded to the
parent's stream.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, StructuredTool
from pydantic import Field, create_model

from llm_kit.core import config as settings
from llm_kit.core.agents.limits import STACK_KEY, TREE_KEY, Tree, is_ephemeral, stack_of, tree_of
from llm_kit.core.agents.results import AgentResult, from_state
from llm_kit.core.agents.spec import AgentSpec, UnknownAgent, get_agent
from llm_kit.core.errors import LLMKitError
from llm_kit.core.runs import SUBAGENT_EVENT

logger = logging.getLogger("llm_kit.agents")

DELEGATE = "delegate"


class Refused(LLMKitError):
    """A delegation that limits or declarations don't allow. The message is for the model."""


def effective(spec: AgentSpec, env) -> AgentSpec:
    """The registered declaration (module reloads replace it), as an admin narrowed it."""
    try:
        current = get_agent(spec.name)
    except UnknownAgent:
        current = spec
    return current.with_override(env.override(current.name))


def _max_depth(config) -> int:
    tree = tree_of(config)
    return tree.max_depth if tree is not None else int(settings.get("AGENT_MAX_DEPTH"))


def available(spec: AgentSpec, config, env) -> list[AgentSpec]:
    """The subagents ``spec`` may hand work to right now: not an agent above it, not too deep, not paused."""
    stack = stack_of(config) or (spec.name,)
    if len(stack) > _max_depth(config):
        return []
    out = []
    for name in spec.subagents:
        if name in stack:
            continue
        try:
            sub = effective(get_agent(name), env)
        except UnknownAgent:
            continue
        if not sub.paused:
            out.append(sub)
    return out


def child_config(config, spec: AgentSpec, thread_id: str) -> dict[str, Any]:
    """Config for a subagent's run: same caller and limits, its own thread, no parent graph internals."""
    parent = (config or {}).get("configurable") or {}
    stack = stack_of(config)
    configurable = {k: v for k, v in parent.items() if str(k).startswith("__llm_kit_")}
    if configurable.get(TREE_KEY) is None:
        configurable[TREE_KEY] = Tree.new()
    configurable[STACK_KEY] = (*stack, spec.name)
    configurable["thread_id"] = thread_id
    configurable[TREE_KEY].add_thread(thread_id)  # so a throwaway run can delete it afterwards
    if "prompt_vars" in parent:
        configurable["prompt_vars"] = parent["prompt_vars"]
    run_id = uuid.uuid4()
    meta = (config or {}).get("metadata") or {}
    metadata = {k: v for k, v in meta.items() if str(k).startswith("llm_kit_") or k in ("user_id", "tenant")}
    metadata.update(
        llm_kit_parent_agent_run=str(meta.get("llm_kit_agent_run") or ""),
        llm_kit_thread=thread_id,
        llm_kit_agent=spec.name,
        llm_kit_agent_run=str(run_id),
        llm_kit_parent_agent=stack[-1] if stack else "",
        llm_kit_depth=len(stack),
    )
    child: dict[str, Any] = {
        "configurable": configurable,
        "metadata": metadata,
        "run_id": run_id,
        "run_name": spec.name,
        "recursion_limit": 2 * spec.max_steps + 8,
    }
    if (callbacks := (config or {}).get("callbacks")) is not None:
        child["callbacks"] = callbacks
    return child


def _first_messages(spec: AgentSpec, task: str, parent_messages) -> list:
    history = []
    if spec.fork and parent_messages:
        history = list(parent_messages)
        if history and isinstance(history[-1], AIMessage) and history[-1].tool_calls:
            history = history[:-1]  # the call that is starting this subagent has no reply yet
    return [*history, HumanMessage(content=task)]


def _check(parent: AgentSpec, name: str, config, env) -> AgentSpec:
    if name not in parent.subagents:
        raise Refused(f"{name!r} is not one of the agents you can hand work to.")
    stack = stack_of(config) or (parent.name,)
    if name in stack:
        raise Refused(f"{name} is already working on this request above you; do the task yourself.")
    if len(stack) > _max_depth(config):
        raise Refused("Subagents can't be nested any deeper; do the task yourself.")
    try:
        sub = effective(get_agent(name), env)
    except UnknownAgent:
        raise Refused(f"There is no agent called {name!r}.") from None
    if sub.paused:
        raise Refused(f"{name} is paused: {sub.paused}")
    tree = tree_of(config)
    if tree is not None:
        if tree.timed_out:
            raise Refused("Out of time for this request; answer with what you have.")
        if not tree.take_delegation():
            raise Refused(f"This request already handed off {tree.max_delegations} tasks; finish it yourself.")
    return sub


def _setup(parent, name, task, config, env, call_id, parent_messages):
    if not (task or "").strip():
        raise Refused("Give the task to hand over.")
    sub = _check(parent, name, config, env)
    parent_thread = ((config or {}).get("configurable") or {}).get("thread_id") or "run"
    thread_id = f"{parent_thread}/{name}/{call_id or uuid.uuid4().hex}"
    if tree_of(config) is None:  # a run started without llm-kit's config: limits start here
        config = {
            **(config or {}),
            "configurable": {**((config or {}).get("configurable") or {}), TREE_KEY: Tree.new()},
        }
    child = child_config(config, sub, thread_id)
    graph = env.graph(name, ephemeral=is_ephemeral(child))
    return sub, graph, {"messages": _first_messages(sub, task, parent_messages)}, child, thread_id


_MODES = ["values", "messages", "custom"]


def _writer():
    """The parent run's custom stream (a no-op when nobody is streaming it)."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # not inside a graph run
        return lambda chunk: None


def _forward(write, mode: str, chunk) -> None:
    """Pass a subagent's messages up to the parent's stream, labelled with the agent and depth.

    LangGraph keeps a caller's stream handlers out of a graph running on its own thread, so
    the subagent is streamed here and its messages re-sent as custom events, which
    ``astream_run`` turns back into ``token``/``message`` events. Deeper subagents' events
    are already labelled and pass through as they are.
    """
    if mode == "messages":
        message, meta = chunk
        write(
            {
                SUBAGENT_EVENT: True,
                "agent": meta.get("llm_kit_agent"),
                "depth": int(meta.get("llm_kit_depth") or 0),
                "node": meta.get("langgraph_node"),
                "message": message,
            }
        )
    elif mode == "custom":
        write(chunk)


def run_subagent(parent, name, task, config, env, *, call_id=None, parent_messages=None) -> AgentResult:
    sub, graph, start, child, thread_id = _setup(parent, name, task, config, env, call_id, parent_messages)
    write, state = _writer(), {}
    for mode, chunk in graph.stream(start, child, stream_mode=_MODES):
        if mode == "values":
            state = chunk
        else:
            _forward(write, mode, chunk)
    return from_state(sub, state, thread_id)


async def arun_subagent(parent, name, task, config, env, *, call_id=None, parent_messages=None) -> AgentResult:
    from asgiref.sync import sync_to_async

    sub, graph, start, child, thread_id = await sync_to_async(_setup, thread_sensitive=True)(
        parent, name, task, config, env, call_id, parent_messages
    )
    write, state = _writer(), {}
    async for mode, chunk in graph.astream(start, child, stream_mode=_MODES):
        if mode == "values":
            state = chunk
        else:
            _forward(write, mode, chunk)
    return from_state(sub, state, thread_id)


def _reply(name: str, result: AgentResult) -> str:
    if result.stopped == "done":
        return result.for_model()
    return f"{name} stopped early ({result.stopped}). What it had:\n{result.for_model()}"


def delegate_tool(parent: AgentSpec, subagents: list[AgentSpec], env, parent_messages=None) -> StructuredTool:
    """One tool for every subagent: ``agent`` is an enum of their names, so the schema stays small."""
    names = tuple(s.name for s in subagents)
    args = create_model(
        "Delegate",
        agent=(Literal[names], Field(description="Which agent does the task.")),  # type: ignore[valid-type]
        task=(str, Field(description="The task, with everything the agent needs to do it.")),
        tool_call_id=(Annotated[str, InjectedToolCallId], ...),
    )
    lines = "\n".join(
        f"- {s.name}: {s.description}" + (" (sees this conversation)" if s.fork else "") for s in subagents
    )
    description = (
        "Hand a self-contained task to another agent and get its answer back. It doesn't see this "
        f"conversation unless noted, so include everything it needs.\n{lines}"
    )

    def run(agent: str, task: str, tool_call_id: str, config: RunnableConfig) -> str:
        try:
            result = run_subagent(
                parent, agent, task, config, env, call_id=tool_call_id, parent_messages=parent_messages
            )
        except Refused as e:
            return f"Refused: {e}"
        except Exception:
            logger.exception("Subagent %s failed", agent)
            return f"{agent} failed before finishing; do the task another way or tell the user."
        return _reply(agent, result)

    async def arun(agent: str, task: str, tool_call_id: str, config: RunnableConfig) -> str:
        try:
            result = await arun_subagent(
                parent, agent, task, config, env, call_id=tool_call_id, parent_messages=parent_messages
            )
        except Refused as e:
            return f"Refused: {e}"
        except Exception:
            logger.exception("Subagent %s failed", agent)
            return f"{agent} failed before finishing; do the task another way or tell the user."
        return _reply(agent, result)

    return StructuredTool(
        name=DELEGATE,
        description=description,
        args_schema=args,
        func=run,
        coroutine=arun,
        metadata={"llm_kit_kind": "delegate"},
    )
