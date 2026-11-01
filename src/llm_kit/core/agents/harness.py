"""The harness: one LangGraph loop that runs every agent.

model -> tools -> model ... until the model answers without calling a tool, or reaches
``max_steps``. Each step reads the declaration again (with admin overrides), so a change
in the admin applies on the next step:

1. resolve the model through the agent's touchpoint (client choice, overrides, fallbacks)
2. build the system prompt: the agent's prompt, tool guidance, the agents it can hand work to
3. bind its tools, limited to what the caller may use, plus ``delegate`` when it has subagents

The tool node runs only tools from that same set; a tool call outside it is refused,
so a parent's tools never leak into a subagent. Code agents get a one-node graph that
runs their function.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from asgiref.sync import sync_to_async
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from llm_kit.core.agents.deferral import FIND_TOOLS, find_tools_tool, found_reply, match, union
from llm_kit.core.agents.delegate import (
    DELEGATE,
    Refused,
    arun_subagent,
    available,
    delegate_tool,
    effective,
    run_subagent,
)
from llm_kit.core.agents.env import AgentEnv, default_env
from llm_kit.core.agents.limits import depth_of, tree_of
from llm_kit.core.agents.prompt import build_prompt, context_for
from llm_kit.core.agents.results import AgentResult, this_turn
from llm_kit.core.agents.spec import AgentSpec

logger = logging.getLogger("llm_kit.agents")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    output: Any  # the final answer in the agent's ``output`` shape (JSON), if it has one
    stopped: str  # "" while working or done; max_steps | timeout | paused
    loaded_tools: Annotated[list[str], union]  # with defer_tools: the tools found so far


def _synthetic(text: str) -> AIMessage:
    """A reply the harness wrote, not the model (not counted as a step)."""
    return AIMessage(content=text, response_metadata={"llm_kit_synthetic": True})


def _not_run(message: AIMessage, why: str) -> list[ToolMessage]:
    return [
        ToolMessage(content=f"Not run: {why}", tool_call_id=c["id"], name=c["name"], status="error")
        for c in message.tool_calls
    ]


def model_steps(messages: list) -> int:
    return sum(
        1
        for m in this_turn(messages)
        if isinstance(m, AIMessage) and not (m.response_metadata or {}).get("llm_kit_synthetic")
    )


def resolve_tools(spec: AgentSpec, config, env: AgentEnv) -> list[BaseTool]:
    """The agent's tools for this run's caller, after any admin narrowing."""
    raw = spec.tools
    if raw is None:
        tools: list[BaseTool] = []
    elif callable(raw):
        tools = list(raw(config) or [])
    else:
        names = [t for t in raw if isinstance(t, str)]
        tools = []
        for item in raw:
            if isinstance(item, BaseTool):
                tools.append(item)
            elif callable(item):  # tools.select(...) or fn(config)
                tools += list(item(config) or [])
        if names:
            tools += env.named_tools(names, config)
    tools = env.permitted(tools, config)
    if spec.allowed_tools is not None:
        tools = [t for t in tools if t.name in spec.allowed_tools]
    seen, out = set(), []
    for t in tools:
        if t.name not in seen and t.name not in (DELEGATE, FIND_TOOLS):
            seen.add(t.name)
            out.append(t)
    return out


@dataclass
class Step:
    """Everything one model step needs, read fresh from the declaration and overrides."""

    spec: AgentSpec
    tools: list[BaseTool]  # every tool it may run this step
    subagents: list[AgentSpec]
    prompt: str
    last: bool
    llm: Any = None
    deferred: list[BaseTool] = field(default_factory=list)  # with defer_tools: not loaded yet

    @property
    def bound(self) -> list[BaseTool]:
        """The tools the model is shown."""
        names = {t.name for t in self.deferred}
        return [t for t in self.tools if t.name not in names]

    def run_config(self, config) -> dict:
        metadata = {
            **(config.get("metadata") or {}),
            "llm_kit_agent": self.spec.name,
            "llm_kit_depth": depth_of(config),
        }
        return {**config, "metadata": metadata}


def _stop_for(spec: AgentSpec, config) -> dict | None:
    if spec.paused:
        return {"messages": [_synthetic(f"{spec.name} is paused: {spec.paused}")], "stopped": "paused"}
    tree = tree_of(config)
    if tree is not None and tree.timed_out:
        return {"messages": [_synthetic("Stopped: this request ran out of time.")], "stopped": "timeout"}
    return None


def prepare(declared: AgentSpec, state, config, env: AgentEnv, *, with_model: bool = True) -> Step | dict:
    spec = effective(declared, env)
    if (stop := _stop_for(spec, config)) is not None:
        return stop
    tools = resolve_tools(spec, config, env)
    deferred = []
    if spec.defer_tools:
        loaded = {*spec.essential, *(state.get("loaded_tools") or [])}
        deferred = [t for t in tools if t.name not in loaded]
    subagents = available(spec, config, env)
    last = model_steps(state.get("messages") or []) + 1 >= spec.max_steps
    step = Step(spec=spec, tools=tools, subagents=subagents, prompt="", last=last, deferred=deferred)
    guidance = env.guidance(step.bound)
    if deferred:
        guidance = (guidance + "\n" if guidance else "") + (
            f"- {len(deferred)} more tools are available: call `{FIND_TOOLS}` with what you need to load them."
        )
    step.prompt = build_prompt(
        spec,
        context_for(spec, config, env),
        guidance=guidance,
        subagents=subagents,
        last_step=last and bool(tools or subagents),
    )
    if with_model:
        llm = spec.get_touchpoint().model(config=config)
        offered = step.bound + ([find_tools_tool()] if deferred else [])
        offered += [delegate_tool(spec, subagents, env)] if subagents else []
        step.llm = llm.bind_tools(offered) if offered else llm
    return step


def _after_reply(step: Step, reply: BaseMessage) -> dict:
    update: dict[str, Any] = {"messages": [reply], "stopped": ""}
    if isinstance(reply, AIMessage) and reply.tool_calls and step.last:
        from llm_kit.core.serialize import message_text

        said = message_text(reply).strip()
        note = f"(Stopped: reached the limit of {step.spec.max_steps} steps before finishing.)"
        update["messages"] += [
            *_not_run(reply, "the step limit was reached."),
            _synthetic(f"{said}\n\n{note}" if said else note),
        ]
        update["stopped"] = "max_steps"
    return update


# ------------------------------------------------------------------ tools


def _tool_set(step: Step, state, env) -> dict[str, BaseTool]:
    tools = {t.name: t for t in step.tools}
    if step.subagents:
        tools[DELEGATE] = delegate_tool(step.spec, step.subagents, env, parent_messages=state.get("messages"))
    return tools


def _refusal(call, step: Step) -> ToolMessage:
    return ToolMessage(
        content=f"Refused: {call['name']} is not one of {step.spec.name}'s tools.",
        tool_call_id=call["id"],
        name=call["name"],
        status="error",
    )


def _as_tool_message(result: Any, call) -> ToolMessage:
    if isinstance(result, ToolMessage):
        return result
    return ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"])


def _failed(call, exc: Exception) -> ToolMessage:
    logger.warning("Tool %s failed: %s", call["name"], exc, exc_info=True)
    return ToolMessage(
        content=f"Error: {call['name']} failed ({type(exc).__name__}: {exc}).",
        tool_call_id=call["id"],
        name=call["name"],
        status="error",
    )


def _find(call, step: Step) -> tuple[ToolMessage, list[str]]:
    found = match(str((call.get("args") or {}).get("query") or ""), step.deferred)
    reply = found_reply(found, len(step.deferred))
    return ToolMessage(content=reply, tool_call_id=call["id"], name=FIND_TOOLS), [t.name for t in found]


def _calls(state) -> tuple[AIMessage, list]:
    last = state["messages"][-1]
    return last, list(getattr(last, "tool_calls", None) or [])


# ------------------------------------------------------------------ building


def build(spec: AgentSpec, checkpointer: Any = None, env: AgentEnv | None = None):
    """The compiled graph for ``spec``. ``env`` defaults to the host's (read on every step)."""
    get_env = (lambda: env) if env is not None else default_env
    return (_build_code(spec, get_env) if spec.is_code else _build_model(spec, get_env)).compile(
        checkpointer=checkpointer, name=spec.name
    )


def window(spec: AgentSpec, messages: list) -> list:
    """The part of the conversation a model call sees: all of it, or the newest ``spec.history``
    tokens (approximate), starting at a user message so tool calls keep their replies."""
    if not spec.history or not messages:
        return list(messages)
    from langchain_core.messages import trim_messages
    from langchain_core.messages.utils import count_tokens_approximately

    kept = trim_messages(
        messages,
        max_tokens=spec.history,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    if not kept:  # the latest turn alone is over the limit: send it anyway, from its user message
        last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=0)
        kept = list(messages[last_human:])
    return kept


def _build_model(spec: AgentSpec, get_env) -> StateGraph:
    def model(state, config):
        step = prepare(spec, state, config, get_env())
        if isinstance(step, dict):
            return step
        reply = step.llm.invoke(
            [SystemMessage(step.prompt), *window(step.spec, state["messages"])], step.run_config(config)
        )
        return _after_reply(step, reply)

    async def amodel(state, config):
        step = await sync_to_async(prepare, thread_sensitive=True)(spec, state, config, get_env())
        if isinstance(step, dict):
            return step
        reply = await step.llm.ainvoke(
            [SystemMessage(step.prompt), *window(step.spec, state["messages"])], step.run_config(config)
        )
        return _after_reply(step, reply)

    def tools(state, config):
        last, calls = _calls(state)
        step = prepare(spec, state, config, get_env(), with_model=False)
        if isinstance(step, dict):
            return {**step, "messages": [*_not_run(last, "the agent stopped."), *step["messages"]]}
        available_tools = _tool_set(step, state, get_env())
        out, loaded = [], []
        for call in calls:
            if call["name"] == FIND_TOOLS and step.deferred:
                message, names = _find(call, step)
                out.append(message)
                loaded += names
                continue
            tool = available_tools.get(call["name"])
            if tool is None:
                out.append(_refusal(call, step))
                continue
            try:
                out.append(_as_tool_message(tool.invoke({**call, "type": "tool_call"}, config), call))
            except GraphBubbleUp:
                raise
            except Exception as e:  # the model reads the error and can recover
                out.append(_failed(call, e))
        return {"messages": out, "loaded_tools": loaded}

    async def atools(state, config):
        last, calls = _calls(state)
        env = get_env()
        step = await sync_to_async(prepare, thread_sensitive=True)(spec, state, config, env, with_model=False)
        if isinstance(step, dict):
            return {**step, "messages": [*_not_run(last, "the agent stopped."), *step["messages"]]}
        available_tools = _tool_set(step, state, env)
        loaded = []

        async def one(call):
            if call["name"] == FIND_TOOLS and step.deferred:
                message, names = _find(call, step)
                loaded.extend(names)
                return message
            tool = available_tools.get(call["name"])
            if tool is None:
                return _refusal(call, step)
            try:
                return _as_tool_message(await tool.ainvoke({**call, "type": "tool_call"}, config), call)
            except GraphBubbleUp:
                raise
            except Exception as e:
                return _failed(call, e)

        messages = list(await asyncio.gather(*(one(c) for c in calls)))
        return {"messages": messages, "loaded_tools": loaded}

    def respond(state, config):
        step = prepare(spec, state, config, get_env(), with_model=False)
        if isinstance(step, dict):
            return {"stopped": step["stopped"]}
        llm = step.spec.get_touchpoint().model(config=config)
        return {"output": _structured_answer(llm, step, state, config)}

    async def arespond(state, config):
        step = await sync_to_async(prepare, thread_sensitive=True)(spec, state, config, get_env(), with_model=False)
        if isinstance(step, dict):
            return {"stopped": step["stopped"]}
        llm = await sync_to_async(step.spec.get_touchpoint().model, thread_sensitive=True)(config=config)
        return {"output": await _astructured_answer(llm, step, state, config)}

    def route(state):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        if spec.output is not None and not state.get("stopped"):
            return "respond"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("model", RunnableLambda(model, afunc=amodel, name="model"))
    graph.add_node("tools", RunnableLambda(tools, afunc=atools, name="tools"))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, ["tools", "respond", END] if spec.output else ["tools", END])
    graph.add_edge("tools", "model")
    if spec.output is not None:
        graph.add_node("respond", RunnableLambda(respond, afunc=arespond, name="respond"))
        graph.add_edge("respond", END)
    return graph


# ------------------------------------------------------------------ structured answers

_SHAPE = "Put your final answer above into the required structure. Use only what you found."


def _structured_messages(step: Step, state) -> list:
    return [SystemMessage(step.prompt), *window(step.spec, state["messages"]), HumanMessage(_SHAPE)]


def _structured_answer(llm, step: Step, state, config) -> Any:
    from llm_kit.core.touchpoints.structured import parse_as
    from llm_kit.core.touchpoints.touchpoint import _parsed, _structured, _text

    schema = step.spec.output
    messages = _structured_messages(step, state)
    structured = _structured(llm, schema)
    if structured is not None:
        value = _parsed(structured.invoke(messages, step.run_config(config)), schema)
    else:
        value = parse_as(_text(llm.invoke(messages, step.run_config(config))), schema)
    return value.model_dump(mode="json")


async def _astructured_answer(llm, step: Step, state, config) -> Any:
    from llm_kit.core.touchpoints.structured import parse_as
    from llm_kit.core.touchpoints.touchpoint import _parsed, _structured, _text

    schema = step.spec.output
    messages = _structured_messages(step, state)
    structured = _structured(llm, schema)
    if structured is not None:
        value = _parsed(await structured.ainvoke(messages, step.run_config(config)), schema)
    else:
        value = parse_as(_text(await llm.ainvoke(messages, step.run_config(config))), schema)
    return value.model_dump(mode="json")


# ------------------------------------------------------------------ code agents


@dataclass
class AgentContext:
    """What a code agent gets: run its subagents, read the caller and thread."""

    agent: AgentSpec
    config: Any
    env: AgentEnv
    messages: list

    @property
    def caller(self) -> Any:
        return ((self.config or {}).get("configurable") or {}).get("__llm_kit_caller")

    @property
    def thread_id(self) -> str:
        return ((self.config or {}).get("configurable") or {}).get("thread_id") or ""

    def run(self, name: str, task: str) -> AgentResult:
        """Run one of this agent's subagents. Raises ``Refused`` when limits don't allow it."""
        return run_subagent(self.agent, name, task, self.config, self.env, parent_messages=self.messages)

    async def arun(self, name: str, task: str) -> AgentResult:
        return await arun_subagent(self.agent, name, task, self.config, self.env, parent_messages=self.messages)


def _task(state) -> str:
    from llm_kit.core.serialize import message_text

    human = [m for m in state.get("messages") or [] if isinstance(m, HumanMessage)]
    return message_text(human[-1]) if human else ""


def _code_update(result: Any) -> dict:
    if isinstance(result, AgentResult):
        result = result.output if result.output is not None else result.text
    if isinstance(result, BaseModel):
        output = result.model_dump(mode="json")
        return {"messages": [AIMessage(content=json.dumps(output))], "output": output, "stopped": ""}
    if isinstance(result, dict):
        return {"messages": [AIMessage(content=json.dumps(result, default=str))], "output": result, "stopped": ""}
    return {"messages": [AIMessage(content="" if result is None else str(result))], "output": None, "stopped": ""}


def _build_code(spec: AgentSpec, get_env) -> StateGraph:
    def run(state, config):
        env = get_env()
        current = effective(spec, env)
        if (stop := _stop_for(current, config)) is not None:
            return stop
        ctx = AgentContext(agent=current, config=config, env=env, messages=list(state.get("messages") or []))
        result = current.code(_task(state), ctx)
        if inspect.isawaitable(result):
            raise TypeError(f"Code agent {spec.name} is async; run it with ainvoke/astream.")
        return _code_update(result)

    async def arun(state, config):
        if not inspect.iscoroutinefunction(spec.code):
            return await sync_to_async(run, thread_sensitive=True)(state, config)
        env = get_env()
        current = await sync_to_async(effective, thread_sensitive=True)(spec, env)
        if (stop := _stop_for(current, config)) is not None:
            return stop
        ctx = AgentContext(agent=current, config=config, env=env, messages=list(state.get("messages") or []))
        return _code_update(await current.code(_task(state), ctx))

    graph = StateGraph(AgentState)
    graph.add_node("run", RunnableLambda(run, afunc=arun, name="run"))
    graph.add_edge(START, "run")
    graph.add_edge("run", END)
    return graph


__all__ = ["AgentContext", "AgentState", "Refused", "build", "prepare", "resolve_tools"]
