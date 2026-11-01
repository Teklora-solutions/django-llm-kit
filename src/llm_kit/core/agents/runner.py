"""Running an agent from code: ``run(name, task)`` returns an ``AgentResult``.

The run gets fresh limits (depth, hand-offs, timeout) shared with every subagent it starts.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from llm_kit.core.agents.env import AgentEnv, default_env
from llm_kit.core.agents.limits import STACK_KEY, TREE_KEY, Tree
from llm_kit.core.agents.results import AgentResult, from_state
from llm_kit.core.agents.spec import AgentSpec, get_agent


def root_config(spec: AgentSpec, config: dict[str, Any] | None = None, thread_id: str | None = None) -> dict:
    """``config`` for the agent a request starts: its own run tree, the agent named in the metadata."""
    config = dict(config or {})
    configurable = dict(config.get("configurable") or {})
    configurable["thread_id"] = str(thread_id or configurable.get("thread_id") or uuid.uuid4())
    configurable.setdefault(TREE_KEY, Tree.new())
    configurable.setdefault(STACK_KEY, (spec.name,))
    config["configurable"] = configurable
    config["metadata"] = {
        "llm_kit_graph": spec.name,
        "llm_kit_thread": configurable["thread_id"],
        **(config.get("metadata") or {}),
        "llm_kit_agent": spec.name,
        "llm_kit_depth": len(configurable[STACK_KEY]) - 1,
    }
    config.setdefault("recursion_limit", 2 * spec.max_steps + 8)
    return config


def _start(name, task, config, env, thread_id):
    spec = name if isinstance(name, AgentSpec) else get_agent(name)
    env = env or default_env()
    config = root_config(spec, config, thread_id)
    return spec, env.graph(spec.name), {"messages": [HumanMessage(content=task)]}, config


def run(
    name: str | AgentSpec,
    task: str,
    *,
    config: dict | None = None,
    env: AgentEnv | None = None,
    thread_id: str | None = None,
) -> AgentResult:
    spec, graph, start, config = _start(name, task, config, env, thread_id)
    state = graph.invoke(start, config)
    return from_state(spec, state, config["configurable"]["thread_id"])


async def arun(
    name: str | AgentSpec,
    task: str,
    *,
    config: dict | None = None,
    env: AgentEnv | None = None,
    thread_id: str | None = None,
) -> AgentResult:
    spec, graph, start, config = _start(name, task, config, env, thread_id)
    state = await graph.ainvoke(start, config)
    return from_state(spec, state, config["configurable"]["thread_id"])
