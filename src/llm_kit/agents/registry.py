"""Agents as graphs, and the start-up check of every declaration.

Every declared agent is a graph by its name: the SSE, API and WebSocket endpoints, threads,
the model picker and checkpoints work for it without listing it in ``GRAPHS``.
"""

from __future__ import annotations

import functools

from django.core.exceptions import ImproperlyConfigured

from llm_kit.core.agents.harness import build
from llm_kit.core.agents.spec import AgentSpec, all_agents, problems


def graph_targets() -> dict[str, dict]:
    targets = {}
    for spec in all_agents():
        target = {"graph": functools.partial(_compile, spec.name), "agent": spec.name}
        if spec.touchpoint:
            target["touchpoint"] = spec.touchpoint
        targets[spec.name] = target
    return targets


def _compile(name: str, checkpointer):
    from llm_kit.core.agents.spec import get_agent

    return build(get_agent(name), checkpointer)


def is_agent(graph: str) -> bool:
    from llm_kit.core.agents.spec import _registry

    return graph in _registry


def _tool_names(spec: AgentSpec) -> list[str]:
    from llm_kit.tools.registry import ToolSelection

    tools = spec.tools
    if isinstance(tools, ToolSelection):
        return tools.unknown_names()
    if isinstance(tools, (list, tuple)):
        from llm_kit.tools import registry

        known = {e.name for e in registry.catalogue()}
        unknown = [t for t in tools if isinstance(t, str) and t not in known]
        return unknown + [n for t in tools if isinstance(t, ToolSelection) for n in t.unknown_names()]
    return []


def declared_tool_names(spec: AgentSpec) -> list[str]:
    """Every tool the declaration names or selects, whoever runs it (``fn(config)`` tools can't be listed)."""
    from llm_kit.tools.registry import ToolSelection

    items = spec.tools if isinstance(spec.tools, (list, tuple)) else [spec.tools] if spec.tools else []
    names = []
    for t in items:
        if isinstance(t, ToolSelection):
            names += [e.name for e in t.entries()]
        elif isinstance(t, str) or hasattr(t, "name"):
            names.append(t if isinstance(t, str) else t.name)
    return list(dict.fromkeys(names))


def check() -> list[str]:
    """Problems with the agent declarations (empty when all is well)."""
    from llm_kit import conf
    from llm_kit.core.touchpoints import all_touchpoints

    found = problems(known_touchpoints=[t.name for t in all_touchpoints()])
    graphs = conf.get("GRAPHS")
    for spec in all_agents():
        if spec.name in graphs:
            found.append(f"{spec.name}: an agent and a GRAPHS entry share this name")
        for name in _tool_names(spec):
            found.append(f"{spec.name}: tool {name!r} is not in the tool catalogue")
    return found


def validate() -> None:
    """Called at start-up: a bad declaration stops it here, not at the first run."""
    found = check()
    if found:
        raise ImproperlyConfigured("Agent declarations have problems:\n- " + "\n- ".join(found))
