"""Agents: one harness, and every agent (and subagent) is a declaration run by it.

Declare in ``<your_app>/agents.py`` (imported at start-up)::

    from llm_kit import agent, tools

    agent(
        "crm.assistant",
        description="Answers staff questions about customers and invoices.",
        touchpoint="crm.assistant",
        prompt="You help {user} with customers and invoices. Today is {date}.",
        tools=tools.select(categories=["crm"]),
        subagents=["crm.researcher"],
    )

Each agent is a graph by its name (streaming endpoints, threads, checkpoints), can hand
work to its subagents with the ``delegate`` tool, and can be run from code::

    llm_kit.agents.run("crm.researcher", "Which invoices are overdue?", user=request.user)
"""

from llm_kit.agents.run import arun, config_for, run, stream
from llm_kit.core.agents import (
    AgentContext,
    AgentOverride,
    AgentResult,
    AgentSpec,
    PromptContext,
    Refused,
    UnknownAgent,
    agent,
    all_agents,
    get_agent,
)

__all__ = [
    "AgentContext",
    "AgentOverride",
    "AgentResult",
    "AgentSpec",
    "PromptContext",
    "Refused",
    "UnknownAgent",
    "agent",
    "all_agents",
    "arun",
    "config_for",
    "get_agent",
    "run",
    "stream",
]
