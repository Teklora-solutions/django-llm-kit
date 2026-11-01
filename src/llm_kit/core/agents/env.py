"""What the harness needs from its host. The Django adapter supplies ``DjangoAgentEnv``."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool

from llm_kit.core.agents.spec import AgentOverride
from llm_kit.core.errors import ConfigError


class AgentEnv:
    """Defaults for running without a framework: tool objects only, in-memory checkpoints."""

    def __init__(self) -> None:
        self._graphs: dict[str, Any] = {}
        self._lock = threading.Lock()

    def named_tools(self, names: Sequence[str], config: Any) -> list[BaseTool]:
        """Tools from a catalogue by name, for the run's caller."""
        raise ConfigError(f"Tool names {list(names)} need a tool catalogue; pass tool objects instead.")

    def permitted(self, tools: Sequence[BaseTool], config: Any) -> list[BaseTool]:
        """The tools the run's caller may use (catalogue tools are checked; others pass)."""
        return list(tools)

    def guidance(self, tools: Sequence[BaseTool]) -> str:
        """System-prompt lines about these tools."""
        return ""

    def prompt_vars(self, config: Any) -> dict[str, Any]:
        """Values for prompt templates beyond date/now/agent (e.g. user, tenant)."""
        return {}

    def override(self, name: str) -> AgentOverride | None:
        return None

    def graph(self, name: str) -> Any:
        """The compiled graph for an agent, with a checkpointer (subagents run on it)."""
        from langgraph.checkpoint.memory import InMemorySaver

        from llm_kit.core.agents.harness import build
        from llm_kit.core.agents.spec import get_agent

        with self._lock:
            if name not in self._graphs:
                self._graphs[name] = build(get_agent(name), InMemorySaver())
            return self._graphs[name]


_default: AgentEnv | None = None


def set_default_env(env: AgentEnv) -> None:
    global _default
    _default = env


def default_env() -> AgentEnv:
    global _default
    if _default is None:
        _default = AgentEnv()
    return _default
