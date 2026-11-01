"""The harness's view of Django: the tool catalogue, the caller, overrides, checkpointed graphs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool

from llm_kit.core.agents.env import AgentEnv
from llm_kit.core.agents.spec import AgentOverride


class DjangoAgentEnv(AgentEnv):
    def named_tools(self, names: Sequence[str], config: Any) -> list[BaseTool]:
        from llm_kit.tools import registry

        return registry.for_config(config, names=set(names))

    def permitted(self, tools: Sequence[BaseTool], config: Any) -> list[BaseTool]:
        """Catalogue tools only for a caller allowed to use them; the developer's own tools pass."""
        from llm_kit.tools import registry
        from llm_kit.tools.caller import caller_from

        entries = {e.name: e for e in registry.catalogue()}
        caller = caller_from(config)
        return [t for t in tools if t.name not in entries or entries[t.name].allowed(caller)]

    def guidance(self, tools: Sequence[BaseTool]) -> str:
        from llm_kit.tools import registry

        return registry.guidance(tools)

    def prompt_vars(self, config: Any) -> dict[str, Any]:
        from llm_kit.tools.caller import caller_from

        caller = caller_from(config)
        if caller is None or not caller.is_authenticated:
            return {}
        user = caller.user
        name = (getattr(user, "get_full_name", lambda: "")() or "").strip() or user.get_username()
        return {"user": name, "tenant": "" if caller.tenant is None else str(caller.tenant)}

    def override(self, name: str) -> AgentOverride | None:
        from llm_kit.agents.overrides import override_for

        return override_for(name)

    def graph(self, name: str) -> Any:
        from llm_kit.runtime.graphs import get_graph

        return get_graph(name)
