"""Admin overrides for agents (``AgentOverride`` rows), cached per process and reloaded when edited."""

from __future__ import annotations

from llm_kit.core.agents.spec import AgentOverride
from llm_kit.runtime.versioned_cache import VersionedCache

_cache = VersionedCache("agent_overrides")


def _load() -> dict[str, AgentOverride]:
    from llm_kit.models import AgentOverride as Row

    return {
        row.agent: AgentOverride(
            prompt=row.prompt or "",
            prompt_mode=row.prompt_mode,
            max_steps=row.max_steps,
            tools=tuple(row.tools) if row.tools else None,
            paused=row.paused or "",
        )
        for row in Row.objects.all()
    }


def override_for(name: str) -> AgentOverride | None:
    return _cache.get(_load).get(name)


def invalidate() -> None:
    _cache.bump()
