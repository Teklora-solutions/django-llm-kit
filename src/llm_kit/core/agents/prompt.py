"""An agent's system prompt: its own prompt, then tool guidance and the agents it can hand work to."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_VAR = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


@dataclass
class PromptContext:
    """What a prompt function gets (``prompt=fn``); templates see ``vars`` as ``{name}``."""

    agent: Any  # the AgentSpec
    config: Any
    vars: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__["vars"][name]
        except KeyError:
            raise AttributeError(name) from None


def context_for(spec, config, env) -> PromptContext:
    now = datetime.now().astimezone()
    variables: dict[str, Any] = {
        "agent": spec.name,
        "date": now.date().isoformat(),
        "now": now.strftime("%Y-%m-%d %H:%M %Z").strip(),
        "user": "",
        "tenant": "",
    }
    variables.update(env.prompt_vars(config) or {})
    variables.update(((config or {}).get("configurable") or {}).get("prompt_vars") or {})
    return PromptContext(agent=spec, config=config, vars=variables)


def render(prompt, ctx: PromptContext) -> str:
    """A template's known ``{names}`` are filled in; other braces (JSON examples) are left alone."""
    if callable(prompt):
        return str(prompt(ctx) or "")
    return _VAR.sub(lambda m: str(ctx.vars[m.group(1)]) if m.group(1) in ctx.vars else m.group(0), prompt or "")


def build_prompt(spec, ctx: PromptContext, *, guidance: str = "", subagents: list = (), last_step: bool = False) -> str:
    parts = [render(spec.prompt, ctx).strip()]
    if guidance.strip():
        parts.append("## Tools\n" + guidance.strip())
    if subagents:
        lines = "\n".join(f"- `{s.name}`: {s.description}" for s in subagents)
        parts.append(
            "## Other agents\nYou can hand a self-contained task to one of these with the `delegate` tool. "
            "It does the work and replies with its answer. It doesn't see this conversation unless noted, so put "
            "everything it needs in the task.\n" + lines
        )
    if last_step:
        parts.append("This is your last step: answer now with what you have. Don't call any more tools.")
    return "\n\n".join(p for p in parts if p)
