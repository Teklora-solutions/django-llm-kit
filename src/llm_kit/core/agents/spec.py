"""Declaring agents. One harness runs them all; an agent is only this declaration.

    from llm_kit import agent

    researcher = agent(
        "crm.researcher",
        description="Digs through invoices and notes to answer one narrow question.",
        touchpoint="crm.cheap",
        prompt="Investigate the task. Report facts with record ids. Today is {date}.",
        tools=tools.select(kinds=["find"]),
    )

    @agent.code("editorial.newsroom", description="One full editorial cycle.", subagents=["editorial.writer"])
    def newsroom(task, ctx):
        draft = ctx.run("editorial.writer", task)
        return draft.text

A subagent is the same harness with a different declaration: model (touchpoint),
prompt and tools change, everything else is shared.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from llm_kit.core.errors import ConfigError, LLMKitError

DEFAULT_MAX_STEPS = 12
_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


class UnknownAgent(LLMKitError, LookupError):
    pass


@dataclass(frozen=True)
class AgentOverride:
    """What an admin changed (``AgentOverride`` rows). It can narrow an agent, never widen it."""

    prompt: str = ""
    prompt_mode: str = "append"  # "append" | "replace"
    max_steps: int | None = None  # only lowers
    tools: tuple[str, ...] | None = None  # only these of the agent's tools (None = all of them)
    paused: str = ""  # why; a paused agent answers with this instead of calling a model


@dataclass(frozen=True, eq=False)
class AgentSpec:
    name: str
    description: str
    touchpoint: str | None = None
    prompt: str | Callable[..., str] = ""
    tools: Any = None  # None | [BaseTool | "name" | fn(config), ...] | fn(config) -> [BaseTool]
    subagents: tuple[str, ...] = ()
    output: type[BaseModel] | None = None
    max_steps: int = DEFAULT_MAX_STEPS
    fork: bool = False
    defer_tools: bool | None = None  # None: when it has more than DEFER_TOOLS_ABOVE tools
    essential: tuple[str, ...] = ()  # when deferring: the tools bound from the start
    history: int | None = None  # tokens of conversation sent to the model (newest kept); None = all
    tool_result_chars: int | None = None  # each tool result cut to this many characters; None = no cap
    code: Callable[..., Any] | None = None
    paused: str = ""  # set by an override; see ``with_override``
    allowed_tools: frozenset[str] | None = None  # set by an override: only these tool names

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise ConfigError(f"Agent name {self.name!r} must be lowercase dotted words, like 'crm.researcher'.")
        if not (self.description or "").strip():
            raise ConfigError(f"Agent {self.name!r}: give a description; it is how other agents pick it.")
        if isinstance(self.subagents, str):
            object.__setattr__(self, "subagents", (self.subagents,))
        object.__setattr__(self, "subagents", tuple(self.subagents))
        object.__setattr__(self, "essential", tuple(self.essential))
        if self.name in self.subagents:
            raise ConfigError(f"Agent {self.name!r} lists itself as a subagent.")
        if self.max_steps < 1:
            raise ConfigError(f"Agent {self.name!r}: max_steps must be at least 1.")
        if self.tool_result_chars is not None and self.tool_result_chars < 1:
            raise ConfigError(f"Agent {self.name!r}: tool_result_chars must be positive, or None.")
        if self.history is not None and self.history < 1:
            raise ConfigError(f"Agent {self.name!r}: history must be a positive number of tokens, or None.")
        if self.output is not None and not (isinstance(self.output, type) and issubclass(self.output, BaseModel)):
            raise ConfigError(f"Agent {self.name!r}: output must be a pydantic model class.")
        if self.code is not None:
            if self.touchpoint or self.tools or self.prompt:
                raise ConfigError(f"Code agent {self.name!r} runs Python, so it takes no touchpoint, prompt or tools.")
            return
        if not self.touchpoint:
            raise ConfigError(f"Agent {self.name!r}: give a touchpoint (its model policy).")
        if not (isinstance(self.prompt, str) or callable(self.prompt)):
            raise ConfigError(f"Agent {self.name!r}: prompt must be a string or fn(ctx) -> str.")
        if self.essential and self.defer_tools is False:
            raise ConfigError(
                f"Agent {self.name!r}: essential= only applies when tools are deferred, not defer_tools=False."
            )
        tools = self.tools
        if tools is not None and not callable(tools):
            if isinstance(tools, (str, BaseTool)) or not isinstance(tools, Sequence):
                raise ConfigError(f"Agent {self.name!r}: tools must be a list, tools.select(...), or fn(config).")
            for t in tools:
                if not (isinstance(t, (str, BaseTool)) or callable(t)):
                    raise ConfigError(f"Agent {self.name!r}: {t!r} is not a tool, a tool name or tools.select(...).")

    @property
    def is_code(self) -> bool:
        return self.code is not None

    def get_touchpoint(self):
        from llm_kit.core.touchpoints import get_touchpoint

        return get_touchpoint(self.touchpoint)

    def with_override(self, override: AgentOverride | None) -> AgentSpec:
        """This declaration as an admin narrowed it."""
        if override is None:
            return self
        changes: dict[str, Any] = {"paused": override.paused or ""}
        if override.prompt.strip() and not self.is_code:
            if override.prompt_mode == "replace":
                changes["prompt"] = override.prompt
            else:
                changes["prompt"] = _AppendedPrompt(self.prompt, override.prompt)
        if override.max_steps:
            changes["max_steps"] = max(1, min(self.max_steps, int(override.max_steps)))
        if override.tools is not None:
            changes["allowed_tools"] = frozenset(override.tools)
        return replace(self, **changes)


@dataclass(frozen=True)
class _AppendedPrompt:
    base: Any
    extra: str

    def __call__(self, ctx) -> str:
        from llm_kit.core.agents.prompt import render

        return render(self.base, ctx) + "\n\n" + render(self.extra, ctx)


# ------------------------------------------------------------------ registry

_registry: dict[str, AgentSpec] = {}
_lock = threading.Lock()


def _register(spec: AgentSpec) -> AgentSpec:
    with _lock:
        _registry[spec.name] = spec
    return spec


def agent(
    name: str,
    *,
    description: str,
    touchpoint: Any = None,
    prompt: str | Callable[..., str] = "",
    tools: Any = None,
    subagents: Sequence[str] = (),
    output: type[BaseModel] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    fork: bool = False,
    defer_tools: bool | None = None,
    essential: Sequence[str] = (),
    history: int | None = None,
    tool_result_chars: int | None = None,
) -> AgentSpec:
    """Declare and register an agent run by the shared harness. Re-declaring a name replaces it.

    Args:
        touchpoint: its model policy (a touchpoint or its name): default model, client choice,
            admin overrides, fallbacks, recording.
        prompt: a template (``{date}``, ``{now}``, ``{agent}``, ``{user}``, ``{tenant}``, plus any
            ``configurable["prompt_vars"]``) or ``fn(ctx) -> str``.
        tools: ``tools.select(...)``, ``fn(config) -> tools``, or a list mixing tools, tool names
            and selections. Catalogue tools are always limited to the ones the caller may use.
        subagents: agents it may hand work to with the ``delegate`` tool.
        output: a pydantic model; the final answer is also returned in that shape.
        max_steps: model calls per turn; on the last one it is told to answer.
        fork: as a subagent, start from a copy of the parent's conversation instead of only the task.
        defer_tools: bind only ``essential`` tools plus ``find_tools``; the rest load when found.
            None (default): defer when the agent can use more than ``DEFER_TOOLS_ABOVE`` tools.
        history: about how many tokens of the conversation each model call sees, newest first,
            always starting at a user message (the checkpoint keeps everything). None: all of it.
        tool_result_chars: cut each tool result to this many characters, with a note saying so
            (fetched pages and documents are the largest thing in a context window). None: no cap.
    """
    tp_name = getattr(touchpoint, "name", touchpoint)
    return _register(
        AgentSpec(
            name=name,
            description=description,
            touchpoint=tp_name,
            prompt=prompt,
            tools=tools,
            subagents=tuple(subagents),
            output=output,
            max_steps=max_steps,
            fork=fork,
            defer_tools=defer_tools,
            essential=tuple(essential),
            history=history,
            tool_result_chars=tool_result_chars,
        )
    )


def _code(name: str, *, description: str, subagents: Sequence[str] = ()) -> Callable:
    """Declare a code agent: ``fn(task, ctx)`` (sync or async) instead of a model loop.

    It returns text, a dict or a pydantic model. ``ctx.run(name, task)`` runs one of its
    ``subagents``. Code agents are graphs, can be delegated to, and are recorded like any other.
    """

    def decorate(fn: Callable) -> AgentSpec:
        return _register(AgentSpec(name=name, description=description, subagents=tuple(subagents), code=fn))

    return decorate


agent.code = _code  # type: ignore[attr-defined]


def get_agent(name: str) -> AgentSpec:
    try:
        return _registry[name]
    except KeyError:
        raise UnknownAgent(f"Unknown agent {name!r}. Registered: {', '.join(sorted(_registry)) or 'none'}.") from None


def all_agents() -> list[AgentSpec]:
    return sorted(_registry.values(), key=lambda a: a.name)


def _unregister(name: str) -> None:
    """For tests."""
    _registry.pop(name, None)


def problems(known_touchpoints: Sequence[str] | None = None) -> list[str]:
    """What is wrong with the declarations as a whole: unknown subagents or touchpoints, cycles."""
    out = []
    for spec in all_agents():
        for sub in spec.subagents:
            if sub not in _registry:
                out.append(f"{spec.name}: subagent {sub!r} is not a declared agent")
        if known_touchpoints is not None and spec.touchpoint and spec.touchpoint not in known_touchpoints:
            out.append(f"{spec.name}: touchpoint {spec.touchpoint!r} is not declared")
    for cycle in _cycles():
        out.append(f"agents delegate in a circle: {' -> '.join(cycle)}")
    return out


def _cycles() -> list[list[str]]:
    found, done = [], set()

    def visit(name: str, path: list[str]) -> None:
        if name in path:
            cycle = path[path.index(name) :] + [name]
            if frozenset(cycle) not in done:
                done.add(frozenset(cycle))
                found.append(cycle)
            return
        spec = _registry.get(name)
        for sub in spec.subagents if spec else ():
            visit(sub, [*path, name])

    for name in sorted(_registry):
        visit(name, [])
    return found
