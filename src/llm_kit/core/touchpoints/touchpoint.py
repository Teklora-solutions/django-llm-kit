"""Touchpoints: each place your code calls an LLM, declared once with its model policy.

    from llm_kit import touchpoint, ClientChoice

    summarise = touchpoint(
        "crm.summarise_lead",
        default="anthropic/claude-haiku-5-5",
        fallbacks=["gemini/gemini-3.8-flash"],
        params={"temperature": 0},
        requires={"structured_output": True},
    )

    llm = summarise.model()                       # LangChain chat model (with fallbacks)
    lead = summarise.complete("...", schema=Lead) # one-shot structured call

In a graph node, pass the node's ``config`` so the run's requested model
(``configurable.model``) and tracing metadata flow through::

    def node(state, config):
        return {"messages": [chat.model(config=config).invoke(state["messages"], config)]}
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from llm_kit.core.catalogue.match import REQUIREMENT_KEYS
from llm_kit.core.errors import ConfigError, UnknownTouchpoint
from llm_kit.core.touchpoints.env import ResolutionEnv, default_env
from llm_kit.core.touchpoints.policy import Choice, ServerChoice
from llm_kit.core.touchpoints.resolve import Resolution, resolve
from llm_kit.core.touchpoints.retry import RetryPolicy, policy_for
from llm_kit.core.touchpoints.structured import parse_as

_registry: dict[str, Touchpoint] = {}
_lock = threading.Lock()


def _requested_from(config: RunnableConfig | None) -> str | None:
    return ((config or {}).get("configurable") or {}).get("model")


def _messages(prompt: str | Sequence[BaseMessage] | Sequence[Any]) -> list:
    return [HumanMessage(content=prompt)] if isinstance(prompt, str) else list(prompt)


@dataclass(frozen=True, eq=False)
class Touchpoint:
    name: str
    default: str
    fallbacks: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    choice: Choice = field(default_factory=ServerChoice)
    requires: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    backend: str | None = None  # None = LLM_KIT["MODEL_BACKEND"]
    capture: bool | None = None  # store prompts/responses; None = LLM_KIT["CAPTURE_CONTENT"]
    retry: RetryPolicy | None = None  # waits before falling back; None = LLM_KIT["RETRY"]

    def __post_init__(self) -> None:
        bad = [k for k in self.requires if k not in REQUIREMENT_KEYS]
        if bad:
            raise ConfigError(
                f"Touchpoint {self.name!r}: unknown requirement(s) {bad}. Use: {', '.join(REQUIREMENT_KEYS)}."
            )
        if isinstance(self.fallbacks, str):
            object.__setattr__(self, "fallbacks", (self.fallbacks,))
        object.__setattr__(self, "retry", RetryPolicy.coerce(self.retry))

    # ----------------------------------------------------------- resolution

    def resolve(
        self, requested: str | None = None, *, config: RunnableConfig | None = None, env: ResolutionEnv | None = None
    ) -> Resolution:
        """Which model(s) a call would use, and why. Raises ``ModelNotAllowed`` / ``TouchpointDisabled``."""
        return resolve(self, requested if requested is not None else _requested_from(config), env or default_env())

    # --------------------------------------------------------------- models

    def model(
        self,
        requested: str | None = None,
        *,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        **params: Any,
    ) -> Runnable:
        """A chat model for this touchpoint, wrapped with its fallbacks.

        ``params`` override the touchpoint's params for this call (e.g. ``max_tokens``).
        ``retry`` overrides how long a short rate limit is waited out before falling back.
        ``fallbacks=False`` uses only the resolved model for this call (results compared
        across runs shouldn't come from whichever model answered); it still waits per ``retry``.
        The returned runnable supports ``invoke``/``stream``/``bind_tools``/``with_structured_output``.
        """
        from llm_kit.core.touchpoints.backends import build_for

        resolution = self.resolve(requested, config=config, env=env)
        if not fallbacks:
            resolution.fallbacks = []
        if params:
            resolution.params = {**resolution.params, **params}
        # Inside a running agent, its retry policy applies (but not its requested model).
        run = config if config is not None else _running()
        return build_for(self, resolution, env or default_env(), retry=policy_for(self, run, retry))

    # ------------------------------------------------------- one-shot calls

    def complete(
        self,
        prompt: str | Sequence[BaseMessage] | Sequence[Any],
        *,
        schema: type[BaseModel] | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        **params: Any,
    ) -> Any:
        """Call the model once. With ``schema``, return a validated instance of it.

        Uses native structured output when the model supports it, otherwise
        parses JSON out of the reply (code fences and surrounding prose are tolerated).
        """
        llm = self.model(requested, config=config, env=env, retry=retry, fallbacks=fallbacks, **params)
        messages = _messages(prompt)
        run_config = self._run_config(config)
        if schema is None:
            return _as_run(llm).invoke(messages, run_config)
        structured = _structured(llm, schema)
        if structured is not None:
            return _parsed(structured.invoke(messages, run_config), schema)
        reply = _as_run(llm).invoke(messages, run_config)
        return parse_as(_text(reply), schema)

    async def acomplete(
        self,
        prompt: str | Sequence[BaseMessage] | Sequence[Any],
        *,
        schema: type[BaseModel] | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        **params: Any,
    ) -> Any:
        llm = self.model(requested, config=config, env=env, retry=retry, fallbacks=fallbacks, **params)
        messages = _messages(prompt)
        run_config = self._run_config(config)
        if schema is None:
            return await _as_run(llm).ainvoke(messages, run_config)
        structured = _structured(llm, schema)
        if structured is not None:
            return _parsed(await structured.ainvoke(messages, run_config), schema)
        reply = await _as_run(llm).ainvoke(messages, run_config)
        return parse_as(_text(reply), schema)

    def run_config(self, config: RunnableConfig | None = None) -> RunnableConfig | None:
        """The config to invoke this touchpoint's ``model()`` with, so the call is recorded.

        Inside a graph node (config already has callbacks) the call joins that run. Called
        without one from inside a running agent (a tool, a code agent, code they call), it
        joins that agent's run too: recorded in its run tree and counted in its totals.
        Otherwise it is a recorded run of its own, named after the touchpoint. Metadata you
        pass (for example ``{"llm_kit_agent": "crm.writer"}``) is recorded with the call."""
        if config and config.get("callbacks"):
            return config
        if (parent := _running()) is not None and parent.get("callbacks") is not None:
            mine = config or {}
            return {
                **mine,
                "callbacks": parent["callbacks"],
                "metadata": {**(parent.get("metadata") or {}), **(mine.get("metadata") or {})},
                "configurable": {**(parent.get("configurable") or {}), **(mine.get("configurable") or {})},
                "run_name": mine.get("run_name") or self.name,
            }
        from llm_kit.core.observability.handler import get_handler

        handler = get_handler()
        if handler is None:
            return config
        return {**(config or {}), "callbacks": [handler], "run_name": self.name}

    _run_config = run_config


def _running() -> RunnableConfig | None:
    from llm_kit.core.agents.limits import running_config

    return running_config()


def _as_run(llm: Runnable) -> Runnable:
    """A one-step chain, so the call gets a root run (and a Run record)."""
    from langchain_core.runnables import RunnablePassthrough

    return RunnablePassthrough() | llm


def _structured(llm: Runnable, schema: type[BaseModel]) -> Runnable | None:
    try:
        return llm.with_structured_output(schema, include_raw=True)
    except (NotImplementedError, AttributeError):
        return None


def _parsed(result: dict[str, Any], schema: type[BaseModel]) -> Any:
    """Native structured output can return nothing when the model answers in text; parse the text then."""
    if result.get("parsed") is not None:
        return result["parsed"]
    return parse_as(_text(result["raw"]), schema)


def _text(reply: Any) -> str:
    from llm_kit.core.serialize import message_text

    return message_text(reply) if not isinstance(reply, str) else reply


def touchpoint(
    name: str,
    *,
    default: str,
    fallbacks: Sequence[str] = (),
    params: dict[str, Any] | None = None,
    choice: Choice | None = None,
    requires: dict[str, Any] | None = None,
    description: str = "",
    backend: str | None = None,
    capture: bool | None = None,
    retry: RetryPolicy | dict | str | None = None,
) -> Touchpoint:
    """Declare and register a touchpoint. Re-declaring the same name replaces it (module reloads)."""
    tp = Touchpoint(
        name=name,
        default=default,
        fallbacks=tuple(fallbacks),
        params=dict(params or {}),
        choice=choice or ServerChoice(),
        requires=dict(requires or {}),
        description=description,
        backend=backend,
        capture=capture,
        retry=retry,
    )
    with _lock:
        _registry[name] = tp
    return tp


def get_touchpoint(name: str) -> Touchpoint:
    try:
        return _registry[name]
    except KeyError:
        raise UnknownTouchpoint(
            f"Unknown touchpoint {name!r}. Registered: {', '.join(sorted(_registry)) or 'none'}."
        ) from None


def all_touchpoints() -> list[Touchpoint]:
    return sorted(_registry.values(), key=lambda t: t.name)


def _unregister(name: str) -> None:
    """For tests."""
    _registry.pop(name, None)


__all__ = ["RetryPolicy", "Touchpoint", "all_touchpoints", "get_touchpoint", "touchpoint", "BaseChatModel"]
