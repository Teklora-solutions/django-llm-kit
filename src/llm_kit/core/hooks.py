"""Run hooks: per-run policy your server sets, applied wherever a graph runs.

``LLM_KIT["RUN_HOOKS"] = ["myapp.llm:prepare"]`` (every graph and agent), or per graph
``GRAPHS = {"support": {"graph": ..., "prepare": "myapp.llm:prepare"}}``. Each hook is a plain
function (it may query the database) called with a ``RunSetup`` just before the run starts, in
the process that runs it: the view for a streamed or REST run, the worker for a background run
(from the saved payload and owner, every time, so it sees permissions as they are then)::

    def prepare(run):
        run.set_state(audience="staff", can_refund=run.owner.user.has_perm("shop.refund"))
        run.set_configurable(__corpus="kb-2026")      # server-only config, never from a client
        run.add_callback(ActivityRecorder(run.thread))
        return [use_channel("web"), telemetry_row(run)]  # context managers, sync or async

* ``set_state(...)``: keys merged over the graph's input (or the resumed run's state), after
  the client's: nothing a client sends can set or replace them. Set every key your graph relies
  on, every time.
* ``set_configurable(...)``: ``configurable`` keys, over the client's.
* ``run.input`` may also be changed in place (turn attachments into content blocks, ...).
* Context managers (returned, or added with ``enter()``) are entered around the whole run, in the
  order given, and exited when it ends however it ends: set ContextVars, write a row in
  ``finally``.
* ``run.context``: facts only your server knows (the channel, an API key's narrowed permissions,
  attachment ids already checked), passed by your own code with ``context=``
  (``background.start``, ``agents.run``/``start``, ``runtime.runs.prepare``). No transport sets it
  from a request, so a client can't. A small JSON object, saved with a background run.

A hook that raises fails the run before anything runs: ``RunRefused`` (and ``BadRequest``)
refuse it (403/400); anything else is an ``error`` event, a 500, or a background run in status
``error``. Hooks run in order: ``RUN_HOOKS``, then the graph's own.
"""

from __future__ import annotations

import contextlib
import copy
import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from llm_kit.core import config as settings
from llm_kit.core.errors import BadRequest, LLMKitError

logger = logging.getLogger("llm_kit")

CONTEXT_KEY = "__llm_kit_context"  # configurable: the run's server-set context (never from a client)
MAX_CONTEXT_BYTES = 16_384  # it is saved with a background run, for the worker

__all__ = ["CONTEXT_KEY", "HookFailed", "RunRefused", "RunSetup", "aentered", "entered", "hooks_for", "prepare"]


class RunRefused(LLMKitError):
    """Raised by a run hook to refuse a run (403; the message is shown)."""


class HookFailed(LLMKitError):
    """A run hook raised something other than ``RunRefused``/``BadRequest``."""


@dataclass
class RunSetup:
    """What a run hook sees and changes. ``payload`` is a read-only copy of what the client sent."""

    graph: str
    input: Any
    config: dict[str, Any]
    payload: Mapping[str, Any]
    owner: Any = None  # who the run acts for (the Django side: an ``access.Owner``)
    thread: Any = None  # the ``Thread``, when the run has one
    background: bool = False
    # Facts only the server knows, passed by your own code (``context=``): never from a request.
    context: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    state: dict[str, Any] = field(default_factory=dict)
    contexts: list[Any] = field(default_factory=list)

    def set_state(self, **values: Any) -> None:
        self.state.update(values)

    def set_configurable(self, **values: Any) -> None:
        self.config.setdefault("configurable", {}).update(values)

    def add_callback(self, callback: Any) -> None:
        current = self.config.get("callbacks")
        if current is None or isinstance(current, list):
            self.config["callbacks"] = [*(current or []), callback]
        else:  # a callback manager
            current.add_handler(callback, inherit=True)

    def enter(self, manager: Any) -> None:
        """Enter ``manager`` (a sync or async context manager) around the run."""
        if not (hasattr(manager, "__enter__") or hasattr(manager, "__aenter__")):
            raise HookFailed(f"{manager!r} is not a context manager.")
        self.contexts.append(manager)


def check_context(context: Any) -> dict[str, Any]:
    """``context`` as it will be stored: a small JSON object. Raises ``ValueError``."""
    import json

    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise ValueError("context must be a dict.")
    try:
        encoded = json.dumps(dict(context))
    except (TypeError, ValueError) as e:
        raise ValueError(f"context must be JSON-serialisable: {e}") from None
    if len(encoded.encode()) > MAX_CONTEXT_BYTES:
        raise ValueError(f"context is over {MAX_CONTEXT_BYTES} bytes; pass ids, not data.")
    return json.loads(encoded)


def context_of(config: Any) -> dict[str, Any]:
    """The server-set context of the run ``config`` belongs to (for tools)."""
    return dict(((config or {}).get("configurable") or {}).get(CONTEXT_KEY) or {})


def _load(hook: Any) -> Any:
    if isinstance(hook, str):
        from llm_kit.core.graphs import import_target

        return import_target(hook)
    return hook


def hooks_for(graph_hook: Any = None) -> list[Any]:
    """``RUN_HOOKS`` then the graph's own ``prepare``."""
    hooks = settings.get("RUN_HOOKS") or []
    if isinstance(hooks, str) or callable(hooks):
        hooks = [hooks]
    hooks = list(hooks)
    if graph_hook:
        hooks.extend(graph_hook if isinstance(graph_hook, (list, tuple)) else [graph_hook])
    return [_load(h) for h in hooks]


def _merge_state(input: Any, state: dict[str, Any]) -> Any:
    if not state:
        return input
    from langgraph.types import Command

    if isinstance(input, Command):
        update = input.update
        if update is None:
            update = {}
        elif not isinstance(update, dict):
            raise HookFailed("set_state() can't be combined with a non-dict Command update.")
        return Command(graph=input.graph, update={**update, **state}, resume=input.resume, goto=input.goto)
    if input is None:
        return Command(update=dict(state))
    if not isinstance(input, dict):
        raise HookFailed(f"set_state() needs the run's input to be a dict, not {type(input).__name__}.")
    return {**input, **state}


def prepare(
    graph: str,
    input: Any,
    config: dict[str, Any],
    *,
    payload: Mapping[str, Any] | None = None,
    owner: Any = None,
    thread: Any = None,
    background: bool = False,
    graph_hook: Any = None,
    context: Mapping[str, Any] | None = None,
) -> RunSetup:
    """Run the hooks for a run of ``graph``; returns the setup with the final input, config and
    context managers. Raises ``RunRefused``/``BadRequest``, or ``HookFailed`` for anything else."""
    run = RunSetup(
        graph=graph,
        input=input,
        config=config,
        payload=MappingProxyType(copy.deepcopy(dict(payload or {}))),
        owner=owner,
        thread=thread,
        background=background,
        context=MappingProxyType(copy.deepcopy(dict(context or {}))),
    )
    if context:  # tools and background subagents started by this run see it too
        config.setdefault("configurable", {})[CONTEXT_KEY] = dict(run.context)
    for hook in hooks_for(graph_hook):
        try:
            result = hook(run)
        except (RunRefused, BadRequest):
            raise
        except Exception as e:
            name = getattr(hook, "__qualname__", repr(hook))
            raise HookFailed(f"Run hook {name} failed: {type(e).__name__}: {e}") from e
        if result is None:
            continue
        if hasattr(result, "__enter__") or hasattr(result, "__aenter__"):
            run.enter(result)
        else:
            for manager in result:
                run.enter(manager)
    run.input = _merge_state(run.input, run.state)
    return run


@contextlib.asynccontextmanager
async def aentered(contexts) -> AsyncIterator[None]:
    """Enter sync and async context managers, in order, around an async run."""
    async with contextlib.AsyncExitStack() as stack:
        for manager in contexts or ():
            if hasattr(manager, "__aenter__"):
                await stack.enter_async_context(manager)
            else:
                stack.enter_context(manager)
        yield


@contextlib.contextmanager
def entered(contexts) -> Iterator[None]:
    """Enter context managers around a sync run; an async one is entered on an event loop."""
    with contextlib.ExitStack() as stack:
        for manager in contexts or ():
            if hasattr(manager, "__enter__"):
                stack.enter_context(manager)
            else:
                from asgiref.sync import async_to_sync

                async_to_sync(manager.__aenter__)()
                stack.push(lambda *exc, m=manager: async_to_sync(m.__aexit__)(*exc))
        yield
