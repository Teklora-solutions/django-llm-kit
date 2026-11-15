"""How long one tool call may take, and what the model is told when it takes longer.

A tool that hangs (a page that never loads) used to hold the agent until the request gave up:
the person got nothing of what had been found, and the model never learned which step failed.
Now each call has a limit; past it, the call ends as a tool **error** the model reads
(``status="error"``: the tool didn't finish, its result is unknown), and the run goes on.

The limit is the lesser of the tool's own timeout (``TOOL_TIMEOUT``, or per tool
``tools.register(..., timeout=)`` / ``metadata={"llm_kit_timeout": s}``; 0 for none) and the
time the run has left (``AGENT_TIMEOUT``) less ``AGENT_ANSWER_RESERVE``, kept back for the
model to write its answer. With less than ``MIN_TOOL_SECONDS`` of that, a tool isn't started,
and the agent stops offering tools and is told to answer now with what it has.

An async tool is cancelled at its limit. A sync tool can't be stopped: it runs on a thread of
its own (a small pool, ``TOOL_THREADS``) and is abandoned at the limit, finishing on that
thread, its result dropped; the host closes that thread's database connections when it ends.
Inside a database transaction a sync tool runs where it was called (it must see the
transaction's data), so it can't be abandoned there.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import functools
import inspect
import logging
import threading
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, ToolException

from llm_kit.core import config as settings
from llm_kit.core.agents.limits import running_config, tree_of

logger = logging.getLogger("llm_kit.agents")

MIN_TOOL_SECONDS = 5.0  # less than this left is not worth starting a tool for
TIMEOUT_KEY = "llm_kit_timeout"  # tool.metadata: this tool's own limit in seconds (0: none)
SYNC_KEY = "llm_kit_sync"  # tool.metadata: its coroutine only hands sync work to a thread

__all__ = ["MIN_TOOL_SECONDS", "TIMEOUT_KEY", "TimeLimited", "seconds_for", "time_left"]


def time_left(config: Any = None) -> float | None:
    """Seconds until this run's deadline (``AGENT_TIMEOUT``), or None outside an agent run.

    For tools: ``time_left(config)`` with the ``config`` they're given, or with no argument
    inside the run."""
    tree = tree_of(config if config is not None else running_config())
    return None if tree is None else tree.deadline - time.monotonic()


def _reserve() -> float:
    return float(settings.get("AGENT_ANSWER_RESERVE") or 0)


def own_timeout(tool: BaseTool) -> float | None:
    value = (tool.metadata or {}).get(TIMEOUT_KEY)
    if value is None:
        value = settings.get("TOOL_TIMEOUT")
    return float(value) if value else None


def seconds_for(tool: BaseTool | None, config: Any, *, own: bool = True) -> float | None:
    """How long the next call of ``tool`` may take: None for no limit, 0 for none at all."""
    limit = own_timeout(tool) if own and tool is not None else None
    left = time_left(config)
    if left is None:
        return limit
    allowed = left - _reserve()
    if allowed < MIN_TOOL_SECONDS:
        return 0.0
    return allowed if limit is None else min(limit, allowed)


def out_of_time(config: Any) -> bool:
    """Too little time left to run a tool and still answer."""
    left = time_left(config)
    return left is not None and left - _reserve() < MIN_TOOL_SECONDS


def no_time(name: str) -> str:
    return (
        f"Not run: there is no time left in this request to run {name}. Answer now with what you "
        "already have, and say plainly what you could not check."
    )


def took_too_long(name: str, seconds: float) -> str:
    return (
        f"{name} did not finish within {round(seconds, 1):g} seconds and was stopped. Its result is unknown: "
        "don't guess it. Try something else, or answer with what you have and say what you could not check."
    )


# ------------------------------------------------------------------ the thread pool for sync tools

_pool: concurrent.futures.ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _executor() -> concurrent.futures.ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            size = int(settings.get("TOOL_THREADS") or 32)
            _pool = concurrent.futures.ThreadPoolExecutor(max_workers=size, thread_name_prefix="llm_kit-tool")
        return _pool


def _submit(env, call) -> concurrent.futures.Future:
    context = contextvars.copy_context()  # the run's config, caller and the host's ContextVars
    return _executor().submit(context.run, env.run_tool_thread, call)


def is_async(tool: BaseTool) -> bool:
    """Whether ``tool`` runs natively async (so cancelling it stops it)."""
    if (tool.metadata or {}).get(SYNC_KEY):
        return False
    if isinstance(tool, StructuredTool):
        return tool.coroutine is not None
    return type(tool)._arun is not BaseTool._arun


def _bound(fn, args, kwargs, config, run_manager):
    params = inspect.signature(fn).parameters
    extra = {}
    if "config" in params:
        extra["config"] = config
    if "run_manager" in params:
        extra["run_manager"] = run_manager
    return functools.partial(fn, *args, **extra, **kwargs)


class TimeLimited(BaseTool):
    """``inner`` under a time limit. Same name, description and arguments, so the model and
    LangChain's callbacks see the one tool: a call that runs out of time ends as that tool's
    error (``on_tool_error``), like any other failure."""

    inner: BaseTool
    seconds: float
    env: Any = None

    @classmethod
    def around(cls, inner: BaseTool, seconds: float, env: Any) -> TimeLimited:
        return cls(
            name=inner.name,
            description=inner.description,
            args_schema=inner.args_schema,
            return_direct=inner.return_direct,
            response_format=inner.response_format,
            handle_tool_error=inner.handle_tool_error,
            handle_validation_error=inner.handle_validation_error,
            metadata=inner.metadata,
            tags=inner.tags,
            callbacks=inner.callbacks,
            inner=inner,
            seconds=seconds,
            env=env,
        )

    def _expired(self) -> ToolException:
        logger.warning("Tool %s did not finish within %.0fs; stopped", self.name, self.seconds)
        return ToolException(took_too_long(self.name, self.seconds))

    def _run(self, *args: Any, config: RunnableConfig, run_manager=None, **kwargs: Any) -> Any:
        call = _bound(self.inner._run, args, kwargs, config, run_manager)
        if self.env.in_transaction():  # it must see the transaction's data: run it here
            return call()
        future = _submit(self.env, call)
        done, _ = concurrent.futures.wait([future], timeout=self.seconds)
        if not done:
            future.cancel()  # if it never started (all threads busy), it never will
            raise self._expired()
        return future.result()

    async def _arun(self, *args: Any, config: RunnableConfig, run_manager=None, **kwargs: Any) -> Any:
        if is_async(self.inner):
            task = asyncio.ensure_future(_bound(self.inner._arun, args, kwargs, config, run_manager)())
            done, _ = await asyncio.wait({task}, timeout=self.seconds)
            if not done:
                task.cancel()
                with contextlib.suppress(BaseException):
                    await asyncio.wait({task}, timeout=1)  # let it clean up, briefly
                raise self._expired()
            return task.result()
        from asgiref.sync import sync_to_async

        sync_manager = run_manager.get_sync() if run_manager is not None else None
        if await sync_to_async(self.env.in_transaction, thread_sensitive=True)():
            return await _bound(self.inner._arun, args, kwargs, config, run_manager)()
        future = _submit(self.env, _bound(self.inner._run, args, kwargs, config, sync_manager))
        done, _ = await asyncio.wait({asyncio.wrap_future(future)}, timeout=self.seconds)
        if not done:
            future.cancel()
            raise self._expired()
        return future.result()
