"""Limits shared by one run tree: an agent and every subagent it starts.

The tree lives in the run's config (``configurable["__llm_kit_tree"]``). Like the caller,
it is set by the server; LangGraph never writes ``"__"`` keys to checkpoints, and clients
can't send them.
"""

from __future__ import annotations

import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from llm_kit.core import config as settings

TREE_KEY = "__llm_kit_tree"
STACK_KEY = "__llm_kit_agents"  # names from the root agent down to the running one
# Set on a throwaway run (no thread to continue): it and its subagents keep their state in
# memory, never in the database. Copied to subagents with the other "__llm_kit_" keys.
EPHEMERAL_KEY = "__llm_kit_ephemeral"
# Metadata key naming the request's tree, so the recording handler can add each call's tokens
# to it as the call ends (the tree itself stays out of metadata, which tracers serialise).
REQUEST_KEY = "llm_kit_request"

_live: weakref.WeakValueDictionary[str, Tree] = weakref.WeakValueDictionary()


def is_ephemeral(config) -> bool:
    return bool(((config or {}).get("configurable") or {}).get(EPHEMERAL_KEY))


@dataclass
class Tree:
    max_depth: int
    max_delegations: int
    deadline: float  # time.monotonic() after which no new model call starts
    delegations: int = 0
    threads: list[str] = field(default_factory=list)  # subagent threads started in this tree
    budget: Decimal | None = None  # USD this request may spend; None = no limit
    # model -> [input, output, cache_read, cache_write, reasoning] tokens used so far
    usage: dict[str, list[int]] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        _live[self.id] = self

    @classmethod
    def new(cls, budget: Any = None) -> Tree:
        if budget is None:
            budget = settings.get("AGENT_BUDGET")
        return cls(
            max_depth=int(settings.get("AGENT_MAX_DEPTH")),
            max_delegations=int(settings.get("AGENT_MAX_DELEGATIONS")),
            deadline=time.monotonic() + float(settings.get("AGENT_TIMEOUT")),
            budget=None if budget is None else Decimal(str(budget)),
        )

    def add_usage(self, model: str, tokens: tuple[int, int, int, int, int]) -> None:
        with self._lock:
            total = self.usage.setdefault(model or "", [0, 0, 0, 0, 0])
            for i, n in enumerate(tokens):
                total[i] += int(n or 0)

    def spent(self) -> Decimal:
        """USD spent so far, from catalogue prices (models with no known price count as 0)."""
        from llm_kit.core.observability.cost import price_call
        from llm_kit.core.observability.recorder import recorder

        lookup = recorder.price_lookup
        with self._lock:
            usage = {m: list(t) for m, t in self.usage.items()}
        total = Decimal(0)
        for model, (i, o, cr, cw, r) in usage.items():
            try:
                prices = lookup(model) if lookup is not None and model else None
            except Exception:
                prices = None
            cost = price_call(
                prices, input_tokens=i, output_tokens=o, cache_read_tokens=cr, cache_write_tokens=cw, reasoning_tokens=r
            )
            total += cost or 0
        return total

    @property
    def over_budget(self) -> bool:
        return self.budget is not None and self.spent() >= self.budget

    def take_delegation(self) -> bool:
        with self._lock:
            if self.delegations >= self.max_delegations:
                return False
            self.delegations += 1
            return True

    def add_thread(self, thread_id: str) -> None:
        with self._lock:
            self.threads.append(thread_id)

    @property
    def timed_out(self) -> bool:
        return time.monotonic() > self.deadline


def live_tree(request_id: str) -> Tree | None:
    """The tree of a request still running, by its ``REQUEST_KEY`` metadata value."""
    return _live.get(request_id) if request_id else None


def tree_of(config) -> Tree | None:
    return ((config or {}).get("configurable") or {}).get(TREE_KEY)


def running_config() -> dict | None:
    """The config of the llm-kit run this code is running inside (a tool, a code agent, code
    they call), or None outside one."""
    from langchain_core.runnables.config import var_child_runnable_config

    config = var_child_runnable_config.get()
    return config if tree_of(config) is not None else None


def stack_of(config) -> tuple[str, ...]:
    return tuple(((config or {}).get("configurable") or {}).get(STACK_KEY) or ())


def depth_of(config) -> int:
    """0 for the agent a request started, 1 for its subagents, ..."""
    return max(0, len(stack_of(config)) - 1)
