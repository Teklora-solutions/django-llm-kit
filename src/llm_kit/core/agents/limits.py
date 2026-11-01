"""Limits shared by one run tree: an agent and every subagent it starts.

The tree lives in the run's config (``configurable["__llm_kit_tree"]``). Like the caller,
it is set by the server; LangGraph never writes ``"__"`` keys to checkpoints, and clients
can't send them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from llm_kit.core import config as settings

TREE_KEY = "__llm_kit_tree"
STACK_KEY = "__llm_kit_agents"  # names from the root agent down to the running one


@dataclass
class Tree:
    max_depth: int
    max_delegations: int
    deadline: float  # time.monotonic() after which no new model call starts
    delegations: int = 0
    threads: list[str] = field(default_factory=list)  # subagent threads started in this tree
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def new(cls) -> Tree:
        return cls(
            max_depth=int(settings.get("AGENT_MAX_DEPTH")),
            max_delegations=int(settings.get("AGENT_MAX_DELEGATIONS")),
            deadline=time.monotonic() + float(settings.get("AGENT_TIMEOUT")),
        )

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


def tree_of(config) -> Tree | None:
    return ((config or {}).get("configurable") or {}).get(TREE_KEY)


def stack_of(config) -> tuple[str, ...]:
    return tuple(((config or {}).get("configurable") or {}).get(STACK_KEY) or ())


def depth_of(config) -> int:
    """0 for the agent a request started, 1 for its subagents, ..."""
    return max(0, len(stack_of(config)) - 1)
