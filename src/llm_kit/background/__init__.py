"""Background runs: runs that keep going without the request that started them, and that
anyone allowed to use the thread can watch or cancel::

    from llm_kit import background

    run = background.start(thread, {"message": "Audit last month's invoices"}, user=request.user)
    ...
    async for event in background.watch(run):   # a snapshot, then live events
        ...
    run.cancel()

Agents: ``llm_kit.agents.start(name, task, user=...)``. Over HTTP and WebSocket: send
``"background": true`` with a run, and watch with ``llm/runs/<id>/watch/`` (see the
streaming guide). Where runs execute: ``RUN_EXECUTOR``; how events reach watchers:
``RUN_EVENTS``.
"""

from llm_kit.background.service import (
    RunBusy,
    aadvance,
    advance,
    aexecute,
    astart,
    cancel,
    execute,
    get_run,
    refresh,
    runs_for,
    start,
    waiting_runs,
    watch,
)
from llm_kit.background.signals import run_finished

__all__ = [
    "RunBusy",
    "aadvance",
    "advance",
    "aexecute",
    "astart",
    "cancel",
    "execute",
    "get_run",
    "refresh",
    "run_finished",
    "runs_for",
    "start",
    "waiting_runs",
    "watch",
]
