"""Where background runs execute: ``LLM_KIT["RUN_EXECUTOR"]``.

* ``"thread"`` (default): a thread in this process. Nothing to set up; a run is lost if the
  process stops (a deploy, a restart), and is then marked ``lost``.
* ``"celery"``: the ``llm_kit.run_in_background`` task on your Celery workers.
* ``"django_q"``: ``llm_kit.background.tasks.run_in_background`` through Django-Q2.
* ``"pkg.module:function"``: your own; it is called with the run id (a string) once the run is
  saved, and must arrange for ``llm_kit.background.execute(run_id)`` to be called somewhere.

With a worker executor, set ``CHANNEL_LAYERS`` (e.g. Redis) so watchers in the web processes
see the worker's events (``RUN_EVENTS``).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable

_lock = threading.Lock()
_running: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Task | None]] = {}
_threads: dict[str, threading.Thread] = {}


def thread_executor(run_id: str) -> None:
    from llm_kit.background.service import execute

    def target():
        from django.db import connections

        try:
            execute(run_id)
        finally:
            connections.close_all()
            with _lock:
                _threads.pop(run_id, None)

    thread = threading.Thread(target=target, daemon=True, name=f"llm-kit-run-{run_id[:8]}")
    with _lock:
        _threads[run_id] = thread
    thread.start()


def celery_executor(run_id: str) -> None:
    from llm_kit.background.tasks import run_in_background_task

    if run_in_background_task is None:
        from llm_kit.core.errors import ConfigError

        raise ConfigError('RUN_EXECUTOR="celery" needs Celery installed.')
    run_in_background_task.delay(run_id)


def django_q_executor(run_id: str) -> None:
    from django_q.tasks import async_task

    async_task("llm_kit.background.tasks.run_in_background", run_id)


BUILT_IN: dict[str, Callable[[str], None]] = {
    "thread": thread_executor,
    "celery": celery_executor,
    "django_q": django_q_executor,
}


def get_executor(name: str) -> Callable[[str], None]:
    if name in BUILT_IN:
        return BUILT_IN[name]
    if ":" not in name and "." not in name:
        from llm_kit.core.errors import ConfigError

        raise ConfigError(
            f'RUN_EXECUTOR must be "thread", "celery", "django_q" or "pkg.module:function", not {name!r}.'
        )
    from llm_kit.core.graphs import import_target

    return import_target(name)


# ------------------------------------------------------------------ runs executing in this process


def register_local(run_id: str, loop: asyncio.AbstractEventLoop, task: asyncio.Task | None) -> None:
    with _lock:
        _running[run_id] = (loop, task)


def unregister_local(run_id: str) -> None:
    with _lock:
        _running.pop(run_id, None)


def cancel_local(run_id: str) -> bool:
    """Cancel a run executing in this process at once (others notice the flag within a second)."""
    with _lock:
        entry = _running.get(run_id)
    if entry is None or entry[1] is None:
        return False
    loop, task = entry
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(task.cancel)
        return True
    return False


def join(run_id: str, timeout: float | None = None) -> bool:
    """Wait for a run started by the thread executor in this process (tests, management commands).
    True once it has finished (or was never running here)."""
    with _lock:
        thread = _threads.get(run_id)
    if thread is None:
        return True
    thread.join(timeout)
    return not thread.is_alive()
