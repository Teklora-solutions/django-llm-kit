"""LangGraph's own Postgres checkpointer, on the database Django already uses.

``pip install django-llm-kit[postgres]``, then::

    LLM_KIT = {"CHECKPOINTER": "postgres"}

which means ``{"sync": sync_saver, "async": async_saver}``: a pooled ``PostgresSaver`` for sync
code, and an ``AsyncPostgresSaver`` with its own pool for each event loop. Tables are created
on first use (``setup()``). Pass your own factories for other databases, pool sizes or options::

    LLM_KIT = {"CHECKPOINTER": {"sync": "myapp.ai:saver", "async": "myapp.ai:asaver"}}
"""

from __future__ import annotations

from typing import Any

SAVERS = {"sync": "llm_kit.persistence.postgres:sync_saver", "async": "llm_kit.persistence.postgres:async_saver"}

_CONNECTION = {"autocommit": True, "prepare_threshold": 0}


def conninfo(alias: str = "default") -> str:
    """A libpq connection string for Django's database ``alias`` (the test database under tests)."""
    from django.db import connections
    from psycopg.conninfo import make_conninfo

    s = connections[alias].settings_dict
    if s.get("ENGINE", "").rsplit(".", 1)[-1] not in ("postgresql", "postgis"):
        raise ValueError(f"Database {alias!r} is not PostgreSQL; the Postgres checkpointer needs it.")
    params: dict[str, Any] = {
        "dbname": s.get("NAME"),
        "user": s.get("USER") or None,
        "password": s.get("PASSWORD") or None,
        "host": s.get("HOST") or None,
        "port": str(s["PORT"]) if s.get("PORT") else None,
    }
    for key in ("sslmode", "sslrootcert", "sslcert", "sslkey", "connect_timeout", "options", "application_name"):
        if key in (s.get("OPTIONS") or {}):
            params[key] = s["OPTIONS"][key]
    return make_conninfo(**{k: v for k, v in params.items() if v is not None})


def sync_saver(alias: str = "default", max_size: int = 10):
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo(alias), kwargs={**_CONNECTION, "row_factory": dict_row}, max_size=max_size, open=True
    )
    saver = PostgresSaver(pool)
    saver.setup()
    saver.llm_kit_close = pool.close  # CheckpointRouter.close()
    return saver


async def async_saver(alias: str = "default", max_size: int = 10):
    from asgiref.sync import sync_to_async
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row

    info = await sync_to_async(conninfo)(alias)
    # One pool per event loop, and loops can be short-lived (asyncio.run in a task): start small.
    pool = _loop_pool_class()(
        info, kwargs={**_CONNECTION, "row_factory": dict_row}, min_size=1, max_size=max_size, open=False
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    saver.llm_kit_close = lambda: _finish(pool)
    return saver


_LOOP_POOL = None


def _loop_pool_class():
    """``AsyncConnectionPool`` whose worker tasks stop when cancelled.

    psycopg_pool's workers catch ``CancelledError`` raised while they run a job (adding a
    connection, say) and go back to waiting for the next one. When ``asyncio.run`` ends and
    cancels the tasks still running, a worker caught mid-job then never ends, and
    ``asyncio.run`` waits for it forever. These workers check ``Task.cancelling()`` after
    each job and stop.
    """
    global _LOOP_POOL
    if _LOOP_POOL is not None:
        return _LOOP_POOL
    import asyncio
    import logging

    from psycopg_pool import AsyncConnectionPool

    try:
        from psycopg_pool.pool_async import StopWorker
    except ImportError:  # a psycopg_pool without it: its own workers
        _LOOP_POOL = AsyncConnectionPool
        return _LOOP_POOL

    logger = logging.getLogger("psycopg.pool")

    class LoopPool(AsyncConnectionPool):
        @classmethod
        async def worker(cls, q) -> None:
            me = asyncio.current_task()
            while True:
                job = await q.get()
                if isinstance(job, StopWorker):
                    return
                try:
                    await job.run()
                except Exception as ex:
                    logger.warning("task run %s failed: %s: %s", job, ex.__class__.__name__, ex)
                if me is not None and me.cancelling():
                    raise asyncio.CancelledError

    _LOOP_POOL = LoopPool
    return _LOOP_POOL


def _finish(pool) -> None:
    """Close an async pool's idle connections without its event loop, which may be gone.

    The pool's own ``close()`` is a coroutine of the loop that opened it; once ``asyncio.run``
    has returned that loop is closed, and the connections would stay open until garbage
    collection. libpq's ``finish()`` closes them synchronously.
    """
    idle = getattr(pool, "_pool", None) or ()
    while idle:
        conn = idle.pop()
        try:
            conn.pgconn.finish()
        except Exception:  # already closed
            pass
