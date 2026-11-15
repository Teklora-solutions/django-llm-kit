"""LangGraph's own Postgres checkpointer, on the database Django already uses.

``pip install django-llm-kit[postgres]``, then::

    LLM_KIT = {"CHECKPOINTER": "postgres"}

which means ``{"sync": sync_saver, "async": async_saver}``: a pooled ``PostgresSaver`` for sync
code, and an ``AsyncPostgresSaver`` with its own pool for each event loop. Tables are created
on first use: LangGraph's ``setup()``, once per process and database, under a Postgres advisory
lock, so processes and event loops starting together don't race to apply the same migration.
Pass your own factories for other databases, pool sizes or options::

    LLM_KIT = {"CHECKPOINTER": {"sync": "myapp.ai:saver", "async": "myapp.ai:asaver"}}
"""

from __future__ import annotations

import threading
from typing import Any

SAVERS = {"sync": "llm_kit.persistence.postgres:sync_saver", "async": "llm_kit.persistence.postgres:async_saver"}

_CONNECTION = {"autocommit": True, "prepare_threshold": 0}

# Session advisory lock held around LangGraph's setup(), which reads the highest applied
# migration and then inserts the missing ones: two at once insert the same version and one
# fails. A session lock (not a transaction one): some migrations use CREATE INDEX CONCURRENTLY,
# which can't run in a transaction.
SETUP_LOCK = 0x6C6C6D5F6B6974  # "llm_kit"-ish; any bigint shared by every process

_ready: set[str] = set()  # databases whose tables this process has set up
_ready_lock = threading.Lock()


def ensure_tables(info: str) -> None:
    """Create or migrate LangGraph's checkpoint tables in the database at ``info``: once per
    process, and one process at a time."""
    if info in _ready:
        return
    with _ready_lock:
        if info in _ready:
            return
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row

        with psycopg.connect(info, **_CONNECTION, row_factory=dict_row) as conn:
            conn.execute("SELECT pg_advisory_lock(%s)", (SETUP_LOCK,))
            try:
                PostgresSaver(conn).setup()
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (SETUP_LOCK,))
        _ready.add(info)


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

    info = conninfo(alias)
    ensure_tables(info)
    pool = ConnectionPool(info, kwargs={**_CONNECTION, "row_factory": dict_row}, max_size=max_size, open=True)
    saver = PostgresSaver(pool)
    saver.llm_kit_close = pool.close  # CheckpointRouter.close()
    return saver


async def async_saver(alias: str = "default", max_size: int = 10):
    import asyncio

    from asgiref.sync import sync_to_async
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row

    info = await sync_to_async(conninfo)(alias)
    if info not in _ready:
        await asyncio.to_thread(ensure_tables, info)  # plain psycopg: no Django thread needed
    # One pool per event loop, and loops can be short-lived (asyncio.run in a task): start small.
    pool = _loop_pool_class()(
        info, kwargs={**_CONNECTION, "row_factory": dict_row}, min_size=1, max_size=max_size, open=False
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)
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
