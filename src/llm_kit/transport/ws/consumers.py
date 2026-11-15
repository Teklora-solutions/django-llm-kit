"""Django Channels WebSocket consumer (``pip install django-llm-kit[channels]``).

Routes (see ``llm_kit.transport.ws.routing``)::

    ws/llm/threads/<uuid>/   an existing thread
    ws/llm/threads/          a new thread, created on the first run

Client -> server::

    {"type": "run", "message": "hi", "model": "fast"}    # any llm_kit.runtime.runs payload
    {"type": "run", "resume": {"approved": true}}         # answer an interrupt
    {"type": "run", "graph": "chat"}                      # (new-thread route) pick a graph
    {"type": "run", "message": "hi", "background": true} # start detached, and watch it
    {"type": "run", "message": "hi", "queue": true}      # the same, waiting its turn if busy
    {"type": "watch", "run": "<run id>"}                  # watch a background run
    {"type": "watch", "run": "<run id>", "after": 41}     # ... from the event after seq 41
    {"type": "cancel"}                                    # stop the run (or the one watched)

Server -> client: ``{"event": ..., "data": ...}`` with the events listed in
``llm_kit.runtime.runs``, plus ``thread`` (new thread created), ``cancelled`` and ``busy``,
and for background runs ``run``, ``snapshot`` and ``status``; a background run's events carry
``seq``. Disconnecting stops a run started
on this connection, but never a background run: it carries on, and can be watched again.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from llm_kit.core.errors import BudgetExceeded
from llm_kit.models import Thread
from llm_kit.runtime import access, runs
from llm_kit.runtime.graphs import UnknownGraph, default_graph_name, get_graph

logger = logging.getLogger("llm_kit")

CLOSE_UNAUTHORIZED = 4401
CLOSE_FORBIDDEN = 4403
CLOSE_NOT_FOUND = 4404


class ThreadConsumer(AsyncJsonWebsocketConsumer):
    thread: Thread | None = None
    task: asyncio.Task | None = None
    watched = None  # the BackgroundRun this connection is watching

    @property
    def user(self):
        return self.scope.get("user")

    async def connect(self):
        if not access.can_use_api(self.user):
            await self.close(code=CLOSE_UNAUTHORIZED)
            return
        try:  # a TENANT_RESOLVER that refuses this scope refuses the connection
            await database_sync_to_async(access.owner_from_scope)(self.scope)
        except access.NotAllowed:
            await self.close(code=CLOSE_FORBIDDEN)
            return
        thread_id = self.scope["url_route"]["kwargs"].get("thread_id")
        if thread_id is not None:
            try:
                self.thread = await database_sync_to_async(self._get_thread)(thread_id)
            except Thread.DoesNotExist:
                await self.close(code=CLOSE_NOT_FOUND)
                return
            except access.NotAllowed:
                await self.close(code=CLOSE_FORBIDDEN)
                return
        await self.accept()

    def _get_thread(self, thread_id):
        return access.get_thread(thread_id, access.owner_from_scope(self.scope))

    async def disconnect(self, code):
        if self.task and not self.task.done():
            self.task.cancel()

    async def send_event(self, event: str, data=None, seq: int | None = None):
        message = {"event": event, "data": data}
        if seq is not None:
            message["seq"] = seq
        await self.send_json(message)

    async def receive_json(self, content, **kwargs):
        kind = content.get("type") if isinstance(content, dict) else None
        if kind == "cancel":
            if self.watched is not None:
                await database_sync_to_async(self.watched.cancel)()  # its status event ends the watch
            elif self.task and not self.task.done():
                self.task.cancel()
            return
        if kind not in ("run", "watch"):
            await self.send_event("error", {"message": "Expected a message with type 'run', 'watch' or 'cancel'."})
            return
        if self.task and not self.task.done():
            await self.send_event("busy", {"message": "A run is already in progress on this connection."})
            return
        if kind == "watch":
            after = content.get("after")
            if after is not None and (not isinstance(after, int) or isinstance(after, bool) or after < 0):
                await self.send_event("error", {"message": "after must be an event's seq (a whole number)."})
                return
            self.task = asyncio.create_task(self._watch(content.get("run"), after=after))
        elif content.get("background") is True or content.get("queue") is True:
            self.task = asyncio.create_task(self._start_background(content))
        else:
            self.task = asyncio.create_task(self._run(content))

    def _get_run(self, run_id):
        from llm_kit import background

        return background.get_run(run_id, access.owner_from_scope(self.scope))

    async def _watch(self, run_id, run=None, after=None):
        from llm_kit import background
        from llm_kit.models import BackgroundRun

        if run is None:
            try:
                run = await database_sync_to_async(self._get_run)(uuid.UUID(str(run_id)))
            except (BackgroundRun.DoesNotExist, access.NotAllowed, ValueError):
                await self.send_event("error", {"message": "Run not found."})
                return
        self.watched = run
        try:
            async for item in background.watch(run, after=after):
                await self.send_event(item["event"], item["data"], item.get("seq"))
        finally:
            self.watched = None

    async def _start_background(self, payload):
        from llm_kit import background

        queue = payload.get("queue")
        payload = {k: v for k, v in payload.items() if k not in ("type", "background", "queue")}
        try:
            created = await self._prepare(payload)
            owner = await database_sync_to_async(access.owner_from_scope)(self.scope)
            run = await background.astart(self.thread, payload, owner, queue=queue if isinstance(queue, bool) else None)
        except background.RunBusy as e:
            await self.send_event("busy", {"message": str(e), "run": e.run.to_json() if e.run else None})
            return
        except (UnknownGraph, runs.BadRequest, access.NotAllowed, BudgetExceeded) as e:
            await self.send_event("error", {"message": str(e)})
            return
        if created:
            await self.send_event("thread", {"id": self.thread.thread_id, "graph": self.thread.graph})
        await self.send_event("run", {"id": run.run_id, "status": run.status})
        await self._watch(None, run, after=0)  # from its first event

    @database_sync_to_async
    def _prepare(self, payload):
        if self.thread is None:
            graph = payload.get("graph") or default_graph_name()
            get_graph(graph)
            runs.validate_choice(graph, payload.get("model"))  # before creating anything
            fields = access.owner_fields(access.owner_from_scope(self.scope))
            self.thread = Thread.objects.create(graph=graph, **fields)
            session = self.scope.get("session")
            if fields.get("session_key") and session is not None and getattr(session, "modified", False):
                session.save()  # the visitor id must outlive this connection
            return True
        runs.validate_choice(self.thread.graph, payload.get("model"))
        return False

    def _setup(self, payload):
        return runs.prepare(self.thread, payload, owner=access.owner_from_scope(self.scope))

    async def _run(self, payload):
        try:
            await self._run_now(payload)
        except asyncio.CancelledError:  # cancelled at any point, also before the graph started
            await self.send_event("cancelled", None)
            raise

    async def _run_now(self, payload):
        try:
            created = await self._prepare(payload)
            from llm_kit.background.service import RunBusy, active_run

            if (busy := await database_sync_to_async(active_run)(self.thread)) is not None:
                await self.send_event("busy", {"message": str(RunBusy(busy)), "run": busy.to_json()})
                return
            graph = get_graph(self.thread.graph)
            # Sync: TENANT_RESOLVER and run hooks may query the database.
            setup = await database_sync_to_async(self._setup)(payload)
        except (UnknownGraph, runs.BadRequest, access.NotAllowed, BudgetExceeded, runs.RunRefused) as e:
            await self.send_event("error", {"message": str(e)})
            return
        except runs.HookFailed as e:
            logger.exception("Run hook failed for thread %s", self.thread.pk)
            await self.send_event("error", {"message": runs.error_message(e)})
            return
        if created:
            await self.send_event("thread", {"id": self.thread.thread_id, "graph": self.thread.graph})
        last = None
        async for item in runs.astream_run(graph, setup.input, setup.config, setup.contexts):
            if last is not None:
                await self.send_event(last["event"], last["data"])
            last = item
        await database_sync_to_async(self.thread.save)(update_fields=["updated_at"])
        from llm_kit.background.service import aadvance

        await aadvance(self.thread)  # a run queued meanwhile may start now
        if last is not None:  # end (or error) last: once a client sees it, the connection takes a new run
            await self.send_event(last["event"], last["data"])
