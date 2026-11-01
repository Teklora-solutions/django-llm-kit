"""Django Channels WebSocket consumer (``pip install django-llm-kit[channels]``).

Routes (see ``llm_kit.transport.ws.routing``)::

    ws/llm/threads/<uuid>/   an existing thread
    ws/llm/threads/          a new thread, created on the first run

Client -> server::

    {"type": "run", "message": "hi", "model": "fast"}    # any llm_kit.runtime.runs payload
    {"type": "run", "resume": {"approved": true}}         # answer an interrupt
    {"type": "run", "graph": "chat"}                      # (new-thread route) pick a graph
    {"type": "cancel"}

Server -> client: ``{"event": ..., "data": ...}`` with the events listed in
``llm_kit.runtime.runs``, plus ``thread`` (new thread created), ``cancelled`` and ``busy``.
"""

from __future__ import annotations

import asyncio

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from llm_kit.models import Thread
from llm_kit.runtime import access, runs
from llm_kit.runtime.graphs import UnknownGraph, default_graph_name, get_graph

CLOSE_UNAUTHORIZED = 4401
CLOSE_FORBIDDEN = 4403
CLOSE_NOT_FOUND = 4404


class ThreadConsumer(AsyncJsonWebsocketConsumer):
    thread: Thread | None = None
    task: asyncio.Task | None = None

    @property
    def user(self):
        return self.scope.get("user")

    async def connect(self):
        if not access.can_use_api(self.user):
            await self.close(code=CLOSE_UNAUTHORIZED)
            return
        thread_id = self.scope["url_route"]["kwargs"].get("thread_id")
        if thread_id is not None:
            try:
                self.thread = await database_sync_to_async(access.get_thread)(thread_id, self.user)
            except Thread.DoesNotExist:
                await self.close(code=CLOSE_NOT_FOUND)
                return
            except access.NotAllowed:
                await self.close(code=CLOSE_FORBIDDEN)
                return
        await self.accept()

    async def disconnect(self, code):
        if self.task and not self.task.done():
            self.task.cancel()

    async def send_event(self, event: str, data=None):
        await self.send_json({"event": event, "data": data})

    async def receive_json(self, content, **kwargs):
        kind = content.get("type") if isinstance(content, dict) else None
        if kind == "cancel":
            if self.task and not self.task.done():
                self.task.cancel()
            return
        if kind != "run":
            await self.send_event("error", {"message": "Expected a message with type 'run' or 'cancel'."})
            return
        if self.task and not self.task.done():
            await self.send_event("busy", {"message": "A run is already in progress on this connection."})
            return
        self.task = asyncio.create_task(self._run(content))

    @database_sync_to_async
    def _prepare(self, payload):
        if self.thread is None:
            graph = payload.get("graph") or default_graph_name()
            get_graph(graph)
            runs.validate_choice(graph, payload.get("model"))  # before creating anything
            self.thread = Thread.objects.create(graph=graph, user=access.owner_for_new_thread(self.user))
            return True
        runs.validate_choice(self.thread.graph, payload.get("model"))
        return False

    async def _run(self, payload):
        try:
            created = await self._prepare(payload)
            graph = get_graph(self.thread.graph)
            graph_input = runs.build_input(payload)
            config = runs.build_config(self.thread, payload, self.user)
        except (UnknownGraph, runs.BadRequest) as e:
            await self.send_event("error", {"message": str(e)})
            return
        if created:
            await self.send_event("thread", {"id": self.thread.thread_id, "graph": self.thread.graph})
        try:
            async for item in runs.astream_run(graph, graph_input, config):
                await self.send_event(item["event"], item["data"])
        except asyncio.CancelledError:
            await self.send_event("cancelled", None)
            raise
        await database_sync_to_async(self.thread.save)(update_fields=["updated_at"])
