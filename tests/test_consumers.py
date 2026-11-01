import pytest
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator

from llm_kit.models import Thread
from llm_kit.routing import websocket_urlpatterns

pytestmark = pytest.mark.django_db(transaction=True)

app = URLRouter(websocket_urlpatterns)


def communicator(path, user=None):
    c = WebsocketCommunicator(app, path)
    from django.contrib.auth.models import AnonymousUser

    c.scope["user"] = user or AnonymousUser()
    return c


async def collect(c, until=("end", "error")):
    events = []
    while True:
        msg = await c.receive_json_from(timeout=10)
        events.append(msg)
        if msg["event"] in until:
            return events


async def test_new_thread_then_reuse(user):
    c = communicator("/ws/llm/threads/", user)
    assert (await c.connect())[0]
    await c.send_json_to({"type": "run", "message": "hey"})
    events = await collect(c)
    assert events[0]["event"] == "thread"
    assert "".join(e["data"]["message"]["content"] for e in events if e["event"] == "token") == "echo: hey"
    # Same connection, same thread.
    await c.send_json_to({"type": "run", "message": "again"})
    events = await collect(c)
    assert events[0]["event"] != "thread"
    await c.disconnect()

    thread = await Thread.objects.aget(user=user)
    c = communicator(f"/ws/llm/threads/{thread.id}/", user)
    assert (await c.connect())[0]
    await c.disconnect()


async def test_interrupt_resume_over_websocket(user):
    thread = await Thread.objects.acreate(graph="approval", user=user)
    c = communicator(f"/ws/llm/threads/{thread.id}/", user)
    await c.connect()
    await c.send_json_to({"type": "run", "input": {"request": "ship", "log": []}})
    events = await collect(c)
    assert any(e["event"] == "interrupt" for e in events)
    await c.send_json_to({"type": "run", "resume": "y"})
    events = await collect(c)
    assert events[-1] == {"event": "end", "data": {"next": [], "interrupts": []}}
    await c.disconnect()


async def test_rejections(user, other_user):
    c = communicator("/ws/llm/threads/")
    connected, code = await c.connect()
    assert not connected and code == 4401

    theirs = await Thread.objects.acreate(graph="chat", user=other_user)
    connected, code = await communicator(f"/ws/llm/threads/{theirs.id}/", user).connect()
    assert not connected and code == 4403

    connected, code = await communicator("/ws/llm/threads/00000000-0000-0000-0000-000000000000/", user).connect()
    assert not connected and code == 4404

    c = communicator("/ws/llm/threads/", user)
    await c.connect()
    await c.send_json_to({"type": "bogus"})
    assert (await c.receive_json_from())["event"] == "error"
    await c.send_json_to({"type": "run", "graph": "nope"})
    assert (await c.receive_json_from())["event"] == "error"
    await c.disconnect()


async def test_cancel_and_busy(user):
    thread = await Thread.objects.acreate(graph="slow", user=user)
    c = communicator(f"/ws/llm/threads/{thread.id}/", user)
    await c.connect()
    await c.send_json_to({"type": "run", "message": "zzz"})
    await c.send_json_to({"type": "run", "message": "again"})
    assert (await c.receive_json_from(timeout=5))["event"] == "busy"
    await c.send_json_to({"type": "cancel"})
    events = await collect(c, until=("cancelled",))
    assert events[-1] == {"event": "cancelled", "data": None}
    await c.disconnect()
