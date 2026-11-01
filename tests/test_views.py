import json

import pytest

from llm_kit.models import Thread

pytestmark = pytest.mark.django_db(transaction=True)


async def read_events(response):
    assert response["Content-Type"] == "text/event-stream"
    text = "".join([c.decode() async for c in response.streaming_content])
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def post(client, url, body):
    return await client.post(url, json.dumps(body), content_type="application/json")


async def test_new_thread_streams_tokens_and_persists(async_client, user):
    await async_client.aforce_login(user)
    events = await read_events(await post(async_client, "/llm/stream/", {"message": "hello there"}))
    names = [e for e, _ in events]
    assert names[0] == "thread" and names[-1] == "end"
    tokens = "".join(d["message"]["content"] for e, d in events if e == "token")
    assert tokens == "echo: hello there"
    assert "update" in names

    thread = await Thread.objects.aget(pk=events[0][1]["id"])
    assert thread.user_id == user.pk and thread.graph == "chat" and thread.title == "hello there"

    events = await read_events(await post(async_client, f"/llm/threads/{thread.id}/stream/", {"message": "more"}))
    assert "".join(d["message"]["content"] for e, d in events if e == "token") == "echo: more"


async def test_interrupt_then_resume(async_client, user):
    await async_client.aforce_login(user)
    events = await read_events(
        await post(async_client, "/llm/stream/", {"graph": "approval", "input": {"request": "deploy", "log": []}})
    )
    interrupt = next(d for e, d in events if e == "interrupt")
    assert interrupt["interrupts"][0]["value"] == {"question": "Approve deploy?"}
    assert events[-1] == ("end", {"next": ["ask"], "interrupts": interrupt["interrupts"]})

    tid = events[0][1]["id"]
    events = await read_events(await post(async_client, f"/llm/threads/{tid}/stream/", {"resume": "ok"}))
    assert events[-1] == ("end", {"next": [], "interrupts": []})
    assert {"ask": {"log": ["approved=ok"]}} in [d for e, d in events if e == "update"]


@pytest.fixture
def synced_catalogue(db):
    from llm_kit import catalogue
    from tests.catalogue_payloads import FakeSources

    catalogue.sync(fetch=FakeSources())


async def test_model_choice_goes_through_the_touchpoint(async_client, user, built, keys, synced_catalogue):
    await async_client.aforce_login(user)
    body = {"graph": "catalogue", "message": "hi", "model": "google/gemini-3.8-flash"}
    events = await read_events(await post(async_client, "/llm/stream/", body))
    assert events[-1][0] == "end"
    assert [b[0] for b in built] == ["gemini/gemini-3.8-flash"]  # client's pick, normalised; no fallback needed

    built.clear()
    events = await read_events(await post(async_client, "/llm/stream/", {"graph": "catalogue", "message": "hi"}))
    assert [b[0] for b in built] == [
        "anthropic/claude-sonnet-5-5",
        "gemini/gemini-3.8-flash",
    ]  # code default + fallback


async def test_disallowed_model_is_rejected_before_a_thread_exists(async_client, user, keys, synced_catalogue):
    await async_client.aforce_login(user)
    before = await Thread.objects.acount()
    r = await post(async_client, "/llm/stream/", {"graph": "catalogue", "message": "x", "model": "openai/gpt-5.5"})
    assert r.status_code == 400 and "not in allowed patterns" in json.loads(r.content)["detail"]
    r = await post(
        async_client, "/llm/stream/", {"graph": "chat", "message": "x", "model": "anthropic/claude-sonnet-5-5"}
    )
    assert r.status_code == 400 and "does not accept a model choice" in json.loads(r.content)["detail"]
    assert await Thread.objects.acount() == before


async def test_errors(async_client, user, other_user):
    # Anonymous: auth required by default.
    assert (await post(async_client, "/llm/stream/", {"message": "x"})).status_code == 401
    await async_client.aforce_login(user)
    assert (await post(async_client, "/llm/stream/", {"graph": "nope"})).status_code == 400
    assert (await post(async_client, "/llm/stream/", {"model": 5, "message": "x"})).status_code == 400
    assert (await async_client.post("/llm/stream/", "not json", content_type="application/json")).status_code == 400
    assert (await async_client.get("/llm/stream/")).status_code == 405

    theirs = await Thread.objects.acreate(graph="chat", user=other_user)
    assert (await post(async_client, f"/llm/threads/{theirs.id}/stream/", {"message": "x"})).status_code == 403
    missing = "00000000-0000-0000-0000-000000000000"
    assert (await post(async_client, f"/llm/threads/{missing}/stream/", {"message": "x"})).status_code == 404


async def test_graph_failure_becomes_error_event(async_client, user, settings):
    await async_client.aforce_login(user)
    events = await read_events(await post(async_client, "/llm/stream/", {"graph": "boom", "message": "x"}))
    assert events[-1] == ("error", {"message": "The run failed."})  # details hidden unless DEBUG
    settings.DEBUG = True
    events = await read_events(await post(async_client, "/llm/stream/", {"graph": "boom", "message": "x"}))
    assert events[-1] == ("error", {"message": "RuntimeError: kaboom"})


async def test_anonymous_mode(async_client, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "REQUIRE_AUTH": False}
    events = await read_events(await post(async_client, "/llm/stream/", {"message": "anon"}))
    assert events[-1][0] == "end"
    thread = await Thread.objects.aget(pk=events[0][1]["id"])
    assert thread.user_id is None


async def test_keepalive_is_sent_while_quiet():
    import asyncio

    from llm_kit.transport.sse.views import with_keepalive

    async def slow():
        await asyncio.sleep(0.05)
        yield {"event": "end", "data": {}}

    out = [chunk async for chunk in with_keepalive(slow(), 0.01)]
    assert out[0] == ": keepalive\n\n" and out[-1] == "event: end\ndata: {}\n\n"
