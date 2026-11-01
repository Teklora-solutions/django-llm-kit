"""Who owns a thread: users, anonymous visitors (adopted at login), keys chosen by code, tenants."""

import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from llm_kit import agents
from llm_kit.models import Thread
from llm_kit.runtime import access
from llm_kit.runtime.threads import open_thread
from tests.graphs import SCRIPTS

pytestmark = pytest.mark.django_db(transaction=True)

THREADS = "/llm/api/threads/"


@pytest.fixture
def anonymous_mode(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "REQUIRE_AUTH": False}


def new_thread(client, **body):
    r = client.post(THREADS, {"graph": "chat", **body}, content_type="application/json")
    assert r.status_code == 201, r.content
    return r.json()["id"]


def run(client, thread_id, message="hi"):
    return client.post(f"{THREADS}{thread_id}/runs/", {"message": message}, content_type="application/json")


# ------------------------------------------------------------------ visitors


def test_a_visitor_owns_their_threads_through_the_session(anonymous_mode):
    visitor, stranger = Client(), Client()
    tid = new_thread(visitor)
    thread = Thread.objects.get(pk=tid)
    assert thread.user_id is None and len(thread.session_key) == 32
    assert run(visitor, tid).status_code == 200
    assert [t["id"] for t in visitor.get(THREADS).json()] == [tid]  # listed for its visitor
    # Another visitor knowing the id gets nothing, and lists nothing.
    assert run(stranger, tid).status_code == 403
    assert stranger.get(f"{THREADS}{tid}/").status_code == 403
    assert stranger.get(THREADS).json() == []


def test_visitors_need_anonymous_mode():
    r = Client().post(THREADS, {"graph": "chat"}, content_type="application/json")
    assert r.status_code == 403


def test_signing_in_adopts_the_visitors_threads(anonymous_mode, user):
    visitor = Client()
    tid = new_thread(visitor)
    visitor.force_login(user)  # login() rotates the session key; the visitor id in the data survives
    thread = Thread.objects.get(pk=tid)
    assert (thread.user_id, thread.session_key) == (user.pk, "")
    assert [t["id"] for t in visitor.get(THREADS).json()] == [tid]
    assert run(visitor, tid).status_code == 200


def test_adoption_can_be_turned_off(anonymous_mode, user, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "ADOPT_SESSION_THREADS": False}
    visitor = Client()
    tid = new_thread(visitor)
    visitor.force_login(user)
    assert Thread.objects.get(pk=tid).user_id is None
    assert run(visitor, tid).status_code == 403  # signed in now, and it isn't theirs


def test_ownerless_threads_are_for_code_and_staff_only(anonymous_mode, user, other_user):
    thread = Thread.objects.create(graph="chat")
    anon = Client()
    assert run(anon, thread.pk).status_code == 403  # used to be open to anyone with the id
    member = Client()
    member.force_login(user)
    assert run(member, thread.pk).status_code == 403
    other_user.is_staff = True
    other_user.save()
    staff = Client()
    staff.force_login(other_user)
    assert staff.get(f"{THREADS}{thread.pk}/").status_code == 200


def test_staff_cannot_open_another_users_thread(user, other_user):
    thread = Thread.objects.create(graph="chat", user=user)
    other_user.is_staff = True
    other_user.save()
    staff = Client()
    staff.force_login(other_user)
    assert staff.get(f"{THREADS}{thread.pk}/").status_code == 403


async def test_sse_visitor_threads_belong_to_the_session(anonymous_mode, async_client):
    import json

    r = await async_client.post("/llm/stream/", json.dumps({"message": "anon"}), content_type="application/json")
    first = [chunk async for chunk in r.streaming_content][0].decode()
    tid = json.loads(first.split("data: ", 1)[1])["id"]
    thread = await Thread.objects.aget(pk=tid)
    assert thread.user_id is None and thread.session_key
    again = await async_client.post(
        f"/llm/threads/{tid}/stream/", json.dumps({"message": "x"}), content_type="application/json"
    )
    assert again.status_code == 200
    from django.test import AsyncClient

    other = await AsyncClient().post(
        f"/llm/threads/{tid}/stream/", json.dumps({"message": "x"}), content_type="application/json"
    )
    assert other.status_code == 403


# ------------------------------------------------------------------- tenants


def test_threads_are_stamped_and_checked_by_tenant(world):
    alice = world["alice"]
    client = Client()
    client.force_login(alice)
    tid = new_thread(client)
    assert Thread.objects.get(pk=tid).tenant == str(world["acme"].pk)
    # The same user acting for another tenant can't use it, and doesn't list it.
    elsewhere = access.Owner(user=alice, tenant=world["globex"])
    with pytest.raises(access.NotAllowed):
        access.get_thread(tid, elsewhere)
    assert not access.threads_for(elsewhere).exists()
    assert access.threads_for(access.owner_for(alice)).count() == 1


# --------------------------------------------------------------- keyed threads


def test_open_thread_gets_or_creates_by_key_per_tenant(world):
    a = open_thread("chat", key="whatsapp:2547", tenant=world["acme"])
    assert open_thread("chat", key="whatsapp:2547", tenant=world["acme"]).pk == a.pk
    b = open_thread("chat", key="whatsapp:2547", tenant=world["globex"].pk)
    assert b.pk != a.pk and (a.tenant, b.tenant) == (str(world["acme"].pk), str(world["globex"].pk))
    with pytest.raises(ValueError, match="runs graph 'chat'"):
        open_thread("tools", key="whatsapp:2547", tenant=world["acme"])
    with pytest.raises(ValueError, match="non-empty key"):
        open_thread("chat", key="")


def test_keyed_threads_are_not_reachable_by_clients(anonymous_mode):
    thread = open_thread("chat", key="ticket:9")
    assert run(Client(), thread.pk).status_code == 403
    assert Client().get(THREADS).json() == []


def test_agents_run_continues_a_keyed_thread(world):
    SCRIPTS.clear()
    SCRIPTS["ollama/script-assistant"] = ["first", "second"]
    thread = open_thread("tests.assistant", key="whatsapp:1", user=world["alice"], tenant=world["acme"])
    agents.run("tests.assistant", "one", thread=thread)  # user and tenant come from the thread
    result = agents.run("tests.assistant", "two", thread=thread)
    assert result.text == "second" and result.thread_id == thread.thread_id
    from llm_kit.runtime.graphs import get_graph

    state = get_graph("tests.assistant").get_state(thread.config)
    assert [m.content for m in state.values["messages"] if m.type == "human"] == ["one", "two"]  # one conversation
    with pytest.raises(ValueError, match="runs 'tests.assistant'"):
        agents.run("tests.helper", "x", thread=thread)
    SCRIPTS.clear()


def test_deleting_a_user_deletes_adopted_threads(anonymous_mode):
    visitor = Client()
    tid = new_thread(visitor)
    user = get_user_model().objects.create_user("dana")
    visitor.force_login(user)
    user.delete()
    assert not Thread.objects.filter(pk=tid).exists()
