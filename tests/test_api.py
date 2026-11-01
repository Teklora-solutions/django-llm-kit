import pytest
from rest_framework.test import APIClient

from llm_kit.models import Thread
from llm_kit.runtime.graphs import get_checkpointer

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def api(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


def test_thread_crud_run_state_history(api, user):
    r = api.post("/llm/api/threads/", {"title": "First"}, format="json")
    assert r.status_code == 201, r.data
    tid = r.data["id"]
    assert r.data["graph"] == "chat"

    r = api.post(f"/llm/api/threads/{tid}/runs/", {"message": "hi"}, format="json")
    assert r.status_code == 200, r.data
    assert [m["content"] for m in r.data["values"]["messages"]] == ["hi", "echo: hi"]
    assert r.data["next"] == []

    state = api.get(f"/llm/api/threads/{tid}/state/").data
    assert state["checkpoint_id"] == r.data["checkpoint_id"]
    history = api.get(f"/llm/api/threads/{tid}/history/?limit=2").data
    assert len(history) == 2 and history[0]["checkpoint_id"] == state["checkpoint_id"]

    assert [t["id"] for t in api.get("/llm/api/threads/").data] == [tid]
    assert api.patch(f"/llm/api/threads/{tid}/", {"title": "Renamed"}, format="json").data["title"] == "Renamed"
    assert api.patch(f"/llm/api/threads/{tid}/", {"graph": "approval"}, format="json").status_code == 400

    saved = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    assert get_checkpointer().get_tuple(saved) is not None  # whichever checkpointer is configured
    assert api.delete(f"/llm/api/threads/{tid}/").status_code == 204
    assert get_checkpointer().get_tuple(saved) is None  # state deleted with the thread


def test_interrupt_resume_via_api(api):
    tid = api.post("/llm/api/threads/", {"graph": "approval"}, format="json").data["id"]
    r = api.post(f"/llm/api/threads/{tid}/runs/", {"input": {"request": "pay", "log": []}}, format="json")
    assert r.data["next"] == ["ask"]
    assert r.data["interrupts"][0]["value"] == {"question": "Approve pay?"}
    r = api.post(f"/llm/api/threads/{tid}/runs/", {"resume": False}, format="json")
    assert r.data["values"]["log"] == ["approved=False"] and r.data["next"] == []


def test_isolation_between_users(api, other_user):
    theirs = Thread.objects.create(graph="chat", user=other_user)
    assert api.get("/llm/api/threads/").data == []
    assert api.get(f"/llm/api/threads/{theirs.id}/").status_code == 403
    assert api.post(f"/llm/api/threads/{theirs.id}/runs/", {"message": "x"}, format="json").status_code == 403
    assert APIClient().get("/llm/api/threads/").status_code == 403


def test_validation(api):
    assert api.post("/llm/api/threads/", {"graph": "nope"}, format="json").status_code == 400
    tid = api.post("/llm/api/threads/", {}, format="json").data["id"]
    r = api.post(
        f"/llm/api/threads/{tid}/runs/", {"message": "x", "model": "anthropic/claude-sonnet-5-5"}, format="json"
    )
    assert r.status_code == 400  # the default "chat" graph declares no touchpoint
    assert api.get(f"/llm/api/threads/{tid}/history/?limit=abc").status_code == 400


def test_touchpoints_graphs_and_model_picker(api, keys):
    from llm_kit import catalogue
    from tests.catalogue_payloads import FakeSources

    catalogue.sync(fetch=FakeSources())
    tps = {t["name"]: t for t in api.get("/llm/api/touchpoints/").data}
    assert tps["tests.chat"]["choice"] == "client" and tps["tests.chat"]["default"] == "anthropic/claude-sonnet-5-5"

    data = api.get("/llm/api/touchpoints/tests.chat/models/").data
    assert data["default"] == "anthropic/claude-sonnet-5-5"
    # allow=anthropic/* or gemini/*, requires tool_call: claude-old-1 has no tool calling
    assert [m["ref"] for m in data["models"]] == ["anthropic/claude-sonnet-5-5", "gemini/gemini-3.8-flash"]
    assert data["models"][0]["price_input"] == "2.00000000"
    assert api.get("/llm/api/touchpoints/nope/models/").status_code == 404

    graphs = {g["name"]: g["touchpoint"] for g in api.get("/llm/api/graphs/").data}
    assert graphs["catalogue"] == "tests.chat" and graphs["chat"] is None
