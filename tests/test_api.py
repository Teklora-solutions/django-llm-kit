import pytest
from rest_framework.test import APIClient

from llm_kit.models import Checkpoint, LLMModel, Thread

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

    assert Checkpoint.objects.filter(thread_id=tid).exists()
    assert api.delete(f"/llm/api/threads/{tid}/").status_code == 204
    assert not Checkpoint.objects.filter(thread_id=tid).exists()  # state deleted with the thread


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
    r = api.post(f"/llm/api/threads/{tid}/runs/", {"message": "x", "model": "nope"}, format="json")
    assert r.status_code == 400
    assert api.get(f"/llm/api/threads/{tid}/history/?limit=abc").status_code == 400


def test_models_and_graphs(api):
    LLMModel.objects.create(name="smart", model="anthropic:claude-opus-5-5", description="Best")
    data = api.get("/llm/api/models/").data
    assert [(m["name"], m["source"]) for m in data] == [("fake", "settings"), ("smart", "db")]
    assert "api_key_env" not in data[0]
    assert {g["name"] for g in api.get("/llm/api/graphs/").data} == {"chat", "approval", "boom", "catalogue", "slow"}
