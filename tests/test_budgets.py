"""Spend budgets: per request (AGENT_BUDGET, agents.run(budget=)) and per day (DAILY_BUDGETS)."""

from datetime import timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from django.test import Client
from django.utils import timezone

from llm_kit import agents
from llm_kit.core.errors import BudgetExceeded, ConfigError
from llm_kit.core.observability import recorder
from tests.graphs import SCRIPTS, SEEN, call

pytestmark = pytest.mark.django_db(transaction=True)

A, R = "ollama/script-assistant", "ollama/script-researcher"


@pytest.fixture(autouse=True)
def scripts():
    SCRIPTS.clear()
    SEEN.clear()
    yield SCRIPTS
    SCRIPTS.clear()
    SEEN.clear()


@pytest.fixture
def priced(monkeypatch):
    """Every model costs $1 per 1,000 tokens."""
    monkeypatch.setattr(recorder, "price_lookup", lambda ref: {"price_input": "1000", "price_output": "1000"})


def test_a_request_stops_when_its_budget_is_spent(world, scripts, priced):
    scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "A-1?"}, id="d1"), "done"]
    scripts[R] = [call("Report", {"answer": "open"})]
    result = agents.run("tests.assistant", "check A-1", user=world["alice"], budget="0.0001")
    assert result.stopped == "budget"
    assert "budget of $0.0001" in result.text
    assert [s["ref"] for s in SEEN] == [A]  # neither the subagent nor a second step called a model
    assert result.messages[1].content == "Not run: the agent stopped."  # its pending hand-off too


def test_the_budget_setting_applies_and_a_generous_budget_changes_nothing(world, scripts, priced, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "AGENT_BUDGET": 100}
    scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "A-1?"}, id="d2"), "done"]
    scripts[R] = [call("Report", {"answer": "open"})]
    result = agents.run("tests.assistant", "check A-1", user=world["alice"])
    assert result.stopped == "done" and result.text == "done"

    settings.LLM_KIT = {**settings.LLM_KIT, "AGENT_BUDGET": "0.0001"}
    scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "A-1?"}, id="d3"), "done"]
    assert agents.run("tests.assistant", "again", user=world["alice"]).stopped == "budget"


def test_unpriced_models_do_not_use_up_a_budget(world, scripts, monkeypatch):
    monkeypatch.setattr(recorder, "price_lookup", lambda ref: None)
    scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "A-1?"}, id="d4"), "done"]
    scripts[R] = [call("Report", {"answer": "open"})]
    assert agents.run("tests.assistant", "x", user=world["alice"], budget="0.0001").stopped == "done"


def test_a_nested_run_shares_its_requests_budget(world, scripts):
    def orchestrate(task, ctx):
        try:
            agents.run("tests.researcher", task, budget=1)
        except ConfigError as e:
            return str(e)
        return "ran"

    agents.agent.code("t.budgeted", description="for a test")(orchestrate)
    try:
        assert "applies to a whole request" in agents.run("t.budgeted", "x", user=world["alice"]).text
    finally:
        from llm_kit.core.agents import spec

        spec._unregister("t.budgeted")


# ------------------------------------------------------------------ daily


def _spent(amount, **who):
    from llm_kit.models import LLMCall

    LLMCall.objects.create(call_id=uuid4(), started_at=timezone.now(), cost=Decimal(amount), **who)


def test_a_user_over_their_daily_budget_is_refused(world, scripts, settings):
    from llm_kit.models import LLMCall

    settings.LLM_KIT = {**settings.LLM_KIT, "DAILY_BUDGETS": {"user": 5}}
    alice = world["alice"]
    _spent("3", user=str(alice.pk))
    old = LLMCall.objects.create(call_id=uuid4(), started_at=timezone.now() - timedelta(days=2), cost=50)
    old.user = str(alice.pk)
    old.save()  # yesterday's spend doesn't count
    scripts[A] = ["fine"]
    assert agents.run("tests.assistant", "x", user=alice).text == "fine"

    _spent("2.5", user=str(alice.pk))
    with pytest.raises(BudgetExceeded, match=r"user budget of \$5 is used up \(\$5.50 spent\)"):
        agents.run("tests.assistant", "x", user=alice)
    agents.run("tests.assistant", "x", user=world["bob"])  # others are unaffected


def test_a_tenant_over_its_daily_budget_is_refused_by_the_views(world, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "DAILY_BUDGETS": {"tenant": 1}}
    _spent("1", tenant=str(world["acme"].pk))
    client = Client()
    client.force_login(world["carol"])  # in acme too
    r = client.post("/llm/api/threads/", {"graph": "chat"}, content_type="application/json")
    tid = r.json()["id"]
    r = client.post(f"/llm/api/threads/{tid}/runs/", {"message": "hi"}, content_type="application/json")
    assert r.status_code == 429 and "tenant budget" in r.json()["detail"]
    r = client.post("/llm/stream/", {"message": "hi"}, content_type="application/json")
    assert r.status_code == 429


def test_unknown_daily_budget_keys_are_refused(world, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "DAILY_BUDGETS": {"users": 5}}
    with pytest.raises(ConfigError, match="unknown key"):
        agents.run("tests.assistant", "x", user=world["alice"])
