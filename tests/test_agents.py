"""Agents: one harness for every agent, delegation to subagents, limits, and the run tree."""

import json

import pytest
from langchain_core.messages import ToolMessage

from llm_kit import agents
from llm_kit.core.agents import spec as agent_spec
from llm_kit.core.errors import ConfigError
from llm_kit.runtime.graphs import get_graph, graph_names
from tests.graphs import SCRIPTS, SEEN, call

pytestmark = pytest.mark.django_db(transaction=True)

A, R, H = "ollama/script-assistant", "ollama/script-researcher", "ollama/script-helper"


@pytest.fixture(autouse=True)
def scripts():
    SCRIPTS.clear()
    SEEN.clear()
    yield SCRIPTS
    SCRIPTS.clear()
    SEEN.clear()


@pytest.fixture
def temp_agents():
    """Declare agents for one test; they are removed afterwards."""
    names = []

    def declare(name, **options):
        names.append(name)
        options.setdefault("description", f"{name} for a test")
        options.setdefault("touchpoint", "tests.helper")
        return agents.agent(name, **options)

    yield declare
    for name in names:
        agent_spec._unregister(name)


def seen(ref):
    return [s for s in SEEN if s["ref"] == ref]


def tool_replies(result):
    return [m.content for m in result.messages if isinstance(m, ToolMessage)]


# ------------------------------------------------------------- declarations


@pytest.mark.parametrize(
    "name, options, error",
    [
        ("Bad Name", {"description": "x", "touchpoint": "t"}, "lowercase dotted"),
        ("t.a", {"description": " ", "touchpoint": "t"}, "give a description"),
        ("t.a", {"description": "x"}, "give a touchpoint"),
        ("t.a", {"description": "x", "touchpoint": "t", "subagents": ["t.a"]}, "lists itself"),
        ("t.a", {"description": "x", "touchpoint": "t", "tools": "find_invoices"}, "tools must be a list"),
        ("t.a", {"description": "x", "touchpoint": "t", "tools": [42]}, "not a tool"),
        ("t.a", {"description": "x", "touchpoint": "t", "max_steps": 0}, "at least 1"),
        ("t.a", {"description": "x", "touchpoint": "t", "essential": ["x"]}, "defer_tools"),
        ("t.a", {"description": "x", "touchpoint": "t", "output": dict}, "pydantic"),
    ],
)
def test_bad_declarations_are_refused(name, options, error):
    with pytest.raises(ConfigError, match=error):
        agent_spec.AgentSpec(name=name, **options)


def test_code_agents_take_no_model_settings():
    with pytest.raises(ConfigError, match="no touchpoint"):
        agent_spec.AgentSpec(name="t.code", description="x", touchpoint="t", code=lambda task, ctx: "")


def test_startup_check_reports_what_does_not_fit(temp_agents, settings):
    from llm_kit.agents.registry import check

    assert check() == []
    temp_agents("t.lost", subagents=["t.nobody"], touchpoint="t.unknown", tools=["no_such_tool"])
    temp_agents("t.ping", subagents=["t.pong"])
    temp_agents("t.pong", subagents=["t.ping"])
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "GRAPHS": {**settings.LLM_KIT["GRAPHS"], "t.ping": "tests.graphs:chat_builder"},
    }
    found = "\n".join(check())
    assert "subagent 't.nobody' is not a declared agent" in found
    assert "touchpoint 't.unknown' is not declared" in found
    assert "tool 'no_such_tool' is not in the tool catalogue" in found
    assert "delegate in a circle: t.ping -> t.pong -> t.ping" in found
    assert "t.ping: an agent and a GRAPHS entry share this name" in found


def test_every_agent_is_a_graph(client, user):
    assert {"tests.assistant", "tests.researcher", "tests.newsroom"} <= set(graph_names())
    assert get_graph("tests.assistant").name == "tests.assistant"
    client.force_login(user)
    listed = {g["name"]: g["touchpoint"] for g in client.get("/llm/api/graphs/").json()}
    assert listed["tests.assistant"] == "tests.assistant"
    assert listed["tests.newsroom"] is None


# ------------------------------------------------------------- the harness


def test_prompt_and_tools_come_from_the_declaration_and_the_caller(world, scripts):
    scripts[A] = ["hello"]
    result = agents.run("tests.assistant", "hi", user=world["alice"])
    assert (result.text, result.steps, result.stopped) == ("hello", 1, "done")
    step = seen(A)[0]
    assert f"You help Alice of tenant {world['acme'].pk}" in step["system"]
    assert 'Keep {braces} and {"json": 1} as written.' in step["system"]
    assert "`whoami` says who you are." in step["system"]  # tool guidance
    assert "`tests.researcher`: Looks invoices up" in step["system"]  # the agents it can hand work to
    assert {"find_invoices", "save_invoice", "find_customers", "whoami", "delegate"} <= set(step["tools"])
    assert "find_accounts" not in step["tools"]  # category "auth", not selected


def test_a_caller_without_permissions_gets_fewer_tools(world, scripts):
    agents.run("tests.assistant", "hi", user=world["carol"])
    assert set(seen(A)[0]["tools"]) == {"whoami", "delegate"}


def test_tools_act_for_the_caller(world, scripts):
    scripts[A] = [call("find_invoices"), "two invoices"]
    result = agents.run("tests.assistant", "list invoices", user=world["alice"])
    (reply,) = tool_replies(result)
    assert "A-1" in reply and "A-2" in reply and "G-1" not in reply
    assert result.steps == 2


def test_a_tool_outside_the_agents_set_is_refused(world, scripts):
    from tests.testapp.models import Invoice

    invoice = Invoice.objects.get(number="A-1")
    scripts[R] = [call("save_invoice", {"id": invoice.pk, "total": 1}), "could not", call("Report", {"answer": "no"})]
    result = agents.run("tests.researcher", "change A-1", user=world["alice"])
    assert tool_replies(result) == ["Refused: save_invoice is not one of tests.researcher's tools."]
    invoice.refresh_from_db()
    assert invoice.total == 10


def test_the_last_step_asks_for_an_answer_then_stops(world, scripts):
    scripts[A] = [call("whoami") for _ in range(4)]
    result = agents.run("tests.assistant", "loop", user=world["alice"])
    assert (result.stopped, result.steps) == ("max_steps", 4)
    assert "This is your last step" in seen(A)[-1]["system"]
    assert "This is your last step" not in seen(A)[-2]["system"]
    assert tool_replies(result)[-1] == "Not run: the step limit was reached."
    assert "reached the limit of 4 steps" in result.text
    # The conversation is still valid: the next turn runs.
    scripts[A] = ["fine"]
    assert agents.run("tests.assistant", "again", user=world["alice"], thread_id=result.thread_id).text == "fine"


def test_structured_output(world, scripts):
    scripts[R] = [
        call("find_invoices", {"query": "A-1"}),
        "A-1 is open",
        call("Report", {"answer": "open", "ids": [1]}),
    ]
    result = agents.run("tests.researcher", "is A-1 open?", user=world["alice"])
    assert result.output.answer == "open" and result.output.ids == [1]
    assert result.for_model() == '{"answer":"open","ids":[1]}'


# ------------------------------------------------------------- delegation


def test_delegating_to_a_subagent(world, scripts):
    scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "Is A-1 paid?"}, id="d1"), "A-1 is open"]
    scripts[R] = [call("find_invoices", {"query": "A-1"}), "open", call("Report", {"answer": "open", "ids": [7]})]
    result = agents.run("tests.assistant", "is A-1 paid?", user=world["alice"])

    assert result.text == "A-1 is open"
    assert tool_replies(result) == ['{"answer":"open","ids":[7]}']
    first = seen(R)[0]
    assert [m.type for m in first["messages"]] == ["system", "human"]  # fresh: only the task
    assert first["messages"][-1].content == "Is A-1 paid?"
    assert "save_invoice" not in first["tools"] and "find_invoices" in first["tools"]
    assert "delegate" in first["tools"]  # depth 1 may still hand off (to tests.helper)
    assert "A-1" in seen(R)[1]["messages"][-1].content  # the find ran for alice
    # The subagent's own thread is checkpointed.
    child = get_graph("tests.researcher").get_state(
        {"configurable": {"thread_id": f"{result.thread_id}/tests.researcher/d1"}}
    )
    assert child.values["output"] == {"answer": "open", "ids": [7]}


def test_fork_starts_from_the_parents_conversation(world, scripts, temp_agents):
    temp_agents("t.parent", touchpoint="tests.assistant", subagents=["t.forked"])
    temp_agents("t.forked", fork=True)
    scripts[A] = [call("delegate", {"agent": "t.forked", "task": "continue"}), "ok"]
    agents.run("t.parent", "remember the word kiwi", user=world["alice"])
    messages = seen(H)[0]["messages"]
    assert [m.type for m in messages] == ["system", "human", "human"]
    assert messages[1].content == "remember the word kiwi" and messages[2].content == "continue"


def test_subagents_are_not_offered_past_the_depth_limit(world, scripts, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "AGENT_MAX_DEPTH": 1}
    scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "x"}), "ok"]
    scripts[R] = ["nothing", call("Report", {"answer": "-"})]
    agents.run("tests.assistant", "x", user=world["alice"])
    assert "delegate" in seen(A)[0]["tools"]
    assert "delegate" not in seen(R)[0]["tools"]


def test_no_handing_back_up_the_chain(world, scripts, temp_agents):
    temp_agents("t.ping", touchpoint="tests.assistant", subagents=["t.pong"])
    temp_agents("t.pong", subagents=["t.ping"])
    scripts[A] = [call("delegate", {"agent": "t.pong", "task": "x"}), "ok"]
    scripts[H] = [call("delegate", {"agent": "t.ping", "task": "back to you"}), "fine, I did it"]
    result = agents.run("t.ping", "x", user=world["alice"])
    assert "delegate" not in seen(H)[0]["tools"]  # t.ping is above it
    assert tool_replies(result) == ["fine, I did it"]
    pong = seen(H)[1]["messages"]
    assert pong[-1].content == "Refused: delegate is not one of t.pong's tools."


def test_hand_offs_per_request_are_limited(world, scripts, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "AGENT_MAX_DELEGATIONS": 1}
    task = {"agent": "tests.researcher", "task": "x"}
    scripts[A] = [call("delegate", task), call("delegate", task), "done"]
    scripts[R] = ["r", call("Report", {"answer": "r"})]
    result = agents.run("tests.assistant", "x", user=world["alice"])
    assert tool_replies(result) == [
        '{"answer":"r","ids":[]}',
        "Refused: This request already handed off 1 tasks; finish it yourself.",
    ]


def test_a_failing_subagent_becomes_a_message(world, scripts, temp_agents):
    temp_agents("t.parent", touchpoint="tests.assistant", subagents=["t.broken"])

    @agents.agent.code("t.broken", description="always fails")
    def broken(task, ctx):
        raise RuntimeError("boom")

    try:
        scripts[A] = [call("delegate", {"agent": "t.broken", "task": "x"}), "sorry"]
        result = agents.run("t.parent", "x", user=world["alice"])
    finally:
        agent_spec._unregister("t.broken")
    assert tool_replies(result) == ["t.broken failed before finishing; do the task another way or tell the user."]
    assert result.text == "sorry"


def test_code_agents_run_subagents(world, scripts):
    scripts[R] = ["found", call("Report", {"answer": "all paid"})]
    result = agents.run("tests.newsroom", "any unpaid invoices?", user=world["alice"])
    assert result.output["headline"] == "ALL PAID"
    assert result.output["user"] == "alice"
    assert seen(R)[0]["messages"][-1].content == "any unpaid invoices?"


def test_a_model_agent_can_delegate_to_a_code_agent(world, scripts):
    scripts[A] = [call("delegate", {"agent": "tests.newsroom", "task": "headline please"}), "done"]
    scripts[R] = ["found", call("Report", {"answer": "quiet day"})]
    result = agents.run("tests.assistant", "x", user=world["alice"])
    assert json.loads(tool_replies(result)[0])["headline"] == "QUIET DAY"


def test_clients_cannot_set_the_run_tree(world):
    from llm_kit.models import Thread
    from llm_kit.runtime.runs import build_config

    thread = Thread.objects.create(graph="tests.assistant", user=world["alice"])
    evil = {"config": {"configurable": {"__llm_kit_agents": ["x"], "__llm_kit_tree": "x"}}}
    config = build_config(thread, evil, world["alice"])
    assert config["configurable"]["__llm_kit_agents"] == ("tests.assistant",)
    assert config["configurable"]["__llm_kit_tree"].max_delegations == 8


# ------------------------------------------------------------- streaming and recording


async def test_subagent_output_streams_labelled(async_client, world, scripts):
    scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "look"}), "all good"]
    scripts[R] = ["looked", call("Report", {"answer": "fine"})]
    await async_client.aforce_login(world["alice"])
    body = {"graph": "tests.assistant", "message": "check"}
    r = await async_client.post("/llm/stream/", json.dumps(body), content_type="application/json")
    text = "".join([c.decode() async for c in r.streaming_content])
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
    labelled = {(e.get("agent"), e.get("depth")) for e in events if isinstance(e, dict) and "agent" in e}
    assert ("tests.assistant", 0) in labelled
    assert ("tests.researcher", 1) in labelled
    assert "event: end" in text


def test_a_request_is_a_tree_of_runs(world, scripts, settings):
    from django.db.models import Count

    from llm_kit import monitoring
    from llm_kit.core.observability import flush, recorder
    from llm_kit.models import LLMCall, LLMRun
    from llm_kit.observability.setup import configure

    flush()  # records still queued from earlier tests must not land in this one
    settings.LLM_KIT = {**settings.LLM_KIT, "RECORD_SINKS": ["database"]}
    configure()
    try:
        scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "look"}), "all good"]
        scripts[R] = [call("find_invoices"), "looked", call("Report", {"answer": "fine"})]
        result = agents.run("tests.assistant", "check", user=world["alice"])
        assert flush()
    finally:
        recorder.configure([], None)

    root = LLMRun.objects.get(thread_id=result.thread_id)
    sub = LLMRun.objects.get(parent_run_id=root.run_id)
    assert (root.agent, root.depth, root.parent_run_id) == ("tests.assistant", 0, None)
    call_id = result.messages[0].tool_calls[0]["id"]
    assert (sub.agent, sub.depth, sub.thread_id) == (
        "tests.researcher",
        1,
        f"{result.thread_id}/tests.researcher/{call_id}",
    )
    assert (root.calls, sub.calls) == (5, 3)  # the request's run includes its subagent's calls
    assert (root.tool_calls, sub.tool_calls) == (2, 1)
    by_agent = dict(LLMCall.objects.filter(run_id=root.run_id).values_list("agent").annotate(n=Count("id")))
    assert by_agent == {"tests.assistant": 2, "tests.researcher": 3}

    assert monitoring.summary(period="1h")["runs"] == 1  # requests, not their subagent runs
    assert monitoring.summary(period="1h", agent="tests.researcher")["runs"] == 1
    rows = {r["key"]: r["calls"] for r in monitoring.breakdown(by="agent", period="1h")["rows"]}
    assert rows == {"tests.assistant": 2, "tests.researcher": 3}


async def test_arun(world, scripts):
    scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "x"}), "async done"]
    scripts[R] = ["r", call("Report", {"answer": "r"})]
    result = await agents.arun("tests.assistant", "x", user=world["alice"])
    assert result.text == "async done"
    assert tool_replies(result) == ['{"answer":"r","ids":[]}']


# ------------------------------------------------------------- admin overrides


def override(agent, **fields):
    from llm_kit.models import AgentOverride

    return AgentOverride.objects.create(agent=agent, **fields)


def test_an_override_appends_to_or_replaces_the_prompt(world, scripts):
    row = override("tests.assistant", prompt="Answer in French for {user}.")
    agents.run("tests.assistant", "hi", user=world["alice"])
    system = seen(A)[-1]["system"]
    assert "You help Alice" in system and "Answer in French for Alice." in system

    row.prompt_mode = "replace"
    row.save()
    agents.run("tests.assistant", "hi", user=world["alice"])
    system = seen(A)[-1]["system"]
    assert "You help Alice" not in system and system.startswith("Answer in French for Alice.")


def test_an_override_only_narrows(world, scripts):
    override("tests.assistant", max_steps=50, tools=["find_invoices", "find_accounts"])
    scripts[A] = [call("save_invoice", {"id": 1, "total": 0}), "ok"]
    result = agents.run("tests.assistant", "x", user=world["alice"])
    assert set(seen(A)[0]["tools"]) == {"find_invoices", "delegate"}  # find_accounts was never its tool
    assert tool_replies(result) == ["Refused: save_invoice is not one of tests.assistant's tools."]

    from llm_kit.models import AgentOverride

    AgentOverride.objects.filter(agent="tests.assistant").update(max_steps=1)
    from llm_kit.agents.overrides import invalidate

    invalidate()  # .update() sends no signal
    agents.run("tests.assistant", "x", user=world["alice"])
    assert "This is your last step" in seen(A)[-1]["system"]


def test_a_paused_agent_answers_without_a_model(world, scripts):
    override("tests.assistant", paused="being retrained")
    result = agents.run("tests.assistant", "hi", user=world["alice"])
    assert (result.text, result.stopped, result.steps) == ("tests.assistant is paused: being retrained", "paused", 0)
    assert seen(A) == []


def test_a_paused_subagent_is_not_offered(world, scripts):
    override("tests.researcher", paused="maintenance")
    scripts[A] = [call("delegate", {"agent": "tests.researcher", "task": "x"}), "ok"]
    result = agents.run("tests.assistant", "x", user=world["alice"])
    assert "tests.researcher" not in seen(A)[0]["system"]
    (reply,) = tool_replies(result)  # not in the delegate tool's choices any more
    assert reply.startswith("Error: delegate failed") and "Input should be 'tests.newsroom'" in reply


def test_admin_form_checks_overrides(admin_client):
    data = {"agent": "tests.assistant", "prompt_mode": "append", "tools": '["find_invoices", "nope"]'}
    r = admin_client.post("/admin/llm_kit/agentoverride/add/", data)
    assert "Not among this agent&#x27;s tools: nope." in r.content.decode()
    r = admin_client.post(
        "/admin/llm_kit/agentoverride/add/", {"agent": "tests.newsroom", "prompt_mode": "append", "prompt": "x"}
    )
    assert "A code agent runs Python" in r.content.decode()
    r = admin_client.post("/admin/llm_kit/agentoverride/add/", {**data, "tools": '["find_invoices"]'})
    assert r.status_code == 302
    from llm_kit.models import AgentOverride

    pk = AgentOverride.objects.get(agent="tests.assistant").pk
    page = admin_client.get(f"/admin/llm_kit/agentoverride/{pk}/change/").content.decode()
    assert "tests.assistant" in page and "find_invoices" in page and "You help {user}" in page


# ------------------------------------------------------------- loading tools on demand


def test_deferred_tools_load_when_found(world, scripts, temp_agents):
    from llm_kit import tools

    temp_agents(
        "t.lazy",
        touchpoint="tests.assistant",
        tools=tools.select(categories=["testapp"]),
        defer_tools=True,
        essential=["whoami"],
    )
    scripts[A] = [call("find_tools", {"query": "overdue invoices"}), call("find_invoices"), "found them"]
    result = agents.run("t.lazy", "overdue?", user=world["alice"])
    first, second = seen(A)[0], seen(A)[1]
    assert set(first["tools"]) == {"whoami", "find_tools"}
    assert "more tools are available: call `find_tools`" in first["system"]
    assert "find_invoices" in tool_replies(result)[0]
    assert {"find_invoices", "save_invoice"} <= set(second["tools"])  # loaded for the next step
    assert "A-1" in tool_replies(result)[1]
    # Loaded tools stay loaded in the thread.
    agents.run("t.lazy", "again", user=world["alice"], thread_id=result.thread_id)
    assert "find_invoices" in seen(A)[-1]["tools"]


# ------------------------------------------------------------- command and admin


def test_agents_command(world):
    from io import StringIO

    from django.core.management import call_command

    override("tests.helper", paused="off for now")
    out = StringIO()
    call_command("llm_kit_agents", stdout=out)
    text = out.getvalue()
    assert "tests.assistant  [touchpoint tests.assistant]" in text
    assert "subagents: tests.researcher, tests.newsroom" in text
    assert "tests.newsroom  [code]" in text
    assert "paused: off for now" in text
    assert "problem" not in text

    out = StringIO()
    call_command("llm_kit_agents", "--user", "carol", stdout=out)
    assert "tools: whoami" in out.getvalue()


def test_run_tree_in_the_admin(admin_client):
    import uuid

    from django.utils import timezone

    from llm_kit.models import LLMRun

    root = LLMRun.objects.create(
        run_id=uuid.uuid4(), name="tests.assistant", agent="tests.assistant", started_at=timezone.now()
    )
    child = LLMRun.objects.create(
        run_id=uuid.uuid4(),
        name="tests.researcher",
        agent="tests.researcher",
        parent_run_id=root.run_id,
        depth=1,
        started_at=timezone.now(),
    )
    page = admin_client.get(f"/admin/llm_kit/llmrun/{child.pk}/change/").content.decode()
    assert "tests.assistant</a>" in page and "tests.researcher</a>" in page and "(this run)" in page
    listing = admin_client.get("/admin/llm_kit/llmrun/").content.decode()
    assert "↳ tests.researcher (depth 1)" in listing


# ----------------------------------------------------- public tools, history


def test_public_tools_reach_visitors_and_runs_without_a_user(temp_agents, scripts, world):
    from llm_kit.tools import registry as tool_registry

    temp_agents("t.front", tools=tools_select_public(), touchpoint="tests.helper")
    scripts[H] = ["ok", "ok", "ok"]
    agents.run("t.front", "hours?")  # a Celery-style run: no user
    assert seen(H)[-1]["tools"] == ["opening_hours"]
    from django.contrib.auth.models import AnonymousUser

    agents.run("t.front", "hours?", user=AnonymousUser())
    assert seen(H)[-1]["tools"] == ["opening_hours"]
    # Anyone gets public tools, and only those: carol (no permissions) gets whoami too.
    assert [e.name for e in tool_registry.entries_for(None)] == ["opening_hours"]
    assert {"opening_hours", "whoami"} <= {e.name for e in tool_registry.entries_for(world["carol"])}


def test_model_tools_cannot_be_public():
    from django.core.exceptions import ImproperlyConfigured

    from llm_kit.tools import registry as tool_registry
    from llm_kit.tools.caller import PUBLIC

    entry = tool_registry.ToolEntry("x", "x", "c", "find", (PUBLIC,), None)
    with pytest.raises(ImproperlyConfigured, match="can't be PUBLIC"):
        tool_registry._add_entry(entry)


def test_history_limits_what_the_model_sees_not_what_is_kept(temp_agents, scripts):
    temp_agents("t.short", history=60, touchpoint="tests.helper")
    scripts[H] = ["first answer " * 10, "second answer " * 10, "third"]
    thread = "short-1"
    agents.run("t.short", "first question " * 10, thread_id=thread)
    agents.run("t.short", "second question " * 10, thread_id=thread)
    agents.run("t.short", "third question", thread_id=thread)
    sent = [m.content for m in seen(H)[-1]["messages"] if m.type != "system"]
    assert not any("first question" in c for c in sent)  # trimmed away
    assert sent[-1] == "third question"
    state = get_graph("t.short").get_state({"configurable": {"thread_id": thread}})
    assert len(state.values["messages"]) == 6  # the checkpoint keeps everything


def tools_select_public():
    from llm_kit import tools

    return tools.select(categories=["public"])


def test_long_tool_results_are_cut_with_a_note(temp_agents, scripts):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    @tool
    def big_page() -> str:
        """A long page."""
        return "x" * 500

    temp_agents("t.reader", tools=[big_page], tool_result_chars=100)
    scripts[H] = [AIMessage(content="", tool_calls=[{"name": "big_page", "args": {}, "id": "b1"}]), "read it"]
    result = agents.run("t.reader", "read")
    (reply,) = tool_replies(result)
    assert reply.startswith("x" * 100) and "400 characters omitted" in reply
