"""Run hooks: server-set state, configurable and context managers, wherever a graph runs."""

import contextlib
import json

import pytest
from django.test import Client
from langchain_core.messages import AIMessage
from langgraph.types import Command

from llm_kit import agents, background
from llm_kit.core.hooks import HookFailed, RunRefused, prepare
from llm_kit.models import BackgroundRun, Thread
from tests.graphs import CHANNEL, HOOKED_CALLS, SCRIPTS
from tests.test_background import finish
from tests.test_views import post, read_events

pytestmark = pytest.mark.django_db(transaction=True)

LOG: list = []  # what the hooks and their context managers did, in order
POLICY = {"audience": "staff"}  # what the hook reads when it runs (a permission, say)


@contextlib.contextmanager
def channel(name):
    token = CHANNEL.set(name)
    LOG.append(f"enter {name}")
    try:
        yield
    finally:
        CHANNEL.reset(token)
        LOG.append(f"exit {name}")


@contextlib.asynccontextmanager
async def telemetry():
    LOG.append("telemetry start")
    try:
        yield
    finally:
        LOG.append("telemetry row")


def graph_hook(run):
    """The "hooked" graph's own hook (GRAPHS["hooked"]["prepare"])."""
    LOG.append(f"graph hook background={run.background} user={getattr(run.owner.user, 'username', None)}")
    run.set_state(audience=POLICY["audience"])
    run.set_configurable(__test_scope="kb")
    return [channel("web"), telemetry()]


def global_hook(run):
    LOG.append(f"global hook {run.graph}")


def refusing_hook(run):
    raise RunRefused("Not during maintenance.")


def broken_hook(run):
    raise KeyError("oops")


@contextlib.contextmanager
def failing_enter():
    raise RuntimeError("cannot open the channel")
    yield


def failing_cm_hook(run):
    return failing_enter()


def noop(run_id):
    pass


@pytest.fixture(autouse=True)
def clean():
    LOG.clear()
    HOOKED_CALLS.clear()
    POLICY["audience"] = "staff"
    yield
    LOG.clear()
    HOOKED_CALLS.clear()


def _hooks(settings, *hooks):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_HOOKS": [f"tests.test_run_hooks:{h}" for h in hooks]}


def _client_input():
    # A client trying to set what the server decides.
    return {
        "input": {"messages": [{"role": "user", "content": "hi"}], "audience": "admin"},
        "config": {"configurable": {"__test_scope": "everything"}},
    }


async def test_sse_runs_get_server_state_config_and_contexts(async_client, user, settings):
    _hooks(settings, "global_hook")
    await async_client.aforce_login(user)
    events = await read_events(await post(async_client, "/llm/stream/", {"graph": "hooked", **_client_input()}))
    updates = [d for e, d in events if e == "update"]
    assert updates[0]["answer"]["messages"][0]["content"] == "staff|web|kb"
    assert LOG == [
        "global hook hooked",
        "graph hook background=False user=alice",
        "enter web",
        "telemetry start",
        "telemetry row",
        "exit web",
    ]


def test_rest_runs_enter_async_context_managers_too(user):
    client = Client()
    client.force_login(user)
    thread = Thread.objects.create(graph="hooked", user=user)
    r = client.post(f"/llm/api/threads/{thread.id}/runs/", _client_input(), "application/json")
    assert r.status_code == 200
    assert r.json()["values"]["messages"][-1]["content"] == "staff|web|kb"
    assert LOG[-2:] == ["telemetry row", "exit web"]


async def test_websocket_runs(user):
    from tests.test_consumers import collect, communicator

    thread = await Thread.objects.acreate(graph="hooked", user=user)
    c = communicator(f"/ws/llm/threads/{thread.id}/", user)
    assert (await c.connect())[0]
    await c.send_json_to({"type": "run", **_client_input()})
    events = await collect(c)
    update = next(e for e in events if e["event"] == "update")
    assert update["data"]["answer"]["messages"][0]["content"] == "staff|web|kb"
    await c.disconnect()


def test_background_runs_hook_in_the_worker_with_policy_as_it_is_then(user, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_run_hooks:noop"}
    thread = Thread.objects.create(graph="hooked", user=user)
    run = background.start(thread, _client_input(), user=user)
    assert LOG == []  # nothing ran when it was started
    POLICY["audience"] = "customer"  # changed between start and execution
    background.execute(run.run_id)
    run.refresh_from_db()
    assert run.status == "done" and HOOKED_CALLS == ["customer"]
    assert "graph hook background=True user=alice" in LOG and LOG[-1] == "exit web"


def test_a_refusing_hook_is_a_403_and_never_runs(user, settings):
    _hooks(settings, "refusing_hook")
    client = Client()
    client.force_login(user)
    thread = Thread.objects.create(graph="hooked", user=user)
    r = client.post(f"/llm/api/threads/{thread.id}/runs/", {"message": "hi"}, "application/json")
    assert r.status_code == 403 and "maintenance" in r.json()["detail"]
    assert HOOKED_CALLS == []


async def test_a_broken_hook_is_an_error_event_and_never_runs(async_client, user, settings):
    _hooks(settings, "broken_hook")
    await async_client.aforce_login(user)
    events = await read_events(await post(async_client, "/llm/stream/", {"graph": "hooked", "message": "hi"}))
    assert [e for e, _ in events] == ["thread", "error"] and HOOKED_CALLS == []


def test_a_broken_hook_fails_a_background_run_cleanly(user, settings):
    _hooks(settings, "broken_hook")
    thread = Thread.objects.create(graph="hooked", user=user)
    run = finish(background.start(thread, {"message": "hi"}, user=user))
    assert run.status == "error" and HOOKED_CALLS == []
    _hooks(settings, "refusing_hook")
    run = finish(background.start(thread, {"message": "hi"}, user=user))
    assert run.status == "error" and run.error == "Not during maintenance."


async def test_a_context_manager_that_fails_to_enter_stops_the_run(async_client, user, settings):
    _hooks(settings, "failing_cm_hook")
    await async_client.aforce_login(user)
    events = await read_events(await post(async_client, "/llm/stream/", {"graph": "hooked", "message": "hi"}))
    assert events[-1][0] == "error" and HOOKED_CALLS == []
    assert await BackgroundRun.objects.acount() == 0


def test_agents_run_from_code_gets_the_hooks(world, settings):
    _hooks(settings, "global_hook", "agent_hook")
    SCRIPTS["ollama/script-assistant"] = [lambda messages: AIMessage(content=f"channel={CHANNEL.get()}")]
    try:
        result = agents.run("tests.assistant", "hi", user=world["alice"])
    finally:
        SCRIPTS.clear()
    assert result.text == "channel=code" and LOG == ["global hook tests.assistant", "enter code", "exit code"]


def agent_hook(run):
    assert run.payload["message"] == "hi" and run.owner.user.username == "alice"
    return channel("code")


async def test_agents_arun_and_stream_get_the_hooks(world, settings):
    _hooks(settings, "agent_hook")
    SCRIPTS["ollama/script-assistant"] = [
        lambda messages: AIMessage(content=f"channel={CHANNEL.get()}"),
        lambda messages: AIMessage(content=f"channel={CHANNEL.get()}"),
    ]
    try:
        assert (await agents.arun("tests.assistant", "hi", user=world["alice"])).text == "channel=code"
        events = [e async for e in agents.stream("tests.assistant", "hi", user=world["alice"])]
    finally:
        SCRIPTS.clear()
    assert events[-1]["event"] == "end" and LOG.count("exit code") == 2


def test_state_is_merged_over_the_input_and_into_a_resume():
    def hook(run):
        run.set_state(audience="staff")

    setup = prepare("g", {"messages": [], "audience": "admin"}, {}, graph_hook=hook)
    assert setup.input == {"messages": [], "audience": "staff"}
    resumed = prepare("g", Command(resume="yes"), {}, graph_hook=hook).input
    assert resumed.resume == "yes" and resumed.update == {"audience": "staff"}
    with pytest.raises(HookFailed, match="mappingproxy"):  # the payload is read-only
        prepare("g", {}, {}, payload={"a": 1}, graph_hook=lambda run: run.payload.__setitem__("a", 2))
    with pytest.raises(HookFailed, match="not a context manager"):
        prepare("g", {}, {}, graph_hook=lambda run: [object()])


def test_payload_seen_by_hooks_is_a_copy():
    seen = {}

    def hook(run):
        seen["payload"] = run.payload

    payload = {"message": "hi", "config": {"configurable": {"k": 1}}}
    prepare("g", {}, {}, payload=payload, graph_hook=hook)
    payload["config"]["configurable"]["k"] = 2
    assert seen["payload"]["config"]["configurable"]["k"] == 1
    assert json.dumps(dict(seen["payload"]))


# ------------------------------------------------------------------ server-set context

SEEN_CONTEXT: list = []


def context_hook(run):
    SEEN_CONTEXT.append((run.background, dict(run.context)))
    run.set_state(audience=run.context.get("audience", "nobody"))


@pytest.fixture
def seen_context():
    SEEN_CONTEXT.clear()
    yield SEEN_CONTEXT
    SEEN_CONTEXT.clear()


async def test_a_client_can_never_set_the_context(async_client, user, settings, seen_context):
    _hooks(settings, "context_hook")
    await async_client.aforce_login(user)
    sneaky = {
        "graph": "hooked",
        "message": "hi",
        "context": {"audience": "admin"},
        "config": {"configurable": {"__llm_kit_context": {"audience": "admin"}}},
    }
    await read_events(await post(async_client, "/llm/stream/", sneaky))
    assert seen_context == [(False, {})] and HOOKED_CALLS == ["staff"]  # the graph hook still sets "staff"


def test_rest_and_background_requests_never_set_it_either(user, settings, seen_context):
    _hooks(settings, "context_hook")
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_run_hooks:noop"}
    client = Client()
    client.force_login(user)
    thread = Thread.objects.create(graph="hooked", user=user)
    body = {"message": "hi", "background": True, "context": {"audience": "admin"}}
    r = client.post(f"/llm/api/threads/{thread.id}/runs/", body, "application/json")
    assert r.status_code == 202
    run = BackgroundRun.objects.get(pk=r.json()["id"])
    assert run.context == {} and "context" not in run.payload


def test_a_background_runs_hook_sees_the_context_in_the_worker(user, settings, seen_context):
    _hooks(settings, "context_hook")
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_run_hooks:noop"}
    thread = Thread.objects.create(graph="hooked", user=user)
    run = background.start(thread, {"message": "hi"}, user=user, context={"channel": "chat", "audience": "member"})
    assert seen_context == []  # nothing ran yet
    background.execute(run.run_id)
    run.refresh_from_db()
    assert run.status == "done" and seen_context == [(True, {"channel": "chat", "audience": "member"})]
    assert run.context == {"channel": "chat", "audience": "member"}


def test_context_must_be_small_json(user):
    thread = Thread.objects.create(graph="hooked", user=user)
    with pytest.raises(ValueError, match="JSON"):
        background.start(thread, {"message": "hi"}, user=user, context={"when": object()})
    with pytest.raises(ValueError, match="bytes"):
        background.start(thread, {"message": "hi"}, user=user, context={"blob": "x" * 20_000})
    assert not BackgroundRun.objects.exists()


def test_agents_run_passes_context_to_hooks_and_tools(world, settings, seen_context):
    from llm_kit.core.hooks import context_of

    _hooks(settings, "context_hook")
    seen = {}

    def reply(messages):
        from langchain_core.runnables.config import var_child_runnable_config

        seen["tool_view"] = context_of(var_child_runnable_config.get())
        return AIMessage(content="ok")

    SCRIPTS["ollama/script-assistant"] = [reply]
    try:
        agents.run("tests.assistant", "hi", user=world["alice"], context={"channel": "api"})
    finally:
        SCRIPTS.clear()
    assert seen_context == [(False, {"channel": "api"})] and seen["tool_view"] == {"channel": "api"}


def test_background_subagents_and_their_reports_keep_the_context(world, settings):
    from llm_kit.agents.env import DjangoAgentEnv
    from llm_kit.core.hooks import CONTEXT_KEY

    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_run_hooks:noop"}
    thread = Thread.objects.create(graph="tests.assistant", user=world["alice"])
    config = {"configurable": {"thread_id": str(thread.pk), CONTEXT_KEY: {"channel": "chat"}}}
    started = DjangoAgentEnv().start_detached("tests.researcher", "look", config, {"agents": ["tests.assistant"]})
    task = BackgroundRun.objects.get(pk=started["id"])
    assert task.context == {"channel": "chat"}
    task.cancel()
    report = BackgroundRun.objects.get(reports_on=task)
    assert report.context == {"channel": "chat"}
