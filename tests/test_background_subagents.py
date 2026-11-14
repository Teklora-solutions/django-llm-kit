"""Background subagents: delegate(..., background=true) starts the child detached, and its answer
comes back to the parent's thread as a queued run."""

import time

import pytest

from llm_kit import agent, background
from llm_kit.agents.env import DjangoAgentEnv
from llm_kit.core.agents import spec as specs
from llm_kit.core.agents.delegate import CANCEL_TASK, CHECK_TASK, background_names
from llm_kit.core.agents.env import AgentEnv
from llm_kit.core.errors import ConfigError
from llm_kit.models import BackgroundRun, Thread
from llm_kit.runtime.graphs import get_graph
from tests.graphs import SCRIPTS, SEEN, call
from tests.test_background import finish

pytestmark = pytest.mark.django_db(transaction=True)

BOSS, RESEARCHER = "ollama/script-assistant", "ollama/script-researcher"


@pytest.fixture(autouse=True)
def boss():
    SCRIPTS.clear()
    SEEN.clear()
    agent(
        "tests.boss",
        description="Hands long research to the researcher in the background.",
        touchpoint="tests.assistant",
        prompt="Help.",
        subagents=["tests.researcher"],
        background_subagents=True,
    )
    yield
    specs._unregister("tests.boss")
    SCRIPTS.clear()
    SEEN.clear()


def settle(thread, timeout=15):
    """Run everything started from the thread (its tasks and their reports) to the end."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending = list(BackgroundRun.objects.filter(status__in=("waiting", "queued", "running")))
        if not pending:
            return
        for run in pending:
            finish(run, timeout=1) if run.status == "running" else time.sleep(0.02)
    raise AssertionError(f"still pending: {[(r.graph, r.status) for r in pending]}")


def test_the_answer_comes_back_to_the_parent_who_tells_the_user(world):
    alice = world["alice"]
    SCRIPTS[BOSS] = [
        call("delegate", {"agent": "tests.researcher", "task": "Is A-1 paid?", "background": True}, id="d1"),
        "I've started on that.",
        "The researcher says A-1 is open.",
    ]
    SCRIPTS[RESEARCHER] = ["A-1 is open", call("Report", {"answer": "open", "ids": [1]})]
    thread = Thread.objects.create(graph="tests.boss", user=alice, tenant=str(world["acme"].pk))
    first = finish(background.start(thread, {"message": "check A-1"}, user=alice))
    assert first.status == "done"
    settle(thread)

    task = BackgroundRun.objects.get(parent_thread=thread)
    assert task.status == "done" and task.graph == "tests.researcher" and task.thread != thread
    assert task.thread.user == alice and task.user == alice and task.tenant == str(world["acme"].pk)
    assert task.agents == ["tests.boss", "tests.researcher"]
    assert str(task.parent_run_id) == first.run_id  # recorded under the run that started it

    reported = BackgroundRun.objects.get(reports_on=task)
    assert reported.thread == thread and reported.status == "done"
    messages = get_graph("tests.boss").get_state(thread.config).values["messages"]
    tool_reply = next(m for m in messages if m.type == "tool")
    assert f"task {task.run_id}" in tool_reply.content and "Tell the user you've started" in tool_reply.content
    note = messages[-2]
    assert note.type == "human" and note.content.startswith("[Automatic message, not from the user]")
    assert '"answer":"open"' in note.content and "is done" in note.content
    assert messages[-1].content == "The researcher says A-1 is open."


def test_the_child_inherits_the_requests_limits(world, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "AGENT_MAX_DELEGATIONS": 3, "AGENT_BUDGET": 5}
    SCRIPTS[BOSS] = [call("delegate", {"agent": "tests.researcher", "task": "x", "background": True}), "ok"]
    thread = Thread.objects.create(graph="tests.boss", user=world["alice"])
    finish(background.start(thread, {"message": "go"}, user=world["alice"]))
    task = BackgroundRun.objects.get(parent_thread=thread)
    assert task.limits == {"budget": "5", "delegations": 2, "max_depth": 2}  # one hand-off used to start it
    settle(thread)


def test_check_and_cancel_only_this_threads_tasks(world, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}
    env, alice = DjangoAgentEnv(), world["alice"]
    thread = Thread.objects.create(graph="tests.boss", user=alice)
    other = Thread.objects.create(graph="tests.boss", user=alice)
    mine = {"configurable": {"thread_id": str(thread.pk)}}
    started = env.start_detached("tests.researcher", "look", mine, {"agents": ["tests.boss", "tests.researcher"]})
    assert env.task_status(started["id"], mine)["status"] == "queued"
    assert env.task_status(started["id"], {"configurable": {"thread_id": str(other.pk)}}) is None
    assert env.task_status("nonsense", mine) is None
    assert env.cancel_task(started["id"], {"configurable": {"thread_id": str(other.pk)}}) is None
    assert env.cancel_task(started["id"], mine) is True
    assert env.task_status(started["id"], mine)["status"] == "cancelled"
    assert env.cancel_task(started["id"], mine) is False
    report = BackgroundRun.objects.get(reports_on_id=started["id"])  # cancelled tasks are reported too
    assert report.thread == thread and "was cancelled" in report.payload["input"]["messages"][0]["content"]


def test_the_model_can_check_on_a_task(world):
    SCRIPTS[BOSS] = [
        call("delegate", {"agent": "tests.researcher", "task": "x", "background": True}, id="d1"),
        lambda messages: call(CHECK_TASK, {"task_id": _task_id(messages)}),
        "ok",
    ]
    SCRIPTS[RESEARCHER] = ["open", call("Report", {"answer": "open"})]
    thread = Thread.objects.create(graph="tests.boss", user=world["alice"])
    finish(background.start(thread, {"message": "go"}, user=world["alice"]))
    messages = get_graph("tests.boss").get_state(thread.config).values["messages"]
    checked = [m for m in messages if m.type == "tool"][1]
    assert checked.content.startswith("Task ") and "tests.researcher" in checked.content
    assert {CHECK_TASK, CANCEL_TASK} <= set(SEEN[0]["tools"])
    settle(thread)


def _task_id(messages):
    import re

    tool = next(m for m in reversed(messages) if m.type == "tool")
    return re.search(r"task ([0-9a-f-]{36})", tool.content).group(1)


def test_no_background_option_without_a_saved_thread(world):
    """A throwaway run (agents.run without a thread) has nowhere to send the answer later."""
    from llm_kit import agents

    SCRIPTS[BOSS] = ["fine"]
    agents.run("tests.boss", "x", user=world["alice"])
    assert "check_task" not in SEEN[0]["tools"]
    assert background_names(specs.get_agent("tests.boss"), [specs.get_agent("tests.researcher")], {}, AgentEnv()) == ()


def test_a_background_subagent_must_be_one_of_the_subagents():
    with pytest.raises(ConfigError, match="background_subagents"):
        agent("tests.bad", description="x", touchpoint="tests.assistant", subagents=[], background_subagents=["x.y"])
    a = agent(
        "tests.some",
        description="x",
        touchpoint="tests.assistant",
        subagents=["tests.researcher", "tests.helper"],
        background_subagents=["tests.helper"],
    )
    try:
        assert a.background_subagents == ("tests.helper",)
    finally:
        specs._unregister("tests.some")


def test_a_detached_run_is_recorded_under_its_parent():
    from uuid import uuid4

    from llm_kit.core.observability.handler import ObservabilityHandler

    handler, run_id, parent = ObservabilityHandler(), uuid4(), str(uuid4())
    handler.on_chain_start({}, {}, run_id=run_id, metadata={"llm_kit_detached_from": parent, "llm_kit_depth": 1})
    rec = handler._runs[run_id][0]
    assert rec.parent_run_id == parent and rec.depth == 1
