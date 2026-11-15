"""Tool calls under a time limit; tool failures as error results; a deadline that leaves time to answer."""

import asyncio
import threading
import time

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool

from llm_kit import agents
from llm_kit.core.agents import spec as agent_spec
from llm_kit.core.agents import toolcalls
from llm_kit.core.agents.limits import TREE_KEY, Tree
from tests.graphs import SCRIPTS, SEEN, call

H = "ollama/script-helper"
RELEASE = threading.Event()
STATE = {}


@pytest.fixture(autouse=True)
def clean():
    SCRIPTS.clear()
    SEEN.clear()
    RELEASE.clear()
    STATE.clear()
    yield
    RELEASE.set()  # let an abandoned tool finish
    SCRIPTS.clear()
    SEEN.clear()


@pytest.fixture
def declare():
    names = []

    def declare_(name, **options):
        names.append(name)
        options.setdefault("description", f"{name} for a test")
        options.setdefault("touchpoint", "tests.helper")
        return agents.agent(name, **options)

    yield declare_
    for name in names:
        agent_spec._unregister(name)


@tool
def hang(url: str) -> str:
    """Read a web page (this one never loads)."""
    STATE["thread"] = threading.get_ident()
    RELEASE.wait(10)
    STATE["finished"] = True
    return "the page"


hang.metadata = {toolcalls.TIMEOUT_KEY: 0.5}


@tool
async def ahang(url: str) -> str:
    """Read a web page, async (this one never loads)."""
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        STATE["cancelled"] = True
        raise
    return "the page"


ahang.metadata = {toolcalls.TIMEOUT_KEY: 0.5}


@tool
def broken(what: str) -> str:
    """Fails."""
    if what == "reported":
        raise ToolException("the CRM is down for maintenance")
    raise KeyError("secret-internal-key")


class Recorder(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    def on_tool_start(self, *args, **kwargs):
        self.events.append("start")

    def on_tool_end(self, *args, **kwargs):
        self.events.append("end")

    def on_tool_error(self, *args, **kwargs):
        self.events.append("error")


def _tool_messages(result):
    return [m for m in result.messages if isinstance(m, ToolMessage)]


@pytest.mark.django_db(transaction=True)  # outside a transaction: a sync tool runs on a thread of its own
def test_a_tool_that_hangs_gives_an_error_result_and_the_run_still_answers(declare):
    declare("tests.reader", tools=[hang])
    SCRIPTS[H] = [call("hang", {"url": "https://example.com"}), "I couldn't read the page."]
    recorder = Recorder()
    started = time.monotonic()
    result = agents.run("tests.reader", "read it", callbacks=[recorder])
    assert time.monotonic() - started < 5
    (reply,) = _tool_messages(result)
    assert reply.status == "error"
    assert "hang did not finish within 0.5 seconds" in reply.content and "result is unknown" in reply.content
    assert result.text == "I couldn't read the page." and result.stopped == "done"
    assert recorder.events == ["start", "error"]  # the call was closed, as a failure
    assert STATE["thread"] != threading.get_ident() and "finished" not in STATE  # abandoned, still running
    RELEASE.set()


@pytest.mark.django_db(transaction=True)
async def test_an_async_tool_that_hangs_is_cancelled(declare):
    declare("tests.areader", tools=[ahang])
    SCRIPTS[H] = [call("ahang", {"url": "https://example.com"}), "No luck."]
    result = await agents.arun("tests.areader", "read it")
    (reply,) = _tool_messages(result)
    assert reply.status == "error" and "did not finish within 0.5 seconds" in reply.content
    assert result.text == "No luck." and STATE.get("cancelled") is True


@pytest.mark.django_db(transaction=True)
async def test_a_sync_tool_that_hangs_in_an_async_run(declare):
    declare("tests.reader2", tools=[hang])
    SCRIPTS[H] = [call("hang", {"url": "x"}), "No luck."]
    result = await agents.arun("tests.reader2", "read it")
    (reply,) = _tool_messages(result)
    assert reply.status == "error" and "did not finish" in reply.content and result.text == "No luck."


@pytest.mark.django_db  # inside the test's transaction: the tool must see its data, so it runs here
def test_inside_a_transaction_a_sync_tool_runs_where_it_was_called(declare):
    @tool
    def where() -> str:
        """Which thread."""
        STATE["thread"] = threading.get_ident()
        return "here"

    declare("tests.where", tools=[where])
    SCRIPTS[H] = [call("where"), "ok"]
    agents.run("tests.where", "x")
    assert STATE["thread"] == threading.get_ident()


@pytest.mark.django_db
def test_tool_failures_are_error_results_the_model_reads(declare, settings):
    declare("tests.breaker", tools=[broken])
    SCRIPTS[H] = [call("broken", {"what": "reported"}), call("broken", {"what": "bug"}), "done"]
    first, second = _tool_messages(agents.run("tests.breaker", "x"))
    assert first.status == second.status == "error"
    assert first.content == "Error: broken failed: the CRM is down for maintenance"
    assert second.content == "Error: broken failed: KeyError"  # nothing internal reaches the model

    settings.LLM_KIT = {**settings.LLM_KIT, "EXPOSE_ERRORS": True}
    SCRIPTS[H] = [call("broken", {"what": "bug"}), "done"]
    (reply,) = _tool_messages(agents.run("tests.breaker", "x"))
    assert "secret-internal-key" in reply.content


@pytest.mark.django_db
def test_near_the_deadline_no_tool_starts_and_the_agent_is_told_to_answer(declare, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "AGENT_TIMEOUT": 20, "AGENT_ANSWER_RESERVE": 25}
    declare("tests.hurried", tools=[hang])
    SCRIPTS[H] = ["Here is what I have."]
    result = agents.run("tests.hurried", "x")
    assert result.text == "Here is what I have."
    assert SEEN[0]["tools"] == [] and "Time is nearly up" in SEEN[0]["system"]


@pytest.mark.django_db
def test_a_tool_call_made_anyway_is_not_run(declare, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "AGENT_TIMEOUT": 20, "AGENT_ANSWER_RESERVE": 25}
    declare("tests.hurried2", tools=[hang])
    SCRIPTS[H] = [call("hang", {"url": "x"})]  # a model that calls a tool it wasn't offered
    result = agents.run("tests.hurried2", "x")
    (reply,) = _tool_messages(result)
    assert reply.status == "error" and "no time left" in reply.content
    assert result.stopped == "timeout" and "thread" not in STATE


def test_each_call_may_take_the_lesser_of_its_timeout_and_the_time_left_less_the_reserve(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "AGENT_ANSWER_RESERVE": 25, "TOOL_TIMEOUT": 90}
    tree = Tree.new()
    config = {"configurable": {TREE_KEY: tree}}
    tree.deadline = time.monotonic() + 40
    assert 14 < toolcalls.seconds_for(hang.model_copy(update={"metadata": {}}), config) <= 15
    assert toolcalls.seconds_for(hang, config) == 0.5  # its own, shorter
    assert toolcalls.seconds_for(hang, config, own=False) > 14  # a subagent: the deadline only
    tree.deadline = time.monotonic() + 29  # 4 s once the reserve is kept back
    assert toolcalls.seconds_for(hang, config) == 0
    assert toolcalls.seconds_for(hang, {}) == 0.5  # outside an agent run: its own limit
    assert 3 < toolcalls.time_left(config) <= 29 and toolcalls.time_left({}) is None


@pytest.mark.django_db
def test_tools_see_the_time_left(declare):
    from langchain_core.runnables import RunnableConfig

    @tool
    def clock(config: RunnableConfig) -> str:
        """How long is left."""
        STATE["left"] = agents.time_left(config)
        return "ok"

    declare("tests.clock", tools=[clock])
    SCRIPTS[H] = [call("clock"), "ok"]
    agents.run("tests.clock", "x")
    assert 0 < STATE["left"] <= 300


@pytest.mark.django_db
def test_registered_tools_take_a_timeout():
    from llm_kit.tools import registry

    entry = registry.ToolEntry(
        name="slow_thing",
        description="x",
        category="custom",
        kind="custom",
        permissions=("x",),
        args_schema=hang.args_schema,
        inner=hang,
        timeout=12,
    )
    made = entry.tool()
    assert made.metadata[toolcalls.TIMEOUT_KEY] == 12 and made.metadata[toolcalls.SYNC_KEY] is True
    assert not toolcalls.is_async(made) and toolcalls.is_async(ahang)
