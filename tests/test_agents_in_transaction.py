"""A throwaway agent run inside a test transaction (Django's TestCase), as most projects test.

Its state stays in memory, so LangGraph's background thread never waits for a database
lock the test's transaction holds (on SQLite that was a 20-second hang, then an error).
"""

import pytest

from llm_kit import agents
from llm_kit.models import Checkpoint
from tests.graphs import SCRIPTS, call

pytestmark = pytest.mark.django_db  # not transaction=True: the test runs inside one transaction


def test_a_throwaway_run_with_a_subagent_never_writes_checkpoints(world):
    SCRIPTS.clear()
    SCRIPTS["ollama/script-assistant"] = [call("delegate", {"agent": "tests.researcher", "task": "A-1?"}), "done"]
    SCRIPTS["ollama/script-researcher"] = [call("Report", {"answer": "open"})]
    before = Checkpoint.objects.count()
    result = agents.run("tests.assistant", "check A-1", user=world["alice"])
    assert result.text == "done"
    assert Checkpoint.objects.count() == before
    from llm_kit.runtime.graphs import memory_checkpointer

    assert not list(memory_checkpointer().list(None))  # and freed from memory when it ended
    SCRIPTS.clear()
