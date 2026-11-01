"""llm_kit.core must stay importable and usable without Django."""

import subprocess
import sys
import textwrap


def run_isolated(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": ""},
    )


def test_core_never_imports_django():
    result = run_isolated(
        """
        import pkgutil, importlib, sys
        import llm_kit.core as core
        for mod in pkgutil.walk_packages(core.__path__, "llm_kit.core."):
            importlib.import_module(mod.name)
        bad = sorted(m for m in sys.modules if m == "django" or m.startswith("django."))
        assert not bad, bad
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_core_runs_a_graph_without_django():
    result = run_isolated(
        """
        import asyncio
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import START, END, MessagesState, StateGraph
        from langchain_core.messages import AIMessage
        from llm_kit.core import config
        from llm_kit.core.graphs import GraphRegistry
        from llm_kit.core.runs import astream_run, build_config, build_input

        def reply(state):
            return {"messages": [AIMessage(content=[{"type": "text", "text": "hi"}])]}

        b = StateGraph(MessagesState)
        b.add_node("reply", reply)
        b.add_edge(START, "reply")
        b.add_edge("reply", END)

        config.configure({"STREAM_MODES": ["updates"]})
        reg = GraphRegistry(targets=lambda: {"g": b}, checkpointer=InMemorySaver)

        async def main():
            cfg = build_config("t1", {"config": {"configurable": {"thread_id": "evil"}}})
            assert cfg["configurable"]["thread_id"] == "t1"
            return [e["event"] async for e in astream_run(reg.get(), build_input({"message": "x"}), cfg)]

        print(asyncio.run(main()))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['update', 'end']"


def test_message_text_flattens_blocks():
    from langchain_core.messages import AIMessage

    from llm_kit.core.serialize import message_text

    assert message_text(AIMessage(content="plain")) == "plain"
    assert message_text(AIMessage(content=[{"type": "text", "text": "a"}, {"type": "image_url"}, "b"])) == "ab"
    assert message_text(None) == ""
