"""Graphs used by the test-suite. The chat model is a fake that echoes."""

from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt


class EchoModel(GenericFakeChatModel):
    """Replies 'echo: <last message>' and streams it word by word."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.messages = iter([AIMessage(content=f"echo: {messages[-1].content}")])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def call_model(state: MessagesState, config):
    model = EchoModel(messages=iter([]))
    return {"messages": [model.invoke(state["messages"], config)]}


chat_builder = StateGraph(MessagesState)
chat_builder.add_node("model", call_model)
chat_builder.add_edge(START, "model")
chat_builder.add_edge("model", END)


def _append(a: list, b: list) -> list:
    return a + b


class ApprovalState(TypedDict):
    request: str
    log: Annotated[list, _append]


def ask(state: ApprovalState):
    answer = interrupt({"question": f"Approve {state['request']}?"})
    return {"log": [f"approved={answer}"]}


def make_approval_graph(checkpointer):
    b = StateGraph(ApprovalState)
    b.add_node("ask", ask)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    return b.compile(checkpointer=checkpointer)


def explode(state: MessagesState):
    raise RuntimeError("kaboom")


_boom = StateGraph(MessagesState)
_boom.add_node("explode", explode)
_boom.add_edge(START, "explode")
boom_graph = _boom.compile()


# --- touchpoints, with a fake backend recording what it was asked to build

BUILT: list[tuple[str, dict, list]] = []


def fake_backend(ref, params, api_keys):
    """MODEL_BACKEND for tests: an echo model; refs starting with "down/" always fail."""
    BUILT.append((ref, dict(params), list(api_keys)))
    model = FailingModel(messages=iter([])) if ref.startswith("down/") else EchoModel(messages=iter([]))
    return model


class FailingModel(GenericFakeChatModel):
    def _generate(self, *a, **k):
        raise ConnectionError("provider down")

    def _stream(self, *a, **k):
        raise ConnectionError("provider down")


from llm_kit import ClientChoice, touchpoint  # noqa: E402

chat_tp = touchpoint(
    "tests.chat",
    default="anthropic/claude-sonnet-5-5",
    fallbacks=["gemini/gemini-3.8-flash"],
    params={"temperature": 0},
    choice=ClientChoice(allow=("anthropic/*", "gemini/*"), requires={"tool_call": True}),
)


def call_catalogue_model(state: MessagesState, config):
    return {"messages": [chat_tp.model(config=config).invoke(state["messages"], config)]}


catalogue_builder = StateGraph(MessagesState)
catalogue_builder.add_node("model", call_catalogue_model)
catalogue_builder.add_edge(START, "model")
catalogue_builder.add_edge("model", END)


async def sleepy(state: MessagesState):
    import asyncio

    await asyncio.sleep(30)
    return {}


slow_builder = StateGraph(MessagesState)
slow_builder.add_node("sleepy", sleepy)
slow_builder.add_edge(START, "sleepy")


# --- an agent that calls llm-kit model tools. The user message is JSON: {"tool": name, "args": {...}}


def make_tools_graph(checkpointer):
    import json

    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode

    from llm_kit import tools

    def agent(state: MessagesState, config):
        last = state["messages"][-1]
        if last.type == "tool":
            return {"messages": [AIMessage(content=f"tool said: {last.content}")]}
        ask = json.loads(last.content)
        offered = [t.name for t in tools.for_config(config)]  # what the model would be bound with
        call = {"name": ask["tool"], "args": ask.get("args", {}), "id": "call-1", "type": "tool_call"}
        return {"messages": [AIMessage(content=f"offered: {','.join(offered)}", tool_calls=[call])]}

    def route(state: MessagesState):
        return "tools" if state["messages"][-1].tool_calls else END

    b = StateGraph(MessagesState)
    b.add_node("agent", agent)
    b.add_node("tools", ToolNode(tools.all_tools()))
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", route, ["tools", END])
    b.add_edge("tools", "agent")
    return b.compile(checkpointer=checkpointer)
