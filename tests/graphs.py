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


def call_catalogue_model(state: MessagesState, config):
    from llm_kit.catalogue import model_from_config

    return {"messages": [model_from_config(config).invoke(state["messages"], config)]}


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
