"""A small tool-calling assistant. The model is chosen per run from the catalogue."""

import os
from datetime import UTC, datetime

from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from llm_kit import model_from_config


@tool
def current_time() -> str:
    """Return the current UTC date and time."""
    return datetime.now(UTC).isoformat(timespec="seconds")


TOOLS = [current_time]


def call_model(state: MessagesState, config):
    if os.environ.get("LLM_KIT_DEMO_OFFLINE"):  # try the UI with no API key
        from chat.offline import EchoModel

        return {"messages": [EchoModel(messages=iter([])).invoke(state["messages"], config)]}
    llm = model_from_config(config).bind_tools(TOOLS)
    return {"messages": [llm.invoke(state["messages"], config)]}


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")
