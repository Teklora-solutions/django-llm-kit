"""A small tool-calling assistant. Its model policy lives in chat/touchpoints.py."""

from datetime import UTC, datetime

from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chat.touchpoints import assistant


@tool
def current_time() -> str:
    """Return the current UTC date and time."""
    return datetime.now(UTC).isoformat(timespec="seconds")


TOOLS = [current_time]


def call_model(state: MessagesState, config):
    # config carries the run's requested model ("model" in the request), checked against the touchpoint.
    llm = assistant.model(config=config).bind_tools(TOOLS)
    return {"messages": [llm.invoke(state["messages"], config)]}


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")
