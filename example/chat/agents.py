"""Agents for the chat page. Both run on llm-kit's one harness; only model, prompt and tools differ.

* ``chat.assistant`` talks to the user, can save notes, and hands research to:
* ``chat.researcher``: a cheaper model with read-only tools, starting from only the task.

Each is also a graph by name, so ``/llm/stream/`` with ``{"graph": "chat.assistant"}`` runs it.
"""

from chat.graphs import current_time
from llm_kit import agent, tools

agent(
    "chat.assistant",
    description="The chat page's assistant.",
    touchpoint="chat.assistant",
    prompt=(
        "You are the assistant on the llm-kit demo page, helping {user}. Today is {date}. "
        "Be brief. For questions that need looking through the user's notes, hand them to the researcher."
    ),
    tools=[current_time, tools.select(categories=["chat"])],
    subagents=["chat.researcher"],
    max_steps=8,
)

agent(
    "chat.researcher",
    description="Looks through the user's notes to answer one question; reports facts with note ids. Changes nothing.",
    touchpoint="chat.researcher",
    prompt="Answer the task from the user's notes. Quote note ids. Never guess; say what you could not find.",
    tools=[current_time, tools.select(kinds=["find"])],
    max_steps=6,
)
