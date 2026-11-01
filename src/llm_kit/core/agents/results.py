"""What running an agent returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


@dataclass
class AgentResult:
    agent: str
    text: str  # the final answer
    output: Any = None  # an instance of the agent's ``output`` schema, if it has one
    messages: list = field(default_factory=list)  # this turn's messages
    steps: int = 0  # model calls this turn
    stopped: str = "done"  # done | max_steps | timeout | paused
    thread_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.stopped == "done"

    def for_model(self) -> str:
        """What a parent agent gets back from ``delegate``."""
        if self.output is not None and hasattr(self.output, "model_dump_json"):
            return self.output.model_dump_json()
        return self.text or "(no answer)"


def this_turn(messages: list) -> list:
    """Messages after the last human message (the current turn)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1 :]
    return list(messages)


def from_state(spec, state: dict, thread_id: str = "") -> AgentResult:
    from llm_kit.core.serialize import message_text

    messages = list(state.get("messages") or [])
    turn = this_turn(messages)
    ai = [m for m in turn if isinstance(m, AIMessage)]
    final = next((m for m in reversed(ai) if not m.tool_calls), ai[-1] if ai else None)
    output = state.get("output")
    if output is not None and spec.output is not None:
        output = spec.output.model_validate(output)
    usage = [m.usage_metadata or {} for m in ai]
    return AgentResult(
        agent=spec.name,
        text=message_text(final) if final is not None else "",
        output=output,
        messages=turn,
        steps=sum(1 for m in ai if not (m.response_metadata or {}).get("llm_kit_synthetic")),
        stopped=state.get("stopped") or "done",
        thread_id=thread_id,
        input_tokens=sum(int(u.get("input_tokens") or 0) for u in usage),
        output_tokens=sum(int(u.get("output_tokens") or 0) for u in usage),
    )
