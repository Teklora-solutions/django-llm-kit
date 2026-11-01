"""Loading tools on demand (``defer_tools=True``): only ``essential`` tools are bound at first,
plus ``find_tools``. Tools it finds are bound from the next step on, and stay loaded for the
rest of the thread (``loaded_tools`` in the checkpointed state). Keeps prompts small when an
agent can reach many tools.

Matching is by keywords over tool names and descriptions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

FIND_TOOLS = "find_tools"
MAX_FOUND = 5
_WORD = re.compile(r"[a-z0-9]+")


class _FindArgs(BaseModel):
    query: str = Field(description="What you need to do, in a few words, e.g. 'overdue invoices'.")


def find_tools_tool() -> StructuredTool:
    """The tool the model sees. The harness's tool node answers it (it needs the run's tools)."""

    def run(query: str) -> str:
        return "find_tools only works inside an agent run."

    return StructuredTool(
        name=FIND_TOOLS,
        description="Find more tools for a task. Tools it finds can be used from your next step.",
        args_schema=_FindArgs,
        func=run,
        metadata={"llm_kit_kind": "find_tools"},
    )


def _words(text: str) -> set[str]:
    words = set(_WORD.findall(text.lower()))
    return words | {w[:-1] for w in words if len(w) > 3 and w.endswith("s")}  # invoices ~ invoice


def match(query: str, tools: Sequence[BaseTool], limit: int = MAX_FOUND) -> list[BaseTool]:
    wanted = _words(query)
    scored = []
    for index, tool in enumerate(tools):
        name, text = _words(tool.name.replace("_", " ")), _words(tool.description or "")
        score = 3 * len(wanted & name) + len(wanted & text)
        if score:
            scored.append((-score, index, tool))
    return [tool for _, _, tool in sorted(scored)[:limit]]


def found_reply(found: Sequence[BaseTool], remaining: int) -> str:
    if not found:
        return f"No tools match. {remaining} more are available: try other words."
    lines = "\n".join(f"- {t.name}: {(t.description or '').strip().splitlines()[0][:200]}" for t in found)
    return f"Loaded these tools; you can call them from your next step:\n{lines}"


def union(old: list | None, new: list | None) -> list:
    """State reducer for ``loaded_tools``."""
    return list(dict.fromkeys([*(old or []), *(new or [])]))
