"""Loading tools on demand, so an agent that can reach hundreds of tools doesn't send them all.

With deferral on, the model is shown its ``essential`` tools, ``find_tools``, and a one-line
overview of what else exists (categories and counts). ``find_tools`` searches the rest and
binds the best matches from the next step on; they stay loaded for the thread
(``loaded_tools`` in the checkpointed state), up to ``MAX_LOADED_TOOLS``, the most recently
found first.

Deferral is on when an agent declares ``defer_tools=True``, or, by default, when it can use
more than ``DEFER_TOOLS_ABOVE`` tools.

Search ranks tools by the query's words in their names, categories, descriptions and
argument names, rarer words counting more. With ``TOOL_SEARCH_TOUCHPOINT`` (an embedding
touchpoint), it ranks by meaning too, so "money owed" finds ``find_invoices``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from llm_kit.core import config as settings

logger = logging.getLogger("llm_kit.agents")

FIND_TOOLS = "find_tools"
MAX_FOUND = 6
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and the of to for in on with by from my me i is are be it this that what all any".split())
OTHER = "other"


class _FindArgs(BaseModel):
    query: str = Field(default="", description="What you need to do, in a few words, e.g. 'overdue invoices'.")
    category: str | None = Field(default=None, description="Only tools in this category (see the overview).")


def find_tools_tool() -> StructuredTool:
    """The tool the model sees. The harness's tool node answers it (it needs the run's tools)."""

    def run(query: str = "", category: str | None = None) -> str:
        return "find_tools only works inside an agent run."

    return StructuredTool(
        name=FIND_TOOLS,
        description=(
            "Find more tools for a task and load them. Describe the task in a few words; optionally "
            "name a category. Tools it finds can be used from your next step."
        ),
        args_schema=_FindArgs,
        func=run,
        metadata={"llm_kit_kind": "find_tools"},
    )


def should_defer(declared: bool | None, count: int) -> bool:
    """``defer_tools`` as declared, or (None) whether the agent has more tools than ``DEFER_TOOLS_ABOVE``."""
    if declared is not None:
        return bool(declared)
    above = settings.get("DEFER_TOOLS_ABOVE")
    return above is not None and count > int(above)


def keep_loaded(loaded: Sequence[str]) -> list[str]:
    """The tools that stay bound: the ``MAX_LOADED_TOOLS`` most recently found."""
    limit = settings.get("MAX_LOADED_TOOLS")
    loaded = list(loaded)
    return loaded if limit is None else loaded[-int(limit) :] if int(limit) > 0 else []


# ------------------------------------------------------------------ describing tools


def category_of(tool: BaseTool) -> str:
    return str((tool.metadata or {}).get("llm_kit_category") or OTHER)


def overview(tools: Sequence[BaseTool], limit: int = 15) -> str:
    """``invoices (12), crm (8), other (3)``: what the deferred tools cover, without listing them."""
    counts = Counter(category_of(t) for t in tools)
    parts = [f"{name} ({n})" for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if len(parts) > limit:
        parts = [*parts[:limit], f"{len(parts) - limit} more categories"]
    return ", ".join(parts)


def _words(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    return words + [w[:-1] for w in words if len(w) > 3 and w.endswith("s")]  # invoices ~ invoice


def _arg_text(tool: BaseTool) -> str:
    try:
        schema = tool.args or {}
    except Exception:  # a tool whose schema can't be built: search its name and description
        return ""
    return " ".join(f"{name} {(spec or {}).get('description', '')}" for name, spec in schema.items())


def tool_text(tool: BaseTool) -> str:
    """What a tool is searched (and embedded) by."""
    first = (tool.description or "").strip()
    return f"{tool.name.replace('_', ' ')}. {category_of(tool)}. {first} {_arg_text(tool)}"


def _fields(tool: BaseTool) -> list[tuple[int, list[str]]]:
    return [
        (3, _words(tool.name.replace("_", " "))),
        (2, _words(category_of(tool))),
        (1, _words(tool.description or "")),
        (1, _words(_arg_text(tool))),
    ]


# ------------------------------------------------------------------ searching


Ranker = Callable[[str, Sequence[BaseTool]], list[float]]


def keyword_scores(query: str, tools: Sequence[BaseTool]) -> list[float]:
    """Weighted word matches, each word counting more the fewer tools have it (IDF)."""
    wanted = set(_words(query))
    if not wanted or not tools:
        return [0.0] * len(tools)
    fields = [_fields(t) for t in tools]
    vocab = [{w for _, words in f for w in words} for f in fields]
    n = len(tools)

    def idf(word: str) -> float:
        df = sum(1 for v in vocab if word in v or any(len(word) >= 4 and w.startswith(word) for w in v))
        return math.log(1 + n / df) if df else 0.0

    weights = {w: idf(w) for w in wanted}
    scores = []
    for f in fields:
        score = 0.0
        for word, weight in weights.items():
            best = 0
            for field_weight, words in f:
                if word in words:
                    best = max(best, field_weight)
                elif len(word) >= 4 and any(w.startswith(word) for w in words):
                    best = max(best, field_weight / 2)  # "invoic" ~ "invoicing"
            score += best * weight
        scores.append(score)
    return scores


def match(
    query: str,
    tools: Sequence[BaseTool],
    limit: int = MAX_FOUND,
    *,
    category: str | None = None,
    ranker: Ranker | None = None,
) -> list[BaseTool]:
    """The best ``limit`` tools for ``query`` (in ``category``, if given). With no query, a
    category's tools in order. ``ranker`` (by meaning) is added to the keyword score."""
    pool = [t for t in tools if category is None or category_of(t).lower() == category.strip().lower()]
    if not pool:
        return []
    if not query.strip():
        return pool[:limit] if category is not None else []
    scores = keyword_scores(query, pool)
    top = max(scores) or 1.0
    combined = [s / top for s in scores]  # 0..1
    if ranker is not None:
        try:
            meaning = ranker(query, pool)
        except Exception:
            logger.warning("Tool search by meaning failed; using keywords only.", exc_info=True)
        else:
            combined = [k + 2 * max(m, 0.0) for k, m in zip(combined, meaning, strict=True)]
    ranked = sorted(range(len(pool)), key=lambda i: (-combined[i], i))
    if ranker is None:
        ranked = [i for i in ranked if scores[i] > 0]
    else:  # by meaning everything scores something: keep those near the best
        best = combined[ranked[0]] if ranked else 0
        ranked = [i for i in ranked if combined[i] >= best * 0.6]
    return [pool[i] for i in ranked[:limit]]


def found_reply(found: Sequence[BaseTool], remaining: Sequence[BaseTool], category: str | None = None) -> str:
    if not found:
        where = f" in {category!r}" if category else ""
        return (
            f"No tools match{where}. {len(remaining)} are available, by category: {overview(remaining)}. "
            "Try other words, or a category."
        )
    lines = "\n".join(f"- {t.name}: {(t.description or '').strip().splitlines()[0][:200]}" for t in found)
    return f"Loaded these tools; you can call them from your next step:\n{lines}"


def union(old: list | None, new: list | None) -> list:
    """State reducer for ``loaded_tools``: found again moves a tool to the end (most recent)."""
    new = list(dict.fromkeys(new or []))
    return [*(t for t in (old or []) if t not in new), *new]


# ------------------------------------------------------------------ by meaning


_vectors: dict[tuple[str, str], list[float]] = {}  # (touchpoint, text hash) -> vector
_vectors_lock = threading.Lock()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def semantic_ranker() -> Ranker | None:
    """Ranking by meaning through the embedding touchpoint ``TOOL_SEARCH_TOUCHPOINT``, if set.

    Tool vectors are cached in memory by their text, so each tool is embedded once per process."""
    name = settings.get("TOOL_SEARCH_TOUCHPOINT")
    if not name:
        return None
    from llm_kit.core.touchpoints import get_touchpoint

    tp = get_touchpoint(name)

    def rank(query: str, tools: Sequence[BaseTool]) -> list[float]:
        keys = [(tp.name, hashlib.sha256(tool_text(t).encode()).hexdigest()) for t in tools]
        with _vectors_lock:
            missing = [i for i, k in enumerate(keys) if k not in _vectors]
        if missing:
            vectors = tp.embed([tool_text(tools[i]) for i in missing])
            with _vectors_lock:
                if len(_vectors) > 20_000:
                    _vectors.clear()
                for i, v in zip(missing, vectors, strict=True):
                    _vectors[keys[i]] = list(v)
        [q] = tp.embed([query])
        with _vectors_lock:
            return [_cosine(q, _vectors.get(k) or []) for k in keys]

    return rank
