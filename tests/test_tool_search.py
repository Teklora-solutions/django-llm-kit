"""find_tools: ranking, categories, overviews and search by meaning."""

import pytest
from langchain_core.tools import StructuredTool

from llm_kit.core.agents import deferral


def tool(name, description, category="other", **args):
    def run(**kw):
        return ""

    from pydantic import Field, create_model

    schema = create_model(f"{name}_args", **{k: (str, Field(description=v)) for k, v in args.items()})
    return StructuredTool(
        name=name, description=description, func=run, args_schema=schema, metadata={"llm_kit_category": category}
    )


TOOLS = [
    tool("find_invoices", "Find invoices by customer, status or date.", "billing", status="draft, sent or paid"),
    tool("save_invoice", "Create or change an invoice.", "billing"),
    tool("find_customers", "Find customers by name or email.", "crm"),
    tool("send_email", "Send an email to a customer.", "messaging", to="address"),
    tool("find_notes", "Find your notes.", "notes"),
    *[tool(f"report_{i}", f"Monthly report number {i} about invoices.", "reports") for i in range(10)],
]


def names(found):
    return [t.name for t in found]


def test_names_count_more_and_common_words_count_less():
    found = deferral.match("find invoices", TOOLS)
    assert names(found)[0] == "find_invoices"  # name match; "invoices" is in 11 descriptions
    assert names(deferral.match("customer email", TOOLS))[:2] == ["send_email", "find_customers"]


def test_argument_names_categories_and_word_stems_are_searched():
    assert names(deferral.match("paid", TOOLS)) == ["find_invoices"]  # an argument's description
    assert "find_invoices" in names(deferral.match("billing", TOOLS))  # its category
    assert "find_invoices" in names(deferral.match("invoic", TOOLS))  # a prefix


def test_a_category_narrows_or_lists():
    assert names(deferral.match("find", TOOLS, category="crm")) == ["find_customers"]
    assert len(deferral.match("", TOOLS, category="reports", limit=4)) == 4  # no query: browse
    assert deferral.match("", TOOLS) == []
    assert deferral.match("zebra", TOOLS) == []


def test_overview_and_no_match_reply():
    assert deferral.overview(TOOLS) == "reports (10), billing (2), crm (1), messaging (1), notes (1)"
    reply = deferral.found_reply([], TOOLS)
    assert "15 are available" in reply and "reports (10)" in reply


def test_ranking_by_meaning(monkeypatch):
    def ranker(query, tools):
        return [1.0 if t.name == "find_invoices" else 0.1 for t in tools]

    assert names(deferral.match("money people owe us", TOOLS, ranker=ranker)) == ["find_invoices"]

    def broken(query, tools):
        raise RuntimeError("embedding provider down")

    assert names(deferral.match("find invoices", TOOLS, ranker=broken))[0] == "find_invoices"  # keywords still work


def test_semantic_ranker_embeds_each_tool_once(settings, monkeypatch):
    from llm_kit.core.touchpoints import touchpoint

    embedded = []

    class Fake:
        def embed(self, inputs):
            embedded.extend(inputs)
            return [[1.0, 0.0] if "invoice" in i.lower() or "owe" in i else [0.0, 1.0] for i in inputs]

    tp = touchpoint("t.toolsearch", default="openai/text-embedding-3-small")
    monkeypatch.setattr(type(tp), "embed", lambda self, inputs, **kw: Fake().embed(inputs))
    settings.LLM_KIT = {**settings.LLM_KIT, "TOOL_SEARCH_TOUCHPOINT": "t.toolsearch"}
    deferral._vectors.clear()
    rank = deferral.semantic_ranker()
    scores = rank("what do they owe", TOOLS[:3])
    assert scores[0] == pytest.approx(1.0) and scores[2] == pytest.approx(0.0)
    rank("again", TOOLS[:3])
    assert len(embedded) == 3 + 2  # tools once, then only the two queries


def test_found_again_moves_to_the_end_and_the_cap_keeps_the_latest(settings):
    assert deferral.union(["a", "b", "c"], ["a", "d"]) == ["b", "c", "a", "d"]
    settings.LLM_KIT = {**settings.LLM_KIT, "MAX_LOADED_TOOLS": 2}
    assert deferral.keep_loaded(["b", "c", "a", "d"]) == ["a", "d"]
