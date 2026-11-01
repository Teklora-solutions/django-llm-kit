"""Framework-independent catalogue logic: parsing, merging, syncing, diffing."""

import pytest

from llm_kit.core.catalogue import MemoryStore, merge, sync
from llm_kit.core.catalogue.fields import normalise_provider, per_mtok
from llm_kit.core.catalogue.match import unmet
from llm_kit.core.catalogue.snapshot import load as load_snapshot
from llm_kit.core.catalogue.sources import (
    LITELLM,
    MODELS_DEV,
    OPENROUTER,
    parse_litellm,
    parse_models_dev,
    parse_openrouter,
)
from tests.catalogue_payloads import LITELLM_DATA, MODELS_DEV_DATA, OPENROUTER_DATA, FakeSources


def test_price_units():
    assert per_mtok("0.000000825", per_token=True) == "0.825"
    assert per_mtok(2e-06, per_token=True) == "2"
    assert per_mtok(0.75, per_token=False) == "0.75"
    assert per_mtok("-1", per_token=True) is None  # OpenRouter "varies"
    assert per_mtok(None, per_token=True) is None
    assert per_mtok("abc", per_token=True) is None


def test_provider_names_normalised_to_litellm():
    assert normalise_provider("google") == "gemini"
    assert normalise_provider("amazon-bedrock") == "bedrock"
    assert normalise_provider("bedrock_converse") == "bedrock"
    assert normalise_provider("vertex_ai-anthropic_models") == "vertex_ai"
    assert normalise_provider("Anthropic") == "anthropic"


def test_parse_models_dev():
    r = parse_models_dev(MODELS_DEV_DATA)
    m = r.models[("gemini", "gemini-3.8-flash")]
    assert m["price_input"] == "0.75" and m["context_window"] == 1048576
    assert m["price_tiers"] == [{"above_tokens": 200000, "price_input": "1.5", "price_output": "7.5"}]
    assert m["input_modalities"] == ["text", "image", "audio", "video", "pdf"]
    assert r.providers["gemini"]["env"] == ["GOOGLE_API_KEY", "GEMINI_API_KEY"]
    assert ("openrouter", "google/gemini-3.8-flash") in r.models


def test_parse_openrouter():
    r = parse_openrouter(OPENROUTER_DATA)
    m = r.models[("openrouter", "google/gemini-3.8-flash")]
    assert m["price_input"] == "0.825" and m["price_cache_read"] == "0.0825"
    assert m["tool_call"] is True and m["structured_output"] is True
    assert m["release_date"] == "2026-09-01"
    auto = r.models[("openrouter", "openrouter/auto")]
    assert "price_input" not in auto  # -1 means variable pricing, not negative
    assert "tool_call" not in auto  # no supported_parameters -> unknown, not False


def test_parse_litellm_keys_and_kinds():
    r = parse_litellm(LITELLM_DATA)
    assert set(r.models) == {
        ("anthropic", "claude-sonnet-5-5"),
        ("gemini", "gemini-3.8-flash"),
        ("bedrock", "us.anthropic.claude-sonnet-5-5-v1:0"),
        ("vertex_ai", "claude-sonnet-5-5"),
        ("openai", "text-embedding-3-large"),
    }
    assert r.models[("openai", "text-embedding-3-large")]["kind"] == "embedding"
    assert r.models[("anthropic", "claude-sonnet-5-5")]["input_modalities"] == ["text", "image"]


def test_merge_precedence_provenance_conflicts():
    md = parse_models_dev(MODELS_DEV_DATA).models
    ll = parse_litellm(LITELLM_DATA).models
    key = ("anthropic", "claude-sonnet-5-5")
    m = merge({"models.dev": md[key], "litellm": ll[key]}, provider="anthropic")
    assert m.values["price_input"] == "2" and m.provenance["price_input"] == "litellm"  # prices: LiteLLM first
    assert m.values["structured_output"] is True and m.provenance["structured_output"] == "models.dev"
    assert m.conflicts["price_input"] == {"litellm": "2", "models.dev": "2.2"}  # >1% apart
    assert "price_cache_read" not in m.conflicts  # same value

    m = merge({"models.dev": md[key], "litellm": ll[key]}, {"price_input": "1.5", "hidden": True}, provider="anthropic")
    assert m.values["price_input"] == "1.5" and m.provenance["price_input"] == "override"
    assert m.hidden is True


def test_openrouter_is_authoritative_for_its_own_models():
    key = ("openrouter", "google/gemini-3.8-flash")
    md = parse_models_dev(MODELS_DEV_DATA).models[key]
    orr = parse_openrouter(OPENROUTER_DATA).models[key]
    m = merge({"models.dev": md, "openrouter": orr}, provider="openrouter")
    assert m.values["price_input"] == "0.825" and m.provenance["price_input"] == "openrouter"
    assert m.values["input_modalities"] == ["text", "image", "file"]


def test_requirements():
    v = {
        "tool_call": True,
        "reasoning": None,
        "input_modalities": ["text", "image"],
        "context_window": 1000,
        "kind": "chat",
    }
    assert unmet(v, {"tool_call": True, "input": ["image"], "min_context": 1000, "kind": "chat"}) == []
    assert unmet(v, {"reasoning": True})  # unknown capability is not accepted
    assert unmet(v, {"input": "audio"}) == ["input modalities lack ['audio']"]
    assert unmet(v, {"min_context": 2000})
    with pytest.raises(ValueError):
        unmet(v, {"vision": True})


def test_sync_baseline_then_conditional_requests():
    store, fetch = MemoryStore(), FakeSources()
    r = sync(store, fetch=fetch)
    assert r.baseline and r.changes == [] and r.active == 9
    assert all(s.last_status == "fetched" for s in r.sources.values())
    assert {k for k, e in store.state.entries.items() if len(e.source_values) > 1} >= {
        ("anthropic", "claude-sonnet-5-5"),
        ("gemini", "gemini-3.8-flash"),
        ("openrouter", "google/gemini-3.8-flash"),
    }
    first_requests = len(fetch.requests)

    r = sync(store, fetch=fetch)
    sent = fetch.requests[first_requests:]
    assert all(h.get("If-None-Match") == '"v1"' for _, h in sent)
    assert all(s.last_status == "not_modified" for s in r.sources.values())
    assert r.changes == [] and r.active == 9


def test_sync_detects_price_change_removal_restore_deprecation():
    store, fetch = MemoryStore(), FakeSources()
    sync(store, fetch=fetch)

    fetch.data[LITELLM.url]["claude-sonnet-5-5"]["input_cost_per_token"] = 2.5e-06
    del fetch.data[MODELS_DEV.url]["anthropic"]["models"]["claude-old-1"]
    fetch.data[MODELS_DEV.url]["google"]["models"]["gemini-3.8-flash"]["status"] = "deprecated"
    fetch.bump(LITELLM.url)
    fetch.bump(MODELS_DEV.url)
    r = sync(store, fetch=fetch)
    kinds = {(c.kind, c.provider, c.model_id, c.field) for c in r.changes}
    assert ("price_changed", "anthropic", "claude-sonnet-5-5", "price_input") in kinds
    assert ("removed", "anthropic", "claude-old-1", "") in kinds
    assert ("deprecated", "gemini", "gemini-3.8-flash", "status") in kinds
    old = store.state.entries[("anthropic", "claude-old-1")]
    assert old.removed and old.removed_at is not None

    fetch.data[MODELS_DEV.url]["anthropic"]["models"]["claude-old-1"] = MODELS_DEV_DATA["anthropic"]["models"][
        "claude-old-1"
    ]
    fetch.bump(MODELS_DEV.url)
    r = sync(store, fetch=fetch)
    assert [(c.kind, c.model_id) for c in r.changes] == [("restored", "claude-old-1")]
    assert store.state.entries[("anthropic", "claude-old-1")].removed_at is None


def test_failed_source_keeps_its_data():
    store, fetch = MemoryStore(), FakeSources()
    sync(store, fetch=fetch)
    fetch.fail.add(OPENROUTER.url)
    r = sync(store, fetch=fetch, force=True)
    assert r.sources["openrouter"].last_status == "failed"
    assert "OSError" in r.sources["openrouter"].last_error
    assert not store.state.entries[("openrouter", "openrouter/auto")].removed
    assert [c for c in r.changes if c.kind == "removed"] == []


def test_model_dropped_by_one_source_stays_if_another_lists_it():
    store, fetch = MemoryStore(), FakeSources()
    sync(store, fetch=fetch)
    del fetch.data[LITELLM.url]["claude-sonnet-5-5"]
    fetch.bump(LITELLM.url)
    r = sync(store, fetch=fetch)
    e = store.state.entries[("anthropic", "claude-sonnet-5-5")]
    assert not e.removed and set(e.source_values) == {"models.dev"}
    assert e.values["price_input"] == "2.2"  # now from models.dev
    assert ("price_changed", "price_input") in {(c.kind, c.field) for c in r.changes}


def test_override_creates_private_model():
    store, fetch = MemoryStore(), FakeSources()
    store.state.overrides[("openai", "my-finetune")] = {"kind": "chat", "price_input": "4", "tool_call": True}
    sync(store, fetch=fetch)
    e = store.state.entries[("openai", "my-finetune")]
    assert not e.removed and e.values["price_input"] == "4" and e.provenance["price_input"] == "override"


def test_bundled_snapshot_fallback_and_offline():
    snap = load_snapshot()
    assert len(snap["models.dev"].models) > 1000 and len(snap["litellm"].models) > 1000

    store, fetch = MemoryStore(), FakeSources()
    fetch.fail.add(LITELLM.url)  # never fetched before -> snapshot fills in
    r = sync(store, fetch=fetch)
    assert r.sources["litellm"].last_status == "snapshot"
    assert r.active > 1000

    store = MemoryStore()
    r = sync(store, fetch=lambda *a: pytest.fail("network used"), snapshot="only")
    assert all(s.last_status == "snapshot" for s in r.sources.values()) and r.active > 1000


def test_transient_errors_are_retried(monkeypatch):
    import llm_kit.core.catalogue.refresh as sync_mod

    monkeypatch.setattr(sync_mod.time, "sleep", lambda s: None)
    store, fetch = MemoryStore(), FakeSources()
    calls = {"n": 0}

    def flaky(url, headers):
        if url == MODELS_DEV.url and calls["n"] == 0:
            calls["n"] += 1
            raise ConnectionResetError(104, "Connection reset by peer")
        return fetch(url, headers)

    r = sync(store, fetch=flaky)
    assert r.sources["models.dev"].last_status == "fetched" and calls["n"] == 1
    r = sync(store, fetch=lambda u, h: (_ for _ in ()).throw(ConnectionResetError()), retries=0, force=True)
    assert all(s.last_status == "failed" for s in r.sources.values())


def test_only_changed_entries_are_written():
    class CountingStore(MemoryStore):
        def save(self, entries, *args):
            self.last_written = set(entries)
            super().save(entries, *args)

    store, fetch = CountingStore(), FakeSources()
    sync(store, fetch=fetch)
    assert len(store.last_written) == 9
    first_seen = store.state.entries[("anthropic", "claude-sonnet-5-5")].last_seen

    fetch.data[LITELLM.url]["claude-sonnet-5-5"]["input_cost_per_token"] = 3e-06
    fetch.bump(LITELLM.url)
    r = sync(store, fetch=fetch)
    assert store.last_written == {("anthropic", "claude-sonnet-5-5")} and r.count("price_changed") == 1
    assert store.state.entries[("gemini", "gemini-3.8-flash")].last_seen > first_seen  # still bumped


def test_connect_tries_ipv4_first_and_falls_through(monkeypatch):
    import socket

    import llm_kit.core.catalogue.refresh as sync_mod

    v6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606::1", 443, 0, 0))
    v4a = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))
    v4b = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.2", 443))
    monkeypatch.setattr(sync_mod.socket, "getaddrinfo", lambda *a, **k: [v6, v4a, v4b])
    attempts = []

    class FakeSocket:
        def __init__(self, family, type_, proto):
            self.family = family

        def settimeout(self, t):
            self.timeout = t

        def connect(self, addr):
            attempts.append((addr[0], self.timeout))
            if addr[0] != "192.0.2.2":
                raise TimeoutError("SYN-SENT forever")

        def close(self):
            pass

    monkeypatch.setattr(sync_mod.socket, "socket", FakeSocket)
    sock = sync_mod._connect("models.dev", 443, timeout=5)
    assert attempts == [("192.0.2.1", 5), ("2606::1", 5), ("192.0.2.2", 5)]
    assert sock.family == socket.AF_INET
