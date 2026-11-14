"""Bring your own key: tenant and user keys, key policies, recording whose key, the keys API."""

import pytest
from django.test import Client
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from llm_kit.core.errors import NoKey
from llm_kit.core.scoped import KeyPolicy, Settable
from llm_kit.core.touchpoints import touchpoint
from llm_kit.core.touchpoints.touchpoint import _unregister
from llm_kit.models import ProviderKey
from llm_kit.tools.caller import CONFIG_KEY, caller_for

pytestmark = pytest.mark.django_db

USED: list[tuple[str, list[str]]] = []
BAD_KEYS: set[str] = set()


class Keyed(GenericFakeChatModel):
    ref: str = ""
    keys: list[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        USED.append((self.ref, list(self.keys)))
        if set(self.keys) & BAD_KEYS:
            raise PermissionError("401 invalid api key")
        self.messages = iter([AIMessage(content=f"{self.ref} ok")])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kw)


def backend(ref, params, keys):
    return Keyed(messages=iter([]), ref=ref, keys=list(keys))


@pytest.fixture(autouse=True)
def setup(settings, monkeypatch, world):
    USED.clear()
    BAD_KEYS.clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "platform-anthropic")
    monkeypatch.setenv("GEMINI_API_KEY", "platform-gemini")
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "BYOK_SCOPES": ("tenant", "user"),
        "SCOPE_PERMISSIONS": {"tenant": "tests.testapp.tenancy:can_manage"},
    }
    yield
    for name in ("t.byok",):
        _unregister(name)


def tp(policy):
    return touchpoint(
        "t.byok",
        default="anthropic/claude-sonnet-5-5",
        fallbacks=["gemini/gemini-3.8-flash"],
        backend="tests.test_byok:backend",
        settable=Settable(keys=policy),
    )


def key(scope, scope_id, provider, secret):
    row = ProviderKey(provider=provider, scope=scope, scope_id=str(scope_id))
    row.set_key(secret)
    row.save()
    return row


def as_(user):
    return {"configurable": {CONFIG_KEY: caller_for(user)}}


def test_platform_only_never_uses_their_keys(world):
    key("tenant", world["acme"].pk, "anthropic", "acme-anthropic")
    tp(KeyPolicy.PLATFORM_ONLY).complete("hi", config=as_(world["carol"]))
    assert USED == [("anthropic/claude-sonnet-5-5", ["platform-anthropic"])]


def test_own_only_uses_their_key_and_skips_providers_they_lack(world):
    key("tenant", world["acme"].pk, "anthropic", "acme-anthropic")
    t = tp(KeyPolicy.OWN_ONLY)
    r = t.resolve(config=as_(world["carol"]))
    assert r.fallbacks == [] and ("gemini/gemini-3.8-flash", "no API key configured") in r.skipped
    BAD_KEYS.add("acme-anthropic")
    with pytest.raises(PermissionError):  # their key failed: nothing of ours is tried
        t.complete("hi", config=as_(world["carol"]))
    assert USED == [("anthropic/claude-sonnet-5-5", ["acme-anthropic"])]


def test_own_only_with_no_key_is_refused_before_any_call(world):
    with pytest.raises(NoKey, match="anthropic"):
        tp(KeyPolicy.OWN_ONLY).complete("hi", config=as_(world["bob"]))  # globex brought nothing
    assert USED == []


def test_own_then_platform_tries_theirs_then_ours(world):
    key("user", world["carol"].pk, "anthropic", "carol-anthropic")
    key("tenant", world["acme"].pk, "anthropic", "acme-anthropic")
    BAD_KEYS.update({"carol-anthropic", "acme-anthropic"})
    reply = tp(KeyPolicy.OWN_THEN_PLATFORM).complete("hi", config=as_(world["carol"]))
    assert reply.content == "anthropic/claude-sonnet-5-5 ok"
    assert [k for _, k in USED] == [["carol-anthropic"], ["acme-anthropic"], ["platform-anthropic"]]


def test_no_caller_and_other_tenants_get_platform_keys(world):
    key("tenant", world["acme"].pk, "anthropic", "acme-anthropic")
    t = tp(KeyPolicy.OWN_THEN_PLATFORM)
    t.complete("hi")  # code with no caller
    t.complete("hi", config=as_(world["bob"]))  # globex
    assert [k for _, k in USED] == [["platform-anthropic"], ["platform-anthropic"]]


def test_whose_key_is_recorded(world, monkeypatch):
    from llm_kit.core.observability import recorder

    seen = []
    monkeypatch.setattr(recorder, "emit", lambda rec: seen.append(rec) if hasattr(rec, "key_owner") else None)
    key("tenant", world["acme"].pk, "anthropic", "acme-anthropic")
    tp(KeyPolicy.OWN_THEN_PLATFORM).complete("hi", config=as_(world["carol"]))
    tp(KeyPolicy.PLATFORM_ONLY).complete("hi", config=as_(world["carol"]))
    assert [r.key_owner for r in seen] == ["tenant", "platform"]


def test_the_tenant_chooses_its_policy(world):
    from llm_kit import scoped

    key("tenant", world["acme"].pk, "anthropic", "acme-anthropic")
    BAD_KEYS.add("acme-anthropic")
    t = tp(KeyPolicy.TENANT_CHOOSES)
    assert t.complete("hi", config=as_(world["carol"])).content.endswith("ok")  # default: then ours
    scoped.save(caller_for(world["alice"]), "tenant", "keys.policy", KeyPolicy.OWN_ONLY)
    with pytest.raises(PermissionError):
        t.complete("hi", config=as_(world["carol"]))


def test_platform_keys_are_not_mixed_with_brought_ones(world):
    from llm_kit.credentials import api_keys

    key("tenant", world["acme"].pk, "anthropic", "acme-anthropic")
    assert "acme-anthropic" not in api_keys("anthropic")


# ------------------------------------------------------------------ API


def test_the_keys_api(world, settings):
    alice, carol = Client(), Client()
    alice.force_login(world["alice"])
    carol.force_login(world["carol"])
    r = alice.post(
        "/llm/api/keys/",
        {"scope": "tenant", "provider": "Anthropic", "key": "sk-ant-abcdef123456"},
        content_type="application/json",
    )
    assert r.status_code == 201
    created = r.json()
    assert created["provider"] == "anthropic" and created["fingerprint"].startswith("…3456")
    assert "sk-ant" not in str(created)

    listed = carol.get("/llm/api/keys/", {"scope": "tenant"}).json()
    assert listed["editable"] is False and [k["id"] for k in listed["keys"]] == [created["id"]]
    assert "anthropic" in listed["providers"]
    assert (
        carol.post(
            "/llm/api/keys/",
            {"scope": "tenant", "provider": "openai", "key": "sk-xxxxxxxx"},
            content_type="application/json",
        ).status_code
        == 403
    )
    assert carol.delete(f"/llm/api/keys/{created['id']}/?scope=tenant").status_code == 403
    assert Client().get("/llm/api/keys/").status_code in (401, 403)

    # A user's own keys; another tenant can't touch acme's.
    assert (
        carol.post(
            "/llm/api/keys/",
            {"scope": "user", "provider": "openai", "key": "sk-carol-1234"},
            content_type="application/json",
        ).status_code
        == 201
    )
    bob = Client()
    bob.force_login(world["bob"])
    assert bob.get("/llm/api/keys/", {"scope": "tenant"}).json()["keys"] == []

    assert alice.delete(f"/llm/api/keys/{created['id']}/?scope=tenant").status_code == 204
    settings.LLM_KIT = {**settings.LLM_KIT, "BYOK_SCOPES": ()}
    assert carol.get("/llm/api/keys/", {"scope": "user"}).status_code == 403
