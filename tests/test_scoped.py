"""Scoped settings: model choices and params saved by tenants and users, custom settings, the API."""

import pytest
from django.test import Client

from llm_kit import scoped
from llm_kit.core.scoped import Param, Settable, Setting
from llm_kit.core.touchpoints import ClientChoice, touchpoint
from llm_kit.core.touchpoints.touchpoint import _unregister
from llm_kit.models import ScopedSetting
from llm_kit.tools.caller import CONFIG_KEY, caller_for

pytestmark = pytest.mark.django_db

CLAUDE, GEMINI = "anthropic/claude-sonnet-5-5", "gemini/gemini-3.8-flash"


@pytest.fixture
def tp(db, keys, settings):
    from llm_kit import catalogue
    from tests.catalogue_payloads import FakeSources

    catalogue.sync(fetch=FakeSources())
    settings.LLM_KIT = {**settings.LLM_KIT, "SCOPE_PERMISSIONS": {"tenant": "tests.testapp.tenancy:can_manage"}}
    t = touchpoint(
        "t.settable",
        default=CLAUDE,
        description="Answers support questions.",
        choice=ClientChoice(allow=["anthropic/*", "gemini/*"]),
        settable=Settable(
            model=("tenant", "user"),
            params={"temperature": Param("number", min=0, max=1, scopes=("user",))},
        ),
    )
    yield t
    _unregister("t.settable")


def run_for(user):
    return {"configurable": {CONFIG_KEY: caller_for(user)}}


def test_a_saved_model_applies_narrowest_scope_first(tp, world):
    alice, carol = world["alice"], world["carol"]  # both in acme
    assert tp.resolve(config=run_for(carol)).model == CLAUDE  # nothing saved: the code's default

    scoped.save(caller_for(alice), "tenant", "model", GEMINI, touchpoint="t.settable")
    r = tp.resolve(config=run_for(carol))
    assert (r.model, r.chosen_by) == (GEMINI, "tenant")  # carol inherits acme's choice

    scoped.save(caller_for(carol), "user", "model", CLAUDE, touchpoint="t.settable")
    r = tp.resolve(config=run_for(carol))
    assert (r.model, r.chosen_by) == (CLAUDE, "user")
    assert tp.resolve(config=run_for(world["bob"])).model == CLAUDE  # globex saved nothing

    # A client's explicit request still comes first; no caller (code) gets the default.
    assert tp.resolve(GEMINI, config=run_for(carol)).chosen_by == "client"
    assert tp.resolve().chosen_by == "code"


def test_a_saved_model_stays_inside_the_touchpoints_choice(tp, world):
    with pytest.raises(ValueError, match="allowed options"):
        scoped.save(caller_for(world["carol"]), "user", "model", "openai/gpt-5", touchpoint="t.settable")
    # A value that is no longer allowed (saved earlier, or written directly) is ignored.
    ScopedSetting.objects.create(
        scope="user", scope_id=str(world["carol"].pk), touchpoint="t.settable", key="model", value="openai/gpt-5"
    )
    r = tp.resolve(config=run_for(world["carol"]))
    assert r.model == CLAUDE and any("not allowed" in n for n in r.notes)


def test_saved_params_merge_and_are_checked(tp, world):
    carol = caller_for(world["carol"])
    scoped.save(carol, "user", "param.temperature", 0.3, touchpoint="t.settable")
    assert tp.resolve(config=run_for(world["carol"])).params["temperature"] == 0.3
    with pytest.raises(ValueError, match="at most 1"):
        scoped.save(carol, "user", "param.temperature", 2, touchpoint="t.settable")
    with pytest.raises(scoped.NotPermitted, match="can't be set by a tenant"):
        scoped.save(caller_for(world["alice"]), "tenant", "param.temperature", 0.1, touchpoint="t.settable")


def test_who_may_change_a_scope(tp, world, settings):
    with pytest.raises(scoped.NotPermitted):  # carol isn't a manager of acme
        scoped.save(caller_for(world["carol"]), "tenant", "model", GEMINI, touchpoint="t.settable")
    settings.LLM_KIT = {**settings.LLM_KIT, "SCOPE_PERMISSIONS": {}}
    with pytest.raises(scoped.NotPermitted):  # nobody, until the developer says who
        scoped.save(caller_for(world["alice"]), "tenant", "model", GEMINI, touchpoint="t.settable")
    with pytest.raises(scoped.NotPermitted):  # the platform needs the permission
        scoped.save(caller_for(world["carol"]), "platform", "model", GEMINI, touchpoint="t.settable")
    with pytest.raises(scoped.UnknownSetting):
        scoped.save(caller_for(world["carol"]), "user", "colour", "red")


def test_custom_settings_inherit(tp, world):
    scoped.register(Setting("t.tone", type="choice", choices=["friendly", "formal"], default="friendly"))
    try:
        alice, carol, bob = (caller_for(world[n]) for n in ("alice", "carol", "bob"))
        assert scoped.value("t.tone", carol) == "friendly"  # the default
        scoped.save(alice, "tenant", "t.tone", "formal")
        assert scoped.value("t.tone", carol) == "formal" and scoped.value("t.tone", bob) == "friendly"
        scoped.save(carol, "user", "t.tone", "friendly")
        assert scoped.value("t.tone", carol) == "friendly"
        scoped.clear(carol, "user", "t.tone")
        assert scoped.value("t.tone", carol) == "formal"
    finally:
        scoped.unregister("t.tone")


# ------------------------------------------------------------------ API

URL = "/llm/api/settings/"


def client_for(user):
    c = Client()
    c.force_login(user)
    return c


def test_the_api_describes_sections_for_a_ui(tp, world):
    c = client_for(world["carol"])
    body = c.get(URL, {"scope": "user"}).json()
    assert body["scope"] == "user" and body["editable"] is True
    section = next(s for s in body["sections"] if s["touchpoint"] == "t.settable")
    assert section["description"] == "Answers support questions."
    model, temperature = section["settings"]
    assert (model["key"], model["type"], model["value"], model["inherited"]) == ("model", "model", None, None)
    options = {o["value"] for o in model["options"]}
    assert {CLAUDE, GEMINI} <= options and all(o.split("/")[0] in ("anthropic", "gemini") for o in options)
    assert (temperature["key"], temperature["min"], temperature["max"]) == ("param.temperature", 0, 1)

    tenant = c.get(URL, {"scope": "tenant"}).json()
    assert tenant["editable"] is False  # carol can see acme's settings but not change them
    assert [s["key"] for s in tenant["sections"][0]["settings"]] == ["model"]  # temperature is user-only


def test_the_api_saves_clears_and_refuses(tp, world):
    alice = client_for(world["alice"])
    r = alice.patch(
        URL,
        {"scope": "tenant", "touchpoint": "t.settable", "key": "model", "value": GEMINI},
        content_type="application/json",
    )
    assert r.status_code == 200 and r.json()["value"] == GEMINI

    carol = client_for(world["carol"])
    mine = carol.get(URL, {"scope": "user", "touchpoint": "t.settable"}).json()["sections"][0]["settings"][0]
    assert mine["inherited"] == {"value": GEMINI, "scope": "tenant"}

    def patch(c, **body):
        return c.patch(URL, body, content_type="application/json")

    assert patch(carol, scope="tenant", touchpoint="t.settable", key="model", value=CLAUDE).status_code == 403
    assert patch(carol, scope="user", touchpoint="t.settable", key="model", value="x/y").status_code == 400
    assert patch(carol, scope="user", key="nope", value=1).status_code == 404
    assert patch(carol, scope="galaxy", key="model", value=1).status_code == 400

    r = alice.delete(
        URL, {"scope": "tenant", "touchpoint": "t.settable", "key": "model"}, content_type="application/json"
    )
    assert r.status_code == 200 and r.json()["value"] is None
    assert not ScopedSetting.objects.exists()
    assert Client().get(URL).status_code in (401, 403)
