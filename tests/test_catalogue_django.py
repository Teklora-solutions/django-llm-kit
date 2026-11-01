from decimal import Decimal
from io import StringIO

import pytest
from django.core.management import call_command

from llm_kit import catalogue
from llm_kit.core.catalogue.sources import LITELLM
from llm_kit.models import CatalogueChange, CatalogueModel, CatalogueOverride, CatalogueProvider, CatalogueSource
from tests.catalogue_payloads import FakeSources

pytestmark = pytest.mark.django_db


@pytest.fixture
def fetch():
    return FakeSources()


@pytest.fixture
def synced(fetch):
    return catalogue.sync(fetch=fetch)


def test_sync_stores_models_providers_sources(synced):
    assert synced.baseline and CatalogueModel.objects.count() == 9
    m = CatalogueModel.objects.get(provider="anthropic", model_id="claude-sonnet-5-5")
    assert m.price_input == Decimal("2") and m.price_output == Decimal("10")
    assert m.context_window == 1000000 and m.tool_call is True and m.kind == "chat"
    assert m.input_modality_list == ["text", "image", "pdf"]
    assert m.provenance["price_input"] == "litellm" and "price_input" in m.conflicts
    assert m.ref == m.litellm_id == "anthropic/claude-sonnet-5-5"
    assert CatalogueProvider.objects.get(provider="gemini").env == ["GOOGLE_API_KEY", "GEMINI_API_KEY"]
    assert set(CatalogueSource.objects.values_list("name", "last_status")) == {
        ("models.dev", "fetched"),
        ("openrouter", "fetched"),
        ("openrouter-embeddings", "fetched"),
        ("litellm", "fetched"),
    }
    assert CatalogueChange.objects.count() == 0  # baseline
    assert not catalogue.is_stale()


def test_second_sync_records_changes(fetch, synced):
    fetch.data[LITELLM.url]["claude-sonnet-5-5"]["output_cost_per_token"] = 1.2e-05
    fetch.bump(LITELLM.url)
    report = catalogue.sync(fetch=fetch)
    assert report.count("price_changed") == 1
    ch = CatalogueChange.objects.get()
    assert (ch.kind, ch.model_id, ch.field, ch.old, ch.new) == (
        "price_changed",
        "claude-sonnet-5-5",
        "price_output",
        "10",
        "12",
    )
    assert CatalogueModel.objects.get(model_id="claude-sonnet-5-5", provider="anthropic").price_output == Decimal("12")
    assert list(catalogue.recent_changes()) == [ch]


def test_get_resolves_reference_styles(synced):
    assert catalogue.get("anthropic/claude-sonnet-5-5").provider == "anthropic"
    assert catalogue.get("anthropic:claude-sonnet-5-5").provider == "anthropic"  # init_chat_model style
    assert catalogue.get("google/gemini-3.8-flash").provider == "gemini"  # alias normalised
    assert catalogue.get("openrouter/google/gemini-3.8-flash").price_input == Decimal("0.825")
    # Bare id served by both anthropic and vertex_ai: the first-party provider wins.
    assert catalogue.get("claude-sonnet-5-5").provider == "anthropic"
    assert catalogue.get("us.anthropic.claude-sonnet-5-5-v1:0").provider == "bedrock"  # unique bare id
    assert catalogue.get("nope/nothing") is None
    assert catalogue.prices("gemini/gemini-3.8-flash")["input"] == Decimal("0.75")


def test_search(synced, monkeypatch):
    refs = lambda qs: sorted(m.ref for m in qs)  # noqa: E731
    assert refs(catalogue.search(requires={"tool_call": True, "input": ["image"]}, max_price_input=1)) == [
        "gemini/gemini-3.8-flash",
        "openrouter/google/gemini-3.8-flash",
    ]
    assert refs(catalogue.search(kind="embedding")) == [
        "openai/text-embedding-3-large",
        "openrouter/openai/text-embedding-3-large",
    ]
    assert refs(catalogue.search(requires={"min_context": 1048576}, provider="google")) == ["gemini/gemini-3.8-flash"]
    with pytest.raises(ValueError):
        catalogue.search(requires={"vision": True})

    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert not catalogue.has_credentials("gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert catalogue.has_credentials("google")
    assert refs(catalogue.search(requires={"tool_call": True}, usable_only=True)) == ["gemini/gemini-3.8-flash"]


def test_overrides_apply_immediately_and_survive_sync(fetch, synced):
    o = CatalogueOverride.objects.create(
        provider="anthropic", model_id="claude-sonnet-5-5", values={"price_input": "1.8"}
    )
    m = CatalogueModel.objects.get(provider="anthropic", model_id="claude-sonnet-5-5")
    assert m.price_input == Decimal("1.8") and m.provenance["price_input"] == "override"

    fetch.bump(LITELLM.url)
    catalogue.sync(fetch=fetch, force=True)
    assert CatalogueModel.objects.get(pk=m.pk).price_input == Decimal("1.8")

    o.values = {"hidden": True}
    o.save()
    assert catalogue.get("anthropic/claude-sonnet-5-5").hidden
    assert "anthropic/claude-sonnet-5-5" not in {x.ref for x in catalogue.search()}

    o.delete()
    m = CatalogueModel.objects.get(pk=m.pk)
    assert m.price_input == Decimal("2") and not m.hidden


def test_private_model_via_override(synced):
    CatalogueOverride.objects.create(
        provider="openai", model_id="ft:gpt-5:acme", values={"kind": "chat", "price_input": "6", "tool_call": True}
    )
    m = catalogue.get("openai/ft:gpt-5:acme")
    assert m is not None and m.price_input == Decimal("6") and m.source_values == {}
    assert catalogue.get("ft:gpt-5:acme") == m  # bare id containing ":" is not split into a provider
    CatalogueOverride.objects.filter(model_id="ft:gpt-5:acme").delete()
    assert catalogue.get("openai/ft:gpt-5:acme") is None  # removed with its override


def test_override_rejects_unknown_fields():
    from django.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        CatalogueOverride(provider="x", model_id="y", values={"vision": True}).full_clean()


def test_management_command_offline_uses_bundled_snapshot():
    out = StringIO()
    call_command("llm_kit_sync_catalogue", "--offline", stdout=out)
    assert "first sync (baseline" in out.getvalue()
    assert CatalogueModel.objects.filter(removed=False).count() > 5000
    assert set(CatalogueSource.objects.values_list("last_status", flat=True)) == {"snapshot"}


def test_management_command_source_filter(monkeypatch):
    out, err = StringIO(), StringIO()
    call_command("llm_kit_sync_catalogue", "--source", "bogus", stdout=out, stderr=err)
    assert "Unknown source(s): bogus" in err.getvalue()


def test_task_function(monkeypatch, fetch):
    from llm_kit import tasks
    from llm_kit.core.catalogue import sync as core_sync

    monkeypatch.setattr(catalogue, "sync", lambda force=False: core_sync(catalogue.DjangoCatalogueStore(), fetch=fetch))
    assert "active models" in tasks.run_catalogue_sync()  # via the Celery-autodiscover module


def test_admin_pages_render(client, django_user_model, synced):
    admin = django_user_model.objects.create_superuser("admin", "a@example.com", "pw")
    client.force_login(admin)
    m = CatalogueModel.objects.get(provider="anthropic", model_id="claude-sonnet-5-5")
    for url in [
        "/admin/llm_kit/cataloguemodel/",
        "/admin/llm_kit/cataloguemodel/?kind__exact=embedding",
        f"/admin/llm_kit/cataloguemodel/{m.pk}/change/",
        "/admin/llm_kit/cataloguechange/",
        "/admin/llm_kit/cataloguesource/",
        "/admin/llm_kit/catalogueprovider/",
        "/admin/llm_kit/catalogueoverride/add/",
    ]:
        r = client.get(url)
        assert r.status_code == 200, url
    page = client.get(f"/admin/llm_kit/cataloguemodel/{m.pk}/change/").content.decode()
    assert "litellm" in page and "Source disagreements" in page

    r = client.post("/admin/llm_kit/cataloguemodel/", {"action": "hide", "_selected_action": [m.pk]}, follow=True)
    assert r.status_code == 200
    assert CatalogueModel.objects.get(pk=m.pk).hidden


def test_top_level_exports():
    import llm_kit

    assert callable(llm_kit.model_from_config) and callable(llm_kit.get_chat_model)
    assert callable(llm_kit.get_graph) and llm_kit.DjangoSaver.__name__ == "DjangoSaver"
