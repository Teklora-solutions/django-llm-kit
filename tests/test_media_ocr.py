"""OCR: tp.ocr(), the LiteLLM backend's ocr, and per-page prices."""

import asyncio
import types
from decimal import Decimal

import pytest

from llm_kit.core.catalogue.fields import is_media_price
from llm_kit.core.catalogue.sources import parse_litellm
from llm_kit.core.media import Media
from llm_kit.core.media import backends as media_backends
from llm_kit.core.observability.cost import price_media
from llm_kit.core.touchpoints import touchpoint
from tests.test_media import CALLS, PNG, env  # noqa: F401
from tests.test_media import _setup as _media_setup  # noqa: F401

PDF = b"%PDF-1.7\n" + b"0" * 32


def reader(**kw):
    kw.setdefault("default", "mistral/mistral-ocr-latest")
    return touchpoint("t.ocr", **kw)


def response(**extra):
    return types.SimpleNamespace(
        pages=[{"index": 0, "markdown": "# Invoice 42"}, {"index": 1, "markdown": "Total: 10 EUR"}],
        usage_info={"pages_processed": 2, "doc_size_bytes": 1234},
        model="mistral-ocr-2505",
        **extra,
    )


@pytest.fixture
def litellm(monkeypatch, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_BACKEND": "litellm"}
    sent = {}

    def ocr(**kw):
        sent["ocr"] = kw
        return response()

    async def aocr(**kw):
        sent["aocr"] = kw
        return response(document_annotation={"total": 10})

    monkeypatch.setattr(media_backends, "_litellm", lambda: types.SimpleNamespace(ocr=ocr, aocr=aocr))
    return sent


def test_ocr_sends_the_document():
    reader().ocr(PDF, pages=[0], env=env(keys={"mistral": ["m"]}))
    ((ref, kind, request, _),) = CALLS
    assert (ref, kind) == ("mistral/mistral-ocr-latest", "ocr") and request["pages"] == [0]
    assert isinstance(request["document"], Media) and request["document"].mime_type == "application/pdf"


def test_litellm_ocr_reads_every_page(litellm):
    result = reader().ocr(PDF, env=env(keys={"mistral": ["m"]}))
    document = litellm["ocr"]["document"]
    assert document["type"] == "document_url" and document["document_url"].startswith("data:application/pdf;base64,")
    assert litellm["ocr"]["model"] == "mistral/mistral-ocr-latest" and litellm["ocr"]["api_key"] == "m"
    assert result.text == "# Invoice 42\n\nTotal: 10 EUR" and len(result.pages) == 2
    assert result.units == {"pages": 2} and result.served_model == "mistral-ocr-2505"


def test_litellm_ocr_of_an_image_url_async(litellm):
    result = asyncio.run(reader().aocr("https://docs.example/scan.png", env=env(keys={"mistral": ["m"]})))
    assert litellm["aocr"]["document"] == {"type": "image_url", "image_url": "https://docs.example/scan.png"}
    assert result.annotation == {"total": 10}


def test_per_page_prices():
    assert is_media_price("ocr_cost_per_page") and is_media_price("annotation_cost_per_page")
    assert not is_media_price("ocr_cost_per_page_batches")
    prices = {"price_units": {"ocr_cost_per_page": "0.004", "annotation_cost_per_page": "0.005"}}
    assert price_media(prices, {"pages": 10, "annotated_pages": 2}) == Decimal("0.05")
    models = parse_litellm(
        {"mistral/mistral-ocr-latest": {"litellm_provider": "mistral", "mode": "ocr", "ocr_cost_per_page": 0.004}}
    ).models
    assert models[("mistral", "mistral-ocr-latest")]["price_units"] == {"ocr_cost_per_page": "0.004"}


@pytest.mark.django_db
def test_a_model_of_your_own_kind_from_a_catalogue_override():
    from llm_kit import catalogue
    from llm_kit.models import CatalogueOverride
    from llm_kit.observability import prices

    CatalogueOverride.objects.create(
        provider="acme",
        model_id="song-2",
        values={"kind": "music", "price_units": {"output_cost_per_second": "0.002"}},
    )
    prices.clear()
    model = catalogue.get("acme/song-2")
    assert model.kind == "music" and model.values["price_units"] == {"output_cost_per_second": "0.002"}
    assert price_media(prices.lookup("acme/song-2"), {"output_seconds": 15}) == Decimal("0.03")
