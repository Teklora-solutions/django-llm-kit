"""Images: tp.image() / edit_image(), and the LiteLLM backend's image generation and editing."""

import asyncio
import base64
import types

import pytest

from llm_kit.core.media import Media
from llm_kit.core.media import backends as media_backends
from llm_kit.core.media import types as media_types
from llm_kit.core.touchpoints import touchpoint
from tests.test_media import CALLS, PNG, env  # noqa: F401  (the fixture comes from test_media)
from tests.test_media import _setup as _media_setup  # noqa: F401

B64 = base64.b64encode(PNG).decode()


def tp(**kw):
    kw.setdefault("default", "openai/gpt-image-1")
    kw.setdefault("fallbacks", ["gemini/imagen-4"])
    return touchpoint("t.images", **kw)


def test_image_sends_only_what_was_given():
    result = tp().image("a lighthouse", n=2, size="1024x1024", env=env(), background="transparent")
    assert len(result.media) == 2 and result.units == {"images": 2, "size": "1024x1024"}
    ((ref, kind, request, params),) = CALLS
    assert kind == "image" and request == {"prompt": "a lighthouse", "n": 2, "size": "1024x1024"}
    assert params == {"background": "transparent"}


def test_edit_image_takes_images_and_a_mask():
    tp().edit_image("make it night", [PNG, "data:image/png;base64," + B64], mask=PNG, env=env())
    ((_, kind, request, _),) = CALLS
    assert kind == "image_edit" and [m.read() for m in request["images"]] == [PNG, PNG]
    assert isinstance(request["mask"], Media)


def test_async_image():
    result = asyncio.run(tp().aimage("x", env=env()))
    assert result.first.read() == PNG


# ------------------------------------------------------------------ LiteLLM


class Response(types.SimpleNamespace):
    pass


def image_response(**extra):
    response = Response(
        data=[
            {"b64_json": B64, "revised_prompt": "A lighthouse at dawn"},
            {"url": "https://images.example/x.png"},
        ],
        output_format="png",
        size="1024x1024",
        usage={"input_tokens": 50, "output_tokens": 4160, "input_tokens_details": {"text_tokens": 50}},
        **extra,
    )
    response._hidden_params = {"response_cost": 0.167}
    return response


@pytest.fixture
def litellm(monkeypatch):
    sent = {}

    def image_generation(**kw):
        sent["image"] = kw
        return image_response()

    async def aimage_generation(**kw):
        sent["aimage"] = kw
        return image_response()

    def image_edit(**kw):
        sent["edit"] = kw
        return image_response()

    fake = types.SimpleNamespace(
        image_generation=image_generation, aimage_generation=aimage_generation, image_edit=image_edit
    )
    monkeypatch.setattr(media_backends, "_litellm", lambda: fake)

    def download(url, size):
        sent.setdefault("downloaded", []).append(url)
        yield PNG

    monkeypatch.setattr(media_types, "_download", download)
    return sent


def test_litellm_image_generation(litellm, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_BACKEND": "litellm"}
    result = tp(fallbacks=[]).image("a lighthouse", n=2, quality="high", env=env(), output_format="png")
    assert litellm["image"] == {
        "model": "openai/gpt-image-1",
        "prompt": "a lighthouse",
        "n": 2,
        "quality": "high",
        "output_format": "png",
        "api_key": "o",
    }
    first, second = result.media
    assert (
        first.read() == PNG
        and first.mime_type == "image/png"
        and first.meta["revised_prompt"] == "A lighthouse at dawn"
    )
    assert second.data == PNG and litellm["downloaded"] == ["https://images.example/x.png"]  # fetched at once
    assert result.revised_prompt == "A lighthouse at dawn" and str(result.cost) == "0.167"
    assert result.units == {"images": 2, "size": "1024x1024", "quality": "high", "output_image_tokens": 4160}
    assert result.input_tokens == 50


def test_urls_can_be_kept(litellm, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_BACKEND": "litellm"}
    result = tp(fallbacks=[]).image("x", env=env(), fetch=False)
    assert result.media[1].url == "https://images.example/x.png" and "downloaded" not in litellm


def test_litellm_image_generation_async(litellm, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_BACKEND": "litellm"}
    result = asyncio.run(tp(fallbacks=[]).aimage("x", size="512x512", env=env()))
    assert litellm["aimage"]["size"] == "512x512" and result.media[1].data == PNG


def test_litellm_image_edit(litellm, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_BACKEND": "litellm"}
    tp(fallbacks=[]).edit_image("night", PNG, mask=PNG, env=env())
    sent = litellm["edit"]
    assert sent["prompt"] == "night" and sent["image"] == ("file.png", PNG, "image/png")
    assert sent["mask"] == ("file.png", PNG, "image/png") and sent["model"] == "openai/gpt-image-1"
