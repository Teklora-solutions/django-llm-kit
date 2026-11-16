"""Generated media: the call path every kind shares (resolution, keys, fallbacks, refusals,
long jobs, records, prices), the Media type, and the catalogue's kinds and unit prices."""

import asyncio
from decimal import Decimal

import pytest

from llm_kit.core.catalogue.fields import is_media_price
from llm_kit.core.catalogue.merge import merge
from llm_kit.core.catalogue.sources import parse_litellm, parse_models_dev
from llm_kit.core.errors import AllModelsFailed, ConfigError
from llm_kit.core.media import (
    DONE,
    FAILED,
    RUNNING,
    ContentRefused,
    JobState,
    Media,
    MediaJobTimeout,
    MediaKind,
    MediaResult,
    UnsafeURL,
    is_refusal,
    register_kind,
)
from llm_kit.core.media import calls as media_calls
from llm_kit.core.observability import recorder
from llm_kit.core.observability.cost import price_media
from llm_kit.core.touchpoints import backends, touchpoint
from llm_kit.core.touchpoints.env import ModelInfo
from tests.test_retry import GEMINI_MINUTE, RateLimitError
from tests.test_touchpoints_core import FakeEnv

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
FAILURES: dict[str, list[BaseException]] = {}
CALLS: list[tuple[str, str, dict, dict]] = []
JOBS: dict[str, list[JobState]] = {}


class Scripted:
    """A media backend that makes a PNG per image, reads text back, and runs scripted jobs."""

    def __init__(self, ref, params, keys):
        self.ref, self.params, self.keys = ref, params, keys

    def generate(self, kind, request, **params):
        CALLS.append((self.ref, kind, dict(request), {**self.params, **params}))
        if FAILURES.get(self.ref):
            raise FAILURES[self.ref].pop(0)
        if kind == "transcription":
            return MediaResult(kind, text="hello there", data={"duration": 12.5})
        n = int(request.get("n") or 1)
        return MediaResult(
            kind, media=[Media(data=PNG, mime_type="image/png") for _ in range(n)], served_model=self.ref
        )

    async def agenerate(self, kind, request, **params):
        return self.generate(kind, request, **params)

    def submit(self, kind, request, **params):
        CALLS.append((self.ref, f"submit:{kind}", dict(request), {**self.params, **params}))
        if FAILURES.get(self.ref):
            raise FAILURES[self.ref].pop(0)
        return f"job-{self.ref}"

    def poll(self, kind, job_id):
        states = JOBS.get(job_id) or [JobState(DONE, result=MediaResult(kind, media=[Media(data=b"mp4")]))]
        return states.pop(0) if len(states) > 1 else states[0]


def backend(ref, params, keys):
    return Scripted(ref, params, keys)


IMAGE = {**{"kind": "image"}}
CATALOGUE = {
    "openai/gpt-image-1": ModelInfo("openai/gpt-image-1", IMAGE),
    "gemini/imagen-4": ModelInfo("gemini/imagen-4", IMAGE),
    "openai/sora-2": ModelInfo("openai/sora-2", {"kind": "video"}),
    "gemini/veo-3": ModelInfo("gemini/veo-3", {"kind": "video"}),
}


def env(**kw):
    return FakeEnv(catalogue=CATALOGUE, **kw)


@pytest.fixture(autouse=True)
def _setup(settings, monkeypatch):
    FAILURES.clear()
    CALLS.clear()
    JOBS.clear()
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "MEDIA_BACKEND": "tests.test_media:backend",
        "OBSERVABILITY": False,
        "MEDIA_POLL_EVERY": 0,
    }
    monkeypatch.setattr(backends, "_sleep", lambda s: None)

    async def asleep(s):
        return None

    monkeypatch.setattr(backends, "_asleep", asleep)
    monkeypatch.setattr(media_calls.time, "sleep", lambda s: None)
    monkeypatch.setattr(recorder, "price_lookup", None)  # the catalogue needs a database


def tp(**kw):
    kw.setdefault("default", "openai/gpt-image-1")
    kw.setdefault("fallbacks", ["gemini/imagen-4"])
    kw.setdefault("requires", {"kind": "image"})
    return touchpoint("t.media", **kw)


# ------------------------------------------------------------------ calls


def test_generate_returns_media_from_the_first_model():
    result = tp(params={"quality": "high"}).generate("image", prompt="a cat", n=2, env=env())
    assert result.kind == "image" and result.model == "openai/gpt-image-1"
    assert len(result.media) == 2 and result.first.read() == PNG and result.first.mime_type == "image/png"
    assert result.units == {"images": 2}
    ((ref, kind, request, params),) = CALLS
    assert (ref, kind, request, params) == (
        "openai/gpt-image-1",
        "image",
        {"prompt": "a cat", "n": 2},
        {"quality": "high"},
    )


def test_each_model_gets_its_own_params():
    point = tp(
        params={"size": "1024x1024"},
        model_params={"gemini/*": {"aspect_ratio": "1:1", "size": None}, "openai/gpt-image-1": {"background": "clear"}},
    )
    FAILURES["openai/gpt-image-1"] = [ConnectionError("down")]
    point.generate("image", prompt="x", env=env())
    assert CALLS[0][3] == {"size": "1024x1024", "background": "clear"}
    assert CALLS[1][3] == {"size": None, "aspect_ratio": "1:1"}


def test_model_params_reach_chat_and_embedding_factories_too():
    chat = touchpoint(
        "t.chatmp", default="openai/gpt-5.5", params={"temperature": 0}, model_params={"openai/*": {"seed": 1}}
    )
    assert chat.params_for("openai/gpt-5.5", {"temperature": 0}) == {"temperature": 0, "seed": 1}
    assert chat.params_for("anthropic/claude-sonnet-5-5", {"temperature": 0}) == {"temperature": 0}


def test_an_outage_falls_back_to_the_next_model():
    FAILURES["openai/gpt-image-1"] = [ConnectionError("down")]
    assert tp().generate("image", prompt="x", env=env()).model == "gemini/imagen-4"


def test_a_refusal_stops_at_the_first_model():
    FAILURES["openai/gpt-image-1"] = [RuntimeError("Error code: 400 - content_policy_violation: rejected")]
    with pytest.raises(ContentRefused):
        tp().generate("image", prompt="x", env=env())
    assert [c[0] for c in CALLS] == ["openai/gpt-image-1"]


def test_a_refusal_may_fall_back_when_the_touchpoint_says_so():
    FAILURES["openai/gpt-image-1"] = [ContentRefused("no")]
    assert tp(fallback_on_refusal=True).generate("image", prompt="x", env=env()).model == "gemini/imagen-4"


def test_a_rate_limit_is_waited_out_then_retried():
    FAILURES["openai/gpt-image-1"] = [RateLimitError(GEMINI_MINUTE)]
    point = tp(retry={"attempts": 2, "max_wait": 60})
    assert point.generate("image", prompt="x", env=env()).model == "openai/gpt-image-1"


def test_every_model_failing():
    FAILURES["openai/gpt-image-1"] = [ConnectionError("a")]
    FAILURES["gemini/imagen-4"] = [ConnectionError("b")]
    with pytest.raises(AllModelsFailed):
        tp().generate("image", prompt="x", env=env())


def test_a_model_without_a_key_is_skipped():
    result = tp().generate("image", prompt="x", env=env(keys={"gemini": ["g"]}))
    assert result.model == "gemini/imagen-4"


def test_unknown_kinds_and_backends():
    with pytest.raises(ConfigError, match="Unknown media kind"):
        tp().generate("hologram", prompt="x", env=env())
    with pytest.raises(ConfigError, match="doesn't make"):
        tp(media_backend="litellm").generate("ocr", document=Media(data=b"%PDF"), env=env(), fallbacks=False)


def test_an_empty_answer_is_a_failure_that_falls_back(monkeypatch):
    original = Scripted.generate

    def empty(self, kind, request, **params):
        if self.ref == "openai/gpt-image-1":
            return MediaResult(kind)
        return original(self, kind, request, **params)

    monkeypatch.setattr(Scripted, "generate", empty)
    assert tp().generate("image", prompt="x", env=env()).model == "gemini/imagen-4"


def test_a_kind_of_your_own():
    register_kind(
        MediaKind("jingle", long=False, measure=lambda request, result: {"output_seconds": request["seconds"]})
    )
    result = tp().generate("jingle", prompt="upbeat", seconds=15, env=env())
    assert result.kind == "jingle" and result.units == {"output_seconds": 15}


def test_async_generate():
    FAILURES["openai/gpt-image-1"] = [ConnectionError("down")]
    result = asyncio.run(tp().agenerate("image", prompt="x", env=env()))
    assert result.model == "gemini/imagen-4"


# ------------------------------------------------------------------ long jobs


def video(**kw):
    kw.setdefault("default", "openai/sora-2")
    kw.setdefault("fallbacks", ["gemini/veo-3"])
    return touchpoint("t.video", **kw)


def test_a_long_job_is_submitted_then_checked_until_done():
    JOBS["job-openai/sora-2"] = [
        JobState(RUNNING, progress=10),
        JobState(RUNNING, progress=60),
        JobState(
            DONE, result=MediaResult("video", media=[Media(data=b"\x00\x00\x00\x18ftypmp42")], data={"seconds": 8})
        ),
    ]
    result = video().generate("video", prompt="waves", seconds=8, env=env())
    assert result.job_id == "job-openai/sora-2" and result.first.mime_type == ""  # as the backend gave it
    assert result.units == {"videos": 1, "output_seconds": 8.0}


def test_a_failed_job_falls_back():
    JOBS["job-openai/sora-2"] = [JobState(FAILED, error="internal error")]
    assert video().generate("video", prompt="x", env=env()).model == "gemini/veo-3"


def test_a_job_refused_by_policy_does_not_fall_back():
    JOBS["job-openai/sora-2"] = [
        JobState(FAILED, error="Your request was blocked by our moderation system: moderation_blocked")
    ]
    with pytest.raises(ContentRefused):
        video().generate("video", prompt="x", env=env())


def test_a_job_that_runs_too_long_is_not_sent_to_the_next_model(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_POLL_EVERY": 1}
    JOBS["job-openai/sora-2"] = [JobState(RUNNING)]
    with pytest.raises(MediaJobTimeout) as caught:
        video().generate("video", prompt="x", env=env(), wait=0.5)
    assert caught.value.job_id == "job-openai/sora-2" and [c[0] for c in CALLS] == ["openai/sora-2"]


def test_a_long_job_async():
    JOBS["job-openai/sora-2"] = [
        JobState(RUNNING),
        JobState(DONE, result=MediaResult("video", media=[Media(data=b"v")])),
    ]
    assert asyncio.run(video().agenerate("video", prompt="x", env=env())).job_id == "job-openai/sora-2"


# ------------------------------------------------------------------ records and prices


def test_calls_are_recorded_with_their_kind_and_units(monkeypatch, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "OBSERVABILITY": True, "CAPTURE_CONTENT": True}
    seen = []
    monkeypatch.setattr(recorder, "emit", seen.append)
    monkeypatch.setattr(recorder, "price_lookup", lambda ref: {"price_units": {"output_cost_per_image": "0.04"}})
    FAILURES["openai/gpt-image-1"] = [ConnectionError("down")]
    result = tp().generate("image", prompt="a cat", n=2, env=env())
    failed, ok = seen
    assert (failed.kind, failed.status, failed.model) == ("image", "error", "openai/gpt-image-1")
    assert (ok.kind, ok.status, ok.fallback_index, ok.units) == ("image", "ok", 1, {"images": 2})
    assert ok.cost == result.cost == Decimal("0.08")
    assert ok.prompt == {"prompt": "a cat", "n": 2}
    assert ok.completion["media"][0] == {"mime_type": "image/png", "bytes": len(PNG)}  # never the bytes


def test_a_cost_the_backend_reports_wins(monkeypatch, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "OBSERVABILITY": True}
    seen = []
    monkeypatch.setattr(recorder, "emit", seen.append)
    original = Scripted.generate

    def priced(self, kind, request, **params):
        result = original(self, kind, request, **params)
        result.cost = 0.123
        return result

    monkeypatch.setattr(Scripted, "generate", priced)
    tp().generate("image", prompt="x", env=env())
    assert seen[0].cost == Decimal("0.123")


def test_refusals_are_recorded_as_refused(monkeypatch, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "OBSERVABILITY": True}
    seen = []
    monkeypatch.setattr(recorder, "emit", seen.append)
    FAILURES["openai/gpt-image-1"] = [ContentRefused("no")]
    with pytest.raises(ContentRefused):
        tp().generate("image", prompt="x", env=env())
    assert seen[0].error_class == "refused"


def test_a_request_budget_counts_media_costs(monkeypatch):
    from llm_kit.core.agents.limits import Tree

    tree = Tree.new(budget=1)
    tree.add_cost(Decimal("0.25"))
    tree.add_cost(None)
    assert tree.spent() == Decimal("0.25")


def test_price_media_by_unit():
    prices = {
        "price_units": {
            "output_cost_per_image": "0.04",
            "output_cost_per_image_2K": "0.08",
            "output_cost_per_second_720p": "0.1",
            "output_cost_per_second": "0.3",
            "input_cost_per_second": "0.0001",
            "input_cost_per_character": "0.000015",
        }
    }
    assert price_media(prices, {"images": 2}) == Decimal("0.08")
    assert price_media(prices, {"images": 1, "size": "2048x2048"}) == Decimal("0.08")
    assert price_media(prices, {"output_seconds": 8, "resolution": "720p"}) == Decimal("0.8")
    assert price_media(prices, {"output_seconds": 8, "resolution": "1080p"}) == Decimal("2.4")
    assert price_media(prices, {"input_seconds": 60}) == Decimal("0.006")
    assert price_media(prices, {"characters": 1000}) == Decimal("0.015")
    assert price_media(
        {"price_units": {"output_cost_per_pixel": "0.00000001"}}, {"images": 1, "size": "1000x1000"}
    ) == Decimal("0.01")
    assert price_media(prices, {"pages": 3}) is None and price_media(None, {"images": 1}) is None


def test_the_recorder_prices_units_and_keeps_a_known_cost():
    from datetime import UTC, datetime

    from llm_kit.core.observability.recorder import Recorder
    from llm_kit.core.observability.records import CallRecord

    r = Recorder()
    r.price_lookup = lambda ref: {"price_units": {"input_cost_per_second": "0.0001"}}
    call = CallRecord(
        call_id="1", run_id=None, started_at=datetime.now(UTC), kind="transcription", units={"input_seconds": 100}
    )
    r._price(call)
    assert call.cost == Decimal("0.01") and not call.unpriced
    known = CallRecord(call_id="2", run_id=None, started_at=datetime.now(UTC), kind="image", cost=Decimal("0.5"))
    r._price(known)
    assert known.cost == Decimal("0.5")


# ------------------------------------------------------------------ Media


def test_media_of_what_you_have(tmp_path):
    assert Media.of(PNG).mime_type == "image/png"
    assert Media.of("data:audio/wav;base64,UklGRg==").read() == b"RIFF"
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"ID3abc")
    from_path = Media.of(path)
    assert from_path.mime_type == "audio/mpeg" and from_path.name == "clip.mp3" and from_path.read() == b"ID3abc"
    with open(path, "rb") as f:
        assert Media.of(f).read() == b"ID3abc"
    assert Media.of({"data": "SUQz", "mime_type": "audio/mpeg"}).read() == b"ID3"
    assert Media.from_path(str(path)).read() == b"ID3abc"
    assert Media.of("https://example.com/a.png").url == "https://example.com/a.png"
    with pytest.raises(Exception, match="Can't use"):
        Media.of(3)


def test_saving_streams_into_storage(settings):
    from django.core.files.storage import InMemoryStorage

    storage = InMemoryStorage()
    name = Media.of(PNG).save(storage)
    assert name.startswith("llm_kit/media/") and name.endswith(".png") and storage.open(name).read() == PNG
    result = MediaResult("image", media=[Media.of(PNG), Media.of(PNG)])
    names = result.save(storage, "covers/cat")
    assert names == ["covers/cat-1.png", "covers/cat-2.png"]
    assert result.to_json()["media"][0]["meta"] == {"saved_as": "covers/cat-1.png"}


def test_private_and_odd_urls_are_not_fetched():
    with pytest.raises(UnsafeURL, match="private"):
        Media.of("http://127.0.0.1/secret").read()
    with pytest.raises(UnsafeURL, match="private"):
        Media.of("http://localhost:8000/admin").read()
    with pytest.raises(UnsafeURL):
        Media(url="file:///etc/passwd").read()


def test_a_string_is_never_read_as_a_path():
    for value in ("/etc/passwd", "../settings.py", {"url": "/etc/passwd"}, {"url": "file:///etc/passwd"}):
        with pytest.raises(UnsafeURL):
            Media.of(value)


def test_the_address_is_checked_when_connecting(monkeypatch):
    """The check is made when the connection is opened, on the address it uses: a name that
    resolves to a private address then (DNS rebinding) is refused, whatever it said before."""
    from llm_kit.core.media import types as media_types

    looked_up = []

    def resolve(host, port, **kw):
        looked_up.append(host)
        return [(2, 1, 6, "", ("10.0.0.5", port))]

    monkeypatch.setattr(media_types.socket, "getaddrinfo", resolve)
    with pytest.raises(UnsafeURL, match="private"):
        Media.of("http://rebind.example/x").read()
    assert looked_up == ["rebind.example"]  # by the connection, not a separate check


def test_media_too_large(settings):
    from llm_kit.core.media import MediaTooLarge

    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_MAX_BYTES": 4}
    with pytest.raises(MediaTooLarge):
        Media(file=__import__("io").BytesIO(b"123456")).read()


def test_refusals_are_recognised():
    assert is_refusal(RuntimeError("Your request was rejected as a result of our safety system"))
    assert is_refusal(type("ContentPolicyViolationError", (Exception,), {})("x"))
    assert not is_refusal(ConnectionError("reset"))


def test_result_extras_read_as_attributes():
    result = MediaResult("transcription", text="hi", data={"language": "en"})
    assert result.language == "en"
    with pytest.raises(AttributeError):
        result.segments  # noqa: B018


# ------------------------------------------------------------------ catalogue


LITELLM_MEDIA = {
    "gpt-image-1": {"litellm_provider": "openai", "mode": "image_generation", "output_cost_per_image_token": 4e-05},
    "gemini/imagen-4": {"litellm_provider": "gemini", "mode": "image_generation", "output_cost_per_image": 0.04},
    "whisper-1": {"litellm_provider": "openai", "mode": "audio_transcription", "input_cost_per_second": 0.0001},
    "tts-1": {"litellm_provider": "openai", "mode": "audio_speech", "input_cost_per_character": 1.5e-05},
    "sora-2": {
        "litellm_provider": "openai",
        "mode": "video_generation",
        "output_cost_per_second": 0.1,
        "output_cost_per_second_720p": 0.1,
        "input_cost_per_token_batches": 1,
    },
    "gpt-image-edit": {"litellm_provider": "openai", "mode": "image_edit"},
    "mistral/mistral-ocr-latest": {"litellm_provider": "mistral", "mode": "ocr"},
    "gpt-realtime": {"litellm_provider": "openai", "mode": "realtime"},
}


def test_the_catalogue_knows_what_each_media_model_does():
    models = parse_litellm(LITELLM_MEDIA).models
    kinds = {key[1]: m["kind"] for key, m in models.items()}
    assert kinds == {
        "gpt-image-1": "image",
        "imagen-4": "image",
        "whisper-1": "transcription",
        "tts-1": "speech",
        "sora-2": "video",
        "gpt-image-edit": "image_edit",
        "mistral-ocr-latest": "ocr",
        "gpt-realtime": "realtime",
    }
    assert models[("gemini", "imagen-4")]["price_units"] == {"output_cost_per_image": "0.04"}
    assert models[("openai", "sora-2")]["price_units"] == {
        "output_cost_per_second": "0.1",
        "output_cost_per_second_720p": "0.1",
    }
    assert models[("openai", "tts-1")]["price_units"] == {"input_cost_per_character": "0.000015"}


def test_media_price_keys():
    assert is_media_price("output_cost_per_image") and is_media_price("output_cost_per_second_1080p")
    assert not is_media_price("input_cost_per_token") and not is_media_price("input_cost_per_token_batches")
    assert not is_media_price("cache_read_input_token_cost")


def test_litellm_decides_the_kind_over_other_sources():
    merged = merge({"models.dev": {"kind": "chat"}, "litellm": {"kind": "speech"}})
    assert merged.values["kind"] == "speech" and merged.provenance["kind"] == "litellm"
    assert merge({"models.dev": {"kind": "chat"}}, {"kind": "image"}).values["kind"] == "image"  # an override wins


def test_models_dev_media_models_get_media_kinds():
    data = {
        "openai": {
            "id": "openai",
            "models": {
                "gpt-image-x": {"modalities": {"input": ["text"], "output": ["image"]}},
                "tts-x": {"modalities": {"input": ["text"], "output": ["audio"]}},
                "omni": {"modalities": {"input": ["text"], "output": ["text", "audio"]}},
            },
        }
    }
    models = parse_models_dev(data).models
    assert [models[("openai", m)]["kind"] for m in ("gpt-image-x", "tts-x", "omni")] == ["image", "speech", "chat"]
