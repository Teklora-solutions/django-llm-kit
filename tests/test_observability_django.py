import json
import logging

import pytest

from llm_kit import catalogue
from llm_kit.core.observability import flush, recorder
from llm_kit.models import LLMCall, LLMRun
from llm_kit.observability import prices
from llm_kit.observability.setup import build_sinks, configure
from tests.catalogue_payloads import FakeSources

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def recording(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RECORD_SINKS": ["database", "log"]}
    prices.clear()
    configure()
    yield
    flush()
    recorder.configure([], None)
    prices.clear()


async def test_sse_run_is_recorded_and_priced_from_the_catalogue(async_client, user, keys, built, recording):
    from asgiref.sync import sync_to_async

    await sync_to_async(catalogue.sync)(fetch=FakeSources())
    await async_client.aforce_login(user)
    body = {"graph": "catalogue", "message": "price me", "model": "anthropic/claude-sonnet-5-5"}
    r = await async_client.post("/llm/stream/", json.dumps(body), content_type="application/json")
    text = "".join([c.decode() async for c in r.streaming_content])
    thread_id = json.loads(text.split("data: ", 1)[1].split("\n", 1)[0])["id"]
    assert await sync_to_async(flush)()

    run = await LLMRun.objects.aget(thread_id=thread_id)
    assert (run.graph, run.user, run.status, run.calls) == ("catalogue", str(user.pk), "ok", 1)
    call = await LLMCall.objects.aget(run_id=run.run_id)
    assert (call.touchpoint, call.model, call.chosen_by, call.node) == (
        "tests.chat",
        "anthropic/claude-sonnet-5-5",
        "client",
        "model",
    )
    assert call.usage_estimated and call.input_tokens > 0  # the echo fake reports no usage
    assert call.cost is not None and not call.unpriced  # priced from the synced catalogue (2 / 10 per 1M)
    assert run.cost == call.cost
    assert call.streamed and call.time_to_first_token_ms is not None


def test_log_sink_writes_json_without_content(caplog, recording, keys):
    from llm_kit.core.touchpoints import get_touchpoint

    with caplog.at_level(logging.INFO, logger="llm_kit.calls"):
        get_touchpoint("tests.chat").complete("secret prompt text")
        flush()
    lines = [json.loads(r.message) for r in caplog.records if r.name == "llm_kit.calls"]
    kinds = {line["event"] for line in lines}
    assert {"llm_kit.call", "llm_kit.run"} <= kinds
    assert all("secret prompt text" not in json.dumps(line) for line in lines)
    assert LLMCall.objects.filter(touchpoint="tests.chat").count() == 1


def test_admin_pages(client, django_user_model, recording, keys):
    from llm_kit.core.touchpoints import get_touchpoint

    get_touchpoint("tests.chat").complete("hi")
    flush()
    client.force_login(django_user_model.objects.create_superuser("admin", "a@example.com", "pw"))
    call = LLMCall.objects.get()
    for url in [
        "/admin/llm_kit/llmcall/",
        f"/admin/llm_kit/llmcall/{call.pk}/change/",
        "/admin/llm_kit/llmrun/",
        f"/admin/llm_kit/llmrun/{LLMRun.objects.get().pk}/change/",
        "/admin/llm_kit/toolcall/",
    ]:
        assert client.get(url).status_code == 200, url
    page = client.get(f"/admin/llm_kit/llmcall/{call.pk}/change/").content.decode()
    assert "Not captured" in page


def test_unknown_sink_names_and_otel():
    from llm_kit.core.errors import ConfigError

    with pytest.raises(ConfigError):
        build_sinks(["nope.module:Sink"])
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        with pytest.raises(ConfigError, match="opentelemetry"):
            build_sinks(["otel"])
    else:
        assert type(build_sinks(["otel"])[0]).__name__ == "OTelSink"


def test_failing_provider_trips_the_breaker_and_resolution_moves_on(recording, keys, monkeypatch):
    from llm_kit.core.touchpoints import touchpoint
    from llm_kit.monitoring.breaker import breaker

    breaker.reset()
    monkeypatch.setenv("DOWN_API_KEY", "x")
    tp = touchpoint("tests.flaky", default="down/model", fallbacks=["gemini/gemini-3.8-flash"])
    for _ in range(5):
        tp.complete("hi")  # primary fails (ConnectionError), fallback answers
    assert flush()
    assert breaker.is_open("down")
    r = tp.resolve()
    assert r.model == "gemini/gemini-3.8-flash"
    assert ("down/model", "provider circuit open (recent failures)") in r.skipped
    breaker.reset()
