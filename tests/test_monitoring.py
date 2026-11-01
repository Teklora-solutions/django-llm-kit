import json
import threading
import uuid
from datetime import timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO

import pytest
from django.core.management import CommandError, call_command
from django.utils import timezone
from rest_framework.test import APIClient

from llm_kit import monitoring
from llm_kit.models import AlertEvent, AlertRule, CatalogueChange, LLMCall, LLMRun
from llm_kit.monitoring.alerts import alert_fired, check, check_all
from llm_kit.monitoring.breaker import breaker

pytestmark = pytest.mark.django_db


def call(minutes_ago=5, **kw):
    defaults = dict(
        call_id=uuid.uuid4(),
        started_at=timezone.now() - timedelta(minutes=minutes_ago),
        touchpoint="chat",
        model="anthropic/claude-sonnet-5-5",
        provider="anthropic",
        chosen_by="code",
        input_tokens=100,
        output_tokens=10,
        cost=Decimal("0.001"),
        latency_ms=100,
        status="ok",
    )
    return LLMCall.objects.create(**{**defaults, **kw})


@pytest.fixture(autouse=True)
def _clean_breaker():
    breaker.reset()
    yield
    breaker.reset()


@pytest.fixture
def data():
    for i in range(18):
        call(latency_ms=100 + i * 10, time_to_first_token_ms=20, user="u1")
    call(model="gemini/gemini-3.8-flash", provider="gemini", fallback_index=1, cost=Decimal("0.0005"), user="u2")
    call(status="error", error_class="rate_limit", cost=None, latency_ms=5, user="u2")
    call(touchpoint="summarise", model="openai/gpt-5.5", provider="openai", cost=Decimal("0.01"), user="u3")
    call(minutes_ago=60 * 30, touchpoint="old", cost=Decimal("5"))  # outside 24h
    call(touchpoint="chat", unpriced=True, cost=None, model="mystery/x", provider="mystery")
    LLMRun.objects.create(run_id=uuid.uuid4(), name="LangGraph", started_at=timezone.now(), tool_calls=3, calls=4)


def test_summary(data):
    s = monitoring.summary(period="24h")
    assert s["calls"] == 22 and s["errors"] == 1 and s["error_rate"] == round(1 / 22, 4)
    assert s["cost"] == Decimal("0.0285")  # 18*0.001 + 0.0005 + 0.01
    assert s["fallbacks"] == 1 and s["unpriced"] == 1
    # 21 successful calls: four at 100 ms, then 110..270 (nearest-rank percentiles)
    assert s["latency_ms"]["p50"] == 170 and s["latency_ms"]["p95"] == 260
    assert s["time_to_first_token_ms"]["p50"] == 20
    assert (s["runs"], s["tool_calls"]) == (1, 3)
    assert monitoring.summary(period="7d")["calls"] == 23
    assert monitoring.summary(period="24h", touchpoint="summarise")["cost"] == Decimal("0.01")


def test_filters_validate():
    with pytest.raises(ValueError, match="period"):
        monitoring.summary(period="yesterday")
    with pytest.raises(ValueError, match="Unknown filter"):
        monitoring.summary(color="red")
    with pytest.raises(ValueError, match="before"):
        monitoring.summary(since="2026-02-01T00:00:00", until="2026-01-01T00:00:00")


def test_timeseries_breakdown_errors_top(data):
    ts = monitoring.timeseries("calls", bucket="day", period="7d", group_by="touchpoint")
    by_group = {}
    for p in ts["points"]:
        by_group[p["group"]] = by_group.get(p["group"], 0) + p["value"]
    assert by_group == {"chat": 21, "summarise": 1, "old": 1}

    b = monitoring.breakdown("model", period="24h")
    assert b["rows"][0]["key"] == "anthropic/claude-sonnet-5-5"  # most expensive first
    assert b["rows"][0]["calls"] == 19 and b["rows"][0]["error_rate"] == round(1 / 19, 4)
    assert b["rows"][0]["latency_ms"]["p95"] == 270

    e = monitoring.errors(period="24h")
    assert e["by_class"] == [{"error_class": "rate_limit", "count": 1}] and len(e["recent"]) == 1

    t = monitoring.top("user", metric="cost", period="24h")
    assert [r["key"] for r in t["rows"]] == ["u1", "u3", "u2"]
    with pytest.raises(ValueError):
        monitoring.timeseries("vibes")


def test_provider_health_includes_circuit_state(data):
    for _ in range(5):
        breaker.record("gemini", False, "overloaded")
    rows = {r["provider"]: r for r in monitoring.provider_health(period="24h")["providers"]}
    assert rows["anthropic"]["calls"] == 19 and rows["anthropic"]["retryable"] == 1
    assert rows["gemini"]["circuit"]["state"] == "open" and rows["anthropic"]["circuit"]["state"] == "closed"


def test_api(data, django_user_model):
    api = APIClient()
    api.force_authenticate(django_user_model.objects.create_user("plain"))
    assert api.get("/llm/api/monitoring/summary/").status_code == 403  # staff only by default
    api.force_authenticate(django_user_model.objects.create_user("staff", is_staff=True))
    r = api.get("/llm/api/monitoring/summary/?period=24h&touchpoint=chat")
    assert r.status_code == 200 and r.data["calls"] == 21
    assert api.get("/llm/api/monitoring/timeseries/?metric=cost&bucket=hour&group_by=model").status_code == 200
    assert api.get("/llm/api/monitoring/breakdown/?by=provider&limit=2").data["rows"][0]["key"] == "anthropic"
    assert api.get("/llm/api/monitoring/top/?by=user&n=1").data["rows"][0]["key"] == "u1"
    assert api.get("/llm/api/monitoring/errors/").status_code == 200
    assert api.get("/llm/api/monitoring/providers/?period=1h").status_code == 200
    assert api.get("/llm/api/monitoring/alerts/").status_code == 200
    assert api.get("/llm/api/monitoring/summary/?period=nope").status_code == 400
    assert api.get("/llm/api/monitoring/summary/?colour=red").status_code == 400
    assert api.get("/llm/api/monitoring/top/?n=x").status_code == 400


# ---------------------------------------------------------------- alerts


class Hook(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        Hook.received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def webhook():
    Hook.received = []
    server = HTTPServer(("127.0.0.1", 0), Hook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/hook", Hook.received
    server.shutdown()


def test_spend_alert_fires_once_and_delivers(data, mailoutbox, webhook):
    url, received = webhook
    seen = []
    alert_fired.connect(lambda **kw: seen.append(kw["event"]), weak=False, dispatch_uid="test-alert")
    rule = AlertRule.objects.create(
        name="Spend",
        kind="spend",
        threshold=Decimal("0.02"),
        window_minutes=60,
        emails=["ops@example.com"],
        webhook_url=url,
    )
    event = check(rule)
    assert event is not None and "Spend $0.0285" in event.message
    assert event.delivered == {"email": "ok", "webhook": "ok"}
    assert mailoutbox[0].subject == "[llm-kit] Spend" and received[0]["kind"] == "spend"
    assert seen == [event]
    assert check(rule) is None  # cooling down
    assert AlertEvent.objects.count() == 1
    alert_fired.disconnect(dispatch_uid="test-alert")


def test_rate_alerts_respect_min_calls(data):
    rule = AlertRule.objects.create(name="Errors", kind="error_rate", threshold=Decimal("0.01"), min_calls=50)
    assert check(rule) is None
    rule.min_calls = 10
    assert "Error rate 4.5%" in check(rule).message
    p95 = AlertRule.objects.create(name="Slow", kind="latency_p95", threshold=200, min_calls=1)
    assert "p95 latency 260 ms" in check(p95).message


def test_provider_down_and_catalogue_change_alerts(keys):
    for _ in range(5):
        breaker.record("gemini", False, "rate_limit")
    rule = AlertRule.objects.create(name="Down", kind="provider_down", filters={"provider": "gemini"})
    assert "gemini" in check(rule).message

    rule = AlertRule.objects.create(name="Catalogue", kind="catalogue_change", window_minutes=60)
    CatalogueChange.objects.create(
        synced_at=timezone.now(),
        provider="anthropic",
        model_id="claude-sonnet-5-5",
        kind="price_changed",
        field="price_input",
        old="2",
        new="3",
    )
    CatalogueChange.objects.create(synced_at=timezone.now(), provider="nobody", model_id="unused", kind="removed")
    event = check(rule)
    assert "anthropic/claude-sonnet-5-5 price_changed price_input: 2 → 3 (used by tests.chat)" in event.message
    assert "unused" not in event.message
    assert len(check_all()) == 0  # both cooling down


# ----------------------------------------------------------- operations


def test_status_command(data, keys):
    out = StringIO()
    with pytest.raises(CommandError, match="issue"):
        call_command("llm_kit_status", stdout=out)  # catalogue never synced
    assert "catalogue was never synced" in out.getvalue() and "calls 22" in out.getvalue()
    call_command("llm_kit_status", "--no-fail", stdout=StringIO())


def test_metrics_endpoint(client, data, settings, django_user_model):
    assert client.get("/llm/metrics/").status_code == 403
    settings.LLM_KIT = {**settings.LLM_KIT, "METRICS_TOKEN": "s3cret"}
    assert client.get("/llm/metrics/", HTTP_AUTHORIZATION="Bearer wrong").status_code == 403
    body = client.get("/llm/metrics/", HTTP_AUTHORIZATION="Bearer s3cret").content.decode()
    assert "# TYPE llm_kit_calls_1h gauge" in body
    assert 'llm_kit_calls_1h{touchpoint="chat",model="anthropic/claude-sonnet-5-5",provider="anthropic"} 19' in body
    assert 'llm_kit_latency_ms_1h{quantile="95"} 260' in body
    client.force_login(django_user_model.objects.create_user("s", is_staff=True))
    assert client.get("/llm/metrics/").status_code == 200


def test_prune(data, keys):
    old = call(minutes_ago=60 * 24 * 100)
    captured = call(
        minutes_ago=60 * 24 * 10, prompt=[{"type": "human", "content": "secret"}], completion={"content": "x"}
    )
    out = StringIO()
    call_command("llm_kit_prune", "--dry-run", stdout=out)
    assert "would delete 1 calls" in out.getvalue() and LLMCall.objects.filter(pk=old.pk).exists()
    call_command("llm_kit_prune", stdout=StringIO())
    assert not LLMCall.objects.filter(pk=old.pk).exists()
    captured.refresh_from_db()
    assert captured.prompt is None and captured.completion is None


def test_admin_dashboard(client, data, django_user_model, keys):
    client.force_login(django_user_model.objects.create_superuser("admin", "a@example.com", "pw"))
    for _ in range(5):
        breaker.record("gemini", False, "overloaded")
    for period in ("24h", "7d", "30d", "bogus"):
        r = client.get(f"/admin/llm_kit/monitoringdashboard/?period={period}")
        assert r.status_code == 200, period
    page = client.get("/admin/llm_kit/monitoringdashboard/?period=24h").content.decode()
    assert "$0.0285" in page and "4.5%" in page and ">22<" in page  # spend, error rate, calls
    assert "Open · retry in" in page and "Spend per day" in page and '<path class="bar"' in page
    assert "Needs attention" in page  # catalogue never synced, circuit open
    assert client.get("/admin/llm_kit/alertrule/add/").status_code == 200
