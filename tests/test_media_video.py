"""Video: LiteLLM's job API, background media jobs on BackgroundRun, and provider webhooks."""

import asyncio
import base64
import hashlib
import hmac
import json
import time
import types

import pytest
from asgiref.sync import sync_to_async

from llm_kit import background
from llm_kit.core.errors import BadRequest
from llm_kit.core.media import DONE, FAILED, RUNNING, JobState, Media, MediaResult
from llm_kit.core.media import backends as media_backends
from llm_kit.core.touchpoints import touchpoint
from llm_kit.media import jobs, webhooks
from llm_kit.media.signals import media_job_finished
from llm_kit.models import BackgroundRun
from llm_kit.runtime import access
from tests.test_background import finish, wait_for
from tests.test_media import CALLS, JOBS, PNG, Scripted, env  # noqa: F401
from tests.test_media import _setup as _media_setup  # noqa: F401

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16
CANCELLED: list[str] = []


def video_tp(**kw):
    kw.setdefault("default", "openai/sora-2")
    kw.setdefault("fallbacks", ["gemini/veo-3"])
    return touchpoint("t.video", **kw)


@pytest.fixture(autouse=True)
def _jobs(settings, monkeypatch):
    CANCELLED.clear()
    settings.STORAGES = {
        **settings.STORAGES,
        "llm_test": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
    }
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_POLL_EVERY": 0.05, "MEDIA_STORAGE": "llm_test"}
    monkeypatch.setattr(Scripted, "cancel", lambda self, kind, job_id: CANCELLED.append(job_id), raising=False)


def done(seconds=8):
    return JobState(DONE, result=MediaResult("video", media=[Media(data=MP4)], data={"seconds": seconds}))


# ------------------------------------------------------------------ LiteLLM


@pytest.fixture
def litellm(monkeypatch, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_BACKEND": "litellm"}
    sent = {"status": []}
    statuses = iter(["queued", "in_progress", "completed"])

    def video_generation(**kw):
        sent["generate"] = kw
        return {"id": "video_abc", "status": "queued"}

    def video_status(**kw):
        sent["status"].append(kw)
        status = next(statuses)
        return types.SimpleNamespace(status=status, progress=50, seconds="8", size="1280x720", model="sora-2")

    def video_content(**kw):
        sent["content"] = kw
        return MP4

    async def avideo_generation(**kw):
        return video_generation(**kw)

    async def avideo_status(**kw):
        return video_status(**kw)

    async def avideo_content(**kw):
        return video_content(**kw)

    fake = types.SimpleNamespace(
        video_generation=video_generation,
        video_status=video_status,
        video_content=video_content,
        avideo_generation=avideo_generation,
        avideo_status=avideo_status,
        avideo_content=avideo_content,
        decode_video_id_with_provider=lambda job_id: {"video_id": job_id.removeprefix("enc_")},
    )
    monkeypatch.setattr(media_backends, "_litellm", lambda: fake)
    return sent


def test_litellm_video_is_submitted_polled_and_downloaded(litellm):
    result = video_tp(fallbacks=[]).video("waves", image=PNG, seconds=8, size="1280x720", env=env())
    sent = litellm["generate"]
    assert sent["model"] == "openai/sora-2" and sent["prompt"] == "waves" and sent["seconds"] == "8"
    assert sent["input_reference"].read() == PNG and sent["api_key"] == "o"
    assert [s["video_id"] for s in litellm["status"]] == ["video_abc"] * 3
    assert litellm["status"][0] == {"video_id": "video_abc", "api_key": "o"}  # no generation params
    assert result.job_id == "video_abc" and result.first.read() == MP4 and result.first.mime_type == "video/mp4"
    assert result.units == {"videos": 1, "output_seconds": 8.0, "resolution": "720p", "size": "1280x720"}


def test_status_checks_reach_the_same_endpoint(litellm):
    point = video_tp(fallbacks=[], params={"api_base": "https://proxy.example/v1", "size": "1280x720"})
    point.video("waves", env=env())
    assert litellm["status"][0] == {"video_id": "video_abc", "api_base": "https://proxy.example/v1", "api_key": "o"}
    assert litellm["content"] == {"video_id": "video_abc", "api_base": "https://proxy.example/v1", "api_key": "o"}


def test_litellm_video_async(litellm):
    result = asyncio.run(video_tp(fallbacks=[]).avideo("waves", env=env()))
    assert result.first.read() == MP4


def test_video_states():
    state = media_backends.video_state({"status": "failed", "error": {"message": "moderation_blocked"}})
    assert state.status == FAILED and state.error == "moderation_blocked"
    assert media_backends.video_state({"status": "in_progress", "progress": 40}).progress == 40
    assert media_backends.video_state({"status": "queued"}).status == "pending"
    assert media_backends.LiteLLMMedia.provider_job_id("plain") in ("plain",)


# ------------------------------------------------------------------ background jobs


pytestmark = pytest.mark.django_db(transaction=True)


def test_a_background_video_job_is_saved_and_announced(user):
    JOBS["job-openai/sora-2"] = [JobState(RUNNING, progress=10), JobState(RUNNING, progress=70), done()]
    heard = []

    def receiver(sender, run, status, result, error, **kw):
        heard.append((status, result["media"][0]["name"] if result else None))

    media_job_finished.connect(receiver)
    try:
        run = video_tp().start_video("waves", image=PNG, seconds=8, user=user)
        assert run.kind == "media" and run.thread_id is None and run.payload["kind"] == "video"
        assert run.payload["request"]["image"]["__media__"]["data"] == base64.b64encode(PNG).decode()
        run = finish(run)
    finally:
        media_job_finished.disconnect(receiver)
    assert run.status == "done", run.error
    (entry,) = run.result["media"]
    from django.core.files.storage import storages

    assert (
        entry["name"] == f"llm_kit/media/jobs/{run.run_id}/1.mp4"
        and storages["llm_test"].open(entry["name"]).read() == MP4
    )
    assert run.result["job"] == {
        "id": "job-openai/sora-2",
        "provider_id": "job-openai/sora-2",
        "model": "openai/sora-2",
    }
    assert run.result["units"] == {"videos": 1, "output_seconds": 8.0}
    assert heard == [("done", entry["name"])]
    assert CALLS[0][1] == "submit:video" and isinstance(CALLS[0][2]["image"], Media)


async def test_watching_a_media_job(user):
    JOBS["job-openai/sora-2"] = [JobState(RUNNING, progress=50), done()]
    run = await sync_to_async(video_tp().start_video)("waves", user=user)
    await sync_to_async(finish)(run)
    events = [e async for e in background.watch(run, after=0)]
    names = [e["event"] for e in events]
    assert names[0] == "status" and names[-1] == "status" and events[-1]["data"]["status"] == "done"
    progress = [e["data"] for e in events if e["event"] == "progress"]
    assert progress[0] == {"status": "submitted", "model": "openai/sora-2", "job": "job-openai/sora-2"}
    assert {"status": "running", "progress": 50, "model": "openai/sora-2"} in progress
    (media,) = [e["data"] for e in events if e["event"] == "media"]
    assert media["media"][0]["mime_type"] == "video/mp4" and media["media"][0]["name"].endswith("/1.mp4")
    snapshot = [e async for e in background.watch(run)][0]
    assert snapshot["event"] == "snapshot" and snapshot["data"]["state"] is None


def test_cancelling_a_media_job_cancels_it_at_the_provider(user):
    JOBS["job-openai/sora-2"] = [JobState(RUNNING)]
    run = video_tp().start_video("waves", user=user)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run.refresh_from_db()
        if (run.result or {}).get("job"):
            break
        time.sleep(0.02)
    assert run.cancel()
    run = finish(run)
    assert run.status == "cancelled" and CANCELLED == ["job-openai/sora-2"]
    assert run.result["job"]["id"] == "job-openai/sora-2"


def test_a_job_that_waits_too_long_ends_in_error_with_its_id(user):
    JOBS["job-openai/sora-2"] = [JobState(RUNNING)]
    run = finish(video_tp().start_video("waves", user=user, wait=0.2))
    assert run.status == "error" and "did not finish in time" in run.error
    assert run.result == {"job": {"id": "job-openai/sora-2", "model": "openai/sora-2"}}


def test_a_failing_job_tries_the_next_model(user, keys):
    JOBS["job-openai/sora-2"] = [JobState(FAILED, error="internal")]
    run = finish(video_tp().start_video("waves", user=user))
    assert run.status == "done" and run.result["model"] == "gemini/veo-3"


def test_only_its_owner_sees_a_media_job(user, other_user):
    JOBS["job-openai/sora-2"] = [done()]
    run = finish(video_tp().start_video("waves", user=user))
    assert background.get_run(run.pk, user).pk == run.pk
    with pytest.raises(access.NotAllowed):
        background.get_run(run.pk, other_user)
    assert list(background.runs_for(user)) == [run] and list(background.runs_for(other_user)) == []


def test_media_jobs_in_the_rest_api(client, user):
    JOBS["job-openai/sora-2"] = [done()]
    run = finish(video_tp().start_video("waves", user=user))
    client.force_login(user)
    listed = client.get("/llm/api/runs/?kind=media").json()
    assert [r["id"] for r in listed] == [run.run_id] and listed[0]["kind"] == "media"
    assert client.get("/llm/api/runs/?kind=graph").json() == []
    assert client.get("/llm/api/runs/?kind=x").status_code == 400


def test_starting_refuses_what_cannot_run(user):
    with pytest.raises(BadRequest):
        video_tp().start("video", user=user, prompt="x", params={"bad": object()})
    assert not BackgroundRun.objects.exists()


def test_the_request_is_kept_and_restored():
    packed = jobs.pack({"prompt": "x", "image": Media.of(PNG), "refs": [Media.of("https://a.example/b.png")]})
    restored = jobs.unpack(json.loads(json.dumps(packed)))
    assert restored["image"].read() == PNG and restored["refs"][0].url == "https://a.example/b.png"


# ------------------------------------------------------------------ webhooks


SECRET = "whsec_" + base64.b64encode(b"0123456789abcdef0123456789abcdef").decode()


def signed(body: bytes, secret: str = SECRET, msg_id: str = "msg_1", stamp: int | None = None) -> dict:
    stamp = int(time.time()) if stamp is None else stamp
    key = base64.b64decode(secret.removeprefix("whsec_"))
    signature = base64.b64encode(hmac.new(key, f"{msg_id}.{stamp}.".encode() + body, hashlib.sha256).digest()).decode()
    return {
        "HTTP_WEBHOOK_ID": msg_id,
        "HTTP_WEBHOOK_TIMESTAMP": str(stamp),
        "HTTP_WEBHOOK_SIGNATURE": f"v1,{signature}",
    }


def test_a_webhook_wakes_its_job_at_once(client, user, settings):
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "MEDIA_WEBHOOKS": {"openai": {"secret": SECRET}},
        "MEDIA_WEBHOOK_POLL_EVERY": 60,
    }
    JOBS["job-openai/sora-2"] = [JobState(RUNNING)]
    run = video_tp(fallbacks=[]).start_video("waves", user=user)
    wait_for(run, "running")
    deadline = time.monotonic() + 5
    while not (run.result or {}).get("job") and time.monotonic() < deadline:
        time.sleep(0.02)
        run.refresh_from_db()
    JOBS["job-openai/sora-2"] = [done()]  # finished at the provider; the job checks only every 60 s
    body = json.dumps({"type": "video.completed", "data": {"id": "job-openai/sora-2"}}).encode()
    response = client.post("/llm/media/webhooks/openai/", body, content_type="application/json", **signed(body))
    assert response.status_code == 204
    run = finish(run, timeout=5)
    assert run.status == "done" and run.notified_at is not None


def test_webhooks_that_fail_their_check_change_nothing(client, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_WEBHOOKS": {"openai": {"secret": SECRET}}}
    body = b'{"data": {"id": "x"}}'
    post = client.post
    assert post("/llm/media/webhooks/openai/", body, content_type="application/json").status_code == 401
    forged = signed(body, secret="whsec_" + base64.b64encode(b"another-secret-of-32-bytes-long!").decode())
    assert post("/llm/media/webhooks/openai/", body, content_type="application/json", **forged).status_code == 401
    stale = signed(body, stamp=int(time.time()) - 3600)
    assert post("/llm/media/webhooks/openai/", body, content_type="application/json", **stale).status_code == 401
    assert post("/llm/media/webhooks/replicate/", body, content_type="application/json").status_code == 404
    ok = post("/llm/media/webhooks/openai/", body, content_type="application/json", **signed(body))
    assert ok.status_code == 202  # checked, but no running job has that id
    assert client.get("/llm/media/webhooks/openai/").status_code == 405


def test_hmac_and_your_own_checks(rf, settings):
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "MEDIA_WEBHOOKS": {"acme": {"verify": "hmac", "header": "X-Acme-Signature", "secret": "s3", "id": "job.id"}},
    }
    body = b'{"job": {"id": "j1"}}'
    digest = hmac.new(b"s3", body, hashlib.sha256).hexdigest()
    good = rf.post("/", body, content_type="application/json", HTTP_X_ACME_SIGNATURE=f"sha256={digest}")
    config = webhooks.config_for("acme")
    assert webhooks.verified(good, config) and webhooks.job_id_in(json.loads(body), config["id"]) == "j1"
    bad = rf.post("/", body, content_type="application/json", HTTP_X_ACME_SIGNATURE="sha256=00")
    assert not webhooks.verified(bad, config)


def test_fal_signatures(rf):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    body = b'{"request_id": "r1", "status": "OK"}'
    stamp = str(int(time.time()))
    message = "\n".join(["r1", "u1", stamp, hashlib.sha256(body).hexdigest()]).encode()
    headers = {
        "HTTP_X_FAL_WEBHOOK_REQUEST_ID": "r1",
        "HTTP_X_FAL_WEBHOOK_USER_ID": "u1",
        "HTTP_X_FAL_WEBHOOK_TIMESTAMP": stamp,
        "HTTP_X_FAL_WEBHOOK_SIGNATURE": key.sign(message).hex(),
    }
    config = {**webhooks.DEFAULTS["fal_ai"], "public_keys": [public]}
    assert webhooks.verified(rf.post("/", body, content_type="application/json", **headers), config)
    tampered = rf.post("/", body + b" ", content_type="application/json", **headers)
    assert not webhooks.verified(tampered, config)


def test_the_webhook_url_goes_with_the_job_when_the_provider_takes_one(settings):
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "MEDIA_WEBHOOKS": {"replicate": {"secret": SECRET, "param": "webhook"}, "openai": {"secret": SECRET}},
        "MEDIA_WEBHOOK_URL": "https://app.example/llm/media/webhooks/{provider}/",
    }
    assert webhooks.submit_params("replicate/minimax/video-01") == {
        "webhook": "https://app.example/llm/media/webhooks/replicate/"
    }
    assert webhooks.submit_params("openai/sora-2") == {}  # set in OpenAI's dashboard
    assert webhooks.submit_params("gemini/veo-3") == {}  # webhooks off


def test_webhook_settings_that_cannot_work_are_reported(settings):
    from llm_kit.background.checks import media_webhooks_can_arrive as check

    assert check() == []
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_WEBHOOKS": {"replicate": {"param": "webhook"}}}
    assert [m.id for m in check()] == ["llm_kit.E003", "llm_kit.W004"]
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "MEDIA_WEBHOOKS": {"replicate": {"secret": SECRET, "param": "webhook"}},
        "MEDIA_WEBHOOK_URL": "https://app.example/llm/media/webhooks/{provider}/",
    }
    assert check() == []
