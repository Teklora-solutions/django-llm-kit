"""Providers telling you a media job changed, instead of being asked every few seconds.

Checking a job's status works with every provider and is always on. A webhook only makes it
faster and cheaper: when one arrives, the job checks the provider at once, and while webhooks are
on for a provider its jobs check rarely (``MEDIA_WEBHOOK_POLL_EVERY``) in case one never comes.
What a webhook says is never trusted as the result: the job asks the provider.

Route: ``POST llm/media/webhooks/<provider>/`` (in ``llm_kit.transport.sse.urls``). Turn it on
per provider (LiteLLM's provider names)::

    LLM_KIT = {
        "MEDIA_WEBHOOKS": {
            "replicate": {"secret": "whsec_..."},                   # Standard Webhooks
            "openai": {"secret": "whsec_..."},                      # set the URL in OpenAI's dashboard
            "fal_ai": {"param": "webhook_url"},                     # only with a backend of your own (below)
            "acme": {"verify": "hmac", "header": "X-Acme-Signature", "secret": "...", "id": "job.id"},
            "other": {"verify": "myapp.hooks:verify", "id": "data.job"},
        },
        "MEDIA_WEBHOOK_URL": "https://app.example.com/llm/media/webhooks/{provider}/",
    }

Each provider's entry:

* ``verify``: ``"standard"`` (Standard Webhooks: ``webhook-id``, ``webhook-timestamp``,
  ``webhook-signature`` headers, a ``whsec_`` secret: Replicate, OpenAI), ``"hmac"`` (an
  HMAC-SHA256 of the body in ``header``, hex or base64, ``sha256=`` allowed), ``"fal"`` (fal's
  ED25519 signatures, checked with its published keys; needs ``cryptography``), or a dotted
  path to ``fn(request, config) -> bool``. Replicate, OpenAI and fal have theirs by default.
* ``secret``: for ``standard`` and ``hmac``.
* ``id``: where the job id is in the JSON body (dotted; defaults for the providers above).
* ``param``: the parameter that takes the webhook URL on submission (sent with
  ``MEDIA_WEBHOOK_URL``). Leave it out when the URL is set in the provider's dashboard (OpenAI).
  Through LiteLLM (1.106) no video provider takes one: fal wants a ``fal_webhook`` query
  parameter, which LiteLLM's video calls don't send, and LiteLLM has no Replicate video. Give
  those providers a backend of your own that reads ``param`` (fal's ``fal_client.submit(...,
  webhook_url=)``, Replicate's ``webhook=``). With the LiteLLM backend the URL isn't sent, their
  jobs keep checking every ``MEDIA_POLL_EVERY`` seconds, and ``llm_kit.W005`` says so.

Jobs check rarely only while a webhook can actually come: the URL is set in the provider's
dashboard (no ``param``), or ``param`` and ``MEDIA_WEBHOOK_URL`` are set and the backend sends
it (a backend lists the providers it can't send it to in ``cannot_send_webhook_url``).

A request that fails its check gets 401; one for a job that isn't running gets 202 and changes
nothing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from typing import Any

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from llm_kit import conf
from llm_kit.core.catalogue.fields import normalise_provider
from llm_kit.core.graphs import import_target
from llm_kit.core.touchpoints.env import provider_of

logger = logging.getLogger("llm_kit")

TOLERANCE = 300  # seconds a signed timestamp may be off
DEFAULTS: dict[str, dict[str, Any]] = {
    "replicate": {"verify": "standard", "id": "id"},
    "openai": {"verify": "standard", "id": "data.id"},
    "fal_ai": {"verify": "fal", "id": "request_id"},
}
FAL_KEYS_URL = "https://rest.fal.ai/.well-known/jwks.json"  # fal: cache for at most 24 hours


def config_for(ref_or_provider: str) -> dict[str, Any] | None:
    """The webhook settings for a model's provider, or None when webhooks are off for it."""
    provider = normalise_provider(provider_of(ref_or_provider) if "/" in ref_or_provider else ref_or_provider)
    configured = (conf.get("MEDIA_WEBHOOKS") or {}).get(provider)
    if configured is None:
        return None
    return {**DEFAULTS.get(provider, {}), **dict(configured), "provider": provider}


def _cannot_send(config: dict[str, Any], backend: Any) -> bool:
    return config["provider"] in (getattr(backend, "cannot_send_webhook_url", None) or ())


def submit_params(ref: str, backend: Any = None) -> dict[str, Any]:
    """Provider parameters that give a new job the webhook URL, when the provider takes one and
    ``backend`` can send it."""
    config = config_for(ref)
    url = conf.get("MEDIA_WEBHOOK_URL") or ""
    if not config or not config.get("param") or not url or _cannot_send(config, backend):
        return {}
    return {config["param"]: url.format(provider=config["provider"])}


def expected(ref: str, backend: Any = None) -> bool:
    """Whether a webhook can come for a job of ``ref``'s run through ``backend``: webhooks are on
    for its provider, and the provider knows where to call (its dashboard, or the URL sent with
    the job). Only then do its jobs check rarely (``MEDIA_WEBHOOK_POLL_EVERY``)."""
    config = config_for(ref)
    if config is None:
        return False
    if not config.get("param"):
        return True  # the URL is set at the provider
    return bool(conf.get("MEDIA_WEBHOOK_URL")) and not _cannot_send(config, backend)


# ------------------------------------------------------------------ checks


def _header(request, name: str) -> str:
    return request.headers.get(name) or ""


def verify_standard(request, config: dict[str, Any]) -> bool:
    """Standard Webhooks (standardwebhooks.com): HMAC-SHA256 of ``id.timestamp.body``."""
    secret = str(config.get("secret") or "")
    msg_id = _header(request, "webhook-id") or _header(request, "svix-id")
    stamp = _header(request, "webhook-timestamp") or _header(request, "svix-timestamp")
    signatures = _header(request, "webhook-signature") or _header(request, "svix-signature")
    if not (secret and msg_id and stamp and signatures):
        return False
    try:
        if abs(time.time() - int(stamp)) > TOLERANCE:
            return False
        key = base64.b64decode(secret.removeprefix("whsec_"))
    except ValueError:
        return False
    signed = f"{msg_id}.{stamp}.".encode() + request.body
    expected = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
    for part in signatures.split():
        version, _, signature = part.partition(",")
        if version == "v1" and hmac.compare_digest(signature, expected):
            return True
    return False


def verify_hmac(request, config: dict[str, Any]) -> bool:
    """An HMAC-SHA256 of the body in one header, hex or base64 (``sha256=`` prefix allowed)."""
    secret, header = str(config.get("secret") or ""), str(config.get("header") or "")
    given = _header(request, header).strip().removeprefix("sha256=") if header else ""
    if not secret or not given:
        return False
    digest = hmac.new(secret.encode(), request.body, hashlib.sha256).digest()
    return hmac.compare_digest(given, digest.hex()) or hmac.compare_digest(given, base64.b64encode(digest).decode())


KEYS_FOR = 86400  # fal: cache its keys for at most 24 hours
REFETCH_EVERY = 300  # at most one fetch per 5 minutes after a failed fetch or an unknown signature
_fal: dict[str, Any] = {"keys": [], "fetched": None, "tried": None}
_fal_fetching = threading.Lock()


def _fetch_fal_keys() -> list[bytes]:
    import httpx

    response = httpx.get(FAL_KEYS_URL, timeout=10)
    response.raise_for_status()
    keys = []
    for key in response.json().get("keys") or []:
        x = key.get("x")
        if x:
            keys.append(base64.urlsafe_b64decode(x + "=" * (-len(x) % 4)))
    return keys


def _fal_public_keys(refresh: bool = False) -> list[bytes]:
    """fal's published keys. ``refresh`` fetches again (a signature matched none: fal may have
    rotated them). Fetches are at most every ``REFETCH_EVERY`` seconds, failed ones included, and
    never wait on each other: a request arriving during a fetch uses the keys at hand."""
    now = time.monotonic()
    fresh = _fal["keys"] and _fal["fetched"] is not None and now - _fal["fetched"] < KEYS_FOR
    if fresh and not refresh:
        return _fal["keys"]
    if _fal["tried"] is not None and now - _fal["tried"] < REFETCH_EVERY:
        return _fal["keys"]
    if not _fal_fetching.acquire(blocking=False):
        return _fal["keys"]
    try:
        _fal["tried"] = now
        keys = _fetch_fal_keys()
        if keys:
            _fal.update(keys=keys, fetched=now)
        else:
            logger.warning("fal published no webhook keys at %s", FAL_KEYS_URL)
    except Exception:
        logger.warning("Could not fetch fal's webhook keys", exc_info=True)
    finally:
        _fal_fetching.release()
    return _fal["keys"]


def verify_fal(request, config: dict[str, Any]) -> bool:
    """fal's ED25519 signature over request id, user id, timestamp and the body's SHA-256."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        logger.error("Checking fal webhooks needs the cryptography package")
        return False
    request_id = _header(request, "X-Fal-Webhook-Request-Id")
    user_id = _header(request, "X-Fal-Webhook-User-Id")
    stamp = _header(request, "X-Fal-Webhook-Timestamp")
    signature = _header(request, "X-Fal-Webhook-Signature")
    if not (request_id and user_id and stamp and signature):
        return False
    try:
        if abs(time.time() - int(stamp)) > TOLERANCE:
            return False
        signature_bytes = bytes.fromhex(signature)
    except ValueError:
        return False
    message = "\n".join([request_id, user_id, stamp, hashlib.sha256(request.body).hexdigest()]).encode()

    def signed_by(keys: list[bytes]) -> bool:
        for key in keys:
            try:
                Ed25519PublicKey.from_public_bytes(key).verify(signature_bytes, message)
                return True
            except (InvalidSignature, ValueError):
                continue
        return False

    if config.get("public_keys"):
        return signed_by(config["public_keys"])
    keys = _fal_public_keys()
    if signed_by(keys):
        return True
    newer = _fal_public_keys(refresh=True)  # fal may have rotated its keys
    return newer != keys and signed_by(newer)


VERIFIERS = {"standard": verify_standard, "hmac": verify_hmac, "fal": verify_fal}


def verified(request, config: dict[str, Any]) -> bool:
    how = config.get("verify") or "standard"
    check = VERIFIERS.get(how) if isinstance(how, str) else how
    if check is None:
        check = import_target(how)
    return bool(check(request, config))


def job_id_in(body: Any, path: str) -> str | None:
    value = body
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return str(value) if value not in (None, "") else None


# ------------------------------------------------------------------ the view


@csrf_exempt
@require_POST
def receive(request, provider: str):
    from llm_kit.media.jobs import notify

    config = config_for(provider)
    if config is None:
        return HttpResponse(status=404)
    try:
        ok = verified(request, config)
    except Exception:
        logger.exception("Checking a %s webhook failed", provider)
        ok = False
    if not ok:
        return HttpResponse(status=401)
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        return HttpResponse(status=400)
    job_id = job_id_in(body, str(config.get("id") or "id"))
    if job_id and notify(job_id):
        return HttpResponse(status=204)
    return HttpResponse(status=202)
