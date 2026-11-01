"""Fetch sources, merge, diff, and save the catalogue."""

from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import logging
import socket
import ssl
import time
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from itertools import zip_longest
from typing import Any, Literal

from llm_kit.core.catalogue.fields import CAPABILITY_FIELDS, PRICE_FIELDS, Key
from llm_kit.core.catalogue.merge import merge
from llm_kit.core.catalogue.sources import DEFAULT_SOURCES, ParseResult, Source
from llm_kit.core.catalogue.store import CatalogueStore, Change, Entry, SourceState

logger = logging.getLogger("llm_kit.catalogue")

TRACKED_FIELDS = (*CAPABILITY_FIELDS, "context_window", "max_output", "input_modalities", "output_modalities")


@dataclass
class Response:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


Fetcher = Callable[[str, dict[str, str]], Response]


def _connect(host: str, port: int, timeout: float) -> socket.socket:
    """Connect trying addresses alternately by family, IPv4 first, each with a short timeout.

    A host with a broken IPv6 route would otherwise hang in SYN-SENT for
    minutes (one project hit this too, and disabled IPv6 process-wide instead).
    """
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    v4 = [i for i in infos if i[0] == socket.AF_INET]
    v6 = [i for i in infos if i[0] == socket.AF_INET6]
    ordered = [x for pair in zip_longest(v4, v6) for x in pair if x is not None]
    last_error: Exception | None = None
    for family, type_, proto, _, addr in ordered:
        sock = socket.socket(family, type_, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(addr)
            return sock
        except OSError as e:
            last_error = e
            sock.close()
    raise last_error or OSError(f"could not resolve {host}")


class _HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, *, connect_timeout: float, read_timeout: float) -> None:
        super().__init__(host, timeout=read_timeout, context=ssl.create_default_context())
        self._connect_timeout = connect_timeout

    def connect(self) -> None:
        sock = _connect(self.host, self.port, self._connect_timeout)
        sock.settimeout(self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def urllib_fetcher(
    url: str,
    headers: dict[str, str],
    *,
    connect_timeout: float = 5,
    read_timeout: float = 30,
    deadline: float = 120,
    max_redirects: int = 3,
) -> Response:
    """Fetch ``url`` over HTTPS with a per-address connect timeout and a total deadline.

    Standard library only; follows redirects; decompresses gzip.
    """
    started = time.monotonic()
    for _ in range(max_redirects + 1):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https":
            raise ValueError(f"catalogue sources must use https: {url}")
        conn = _HTTPSConnection(parts.netloc, connect_timeout=connect_timeout, read_timeout=read_timeout)
        try:
            path = parts.path or "/"
            if parts.query:
                path += "?" + parts.query
            conn.request("GET", path, headers={"Host": parts.netloc, **headers})
            resp = conn.getresponse()
            hdrs = {k.lower(): v for k, v in resp.getheaders()}
            if resp.status in (301, 302, 303, 307, 308) and "location" in hdrs:
                url = urllib.parse.urljoin(url, hdrs["location"])
                continue
            chunks = []
            while chunk := resp.read(256 * 1024):
                chunks.append(chunk)
                if time.monotonic() - started > deadline:
                    raise TimeoutError(f"download exceeded {deadline:.0f}s")
            body = b"".join(chunks)
        finally:
            conn.close()
        if hdrs.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        return Response(resp.status, body, hdrs)
    raise RuntimeError(f"too many redirects for {url}")


@dataclass
class SyncReport:
    sources: dict[str, SourceState]
    changes: list[Change]
    total: int
    active: int
    baseline: bool

    def count(self, kind: str) -> int:
        return sum(1 for c in self.changes if c.kind == kind)

    def summary(self) -> str:
        parts = [
            f"{name}: {s.last_status}" + (f" ({s.last_error})" if s.last_error else "")
            for name, s in self.sources.items()
        ]
        if self.baseline:
            change_text = "first sync (baseline, no change events)"
        else:
            kinds = ("added", "removed", "restored", "price_changed", "capability_changed", "deprecated")
            change_text = ", ".join(f"{k} {self.count(k)}" for k in kinds if self.count(k)) or "no changes"
        return f"{self.active} active models ({self.total} total). {change_text}. Sources: {'; '.join(parts)}."


def _decimal_changed(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is not b
    return Decimal(str(a)) != Decimal(str(b))


def diff(key: Key, old: Entry | None, new: Entry) -> list[Change]:
    provider, model_id = key
    if old is None or old.removed:
        if new.removed:
            return []
        return [Change(provider, model_id, "added" if old is None else "restored")]
    if new.removed:
        return [Change(provider, model_id, "removed")]
    changes = []
    o, n = old.values, new.values
    for f in PRICE_FIELDS:
        if _decimal_changed(o.get(f), n.get(f)):
            changes.append(Change(provider, model_id, "price_changed", f, o.get(f), n.get(f)))
    was_dep = o.get("status") == "deprecated" or bool(o.get("deprecation_date"))
    is_dep = n.get("status") == "deprecated" or bool(n.get("deprecation_date"))
    if is_dep and not was_dep:
        changes.append(
            Change(
                provider, model_id, "deprecated", "status", o.get("status"), n.get("deprecation_date") or "deprecated"
            )
        )
    for f in TRACKED_FIELDS:
        if o.get(f) != n.get(f):
            changes.append(Change(provider, model_id, "capability_changed", f, o.get(f), n.get(f)))
    return changes


def _entry_changed(old: Entry, new: Entry) -> bool:
    return (
        old.source_values != new.source_values
        or old.values != new.values
        or old.provenance != new.provenance
        or old.conflicts != new.conflicts
        or old.hidden != new.hidden
        or old.removed != new.removed
    )


def _load_snapshot() -> dict[str, ParseResult]:
    from llm_kit.core.catalogue.snapshot import load

    try:
        return load()
    except FileNotFoundError:
        return {}


TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    socket.timeout,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
)


def _fetch_with_retry(
    fetch: Fetcher, url: str, headers: dict[str, str], *, retries: int, backoff: float = 2.0
) -> Response:
    for attempt in range(retries + 1):
        try:
            return fetch(url, headers)
        except TRANSIENT_ERRORS as e:
            if attempt == retries:
                raise
            logger.info("catalogue fetch %s failed (%s), retrying", url, e)
            time.sleep(backoff * (attempt + 1))
    raise AssertionError("unreachable")


def sync(
    store: CatalogueStore,
    sources: Iterable[Source] = DEFAULT_SOURCES,
    fetch: Fetcher = urllib_fetcher,
    *,
    force: bool = False,
    snapshot: Literal["fallback", "only", "never"] = "fallback",
    retries: int = 1,
    now: datetime | None = None,
) -> SyncReport:
    """Refresh the catalogue.

    * Sources are fetched with ETag / If-Modified-Since unless ``force``.
    * A failed source keeps its previous values; nothing is marked removed
      because of a fetch failure.
    * ``snapshot="fallback"`` uses the bundled snapshot for a source that fails
      and has never been fetched; ``"only"`` uses the snapshot without network.
    * The first sync into an empty store records no change events (baseline).
    * Connection errors are retried ``retries`` times.
    * Only entries that changed are written; ``last_seen`` of every listed
      model is bumped by the store in one step.
    """
    now = now or datetime.now(UTC)
    sources = list(sources)
    state = store.load()
    states = dict(state.sources)
    had_data = {name for e in state.entries.values() for name in e.source_values}
    fresh: dict[str, ParseResult] = {}
    snap: dict[str, ParseResult] | None = None

    for src in sources:
        st = states.get(src.name) or SourceState(name=src.name, url=src.url)
        st.url = src.url
        st.last_fetched_at = now
        st.last_error = ""
        if snapshot == "only":
            snap = snap if snap is not None else _load_snapshot()
            if src.name in snap:
                fresh[src.name] = snap[src.name]
                st.last_status, st.record_count = "snapshot", len(snap[src.name].models)
            else:
                st.last_status, st.last_error = "failed", "not in bundled snapshot"
            states[src.name] = st
            continue
        headers = {"Accept-Encoding": "gzip", "User-Agent": "django-llm-kit (+catalogue sync)"}
        if not force and src.name in had_data:
            if st.etag:
                headers["If-None-Match"] = st.etag
            if st.last_modified:
                headers["If-Modified-Since"] = st.last_modified
        try:
            resp = _fetch_with_retry(fetch, src.url, headers, retries=retries)
            if resp.status == 304:
                st.last_status, st.last_success_at = "not_modified", now
            elif resp.status == 200:
                digest = hashlib.sha256(resp.body).hexdigest()
                st.etag = resp.headers.get("etag", "")
                st.last_modified = resp.headers.get("last-modified", "")
                st.last_success_at = now
                if digest == st.content_hash and not force and src.name in had_data:
                    st.last_status = "not_modified"
                else:
                    parsed = src.parse(json.loads(resp.body))
                    if not parsed.models:
                        raise ValueError("source returned no models")
                    fresh[src.name] = parsed
                    st.content_hash, st.last_status, st.record_count = digest, "fetched", len(parsed.models)
            else:
                raise RuntimeError(f"HTTP {resp.status}")
        except Exception as e:  # keep previous data for this source
            logger.warning("catalogue source %s failed: %s", src.name, e)
            st.last_status, st.last_error = "failed", f"{type(e).__name__}: {e}"[:500]
            if snapshot == "fallback" and src.name not in had_data:
                snap = snap if snap is not None else _load_snapshot()
                if src.name in snap:
                    fresh[src.name] = snap[src.name]
                    st.last_status = "snapshot"
                    st.record_count = len(snap[src.name].models)
        states[src.name] = st

    # Rebuild per-model source values: refreshed sources replace their old values.
    source_values: dict[Key, dict[str, dict]] = {k: dict(e.source_values) for k, e in state.entries.items()}
    for name, parsed in fresh.items():
        for sv in source_values.values():
            sv.pop(name, None)
        for key, partial in parsed.models.items():
            source_values.setdefault(key, {})[name] = partial
    for key in state.overrides:
        source_values.setdefault(key, {})

    entries: dict[Key, Entry] = {}
    dirty: dict[Key, Entry] = {}
    changes: list[Change] = []
    baseline = not state.entries
    for key, sv in source_values.items():
        override = state.overrides.get(key)
        merged = merge(sv, override, provider=key[0])
        removed = not sv and not override
        old = state.entries.get(key)
        entry = Entry(
            source_values=sv,
            values=merged.values,
            provenance=merged.provenance,
            conflicts=merged.conflicts,
            hidden=merged.hidden,
            removed=removed,
            first_seen=(old.first_seen if old and old.first_seen else now),
            last_seen=now if not removed else (old.last_seen if old else None),
            removed_at=((old.removed_at if old and old.removed else now) if removed else None),
        )
        entries[key] = entry
        if not baseline:
            changes.extend(diff(key, old, entry))
        if old is None or _entry_changed(old, entry):
            dirty[key] = entry

    providers: dict[str, dict] = {}
    for parsed in fresh.values():
        providers.update(parsed.providers)
    store.save(dirty, changes, states, providers, now)
    active = sum(1 for e in entries.values() if not e.removed)
    return SyncReport(sources=states, changes=changes, total=len(entries), active=active, baseline=baseline)


def reapply_overrides(key: Key, entry: Entry, override: dict[str, Any] | None) -> Entry:
    """Recompute an entry's effective values after an override changed (no network)."""
    merged = merge(entry.source_values, override, provider=key[0])
    entry.values, entry.provenance, entry.conflicts, entry.hidden = (
        merged.values,
        merged.provenance,
        merged.conflicts,
        merged.hidden,
    )
    entry.removed = not entry.source_values and not override
    return entry
