"""Refreshing the catalogue and re-applying admin overrides."""

from __future__ import annotations

from datetime import timedelta

from django.utils import timezone

from llm_kit import conf
from llm_kit.catalogue.store import DjangoCatalogueStore, _entry_from_row, _row_from_entry
from llm_kit.core.catalogue import DEFAULT_SOURCES, Source, SyncReport, reapply_overrides
from llm_kit.core.catalogue import sync as core_sync
from llm_kit.core.catalogue.store import Entry
from llm_kit.core.graphs import import_target
from llm_kit.models import CatalogueModel, CatalogueOverride, CatalogueSource


def configured_sources() -> list[Source]:
    """``LLM_KIT["CATALOGUE_SOURCES"]``: built-in names and/or dotted paths to ``Source`` objects."""
    wanted = conf.get("CATALOGUE_SOURCES")
    if wanted is None:
        return list(DEFAULT_SOURCES)
    by_name = {s.name: s for s in DEFAULT_SOURCES}
    return [by_name[w] if w in by_name else import_target(w) for w in wanted]


def sync(*, force: bool = False, offline: bool = False, sources: list[Source] | None = None, fetch=None) -> SyncReport:
    """Fetch, merge, diff and store the catalogue. Takes a few seconds; run it in a worker."""
    kwargs = {"fetch": fetch} if fetch else {}
    return core_sync(
        DjangoCatalogueStore(),
        sources if sources is not None else configured_sources(),
        force=force,
        snapshot="only" if offline else "fallback",
        **kwargs,
    )


def last_synced():
    return (
        CatalogueSource.objects.filter(last_success_at__isnull=False)
        .order_by("-last_success_at")
        .values_list("last_success_at", flat=True)
        .first()
    )


def is_stale(max_age: timedelta | None = None) -> bool:
    max_age = max_age or timedelta(hours=conf.get("CATALOGUE_MAX_AGE_HOURS"))
    last = last_synced()
    return last is None or timezone.now() - last > max_age


def apply_override(provider: str, model_id: str) -> CatalogueModel | None:
    """Re-merge one model after its override changed. Creates private models; no network."""
    key = (provider, model_id)
    override = (
        CatalogueOverride.objects.filter(provider=provider, model_id=model_id).values_list("values", flat=True).first()
    )
    row = CatalogueModel.objects.filter(provider=provider, model_id=model_id).first()
    now = timezone.now()
    if row is None:
        if override is None:
            return None
        entry = Entry(source_values={}, values={}, first_seen=now, last_seen=now)
    else:
        entry = _entry_from_row(row)
    reapply_overrides(key, entry, override)
    if not entry.removed:
        entry.removed_at = None
        entry.last_seen = entry.last_seen or now
    elif entry.removed_at is None:
        entry.removed_at = now
    new_row = _row_from_entry(key, entry)
    if row is not None:
        new_row.pk = row.pk
        new_row.save()
    else:
        new_row.save(force_insert=True)
    return new_row
