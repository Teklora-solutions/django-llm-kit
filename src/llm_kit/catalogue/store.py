"""``CatalogueStore`` implementation on the llm_kit catalogue tables."""

from __future__ import annotations

from decimal import Decimal

from django.db import models as djm
from django.db import transaction

from llm_kit.core.catalogue.fields import FIELDS, PRICE_FIELDS, Key
from llm_kit.core.catalogue.store import CatalogueState, Entry, SourceState
from llm_kit.models import (
    CatalogueChange,
    CatalogueModel,
    CatalogueOverride,
    CatalogueProvider,
    CatalogueSource,
)

_COLUMN_FIELDS = [f for f in FIELDS if f not in ("input_modalities", "output_modalities")]


def _modalities(value) -> str:
    return f",{','.join(value)}," if value else ""


def _row_from_entry(key: Key, e: Entry) -> CatalogueModel:
    v = e.values
    row = CatalogueModel(
        provider=key[0],
        model_id=key[1],
        input_modalities=_modalities(v.get("input_modalities")),
        output_modalities=_modalities(v.get("output_modalities")),
        values=v,
        source_values=e.source_values,
        provenance=e.provenance,
        conflicts=e.conflicts,
        hidden=e.hidden,
        removed=e.removed,
        first_seen=e.first_seen,
        last_seen=e.last_seen,
        removed_at=e.removed_at,
    )
    for f in _COLUMN_FIELDS:
        val = v.get(f)
        if f in PRICE_FIELDS:
            val = Decimal(val) if val is not None else None
        elif f == "price_tiers":
            val = val or []
        elif isinstance(field := CatalogueModel._meta.get_field(f), djm.CharField):
            val = str(val)[: field.max_length] if val is not None else ""
        setattr(row, f, val)
    return row


def _entry_from_row(row: CatalogueModel) -> Entry:
    return Entry(
        source_values=row.source_values or {},
        values=row.values or {},
        provenance=row.provenance or {},
        conflicts=row.conflicts or {},
        hidden=row.hidden,
        removed=row.removed,
        first_seen=row.first_seen,
        last_seen=row.last_seen,
        removed_at=row.removed_at,
    )


_UPDATE_FIELDS = [
    f.name for f in CatalogueModel._meta.concrete_fields if f.name not in ("id", "provider", "model_id", "updated_at")
]


class DjangoCatalogueStore:
    """``CatalogueStore`` backed by the llm_kit catalogue tables."""

    def __init__(self, using: str = "default") -> None:
        self.using = using

    def load(self) -> CatalogueState:
        entries = {(r.provider, r.model_id): _entry_from_row(r) for r in CatalogueModel.objects.using(self.using).all()}
        overrides = {
            (o.provider, o.model_id): o.values or {} for o in CatalogueOverride.objects.using(self.using).all()
        }
        sources = {
            s.name: SourceState(
                name=s.name,
                url=s.url,
                etag=s.etag,
                last_modified=s.last_modified,
                content_hash=s.content_hash,
                last_fetched_at=s.last_fetched_at,
                last_success_at=s.last_success_at,
                last_status=s.last_status,
                last_error=s.last_error,
                record_count=s.record_count,
            )
            for s in CatalogueSource.objects.using(self.using).all()
        }
        return CatalogueState(entries, overrides, sources)

    def save(self, entries, changes, sources, providers, now) -> None:
        rows = [_row_from_entry(k, e) for k, e in entries.items()]
        with transaction.atomic(using=self.using):
            CatalogueModel.objects.using(self.using).bulk_create(
                rows,
                batch_size=500,
                update_conflicts=True,
                unique_fields=["provider", "model_id"],
                update_fields=_UPDATE_FIELDS,
            )
            CatalogueModel.objects.using(self.using).filter(removed=False).update(last_seen=now)
            CatalogueChange.objects.using(self.using).bulk_create(
                [
                    CatalogueChange(
                        synced_at=now,
                        provider=c.provider,
                        model_id=c.model_id,
                        kind=c.kind,
                        field=c.field,
                        old=c.old,
                        new=c.new,
                    )
                    for c in changes
                ],
                batch_size=500,
            )
            for st in sources.values():
                CatalogueSource.objects.using(self.using).update_or_create(
                    name=st.name,
                    defaults={
                        "url": st.url,
                        "etag": st.etag,
                        "last_modified": st.last_modified,
                        "content_hash": st.content_hash,
                        "last_fetched_at": st.last_fetched_at,
                        "last_success_at": st.last_success_at,
                        "last_status": st.last_status,
                        "last_error": st.last_error,
                        "record_count": st.record_count,
                    },
                )
            if providers:
                CatalogueProvider.objects.using(self.using).bulk_create(
                    [
                        CatalogueProvider(
                            provider=p, name=i.get("name") or p, env=i.get("env") or [], doc_url=i.get("doc") or ""
                        )
                        for p, i in providers.items()
                    ],
                    update_conflicts=True,
                    unique_fields=["provider"],
                    update_fields=["name", "env", "doc_url", "updated_at"],
                )
