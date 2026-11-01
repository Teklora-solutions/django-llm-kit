"""Storage interface for the catalogue, implemented by framework adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from llm_kit.core.catalogue.fields import Key


@dataclass
class SourceState:
    name: str
    url: str = ""
    etag: str = ""
    last_modified: str = ""
    content_hash: str = ""
    last_fetched_at: datetime | None = None
    last_success_at: datetime | None = None
    last_status: str = ""  # fetched | not_modified | failed | snapshot
    last_error: str = ""
    record_count: int = 0


@dataclass
class Entry:
    source_values: dict[str, dict[str, Any]]  # source name -> partial record
    values: dict[str, Any]  # effective, after merge + override
    provenance: dict[str, str] = field(default_factory=dict)
    conflicts: dict[str, dict[str, Any]] = field(default_factory=dict)
    hidden: bool = False
    removed: bool = False
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    removed_at: datetime | None = None


@dataclass
class Change:
    provider: str
    model_id: str
    kind: str  # added | removed | restored | price_changed | capability_changed | deprecated
    field: str = ""
    old: Any = None
    new: Any = None


@dataclass
class CatalogueState:
    entries: dict[Key, Entry] = field(default_factory=dict)
    overrides: dict[Key, dict[str, Any]] = field(default_factory=dict)
    sources: dict[str, SourceState] = field(default_factory=dict)


class CatalogueStore(Protocol):
    def load(self) -> CatalogueState: ...

    def save(
        self,
        entries: dict[Key, Entry],  # only new or changed entries; others are unchanged
        changes: list[Change],
        sources: dict[str, SourceState],
        providers: dict[str, dict[str, Any]],
        now: datetime,  # also the new last_seen of every model that is not removed
    ) -> None: ...


class MemoryStore:
    """In-memory store, for tests and scripts."""

    def __init__(self) -> None:
        self.state = CatalogueState()
        self.changes: list[Change] = []
        self.providers: dict[str, dict[str, Any]] = {}

    def load(self) -> CatalogueState:
        return CatalogueState(dict(self.state.entries), dict(self.state.overrides), dict(self.state.sources))

    def save(self, entries, changes, sources, providers, now) -> None:
        self.state.entries.update(entries)
        for e in self.state.entries.values():
            if not e.removed:
                e.last_seen = now
        self.state.sources = dict(sources)
        self.changes.extend(changes)
        self.providers.update(providers)
