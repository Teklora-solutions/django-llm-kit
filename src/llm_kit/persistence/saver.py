"""A LangGraph checkpointer backed by the Django ORM.

Works with any database Django supports (SQLite, Postgres, MySQL), so graph
state lives next to your app data, survives restarts and shows in the admin.

    from llm_kit.persistence.saver import DjangoSaver
    graph = builder.compile(checkpointer=DjangoSaver())
"""

from __future__ import annotations

import functools
import json
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from asgiref.sync import sync_to_async
from django.db import DEFAULT_DB_ALIAS, connections, transaction
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
    writes_sort_key,
)

from llm_kit import models as m

_SCALARS = (str, int, float, bool)


def _db(fn):
    """Run ``fn`` against the saver's DB alias, closing any connection it opened.

    LangGraph's sync loop calls the checkpointer from short-lived worker
    threads; without this every run would leak a database connection.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        conn = connections[self.using]
        had_connection = conn.connection is not None
        try:
            return fn(self, *args, **kwargs)
        finally:
            if not had_connection and not conn.in_atomic_block:
                conn.close()

    return wrapper


def _bytes(value) -> bytes:
    return bytes(value) if value is not None else b""


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class DjangoSaver(BaseCheckpointSaver[str]):
    """Stores checkpoints in the ``llm_kit`` Checkpoint / CheckpointBlob / CheckpointWrite tables.

    Args:
        using: Django database alias. Defaults to ``"default"``.
        serde: Optional LangGraph serializer (e.g. an ``EncryptedSerializer``).
    """

    def __init__(self, *, using: str = DEFAULT_DB_ALIAS, serde: SerializerProtocol | None = None) -> None:
        super().__init__(serde=serde)
        self.using = using

    # ------------------------------------------------------------------ helpers

    def _qs(self, model):
        return model.objects.using(self.using)

    @staticmethod
    def _ids(config: RunnableConfig) -> tuple[str, str]:
        c = config["configurable"]
        return str(c["thread_id"]), c.get("checkpoint_ns", "") or ""

    def _load_tuples(self, rows: list[m.Checkpoint]) -> list[CheckpointTuple]:
        """Build CheckpointTuples for rows of one or more threads with 2 extra queries."""
        if not rows:
            return []
        checkpoints = {r.pk: self.serde.loads_typed((r.type, _bytes(r.checkpoint))) for r in rows}

        # Blobs: fetch every (thread, ns, channel, version) referenced, in one query.
        wanted: set[tuple[str, str, str, str]] = set()
        for r in rows:
            for ch, ver in checkpoints[r.pk]["channel_versions"].items():
                wanted.add((r.thread_id, r.checkpoint_ns, ch, str(ver)))
        blobs: dict[tuple[str, str, str, str], Any] = {}
        if wanted:
            qs = self._qs(m.CheckpointBlob).filter(
                thread_id__in={w[0] for w in wanted},
                checkpoint_ns__in={w[1] for w in wanted},
                channel__in={w[2] for w in wanted},
                version__in={w[3] for w in wanted},
            )
            for b in qs:
                key = (b.thread_id, b.checkpoint_ns, b.channel, b.version)
                if key in wanted and b.type != "empty":
                    blobs[key] = self.serde.loads_typed((b.type, _bytes(b.blob)))

        # Pending writes for every checkpoint, in one query.
        writes: dict[tuple[str, str, str], list[m.CheckpointWrite]] = {}
        wqs = self._qs(m.CheckpointWrite).filter(
            thread_id__in={r.thread_id for r in rows},
            checkpoint_id__in={r.checkpoint_id for r in rows},
        )
        for w in wqs:
            writes.setdefault((w.thread_id, w.checkpoint_ns, w.checkpoint_id), []).append(w)

        out = []
        for r in rows:
            ckpt = checkpoints[r.pk]
            values = {}
            for ch, ver in ckpt["channel_versions"].items():
                key = (r.thread_id, r.checkpoint_ns, ch, str(ver))
                if key in blobs:
                    values[ch] = blobs[key]
            pending = sorted(
                writes.get((r.thread_id, r.checkpoint_ns, r.checkpoint_id), []),
                key=lambda w: writes_sort_key(w.task_path, w.task_id, w.idx),
            )
            out.append(
                CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": r.thread_id,
                            "checkpoint_ns": r.checkpoint_ns,
                            "checkpoint_id": r.checkpoint_id,
                        }
                    },
                    checkpoint={**ckpt, "channel_values": values},
                    metadata=self.serde.loads_typed((r.metadata_type, _bytes(r.metadata_blob))),
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": r.thread_id,
                                "checkpoint_ns": r.checkpoint_ns,
                                "checkpoint_id": r.parent_checkpoint_id,
                            }
                        }
                        if r.parent_checkpoint_id
                        else None
                    ),
                    pending_writes=[
                        (w.task_id, w.channel, self.serde.loads_typed((w.type, _bytes(w.blob)))) for w in pending
                    ],
                )
            )
        return out

    # --------------------------------------------------------------- sync API

    @_db
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, ns = self._ids(config)
        qs = self._qs(m.Checkpoint).filter(thread_id=thread_id, checkpoint_ns=ns)
        if checkpoint_id := get_checkpoint_id(config):
            row = qs.filter(checkpoint_id=checkpoint_id).first()
        else:
            row = qs.order_by("-checkpoint_id").first()
        if row is None:
            return None
        return self._load_tuples([row])[0]

    @_db
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        qs = self._qs(m.Checkpoint).all()
        if config:
            c = config["configurable"]
            qs = qs.filter(thread_id=str(c["thread_id"]))
            if c.get("checkpoint_ns") is not None:
                qs = qs.filter(checkpoint_ns=c["checkpoint_ns"])
            if checkpoint_id := get_checkpoint_id(config):
                qs = qs.filter(checkpoint_id=checkpoint_id)
        if before and (before_id := get_checkpoint_id(before)):
            qs = qs.filter(checkpoint_id__lt=before_id)

        # Push simple filters into SQL; verify everything in Python too, since
        # JSON comparisons differ slightly between database backends.
        db_filterable = all(isinstance(v, _SCALARS) and "__" not in k for k, v in (filter or {}).items())
        if filter and db_filterable:
            qs = qs.filter(**{f"metadata__{k}": v for k, v in filter.items()})
        qs = (
            qs.order_by("-thread_id", "checkpoint_ns", "-checkpoint_id")
            if not config
            else qs.order_by("-checkpoint_id")
        )

        if limit is not None and db_filterable:
            qs = qs[: max(limit, 0)]
        rows = list(qs)
        tuples = self._load_tuples(rows)
        if filter:
            tuples = [t for t in tuples if all(t.metadata.get(k) == v for k, v in filter.items())]
        if limit is not None:
            tuples = tuples[: max(limit, 0)]
        return iter(tuples)

    @_db
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, ns = self._ids(config)
        c = checkpoint.copy()
        values: dict[str, Any] = c.pop("channel_values", {})  # type: ignore[misc]
        blobs = []
        for ch, ver in new_versions.items():
            if ch in values:
                type_, data = self.serde.dumps_typed(values[ch])
            else:
                type_, data = "empty", None
            blobs.append(
                m.CheckpointBlob(
                    thread_id=thread_id, checkpoint_ns=ns, channel=ch, version=str(ver), type=type_, blob=data
                )
            )
        ckpt_type, ckpt_data = self.serde.dumps_typed(c)
        meta = get_serializable_checkpoint_metadata(config, metadata)
        meta_type, meta_data = self.serde.dumps_typed(meta)

        # Write-only transaction (no read-then-write), so SQLite never has to
        # upgrade a shared lock while LangGraph writes from several threads.
        with transaction.atomic(using=self.using):
            if blobs:
                self._qs(m.CheckpointBlob).bulk_create(blobs, ignore_conflicts=True)
            self._qs(m.Checkpoint).filter(
                thread_id=thread_id, checkpoint_ns=ns, checkpoint_id=checkpoint["id"]
            ).delete()
            self._qs(m.Checkpoint).create(
                thread_id=thread_id,
                checkpoint_ns=ns,
                checkpoint_id=checkpoint["id"],
                parent_checkpoint_id=config["configurable"].get("checkpoint_id"),
                type=ckpt_type,
                checkpoint=ckpt_data,
                metadata_type=meta_type,
                metadata_blob=meta_data,
                metadata=_json_safe(meta),
            )
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint["id"]}}

    @_db
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, ns = self._ids(config)
        checkpoint_id = config["configurable"]["checkpoint_id"]
        regular, special = [], []
        for i, (channel, value) in enumerate(writes):
            idx = WRITES_IDX_MAP.get(channel, i)
            type_, data = self.serde.dumps_typed(value)
            row = m.CheckpointWrite(
                thread_id=thread_id,
                checkpoint_ns=ns,
                checkpoint_id=checkpoint_id,
                task_id=task_id,
                task_path=task_path,
                idx=idx,
                channel=channel,
                type=type_,
                blob=data,
            )
            (special if idx < 0 else regular).append(row)
        with transaction.atomic(using=self.using):
            # Regular writes are idempotent: keep the first one stored.
            if regular:
                self._qs(m.CheckpointWrite).bulk_create(regular, ignore_conflicts=True)
            # Special writes (errors, interrupts, resumes...) overwrite.
            if special:
                self._qs(m.CheckpointWrite).filter(
                    thread_id=thread_id,
                    checkpoint_ns=ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    idx__in=[row.idx for row in special],
                ).delete()
                self._qs(m.CheckpointWrite).bulk_create(special)

    @_db
    def delete_thread(self, thread_id: str) -> None:
        thread_id = str(thread_id)
        with transaction.atomic(using=self.using):
            for model in (m.Checkpoint, m.CheckpointBlob, m.CheckpointWrite):
                self._qs(model).filter(thread_id=thread_id).delete()

    @_db
    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        source, target = str(source_thread_id), str(target_thread_id)
        with transaction.atomic(using=self.using):
            for model in (m.Checkpoint, m.CheckpointBlob, m.CheckpointWrite):
                rows = list(self._qs(model).filter(thread_id=source))
                for row in rows:
                    row.pk = None
                    row.thread_id = target
                self._qs(model).bulk_create(rows, ignore_conflicts=True)

    def get_next_version(self, current: str | None, channel: None = None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"

    # -------------------------------------------------------------- async API

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await sync_to_async(self.get_tuple)(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        items = await sync_to_async(lambda: [*self.list(config, filter=filter, before=before, limit=limit)])()
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await sync_to_async(self.put)(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await sync_to_async(self.put_writes)(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await sync_to_async(self.delete_thread)(thread_id)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await sync_to_async(self.copy_thread)(source_thread_id, target_thread_id)
