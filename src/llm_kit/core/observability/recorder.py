"""Hands finished records to sinks on a background thread.

Callbacks run inline (even inside async streams), so they never touch a
database or the network themselves; they enqueue records here. One writer
thread prices calls (catalogue lookups may hit the DB), rolls call costs up
into their run, and passes batches to every sink. A sink failing never breaks
a model call, and records are dropped (with a warning) rather than blocking
if the queue is full.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
from collections.abc import Callable
from decimal import Decimal
from typing import Any, Protocol

from llm_kit.core.observability.cost import PriceLookup, price_call
from llm_kit.core.observability.records import CallRecord, RunRecord, ToolRecord

logger = logging.getLogger("llm_kit.observability")

Record = CallRecord | RunRecord | ToolRecord


def _is_marker(item: object) -> bool:
    return not isinstance(item, (CallRecord, RunRecord, ToolRecord))


class Sink(Protocol):
    def write(self, calls: list[CallRecord], runs: list[RunRecord], tools: list[ToolRecord]) -> None: ...


class Recorder:
    def __init__(self, max_queue: int = 10_000, batch_size: int = 200, batch_wait: float = 0.2) -> None:
        self.sinks: list[Sink] = []
        self.price_lookup: PriceLookup | None = None
        self._queue: queue.Queue[Record | Callable[[], None]] = queue.Queue(max_queue)
        self._batch_size = batch_size
        self._batch_wait = batch_wait
        self._thread: threading.Thread | None = None
        self._pid: int | None = None
        self._lock = threading.Lock()
        self._run_costs: dict[str, list[Any]] = {}  # run_id -> [cost, unpriced_calls]
        self.dropped = 0

    # -------------------------------------------------------------- set-up

    def configure(self, sinks: list[Sink], price_lookup: PriceLookup | None) -> None:
        self.sinks = list(sinks)
        self.price_lookup = price_lookup

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive() and self._pid == os.getpid():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._pid == os.getpid():
                return
            if self._pid != os.getpid():  # forked worker: start fresh
                self._queue = queue.Queue(self._queue.maxsize)
                self._run_costs = {}
            self._pid = os.getpid()
            self._thread = threading.Thread(target=self._work, name="llm-kit-recorder", daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------- API

    def emit(self, record: Record) -> None:
        if not self.sinks:
            return
        self._ensure_thread()
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 1000 == 1:
                logger.warning("llm-kit recorder queue full; dropped %s records so far", self.dropped)

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until everything emitted so far has been written. True if it finished in time."""
        if self._thread is None or not self._thread.is_alive():
            return True
        done = threading.Event()
        try:
            self._queue.put(done.set, timeout=timeout)
        except queue.Full:
            return False
        return done.wait(timeout)

    # -------------------------------------------------------------- worker

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            batch: list[Record] = []
            markers: list[Callable[[], None]] = []
            while True:
                if _is_marker(item):
                    markers.append(item)
                    break  # write what we have, then signal the flush
                batch.append(item)
                if len(batch) >= self._batch_size:
                    break
                try:
                    item = self._queue.get(timeout=self._batch_wait)
                except queue.Empty:
                    break
            if batch:
                self._write(batch)
            for marker in markers:
                marker()

    def _write(self, batch: list[Record]) -> None:
        calls, runs, tools = [], [], []
        for r in batch:
            if isinstance(r, CallRecord):
                self._price(r)
                calls.append(r)
            elif isinstance(r, RunRecord):
                cost, unpriced = self._run_costs.pop(r.run_id, [Decimal(0), 0])
                r.cost, r.unpriced_calls = cost, unpriced
                runs.append(r)
            else:
                tools.append(r)
        for sink in self.sinks:
            try:
                sink.write(calls, runs, tools)
            except Exception:
                logger.exception("llm-kit sink %s failed; %s records lost", type(sink).__name__, len(batch))

    def _price(self, call: CallRecord) -> None:
        if call.status == "ok" and self.price_lookup is not None:
            try:
                prices = self.price_lookup(call.model or call.served_model)
            except Exception:
                logger.exception("llm-kit price lookup failed for %s", call.model)
                prices = None
            call.cost = price_call(
                prices,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                cache_read_tokens=call.cache_read_tokens,
                cache_write_tokens=call.cache_write_tokens,
                reasoning_tokens=call.reasoning_tokens,
            )
            call.unpriced = call.cost is None
        if call.run_id:
            if len(self._run_costs) > 10_000:  # roots that never produce a run record (e.g. a bare tool)
                self._run_costs.clear()
            acc = self._run_costs.setdefault(call.run_id, [Decimal(0), 0])
            if call.cost is not None:
                acc[0] += call.cost
            if call.unpriced:
                acc[1] += 1


recorder = Recorder()
atexit.register(lambda: recorder.flush(timeout=5))
