from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from app.traces.models import TurnTraceRecord

logger = logging.getLogger(__name__)


class TraceQueueWorker:
    def __init__(
        self,
        *,
        queue_size: int,
        batch_size: int,
        flush_interval_ms: int,
        persist_batch: Callable[[list[TurnTraceRecord]], asyncio.Future | asyncio.Task | object],
    ) -> None:
        self._queue: asyncio.Queue[TurnTraceRecord] = asyncio.Queue(maxsize=queue_size)
        self._batch_size = batch_size
        self._flush_interval = flush_interval_ms / 1000
        self._persist_batch = persist_batch
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None

    def enqueue(self, record: TurnTraceRecord) -> bool:
        try:
            self._queue.put_nowait(record)
            return True
        except asyncio.QueueFull:
            logger.warning("Trace queue full. Dropping trace flow=%s conversation=%s", record.flow_id, record.conversation_id)
            return False

    async def _run(self) -> None:
        pending: list[TurnTraceRecord] = []
        while self._running or not self._queue.empty():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
                pending.append(item)
                if len(pending) < self._batch_size:
                    continue
            except TimeoutError:
                pass

            if not pending:
                continue

            batch = pending[:]
            pending.clear()
            try:
                result = self._persist_batch(batch)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # pragma: no cover - defensivo
                logger.exception("Failed to persist trace batch of %s records: %s", len(batch), exc)
