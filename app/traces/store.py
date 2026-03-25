from __future__ import annotations

import logging
from typing import Any, Protocol

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.settings import Settings
from app.traces.models import TurnTraceRecord
from app.traces.orm import Base, DiscoveryCallFlowORM, TurnTraceORM
from app.traces.queue import TraceQueueWorker

logger = logging.getLogger(__name__)


class TraceStore(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def enqueue(self, record: TurnTraceRecord) -> bool: ...


class NoOpTraceStore:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def enqueue(self, record: TurnTraceRecord) -> bool:
        del record
        return False


class PostgresTraceStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self._worker = TraceQueueWorker(
            queue_size=settings.trace_capture_queue_size,
            batch_size=settings.trace_capture_batch_size,
            flush_interval_ms=settings.trace_capture_flush_interval_ms,
            persist_batch=self._persist_batch,
        )

    async def start(self) -> None:
        if self._engine is not None:
            return
        if not self._settings.trace_capture_database_url:
            raise ValueError("TRACE_CAPTURE_DATABASE_URL es requerido cuando TRACE_CAPTURE_ENABLED=true")
        self._engine = create_async_engine(self._settings.trace_capture_database_url, future=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self._worker.start()

    async def stop(self) -> None:
        await self._worker.stop()
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None

    def enqueue(self, record: TurnTraceRecord) -> bool:
        return self._worker.enqueue(record)

    async def _persist_batch(self, records: list[TurnTraceRecord]) -> None:
        if not records or self._sessionmaker is None:
            return
        async with self._sessionmaker() as session:
            async with session.begin():
                for record in records:
                    await self._persist_one(session, record)

    async def _persist_one(self, session: AsyncSession, record: TurnTraceRecord) -> None:
        turn = TurnTraceORM(
            id=record.id,
            created_at=record.created_at,
            flow_id=record.flow_id,
            conversation_id=record.conversation_id,
            contact_id=record.contact_id,
            account_id=record.account_id,
            trace_version=record.trace_version,
            llm_backend=record.llm_backend,
            llm_model=record.llm_model,
            next_node=record.next_node,
            intent=record.intent,
            needs_retrieval=record.needs_retrieval,
            handoff_required=record.handoff_required,
            response_sent=record.response_sent,
            has_error=record.has_error,
            webhook_snapshot=record.webhook_snapshot.model_dump(mode="json"),
            state_before=record.state_before.model_dump(mode="json"),
            state_after=record.state_after.model_dump(mode="json") if record.state_after else None,
            route_input=record.route_input.model_dump(mode="json") if record.route_input else None,
            route_output=record.route_output.model_dump(mode="json") if record.route_output else None,
            rag_trace=record.rag_trace.model_dump(mode="json") if record.rag_trace else None,
            discovery_call_trace=(
                record.discovery_call_trace.model_dump(mode="json") if record.discovery_call_trace else None
            ),
            outbound_trace=record.outbound_trace.model_dump(mode="json") if record.outbound_trace else None,
        )
        session.add(turn)
        await session.flush()

        active_flow = await self._get_active_discovery_flow(session, record.conversation_id)
        if record.next_node == "discovery_call" or record.discovery_call_trace is not None:
            if active_flow is None:
                active_flow = DiscoveryCallFlowORM(
                    conversation_id=record.conversation_id,
                    contact_id=record.contact_id,
                    opening_turn_trace_id=turn.id,
                    latest_stage=record.discovery_call_trace.stage_after if record.discovery_call_trace else "",
                    latest_slots=record.discovery_call_trace.slots_after_merge if record.discovery_call_trace else {},
                    status="active",
                )
                session.add(active_flow)
                await session.flush()

            active_flow.updated_at = record.created_at
            active_flow.latest_stage = record.discovery_call_trace.stage_after if record.discovery_call_trace else ""
            active_flow.latest_slots = (
                record.discovery_call_trace.slots_after_merge if record.discovery_call_trace else active_flow.latest_slots
            )
            if record.discovery_call_trace and record.discovery_call_trace.stage_after == "ready_for_handoff":
                active_flow.status = "ready_for_handoff"
                active_flow.closed_at = record.created_at
                active_flow.closing_turn_trace_id = turn.id
                active_flow.final_payload = record.discovery_call_trace.payload_extracted.model_dump(mode="json")

            turn.discovery_call_flow_id = active_flow.id
        elif active_flow is not None:
            active_flow.status = "closed"
            active_flow.closed_at = record.created_at
            active_flow.closing_turn_trace_id = turn.id

    async def _get_active_discovery_flow(
        self, session: AsyncSession, conversation_id: str
    ) -> DiscoveryCallFlowORM | None:
        result = await session.execute(
            select(DiscoveryCallFlowORM).where(
                and_(
                    DiscoveryCallFlowORM.conversation_id == conversation_id,
                    DiscoveryCallFlowORM.status == "active",
                )
            )
        )
        return result.scalar_one_or_none()


def build_trace_store(settings: Settings) -> TraceStore:
    if not settings.trace_capture_enabled:
        return NoOpTraceStore()
    return PostgresTraceStore(settings)
