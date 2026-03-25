import asyncio
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.schemas import ChatwootWebhook, DiscoveryCallIntentPayload
from app.settings import Settings
from app.traces.context import TurnTraceContext
from app.traces.orm import DiscoveryCallFlowORM, TurnTraceORM
from app.traces.store import PostgresTraceStore


def build_webhook(message: str, conversation_id: int = 123) -> ChatwootWebhook:
    return ChatwootWebhook(
        content=message,
        conversation={"id": conversation_id},
        contact={"id": 456, "name": "Ana"},
        account={"id": 999},
        event="message_created",
        message_type="incoming",
    )


def build_trace(
    *,
    message: str,
    next_node: str,
    stage_after: str = "",
    conversation_id: int = 123,
) -> TurnTraceContext:
    context = TurnTraceContext(
        flow_id=f"flow-{conversation_id}",
        webhook=build_webhook(message, conversation_id=conversation_id),
        llm_backend="raw",
        llm_model="gpt-test",
    )
    context.capture_state_after(
        {
            "conversation_id": str(conversation_id),
            "contact_id": "456",
            "contact_name": "Ana",
            "last_user_message": message,
            "next_node": next_node,
            "intent": next_node,
            "stage": stage_after,
        }
    )
    if next_node == "discovery_call":
        payload = DiscoveryCallIntentPayload(
            lead_name="Ana",
            project_need="automatizacion",
            preferred_date="mañana",
            preferred_time="10 am" if stage_after == "ready_for_handoff" else None,
            missing_fields=[] if stage_after == "ready_for_handoff" else ["preferred_time"],
            should_handoff=True,
            confidence=0.9,
        )
        context.capture_discovery_call(
            current_slots_before={"lead_name": "Ana"},
            payload_extracted=payload,
            slots_after_merge=payload.model_dump(),
            pending_question_after="" if stage_after == "ready_for_handoff" else "Necesito la hora preferida.",
            stage_after=stage_after,
        )
    context.capture_outbound(response_text="ok", sent=True)
    return context


async def _fetch_counts(db_url: str) -> tuple[list[TurnTraceORM], list[DiscoveryCallFlowORM]]:
    engine = create_async_engine(db_url, future=True)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as session:
        turns = list((await session.execute(select(TurnTraceORM))).scalars())
        flows = list((await session.execute(select(DiscoveryCallFlowORM))).scalars())
    await engine.dispose()
    return turns, flows


def test_postgres_trace_store_persists_turn_traces_and_discovery_flow(tmp_path: Path) -> None:
    pytest.importorskip("aiosqlite")
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'trace_store.db'}"
    settings = Settings(
        trace_capture_enabled=True,
        trace_capture_database_url=db_url,
        trace_capture_batch_size=10,
        trace_capture_flush_interval_ms=10,
        llm_api_key=None,
        openai_api_key=None,
    )
    store = PostgresTraceStore(settings)

    async def scenario() -> tuple[list[TurnTraceORM], list[DiscoveryCallFlowORM]]:
        await store.start()
        assert store.enqueue(build_trace(message="Quiero agendar una llamada", next_node="discovery_call", stage_after="collecting_slots").freeze())
        assert store.enqueue(build_trace(message="mañana a las 10", next_node="discovery_call", stage_after="ready_for_handoff").freeze())
        assert store.enqueue(build_trace(message="gracias", next_node="conversation", stage_after="open").freeze())
        await store.stop()
        return await _fetch_counts(db_url)

    turns, flows = asyncio.run(scenario())

    assert len(turns) == 3
    assert len(flows) == 1
    flow = flows[0]
    assert flow.status == "ready_for_handoff"
    assert flow.latest_stage == "ready_for_handoff"
    assert flow.final_payload is not None
    linked_turns = [turn for turn in turns if turn.discovery_call_flow_id == flow.id]
    assert len(linked_turns) == 2
