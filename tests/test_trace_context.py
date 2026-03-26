from app.models.schemas import ChatwootWebhook, DiscoveryCallIntentPayload, RoutingPacket, StateRoutingDecision
from app.traces.context import TurnTraceContext
from app.traces.models import RagChunkSnapshot


def build_webhook(message: str = "Quiero una llamada") -> ChatwootWebhook:
    return ChatwootWebhook(
        content=message,
        conversation={"id": 123},
        contact={"id": 456, "name": "Ana"},
        account={"id": 999},
        event="message_created",
        message_type="incoming",
    )


def test_turn_trace_context_captures_route_rag_discovery_and_outbound() -> None:
    context = TurnTraceContext(
        flow_id="flow-123",
        webhook=build_webhook(),
        llm_backend="raw",
        llm_model="gpt-test",
    )

    routing_packet = RoutingPacket(
        user_message="Quiero agendar una llamada",
        active_goal="conversation",
        stage="open",
        memories=["Necesita automatizacion"],
    )
    decision = StateRoutingDecision(
        next_node="discovery_call",
        intent="discovery_call",
        confidence=0.93,
        needs_retrieval=False,
        state_update={"active_goal": "discovery_call", "stage": "collecting_slots"},
        reason="discovery-call-request",
    )
    context.capture_route(routing_packet, decision)
    payload = DiscoveryCallIntentPayload(
        lead_name="Ana",
        project_need="automatizacion",
        preferred_date="mañana",
        missing_fields=["preferred_time"],
        should_handoff=True,
        confidence=0.88,
    )
    context.capture_rag(
        company_context_hash="abc123",
        retrieved_context_preview="contexto",
        chunks=[RagChunkSnapshot(id="chunk-1", score=0.91, source="faq", text="texto")],
        assistant_answer="respuesta rag",
    )
    context.capture_discovery_call(
        current_slots_before={"lead_name": "Ana"},
        payload_extracted=payload,
        slots_after_merge={"lead_name": "Ana", "project_need": "automatizacion"},
        pending_question_after="Ya puedes elegir un horario en Calendly.",
        stage_after="collecting_slots",
    )
    context.capture_state_after(
        {
            "conversation_id": "123",
            "contact_id": "456",
            "contact_name": "Ana",
            "last_user_message": "Quiero agendar una llamada",
            "last_assistant_message": "Necesito la hora preferida",
            "next_node": "discovery_call",
            "intent": "discovery_call",
            "handoff_required": True,
        }
    )
    context.capture_outbound(response_text="Necesito la hora preferida", sent=True)

    record = context.freeze()

    assert record.flow_id == "flow-123"
    assert record.route_input is not None
    assert record.route_input.user_message == "Quiero agendar una llamada"
    assert record.route_output is not None
    assert record.route_output.next_node == "discovery_call"
    assert record.rag_trace is not None
    assert record.rag_trace.chunks[0].id == "chunk-1"
    assert record.discovery_call_trace is not None
    assert record.discovery_call_trace.pending_question_after.startswith("Ya puedes elegir")
    assert record.outbound_trace is not None
    assert record.outbound_trace.sent is True
    assert record.state_after is not None
    assert record.state_after.intent == "discovery_call"
