from __future__ import annotations

import contextvars

from app.models.schemas import ChatwootWebhook, DiscoveryCallIntentPayload, RoutingPacket, StateRoutingDecision
from app.traces.models import (
    DiscoveryCallTracePayload,
    GraphStateSnapshot,
    OutboundTracePayload,
    RagChunkSnapshot,
    RagTracePayload,
    TurnTraceRecord,
    WebhookSnapshot,
)

_trace_ctx_var: contextvars.ContextVar["TurnTraceContext | None"] = contextvars.ContextVar(
    "turn_trace_context", default=None
)


class TurnTraceContext:
    def __init__(
        self,
        *,
        flow_id: str,
        webhook: ChatwootWebhook,
        llm_backend: str,
        llm_model: str,
    ) -> None:
        self._record = TurnTraceRecord(
            flow_id=flow_id,
            conversation_id=webhook.conversation_id,
            contact_id=webhook.contact_id,
            account_id=webhook.account_id,
            llm_backend=llm_backend,
            llm_model=llm_model,
            webhook_snapshot=WebhookSnapshot.from_webhook(webhook),
            state_before=GraphStateSnapshot(
                conversation_id=webhook.conversation_id,
                contact_id=webhook.contact_id,
                contact_name=webhook.contact_name,
                contact_email=webhook.contact_email or "",
                last_user_message=webhook.latest_message,
            ),
        )

    def capture_route(self, routing_packet: RoutingPacket, decision: StateRoutingDecision) -> None:
        self._record.route_input = routing_packet
        self._record.route_output = decision
        self._record.next_node = decision.next_node
        self._record.intent = decision.intent
        self._record.needs_retrieval = decision.needs_retrieval

    def capture_rag(
        self,
        *,
        company_context_hash: str,
        retrieved_context_preview: str,
        chunks: list[RagChunkSnapshot],
        assistant_answer: str,
    ) -> None:
        self._record.rag_trace = RagTracePayload(
            company_context_hash=company_context_hash,
            retrieved_context_preview=retrieved_context_preview,
            chunks=chunks,
            assistant_answer=assistant_answer,
        )

    def capture_discovery_call(
        self,
        *,
        current_slots_before: dict,
        payload_extracted: DiscoveryCallIntentPayload,
        slots_after_merge: dict,
        pending_question_after: str,
        stage_after: str,
    ) -> None:
        self._record.discovery_call_trace = DiscoveryCallTracePayload(
            current_slots_before=current_slots_before,
            payload_extracted=payload_extracted,
            slots_after_merge=slots_after_merge,
            pending_question_after=pending_question_after,
            stage_after=stage_after,
        )
        self._record.handoff_required = payload_extracted.should_handoff

    def capture_state_after(self, state: dict) -> None:
        snapshot = GraphStateSnapshot.from_mapping(state)
        self._record.state_after = snapshot
        self._record.next_node = snapshot.next_node or self._record.next_node
        self._record.intent = snapshot.intent or self._record.intent
        self._record.needs_retrieval = snapshot.needs_retrieval
        self._record.handoff_required = snapshot.handoff_required

    def capture_outbound(self, *, response_text: str, sent: bool, error: Exception | None = None) -> None:
        self._record.outbound_trace = OutboundTracePayload(
            response_text=response_text,
            sent=sent,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
        )
        self._record.response_sent = sent
        if error is not None:
            self._record.has_error = True

    def mark_error(self, exc: Exception) -> None:
        self._record.has_error = True
        if self._record.outbound_trace is None:
            self._record.outbound_trace = OutboundTracePayload(
                response_text="",
                sent=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def freeze(self) -> TurnTraceRecord:
        return self._record.model_copy(deep=True)


def bind_turn_trace_context(context: TurnTraceContext) -> None:
    _trace_ctx_var.set(context)


def get_turn_trace_context() -> TurnTraceContext | None:
    return _trace_ctx_var.get()


def clear_turn_trace_context() -> None:
    _trace_ctx_var.set(None)
