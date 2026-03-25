from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.models.schemas import ChatwootWebhook, DiscoveryCallIntentPayload, RoutingPacket, StateRoutingDecision


class WebhookSnapshot(BaseModel):
    event: str | None = None
    message_type: str | None = None
    latest_message: str = ""
    account_id: str | None = None
    conversation_id: str
    contact_id: str
    contact_name: str

    @classmethod
    def from_webhook(cls, webhook: ChatwootWebhook) -> "WebhookSnapshot":
        return cls(
            event=webhook.event,
            message_type=webhook.message_type,
            latest_message=webhook.latest_message,
            account_id=webhook.account_id,
            conversation_id=webhook.conversation_id,
            contact_id=webhook.contact_id,
            contact_name=webhook.contact_name,
        )


class GraphStateSnapshot(BaseModel):
    conversation_id: str = ""
    contact_id: str = ""
    contact_name: str = ""
    last_user_message: str = ""
    last_assistant_message: str = ""
    conversation_summary: str = ""
    active_goal: str = ""
    stage: str = ""
    pending_action: str = ""
    pending_question: str = ""
    discovery_call_slots: dict = Field(default_factory=dict)
    last_tool_result: str = ""
    next_node: str = ""
    intent: str = ""
    confidence: float = 0.0
    needs_retrieval: bool = False
    routing_reason: str = ""
    response_text: str = ""
    discovery_call_payload: dict = Field(default_factory=dict)
    handoff_required: bool = False
    turn_count: int = 0
    summary_refresh_requested: bool = False

    @classmethod
    def from_mapping(cls, state: dict | None) -> "GraphStateSnapshot":
        state = state or {}
        return cls(
            conversation_id=str(state.get("conversation_id", "") or ""),
            contact_id=str(state.get("contact_id", "") or ""),
            contact_name=str(state.get("contact_name", "") or ""),
            last_user_message=str(state.get("last_user_message", "") or ""),
            last_assistant_message=str(state.get("last_assistant_message", "") or ""),
            conversation_summary=str(state.get("conversation_summary", "") or ""),
            active_goal=str(state.get("active_goal", "") or ""),
            stage=str(state.get("stage", "") or ""),
            pending_action=str(state.get("pending_action", "") or ""),
            pending_question=str(state.get("pending_question", "") or ""),
            discovery_call_slots=dict(state.get("discovery_call_slots") or {}),
            last_tool_result=str(state.get("last_tool_result", "") or ""),
            next_node=str(state.get("next_node", "") or ""),
            intent=str(state.get("intent", "") or ""),
            confidence=float(state.get("confidence", 0.0) or 0.0),
            needs_retrieval=bool(state.get("needs_retrieval", False)),
            routing_reason=str(state.get("routing_reason", "") or ""),
            response_text=str(state.get("response_text", "") or ""),
            discovery_call_payload=dict(state.get("discovery_call_payload") or {}),
            handoff_required=bool(state.get("handoff_required", False)),
            turn_count=int(state.get("turn_count", 0) or 0),
            summary_refresh_requested=bool(state.get("summary_refresh_requested", False)),
        )


class RouteTracePayload(BaseModel):
    routing_packet: RoutingPacket
    decision: StateRoutingDecision


class RagChunkSnapshot(BaseModel):
    id: str
    score: float
    source: str = "unknown"
    text: str = ""


class RagTracePayload(BaseModel):
    company_context_hash: str
    retrieved_context_preview: str
    chunks: list[RagChunkSnapshot] = Field(default_factory=list)
    assistant_answer: str


class DiscoveryCallTracePayload(BaseModel):
    current_slots_before: dict = Field(default_factory=dict)
    payload_extracted: DiscoveryCallIntentPayload
    slots_after_merge: dict = Field(default_factory=dict)
    pending_question_after: str = ""
    stage_after: str = ""


class OutboundTracePayload(BaseModel):
    response_text: str = ""
    sent: bool = False
    error_type: str | None = None
    error_message: str | None = None


class TurnTraceRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    flow_id: str
    conversation_id: str
    contact_id: str
    account_id: str | None = None
    trace_version: int = 1
    llm_backend: str = "raw"
    llm_model: str = ""
    next_node: str = ""
    intent: str = ""
    needs_retrieval: bool = False
    handoff_required: bool = False
    response_sent: bool = False
    has_error: bool = False
    discovery_call_flow_id: UUID | None = None
    webhook_snapshot: WebhookSnapshot
    state_before: GraphStateSnapshot
    state_after: GraphStateSnapshot | None = None
    route_input: RoutingPacket | None = None
    route_output: StateRoutingDecision | None = None
    rag_trace: RagTracePayload | None = None
    discovery_call_trace: DiscoveryCallTracePayload | None = None
    outbound_trace: OutboundTracePayload | None = None
