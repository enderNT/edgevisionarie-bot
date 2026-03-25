from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.models.schemas import DiscoveryCallIntentPayload, RoutingPacket, StateRoutingDecision


def _memories_to_text(memories: list[str]) -> str:
    if not memories:
        return "- Sin memorias relevantes"
    return "\n".join(f"- {memory}" for memory in memories)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _as_str(value).lower() in {"true", "1", "yes", "si", "sí"}


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(_as_str(value))
    except ValueError:
        return 0.0


def _as_missing_fields(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _as_str(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in text.split(",")]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


@dataclass(slots=True)
class ConversationModuleInputs:
    user_message: str
    memories_text: str

    @classmethod
    def from_values(cls, user_message: str, memories: list[str]) -> "ConversationModuleInputs":
        return cls(user_message=user_message.strip(), memories_text=_memories_to_text(memories))

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "memories_text": self.memories_text,
        }


@dataclass(slots=True)
class RagModuleInputs:
    question: str
    memories_text: str
    context: str

    @classmethod
    def from_values(cls, question: str, memories: list[str], context: str) -> "RagModuleInputs":
        return cls(
            question=question.strip(),
            memories_text=_memories_to_text(memories),
            context=context.strip(),
        )

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "memories_text": self.memories_text,
            "context": self.context,
        }


@dataclass(slots=True)
class SummaryModuleInputs:
    current_summary: str
    active_goal: str
    stage: str
    user_message: str
    assistant_message: str

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "current_summary": self.current_summary.strip(),
            "active_goal": self.active_goal.strip(),
            "stage": self.stage.strip(),
            "user_message": self.user_message.strip(),
            "assistant_message": self.assistant_message.strip(),
        }


@dataclass(slots=True)
class RouteModuleInputs:
    user_message: str
    conversation_summary: str
    active_goal: str
    stage: str
    pending_action: str
    pending_question: str
    discovery_call_slots_json: str
    last_tool_result: str
    last_user_message: str
    last_assistant_message: str
    memories_text: str

    @classmethod
    def from_routing_packet(cls, packet: RoutingPacket) -> "RouteModuleInputs":
        return cls(
            user_message=packet.user_message,
            conversation_summary=packet.conversation_summary,
            active_goal=packet.active_goal,
            stage=packet.stage,
            pending_action=packet.pending_action,
            pending_question=packet.pending_question,
            discovery_call_slots_json=_json_dumps(packet.discovery_call_slots),
            last_tool_result=packet.last_tool_result,
            last_user_message=packet.last_user_message,
            last_assistant_message=packet.last_assistant_message,
            memories_text=_memories_to_text(packet.memories),
        )

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "conversation_summary": self.conversation_summary,
            "active_goal": self.active_goal,
            "stage": self.stage,
            "pending_action": self.pending_action,
            "pending_question": self.pending_question,
            "discovery_call_slots_json": self.discovery_call_slots_json,
            "last_tool_result": self.last_tool_result,
            "last_user_message": self.last_user_message,
            "last_assistant_message": self.last_assistant_message,
            "memories_text": self.memories_text,
        }


@dataclass(slots=True)
class RouteModuleOutputs:
    next_node: str
    intent: str
    confidence: float
    needs_retrieval: bool
    next_active_goal: str
    next_stage: str
    next_pending_action: str
    next_pending_question: str
    clear_slots: bool
    clear_last_tool_result: bool
    route_reason: str

    @classmethod
    def from_prediction(cls, prediction: Any) -> "RouteModuleOutputs":
        return cls(
            next_node=_as_str(getattr(prediction, "next_node", "")) or "conversation",
            intent=_as_str(getattr(prediction, "intent", "")) or "conversation",
            confidence=_as_float(getattr(prediction, "confidence", 0.0)),
            needs_retrieval=_as_bool(getattr(prediction, "needs_retrieval", False)),
            next_active_goal=_as_str(getattr(prediction, "next_active_goal", "")),
            next_stage=_as_str(getattr(prediction, "next_stage", "")),
            next_pending_action=_as_str(getattr(prediction, "next_pending_action", "")),
            next_pending_question=_as_str(getattr(prediction, "next_pending_question", "")),
            clear_slots=_as_bool(getattr(prediction, "clear_slots", False)),
            clear_last_tool_result=_as_bool(getattr(prediction, "clear_last_tool_result", False)),
            route_reason=_as_str(getattr(prediction, "route_reason", "")) or "dspy",
        )

    def to_state_routing_decision(self) -> StateRoutingDecision:
        state_update: dict[str, Any] = {}
        if self.next_active_goal:
            state_update["active_goal"] = self.next_active_goal
        if self.next_stage:
            state_update["stage"] = self.next_stage
        if self.next_pending_action:
            state_update["pending_action"] = self.next_pending_action
        if self.next_pending_question:
            state_update["pending_question"] = self.next_pending_question
        if self.clear_slots:
            state_update["discovery_call_slots"] = {}
        if self.clear_last_tool_result:
            state_update["last_tool_result"] = ""
        return StateRoutingDecision(
            next_node=self.next_node if self.next_node in {"conversation", "rag", "discovery_call"} else "conversation",
            intent=self.intent or self.next_node,
            confidence=max(0.0, min(self.confidence, 1.0)),
            needs_retrieval=self.needs_retrieval or self.next_node == "rag",
            state_update=state_update,
            reason=self.route_reason,
        )


@dataclass(slots=True)
class DiscoveryCallModuleInputs:
    user_message: str
    memories_text: str
    company_context: str
    contact_name: str
    current_slots_json: str
    pending_question: str

    @classmethod
    def from_values(
        cls,
        user_message: str,
        memories: list[str],
        company_context: str,
        contact_name: str,
        current_slots: dict[str, Any] | None,
        pending_question: str | None,
    ) -> "DiscoveryCallModuleInputs":
        return cls(
            user_message=user_message.strip(),
            memories_text=_memories_to_text(memories),
            company_context=company_context.strip(),
            contact_name=contact_name.strip(),
            current_slots_json=_json_dumps(current_slots or {}),
            pending_question=(pending_question or "").strip(),
        )

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "memories_text": self.memories_text,
            "company_context": self.company_context,
            "contact_name": self.contact_name,
            "current_slots_json": self.current_slots_json,
            "pending_question": self.pending_question,
        }


@dataclass(slots=True)
class DiscoveryCallModuleOutputs:
    lead_name: str | None
    project_need: str | None
    preferred_date: str | None
    preferred_time: str | None
    missing_fields: list[str]
    should_handoff: bool
    confidence: float

    @classmethod
    def from_prediction(cls, prediction: Any) -> "DiscoveryCallModuleOutputs":
        lead_name = _as_str(getattr(prediction, "lead_name", "")) or None
        project_need = _as_str(getattr(prediction, "project_need", "")) or None
        preferred_date = _as_str(getattr(prediction, "preferred_date", "")) or None
        preferred_time = _as_str(getattr(prediction, "preferred_time", "")) or None
        return cls(
            lead_name=lead_name,
            project_need=project_need,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            missing_fields=_as_missing_fields(getattr(prediction, "missing_fields", [])),
            should_handoff=_as_bool(getattr(prediction, "should_handoff", True)),
            confidence=_as_float(getattr(prediction, "confidence", 0.0)),
        )

    def to_payload(self) -> DiscoveryCallIntentPayload:
        return DiscoveryCallIntentPayload(
            lead_name=self.lead_name,
            project_need=self.project_need,
            preferred_date=self.preferred_date,
            preferred_time=self.preferred_time,
            missing_fields=self.missing_fields,
            should_handoff=self.should_handoff,
            confidence=max(0.0, min(self.confidence, 1.0)),
        )
