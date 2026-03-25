from __future__ import annotations

import logging
import re
from typing import Any

from app.models.schemas import RoutingPacket, StateRoutingDecision
from app.observability.flow_logger import substep
from app.services.llm import SupportLLMService
from app.settings import Settings

logger = logging.getLogger(__name__)


class StateRoutingService:
    def __init__(self, settings: Settings, llm_service: SupportLLMService) -> None:
        self._settings = settings
        self._llm_service = llm_service

    def build_routing_packet(
        self,
        *,
        user_message: str,
        conversation_summary: str,
        active_goal: str,
        stage: str,
        pending_action: str,
        pending_question: str,
        discovery_call_slots: dict[str, Any],
        last_tool_result: str,
        last_user_message: str,
        last_assistant_message: str,
        memories: list[str],
    ) -> RoutingPacket:
        return RoutingPacket(
            user_message=_compact_text(user_message, 400),
            conversation_summary=_compact_text(conversation_summary, 500),
            active_goal=_compact_text(active_goal, 80),
            stage=_compact_text(stage, 80),
            pending_action=_compact_text(pending_action, 120),
            pending_question=_compact_text(pending_question, 200),
            discovery_call_slots={
                key: _compact_text(str(value), 120) for key, value in discovery_call_slots.items()
            },
            last_tool_result=_compact_text(last_tool_result, 280),
            last_user_message=_compact_text(last_user_message, 280),
            last_assistant_message=_compact_text(last_assistant_message, 280),
            memories=[_compact_text(memory, 160) for memory in memories[:3]],
        )

    async def route_state(
        self,
        *,
        user_message: str,
        conversation_summary: str,
        active_goal: str,
        stage: str,
        pending_action: str,
        pending_question: str,
        discovery_call_slots: dict[str, Any],
        last_tool_result: str,
        last_user_message: str,
        last_assistant_message: str,
        memories: list[str],
    ) -> StateRoutingDecision:
        routing_packet = self.build_routing_packet(
            user_message=user_message,
            conversation_summary=conversation_summary,
            active_goal=active_goal,
            stage=stage,
            pending_action=pending_action,
            pending_question=pending_question,
            discovery_call_slots=discovery_call_slots,
            last_tool_result=last_tool_result,
            last_user_message=last_user_message,
            last_assistant_message=last_assistant_message,
            memories=memories,
        )
        guard_hint = self._deterministic_guard(routing_packet)
        if guard_hint is not None:
            substep("state_router_guard", "OK", guard_hint.reason)
            return guard_hint

        decision = await self._llm_service.classify_state_route(routing_packet)
        if decision.next_node == "rag" and not decision.needs_retrieval:
            decision.needs_retrieval = True
        substep(
            "state_router_llm",
            "OK",
            f"next={decision.next_node} intent={decision.intent} confidence={decision.confidence:.2f}",
        )
        return decision

    def summarize_memories(self, memories: list[str]) -> list[str]:
        summarized: list[str] = []
        for memory in memories[:3]:
            compact = _compact_text(memory, 140)
            if compact:
                summarized.append(compact)
        return summarized

    def _deterministic_guard(self, routing_packet: RoutingPacket) -> StateRoutingDecision | None:
        user_message = routing_packet.user_message.lower().strip()
        if not user_message:
            return StateRoutingDecision(
                next_node="conversation",
                intent="conversation",
                confidence=0.3,
                needs_retrieval=False,
                state_update={},
                reason="empty-message",
            )

        if self._discovery_call_follow_up(routing_packet, user_message):
            return StateRoutingDecision(
                next_node="discovery_call",
                intent="discovery_call",
                confidence=0.95,
                needs_retrieval=False,
                state_update={
                    "active_goal": "discovery_call",
                    "stage": "collecting_slots",
                    "pending_action": "collecting_slots",
                },
                reason="discovery-call-follow-up",
            )

        if self._explicit_discovery_call_request(user_message):
            return StateRoutingDecision(
                next_node="discovery_call",
                intent="discovery_call",
                confidence=0.92,
                needs_retrieval=False,
                state_update={
                    "active_goal": "discovery_call",
                    "stage": "collecting_slots",
                    "pending_action": "collecting_slots",
                },
                reason="discovery-call-request",
            )

        if self._explicit_rag_request(user_message):
            return StateRoutingDecision(
                next_node="rag",
                intent="rag",
                confidence=0.86,
                needs_retrieval=True,
                state_update={
                    "active_goal": "information",
                    "stage": "lookup",
                },
                reason="information-request",
            )

        if self._is_simple_conversation(user_message):
            return StateRoutingDecision(
                next_node="conversation",
                intent="conversation",
                confidence=0.9,
                needs_retrieval=False,
                state_update={
                    "active_goal": routing_packet.active_goal or "conversation",
                    "stage": routing_packet.stage or "open",
                },
                reason="simple-conversation",
            )

        return None

    def _discovery_call_follow_up(self, routing_packet: RoutingPacket, user_message: str) -> bool:
        active_discovery_call = routing_packet.active_goal == "discovery_call" or routing_packet.stage in {
            "collecting_slots",
            "ready_for_handoff",
        }
        if not active_discovery_call:
            return False
        if routing_packet.pending_question:
            return True
        if routing_packet.discovery_call_slots and len(user_message) <= 40:
            return True
        return bool(
            re.search(
                r"\b(si|sí|no|claro|mañana|manana|hoy|tarde|noche|am|pm|\d{1,2}:\d{2}|\d{1,2}\s?am|\d{1,2}\s?pm)\b",
                user_message,
            )
        )

    def _explicit_discovery_call_request(self, user_message: str) -> bool:
        discovery_call_keywords = (
            "agendar",
            "llamada",
            "reunión",
            "reunion",
            "cotización",
            "cotizacion",
            "proyecto",
            "hablar con alguien",
            "asesoría",
            "asesoria",
            "demo",
            "sesión",
            "sesion",
        )
        return any(keyword in user_message for keyword in discovery_call_keywords)

    def _explicit_rag_request(self, user_message: str) -> bool:
        rag_keywords = (
            "servicios",
            "precio",
            "precios",
            "costo",
            "costos",
            "tiempo",
            "tiempos",
            "stack",
            "tecnología",
            "tecnologias",
            "tecnología",
            "integraciones",
            "automatización",
            "automatizacion",
            "ia",
            "inteligencia artificial",
            "mantenimiento",
            "soporte",
            "portafolio",
            "casos de éxito",
            "casos de exito",
            "horario",
            "horarios",
            "política",
            "politica",
            "pago",
            "pagos",
        )
        return any(keyword in user_message for keyword in rag_keywords)

    def _is_simple_conversation(self, user_message: str) -> bool:
        compact = " ".join(user_message.split())
        if compact in {"hola", "buenas", "buenos dias", "buenas tardes", "gracias", "ok", "okay", "si", "sí"}:
            return True
        if len(compact) <= 6:
            return True
        return any(marker in compact for marker in ("hola", "gracias", "perfecto", "entendido"))


def _compact_text(value: str, max_len: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[: max_len - 3]}..."
