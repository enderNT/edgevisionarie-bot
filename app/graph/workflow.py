from __future__ import annotations

import hashlib
import logging
import re
from copy import deepcopy
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.models.schemas import ChatwootWebhook, DiscoveryCallIntentPayload
from app.observability.flow_logger import mark_error, step, substep
from app.services.calendly import CalendlyService, format_calendly_time
from app.services.company_config import CompanyConfigLoader
from app.services.llm import SupportLLMService
from app.services.memory import MemoryStore, should_store_memory
from app.services.qdrant import QdrantRetrievalService
from app.services.router import StateRoutingService
from app.settings import Settings
from app.traces.context import get_turn_trace_context
from app.traces.models import RagChunkSnapshot

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    conversation_id: str
    contact_id: str
    contact_name: str
    last_user_message: str
    last_assistant_message: str
    conversation_summary: str
    active_goal: str
    stage: str
    contact_email: str
    pending_action: str
    pending_question: str
    discovery_call_slots: dict[str, Any]
    last_tool_result: str
    memories: list[str]
    next_node: str
    intent: str
    confidence: float
    needs_retrieval: bool
    routing_reason: str
    state_update: dict[str, Any]
    response_text: str
    discovery_call_payload: dict[str, Any]
    handoff_required: bool
    turn_count: int
    summary_refresh_requested: bool


class SupportWorkflow:
    def __init__(
        self,
        router_service: StateRoutingService,
        llm_service: SupportLLMService,
        calendly_service: CalendlyService,
        memory_store: MemoryStore,
        company_config_loader: CompanyConfigLoader,
        qdrant_service: QdrantRetrievalService,
        settings: Settings,
    ) -> None:
        self._router_service = router_service
        self._llm_service = llm_service
        self._calendly_service = calendly_service
        self._memory_store = memory_store
        self._company_config_loader = company_config_loader
        self._qdrant_service = qdrant_service
        self._settings = settings
        self._graph = self._build_graph()

    @property
    def llm_backend_name(self) -> str:
        return getattr(self._llm_service, "backend_name", "raw")

    @property
    def llm_model_name(self) -> str:
        return getattr(self._llm_service, "model_name", "")

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("route", self._route)
        graph.add_node("conversation", self._conversation)
        graph.add_node("rag", self._rag)
        graph.add_node("discovery_call", self._discovery_call)
        graph.add_node("finalize_turn", self._finalize_turn)
        graph.add_node("store_memory", self._store_memory)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "route")
        graph.add_conditional_edges(
            "route",
            self._branch_after_route,
            {
                "conversation": "conversation",
                "rag": "rag",
                "discovery_call": "discovery_call",
            },
        )
        graph.add_edge("conversation", "finalize_turn")
        graph.add_edge("rag", "finalize_turn")
        graph.add_edge("discovery_call", "finalize_turn")
        graph.add_edge("finalize_turn", "store_memory")
        graph.add_edge("store_memory", END)
        return graph.compile(checkpointer=MemorySaver())

    async def run(self, webhook: ChatwootWebhook) -> GraphState:
        initial_state: GraphState = {
            "conversation_id": webhook.conversation_id,
            "contact_id": webhook.contact_id,
            "contact_name": webhook.contact_name,
            "contact_email": webhook.contact_email or "",
            "last_user_message": webhook.latest_message,
        }
        config = {"configurable": {"thread_id": webhook.conversation_id}}
        return await self._graph.ainvoke(initial_state, config=config)

    async def _load_context(self, state: GraphState) -> GraphState:
        try:
            step("2.1 build_context", "RUN", "cargando estado corto y memorias duraderas")
            memories = await self._memory_store.search(
                state["contact_id"],
                query=state.get("last_user_message") or state.get("conversation_summary") or "contexto del usuario",
                limit=self._settings.memory_search_limit,
            )
            substep("mem0_lookup", "OK", f"memories={len(memories)}")
            step("2.1 build_context", "OK")
            turn_count = int(state.get("turn_count", 0)) + 1
            return {
                "turn_count": turn_count,
                "memories": self._router_service.summarize_memories(memories),
            }
        except Exception as exc:
            mark_error("2.1 build_context", exc)
            raise

    async def _route(self, state: GraphState) -> GraphState:
        try:
            step("2.2 state_router", "RUN", "clasificando con estado compacto")
            routing_packet = self._router_service.build_routing_packet(
                user_message=state["last_user_message"],
                conversation_summary=state.get("conversation_summary", ""),
                active_goal=state.get("active_goal", ""),
                stage=state.get("stage", ""),
                pending_action=state.get("pending_action", ""),
                pending_question=state.get("pending_question", ""),
                discovery_call_slots=state.get("discovery_call_slots", {}),
                last_tool_result=state.get("last_tool_result", ""),
                last_user_message=state.get("last_user_message", ""),
                last_assistant_message=state.get("last_assistant_message", ""),
                memories=state.get("memories", []),
            )
            decision = await self._router_service.route_state(
                user_message=state["last_user_message"],
                conversation_summary=state.get("conversation_summary", ""),
                active_goal=state.get("active_goal", ""),
                stage=state.get("stage", ""),
                pending_action=state.get("pending_action", ""),
                pending_question=state.get("pending_question", ""),
                discovery_call_slots=state.get("discovery_call_slots", {}),
                last_tool_result=state.get("last_tool_result", ""),
                last_user_message=state.get("last_user_message", ""),
                last_assistant_message=state.get("last_assistant_message", ""),
                memories=state.get("memories", []),
            )
            trace_context = get_turn_trace_context()
            if trace_context is not None:
                trace_context.capture_route(routing_packet, decision)
            merged_state = self._apply_state_update(state, decision.state_update)
            merged_state.update(
                {
                    "next_node": decision.next_node,
                    "intent": decision.intent,
                    "confidence": decision.confidence,
                    "needs_retrieval": decision.needs_retrieval,
                    "routing_reason": decision.reason,
                    "state_update": decision.state_update,
                    "summary_refresh_requested": merged_state.get("summary_refresh_requested", False)
                    or merged_state.get("active_goal") != state.get("active_goal"),
                }
            )
            step(
                "2.2 state_router",
                "OK",
                f"next={decision.next_node} intent={decision.intent} confidence={decision.confidence:.2f}",
            )
            return merged_state
        except Exception as exc:
            mark_error("2.2 state_router", exc)
            raise

    def _branch_after_route(self, state: GraphState) -> str:
        branch = state.get("next_node", "conversation")
        step("3. branch_selection", "OK", f"selected={branch}")
        if branch == "conversation":
            substep("3.a conversation", "OK", "usando nodo conversacional")
        elif branch == "rag":
            substep("3.b rag", "OK", "usando nodo RAG")
        elif branch == "discovery_call":
            substep("3.c discovery_call", "OK", "usando nodo de discovery call")
        else:
            substep("3.x unknown_branch", "WARN", f"branch={branch}; fallback a conversation")
            return "conversation"
        return branch

    async def _conversation(self, state: GraphState) -> GraphState:
        try:
            step("3.a.1 conversation_node", "RUN", "generando respuesta")
            response_text = await self._llm_service.build_conversation_reply(
                user_message=state["last_user_message"],
                memories=state.get("memories", []),
            )
            step("3.a.1 conversation_node", "OK", f"chars={len(response_text)}")
            return {
                "response_text": response_text,
                "last_assistant_message": response_text,
                "last_tool_result": "",
                "handoff_required": False,
                "discovery_call_payload": {},
            }
        except Exception as exc:
            mark_error("3.a.1 conversation_node", exc)
            raise

    async def _rag(self, state: GraphState) -> GraphState:
        try:
            step("3.b.1 rag_node", "RUN", "consultando contexto RAG")
            company_context = self._company_config_loader.load().to_context_text()
            substep("company_config", "OK", "config estatica cargada")
            rag_context, rag_results = await self._qdrant_service.build_context_bundle(
                query=state["last_user_message"] or "contexto del usuario",
                company_context=company_context,
                memories=state.get("memories", []),
            )
            substep("qdrant_lookup", "OK", "contexto vectorial preparado")
            response_text = await self._llm_service.build_rag_reply(
                user_message=state["last_user_message"],
                memories=state.get("memories", []),
                company_context=rag_context,
            )
            trace_context = get_turn_trace_context()
            if trace_context is not None:
                trace_context.capture_rag(
                    company_context_hash=hashlib.sha256(company_context.encode("utf-8")).hexdigest(),
                    retrieved_context_preview=_shorten(rag_context, 800),
                    chunks=[
                        RagChunkSnapshot(
                            id=result.id,
                            score=result.score,
                            source=str(result.payload.get("source", "unknown")),
                            text=str(result.payload.get("text", "")),
                        )
                        for result in rag_results
                    ],
                    assistant_answer=response_text,
                )
            step("3.b.1 rag_node", "OK", f"chars={len(response_text)}")
            return {
                "last_tool_result": _shorten(rag_context, 240),
                "response_text": response_text,
                "last_assistant_message": response_text,
                "handoff_required": False,
                "discovery_call_payload": {},
            }
        except Exception as exc:
            mark_error("3.b.1 rag_node", exc)
            raise

    async def _discovery_call(self, state: GraphState) -> GraphState:
        try:
            step("3.c.1 discovery_call_node", "RUN", "extrayendo datos de discovery call")
            company_context = self._company_config_loader.load().to_context_text()
            substep("company_config", "OK", "config estatica cargada")
            current_slots_before = dict(state.get("discovery_call_slots", {}))
            booking_link = self._calendly_service.scheduling_url
            current_stage = str(state.get("stage", "") or "")
            if current_stage in {"awaiting_calendar_choice", "awaiting_booking_confirmation", "booking_confirmed"}:
                return await self._handle_discovery_call_booking_stage(
                    state=state,
                    company_context=company_context,
                    current_slots_before=current_slots_before,
                    booking_link=booking_link,
                )

            discovery_call, response_text = await self._llm_service.extract_discovery_call_intent(
                user_message=state["last_user_message"],
                memories=state.get("memories", []),
                company_context=company_context,
                contact_name=state["contact_name"],
                current_slots=current_slots_before,
                pending_question=state.get("pending_question"),
                calendly_link=booking_link,
            )
            discovery_call_slots = _merge_slots(current_slots_before, discovery_call.model_dump())
            missing_fields = [field for field in discovery_call.missing_fields if field in {"lead_name", "project_need"}]
            if missing_fields:
                pending_question = _build_pending_question(missing_fields)
                stage = "collecting_slots"
                pending_action = "collecting_slots"
            else:
                stage = "awaiting_calendar_choice"
                pending_action = "awaiting_calendar_choice"
                pending_question = "Ya puedes elegir un horario en Calendly. Cuando lo tengas, dime si ya lo reservaste."
                response_text = await self._llm_service.build_discovery_call_booking_reply(
                    user_message=state["last_user_message"],
                    contact_name=state["contact_name"],
                    calendly_link=booking_link,
                    stage=stage,
                )
                discovery_call_slots = _merge_slots(
                    discovery_call_slots,
                    {
                        "calendly_booking_status": stage,
                        "calendly_booking_url": booking_link,
                    },
                )
            substep(
                "discovery_call_payload",
                "OK",
                f"missing_fields={len(missing_fields)} handoff={discovery_call.should_handoff}",
            )
            trace_context = get_turn_trace_context()
            if trace_context is not None:
                trace_context.capture_discovery_call(
                    current_slots_before=current_slots_before,
                    payload_extracted=discovery_call,
                    slots_after_merge=discovery_call_slots,
                    pending_question_after=pending_question,
                    stage_after=stage,
                )
            step("3.c.1 discovery_call_node", "OK", f"chars={len(response_text)}")
            return {
                "response_text": response_text,
                "last_assistant_message": response_text,
                "discovery_call_slots": discovery_call_slots,
                "pending_question": pending_question,
                "pending_action": pending_action,
                "active_goal": "discovery_call",
                "stage": stage,
                "last_tool_result": _shorten(
                    (
                        f"discovery_call missing={','.join(missing_fields) or 'none'} "
                        f"confidence={discovery_call.confidence:.2f}"
                    ),
                    200,
                ),
                "handoff_required": discovery_call.should_handoff,
                "discovery_call_payload": discovery_call.model_dump(),
            }
        except Exception as exc:
            mark_error("3.c.1 discovery_call_node", exc)
            raise

    async def _handle_discovery_call_booking_stage(
        self,
        *,
        state: GraphState,
        company_context: str,
        current_slots_before: dict[str, Any],
        booking_link: str,
    ) -> GraphState:
        del company_context
        user_message = state["last_user_message"]
        contact_name = state["contact_name"]
        contact_email = str(state.get("contact_email", "") or "")
        current_stage = str(state.get("stage", "") or "")
        response_text = ""
        discovery_call_slots = dict(current_slots_before)
        pending_action = current_stage

        if current_stage == "awaiting_calendar_choice" and not self._looks_like_booking_confirmation(user_message):
            response_text = await self._llm_service.build_discovery_call_booking_reply(
                user_message=user_message,
                contact_name=contact_name,
                calendly_link=booking_link,
                stage=current_stage,
            )
            discovery_call_slots = _merge_slots(
                discovery_call_slots,
                {
                    "calendly_booking_status": current_stage,
                    "calendly_booking_url": booking_link,
                },
            )
            return {
                "response_text": response_text,
                "last_assistant_message": response_text,
                "discovery_call_slots": discovery_call_slots,
                "pending_question": "Ya puedes elegir un horario en Calendly. Cuando lo tengas, dime si ya lo reservaste.",
                "pending_action": pending_action,
                "active_goal": "discovery_call",
                "stage": current_stage,
                "last_tool_result": _shorten("waiting-for-calendar-choice", 200),
                "handoff_required": False,
                "discovery_call_payload": state.get("discovery_call_payload", {}),
            }

        booking_email = self._extract_email(user_message) or contact_email or str(discovery_call_slots.get("booking_email", "") or "")
        if not booking_email:
            response_text = await self._llm_service.build_discovery_call_booking_reply(
                user_message=user_message,
                contact_name=contact_name,
                calendly_link=booking_link,
                stage="awaiting_booking_confirmation",
            )
            discovery_call_slots = _merge_slots(
                discovery_call_slots,
                {
                    "calendly_booking_status": "awaiting_booking_confirmation",
                    "calendly_booking_url": booking_link,
                },
            )
            return {
                "response_text": response_text,
                "last_assistant_message": response_text,
                "discovery_call_slots": discovery_call_slots,
                "pending_question": "Comparteme el correo con el que reservaste para validar la cita.",
                "pending_action": "awaiting_booking_confirmation",
                "active_goal": "discovery_call",
                "stage": "awaiting_booking_confirmation",
                "last_tool_result": _shorten("waiting-for-booking-email", 200),
                "handoff_required": False,
                "discovery_call_payload": state.get("discovery_call_payload", {}),
            }

        validation = await self._calendly_service.validate_booking_by_email(booking_email)
        if validation.found and validation.match is not None:
            booking_start_time = validation.match.start_time or ""
            formatted_time = format_calendly_time(booking_start_time) if booking_start_time else ""
            response_text = await self._llm_service.build_discovery_call_booking_reply(
                user_message=user_message,
                contact_name=contact_name,
                calendly_link=booking_link,
                stage="booking_confirmed",
                booking_email=booking_email,
                booking_start_time=booking_start_time,
            )
            discovery_call_slots = _merge_slots(
                discovery_call_slots,
                {
                    "booking_email": booking_email,
                    "calendly_booking_status": "booking_confirmed",
                    "calendly_booking_url": booking_link,
                    "calendly_booking_start_time": booking_start_time,
                    "calendly_booking_event_uri": validation.match.scheduled_event_uri,
                    "calendly_booking_invitee_uri": validation.match.invitee_uri,
                },
            )
            trace_context = get_turn_trace_context()
            if trace_context is not None:
                trace_context.capture_discovery_call(
                    current_slots_before=current_slots_before,
                    payload_extracted=DiscoveryCallIntentPayload(
                        lead_name=str(discovery_call_slots.get("lead_name") or contact_name or ""),
                        project_need=str(discovery_call_slots.get("project_need") or ""),
                        preferred_date=str(discovery_call_slots.get("preferred_date") or "") or None,
                        preferred_time=str(discovery_call_slots.get("preferred_time") or "") or None,
                        missing_fields=[],
                        should_handoff=False,
                        confidence=1.0,
                    ),
                    slots_after_merge=discovery_call_slots,
                    pending_question_after="",
                    stage_after="booking_confirmed",
                )
            last_tool_result = _shorten(
                "booking-confirmed "
                + " ".join(
                    fragment
                    for fragment in [
                        f"email={booking_email}",
                        f"start={formatted_time or booking_start_time}",
                        f"event={validation.match.scheduled_event_uri}",
                    ]
                    if fragment
                ),
                240,
            )
            return {
                "response_text": response_text,
                "last_assistant_message": response_text,
                "discovery_call_slots": discovery_call_slots,
                "pending_question": "",
                "pending_action": "",
                "active_goal": "conversation",
                "stage": "open",
                "last_tool_result": last_tool_result,
                "handoff_required": False,
                "discovery_call_payload": {},
                "next_node": "conversation",
            }

        response_text = await self._llm_service.build_discovery_call_booking_reply(
            user_message=user_message,
            contact_name=contact_name,
            calendly_link=booking_link,
            stage="awaiting_calendar_choice",
            booking_email=booking_email,
        )
        if booking_email:
            response_text = (
                f"No encontre una reserva confirmada con {booking_email}. "
                + response_text
            )
        discovery_call_slots = _merge_slots(
            discovery_call_slots,
            {
                "booking_email": booking_email,
                "calendly_booking_status": "awaiting_calendar_choice",
                "calendly_booking_url": booking_link,
            },
        )
        return {
            "response_text": response_text,
            "last_assistant_message": response_text,
            "discovery_call_slots": discovery_call_slots,
            "pending_question": "Ya puedes elegir un horario en Calendly. Cuando lo tengas, dime si ya lo reservaste.",
            "pending_action": "awaiting_calendar_choice",
            "active_goal": "discovery_call",
            "stage": "awaiting_calendar_choice",
            "last_tool_result": _shorten(f"booking-not-found email={booking_email}", 200),
            "handoff_required": False,
            "discovery_call_payload": state.get("discovery_call_payload", {}),
        }

    @staticmethod
    def _looks_like_booking_confirmation(user_message: str) -> bool:
        lowered = " ".join(user_message.lower().split())
        return any(
            marker in lowered
            for marker in (
                "ya elegi",
                "ya elegí",
                "ya lo reserve",
                "ya lo reservé",
                "ya agende",
                "ya agendé",
                "ya esta",
                "ya está",
                "si ya",
                "sí ya",
                "listo",
                "confirmado",
                "reservado",
            )
        ) or bool(
            re.search(
                r"\b(si|sí|ya|listo|confirmado|reservado|agendado|agendada)\b",
                lowered,
            )
        )

    @staticmethod
    def _extract_email(user_message: str) -> str:
        match = re.search(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", user_message)
        return match.group(1) if match else ""

    async def _finalize_turn(self, state: GraphState) -> GraphState:
        try:
            step("3.9 finalize_turn", "RUN", "limpiando estado y refrescando resumen si hace falta")
            cleaned_state = self._cleanup_state(state)
            if cleaned_state.get("summary_refresh_requested") or self._needs_summary_refresh(cleaned_state):
                summary = await self._llm_service.build_state_summary(
                    current_summary=cleaned_state.get("conversation_summary", ""),
                    user_message=cleaned_state.get("last_user_message", ""),
                    assistant_message=cleaned_state.get("last_assistant_message", ""),
                    active_goal=cleaned_state.get("active_goal", ""),
                    stage=cleaned_state.get("stage", ""),
                )
                cleaned_state["conversation_summary"] = _shorten(summary, 700)
                cleaned_state["summary_refresh_requested"] = False
            cleaned_state["turn_count"] = int(cleaned_state.get("turn_count", 0))
            step("3.9 finalize_turn", "OK", "estado limpio")
            return cleaned_state
        except Exception as exc:
            mark_error("3.9 finalize_turn", exc)
            raise

    async def _store_memory(self, state: GraphState) -> GraphState:
        response_text = state.get("response_text", "")
        user_message = state.get("last_user_message", "")
        contact_id = state.get("contact_id")
        route = state.get("next_node", "conversation")
        if response_text and user_message and contact_id:
            memories = should_store_memory(user_message, response_text, route, state)
            if memories:
                step("3.10 store_memory", "RUN", f"persistiendo {len(memories)} memorias utiles")
                try:
                    await self._memory_store.save_memories(contact_id, memories)
                    step("3.10 store_memory", "OK")
                except Exception as exc:
                    mark_error("3.10 store_memory", exc)
                    raise
            else:
                substep("3.10 store_memory", "OK", "sin hechos duraderos para guardar")
        else:
            substep("3.10 store_memory", "WARN", "faltan campos para persistir")
        return {}

    def _apply_state_update(self, state: GraphState, patch: dict[str, Any]) -> GraphState:
        merged: GraphState = deepcopy(state)
        for key, value in patch.items():
            if key == "discovery_call_slots" and isinstance(value, dict):
                existing = merged.get(key, {})
                merged[key] = _merge_slots(existing if isinstance(existing, dict) else {}, value)
            else:
                merged[key] = value
        return merged

    def _cleanup_state(self, state: GraphState) -> GraphState:
        cleaned: GraphState = deepcopy(state)
        if cleaned.get("next_node") != "discovery_call":
            cleaned["pending_action"] = ""
            cleaned["pending_question"] = ""
            cleaned["discovery_call_slots"] = {}
            if cleaned.get("stage") in {
                "collecting_slots",
                "awaiting_calendar_choice",
                "awaiting_booking_confirmation",
                "ready_for_handoff",
                "booking_confirmed",
            }:
                cleaned["stage"] = "open"
            if cleaned.get("active_goal") == "discovery_call" and not cleaned.get("handoff_required", False):
                cleaned["active_goal"] = "conversation"
        if cleaned.get("next_node") != "rag":
            cleaned["last_tool_result"] = ""
        return cleaned

    def _needs_summary_refresh(self, state: GraphState) -> bool:
        summary = state.get("conversation_summary", "")
        turn_count = int(state.get("turn_count", 0))
        if len(summary) >= self._settings.summary_refresh_char_threshold:
            return True
        if turn_count and turn_count % self._settings.summary_refresh_turn_threshold == 0:
            return True
        if state.get("next_node") == "discovery_call" and state.get("stage") in {
            "ready_for_handoff",
            "booking_confirmed",
        }:
            return True
        return False


def _merge_slots(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key in (
        "lead_name",
        "project_need",
        "preferred_date",
        "preferred_time",
        "booking_email",
        "calendly_booking_status",
        "calendly_booking_url",
        "calendly_booking_start_time",
        "calendly_booking_event_uri",
        "calendly_booking_invitee_uri",
    ):
        value = incoming.get(key)
        if value:
            merged[key] = value
    if incoming.get("missing_fields") is not None:
        merged["missing_fields"] = list(incoming.get("missing_fields") or [])
    if "confidence" in incoming:
        merged["confidence"] = incoming["confidence"]
    if "should_handoff" in incoming:
        merged["should_handoff"] = incoming["should_handoff"]
    return merged


def _build_pending_question(missing_fields: list[str]) -> str:
    field_names = {
        "lead_name": "tu nombre",
        "project_need": "el tipo de proyecto o necesidad",
        "preferred_date": "la fecha preferida",
        "preferred_time": "la hora preferida",
    }
    readable = [field_names.get(field, field) for field in missing_fields]
    if not readable:
        return ""
    if len(readable) == 1:
        return f"Necesito {readable[0]} para continuar."
    if len(readable) == 2:
        return f"Necesito {readable[0]} y {readable[1]} para continuar."
    return "Necesito " + ", ".join(readable[:-1]) + f" y {readable[-1]} para continuar."


def _shorten(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
