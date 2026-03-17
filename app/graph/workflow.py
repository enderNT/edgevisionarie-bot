from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.models.schemas import AppointmentIntentPayload, ChatwootWebhook
from app.services.clinic_config import ClinicConfigLoader
from app.services.llm import ClinicLLMService
from app.services.memory import MemoryStore


class GraphState(TypedDict, total=False):
    webhook: ChatwootWebhook
    conversation_id: str
    contact_id: str
    contact_name: str
    user_message: str
    clinic_context: str
    memories: list[str]
    intent: str
    routing_reason: str
    routing_confidence: float
    response_text: str
    appointment_payload: dict[str, Any]
    handoff_required: bool


class ClinicWorkflow:
    def __init__(
        self,
        llm_service: ClinicLLMService,
        memory_store: MemoryStore,
        clinic_config_loader: ClinicConfigLoader,
    ) -> None:
        self._llm_service = llm_service
        self._memory_store = memory_store
        self._clinic_config_loader = clinic_config_loader
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("route", self._route)
        graph.add_node("conversation", self._conversation)
        graph.add_node("appointment", self._appointment)
        graph.add_node("store_memory", self._store_memory)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "route")
        graph.add_conditional_edges(
            "route",
            self._branch_after_route,
            {
                "conversation": "conversation",
                "appointment_intent": "appointment",
            },
        )
        graph.add_edge("conversation", "store_memory")
        graph.add_edge("appointment", "store_memory")
        graph.add_edge("store_memory", END)
        return graph.compile()

    async def run(self, webhook: ChatwootWebhook) -> GraphState:
        initial_state: GraphState = {"webhook": webhook}
        return await self._graph.ainvoke(initial_state)

    async def _load_context(self, state: GraphState) -> GraphState:
        webhook = state["webhook"]
        clinic_context = self._clinic_config_loader.load().to_context_text()
        memories = await self._memory_store.search(
            webhook.contact_id,
            query=webhook.latest_message or "contexto del usuario",
            limit=5,
        )
        return {
            "conversation_id": webhook.conversation_id,
            "contact_id": webhook.contact_id,
            "contact_name": webhook.contact_name,
            "user_message": webhook.latest_message,
            "clinic_context": clinic_context,
            "memories": memories,
        }

    async def _route(self, state: GraphState) -> GraphState:
        decision = await self._llm_service.route_intent(
            user_message=state["user_message"],
            memories=state.get("memories", []),
            clinic_context=state["clinic_context"],
        )
        return {
            "intent": decision.intent,
            "routing_reason": decision.reason,
            "routing_confidence": decision.confidence,
        }

    def _branch_after_route(self, state: GraphState) -> str:
        return state.get("intent", "conversation")

    async def _conversation(self, state: GraphState) -> GraphState:
        response_text = await self._llm_service.build_conversation_reply(
            user_message=state["user_message"],
            memories=state.get("memories", []),
            clinic_context=state["clinic_context"],
        )
        return {
            "response_text": response_text,
            "handoff_required": False,
            "appointment_payload": {},
        }

    async def _appointment(self, state: GraphState) -> GraphState:
        appointment, response_text = await self._llm_service.extract_appointment_intent(
            user_message=state["user_message"],
            memories=state.get("memories", []),
            clinic_context=state["clinic_context"],
            contact_name=state["contact_name"],
        )
        return {
            "response_text": response_text,
            "handoff_required": True,
            "appointment_payload": appointment.model_dump(),
        }

    async def _store_memory(self, state: GraphState) -> GraphState:
        response_text = state.get("response_text")
        user_message = state.get("user_message")
        contact_id = state.get("contact_id")
        if response_text and user_message and contact_id:
            await self._memory_store.save_exchange(contact_id, user_message, response_text)
        return {}
