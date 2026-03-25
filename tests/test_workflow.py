import asyncio

from app.graph.workflow import SupportWorkflow
from app.models.schemas import ChatwootWebhook, DiscoveryCallIntentPayload, StateRoutingDecision
from app.services.company_config import CompanyConfigLoader
from app.services.router import StateRoutingService
from app.settings import Settings


class FakeLLMService:
    def __init__(self):
        self.summary_calls = 0

    async def classify_state_route(self, routing_packet, guard_hint=None):
        del guard_hint
        message = routing_packet.user_message.lower()
        if "llamada" in message or routing_packet.active_goal == "discovery_call":
            return StateRoutingDecision(
                next_node="discovery_call",
                intent="discovery_call",
                confidence=0.9,
                needs_retrieval=False,
                state_update={"active_goal": "discovery_call", "stage": "collecting_slots"},
                reason="test",
            )
        if "horario" in message or "precio" in message or "servicios" in message:
            return StateRoutingDecision(
                next_node="rag",
                intent="rag",
                confidence=0.85,
                needs_retrieval=True,
                state_update={"active_goal": "information", "stage": "lookup"},
                reason="test",
            )
        return StateRoutingDecision(
            next_node="conversation",
            intent="conversation",
            confidence=0.8,
            needs_retrieval=False,
            state_update={"active_goal": "conversation", "stage": "open"},
            reason="test",
        )

    async def build_conversation_reply(self, user_message, memories):
        del memories
        return f"Respuesta para: {user_message}"

    async def build_rag_reply(self, user_message, memories, company_context):
        del memories, company_context
        return f"RAG para: {user_message}"

    async def extract_discovery_call_intent(
        self, user_message, memories, company_context, contact_name, current_slots=None, pending_question=None
    ):
        del memories, company_context, contact_name, pending_question
        current_slots = current_slots or {}
        payload = DiscoveryCallIntentPayload(
            lead_name=current_slots.get("lead_name", "Juan Perez"),
            project_need=current_slots.get("project_need", "automatizacion"),
            preferred_date="manana" if "manana" in user_message.lower() else current_slots.get("preferred_date"),
            preferred_time="10 am" if "10" in user_message else current_slots.get("preferred_time"),
            missing_fields=[] if ("manana" in user_message.lower() and "10" in user_message) else ["preferred_time"],
            should_handoff=True,
            confidence=0.9,
        )
        return payload, f"Discovery call lista: {user_message}"

    async def build_state_summary(self, current_summary, user_message, assistant_message, active_goal, stage):
        self.summary_calls += 1
        return f"{current_summary} | {active_goal}:{stage} | {user_message} -> {assistant_message}".strip(" |")


class FakeMemoryStore:
    def __init__(self):
        self.saved = []

    async def search(self, contact_id, query, limit=5):
        del contact_id, query, limit
        return ["Recuerdo util", "Prefiere horario vespertino"]

    async def save_memories(self, contact_id, memories):
        self.saved.append((contact_id, [memory.model_dump() for memory in memories]))


class FakeQdrantService:
    def __init__(self):
        self.calls = 0

    async def build_context(self, *args, **kwargs):
        self.calls += 1
        del args, kwargs
        return "Contexto RAG simulado"


def build_webhook(message: str, conversation_id: int = 123) -> ChatwootWebhook:
    return ChatwootWebhook(
        content=message,
        conversation={"id": conversation_id},
        contact={"id": 456, "name": "Juan Perez"},
        event="message_created",
        message_type="incoming",
    )


def build_workflow():
    llm = FakeLLMService()
    router = StateRoutingService(Settings(llm_api_key=None, openai_api_key=None), llm)
    memory = FakeMemoryStore()
    qdrant = FakeQdrantService()
    workflow = SupportWorkflow(
        router,
        llm,
        memory,
        CompanyConfigLoader(config_path="config/company.json"),  # type: ignore[arg-type]
        qdrant,
        Settings(),
    )
    return workflow, memory, qdrant, llm


def test_workflow_routes_to_conversation():
    workflow, memory, qdrant, llm = build_workflow()

    result = asyncio.run(workflow.run(build_webhook("Necesito informacion general sobre Metaedgevisionaries")))

    assert result["next_node"] == "conversation"
    assert result["response_text"] == "Respuesta para: Necesito informacion general sobre Metaedgevisionaries"
    assert result["handoff_required"] is False
    assert qdrant.calls == 0
    assert memory.saved
    assert llm.summary_calls >= 1


def test_workflow_routes_to_rag():
    workflow, memory, qdrant, _ = build_workflow()

    result = asyncio.run(workflow.run(build_webhook("Cuales son sus horarios?")))

    assert result["next_node"] == "rag"
    assert result["response_text"] == "RAG para: Cuales son sus horarios?"
    assert result["handoff_required"] is False
    assert qdrant.calls == 1
    assert memory.saved


def test_workflow_keeps_discovery_call_state_across_turns():
    workflow, memory, qdrant, llm = build_workflow()
    conversation_id = 777

    first = asyncio.run(workflow.run(build_webhook("Quiero agendar una llamada", conversation_id=conversation_id)))
    second = asyncio.run(workflow.run(build_webhook("manana a las 10", conversation_id=conversation_id)))

    assert first["next_node"] == "discovery_call"
    assert first["stage"] == "collecting_slots"
    assert second["next_node"] == "discovery_call"
    assert second["stage"] == "ready_for_handoff"
    assert second["discovery_call_slots"]["preferred_time"] == "10 am"
    assert qdrant.calls == 0
    assert memory.saved
    assert llm.summary_calls >= 1
