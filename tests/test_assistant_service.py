import asyncio

from app.models.schemas import ChatwootWebhook
from app.services.assistant_service import AssistantService


class FakeWorkflow:
    llm_backend_name = "raw"
    llm_model_name = "gpt-test"

    async def run(self, payload: ChatwootWebhook) -> dict:
        return {
            "conversation_id": payload.conversation_id,
            "contact_id": payload.contact_id,
            "contact_name": payload.contact_name,
            "last_user_message": payload.latest_message,
            "response_text": "Hola desde el bot",
            "next_node": "conversation",
            "intent": "conversation",
            "needs_retrieval": False,
            "handoff_required": False,
        }


class FakeChatwootClient:
    def __init__(self) -> None:
        self.sent = []

    async def send_message(self, conversation_id: str, content: str, account_id: str | None = None) -> None:
        self.sent.append((conversation_id, content, account_id))


class FakeTraceStore:
    def __init__(self) -> None:
        self.records = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def enqueue(self, record) -> bool:
        self.records.append(record)
        return True


def build_webhook() -> ChatwootWebhook:
    return ChatwootWebhook(
        content="Hola",
        conversation={"id": 123},
        contact={"id": 456, "name": "Ana"},
        account={"id": 999},
        event="message_created",
        message_type="incoming",
    )


def test_assistant_service_enqueues_trace_record() -> None:
    trace_store = FakeTraceStore()
    chatwoot = FakeChatwootClient()
    service = AssistantService(FakeWorkflow(), chatwoot, trace_store=trace_store)

    result = asyncio.run(service.process_webhook(build_webhook(), flow_id="flow-123"))

    assert result["response_text"] == "Hola desde el bot"
    assert chatwoot.sent == [("123", "Hola desde el bot", "999")]
    assert len(trace_store.records) == 1
    assert trace_store.records[0].flow_id == "flow-123"
    assert trace_store.records[0].response_sent is True
