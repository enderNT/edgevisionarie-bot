import asyncio

from app.memory_runtime.runtime import ConversationMemoryRuntime
from app.memory_runtime.store import InMemoryLongTermMemoryStore
from app.memory_runtime.summary import LLMConversationSummaryService
from app.memory_runtime.types import LongTermMemoryRecord, ShortTermState, TurnMemoryInput
from app.services.byteworkers_memory import ByteWorkersMemoryPolicy


class FakeLLMService:
    async def build_state_summary(self, current_summary, user_message, assistant_message, active_goal, stage):
        return f"{current_summary}|{active_goal}:{stage}|{user_message}->{assistant_message}".strip("|")


def test_in_memory_long_term_store_returns_recent_records():
    store = InMemoryLongTermMemoryStore()

    asyncio.run(
        store.save(
            "456",
            [
                LongTermMemoryRecord(kind="profile", text="Cliente prefiere horario matutino"),
                LongTermMemoryRecord(kind="episode", text="Usa WhatsApp para seguimiento"),
            ],
        )
    )

    records = asyncio.run(store.search("456", "automatizacion con ia", limit=5))

    assert [record.text for record in records] == [
        "Cliente prefiere horario matutino",
        "Usa WhatsApp para seguimiento",
    ]


def test_byteworkers_policy_skips_trivial_turns():
    policy = ByteWorkersMemoryPolicy()

    records = policy.select_records(
        TurnMemoryInput(user_message="hola", assistant_message="Hola, te ayudo con gusto", route="conversation"),
        ShortTermState(),
        {},
    )

    assert records == []


def test_byteworkers_policy_persists_discovery_call_facts():
    policy = ByteWorkersMemoryPolicy()

    records = policy.select_records(
        TurnMemoryInput(
            user_message="Quiero una llamada para automatizacion manana",
            assistant_message="Perfecto, lo paso al equipo comercial y tecnico",
            route="discovery_call",
        ),
        ShortTermState(),
        {
            "discovery_call_slots": {
                "lead_name": "Juan Perez",
                "project_need": "automatizacion",
                "preferred_date": "manana",
                "preferred_time": "10 am",
            }
        },
    )

    assert records
    assert records[0].kind == "profile"


def test_conversation_memory_runtime_loads_and_commits():
    store = InMemoryLongTermMemoryStore()
    runtime = ConversationMemoryRuntime(
        store=store,
        summary_service=LLMConversationSummaryService(FakeLLMService()),
        policy=ByteWorkersMemoryPolicy(),
        recall_limit=3,
    )

    asyncio.run(store.save("789", [LongTermMemoryRecord(kind="profile", text="Prefiere WhatsApp")]))

    context = asyncio.run(
        runtime.load_context(
            "session-1",
            "789",
            "agendar discovery call",
            ShortTermState(summary="resumen previo", turn_count=2),
        )
    )
    commit = asyncio.run(
        runtime.commit_turn(
            "session-1",
            "789",
            TurnMemoryInput(
                user_message="Quiero una llamada para automatizacion",
                assistant_message="Te comparto el siguiente paso",
                route="discovery_call",
            ),
            ShortTermState(summary="resumen previo", turn_count=context.turn_count, active_goal="discovery_call", stage="collecting_slots"),
            {
                "discovery_call_slots": {
                    "lead_name": "Juan Perez",
                    "project_need": "automatizacion",
                }
            },
        )
    )

    assert context.turn_count == 3
    assert context.recalled_memories == ["Prefiere WhatsApp"]
    assert commit.turn_count == 3
    assert commit.summary
    assert commit.saved_records
