import asyncio

from app.services.memory import (
    Mem0LocalMemoryStore,
    Mem0PlatformMemoryStore,
    _normalize_mem0_search_results,
    should_store_memory,
)


def test_normalize_mem0_search_results_accepts_v2_dict_shape():
    results = {
        "results": [
            {"memory": "Cliente prefiere horario matutino"},
            {"memory": "Usa WhatsApp para seguimiento"},
            {"text": "Dato alternativo"},
        ]
    }

    normalized = _normalize_mem0_search_results(results, limit=2)

    assert normalized == [
        "Cliente prefiere horario matutino",
        "Usa WhatsApp para seguimiento",
    ]


def test_normalize_mem0_search_results_accepts_list_shape():
    results = [
        {"memory": "Primera memoria"},
        {"memory": "Segunda memoria"},
    ]

    normalized = _normalize_mem0_search_results(results, limit=5)

    assert normalized == ["Primera memoria", "Segunda memoria"]


def test_mem0_local_search_normalizes_dict_results():
    class FakeLocalClient:
        def search(self, query, filters, limit):
            assert query == "automatizacion con ia"
            assert filters == {"user_id": "456"}
            assert limit == 3
            return {"results": [{"memory": "Proyecto previo A"}, {"memory": "Proyecto previo B"}]}

    store = object.__new__(Mem0LocalMemoryStore)
    store._client = FakeLocalClient()

    memories = asyncio.run(store.search("456", "automatizacion con ia", limit=3))

    assert memories == ["Proyecto previo A", "Proyecto previo B"]


def test_mem0_platform_search_normalizes_dict_results():
    class FakePlatformClient:
        def search(self, query, filters, top_k):
            assert query == "agendar discovery call"
            assert filters == {"user_id": "789"}
            assert top_k == 2
            return {"results": [{"memory": "Prefiere WhatsApp"}, {"memory": "No disponible por la tarde"}]}

    store = object.__new__(Mem0PlatformMemoryStore)
    store._client = FakePlatformClient()

    memories = asyncio.run(store.search("789", "agendar discovery call", limit=2))

    assert memories == ["Prefiere WhatsApp", "No disponible por la tarde"]


def test_should_store_memory_skips_trivial_turns():
    memories = should_store_memory("hola", "Hola, te ayudo con gusto", "conversation", {})

    assert memories == []


def test_should_store_memory_persists_discovery_call_facts():
    memories = should_store_memory(
        "Quiero una llamada para automatizacion manana",
        "Perfecto, lo paso al equipo comercial y tecnico",
        "discovery_call",
        {
            "discovery_call_slots": {
                "lead_name": "Juan Perez",
                "project_need": "automatizacion",
                "preferred_date": "manana",
                "preferred_time": "10 am",
            }
        },
    )

    assert memories
    assert memories[0].kind == "profile"
