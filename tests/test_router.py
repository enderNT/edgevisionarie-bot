from app.services.router import ClinicIntentRouterService
from app.settings import Settings


def test_router_falls_back_to_appointment_intent_without_semantic_router():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    decision = service._fallback_route("Quiero agendar una cita con dermatologia")

    assert decision.intent == "appointment_intent"
    assert decision.confidence >= 0.7


def test_router_falls_back_to_rag_without_semantic_router():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    decision = service._fallback_route("Cuales son sus horarios?")

    assert decision.intent == "rag"
    assert decision.confidence >= 0.6


def test_router_falls_back_to_conversation_without_semantic_router():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    decision = service._fallback_route("Hola")

    assert decision.intent == "conversation"
    assert decision.confidence >= 0.5


def test_router_builds_enriched_input_with_memories_and_context():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    router_input = service._build_router_input(
        "Quiero una cita para manana",
        ["Prefiere horario vespertino", "Ya fue paciente de dermatologia"],
        "Clinica: Central\nHorarios: lunes a viernes de 9 a 18\nServicios: dermatologia y medicina general",
    )

    assert "Mensaje actual:" in router_input
    assert "Memoria relevante del usuario:" in router_input
    assert "Referencia clinica:" not in router_input
    assert "Prefiere horario vespertino" in router_input


def test_router_uses_compact_clinic_hints_only_for_institutional_questions():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    clinic_context = (
        "Clinica: Central\n"
        "Servicios:\n- Medicina general: 30 min, 450 MXN\n- Pediatria: 45 min, 600 MXN\n"
        "Doctores:\n- Dra. Elena Ruiz (Medicina general): Lunes a viernes de 09:00 a 14:00\n"
        "Horarios:\n- monday_to_friday: 09:00-18:00\n"
        "Politicas:\n- appointments: Las citas nuevas requieren nombre completo.\n"
    )

    router_input = service._build_router_input(
        "Cuales son sus horarios y que doctores atienden?",
        ["Pregunta seguido por medicina general"],
        clinic_context,
    )

    assert "Referencia clinica:" in router_input
    assert "Servicios:" in router_input
    assert "Doctores:" in router_input
    assert "Horarios:" in router_input
    assert "appointments:" not in router_input
