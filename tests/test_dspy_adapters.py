import json

from app.dspy.adapters import (
    DiscoveryCallModuleInputs,
    DiscoveryCallModuleOutputs,
    RouteModuleInputs,
    RouteModuleOutputs,
)
from app.models.schemas import RoutingPacket


def test_route_module_inputs_serialize_packet_for_dspy():
    packet = RoutingPacket(
        user_message="Quiero agendar una llamada",
        conversation_summary="Lead interesado en automatizacion.",
        active_goal="conversation",
        stage="open",
        pending_action="",
        pending_question="",
        discovery_call_slots={"lead_name": "Ana"},
        last_tool_result="",
        last_user_message="Hola",
        last_assistant_message="Te ayudo con gusto",
        memories=["Prefiere WhatsApp"],
    )

    module_inputs = RouteModuleInputs.from_routing_packet(packet)
    kwargs = module_inputs.model_kwargs()

    assert kwargs["user_message"] == "Quiero agendar una llamada"
    assert kwargs["memories_text"].startswith("- Prefiere WhatsApp")
    assert json.loads(kwargs["discovery_call_slots_json"]) == {"lead_name": "Ana"}


def test_route_module_outputs_build_state_routing_decision():
    prediction = type(
        "Prediction",
        (),
        {
            "next_node": "discovery_call",
            "intent": "discovery_call",
            "confidence": 0.91,
            "needs_retrieval": False,
            "next_active_goal": "discovery_call",
            "next_stage": "collecting_slots",
            "next_pending_action": "collecting_slots",
            "next_pending_question": "Comparte tu horario preferido",
            "clear_slots": False,
            "clear_last_tool_result": True,
            "route_reason": "compiled-program",
        },
    )()

    decision = RouteModuleOutputs.from_prediction(prediction).to_state_routing_decision()

    assert decision.next_node == "discovery_call"
    assert decision.state_update["active_goal"] == "discovery_call"
    assert decision.state_update["last_tool_result"] == ""


def test_discovery_call_module_inputs_serialize_complex_values():
    module_inputs = DiscoveryCallModuleInputs.from_values(
        user_message="Necesito una demo para mi ecommerce",
        memories=["Prefiere agenda por la tarde"],
        company_context="Servicios y politicas",
        contact_name="Mario",
        current_slots={"lead_name": "Mario"},
        pending_question="Necesito el tipo de proyecto",
    )

    kwargs = module_inputs.model_kwargs()

    assert kwargs["contact_name"] == "Mario"
    assert json.loads(kwargs["current_slots_json"]) == {"lead_name": "Mario"}
    assert "Prefiere agenda" in kwargs["memories_text"]


def test_discovery_call_module_outputs_parse_string_fields():
    prediction = type(
        "Prediction",
        (),
        {
            "lead_name": "Mario",
            "project_need": "demo ecommerce",
            "preferred_date": "manana",
            "preferred_time": "10 am",
            "missing_fields": '["preferred_time"]',
            "should_handoff": "true",
            "confidence": "0.83",
        },
    )()

    payload = DiscoveryCallModuleOutputs.from_prediction(prediction).to_payload()

    assert payload.lead_name == "Mario"
    assert payload.project_need == "demo ecommerce"
    assert payload.missing_fields == ["preferred_time"]
    assert payload.should_handoff is True
