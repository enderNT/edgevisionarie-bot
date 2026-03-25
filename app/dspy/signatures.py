from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depende de dependencia opcional
    import dspy
except Exception as exc:  # pragma: no cover - depende de entorno
    dspy = None
    _DSPY_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - exercised when dspy is installed
    _DSPY_IMPORT_ERROR = None


def require_dspy() -> Any:
    if dspy is None:  # pragma: no cover - depende de dependencia opcional
        raise RuntimeError(f"DSPy no esta disponible: {_DSPY_IMPORT_ERROR}")
    return dspy


if dspy is not None:  # pragma: no branch
    class ConversationReplySignature(dspy.Signature):
        user_message: str = dspy.InputField()
        memories_text: str = dspy.InputField()
        reply_text: str = dspy.OutputField(desc="Respuesta breve, cordial y util en espanol.")


    class RagReplySignature(dspy.Signature):
        question: str = dspy.InputField()
        memories_text: str = dspy.InputField()
        context: str = dspy.InputField(desc="Contexto factual recuperado y politicas observables.")
        reply_text: str = dspy.OutputField(desc="Respuesta breve, precisa y grounded en el contexto.")


    class SummarySignature(dspy.Signature):
        current_summary: str = dspy.InputField()
        active_goal: str = dspy.InputField()
        stage: str = dspy.InputField()
        user_message: str = dspy.InputField()
        assistant_message: str = dspy.InputField()
        updated_summary: str = dspy.OutputField(desc="Resumen corto actualizado de una o dos frases.")


    class RouteDecisionSignature(dspy.Signature):
        user_message: str = dspy.InputField()
        conversation_summary: str = dspy.InputField()
        active_goal: str = dspy.InputField()
        stage: str = dspy.InputField()
        pending_action: str = dspy.InputField()
        pending_question: str = dspy.InputField()
        discovery_call_slots_json: str = dspy.InputField()
        last_tool_result: str = dspy.InputField()
        last_user_message: str = dspy.InputField()
        last_assistant_message: str = dspy.InputField()
        memories_text: str = dspy.InputField()
        next_node: str = dspy.OutputField(desc="Uno de: conversation, rag, discovery_call.")
        intent: str = dspy.OutputField()
        confidence: float = dspy.OutputField(desc="Confianza entre 0 y 1.")
        needs_retrieval: bool = dspy.OutputField()
        next_active_goal: str = dspy.OutputField()
        next_stage: str = dspy.OutputField()
        next_pending_action: str = dspy.OutputField()
        next_pending_question: str = dspy.OutputField()
        clear_slots: bool = dspy.OutputField()
        clear_last_tool_result: bool = dspy.OutputField()
        route_reason: str = dspy.OutputField()


    class DiscoveryCallSignature(dspy.Signature):
        user_message: str = dspy.InputField()
        memories_text: str = dspy.InputField()
        company_context: str = dspy.InputField()
        contact_name: str = dspy.InputField()
        current_slots_json: str = dspy.InputField()
        pending_question: str = dspy.InputField()
        lead_name: str = dspy.OutputField()
        project_need: str = dspy.OutputField()
        preferred_date: str = dspy.OutputField()
        preferred_time: str = dspy.OutputField()
        missing_fields: str = dspy.OutputField(desc="JSON array con campos faltantes.")
        should_handoff: bool = dspy.OutputField()
        confidence: float = dspy.OutputField(desc="Confianza entre 0 y 1.")
else:
    class ConversationReplySignature:  # pragma: no cover - fallback placeholder
        pass


    class RagReplySignature:  # pragma: no cover - fallback placeholder
        pass


    class SummarySignature:  # pragma: no cover - fallback placeholder
        pass


    class RouteDecisionSignature:  # pragma: no cover - fallback placeholder
        pass


    class DiscoveryCallSignature:  # pragma: no cover - fallback placeholder
        pass
