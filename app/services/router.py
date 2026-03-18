from __future__ import annotations

import logging
from typing import Any

from app.models.schemas import IntentDecision
from app.observability.flow_logger import substep
from app.settings import Settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency during local development
    from semantic_router import Route
    from semantic_router.encoders import OpenAIEncoder
    from semantic_router.index import LocalIndex
    from semantic_router.routers import SemanticRouter

    SEMANTIC_ROUTER_AVAILABLE = True
except Exception:  # pragma: no cover - import guard for environments without the package
    Route = Any  # type: ignore[assignment]
    OpenAIEncoder = Any  # type: ignore[assignment]
    LocalIndex = Any  # type: ignore[assignment]
    SemanticRouter = Any  # type: ignore[assignment]
    SEMANTIC_ROUTER_AVAILABLE = False
ROUTE_DEFINITIONS = [
    {
        "name": "conversation",
        "utterances": [
            "hola",
            "buenos dias",
            "gracias",
            "perfecto",
            "ok",
            "me puedes ayudar",
            "solo queria saludar",
            "tengo una duda corta",
            "muchas gracias",
            "entendido",
            "de acuerdo",
            "hola buenos dias",
        ],
        "score_threshold": 0.42,
    },
    {
        "name": "rag",
        "utterances": [
            "cuales son sus horarios",
            "que servicios ofrecen",
            "cuanto cuesta la consulta",
            "que doctores tienen",
            "atienden sabados",
            "donde estan ubicados",
            "que especialidades manejan",
            "politicas de pago",
            "cuanto cuesta",
            "que precio tiene la consulta",
            "que doctores atienden",
            "a que hora abren",
        ],
        "score_threshold": 0.46,
    },
    {
        "name": "appointment_intent",
        "utterances": [
            "quiero agendar una cita",
            "necesito una consulta",
            "quiero reservar con un doctor",
            "me ayudas a sacar una cita",
            "busco cita para dermatologia",
            "quiero programar una visita",
            "tengo que agendar una cita para manana",
            "quiero una cita el viernes",
            "me gustaria una cita",
            "quiero reservar una consulta",
            "puedo agendar para hoy",
            "necesito cita con pediatria",
        ],
        "score_threshold": 0.48,
    },
]


class ClinicIntentRouterService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._router = self._build_router()

    def _build_router(self):
        if not SEMANTIC_ROUTER_AVAILABLE or not self._settings.openai_api_key:
            logger.warning(
                "semantic-router unavailable or OPENAI_API_KEY missing; using deterministic fallback router"
            )
            return None

        encoder_kwargs: dict[str, Any] = {
            "name": self._settings.openai_embedding_model,
            "openai_api_key": self._settings.openai_api_key,
        }
        if self._settings.openai_base_url:
            encoder_kwargs["openai_base_url"] = self._settings.openai_base_url.rstrip("/")
        encoder = OpenAIEncoder(**encoder_kwargs)
        routes = [Route(name=item["name"], utterances=item["utterances"], score_threshold=item["score_threshold"]) for item in ROUTE_DEFINITIONS]
        return SemanticRouter(
            encoder=encoder,
            routes=routes,
            index=LocalIndex(),
            auto_sync="local",
        )

    async def route_intent(self, user_message: str, memories: list[str], clinic_context: str) -> IntentDecision:
        if self._router is None:
            return self._fallback_route(user_message)

        try:
            router_input = self._build_router_input(user_message, memories, clinic_context)
            substep(
                "router_prompt_compose",
                "OK",
                f"msg_chars={len(user_message)} memories={len(memories)} router_chars={len(router_input)}",
            )
            route_choice = await self._router.acall(router_input)
            route_name = getattr(route_choice, "name", None)
            score = self._extract_score(route_choice)
            if not route_name:
                substep("router_match", "WARN", "no semantic match, using fallback")
                return self._fallback_route(user_message)

            confidence = self._normalize_score(score)
            score_text = f"{score:.3f}" if score is not None else "n/a"
            substep("router_match", "OK", f"route={route_name} score={score_text}")
            return IntentDecision(
                intent=route_name,
                confidence=confidence,
                reason=f"semantic-router:{route_name}:{score:.3f}" if score is not None else f"semantic-router:{route_name}",
            )
        except Exception as exc:
            logger.warning("semantic-router failed, using fallback: %s", exc)
            substep("router_fallback", "WARN", "semantic-router error")
            return self._fallback_route(user_message)

    def _build_router_input(self, user_message: str, memories: list[str], clinic_context: str) -> str:
        sections = [f"Mensaje actual:\n{_compact_text(user_message, max_len=400)}"]

        memory_lines = [_compact_text(memory, max_len=120) for memory in memories[:2] if memory.strip()]
        if memory_lines:
            sections.append("Memoria relevante del usuario:\n- " + "\n- ".join(memory_lines))

        clinic_hints = self._build_clinic_hints(user_message, clinic_context)
        if clinic_hints:
            sections.append(f"Referencia clinica:\n{clinic_hints}")

        return "\n\n".join(sections)

    def _build_clinic_hints(self, user_message: str, clinic_context: str) -> str:
        lowered = user_message.lower()
        if not any(
            word in lowered
            for word in ("horario", "horarios", "precio", "costo", "doctor", "doctora", "especialidad", "servicio", "ubicacion", "direccion")
        ):
            return ""

        services = self._extract_section_items(clinic_context, "Servicios:", "Doctores:")
        doctors = self._extract_section_items(clinic_context, "Doctores:", "Horarios:")
        hours = self._extract_section_items(clinic_context, "Horarios:", "Politicas:")

        hints: list[str] = []
        if services:
            hints.append("Servicios: " + "; ".join(services[:3]))
        if doctors:
            hints.append("Doctores: " + "; ".join(doctors[:2]))
        if hours:
            hints.append("Horarios: " + "; ".join(hours[:2]))
        return "\n".join(hints)

    def _extract_section_items(self, clinic_context: str, section_start: str, section_end: str) -> list[str]:
        if section_start not in clinic_context:
            return []

        section = clinic_context.split(section_start, 1)[1]
        if section_end in section:
            section = section.split(section_end, 1)[0]

        items = []
        for line in section.splitlines():
            line = line.strip()
            if line.startswith("- "):
                items.append(_compact_text(line[2:], max_len=90))
        return items

    def _extract_score(self, route_choice: Any) -> float | None:
        score = getattr(route_choice, "similarity_score", None)
        if score is None:
            score = getattr(route_choice, "score", None)
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    def _normalize_score(self, score: float | None) -> float:
        if score is None:
            return 0.0
        return max(0.0, min(1.0, score))

    def _fallback_route(self, user_message: str) -> IntentDecision:
        lowered = user_message.lower()
        appointment_keywords = ("cita", "agendar", "agendo", "reservar", "consulta", "doctor", "doctora")
        rag_keywords = ("precio", "costo", "horario", "horarios", "doctor", "especialidad", "servicio", "direccion")
        if any(word in lowered for word in appointment_keywords):
            return IntentDecision(intent="appointment_intent", confidence=0.72, reason="heuristic-fallback")
        if any(word in lowered for word in rag_keywords):
            return IntentDecision(intent="rag", confidence=0.60, reason="heuristic-fallback")
        return IntentDecision(intent="conversation", confidence=0.55, reason="heuristic-fallback")


def _compact_text(value: str, max_len: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[: max_len - 3]}..."
