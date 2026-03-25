from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Protocol, TypedDict

from openai import AsyncOpenAI

from app.dspy.registry import DSPyProgramRegistry, build_dspy_registry
from app.models.schemas import DiscoveryCallIntentPayload, RoutingPacket, StateRoutingDecision
from app.observability.flow_logger import mark_error, step, substep
from app.settings import Settings

logger = logging.getLogger(__name__)


class LLMMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def chat_text(
        self, messages: list[LLMMessage], temperature: float | None = None
    ) -> str: ...

    async def chat_json(
        self, messages: list[LLMMessage], temperature: float | None = None
    ) -> dict[str, Any]: ...


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings) -> None:
        client_kwargs: dict[str, Any] = {"timeout": settings.resolved_llm_timeout_seconds}
        client_kwargs["api_key"] = settings.resolved_llm_api_key or "sk-placeholder"
        if settings.resolved_llm_base_url:
            client_kwargs["base_url"] = settings.resolved_llm_base_url.rstrip("/")
        self._client = AsyncOpenAI(**client_kwargs)
        self._provider_name = settings.resolved_llm_provider
        self._model = settings.resolved_llm_model
        self._temperature = settings.resolved_llm_temperature

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    def _chat_request_kwargs(
        self, messages: list[LLMMessage], temperature: float | None = None
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        selected_temperature = self._temperature if temperature is None else temperature
        if selected_temperature is not None and self._model_supports_temperature():
            request_kwargs["temperature"] = selected_temperature
        return request_kwargs

    def _model_supports_temperature(self) -> bool:
        normalized_model = self._model.strip().lower()
        unsupported_models = {"gpt-5"}
        unsupported_prefixes = ("gpt-5-mini", "gpt-5-nano", "gpt-5-")
        return normalized_model not in unsupported_models and not normalized_model.startswith(
            unsupported_prefixes
        )

    async def chat_text(self, messages: list[LLMMessage], temperature: float | None = None) -> str:
        step(
            "2.2.1 llm_chat_completion",
            "RUN",
            f"provider={self.provider_name} model={self.model_name}",
        )
        try:
            response = await self._client.chat.completions.create(
                **self._chat_request_kwargs(messages=messages, temperature=temperature),
            )
            content = (response.choices[0].message.content or "").strip()
            step("2.2.1 llm_chat_completion", "OK", f"response_chars={len(content)}")
            return content
        except Exception as exc:
            mark_error("2.2.1 llm_chat_completion", exc)
            raise

    async def chat_json(self, messages: list[LLMMessage], temperature: float | None = None) -> dict[str, Any]:
        step(
            "2.2.1 llm_chat_completion",
            "RUN",
            f"provider={self.provider_name} model={self.model_name} json_mode=True",
        )
        request_kwargs = self._chat_request_kwargs(messages=messages, temperature=temperature)
        request_kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.chat.completions.create(**request_kwargs)
            content = (response.choices[0].message.content or "").strip()
            step("2.2.1 llm_chat_completion", "OK", f"response_chars={len(content)}")
            return _extract_json(content)
        except Exception as exc:
            if _should_retry_with_json_schema(exc):
                substep("llm_json_schema_retry", "WARN", "fallback a response_format=json_schema")
                response = await self._client.chat.completions.create(
                    **self._json_schema_request_kwargs(messages=messages, temperature=temperature),
                )
                content = (response.choices[0].message.content or "").strip()
                step("2.2.1 llm_chat_completion", "OK", f"response_chars={len(content)}")
                return _extract_json(content)
            mark_error("2.2.1 llm_chat_completion", exc)
            raise

    def _json_schema_request_kwargs(
        self, messages: list[LLMMessage], temperature: float | None = None
    ) -> dict[str, Any]:
        request_kwargs = self._chat_request_kwargs(messages=messages, temperature=temperature)
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
        }
        return request_kwargs


def build_llm_provider(settings: Settings) -> LLMProvider:
    provider_name = settings.resolved_llm_provider
    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    raise ValueError(f"Unsupported llm provider: {provider_name}")


class RawSupportLLMBackend:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def build_conversation_reply(self, user_message: str, memories: list[str]) -> str:
        system_prompt = (
            "Eres el asistente de atencion al cliente de Metaedgevisionaries, una empresa de creacion y desarrollo de software. "
            "Responde de forma breve, clara y natural. Ayuda con conversacion general, orienta sobre servicios "
            "y si falta un dato puntual ofrece escalar con el equipo comercial o tecnico."
        )
        user_prompt = (
            f"Memorias relevantes: {memories}\n"
            f"Pregunta del usuario: {user_message}\n"
            "Responde en espanol de forma breve y amigable."
        )
        try:
            substep("conversation_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            return await self._provider.chat_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            logger.warning("LLM conversation failed, using deterministic fallback: %s", exc)
            substep("conversation_fallback", "WARN", "mensaje deterministico")
            return (
                "Puedo ayudarte con informacion general de Metaedgevisionaries, servicios de software "
                "y solicitudes para agendar una llamada. Si hace falta un dato puntual, lo canalizo con el equipo humano."
            )

    async def build_rag_reply(self, user_message: str, memories: list[str], company_context: str) -> str:
        system_prompt = (
            "Eres el asistente de Metaedgevisionaries en modo RAG. Usa solo el contexto entregado y no inventes informacion. "
            "Si falta informacion, dilo claramente y escala con el equipo comercial o tecnico."
        )
        user_prompt = (
            f"Contexto recuperado:\n{company_context}\n"
            f"Memoria conversacional: {memories}\n"
            f"Pregunta: {user_message}\n"
            "Responde breve y accionable en espanol."
        )
        try:
            substep("rag_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            return await self._provider.chat_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            logger.warning("LLM rag failed, using deterministic fallback: %s", exc)
            substep("rag_fallback", "WARN", "RAG degradado a respuesta segura")
            return (
                "Puedo responder con la informacion disponible de Metaedgevisionaries. "
                "Si necesitas un dato que no aparece en el contexto actual, lo canalizo con el equipo humano."
            )

    async def build_state_summary(
        self,
        current_summary: str,
        user_message: str,
        assistant_message: str,
        active_goal: str,
        stage: str,
    ) -> str:
        if not user_message.strip() and not assistant_message.strip():
            return current_summary

        system_prompt = (
            "Actualiza un resumen corto de estado conversacional. "
            "Mantente en una o dos frases. No repitas texto inutil."
        )
        user_prompt = (
            f"Resumen actual: {current_summary or 'n/a'}\n"
            f"Objetivo activo: {active_goal or 'n/a'}\n"
            f"Etapa: {stage or 'n/a'}\n"
            f"Ultimo mensaje del usuario: {user_message}\n"
            f"Ultima respuesta del asistente: {assistant_message}\n"
            "Devuelve solo el resumen actualizado."
        )
        try:
            substep("summary_prompt_compose", "OK", f"summary_chars={len(current_summary)}")
            return await self._provider.chat_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("Summary refresh failed, using compact fallback: %s", exc)
            substep("summary_fallback", "WARN", "resumen compacto")
            fragments = [
                current_summary.strip(),
                f"Usuario: {user_message.strip()}",
                f"Asistente: {assistant_message.strip()}",
            ]
            return " ".join(fragment for fragment in fragments if fragment).strip()

    async def classify_state_route(
        self,
        routing_packet: RoutingPacket,
        guard_hint: dict[str, Any] | None = None,
    ) -> StateRoutingDecision:
        system_prompt = (
            "Eres un clasificador de estado para el asistente de Metaedgevisionaries. "
            "Debes devolver JSON estricto con next_node, intent, confidence, needs_retrieval, state_update y reason. "
            "Los valores permitidos para next_node son conversation, rag, discovery_call. "
            "Usa guards y el estado para decidir continuidad conversacional."
        )
        user_prompt = json.dumps(
            {
                "routing_packet": routing_packet.model_dump(),
                "guard_hint": guard_hint or {},
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            substep("state_router_prompt_compose", "OK", f"packet_chars={len(user_prompt)}")
            payload = await self._provider.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            decision = StateRoutingDecision.model_validate(payload)
            substep(
                "state_router_json_parse",
                "OK",
                f"next_node={decision.next_node} confidence={decision.confidence:.2f}",
            )
            return decision
        except Exception as exc:
            logger.warning("State classification failed, using safe fallback: %s", exc)
            substep("state_router_fallback", "WARN", "clasificador degradado")
            return self._fallback_state_route(routing_packet, guard_hint or {})

    async def extract_discovery_call_intent(
        self,
        user_message: str,
        memories: list[str],
        company_context: str,
        contact_name: str,
        current_slots: dict[str, Any] | None = None,
        pending_question: str | None = None,
    ) -> tuple[DiscoveryCallIntentPayload, str]:
        system_prompt = (
            "Extrae intencion para agendar una discovery call. Devuelve JSON estricto con llaves: "
            "lead_name, project_need, preferred_date, preferred_time, missing_fields, should_handoff, confidence."
        )
        user_prompt = (
            f"Nombre de contacto: {contact_name}\n"
            f"Memorias relevantes: {memories}\n"
            f"Slots actuales: {current_slots or {}}\n"
            f"Pendiente: {pending_question or 'n/a'}\n"
            f"Contexto de la empresa:\n{company_context}\n"
            f"Mensaje: {user_message}\n"
            "Interpretalo como una llamada de descubrimiento para entender el proyecto o necesidad del lead. "
            "Si faltan datos, listalos en missing_fields."
        )
        try:
            substep("discovery_call_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            payload = await self._provider.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            discovery_call = DiscoveryCallIntentPayload.model_validate(payload)
            substep("discovery_call_json_parse", "OK")
        except Exception as exc:
            logger.warning("LLM discovery call extraction failed, using heuristic fallback: %s", exc)
            substep("discovery_call_fallback", "WARN", "extraccion heuristica")
            discovery_call = self._fallback_discovery_call(
                user_message,
                contact_name,
                current_slots=current_slots or {},
            )
        reply = self._build_discovery_call_reply(discovery_call)
        return discovery_call, reply

    def _fallback_discovery_call(
        self, user_message: str, contact_name: str, current_slots: dict[str, Any] | None = None
    ) -> DiscoveryCallIntentPayload:
        current_slots = current_slots or {}
        lowered = user_message.lower()
        project_need = None
        for keyword in (
            "automatizacion",
            "automatización",
            "ia",
            "inteligencia artificial",
            "chatbot",
            "web",
            "app",
            "ecommerce",
            "integracion",
            "integración",
            "crm",
            "saas",
            "dashboard",
            "cotizacion",
            "cotización",
        ):
            if keyword in lowered:
                project_need = keyword
                break
        date_match = re.search(
            r"\b(\d{1,2}/\d{1,2}/\d{2,4}|manana|hoy|lunes|martes|miercoles|jueves|viernes|sabado)\b",
            lowered,
        )
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s?(?:am|pm)?|\d{1,2}\s?(?:am|pm))\b", lowered)
        lead_name = (
            current_slots.get("lead_name")
            or (contact_name if contact_name and contact_name != "Cliente" else None)
        )
        project_need = current_slots.get("project_need") or project_need
        preferred_date = current_slots.get("preferred_date") or (date_match.group(1) if date_match else None)
        preferred_time = current_slots.get("preferred_time") or (time_match.group(1) if time_match else None)
        missing_fields = []
        if not lead_name:
            missing_fields.append("lead_name")
        if not project_need:
            missing_fields.append("project_need")
        if not preferred_date:
            missing_fields.append("preferred_date")
        if not preferred_time:
            missing_fields.append("preferred_time")
        return DiscoveryCallIntentPayload(
            lead_name=lead_name,
            project_need=project_need,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            missing_fields=missing_fields,
            should_handoff=True,
            confidence=0.65,
        )

    def _build_discovery_call_reply(self, discovery_call: DiscoveryCallIntentPayload) -> str:
        if discovery_call.missing_fields:
            field_names = {
                "lead_name": "tu nombre",
                "project_need": "el tipo de proyecto o necesidad",
                "preferred_date": "fecha preferida",
                "preferred_time": "hora preferida",
            }
            missing = ", ".join(field_names.get(field, field) for field in discovery_call.missing_fields)
            return (
                "Puedo dejar lista tu solicitud para una discovery call. "
                f"Para continuar necesito: {missing}. "
                "En cuanto los compartas, genero el hand-off para el equipo comercial y tecnico."
            )
        return (
            "Ya tengo lo necesario para preparar tu discovery call. "
            "La pasare al equipo comercial y tecnico con la necesidad del proyecto y tu preferencia de fecha/hora."
        )

    def _fallback_state_route(
        self, routing_packet: RoutingPacket, guard_hint: dict[str, Any]
    ) -> StateRoutingDecision:
        user_message = routing_packet.user_message.lower()
        if guard_hint.get("force_node") == "discovery_call":
            return StateRoutingDecision(
                next_node="discovery_call",
                intent="discovery_call",
                confidence=0.88,
                needs_retrieval=False,
                state_update=guard_hint.get("state_update", {}),
                reason="guard-hint",
            )
        if guard_hint.get("force_node") == "rag":
            return StateRoutingDecision(
                next_node="rag",
                intent="rag",
                confidence=0.84,
                needs_retrieval=True,
                state_update=guard_hint.get("state_update", {}),
                reason="guard-hint",
            )
        if any(
            word in user_message
            for word in ("agendar", "llamada", "reunion", "reunión", "cotizacion", "cotización", "demo", "asesoria")
        ):
            return StateRoutingDecision(
                next_node="discovery_call",
                intent="discovery_call",
                confidence=0.74,
                needs_retrieval=False,
                state_update={"active_goal": "discovery_call", "stage": "collecting_slots"},
                reason="heuristic-fallback",
            )
        if any(
            word in user_message
            for word in (
                "horario",
                "precio",
                "precios",
                "costo",
                "costos",
                "servicio",
                "servicios",
                "stack",
                "tecnologia",
                "tecnologías",
                "tecnologias",
                "ia",
                "integracion",
                "integraciones",
                "mantenimiento",
                "soporte",
                "portafolio",
            )
        ):
            return StateRoutingDecision(
                next_node="rag",
                intent="rag",
                confidence=0.66,
                needs_retrieval=True,
                state_update={"active_goal": "information", "stage": "lookup"},
                reason="heuristic-fallback",
            )
        return StateRoutingDecision(
            next_node="conversation",
            intent="conversation",
            confidence=0.58,
            needs_retrieval=False,
            state_update={
                "active_goal": routing_packet.active_goal or "conversation",
                "stage": routing_packet.stage or "open",
            },
            reason="heuristic-fallback",
        )


class SupportLLMService:
    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings | None = None,
        dspy_registry: DSPyProgramRegistry | None = None,
    ) -> None:
        self._raw_backend = RawSupportLLMBackend(provider)
        self._settings = settings
        if dspy_registry is not None:
            self._dspy_registry = dspy_registry
        elif settings is not None:
            self._dspy_registry = build_dspy_registry(settings)
        else:
            self._dspy_registry = DSPyProgramRegistry(enabled=False, reason="settings-not-provided")

    @property
    def backend_name(self) -> str:
        return "dspy" if self._dspy_registry.enabled else "raw"

    @property
    def model_name(self) -> str:
        provider_name = getattr(self._raw_backend._provider, "model_name", "")
        if self._dspy_registry.enabled:
            return f"{provider_name}|dspy:{self._settings.resolved_dspy_model}" if self._settings else provider_name
        return provider_name

    def _should_fallback_to_raw(self) -> bool:
        return self._settings.dspy_fallback_to_raw if self._settings is not None else True

    async def build_conversation_reply(self, user_message: str, memories: list[str]) -> str:
        if self._dspy_registry.can_serve("conversation"):
            try:
                reply = await self._dspy_registry.conversation_reply(user_message, memories)
                if reply:
                    return reply
            except Exception as exc:
                logger.warning("DSPy conversation failed, using raw backend: %s", exc)
                if not self._should_fallback_to_raw():
                    raise
        return await self._raw_backend.build_conversation_reply(user_message, memories)

    async def build_rag_reply(self, user_message: str, memories: list[str], company_context: str) -> str:
        if self._dspy_registry.can_serve("rag"):
            try:
                reply = await self._dspy_registry.rag_reply(user_message, memories, company_context)
                if reply:
                    return reply
            except Exception as exc:
                logger.warning("DSPy rag failed, using raw backend: %s", exc)
                if not self._should_fallback_to_raw():
                    raise
        return await self._raw_backend.build_rag_reply(user_message, memories, company_context)

    async def build_state_summary(
        self,
        current_summary: str,
        user_message: str,
        assistant_message: str,
        active_goal: str,
        stage: str,
    ) -> str:
        if self._dspy_registry.can_serve("summary"):
            try:
                summary = await self._dspy_registry.build_summary(
                    current_summary=current_summary,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    active_goal=active_goal,
                    stage=stage,
                )
                if summary:
                    return summary
            except Exception as exc:
                logger.warning("DSPy summary failed, using raw backend: %s", exc)
                if not self._should_fallback_to_raw():
                    raise
        return await self._raw_backend.build_state_summary(
            current_summary=current_summary,
            user_message=user_message,
            assistant_message=assistant_message,
            active_goal=active_goal,
            stage=stage,
        )

    async def classify_state_route(
        self,
        routing_packet: RoutingPacket,
        guard_hint: dict[str, Any] | None = None,
    ) -> StateRoutingDecision:
        if self._dspy_registry.can_serve("route"):
            try:
                return await self._dspy_registry.classify_route(routing_packet)
            except Exception as exc:
                logger.warning("DSPy route classification failed, using raw backend: %s", exc)
                if not self._should_fallback_to_raw():
                    raise
        return await self._raw_backend.classify_state_route(routing_packet, guard_hint=guard_hint)

    async def extract_discovery_call_intent(
        self,
        user_message: str,
        memories: list[str],
        company_context: str,
        contact_name: str,
        current_slots: dict[str, Any] | None = None,
        pending_question: str | None = None,
    ) -> tuple[DiscoveryCallIntentPayload, str]:
        if self._dspy_registry.can_serve("discovery_call"):
            try:
                payload = await self._dspy_registry.extract_discovery_call(
                    user_message=user_message,
                    memories=memories,
                    company_context=company_context,
                    contact_name=contact_name,
                    current_slots=current_slots,
                    pending_question=pending_question,
                )
                return payload, self._raw_backend._build_discovery_call_reply(payload)
            except Exception as exc:
                logger.warning("DSPy discovery call extraction failed, using raw backend: %s", exc)
                if not self._should_fallback_to_raw():
                    raise
        return await self._raw_backend.extract_discovery_call_intent(
            user_message=user_message,
            memories=memories,
            company_context=company_context,
            contact_name=contact_name,
            current_slots=current_slots,
            pending_question=pending_question,
        )


def _extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _should_retry_with_json_schema(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format.type" in message and "json_schema" in message
