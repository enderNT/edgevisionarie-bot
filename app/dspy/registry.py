from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.dspy.adapters import (
    ConversationModuleInputs,
    DiscoveryCallModuleInputs,
    DiscoveryCallModuleOutputs,
    RagModuleInputs,
    RouteModuleInputs,
    RouteModuleOutputs,
    SummaryModuleInputs,
)
from app.dspy.programs import build_programs
from app.dspy.signatures import require_dspy
from app.models.schemas import DiscoveryCallIntentPayload, RoutingPacket, StateRoutingDecision
from app.settings import Settings

logger = logging.getLogger(__name__)


class DSPyProgramRegistry:
    def __init__(
        self,
        *,
        enabled: bool,
        reason: str = "",
        programs: dict[str, Any] | None = None,
        program_dir: Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.reason = reason
        self._programs = programs or {}
        self._program_dir = program_dir

    def can_serve(self, task: str) -> bool:
        return self.enabled and task in self._programs

    async def conversation_reply(self, user_message: str, memories: list[str]) -> str:
        prediction = await self._run_program(
            "conversation",
            ConversationModuleInputs.from_values(user_message, memories).model_kwargs(),
        )
        return getattr(prediction, "reply_text", "").strip()

    async def rag_reply(self, question: str, memories: list[str], context: str) -> str:
        prediction = await self._run_program(
            "rag",
            RagModuleInputs.from_values(question, memories, context).model_kwargs(),
        )
        return getattr(prediction, "reply_text", "").strip()

    async def build_summary(
        self,
        current_summary: str,
        user_message: str,
        assistant_message: str,
        active_goal: str,
        stage: str,
    ) -> str:
        prediction = await self._run_program(
            "summary",
            SummaryModuleInputs(
                current_summary=current_summary,
                active_goal=active_goal,
                stage=stage,
                user_message=user_message,
                assistant_message=assistant_message,
            ).model_kwargs(),
        )
        return getattr(prediction, "updated_summary", "").strip()

    async def classify_route(self, routing_packet: RoutingPacket) -> StateRoutingDecision:
        prediction = await self._run_program(
            "route",
            RouteModuleInputs.from_routing_packet(routing_packet).model_kwargs(),
        )
        return RouteModuleOutputs.from_prediction(prediction).to_state_routing_decision()

    async def extract_discovery_call(
        self,
        *,
        user_message: str,
        memories: list[str],
        company_context: str,
        contact_name: str,
        current_slots: dict[str, Any] | None,
        pending_question: str | None,
    ) -> DiscoveryCallIntentPayload:
        prediction = await self._run_program(
            "discovery_call",
            DiscoveryCallModuleInputs.from_values(
                user_message,
                memories,
                company_context,
                contact_name,
                current_slots,
                pending_question,
            ).model_kwargs(),
        )
        return DiscoveryCallModuleOutputs.from_prediction(prediction).to_payload()

    async def _run_program(self, task: str, kwargs: dict[str, Any]) -> Any:
        if not self.can_serve(task):
            raise RuntimeError(f"DSPy no esta habilitado para {task}: {self.reason or 'task-not-registered'}")
        program = self._programs[task]
        return await asyncio.to_thread(program, **kwargs)


def build_dspy_registry(settings: Settings) -> DSPyProgramRegistry:
    if not settings.dspy_enabled:
        return DSPyProgramRegistry(enabled=False, reason="disabled-by-settings")

    try:
        dspy = require_dspy()
        lm_kwargs: dict[str, Any] = {}
        if settings.resolved_llm_api_key:
            lm_kwargs["api_key"] = settings.resolved_llm_api_key
        if settings.resolved_llm_base_url:
            lm_kwargs["api_base"] = settings.resolved_llm_base_url
            lm_kwargs["model_type"] = "chat"
        lm = dspy.LM(settings.resolved_dspy_model, **lm_kwargs)
        dspy.configure(lm=lm)
        programs = build_programs()
        _try_load_programs(programs, settings.dspy_program_dir)
        return DSPyProgramRegistry(
            enabled=True,
            reason="enabled",
            programs=programs,
            program_dir=settings.dspy_program_dir,
        )
    except Exception as exc:  # pragma: no cover - depende de entorno/dspy
        logger.warning("DSPy registry unavailable, using raw backend: %s", exc)
        return DSPyProgramRegistry(enabled=False, reason=f"init-failed:{type(exc).__name__}")


def _try_load_programs(programs: dict[str, Any], program_dir: Path) -> None:
    for task, program in programs.items():
        path = program_dir / f"{task}.json"
        if not path.exists():
            continue
        try:
            program.load(str(path))
        except Exception as exc:  # pragma: no cover - depende de dspy/artefacto
            logger.warning("No se pudo cargar artefacto DSPy para %s desde %s: %s", task, path, exc)
