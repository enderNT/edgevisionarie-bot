from __future__ import annotations

from app.memory_runtime import (
    ConversationMemoryRuntime,
    LLMConversationSummaryService,
    build_long_term_memory_store,
)
from app.services.barbershop_memory import BarbershopMemoryPolicy
from app.services.llm import SupportLLMService
from app.settings import Settings


def build_conversation_memory_runtime(
    settings: Settings,
    llm_service: SupportLLMService,
) -> ConversationMemoryRuntime:
    return ConversationMemoryRuntime(
        store=build_long_term_memory_store(settings),
        summary_service=LLMConversationSummaryService(llm_service),
        policy=BarbershopMemoryPolicy(),
        recall_limit=settings.memory_search_limit,
    )
