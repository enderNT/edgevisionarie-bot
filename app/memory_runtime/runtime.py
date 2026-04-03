from __future__ import annotations

from typing import Any

from app.memory_runtime.policy import MemoryPolicy
from app.memory_runtime.store import LongTermMemoryStore
from app.memory_runtime.summary import ConversationSummaryService
from app.memory_runtime.types import (
    ActorId,
    MemoryCommitResult,
    MemoryContext,
    SessionId,
    ShortTermState,
    TurnMemoryInput,
)


class ConversationMemoryRuntime:
    def __init__(
        self,
        store: LongTermMemoryStore,
        summary_service: ConversationSummaryService,
        policy: MemoryPolicy,
        *,
        recall_limit: int = 5,
    ) -> None:
        self._store = store
        self._summary_service = summary_service
        self._policy = policy
        self._recall_limit = recall_limit

    async def load_context(
        self,
        session_id: SessionId,
        actor_id: ActorId,
        query: str,
        short_term: ShortTermState,
    ) -> MemoryContext:
        del session_id
        raw_records = await self._store.search(actor_id, query=query, limit=self._recall_limit)
        return MemoryContext(
            recalled_memories=_compact_recalled_memories(raw_records),
            raw_records=raw_records,
            turn_count=int(short_term.turn_count) + 1,
        )

    async def commit_turn(
        self,
        session_id: SessionId,
        actor_id: ActorId,
        turn: TurnMemoryInput,
        short_term: ShortTermState,
        domain_state: dict[str, Any],
    ) -> MemoryCommitResult:
        del session_id
        summary = short_term.summary
        if turn.user_message.strip() or turn.assistant_message.strip():
            summary = await self._summary_service.update(short_term.summary, turn, short_term)
        records = self._policy.select_records(turn, short_term, domain_state)
        if records:
            await self._store.save(actor_id, records)
        return MemoryCommitResult(
            summary=summary,
            saved_records=records,
            turn_count=int(short_term.turn_count),
        )

    async def aclose(self) -> None:
        close = getattr(self._store, "aclose", None)
        if close is not None:
            await close()


def _compact_recalled_memories(records: list[Any]) -> list[str]:
    compact: list[str] = []
    for record in records[:3]:
        text = getattr(record, "text", "")
        if not isinstance(text, str):
            continue
        normalized = " ".join(text.split())
        if normalized:
            compact.append(normalized[:140])
    return compact
