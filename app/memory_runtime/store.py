from __future__ import annotations

import asyncio
import hashlib
import logging
from contextlib import AsyncExitStack
from typing import Any, Protocol

from langchain_core.embeddings import Embeddings
from openai import AsyncOpenAI, OpenAI

from app.memory_runtime.types import LongTermMemoryRecord
from app.settings import Settings

logger = logging.getLogger(__name__)


class LongTermMemoryStore(Protocol):
    async def search(self, actor_id: str, query: str, limit: int = 5) -> list[LongTermMemoryRecord]:
        ...

    async def save(self, actor_id: str, records: list[LongTermMemoryRecord]) -> None:
        ...


class InMemoryLongTermMemoryStore:
    def __init__(self) -> None:
        self._store: dict[str, list[LongTermMemoryRecord]] = {}

    async def search(self, actor_id: str, query: str, limit: int = 5) -> list[LongTermMemoryRecord]:
        del query
        records = self._store.get(actor_id, [])
        return records[-limit:]

    async def save(self, actor_id: str, records: list[LongTermMemoryRecord]) -> None:
        snippets = {record.text for record in self._store.get(actor_id, [])}
        bucket = self._store.setdefault(actor_id, [])
        for record in records:
            if record.text not in snippets:
                bucket.append(record)
                snippets.add(record.text)


class OpenAIEmbeddingsAdapter(Embeddings):
    def __init__(self, settings: Settings) -> None:
        client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key or "sk-placeholder"}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url.rstrip("/")
        self._sync_client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(timeout=settings.openai_timeout_seconds, **client_kwargs)
        self._model = settings.openai_embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._sync_client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._sync_client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._async_client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        response = await self._async_client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)


class LangGraphPostgresLongTermMemoryStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._database_url = settings.resolved_memory_database_url
        self._namespace_prefix = ("memories",)
        self._exit_stack = AsyncExitStack()
        self._store: Any | None = None
        self._setup_lock = asyncio.Lock()

    async def search(self, actor_id: str, query: str, limit: int = 5) -> list[LongTermMemoryRecord]:
        store = await self._get_store()
        results = await store.asearch(self._namespace(actor_id), query=query or None, limit=limit)
        return [_coerce_search_item_to_record(item) for item in results]

    async def save(self, actor_id: str, records: list[LongTermMemoryRecord]) -> None:
        if not records:
            return
        store = await self._get_store()
        namespace = self._namespace(actor_id)
        for record in records:
            await store.aput(
                namespace,
                _memory_key(record),
                record.model_dump(mode="json"),
                index=["text"],
            )

    async def aclose(self) -> None:
        await self._exit_stack.aclose()
        self._store = None

    async def _get_store(self) -> Any:
        if self._store is not None:
            return self._store
        async with self._setup_lock:
            if self._store is not None:
                return self._store
            if not self._database_url:
                raise RuntimeError("MEMORY_DATABASE_URL o TRACE_CAPTURE_DATABASE_URL es requerido para langgraph_postgres")
            AsyncPostgresStore = _import_async_postgres_store()
            store_cm = AsyncPostgresStore.from_conn_string(
                self._database_url,
                index={
                    "dims": self._settings.openai_embedding_dimensions,
                    "embed": OpenAIEmbeddingsAdapter(self._settings),
                    "fields": ["text"],
                },
            )
            self._store = await self._exit_stack.enter_async_context(store_cm)
            await self._store.setup()
            return self._store

    def _namespace(self, actor_id: str) -> tuple[str, ...]:
        return (*self._namespace_prefix, actor_id)


def build_long_term_memory_store(settings: Settings) -> LongTermMemoryStore:
    if settings.resolved_memory_backend == "langgraph_postgres":
        if not settings.resolved_memory_database_url:
            logger.warning("Falling back to in-memory store because no Postgres URL is configured")
            return InMemoryLongTermMemoryStore()
        try:
            _import_async_postgres_store()
            return LangGraphPostgresLongTermMemoryStore(settings)
        except Exception as exc:  # pragma: no cover - depende del paquete opcional
            logger.warning("Falling back to in-memory store because LangGraph Postgres store is unavailable: %s", exc)
    return InMemoryLongTermMemoryStore()


def _memory_key(record: LongTermMemoryRecord) -> str:
    payload = f"{record.kind}|{record.text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _coerce_search_item_to_record(item: Any) -> LongTermMemoryRecord:
    value = getattr(item, "value", item)
    if isinstance(value, LongTermMemoryRecord):
        return value
    if isinstance(value, dict):
        return LongTermMemoryRecord.model_validate(value)
    raise TypeError(f"Unsupported search item value: {type(value)!r}")


def _import_async_postgres_store() -> Any:
    from langgraph.store.postgres import AsyncPostgresStore

    return AsyncPostgresStore
