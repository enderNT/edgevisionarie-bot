from __future__ import annotations

import logging
from typing import Protocol

from app.settings import Settings

logger = logging.getLogger(__name__)


class MemoryStore(Protocol):
    async def search(self, contact_id: str, query: str, limit: int = 5) -> list[str]:
        ...

    async def save_exchange(self, contact_id: str, user_message: str, assistant_message: str) -> None:
        ...


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    async def search(self, contact_id: str, query: str, limit: int = 5) -> list[str]:
        del query
        memories = self._store.get(contact_id, [])
        return memories[-limit:]

    async def save_exchange(self, contact_id: str, user_message: str, assistant_message: str) -> None:
        snippets = self._store.setdefault(contact_id, [])
        snippets.append(f"Usuario: {user_message}")
        snippets.append(f"Asistente: {assistant_message}")


class Mem0LocalMemoryStore:
    def __init__(self) -> None:
        from mem0 import Memory

        self._client = Memory()

    async def search(self, contact_id: str, query: str, limit: int = 5) -> list[str]:
        results = self._client.search(query, filters={"user_id": contact_id})
        return [item.get("memory", "") for item in results[:limit] if item.get("memory")]

    async def save_exchange(self, contact_id: str, user_message: str, assistant_message: str) -> None:
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        self._client.add(messages, user_id=contact_id)


class Mem0PlatformMemoryStore:
    def __init__(self, settings: Settings) -> None:
        from mem0 import MemoryClient

        client_kwargs: dict[str, str] = {}
        if settings.mem0_api_key:
            client_kwargs["api_key"] = settings.mem0_api_key
        if settings.mem0_org_id:
            client_kwargs["org_id"] = settings.mem0_org_id
        if settings.mem0_project_id:
            client_kwargs["project_id"] = settings.mem0_project_id
        self._client = MemoryClient(**client_kwargs)

    async def search(self, contact_id: str, query: str, limit: int = 5) -> list[str]:
        results = self._client.search(query, version="v2", filters={"user_id": contact_id})
        return [item.get("memory", "") for item in results[:limit] if item.get("memory")]

    async def save_exchange(self, contact_id: str, user_message: str, assistant_message: str) -> None:
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        self._client.add(messages, user_id=contact_id, version="v2")


def build_memory_store(settings: Settings) -> MemoryStore:
    try:
        if settings.memory_backend == "mem0_local":
            return Mem0LocalMemoryStore()
        if settings.memory_backend == "mem0_platform":
            return Mem0PlatformMemoryStore(settings)
    except Exception as exc:  # pragma: no cover - depende de entorno externo
        logger.warning("Falling back to in-memory store because mem0 failed to initialize: %s", exc)
    return InMemoryMemoryStore()
