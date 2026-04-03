from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langgraph.checkpoint.memory import MemorySaver

from app.settings import Settings


@asynccontextmanager
async def build_checkpointer(settings: Settings) -> AsyncIterator[Any]:
    database_url = settings.resolved_checkpoint_database_url
    if not database_url:
        yield MemorySaver()
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(database_url) as saver:
        await saver.setup()
        yield saver
