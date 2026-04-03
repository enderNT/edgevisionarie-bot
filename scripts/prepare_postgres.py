from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any

import asyncpg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from app.memory_runtime.store import OpenAIEmbeddingsAdapter
from app.settings import Settings
from app.traces.orm import Base
from app.traces.store import PostgresTraceStore


@dataclass
class NormalizedDsn:
    original_scheme: str
    normalized_psycopg_dsn: str
    normalized_asyncpg_dsn: str


async def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the production Postgres database for this app.")
    parser.add_argument("--dsn", required=True, help="Raw Postgres DSN.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    dsn = normalize_dsn(args.dsn)
    settings = Settings(
        trace_capture_enabled=True,
        trace_capture_database_url=dsn.normalized_asyncpg_dsn,
        memory_backend="langgraph_postgres",
        memory_database_url=dsn.normalized_psycopg_dsn,
        checkpoint_database_url=dsn.normalized_psycopg_dsn,
    )

    await bootstrap_trace_store(settings)
    await bootstrap_checkpoint_store(settings)
    await bootstrap_memory_store(settings)
    verification = await inspect_catalog(dsn.normalized_psycopg_dsn)

    payload = {
        "normalized_dsn": {
            "original_scheme": dsn.original_scheme,
            "normalized_psycopg_dsn": redact_dsn(dsn.normalized_psycopg_dsn),
            "normalized_asyncpg_dsn": redact_dsn(dsn.normalized_asyncpg_dsn),
        },
        "verification": verification,
    }
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))


def normalize_dsn(raw_dsn: str) -> NormalizedDsn:
    dsn = raw_dsn.strip()
    if "://" not in dsn:
        raise ValueError("The DSN must include a URL scheme")

    scheme, remainder = dsn.split("://", 1)
    original_scheme = scheme
    normalized_scheme = scheme
    if scheme == "postgres":
        normalized_scheme = "postgresql"
    elif scheme == "postgresql+asyncpg":
        normalized_scheme = "postgresql"
    elif scheme != "postgresql":
        raise ValueError(f"Unsupported DSN scheme: {scheme}")

    psycopg_dsn = f"{normalized_scheme}://{remainder}"
    asyncpg_dsn = f"postgresql+asyncpg://{remainder}"
    return NormalizedDsn(
        original_scheme=original_scheme,
        normalized_psycopg_dsn=psycopg_dsn,
        normalized_asyncpg_dsn=asyncpg_dsn,
    )


async def bootstrap_trace_store(settings: Settings) -> None:
    store = PostgresTraceStore(settings)
    await store.start()
    await store.stop()


async def bootstrap_checkpoint_store(settings: Settings) -> None:
    async with AsyncPostgresSaver.from_conn_string(settings.resolved_checkpoint_database_url) as saver:
        await saver.setup()


async def bootstrap_memory_store(settings: Settings) -> None:
    index_config = {
        "dims": settings.openai_embedding_dimensions,
        "embed": OpenAIEmbeddingsAdapter(settings),
        "fields": ["text"],
    }
    async with AsyncPostgresStore.from_conn_string(
        settings.resolved_memory_database_url,
        index=index_config,
    ) as store:
        await store.setup()


async def inspect_catalog(dsn: str) -> dict[str, Any]:
    conn = await asyncpg.connect(dsn)
    try:
        extensions = await conn.fetch(
            """
            SELECT extname
            FROM pg_extension
            WHERE extname IN ('vector')
            ORDER BY extname
            """
        )
        tables = await conn.fetch(
            """
            SELECT schemaname, tablename
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename IN (
                'checkpoint_migrations',
                'checkpoints',
                'checkpoint_blobs',
                'checkpoint_writes',
                'turn_traces',
                'discovery_call_flows',
                'store',
                'store_vectors',
                'store_migrations',
                'vector_migrations'
              )
            ORDER BY schemaname, tablename
            """
        )
        indexes = await conn.fetch(
            """
            SELECT tablename, indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename IN ('turn_traces', 'discovery_call_flows', 'store', 'store_vectors')
            ORDER BY tablename, indexname
            """
        )
        checkpoint_indexes = await conn.fetch(
            """
            SELECT tablename, indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename IN ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes')
            ORDER BY tablename, indexname
            """
        )
        constraints = await conn.fetch(
            """
            SELECT
              tc.table_name,
              tc.constraint_name,
              tc.constraint_type
            FROM information_schema.table_constraints tc
            WHERE tc.table_schema = 'public'
              AND tc.table_name IN (
                'checkpoint_blobs',
                'checkpoints',
                'checkpoint_writes',
                'turn_traces',
                'discovery_call_flows',
                'store',
                'store_vectors'
              )
            ORDER BY tc.table_name, tc.constraint_type, tc.constraint_name
            """
        )
        foreign_keys = await conn.fetch(
            """
            SELECT
              tc.table_name,
              kcu.column_name,
              ccu.table_name AS foreign_table_name,
              ccu.column_name AS foreign_column_name,
              tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = 'public'
              AND tc.table_name IN ('turn_traces', 'store_vectors')
            ORDER BY tc.table_name, tc.constraint_name
            """
        )
        columns = await conn.fetch(
            """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN (
                'checkpoint_blobs',
                'checkpoints',
                'checkpoint_writes',
                'turn_traces',
                'discovery_call_flows',
                'store',
                'store_vectors'
              )
            ORDER BY table_name, ordinal_position
            """
        )
        return {
            "extensions": [row["extname"] for row in extensions],
            "tables": [dict(row) for row in tables],
            "indexes": [dict(row) for row in [*indexes, *checkpoint_indexes]],
            "constraints": [dict(row) for row in constraints],
            "foreign_keys": [dict(row) for row in foreign_keys],
            "columns": [dict(row) for row in columns],
        }
    finally:
        await conn.close()


def redact_dsn(dsn: str) -> str:
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    creds, tail = rest.split("@", 1)
    username = creds.split(":", 1)[0]
    return f"{scheme}://{username}:***@{tail}"


if __name__ == "__main__":
    asyncio.run(main())
