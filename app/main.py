from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.graph.workflow import SupportWorkflow
from app.observability.flow_logger import configure_flow_logger
from app.services.assistant_service import AssistantService
from app.services.calendly import CalendlyService
from app.services.chatwoot import ChatwootClient
from app.services.company_config import CompanyConfigLoader
from app.services.llm import SupportLLMService, build_llm_provider
from app.services.memory import build_memory_store
from app.services.router import StateRoutingService
from app.services.qdrant import QdrantRetrievalService
from app.settings import get_settings
from app.traces import build_trace_store
from app.webhooks.routes import build_webhook_router


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    configure_flow_logger(getattr(logging, settings.log_level.upper(), logging.INFO))

    company_config_loader = CompanyConfigLoader(settings.company_config_path)
    llm_provider = build_llm_provider(settings)
    llm_service = SupportLLMService(llm_provider, settings=settings)
    router_service = StateRoutingService(settings, llm_service)
    calendly_service = CalendlyService(settings)
    memory_store = build_memory_store(settings)
    qdrant_service = QdrantRetrievalService(settings)
    trace_store = build_trace_store(settings)
    workflow = SupportWorkflow(
        router_service,
        llm_service,
        calendly_service,
        memory_store,
        company_config_loader,
        qdrant_service,
        settings,
    )
    agent_service = AssistantService(workflow, ChatwootClient(settings), trace_store=trace_store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        await trace_store.start()
        try:
            yield
        finally:
            await trace_store.stop()

    app = FastAPI(title=" Assistant", version="0.1.0", lifespan=lifespan)
    app.include_router(build_webhook_router(agent_service))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    return app
