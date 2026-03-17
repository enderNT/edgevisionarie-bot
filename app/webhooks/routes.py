from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import ChatwootWebhook
from app.services.agent import ClinicAgentService

logger = logging.getLogger(__name__)


def build_webhook_router(agent_service: ClinicAgentService) -> APIRouter:
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])

    @router.post("/chatwoot", status_code=status.HTTP_202_ACCEPTED)
    async def chatwoot_webhook(payload: ChatwootWebhook) -> dict[str, str]:
        if not payload.latest_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Webhook sin contenido util para procesar.",
            )
        asyncio.create_task(_safe_process(agent_service, payload))
        return {"status": "accepted", "conversation_id": payload.conversation_id}

    return router


async def _safe_process(agent_service: ClinicAgentService, payload: ChatwootWebhook) -> None:
    try:
        await agent_service.process_webhook(payload)
    except Exception as exc:  # pragma: no cover - logging defensivo
        logger.exception("Background webhook processing failed for %s: %s", payload.conversation_id, exc)
