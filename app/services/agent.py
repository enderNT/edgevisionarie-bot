from __future__ import annotations

import logging
from typing import Any

from app.graph.workflow import ClinicWorkflow
from app.models.schemas import ChatwootWebhook
from app.services.chatwoot import ChatwootClient

logger = logging.getLogger(__name__)


class ClinicAgentService:
    def __init__(self, workflow: ClinicWorkflow, chatwoot_client: ChatwootClient) -> None:
        self._workflow = workflow
        self._chatwoot_client = chatwoot_client

    async def process_webhook(self, payload: ChatwootWebhook) -> dict[str, Any]:
        result = await self._workflow.run(payload)
        response_text = result.get("response_text")
        if response_text:
            try:
                await self._chatwoot_client.send_message(payload.conversation_id, response_text)
            except Exception as exc:  # pragma: no cover - depende de Chatwoot
                logger.exception("Failed to send Chatwoot response for %s: %s", payload.conversation_id, exc)
        return result
