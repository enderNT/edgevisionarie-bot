from __future__ import annotations

import logging

import httpx

from app.settings import Settings

logger = logging.getLogger(__name__)


class ChatwootClient:
    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.chatwoot_reply_enabled
        self._base_url = settings.chatwoot_api_base_url.rstrip("/") if settings.chatwoot_api_base_url else None
        self._token = settings.chatwoot_api_token

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._base_url and self._token)

    async def send_message(self, conversation_id: str, content: str) -> None:
        if not self.enabled:
            logger.info("Chatwoot reply disabled. Conversation %s -> %s", conversation_id, content)
            return
        url = f"{self._base_url}/conversations/{conversation_id}/messages"
        headers = {"Content-Type": "application/json", "api_access_token": self._token}
        payload = {"content": content, "message_type": "outgoing", "private": False}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
