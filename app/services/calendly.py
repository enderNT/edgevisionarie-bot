from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CalendlyBookingMatch:
    invitee_email: str
    invitee_name: str = ""
    invitee_uri: str = ""
    scheduled_event_uri: str = ""
    start_time: str = ""
    end_time: str = ""
    status: str = ""
    cancel_url: str = ""
    reschedule_url: str = ""


@dataclass(slots=True)
class CalendlyValidationResult:
    found: bool
    reason: str = ""
    match: CalendlyBookingMatch | None = None


class CalendlyService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return bool(
            self._settings.calendly_api_token
            and self._settings.calendly_organization_uri
            and self._settings.calendly_scheduling_url
        )

    @property
    def scheduling_url(self) -> str:
        return self._settings.calendly_scheduling_url or ""

    async def validate_booking_by_email(self, email: str) -> CalendlyValidationResult:
        normalized_email = email.strip().lower()
        if not normalized_email:
            return CalendlyValidationResult(found=False, reason="missing-email")
        if not self.enabled:
            return CalendlyValidationResult(found=False, reason="not-configured")

        event_uris = await self._list_active_scheduled_event_uris()
        for event_uri in event_uris:
            event_uuid = event_uri.rsplit("/", 1)[-1]
            invitees = await self._get_invitees(event_uuid)
            for invitee in invitees:
                invitee_email = _first_str(
                    invitee,
                    "email",
                    "invitee_email",
                    "invitee",
                    "contact_email",
                    "email_address",
                )
                if not invitee_email or invitee_email.strip().lower() != normalized_email:
                    continue
                return CalendlyValidationResult(
                    found=True,
                    reason="invitee-found",
                    match=CalendlyBookingMatch(
                        invitee_email=invitee_email.strip(),
                        invitee_name=_first_str(invitee, "name", "invitee_name", "full_name"),
                        invitee_uri=_first_str(invitee, "uri", "invitee_uri"),
                        scheduled_event_uri=event_uri,
                        start_time=_first_str(invitee, "start_time")
                        or _first_str(invitee, "event_start_time")
                        or _first_str(invitee, "scheduled_event_start_time")
                        or _first_str(invitee, "scheduled_event", "start_time"),
                        end_time=_first_str(invitee, "end_time")
                        or _first_str(invitee, "event_end_time")
                        or _first_str(invitee, "scheduled_event_end_time"),
                        status=_first_str(invitee, "status") or _first_str(invitee, "event_status"),
                        cancel_url=_first_str(invitee, "cancel_url") or _first_str(invitee, "cancellation_url"),
                        reschedule_url=_first_str(invitee, "reschedule_url")
                        or _first_str(invitee, "reschedule_link"),
                    ),
                )

        return CalendlyValidationResult(found=False, reason="not-found")

    async def _list_active_scheduled_event_uris(self) -> list[str]:
        payload = await self._get_json(
            "/scheduled_events",
            params={
                "organization": self._settings.calendly_organization_uri,
                "status": "active",
                "count": self._settings.calendly_validation_page_size,
                "sort": "start_time:asc",
            },
        )
        return [_first_str(item, "uri", "scheduled_event_uri") for item in _extract_items(payload) if _first_str(item, "uri", "scheduled_event_uri")]

    async def _get_invitees(self, event_uuid: str) -> list[dict[str, Any]]:
        payload = await self._get_json(f"/scheduled_events/{event_uuid}/invitees")
        return _extract_items(payload)

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {
            "authorization": f"Bearer {self._settings.calendly_api_token}",
            "content-type": "application/json",
        }
        base_url = self._settings.calendly_api_base_url.rstrip("/")
        async with httpx.AsyncClient(base_url=base_url, timeout=20) as client:
            response = await client.get(path, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data
            return {"collection": data}


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("collection", "data", "results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if all(isinstance(value, dict) for value in payload.values()):
        return [value for value in payload.values() if isinstance(value, dict)]
    return []


def _first_str(mapping: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = _first_str(value, "uri", "email", "name", "start_time", "end_time")
            if nested:
                return nested
    for value in mapping.values():
        if isinstance(value, dict):
            nested = _first_str(value, "uri", "email", "name", "start_time", "end_time")
            if nested:
                return nested
    return ""


def format_calendly_time(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return value
