from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import Uuid


JSONType = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    pass


class DiscoveryCallFlowORM(Base):
    __tablename__ = "discovery_call_flows"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    contact_id: Mapped[str] = mapped_column(String(255), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="active")
    opening_turn_trace_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    closing_turn_trace_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    latest_stage: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    latest_slots: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    final_payload: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    __table_args__ = (
        Index("ix_discovery_call_flows_conversation_status", "conversation_id", "status"),
    )


class TurnTraceORM(Base):
    __tablename__ = "turn_traces"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    flow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    contact_id: Mapped[str] = mapped_column(String(255), nullable=False)
    account_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    trace_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    llm_backend: Mapped[str] = mapped_column(String(64), nullable=False, default="raw")
    llm_model: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    next_node: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    intent: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    needs_retrieval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    handoff_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    response_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    discovery_call_flow_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("discovery_call_flows.id"), nullable=True
    )
    webhook_snapshot: Mapped[dict] = mapped_column(JSONType, nullable=False)
    state_before: Mapped[dict] = mapped_column(JSONType, nullable=False)
    state_after: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    route_input: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    route_output: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    rag_trace: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    discovery_call_trace: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    outbound_trace: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    __table_args__ = (
        Index("ix_turn_traces_created_at", "created_at"),
        Index("ix_turn_traces_conversation_created_at", "conversation_id", "created_at"),
        Index("ix_turn_traces_contact_created_at", "contact_id", "created_at"),
        Index("ix_turn_traces_next_node_created_at", "next_node", "created_at"),
    )
