from __future__ import annotations

import logging
from typing import Any

from app.graph.workflow import SupportWorkflow
from app.models.schemas import ChatwootWebhook
from app.observability.flow_logger import bind_flow, clear_flow, end_flow, mark_error, start_flow, step, substep
from app.services.chatwoot import ChatwootClient
from app.traces.context import (
    TurnTraceContext,
    bind_turn_trace_context,
    clear_turn_trace_context,
)
from app.traces.store import NoOpTraceStore, TraceStore

logger = logging.getLogger(__name__)


class AssistantService:
    def __init__(
        self,
        workflow: SupportWorkflow,
        chatwoot_client: ChatwootClient,
        trace_store: TraceStore | None = None,
    ) -> None:
        self._workflow = workflow
        self._chatwoot_client = chatwoot_client
        self._trace_store = trace_store or NoOpTraceStore()

    async def process_webhook(self, payload: ChatwootWebhook, flow_id: str | None = None) -> dict[str, Any]:
        resolved_flow_id = flow_id or payload.conversation_id
        bind_flow(resolved_flow_id, payload.conversation_id)
        start_flow(payload.latest_message)
        trace_context = TurnTraceContext(
            flow_id=resolved_flow_id,
            webhook=payload,
            llm_backend=self._workflow.llm_backend_name,
            llm_model=self._workflow.llm_model_name,
        )
        bind_turn_trace_context(trace_context)
        try:
            step("2. state_routing_and_graph", "RUN", "ejecutando LangGraph")
            result = await self._workflow.run(payload)
            trace_context.capture_state_after(result)
            step(
                "2. state_routing_and_graph",
                "OK",
                f"intent={result.get('intent')} confidence={result.get('confidence', 0):.2f}",
            )

            response_text = result.get("response_text")
            if response_text:
                step("4. outbound_response", "RUN", "enviando respuesta a Chatwoot")
                try:
                    await self._chatwoot_client.send_message(
                        payload.conversation_id,
                        response_text,
                        account_id=payload.account_id,
                    )
                    trace_context.capture_outbound(response_text=response_text, sent=True)
                    step("4. outbound_response", "OK", "respuesta enviada o logueada")
                except Exception as exc:  # pragma: no cover - depende de Chatwoot
                    trace_context.capture_outbound(response_text=response_text, sent=False, error=exc)
                    mark_error("4. outbound_response", exc)
                    logger.exception("Failed to send Chatwoot response for %s: %s", payload.conversation_id, exc)
            else:
                trace_context.capture_outbound(response_text="", sent=False)
                substep("4. outbound_response", "WARN", "sin response_text para enviar")

            end_flow("OK", f"branch={result.get('next_node', result.get('intent', 'unknown'))}")
            return result
        except Exception as exc:
            trace_context.mark_error(exc)
            mark_error("flow_execution", exc)
            end_flow("ERROR")
            raise
        finally:
            self._trace_store.enqueue(trace_context.freeze())
            clear_turn_trace_context()
            clear_flow()
