from __future__ import annotations

from typing import Any

from app.dspy.signatures import (
    ConversationReplySignature,
    DiscoveryCallSignature,
    RagReplySignature,
    RouteDecisionSignature,
    SummarySignature,
    require_dspy,
)


def build_programs() -> dict[str, Any]:
    dspy = require_dspy()
    return {
        "conversation": dspy.Predict(ConversationReplySignature),
        "rag": dspy.ChainOfThought(RagReplySignature),
        "summary": dspy.Predict(SummarySignature),
        "route": dspy.Predict(RouteDecisionSignature),
        "discovery_call": dspy.ChainOfThought(DiscoveryCallSignature),
    }
