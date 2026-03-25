from __future__ import annotations

import json
import re
from typing import Any


def route_metric(example: Any, prediction: Any, trace: Any | None = None) -> float:
    del trace
    score = 0.0
    total = 5.0
    if getattr(prediction, "next_node", "") == getattr(example, "next_node", ""):
        score += 1.0
    if getattr(prediction, "intent", "") == getattr(example, "intent", ""):
        score += 1.0
    if _as_bool(getattr(prediction, "needs_retrieval", False)) == _as_bool(
        getattr(example, "needs_retrieval", False)
    ):
        score += 1.0
    if getattr(prediction, "next_active_goal", "") == getattr(example, "next_active_goal", ""):
        score += 1.0
    if getattr(prediction, "next_stage", "") == getattr(example, "next_stage", ""):
        score += 1.0
    return score / total


def discovery_call_metric(example: Any, prediction: Any, trace: Any | None = None) -> float:
    del trace
    fields = (
        "lead_name",
        "project_need",
        "preferred_date",
        "preferred_time",
        "should_handoff",
    )
    total = float(len(fields) + 1)
    score = 0.0
    for field in fields:
        if _normalize(getattr(prediction, field, "")) == _normalize(getattr(example, field, "")):
            score += 1.0
    if _parse_list(getattr(prediction, "missing_fields", "")) == _parse_list(getattr(example, "missing_fields", "")):
        score += 1.0
    return score / total


def text_overlap_metric(example: Any, prediction: Any, trace: Any | None = None) -> float:
    del trace
    expected = _normalize(getattr(example, "reply_text", "") or getattr(example, "updated_summary", ""))
    predicted = _normalize(getattr(prediction, "reply_text", "") or getattr(prediction, "updated_summary", ""))
    if not expected or not predicted:
        return 0.0
    expected_tokens = set(expected.split())
    predicted_tokens = set(predicted.split())
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & predicted_tokens) / len(expected_tokens)


def _normalize(value: Any) -> str:
    text = str(value).lower().strip()
    return re.sub(r"\s+", " ", text)


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in text.split(",")]
    return [str(item).strip() for item in parsed if str(item).strip()]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "si", "sí"}
