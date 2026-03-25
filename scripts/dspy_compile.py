from __future__ import annotations

import argparse
from pathlib import Path

from app.dspy.datasets import load_jsonl_examples
from app.dspy.metrics import discovery_call_metric, route_metric, text_overlap_metric
from app.dspy.programs import build_programs
from app.dspy.signatures import require_dspy
from app.settings import Settings


TASK_CONFIG = {
    "route": {
        "input_keys": [
            "user_message",
            "conversation_summary",
            "active_goal",
            "stage",
            "pending_action",
            "pending_question",
            "discovery_call_slots_json",
            "last_tool_result",
            "last_user_message",
            "last_assistant_message",
            "memories_text",
        ],
        "output_keys": [
            "next_node",
            "intent",
            "confidence",
            "needs_retrieval",
            "next_active_goal",
            "next_stage",
            "next_pending_action",
            "next_pending_question",
            "clear_slots",
            "clear_last_tool_result",
            "route_reason",
        ],
        "metric": route_metric,
    },
    "discovery_call": {
        "input_keys": [
            "user_message",
            "memories_text",
            "company_context",
            "contact_name",
            "current_slots_json",
            "pending_question",
        ],
        "output_keys": [
            "lead_name",
            "project_need",
            "preferred_date",
            "preferred_time",
            "missing_fields",
            "should_handoff",
            "confidence",
        ],
        "metric": discovery_call_metric,
    },
    "rag": {
        "input_keys": ["question", "memories_text", "context"],
        "output_keys": ["reply_text"],
        "metric": text_overlap_metric,
    },
    "conversation": {
        "input_keys": ["user_message", "memories_text"],
        "output_keys": ["reply_text"],
        "metric": text_overlap_metric,
    },
    "summary": {
        "input_keys": ["current_summary", "active_goal", "stage", "user_message", "assistant_message"],
        "output_keys": ["updated_summary"],
        "metric": text_overlap_metric,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compila un programa DSPy para una tarea concreta.")
    parser.add_argument("task", choices=sorted(TASK_CONFIG))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dspy"))
    args = parser.parse_args()

    settings = Settings()
    dspy = require_dspy()
    config = TASK_CONFIG[args.task]
    dataset_path = args.dataset or Path("datasets/dspy") / f"{args.task}.jsonl"
    examples = load_jsonl_examples(dataset_path, config["input_keys"], config["output_keys"])

    lm_kwargs = {}
    if settings.resolved_llm_api_key:
        lm_kwargs["api_key"] = settings.resolved_llm_api_key
    if settings.resolved_llm_base_url:
        lm_kwargs["api_base"] = settings.resolved_llm_base_url
        lm_kwargs["model_type"] = "chat"
    dspy.configure(lm=dspy.LM(settings.resolved_dspy_model, **lm_kwargs))

    programs = build_programs()
    program = programs[args.task]
    teleprompter = dspy.MIPROv2(metric=config["metric"], auto="light")
    optimized = teleprompter.compile(program, trainset=examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.task}.json"
    optimized.save(str(output_path))
    print(f"Programa compilado guardado en {output_path}")


if __name__ == "__main__":
    main()
