from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.dspy.signatures import require_dspy


def load_jsonl_examples(dataset_path: Path, input_keys: list[str], output_keys: list[str]):
    dspy = require_dspy()
    examples = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            inputs = {key: row["inputs"].get(key, "") for key in input_keys}
            outputs = {key: row["outputs"].get(key, "") for key in output_keys}
            examples.append(dspy.Example(**inputs, **outputs).with_inputs(*input_keys))
    return examples
