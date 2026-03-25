from __future__ import annotations

import json
from pathlib import Path

from app.models.schemas import CompanyConfig


class CompanyConfigLoader:
    def __init__(self, config_path: Path | str) -> None:
        self._config_path = Path(config_path)
        self._cached: CompanyConfig | None = None

    def load(self) -> CompanyConfig:
        if self._cached is None:
            with self._config_path.open("r", encoding="utf-8") as file:
                self._cached = CompanyConfig.model_validate(json.load(file))
        return self._cached
