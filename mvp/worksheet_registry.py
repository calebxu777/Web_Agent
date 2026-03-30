from __future__ import annotations

import json
from pathlib import Path

import yaml

from mvp.worksheet_models import WorksheetDefinition


class WorksheetRegistry:
    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir or Path(__file__).with_name("worksheets"))
        self._definitions: dict[str, WorksheetDefinition] = {}
        self.reload()

    def reload(self) -> None:
        self._definitions = {}
        if not self.base_dir.exists():
            return

        for path in sorted(self.base_dir.iterdir()):
            if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
                continue

            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))

            if not payload:
                continue

            definition = WorksheetDefinition.model_validate(payload)
            self._definitions[definition.name] = definition

    def get(self, name: str) -> WorksheetDefinition:
        if name not in self._definitions:
            raise KeyError(f"Unknown worksheet: {name}")
        return self._definitions[name]

    def find_for_intent(self, intent_name: str) -> WorksheetDefinition | None:
        matches = [
            definition
            for definition in self._definitions.values()
            if intent_name in definition.trigger_intents
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda item: item.priority)[0]
