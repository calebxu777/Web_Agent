from __future__ import annotations

from src.schema import IntentType

from mvp.worksheet_models import WorksheetDefinition, WorksheetInstance
from mvp.worksheet_registry import WorksheetRegistry


class WorksheetRouter:
    def __init__(self, registry: WorksheetRegistry):
        self.registry = registry

    def resolve(
        self,
        intent: IntentType,
        active_instance: WorksheetInstance | None = None,
    ) -> WorksheetDefinition | None:
        if active_instance and active_instance.status not in {"completed", "cancelled"}:
            try:
                return self.registry.get(active_instance.worksheet_name)
            except KeyError:
                pass

        return self.registry.find_for_intent(intent.value)
