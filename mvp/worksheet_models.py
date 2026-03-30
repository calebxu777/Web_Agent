from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


WorksheetStatus = Literal[
    "draft",
    "active",
    "awaiting_input",
    "awaiting_confirmation",
    "completed",
    "cancelled",
]


class WorksheetFieldDefinition(BaseModel):
    name: str
    type: str = "str"
    kind: Literal["input", "internal", "output"] = "input"
    required: bool = False
    dont_ask: bool = False
    confirm: bool = False
    description: str = ""
    predicate: str = ""
    question: str = ""
    default: Any = None


class WorksheetDefinition(BaseModel):
    name: str
    description: str = ""
    trigger_intents: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    completion_rule: str = "all_required_filled"
    priority: int = 100
    fields: list[WorksheetFieldDefinition] = Field(default_factory=list)

    def field_map(self) -> dict[str, WorksheetFieldDefinition]:
        return {field.name: field for field in self.fields}


class WorksheetQueryRecord(BaseModel):
    query_text: str = ""
    rewritten_query: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    source: str = ""
    result_product_ids: list[str] = Field(default_factory=list)
    caption: str = ""
    debug_metadata: dict[str, Any] = Field(default_factory=dict)


class WorksheetInstance(BaseModel):
    worksheet_name: str
    status: WorksheetStatus = "draft"
    values: dict[str, Any] = Field(default_factory=dict)
    missing_required_fields: list[str] = Field(default_factory=list)
    result_refs: dict[str, Any] = Field(default_factory=dict)
    turn_history: list[str] = Field(default_factory=list)
    last_query_record: WorksheetQueryRecord | None = None
    last_updated_at: float = Field(default_factory=time.time)

    def touch(self) -> None:
        self.last_updated_at = time.time()
