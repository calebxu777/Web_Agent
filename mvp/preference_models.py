from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.schema import DecomposedQuery, IntentType


PreferenceKind = Literal[
    "color",
    "brand",
    "size",
    "fit",
    "style",
    "material",
    "budget_max",
    "budget_min",
    "category",
]
PreferencePolarity = Literal["positive", "negative"]

SUPPORTED_PREFERENCE_KINDS: tuple[PreferenceKind, ...] = (
    "color",
    "brand",
    "size",
    "fit",
    "style",
    "material",
    "budget_max",
    "budget_min",
    "category",
)
CATEGORICAL_PREFERENCE_KINDS = {
    "color",
    "brand",
    "size",
    "fit",
    "style",
    "material",
    "category",
}
NUMERIC_PREFERENCE_KINDS = {"budget_max", "budget_min"}
PREFERENCE_CONTEXT_ORDER = [
    "color",
    "avoid_color",
    "brand",
    "avoid_brand",
    "size",
    "fit",
    "style",
    "avoid_style",
    "material",
    "avoid_material",
    "category",
    "budget_min",
    "budget_max",
]


def normalize_preference_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = " ".join(text.split())
    return text


def normalize_numeric_preference(value: Any) -> float | int:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric preferences.")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        cleaned = str(value or "").strip().replace("$", "").replace(",", "")
        if not cleaned:
            raise ValueError("Numeric preference value is empty.")
        number = float(cleaned)
    if number.is_integer():
        return int(number)
    return round(number, 2)


def is_preference_list_key(key: str) -> bool:
    base_kind = key[6:] if key.startswith("avoid_") else key
    return base_kind in CATEGORICAL_PREFERENCE_KINDS


def merge_preference_maps(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for source in (base or {}, incoming or {}):
        for key, value in source.items():
            if value in (None, "", []):
                continue

            if is_preference_list_key(key):
                existing_values = list(merged.get(key) or [])
                seen = {normalize_preference_text(item) for item in existing_values}
                for item in value if isinstance(value, list) else [value]:
                    normalized = normalize_preference_text(item)
                    if normalized and normalized not in seen:
                        existing_values.append(normalized)
                        seen.add(normalized)
                if existing_values:
                    merged[key] = existing_values
                continue

            base_kind = key[6:] if key.startswith("avoid_") else key
            if base_kind in NUMERIC_PREFERENCE_KINDS:
                merged[key] = normalize_numeric_preference(value)
            else:
                normalized = normalize_preference_text(value)
                if normalized:
                    merged[key] = normalized

    return merged


def _preference_label(key: str) -> str:
    avoid = key.startswith("avoid_")
    base_key = key[6:] if avoid else key
    labels = {
        "color": "colors",
        "brand": "brands",
        "size": "sizes",
        "fit": "fits",
        "style": "styles",
        "material": "materials",
        "category": "categories",
        "budget_min": "budget min",
        "budget_max": "budget max",
    }
    label = labels.get(base_key, base_key.replace("_", " "))
    if avoid:
        return f"avoid {label}"
    return label


def _preference_value_to_text(key: str, value: Any) -> str:
    base_key = key[6:] if key.startswith("avoid_") else key
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if base_key in NUMERIC_PREFERENCE_KINDS and value not in (None, ""):
        return f"${value}"
    return str(value)


def build_preference_context(
    profile_or_preferences: dict[str, Any] | "SessionPreferenceProfile" | "StoredPreferenceProfile" | None,
) -> str:
    if profile_or_preferences is None:
        return ""

    if isinstance(profile_or_preferences, dict):
        preferences = profile_or_preferences
    else:
        preferences = profile_or_preferences.preferences

    if not preferences:
        return ""

    ordered_keys = [key for key in PREFERENCE_CONTEXT_ORDER if key in preferences]
    unordered_keys = [key for key in preferences.keys() if key not in ordered_keys]
    lines = []
    for key in ordered_keys + sorted(unordered_keys):
        value = preferences.get(key)
        if value in (None, "", []):
            continue
        lines.append(f"- {_preference_label(key)}: {_preference_value_to_text(key, value)}")

    if not lines:
        return ""

    return "User preference context:\n" + "\n".join(lines)


class PreferenceItem(BaseModel):
    kind: PreferenceKind
    value: str | float | int
    polarity: PreferencePolarity = "positive"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_text: str = ""

    @model_validator(mode="after")
    def _normalize(self) -> "PreferenceItem":
        if self.kind in NUMERIC_PREFERENCE_KINDS:
            self.value = normalize_numeric_preference(self.value)
        else:
            normalized = normalize_preference_text(self.value)
            if not normalized:
                raise ValueError("Preference values must be non-empty.")
            self.value = normalized

        self.source_text = " ".join(str(self.source_text or "").strip().split())
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        return self

    @property
    def storage_key(self) -> str:
        if self.polarity == "negative":
            return f"avoid_{self.kind}"
        return self.kind


class PreferenceExtractionResult(BaseModel):
    preferences: list[PreferenceItem] = Field(default_factory=list)


class TurnAnalysisResult(BaseModel):
    intent: IntentType = IntentType.GENERAL_TALK
    tags: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    rewritten_query: str = ""
    preferences: list[PreferenceItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> "TurnAnalysisResult":
        self.tags = [" ".join(str(tag).strip().split()) for tag in self.tags if str(tag).strip()]

        normalized_filters: dict[str, Any] = {}
        for key, value in dict(self.filters or {}).items():
            if value in (None, ""):
                continue
            if key in {"price_min", "price_max"}:
                normalized_filters[key] = normalize_numeric_preference(value)
            elif isinstance(value, str):
                normalized_filters[key] = " ".join(value.strip().split())
            else:
                normalized_filters[key] = value
        self.filters = normalized_filters
        self.rewritten_query = " ".join(str(self.rewritten_query or "").strip().split())
        return self

    def to_decomposed_query(
        self,
        original_query: str,
        default_intent: IntentType = IntentType.TEXT_SEARCH,
    ) -> DecomposedQuery:
        query_intent = self.intent if self.intent in {
            IntentType.TEXT_SEARCH,
            IntentType.WEB_SEARCH,
            IntentType.IMAGE_SEARCH,
        } else default_intent
        return DecomposedQuery(
            intent=query_intent,
            original_query=original_query,
            tags=list(self.tags),
            filters=dict(self.filters),
            rewritten_query=self.rewritten_query or original_query,
        )


class SessionPreferenceProfile(BaseModel):
    user_id: str
    session_id: str
    preferences: dict[str, Any] = Field(default_factory=dict)
    evidence: list[PreferenceItem] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)

    def merge_items(self, items: list[PreferenceItem]) -> None:
        if not items:
            self.updated_at = time.time()
            return

        next_preferences = dict(self.preferences)
        for item in items:
            key = item.storage_key
            if is_preference_list_key(key):
                current = list(next_preferences.get(key) or [])
                if item.value not in current:
                    current.append(item.value)
                next_preferences[key] = current
            else:
                next_preferences[key] = item.value
            self.evidence.append(item)

        self.preferences = merge_preference_maps({}, next_preferences)
        self.updated_at = time.time()


class StoredPreferenceProfile(BaseModel):
    user_id: str
    preferences: dict[str, Any] = Field(default_factory=dict)
    updated_at: float = Field(default_factory=time.time)
    source_session_id: str = ""
