from __future__ import annotations

import re
import time
from typing import Any

from src.schema import DecomposedQuery, IntentType

from mvp.worksheet_models import (
    WorksheetDefinition,
    WorksheetFieldDefinition,
    WorksheetInstance,
    WorksheetQueryRecord,
)


_PRODUCT_TYPE_KEYWORDS = [
    "jacket",
    "jackets",
    "coat",
    "coats",
    "blazer",
    "blazers",
    "shacket",
    "shackets",
    "parka",
    "parkas",
    "hoodie",
    "hoodies",
    "windbreaker",
    "windbreakers",
    "bomber",
    "bombers",
    "shirt",
    "shirts",
    "dress",
    "dresses",
    "shoe",
    "shoes",
    "sneaker",
    "sneakers",
    "bag",
    "bags",
    "boot",
    "boots",
    "jean",
    "jeans",
    "pant",
    "pants",
    "skirt",
    "skirts",
    "sweater",
    "sweaters",
]

_APPAREL_PRODUCT_TYPES = {
    "jacket",
    "coat",
    "blazer",
    "shacket",
    "parka",
    "hoodie",
    "windbreaker",
    "bomber",
    "shirt",
    "dress",
    "shoe",
    "sneaker",
    "boot",
    "jean",
    "pant",
    "skirt",
    "sweater",
}

_COMPARE_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|which\s+(one|is)\s+better|difference\s+between|"
    r"pros\s+and\s+cons|head\s+to\s+head)\b",
    re.IGNORECASE,
)

_ORDINAL_MAP = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
}


def _singularize(value: str) -> str:
    text = (value or "").strip().lower()
    if text.endswith("ies") and len(text) > 3:
        return text[:-3] + "y"
    if text.endswith("s") and not text.endswith("ss") and len(text) > 3:
        return text[:-1]
    return text


class WorksheetEngine:
    def create_instance(self, definition: WorksheetDefinition) -> WorksheetInstance:
        values: dict[str, Any] = {}
        for field in definition.fields:
            if field.default is not None:
                values[field.name] = field.default
        instance = WorksheetInstance(
            worksheet_name=definition.name,
            status="draft",
            values=values,
        )
        instance.missing_required_fields = self.missing_required_fields(definition, values)
        return instance

    def apply_product_search_turn(
        self,
        definition: WorksheetDefinition,
        instance: WorksheetInstance,
        message: str,
        decomposed: DecomposedQuery,
        include_web: bool = False,
    ) -> WorksheetInstance:
        values = dict(instance.values)
        updates = self._extract_updates(message, decomposed)

        for key, value in updates.items():
            if value in (None, "", [], {}):
                continue
            if key == "style_tags":
                values[key] = self._merge_unique_list(values.get(key, []), value)
            else:
                values[key] = value

        if include_web:
            values["web_search_enabled"] = True
        elif "web_search_enabled" not in values:
            values["web_search_enabled"] = False

        values["rewritten_query"] = decomposed.rewritten_query or message
        values["retrieval_tags"] = self._merge_unique_list(
            values.get("retrieval_tags", []),
            list(decomposed.tags or []),
        )

        instance.values = values
        instance.turn_history = [*instance.turn_history[-9:], message]
        instance.missing_required_fields = self.missing_required_fields(definition, values)
        instance.status = "awaiting_input" if instance.missing_required_fields else "active"
        instance.last_query_record = WorksheetQueryRecord(
            query_text=message,
            rewritten_query=decomposed.rewritten_query or message,
            filters=dict(decomposed.filters or {}),
            tags=list(decomposed.tags or []),
            source="hybrid" if include_web else "local",
            debug_metadata={"worksheet": definition.name},
        )
        instance.last_updated_at = time.time()
        return instance

    def build_query_from_instance(
        self,
        instance: WorksheetInstance,
        fallback_message: str,
    ) -> DecomposedQuery:
        values = instance.values
        filters: dict[str, Any] = {}
        tags = list(values.get("style_tags", []))

        product_type = values.get("product_type")
        if product_type:
            filters["category"] = product_type
            if product_type not in tags:
                tags.append(product_type)

        for key in ["price_min", "price_max", "brand", "color", "size"]:
            value = values.get(key)
            if value not in (None, "", []):
                filters[key] = value

        rewritten_query = self._compose_rewritten_query(values, fallback_message)
        return DecomposedQuery(
            intent=IntentType.TEXT_SEARCH,
            original_query=fallback_message,
            tags=tags,
            filters=filters,
            rewritten_query=rewritten_query,
        )

    def missing_required_fields(
        self,
        definition: WorksheetDefinition,
        values: dict[str, Any],
    ) -> list[str]:
        missing = []
        for field in definition.fields:
            if not field.required:
                continue
            if not field.blocks_progress:
                continue
            if not self.is_field_active(field, values):
                continue
            if values.get(field.name) in (None, "", []):
                missing.append(field.name)
        return missing

    def build_clarification_question(
        self,
        definition: WorksheetDefinition,
        instance: WorksheetInstance,
    ) -> str:
        for field_name in instance.missing_required_fields:
            field = definition.field_map().get(field_name)
            if not field or field.dont_ask:
                continue
            if field.question:
                return field.question
            return f"Could you tell me the {field.name.replace('_', ' ')} you want?"
        return "Could you share a bit more about what you want so I can narrow this down?"

    def build_event_payload(
        self,
        definition: WorksheetDefinition,
        instance: WorksheetInstance,
    ) -> dict[str, Any]:
        visible_values: dict[str, Any] = {}
        for field in definition.fields:
            value = instance.values.get(field.name)
            if value in (None, "", [], {}):
                continue
            if field.kind == "output" and field.name not in {"rewritten_query"}:
                continue
            visible_values[field.name] = value

        result_counts = {
            key: len(value) if isinstance(value, list) else value
            for key, value in instance.result_refs.items()
            if key.endswith("_product_ids") or key.endswith("_count")
        }

        return {
            "type": "worksheet_state",
            "worksheet": {
                "name": definition.name,
                "status": instance.status,
                "values": visible_values,
                "missing_required_fields": list(instance.missing_required_fields),
                "result_counts": result_counts,
                "last_updated_at": instance.last_updated_at,
            },
        }

    def update_result_refs(
        self,
        instance: WorksheetInstance,
        *,
        local_products: list[dict],
        web_products: list[dict],
        reranked_products: list[dict],
        query: DecomposedQuery,
        include_web: bool,
    ) -> WorksheetInstance:
        instance.result_refs = {
            "local_product_ids": self._product_ids(local_products),
            "web_product_ids": self._product_ids(web_products),
            "reranked_product_ids": self._product_ids(reranked_products),
            "local_count": len(local_products),
            "web_count": len(web_products),
            "reranked_count": len(reranked_products),
            "last_products": list(reranked_products),
        }
        instance.status = "active"
        instance.last_query_record = WorksheetQueryRecord(
            query_text=query.original_query,
            rewritten_query=query.rewritten_query,
            filters=dict(query.filters or {}),
            tags=list(query.tags or []),
            source="hybrid" if include_web else "local",
            result_product_ids=self._product_ids(reranked_products),
            debug_metadata={"worksheet": instance.worksheet_name},
        )
        instance.touch()
        return instance

    def create_compare_instance(
        self,
        definition: WorksheetDefinition,
        source_instance: WorksheetInstance | None,
        message: str,
    ) -> WorksheetInstance:
        source_products = self._comparison_pool(source_instance)
        selected_indices = self._extract_compare_indices(message, len(source_products))
        selected_products = [
            source_products[index]
            for index in selected_indices
            if 0 <= index < len(source_products)
        ]

        dimensions = self._extract_comparison_dimensions(message)
        selected_ids = self._product_ids(selected_products)
        values = {
            "selected_product_ids": selected_ids if len(selected_products) >= 2 else [],
            "comparison_dimensions": dimensions,
            "source_label": self._comparison_source_label(source_instance),
        }

        instance = WorksheetInstance(
            worksheet_name=definition.name,
            status="active" if len(selected_products) >= 2 else "awaiting_input",
            values=values,
            turn_history=[message],
            result_refs={
                "comparison_products": selected_products,
                "comparison_count": len(selected_products),
                "source_products": source_products,
                "source_count": len(source_products),
            },
        )
        instance.missing_required_fields = [] if len(selected_products) >= 2 else ["selected_product_ids"]
        instance.last_query_record = WorksheetQueryRecord(
            query_text=message,
            rewritten_query=message,
            source="compare",
            result_product_ids=selected_ids,
            debug_metadata={"worksheet": definition.name},
        )
        instance.touch()
        return instance

    def build_compare_clarification_question(self, instance: WorksheetInstance) -> str:
        source_count = int(instance.result_refs.get("source_count") or 0)
        if source_count < 2:
            return "I need at least two recent results before I can compare them. Ask for some products first, then say something like compare the first two."
        return "Tell me which results you want compared, like compare 1 and 2 or compare the first two."

    @staticmethod
    def is_compare_message(message: str) -> bool:
        return bool(_COMPARE_PATTERNS.search(message or ""))

    def is_field_active(
        self,
        field: WorksheetFieldDefinition,
        values: dict[str, Any],
    ) -> bool:
        if not field.predicate:
            return True
        try:
            return bool(eval(field.predicate, {"__builtins__": {}}, {"values": values}))
        except Exception:
            return True

    def _extract_updates(
        self,
        message: str,
        decomposed: DecomposedQuery,
    ) -> dict[str, Any]:
        filters = dict(decomposed.filters or {})
        text = " ".join(
            [
                message or "",
                decomposed.rewritten_query or "",
                " ".join(decomposed.tags or []),
                str(filters.get("category", "") or ""),
            ]
        )

        product_type = filters.get("category") or self._infer_product_type(text)

        updates: dict[str, Any] = {
            "product_type": _singularize(str(product_type)) if product_type else None,
            "style_tags": list(decomposed.tags or []),
            "price_min": filters.get("price_min"),
            "price_max": filters.get("price_max"),
            "brand": filters.get("brand"),
            "color": filters.get("color"),
            "size": filters.get("size"),
        }
        return updates

    def _infer_product_type(self, text: str) -> str | None:
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        for keyword in _PRODUCT_TYPE_KEYWORDS:
            if keyword in tokens:
                return _singularize(keyword)
        return None

    def _compose_rewritten_query(self, values: dict[str, Any], fallback: str) -> str:
        parts: list[str] = []

        brand = values.get("brand")
        color = values.get("color")
        size = values.get("size")
        style_tags = values.get("style_tags", [])
        product_type = values.get("product_type")
        use_case = values.get("use_case")
        price_min = values.get("price_min")
        price_max = values.get("price_max")

        for value in [brand, color]:
            if value:
                parts.append(str(value))

        if isinstance(style_tags, list):
            parts.extend([str(tag) for tag in style_tags if str(tag).strip()])

        if product_type:
            parts.append(str(product_type))

        if size and _singularize(str(product_type)) in _APPAREL_PRODUCT_TYPES:
            parts.append(f"size {size}")

        if use_case:
            parts.append(f"for {use_case}")

        if price_max not in (None, ""):
            parts.append(f"under ${price_max}")
        if price_min not in (None, ""):
            parts.append(f"over ${price_min}")

        rewritten = " ".join(self._merge_unique_list([], parts)).strip()
        return rewritten or fallback

    def _comparison_pool(self, instance: WorksheetInstance | None) -> list[dict]:
        if not instance:
            return []
        if instance.worksheet_name == "compare_products":
            return list(instance.result_refs.get("source_products", []))
        return list(instance.result_refs.get("last_products", []))

    def _comparison_source_label(self, instance: WorksheetInstance | None) -> str:
        if not instance or not instance.last_query_record:
            return "recent results"
        if instance.last_query_record.source == "hybrid":
            return "recent catalog + web results"
        if instance.last_query_record.source == "local":
            return "recent catalog results"
        return "recent results"

    def _extract_compare_indices(self, message: str, pool_size: int) -> list[int]:
        if pool_size < 2:
            return []

        lowered = (message or "").lower()

        if any(token in lowered for token in ["first two", "top two", "both", "these", "them"]):
            return [0, 1]

        indices: list[int] = []
        for match in re.findall(r"\b([1-9])\b", lowered):
            idx = int(match) - 1
            if 0 <= idx < pool_size:
                indices.append(idx)

        for word, idx in _ORDINAL_MAP.items():
            if re.search(rf"\b{word}\b", lowered) and 0 <= idx < pool_size:
                indices.append(idx)

        deduped: list[int] = []
        seen: set[int] = set()
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            deduped.append(idx)

        if len(deduped) >= 2:
            return deduped[:4]

        return [0, 1]

    def _extract_comparison_dimensions(self, message: str) -> list[str]:
        lowered = (message or "").lower()
        dimensions: list[str] = []
        mapping = {
            "price": ["price", "budget", "cheaper", "value"],
            "quality": ["quality", "durable", "better made"],
            "features": ["feature", "features", "material", "spec"],
            "style": ["style", "look", "design"],
            "reviews": ["review", "reviews", "rating", "ratings"],
        }
        for name, signals in mapping.items():
            if any(signal in lowered for signal in signals):
                dimensions.append(name)
        return dimensions or ["price", "quality", "features", "value_for_money"]

    @staticmethod
    def _merge_unique_list(existing: list[Any], incoming: list[Any]) -> list[Any]:
        merged: list[Any] = []
        seen: set[str] = set()
        for item in [*(existing or []), *(incoming or [])]:
            normalized = str(item).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    @staticmethod
    def _product_ids(products: list[dict]) -> list[str]:
        result = []
        for product in products:
            product_id = product.get("product_id") or product.get("url")
            if product_id:
                result.append(str(product_id))
        return result
