import unittest

from src.schema import DecomposedQuery, IntentType

from mvp.worksheet_engine import WorksheetEngine
from mvp.worksheet_registry import WorksheetRegistry
from mvp.worksheet_store import InMemoryWorksheetStore


class MVPWorksheetTests(unittest.TestCase):
    def setUp(self):
        self.registry = WorksheetRegistry()
        self.definition = self.registry.get("product_search")
        self.engine = WorksheetEngine()

    def test_registry_loads_product_search_definition(self):
        self.assertEqual(self.definition.name, "product_search")
        self.assertIn("text_search", self.definition.trigger_intents)

    def test_missing_required_product_type_triggers_clarification(self):
        instance = self.engine.create_instance(self.definition)
        decomposed = DecomposedQuery(
            intent=IntentType.TEXT_SEARCH,
            original_query="show me something cheaper",
            tags=["cheaper"],
            filters={"price_max": 100},
            rewritten_query="something cheaper under 100",
        )

        updated = self.engine.apply_product_search_turn(
            self.definition,
            instance,
            "show me something cheaper",
            decomposed,
            include_web=False,
        )

        self.assertEqual(updated.status, "awaiting_input")
        self.assertEqual(updated.missing_required_fields, ["product_type"])
        self.assertEqual(
            self.engine.build_clarification_question(self.definition, updated),
            "What kind of product are you shopping for right now?",
        )

    def test_follow_up_turn_reuses_existing_product_type(self):
        instance = self.engine.create_instance(self.definition)
        first = self.engine.apply_product_search_turn(
            self.definition,
            instance,
            "show me black jackets",
            DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query="show me black jackets",
                tags=["black"],
                filters={"category": "jackets", "color": "black"},
                rewritten_query="black jackets",
            ),
            include_web=False,
        )

        second = self.engine.apply_product_search_turn(
            self.definition,
            first,
            "under 120",
            DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query="under 120",
                tags=[],
                filters={"price_max": 120},
                rewritten_query="under 120",
            ),
            include_web=False,
        )

        combined = self.engine.build_query_from_instance(second, "under 120")

        self.assertEqual(second.values["product_type"], "jacket")
        self.assertEqual(second.values["price_max"], 120)
        self.assertEqual(combined.filters["category"], "jacket")
        self.assertEqual(combined.filters["price_max"], 120)
        self.assertIn("jacket", combined.rewritten_query.lower())

    def test_in_memory_store_roundtrip(self):
        store = InMemoryWorksheetStore()
        instance = self.engine.create_instance(self.definition)
        instance.values["product_type"] = "jacket"
        store.save("session-1", instance)

        loaded = store.get("session-1")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.values["product_type"], "jacket")

    def test_compare_instance_selects_requested_products_from_previous_search(self):
        product_search = self.engine.create_instance(self.definition)
        product_search.result_refs["last_products"] = [
            {"product_id": "p1", "title": "First Jacket"},
            {"product_id": "p2", "title": "Second Jacket"},
            {"product_id": "p3", "title": "Third Jacket"},
        ]
        product_search.last_query_record = None

        compare_definition = self.registry.get("compare_products")
        compare_instance = self.engine.create_compare_instance(
            compare_definition,
            product_search,
            "compare 1 and 3 on price and quality",
        )

        self.assertEqual(compare_instance.worksheet_name, "compare_products")
        self.assertEqual(compare_instance.values["selected_product_ids"], ["p1", "p3"])
        self.assertIn("price", compare_instance.values["comparison_dimensions"])
        self.assertIn("quality", compare_instance.values["comparison_dimensions"])
        self.assertEqual(compare_instance.result_refs["comparison_count"], 2)


if __name__ == "__main__":
    unittest.main()
