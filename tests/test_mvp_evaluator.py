import json
import unittest
from pathlib import Path
from uuid import uuid4

from mvp.evaluation.evaluator import (
    evaluate_record_with_python_rubric,
    generate_evaluation_report,
    load_conversation_records,
    write_evaluation_report,
)


class MVPEvaluatorTests(unittest.TestCase):
    def setUp(self):
        runtime_dir = Path("data/test_runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = runtime_dir / f"eval_records_{uuid4().hex}.jsonl"
        self.report_path = runtime_dir / f"eval_report_{uuid4().hex}.json"

    def tearDown(self):
        for path in [self.records_path, self.report_path]:
            if path.exists():
                path.unlink()

    def test_python_rubric_scores_recommendation_signals(self):
        record = {
            "type": "mvp",
            "user_id": "caleb",
            "session_id": "session-1",
            "conversation": [
                {"user": "recommend me some black hoodies under 80", "timestamp": 1.0},
                {
                    "agent": (
                        "I found some strong options for you.\n\n"
                        "1. Black Pullover Hoodie\n"
                        "- Price: $60\n"
                        "- Description: Soft cotton blend with a clean everyday look.\n"
                        "- Trade-off: Warmer but slightly heavier.\n\n"
                        "Would you like more options or a comparison?"
                    ),
                    "timestamp": 2.0,
                },
            ],
            "inferred_preferences": {"color": ["black"], "style": ["simple"]},
        }

        result = evaluate_record_with_python_rubric(record)

        self.assertEqual(result["mode"], "python")
        self.assertGreaterEqual(result["sections"]["catalog_detail_completeness"]["score"], 3.0)
        self.assertGreaterEqual(result["sections"]["recommendation_quality"]["score"], 4.0)
        self.assertGreaterEqual(result["sections"]["closing_helpfulness"]["score"], 3.0)

    def test_report_generation_and_write_roundtrip(self):
        record = {
            "type": "mvp",
            "user_id": "caleb",
            "session_id": "session-2",
            "conversation": [
                {"user": "compare these two jackets", "timestamp": 1.0},
                {"agent": "The first is cheaper, while the second has better features. Want me to compare more?", "timestamp": 2.0},
            ],
            "inferred_preferences": {},
        }
        self.records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        records = load_conversation_records(self.records_path)
        report = generate_evaluation_report(records, mode="python", input_source=str(self.records_path))
        written = write_evaluation_report(report, self.report_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(report["mode"], "python")
        self.assertIn("python_critic", report["conversations"][0])
        payload = json.loads(written.read_text(encoding="utf-8"))
        self.assertEqual(payload["num_conversations"], 1)
        self.assertIn("component_performance", payload["python_critic"])


if __name__ == "__main__":
    unittest.main()
