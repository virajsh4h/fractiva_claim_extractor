import unittest
import json
import os
import tempfile
from eval import evaluate_predictions, format_confusion_matrix

class TestEval(unittest.TestCase):
    def setUp(self):
        self.gold_data = [
            {"claim_id": "C1", "verdict": "overstated"},
            {"claim_id": "C2", "verdict": "supported"},
            {"claim_id": "C3", "verdict": "overstated"},
            {"claim_id": "C4", "verdict": "supported"},
            {"claim_id": "C5", "verdict": "overstated"},
            {"claim_id": "C6", "verdict": "supported"},
            {"claim_id": "C7", "verdict": "overstated"},
        ]

    def test_perfect_predictions(self):
        predictions = [
            {"claim_id": "C1", "verdict": "overstated"},
            {"claim_id": "C2", "verdict": "supported"},
            {"claim_id": "C3", "verdict": "overstated"},
            {"claim_id": "C4", "verdict": "supported"},
            {"claim_id": "C5", "verdict": "overstated"},
            {"claim_id": "C6", "verdict": "supported"},
            {"claim_id": "C7", "verdict": "overstated"},
        ]
        results = evaluate_predictions(predictions, self.gold_data)
        self.assertEqual(results["total"], 7)
        self.assertEqual(results["correct"], 7)
        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["supported_precision"], 1.0)
        self.assertEqual(results["supported_recall"], 1.0)
        self.assertEqual(results["false_supported_count"], 0)
        self.assertEqual(results["false_supported_claims"], [])

    def test_false_supported_detected(self):
        predictions = [
            {"claim_id": "C1", "verdict": "supported"},  # False supported!
            {"claim_id": "C2", "verdict": "supported"},
            {"claim_id": "C3", "verdict": "supported"},  # False supported!
            {"claim_id": "C4", "verdict": "supported"},
            {"claim_id": "C5", "verdict": "overstated"},
            {"claim_id": "C6", "verdict": "supported"},
            {"claim_id": "C7", "verdict": "cannot_verify"},  # Mismatch, but not false supported
        ]
        results = evaluate_predictions(predictions, self.gold_data)
        self.assertEqual(results["total"], 7)
        self.assertEqual(results["correct"], 4)
        self.assertEqual(results["false_supported_count"], 2)
        self.assertEqual(results["false_supported_claims"], ["C1", "C3"])
        self.assertEqual(results["false_overstated_claims"], [])
        self.assertEqual(results["confusion_matrix"]["overstated"]["supported"], 2)
        self.assertEqual(results["confusion_matrix"]["overstated"]["cannot_verify"], 1)
        self.assertEqual(results["trivial_baseline"]["name"], "always_overstated")
        self.assertAlmostEqual(results["trivial_baseline"]["accuracy"], 4 / 7)

    def test_confusion_matrix_structure(self):
        predictions = [
            {"claim_id": "C1", "verdict": "overstated"},
            {"claim_id": "C2", "verdict": "supported"},
        ]
        gold = [
            {"claim_id": "C1", "verdict": "overstated"},
            {"claim_id": "C2", "verdict": "supported"},
        ]
        results = evaluate_predictions(predictions, gold)
        cm = results["confusion_matrix"]
        self.assertIn("supported", cm)
        self.assertIn("overstated", cm)
        self.assertIn("cannot_verify", cm)
        self.assertEqual(cm["supported"]["supported"], 1)
        self.assertEqual(cm["overstated"]["overstated"], 1)

if __name__ == "__main__":
    unittest.main()
