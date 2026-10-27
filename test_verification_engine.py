import unittest

from verification_engine import (
    ArithmeticSpec,
    check_span_presence,
    confidence_levels,
    normalize_text,
    numeric_qualifiers_match,
    page_evidence_text,
    resolve_verdict,
    validate_evidence_span,
    verify_arithmetic,
)


class TestVerificationEngine(unittest.TestCase):
    def test_normalize_text(self):
        raw = "Superior  HbA1c\nreduction (1.3 ± 0.1%) vs   BIAsp 30 (1.1 ± 0.1%); 95% CI −0.4, –0.1"
        normalized = normalize_text(raw)
        self.assertNotIn("\n", normalized)
        self.assertNotIn("  ", normalized)
        self.assertIn("+/-", normalized)
        self.assertIn("-0.4", normalized)
        self.assertIn("-0.1", normalized)

    def test_check_span_presence_exact_and_ligatures(self):
        page_text = (
            "Primary results: superior HbA1c reduction with once-daily iGlarLixi "
            "(1.3 ± 0.1%) vs twice-daily premix insulin BIAsp 30 (1.1 ± 0.1%); "
            "95% CI −0.4, −0.1; p<0.001."
        )
        span1 = (
            "iGlarLixi (1.3 +/- 0.1%) vs twice-daily premix insulin BIAsp 30 "
            "(1.1 +/- 0.1%); 95% CI -0.4, -0.1"
        )
        self.assertTrue(check_span_presence(span1, page_text))
        self.assertFalse(
            check_span_presence(
                "Completely fabricated text that does not exist in the source",
                page_text,
            )
        )

    def test_token_bag_is_not_a_verbatim_quote(self):
        page_text = "Table 2: iGlarLixi (n=468) Baseline 8.1 Week 30 6.5 Change -1.6"
        span = "iGlarLixi Baseline 8.1 Change -1.6 Week 30 6.5"
        self.assertFalse(check_span_presence(span, page_text))

    def test_ellipsis_is_refused(self):
        page_text = "The trial reported a reduction of 1.3 percent and a difference of 0.2."
        self.assertFalse(
            check_span_presence("reduction of 1.3 percent ... difference of 0.2", page_text)
        )
        self.assertTrue(
            check_span_presence("reduction of 1.3 percent and a difference of 0.2", page_text)
        )

    def test_validate_evidence_span(self):
        pages_data = {
            1: {"page_number": 1, "text": "Clinical findings show HbA1c reduction", "has_native_text": True},
            2: {"page_number": 2, "text": "", "has_native_text": False},
        }
        self.assertTrue(validate_evidence_span("HbA1c reduction", 1, pages_data))
        self.assertFalse(validate_evidence_span("Nonexistent sentence", 1, pages_data))
        self.assertFalse(validate_evidence_span("HbA1c reduction", 99, pages_data))
        self.assertFalse(validate_evidence_span("Any span", 2, pages_data))

    def test_table_text_is_valid_evidence(self):
        pages_data = {
            6: {
                "page_number": 6,
                "text": "Figure 2 caption",
                "tables": [
                    "|Participants who reached HbA1c <7 %|31.8 (n=141)<br>"
                    "42.2 (n=187)|1.65 (1.25, 2.19)|||"
                ],
                "has_native_text": True,
            }
        }
        span = (
            "Participants who reached HbA1c <7 %|31.8 (n=141)<br>"
            "42.2 (n=187)|1.65 (1.25, 2.19)|||"
        )
        self.assertTrue(validate_evidence_span(span, 6, pages_data))
        self.assertIn("31.8 (n=141)", page_evidence_text(pages_data[6]))

    def test_frozen_transcript_excludes_unfrozen_table_text(self):
        page = {
            "text": "",
            "tables": ["|stale table value|99|"],
            "transcript": "Fresh transcript value.",
        }
        self.assertEqual(page_evidence_text(page), "Fresh transcript value.")
        self.assertFalse(check_span_presence("stale table value", page_evidence_text(page)))

    def test_verify_arithmetic_ratio(self):
        spec_mismatch = ArithmeticSpec(
            operation="ratio",
            claimed_text="~2x",
            claimed_numeric=2.0,
            source_operands=[1.63, 1.34],
        )
        status, comp = verify_arithmetic(spec_mismatch)
        self.assertEqual(status, "mismatch")
        self.assertEqual(comp["result"], "1.22x")

        spec_match = ArithmeticSpec(
            operation="ratio",
            claimed_text="~2x",
            claimed_numeric=2.0,
            source_operands=[1.63, 0.85],
        )
        status, comp = verify_arithmetic(spec_match)
        self.assertEqual(status, "match")
        self.assertEqual(comp["result"], "1.92x")

    def test_verify_arithmetic_difference(self):
        spec = ArithmeticSpec(
            operation="difference",
            claimed_text="-0.2%",
            claimed_numeric=0.2,
            source_operands=[1.3, 1.1],
        )
        status, comp = verify_arithmetic(spec)
        self.assertEqual(status, "match")
        self.assertEqual(comp["result"], "0.2")

    def test_reported_signed_difference_controls_direction(self):
        spec = ArithmeticSpec(
            operation="difference",
            claimed_text="decrease of 0.2%",
            claimed_numeric=0.2,
            source_operands=[-1.3, -1.1],
            reported_value=-0.2,
            direction="decrease",
        )
        status, comp = verify_arithmetic(spec)
        self.assertEqual(status, "match", comp)
        self.assertEqual(comp["computed_value"], -0.2)

        reversed_spec = ArithmeticSpec(
            operation="difference",
            claimed_text="decrease of 0.2%",
            claimed_numeric=0.2,
            source_operands=[-1.3, -1.1],
            reported_value=0.2,
            direction="decrease",
        )
        status, comp = verify_arithmetic(reversed_spec)
        self.assertEqual(status, "mismatch", comp)

    def test_unsigned_magnitudes_cannot_settle_direction(self):
        spec = ArithmeticSpec(
            operation="difference",
            claimed_numeric=2.0,
            source_operands=[8.5, 6.5],
            direction="decrease",
        )
        status, comp = verify_arithmetic(spec)
        self.assertEqual(status, "unclear", comp)

        signed = ArithmeticSpec(
            operation="difference",
            claimed_numeric=2.0,
            source_operands=[-6.5, -8.5],
            direction="decrease",
        )
        status, comp = verify_arithmetic(signed)
        self.assertEqual(status, "mismatch", comp)

    def test_verify_arithmetic_percentage(self):
        spec = ArithmeticSpec(
            operation="percentage",
            claimed_text="42.20%",
            claimed_numeric=42.2,
            source_operands=[187.0, 443.0],
        )
        status, comp = verify_arithmetic(spec)
        self.assertEqual(status, "match")
        self.assertEqual(comp["result"], "42.21%")

    def test_missing_denominator_is_unclear(self):
        spec = ArithmeticSpec(
            operation="percentage",
            claimed_text="42.20%",
            claimed_numeric=42.2,
            source_operands=[187.0],
        )
        status, comp = verify_arithmetic(spec)
        self.assertEqual(status, "unclear", comp)

    def test_verify_arithmetic_odds_ratio(self):
        spec_crude = ArithmeticSpec(
            operation="odds_ratio",
            claimed_text="0.45",
            claimed_numeric=0.45,
            source_operands=[28.0, 442.0, 57.0, 441.0],
        )
        status, comp = verify_arithmetic(spec_crude)
        self.assertEqual(status, "match")
        self.assertIn("0.46", comp["result"])

        spec_reported = ArithmeticSpec(
            operation="odds_ratio",
            claimed_text="1.65",
            claimed_numeric=1.65,
            source_operands=[187, 443, 141, 444],
            reported_value=1.65,
        )
        status, comp = verify_arithmetic(spec_reported)
        self.assertEqual(status, "match")
        self.assertIn("1.65", comp["result"])

        spec_wrong_claim = ArithmeticSpec(
            operation="odds_ratio",
            claimed_text="1.40",
            claimed_numeric=1.40,
            source_operands=[187, 443, 141, 444],
            reported_value=1.65,
        )
        status, comp = verify_arithmetic(spec_wrong_claim)
        self.assertEqual(status, "mismatch", comp)

    def test_resolve_verdict_matrix(self):
        checks_all_match = {
            "figure": "match",
            "population": "match",
            "drug_indication": "match",
            "actually_stated": "match",
        }
        self.assertEqual(resolve_verdict(checks_all_match, span_verified=True), "supported")
        self.assertEqual(resolve_verdict(checks_all_match, span_verified=False), "cannot_verify")

        checks_mismatch = {
            "figure": "mismatch",
            "population": "match",
            "drug_indication": "match",
            "actually_stated": "mismatch",
        }
        self.assertEqual(resolve_verdict(checks_mismatch, span_verified=True), "overstated")

        checks_unclear = {
            "figure": "unclear",
            "population": "match",
            "drug_indication": "match",
            "actually_stated": "match",
        }
        self.assertEqual(resolve_verdict(checks_unclear, span_verified=True), "cannot_verify")

        checks_mixed = {
            "figure": "unclear",
            "population": "match",
            "drug_indication": "match",
            "actually_stated": "mismatch",
        }
        self.assertEqual(resolve_verdict(checks_mixed, span_verified=True), "overstated")


    def test_confidence_levels_and_p_value_qualifiers(self):
        self.assertEqual(confidence_levels("95% CI -0.4, -0.1"), {95.0})
        self.assertEqual(
            confidence_levels("(97.5% confidence interval)"),
            {97.5},
        )
        self.assertTrue(
            numeric_qualifiers_match(
                "superiority (95% CI); p<0.001",
                "Superiority: (95% CI); p-value -0.2; <0.001",
            )
        )
        self.assertFalse(
            numeric_qualifiers_match(
                "superiority (95% CI); p<0.001",
                "Noninferiority: (97.5% CI); p-value -0.2; <0.001",
            )
        )
        self.assertFalse(
            numeric_qualifiers_match(
                "p<0.0001",
                "p = 0.03",
            )
        )


if __name__ == "__main__":
    unittest.main()
