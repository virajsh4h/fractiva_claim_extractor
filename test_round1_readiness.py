"""Round 1 readiness tests: direction semantics, fail-closed handling and input shapes.

Rewritten for the audited architecture:

1. Positive arm values cannot prove a direction; signed values can.
2. A numeric claim fails closed to ``unclear``, never to an absent figure.
3. Input shape support (list, ``{"claims": [...]}``, single object) is tested
   with a scripted extractor so no test touches the network.
"""

import json
import os
import tempfile
import unittest

from domain import CheckStatus, Verdict
from extractor import ArithmeticExtraction, ClaimExtractionResponse
from verification_engine import ArithmeticSpec, verify_arithmetic
from verify import ClaimVerifier, make_fail_closed_prediction, verify_claims_pipeline


def fail_closed_extraction():
    return ClaimExtractionResponse(
        population_check="unclear",
        drug_indication_check="unclear",
        actually_stated_check="unclear",
        mismatch_bases=[],
        evidence_source=None,
        evidence_page=None,
        evidence_span=None,
        arithmetic=ArithmeticExtraction(operation="none"),
        supporting_values=[],
        page_transcripts=[],
    )


def scripted_verifier(log_path):
    return ClaimVerifier(
        log_path=log_path,
        extractor=lambda **_: fail_closed_extraction(),
        transcriber=lambda *args, **kwargs: [],
    )


class TestTwoOperandDifferenceDirection(unittest.TestCase):
    def test_unsigned_positive_operands_cannot_settle_a_conflict(self):
        spec = ArithmeticSpec(
            operation="difference",
            claimed_numeric=2.0,
            source_operands=[7.2, 5.2],
            direction="decrease",
        )
        status, info = verify_arithmetic(spec)
        self.assertEqual(status, CheckStatus.UNCLEAR.value, info)
        self.assertIn("unsigned", info["result"])

    def test_unsigned_positive_operands_matching_the_direction_still_match(self):
        spec = ArithmeticSpec(
            operation="difference",
            claimed_numeric=2.0,
            source_operands=[5.2, 7.2],
            direction="decrease",
        )
        status, info = verify_arithmetic(spec)
        self.assertEqual(status, CheckStatus.MATCH.value, info)

    def test_unsigned_positive_operands_contradicting_claim_are_unclear(self):
        spec = ArithmeticSpec(
            operation="difference",
            claimed_numeric=2.0,
            source_operands=[8.5, 6.5],
            direction="decrease",
        )
        status, info = verify_arithmetic(spec)
        self.assertEqual(status, CheckStatus.UNCLEAR.value, info)
        self.assertIn("unsigned", info["result"])

    def test_signed_operands_certify_the_direction(self):
        match = ArithmeticSpec(
            operation="difference",
            claimed_numeric=2.0,
            source_operands=[-7.2, -5.2],
            direction="decrease",
        )
        status, info = verify_arithmetic(match)
        self.assertEqual(status, CheckStatus.MATCH.value, info)

        mismatch = ArithmeticSpec(
            operation="difference",
            claimed_numeric=2.0,
            source_operands=[-6.5, -8.5],
            direction="decrease",
        )
        status, info = verify_arithmetic(mismatch)
        self.assertEqual(status, CheckStatus.MISMATCH.value, info)
        self.assertIn("increase", info["result"])


class TestFailClosedNumericHandling(unittest.TestCase):
    def test_numeric_claim_fails_closed_to_unclear(self):
        pred = make_fail_closed_prediction(
            claim_id="C_TEST",
            reason="API key missing",
            claim_text="With SOLIQUA, ~2x greater HbA1c reduction vs basal insulin",
        )
        self.assertEqual(pred["verdict"], Verdict.CANNOT_VERIFY.value)
        self.assertEqual(pred["checks"]["figure"], CheckStatus.UNCLEAR.value)

    def test_non_numeric_claim_fails_closed_to_none(self):
        pred = make_fail_closed_prediction(
            claim_id="C_TEST",
            reason="API key missing",
            claim_text="Indicated for adults with type 2 diabetes",
        )
        self.assertEqual(pred["verdict"], Verdict.CANNOT_VERIFY.value)
        self.assertIsNone(pred["checks"]["figure"])


class TestNewClaimFormats(unittest.TestCase):
    def _run(self, payload):
        with tempfile.TemporaryDirectory() as td:
            claims_path = os.path.join(td, "claims.json")
            output_path = os.path.join(td, "predictions.json")
            log_path = os.path.join(td, "llm_log.jsonl")
            with open(claims_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            preds = verify_claims_pipeline(
                claims_path,
                output_path,
                verifier=scripted_verifier(log_path),
            )
            self.assertTrue(os.path.isfile(output_path))
            return preds

    def test_pipeline_accepts_dict_with_claims_key(self):
        preds = self._run(
            {
                "claims": [
                    {
                        "claim_id": "C_NEW_1",
                        "text": "Indicated for adults with type 2 diabetes",
                        "cited_paper": "SOLIQUA_LBL.pdf",
                    }
                ]
            }
        )
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["claim_id"], "C_NEW_1")
        self.assertEqual(preds[0]["verdict"], Verdict.CANNOT_VERIFY.value)

    def test_pipeline_accepts_single_claim_dict(self):
        preds = self._run(
            {
                "claim_id": "C_SOLO",
                "text": "Trial reported superiority",
                "cited_paper": "papers/SoliMix_Rosenstock2021.pdf",
            }
        )
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["claim_id"], "C_SOLO")
        self.assertEqual(preds[0]["verdict"], Verdict.CANNOT_VERIFY.value)


if __name__ == "__main__":
    unittest.main()
