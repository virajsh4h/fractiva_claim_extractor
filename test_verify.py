import json
import os
import tempfile
import unittest

from extractor import ArithmeticExtraction, ClaimExtractionResponse
from verify import ClaimVerifier, verify_claims_pipeline, verify_single_claim


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


def scripted_verifier(base_dir="."):
    return ClaimVerifier(
        base_dir=base_dir,
        extractor=lambda **_: fail_closed_extraction(),
        transcriber=lambda *args, **kwargs: [],
    )


class TestVerifyPipeline(unittest.TestCase):
    def test_verify_single_claim_structure(self):
        mock_claim = {
            "claim_id": "TEST_C1",
            "text": "Superior HbA1c reduction with once-daily iGlarLixi (1.3 +/- 0.1%) vs twice-daily premix insulin BIAsp 30 (1.1 +/- 0.1%); 95% CI -0.4, -0.1; p<0.001",
            "population": "T2DM suboptimally controlled on basal insulin",
            "molecule": "iGlarLixi vs BIAsp 30",
            "cited_paper": "papers/SoliMix_Rosenstock2021.pdf",
        }
        pred = scripted_verifier().verify(mock_claim)

        self.assertEqual(pred["claim_id"], "TEST_C1")
        self.assertIn(pred["verdict"], ["supported", "overstated", "cannot_verify"])
        self.assertEqual(
            set(pred),
            {
                "claim_id",
                "verdict",
                "checks",
                "evidence_page",
                "evidence_span",
                "computed",
                "reason",
            },
        )
        self.assertEqual(
            set(pred["checks"]),
            {"figure", "population", "drug_indication", "actually_stated"},
        )
        self.assertIn("reason", pred)

    def test_missing_paper_fails_closed(self):
        missing_claim = {
            "claim_id": "TEST_MISSING",
            "text": "Some unsubstantiated assertion",
            "population": "T2DM",
            "molecule": "iGlarLixi",
            "cited_paper": "papers/non_existent_paper_123.pdf",
        }
        pred = verify_single_claim(missing_claim, base_dir=".")
        self.assertEqual(pred["verdict"], "cannot_verify")
        self.assertEqual(pred["checks"]["actually_stated"], "unclear")
        self.assertIn("not found", pred["reason"].lower())

    def test_full_pipeline_file_generation(self):
        with tempfile.TemporaryDirectory() as td:
            in_claims = os.path.join(td, "claims.json")
            out_preds = os.path.join(td, "predictions.json")

            with open(in_claims, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "claim_id": "C_SYNTH",
                            "text": "Sample claim text",
                            "population": "Sample",
                            "molecule": "Sample",
                            "cited_paper": "papers/non_existent.pdf",
                        }
                    ],
                    f,
                )

            predictions = verify_claims_pipeline(
                in_claims,
                out_preds,
                base_dir=".",
                verifier=scripted_verifier(),
            )

            self.assertTrue(os.path.exists(out_preds))
            with open(out_preds, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["claim_id"], "C_SYNTH")
            self.assertEqual(saved[0]["verdict"], "cannot_verify")


if __name__ == "__main__":
    unittest.main()
