import json
import os
import tempfile
import unittest

from audit import append_llm_log, prompt_sha256
from eval import evaluate_predictions, summarize_audit_log
from extractor import (
    ArithmeticExtraction,
    ArithmeticExtractionComponent,
    ClaimExtractionResponse,
    SYSTEM_INSTRUCTION,
    _load_cached_transcript,
    _store_cached_transcript,
    _transcript_cache_path,
    build_prompt,
    parse_structured_json,
)
from ingest import extract_pdf_pages, resolve_source_refs
from verification_engine import (
    ArithmeticComponent,
    ArithmeticSpec,
    build_reason,
    check_span_presence,
    claim_direction,
    claim_has_quantitative_assertion,
    comparison_anchor_numbers,
    deterministic_figure_reason,
    figure_coverage_gaps,
    find_limitation_evidence,
    limitation_hits,
    resolve_verdict,
    span_supports_arithmetic,
    supported_reason,
    verify_arithmetic,
)
from verify import ClaimVerifier, replay_predictions


LIXILAN_ABSTRACT = (
    "Mean body weight decreased by 0.7 kg with iGlarLixi and increased by 0.7 kg "
    "with iGlar (1.4 kg difference, P < 0.0001)."
)


def extraction(**overrides):
    payload = {
        "population_check": "match",
        "drug_indication_check": "match",
        "actually_stated_check": "match",
        "mismatch_bases": [],
        "evidence_source": None,
        "evidence_page": None,
        "evidence_span": None,
        "arithmetic": ArithmeticExtraction(operation="none"),
        "supporting_values": [],
        "page_transcripts": [],
    }
    payload.update(overrides)
    return ClaimExtractionResponse(**payload)


def transcript_injector(label, page_number, text):
    def inject(sources, **_):
        for source in sources:
            if source.get("label") == label:
                source["pages"][page_number]["transcript"] = text
        return []

    return inject


class TestArithmeticFailsClosed(unittest.TestCase):
    def test_missing_operand_is_unclear_not_match(self):
        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="percentage",
                claimed_numeric=42.2,
                source_operands=[187],
            )
        )
        self.assertEqual(status, "unclear", info)

    def test_missing_operation_inputs_cannot_support(self):
        verdict = resolve_verdict(
            {
                "figure": "unclear",
                "population": "match",
                "drug_indication": "match",
                "actually_stated": "match",
            },
            span_verified=True,
        )
        self.assertEqual(verdict, "cannot_verify")

    def test_false_ratio_is_mismatch(self):
        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="ratio",
                claimed_numeric=2.0,
                source_operands=[1.63, 1.34],
            )
        )
        self.assertEqual(status, "mismatch", info)

    def test_reported_signed_difference_wins(self):
        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="difference",
                claimed_numeric=-0.2,
                source_operands=[-1.3, -1.1],
                reported_value=-0.2,
            )
        )
        self.assertEqual(status, "match", info)
        self.assertEqual(info["computed_value"], -0.2)

    def test_reversed_reported_difference_is_mismatch(self):
        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="difference",
                claimed_numeric=-0.2,
                source_operands=[-1.3, -1.1],
                reported_value=0.2,
            )
        )
        self.assertEqual(status, "mismatch", info)
        self.assertIn("opposite", info["result"])

    def test_unsigned_two_operand_direction_conflict_is_unclear(self):
        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="difference",
                claimed_numeric=2.0,
                source_operands=[8.5, 6.5],
                direction="decrease",
            )
        )
        self.assertEqual(status, "unclear", info)
        self.assertIn("unsigned", info["result"])

    def test_signed_two_operand_direction_conflict_is_mismatch(self):
        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="difference",
                claimed_numeric=2.0,
                source_operands=[-6.5, -8.5],
                direction="decrease",
            )
        )
        self.assertEqual(status, "mismatch", info)
        self.assertIn("increase", info["result"])

    def test_reported_adjusted_odds_ratio_beats_the_crude_check(self):
        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="odds_ratio",
                claimed_numeric=1.65,
                source_operands=[187, 443, 141, 444],
                reported_value=1.65,
            )
        )
        self.assertEqual(status, "match", info)

        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="odds_ratio",
                claimed_numeric=1.57,
                source_operands=[187, 443, 141, 444],
                reported_value=1.65,
            )
        )
        self.assertEqual(status, "mismatch", info)

    def test_one_bad_composite_component_fails_the_whole_figure(self):
        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="composite",
                components=[
                    ArithmeticComponent("percentage", 42.2, [187, 443]),
                    ArithmeticComponent("percentage", 31.8, [141, 444]),
                    ArithmeticComponent("odds_ratio", 1.65, [187, 443, 141, 444], reported_value=1.65),
                ],
            )
        )
        self.assertEqual(status, "match", info)
        self.assertEqual(sorted(info["from_source"]), [141, 187, 443, 444])

        status, info = verify_arithmetic(
            ArithmeticSpec(
                operation="composite",
                components=[
                    ArithmeticComponent("percentage", 42.2, [187, 443]),
                    ArithmeticComponent("percentage", 90.0, [141, 444]),
                ],
            )
        )
        self.assertEqual(status, "mismatch", info)


class TestSpanIsExactAndContiguous(unittest.TestCase):
    def test_exact_span_accepts_line_break_and_unicode_minus(self):
        page = "The change was \u22121.63% for iGlarLixi and\n\u22121.34% for iGlar."
        span = "The change was -1.63% for iGlarLixi and -1.34% for iGlar."
        self.assertTrue(check_span_presence(span, page))

    def test_control_character_minus_is_read_as_a_sign(self):
        page = "least squares mean difference \x010.2% (95% CI \x010.4, \x010.1)"
        span = "least squares mean difference -0.2% (95% CI -0.4, -0.1)"
        self.assertTrue(check_span_presence(span, page))
        self.assertFalse(check_span_presence("least squares mean difference 0.2%", page))

    def test_line_break_hyphenation_is_a_layout_artifact(self):
        page = "meeting noninferior-\nity (least squares mean difference -0.2%)"
        span = "meeting noninferiority (least squares mean difference -0.2%)"
        self.assertTrue(check_span_presence(span, page))

    def test_token_bag_is_not_a_verbatim_quote(self):
        page = "Table 2: iGlarLixi (n=468) Baseline 8.1 Week 30 6.5 Change -1.6"
        span = "iGlarLixi Baseline 8.1 Change -1.6 Week 30 6.5"
        self.assertFalse(check_span_presence(span, page))

    def test_ellipsis_is_an_authorial_join_not_a_verbatim_quote(self):
        page = "The trial reported a reduction of 1.3 percent and a difference of 0.2."
        stitched = "reduction of 1.3 percent ... difference of 0.2"
        self.assertFalse(check_span_presence(stitched, page))
        contiguous = "reduction of 1.3 percent and a difference of 0.2"
        self.assertTrue(check_span_presence(contiguous, page))

    def test_span_arithmetic_preserves_signs(self):
        spec = ArithmeticSpec(
            operation="difference",
            claimed_numeric=-1.3,
            source_operands=[1.3, 1.1],
        )
        page = "weight increased by 1.3 kg in the comparator arm"
        self.assertFalse(span_supports_arithmetic("1.3", spec, page, "match"))

    def test_match_requires_a_claimed_value_in_the_span(self):
        spec = ArithmeticSpec(
            operation="difference",
            claimed_numeric=-0.2,
            source_operands=[-1.3, -1.1],
            reported_value=-0.2,
        )
        page = "change -1.3 vs -1.1; difference -0.2 (95% CI -0.4, -0.1)"
        self.assertTrue(span_supports_arithmetic("-0.2 (-0.4, -0.1)", spec, page, "match"))
        self.assertFalse(span_supports_arithmetic("change -1.3 vs -1.1", spec, page, "match"))


class TestQuantitativeDetection(unittest.TestCase):
    def test_unitless_numeric_comparison_is_quantitative(self):
        self.assertTrue(claim_has_quantitative_assertion("HbA1c fell 1.3 vs 1.1"))
        self.assertTrue(claim_has_quantitative_assertion("a difference of 12.2 units"))
        self.assertFalse(claim_has_quantitative_assertion("Weight benefit while switching"))

    def test_the_word_or_is_not_an_odds_ratio(self):
        self.assertFalse(
            claim_has_quantitative_assertion(
                "controlled on basal insulin or oral agents"
            )
        )
        self.assertTrue(claim_has_quantitative_assertion("OR 1.65 (95% CI 1.25-2.19)"))

    def test_direction_words_are_read_only_when_unambiguous(self):
        self.assertEqual(claim_direction("a 1.3% reduction from baseline"), "decrease")
        self.assertEqual(claim_direction("weight gain of 0.7 kg"), "increase")
        self.assertEqual(claim_direction("greater HbA1c reduction"), "decrease")
        self.assertEqual(claim_direction("lower weight gain"), "increase")
        self.assertIsNone(claim_direction("1.3 vs 1.1 with no stated direction"))


class TestFigureCoverage(unittest.TestCase):
    def test_comparison_anchors_ignore_drug_numbers_and_ci_labels(self):
        self.assertEqual(
            comparison_anchor_numbers(
                "Superior HbA1c reduction with iGlarLixi (1.3 +/- 0.1%) vs "
                "twice-daily premix insulin BIAsp 30 (1.1 +/- 0.1%); 95% CI -0.4, -0.1"
            ),
            [1.3, 1.1],
        )
        self.assertEqual(
            comparison_anchor_numbers(
                "42.20% vs 31.80% (OR 1.65; 95% CI 1.25-2.19)"
            ),
            [42.2, 31.8],
        )
        self.assertEqual(
            comparison_anchor_numbers(
                "Lower risk of level-2 hypoglycemia (ADA: <54 mg/dL) with iGlarLixi "
                "vs premix BIAsp 30 [OR 0.45 (0.28, 0.73)]"
            ),
            [],
        )

    def test_signed_operands_still_account_for_unsigned_claim_anchors(self):
        spec = ArithmeticSpec(
            operation="difference",
            claimed_numeric=-0.2,
            source_operands=[-1.3, -1.1],
            reported_value=-0.2,
        )
        self.assertEqual(
            figure_coverage_gaps(
                "Superior HbA1c reduction with iGlarLixi (1.3 +/- 0.1%) vs "
                "BIAsp 30 (1.1 +/- 0.1%)",
                spec,
            ),
            [],
        )

    def test_missing_comparison_components_are_reported(self):
        spec = ArithmeticSpec(
            operation="percentage",
            claimed_numeric=42.2,
            source_operands=[187, 443],
        )
        self.assertIn(
            "comparison percentages",
            figure_coverage_gaps("42.20% vs 31.80%", spec),
        )

    def test_missing_effect_estimate_is_reported(self):
        spec = ArithmeticSpec(operation="percentage", claimed_numeric=42.2, source_operands=[187, 443])
        self.assertTrue(
            any(
                gap.startswith("OR")
                for gap in figure_coverage_gaps(
                    "42.20% vs 31.80% (OR 1.65; 95% CI 1.25-2.19)",
                    spec,
                )
            )
        )


class TestReasonBuilder(unittest.TestCase):
    def test_comparator_substitution_and_arithmetic_are_both_explained(self):
        reason = build_reason(
            claim_text="~2x greater HbA1c reduction vs basal insulin",
            checks={
                "figure": "mismatch",
                "population": "match",
                "drug_indication": "match",
                "actually_stated": "mismatch",
            },
            bases=["comparator_substitution", "numeric"],
            computed={"claimed": "~2x", "from_source": [1.63, 1.34], "result": "1.22x"},
        )
        self.assertIn("different comparator", reason)
        self.assertIn("1.22x", reason)

    def test_study_design_reason_quotes_the_limitation(self):
        reason = build_reason(
            claim_text="shown across three trial panels",
            checks={
                "figure": None,
                "population": "match",
                "drug_indication": "match",
                "actually_stated": "mismatch",
            },
            bases=["study_design"],
            computed=None,
            limitation_span="this report describes post hoc analyses and is hypothesis generating",
        )
        self.assertIn("post hoc", reason)
        self.assertIn("hypothesis generating", reason)

    def test_supporting_values_are_shown_for_a_qualitative_claim(self):
        reason = build_reason(
            claim_text="Weight neutral",
            checks={
                "figure": None,
                "population": "match",
                "drug_indication": "match",
                "actually_stated": "mismatch",
            },
            bases=["not_stated"],
            computed=None,
            supporting_values=[-0.7, 0.7],
        )
        self.assertIn("-0.7, 0.7", reason)

    def test_supported_reason_uses_claim_and_computed_result(self):
        reason = supported_reason(
            "OR 0.45 (0.28, 0.73)",
            {"claimed": "0.45", "from_source": [28, 442, 57, 441], "result": "Reported OR: 0.45"},
        )
        self.assertIn("supports the claim", reason)
        self.assertIn("0.45", reason)


class TestSourceResolution(unittest.TestCase):
    def test_multi_source_citation_resolves_all_three_pdfs(self):
        refs = resolve_source_refs("papers/Tabak2020_PSM.pdf (plus LixiLan-L abstract and SoliMix)")
        names = sorted(ref.label for ref in refs)
        self.assertEqual(
            names,
            ["LixiLan-L_Aroda2016_abstract.pdf", "SoliMix_Rosenstock2021.pdf", "Tabak2020_PSM.pdf"],
        )
        self.assertTrue(all(ref.path and os.path.isfile(ref.path) for ref in refs))

    def test_page_annotation_is_not_treated_as_a_second_source(self):
        for citation in (
            "SOLIQUA_LBL.pdf (Prescribing Information, page 4)",
            "SOLIQUA_LBL.pdf (see page 4)",
        ):
            refs = resolve_source_refs(citation)
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].label, "SOLIQUA_LBL.pdf")


class TestEvaluationHonesty(unittest.TestCase):
    def test_missing_prediction_is_never_credited(self):
        result = evaluate_predictions(
            [],
            [{"claim_id": "C1", "verdict": "cannot_verify"}],
        )
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["present"], 0)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(result["missing_claims"], ["C1"])

    def test_false_supported_and_false_overstated_are_both_counted(self):
        gold = [
            {"claim_id": "A", "verdict": "supported"},
            {"claim_id": "B", "verdict": "overstated"},
            {"claim_id": "C", "verdict": "cannot_verify"},
        ]
        predictions = [
            {"claim_id": "A", "verdict": "overstated"},
            {"claim_id": "B", "verdict": "supported"},
            {"claim_id": "C", "verdict": "cannot_verify"},
        ]
        result = evaluate_predictions(predictions, gold)
        self.assertEqual(result["false_supported_claims"], ["B"])
        self.assertEqual(result["false_overstated_claims"], ["A"])
        self.assertEqual(result["supported_precision"], 0.0)
        self.assertEqual(result["supported_recall"], 0.0)

    def test_checks_page_span_and_computed_are_compared(self):
        gold = [{
            "claim_id": "C2",
            "verdict": "supported",
            "checks": {
                "figure": "match",
                "population": "match",
                "drug_indication": "match",
                "actually_stated": "match",
            },
            "evidence_page": 5,
            "evidence_span": "difference -0.2 (-0.4, -0.1)",
            "computed": {"from_source": [1.3, 1.1]},
        }]
        prediction = [{
            "claim_id": "C2",
            "verdict": "supported",
            "checks": {
                "figure": "match",
                "population": "match",
                "drug_indication": "match",
                "actually_stated": "match",
            },
            "evidence_page": 5,
            "evidence_span": "LS mean difference -0.2 (-0.4, -0.1) %",
            "computed": {"from_source": [1.3, 1.1, -0.2]},
        }]
        result = evaluate_predictions(prediction, gold)
        self.assertEqual(result["checks_match_count"], 1)
        self.assertEqual(result["evidence_page_match_count"], 1)
        self.assertEqual(result["evidence_span_containment_count"], 1)
        self.assertEqual(result["computed_match_count"], 1)
        self.assertEqual(result["trivial_baseline"]["false_supported_count"], 0)

    def test_audit_log_summary_counts_request_outcomes(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "log.jsonl")
            append_llm_log(path, "p", '{"ok": 1}', "m", 0.1, metadata={"claim_id": "C1"})
            append_llm_log(path, "p", "quota", "m", 0.1, metadata={"claim_id": "C2", "error": True})
            summary = summarize_audit_log(path)
            self.assertEqual(summary["requests"], 2)
            self.assertEqual(summary["extraction_success"], 1)
            self.assertEqual(summary["failures"], 1)
            self.assertEqual(summary["successful_claim_ids"], ["C1"])


class TestPromptIsClaimAgnostic(unittest.TestCase):
    def test_system_instruction_has_no_gold_answer_text(self):
        for forbidden in (
            "with Obesity",
            "Weight neutral",
            "lixisenatide monocomponent",
            "three trial panels",
            "SoliMix",
            "overstated",
        ):
            self.assertNotIn(forbidden.lower(), SYSTEM_INSTRUCTION.lower())

    def test_user_prompt_contains_no_gold_answer_text(self):
        claims = {item["claim_id"]: item for item in json.load(open("data/claims.json"))}
        prompt = build_prompt(claims["C1"], [{"label": "x.pdf", "path": "x.pdf", "pages": {}}])
        self.assertIn(claims["C1"]["text"], prompt)
        self.assertNotIn("with Obesity", prompt)
        self.assertNotIn("Weight neutral", prompt)
        self.assertNotIn("overstated", prompt.lower())


class TestAuditLog(unittest.TestCase):
    def test_log_hash_matches_verbatim_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "log.jsonl")
            prompt = "CLAIM TO AUDIT\nClaim ID: X"
            append_llm_log(path, prompt, '{"ok": true}', "test-model", 0.1)
            row = json.loads(open(path).read().strip())
            self.assertEqual(row["prompt"], prompt)
            self.assertEqual(row["prompt_sha256"], prompt_sha256(prompt))

    def test_system_instruction_and_image_parts_are_logged(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "log.jsonl")
            system = "You are a regulatory evidence auditor."
            parts = [{"type": "image", "source": "a.pdf", "page": 5, "sha256": "abc"}]
            append_llm_log(
                path,
                "prompt",
                "{}",
                "test-model",
                0.2,
                system_instruction=system,
                request_parts=parts,
            )
            row = json.loads(open(path).read().strip())
            self.assertEqual(row["system_instruction"], system)
            self.assertEqual(row["system_instruction_sha256"], prompt_sha256(system))
            self.assertEqual(row["request_parts"], parts)


class TestGoldContract(unittest.TestCase):
    def test_gold_matches_the_assignment_example_for_c3(self):
        gold = {item["claim_id"]: item for item in json.load(open("data/gold.json"))}
        self.assertEqual(gold["C3"]["checks"]["drug_indication"], "match")

    def test_non_numeric_claims_have_no_figure_check(self):
        gold = {item["claim_id"]: item for item in json.load(open("data/gold.json"))}
        self.assertIsNone(gold["C5"]["checks"]["figure"])
        self.assertIsNone(gold["C7"]["checks"]["figure"])


class TestTranscriptCache(unittest.TestCase):
    def test_roundtrip_and_model_separation(self):
        with tempfile.TemporaryDirectory() as td:
            image = b"page-image-bytes"
            path = _transcript_cache_path(td, image, "model-a")
            _store_cached_transcript(path, "a.pdf", 4, "exact page text")
            self.assertEqual(_load_cached_transcript(path), "exact page text")
            self.assertNotEqual(path, _transcript_cache_path(td, image, "model-b"))


class TestStructuredJsonParsing(unittest.TestCase):
    def test_repairs_invalid_unicode_escape(self):
        raw = '{"text": "dose 5 \\u12zz"}'
        with self.assertRaises(ValueError):
            json.loads(raw)
        parsed = parse_structured_json(raw)
        self.assertEqual(parsed["text"], "dose 5 \\u12zz")


class TestReplay(unittest.TestCase):
    def test_replay_rebuilds_a_supported_verdict_without_a_model(self):
        claim = next(item for item in json.load(open("data/claims.json")) if item["claim_id"] == "C2")
        payload = {
            "population_check": "match",
            "drug_indication_check": "match",
            "actually_stated_check": "match",
            "mismatch_bases": [],
            "evidence_source": "SoliMix_Rosenstock2021.pdf",
            "evidence_page": 5,
            "evidence_span": "LS mean difference iGlarLixi\nvs BIAsp 30 (95% CI); p-value\n-0.2 (-0.4, -0.1) %; <0.001",
            "arithmetic": {
                "operation": "difference",
                "claimed_text": "1.3 vs 1.1",
                "claimed_numeric": -0.2,
                "source_operands": [-1.3, -1.1],
                "reported_value": -0.2,
            },
            "supporting_values": [],
            "page_transcripts": [],
        }
        with tempfile.TemporaryDirectory() as td:
            claims_path = os.path.join(td, "claims.json")
            log_path = os.path.join(td, "log.jsonl")
            out_path = os.path.join(td, "predictions.json")
            with open(claims_path, "w", encoding="utf-8") as handle:
                json.dump([claim], handle)
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "metadata": {"claim_id": "C2"},
                            "response_raw": json.dumps(payload),
                            "model": "test-model",
                        }
                    )
                    + "\n"
                )
            predictions = replay_predictions(claims_path, log_path, out_path, base_dir=".")
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0]["verdict"], "supported")
            self.assertEqual(predictions[0]["evidence_page"], 5)
            self.assertTrue(os.path.isfile(out_path))


class TestScriptedPipeline(unittest.TestCase):
    """End-to-end verdicts with a deterministic extractor and no network."""

    def test_transcript_cannot_overwrite_native_or_frozen_page_text(self):
        from extractor import PageTranscript

        native = {"text": "native page", "transcript": None, "has_native_text": True, "needs_vision": False}
        scanned = {"text": "", "transcript": None, "has_native_text": False, "needs_vision": True}
        frozen = {"text": "", "transcript": "first transcript", "has_native_text": False, "needs_vision": True}
        sources = [
            {"label": "a.pdf", "path": "a.pdf", "pages": {1: native, 2: scanned, 3: frozen}},
        ]
        verifier = ClaimVerifier(extractor=lambda **_: extraction(), transcriber=lambda *a, **k: [])
        verifier._inject_transcripts(
            extraction(
                page_transcripts=[
                    PageTranscript(source="a.pdf", page_number=1, text="summary"),
                    PageTranscript(source="a.pdf", page_number=2, text="exact scan"),
                    PageTranscript(source="a.pdf", page_number=3, text="rewritten scan"),
                ]
            ),
            sources,
        )
        self.assertIsNone(native["transcript"])
        self.assertEqual(scanned["transcript"], "exact scan")
        self.assertEqual(frozen["transcript"], "first transcript")

    def verify(self, claim_id, extraction_result, transcriber=None):
        claim = next(item for item in json.load(open("data/claims.json")) if item["claim_id"] == claim_id)
        verifier = ClaimVerifier(
            base_dir=".",
            extractor=lambda **_: extraction_result,
            transcriber=transcriber or (lambda *args, **kwargs: []),
        )
        return verifier.verify(claim)

    def test_supported_claim_with_verified_span_and_arithmetic(self):
        result = extraction(
            evidence_source="SoliMix_Rosenstock2021.pdf",
            evidence_page=5,
            evidence_span=(
                "LS mean difference iGlarLixi\n"
                "vs BIAsp 30 (95% CI); p-value\n"
                "-0.2 (-0.4, -0.1) %; <0.001"
            ),
            arithmetic=ArithmeticExtraction(
                operation="difference",
                claimed_text="1.3 +/- 0.1% vs 1.1 +/- 0.1%",
                claimed_numeric=-0.2,
                source_operands=[-1.3, -1.1],
                reported_value=-0.2,
            ),
        )
        prediction = self.verify("C2", result)
        self.assertEqual(prediction["verdict"], "supported")
        self.assertEqual(prediction["checks"]["figure"], "match")
        self.assertEqual(prediction["evidence_page"], 5)
        self.assertIn("supports the claim", prediction["reason"])
        # The span is clean: no y-axis ticks or baseline values.
        self.assertNotIn("-18.0", prediction["evidence_span"])
        self.assertNotIn("71 ± 7", prediction["evidence_span"])

    def test_numeric_mismatch_is_reconciled_when_the_figure_and_qualifiers_match(self):
        result = extraction(
            evidence_source="SoliMix_Rosenstock2021.pdf",
            evidence_page=5,
            evidence_span=(
                "Superiority:\n"
                "LS mean difference iGlarLixi\n"
                "vs BIAsp 30 (95% CI); p-value\n"
                "-0.2 (-0.4, -0.1) %; <0.001"
            ),
            actually_stated_check="mismatch",
            mismatch_bases=["numeric"],
            arithmetic=ArithmeticExtraction(
                operation="difference",
                claimed_text="95% CI -0.4, -0.1",
                claimed_numeric=-0.2,
                source_operands=[-1.3, -1.1],
                reported_value=-0.2,
            ),
        )
        prediction = self.verify("C2", result)
        self.assertEqual(prediction["verdict"], "supported")
        self.assertEqual(prediction["checks"]["actually_stated"], "match")

    def test_numeric_mismatch_is_not_reconciled_by_a_different_confidence_level(self):
        result = extraction(
            evidence_source="SoliMix_Rosenstock2021.pdf",
            evidence_page=5,
            evidence_span=(
                "Noninferiority:\n"
                "LS mean difference iGlarLixi\n"
                "vs BIAsp 30 (97.5% CI); p-value\n"
                "-0.2 (-0.4, -0.1) %; <0.001\u2020"
            ),
            actually_stated_check="mismatch",
            mismatch_bases=["numeric"],
            arithmetic=ArithmeticExtraction(
                operation="difference",
                claimed_numeric=-0.2,
                source_operands=[-1.3, -1.1],
                reported_value=-0.2,
            ),
        )
        prediction = self.verify("C2", result)
        self.assertEqual(prediction["verdict"], "overstated")
        self.assertEqual(prediction["checks"]["actually_stated"], "mismatch")

    def test_comparator_substitution_is_overstated_and_explained(self):
        result = extraction(
            evidence_source="LixiLan-O_Rosenstock2016.pdf",
            evidence_page=13,
            evidence_span=(
                "The least squares (LS) mean changes from baseline to Week 30 in HbA1c "
                "were \u22121.63% for iGlarLixi, \u22121.34% for iGlar, and \u22120.85% for Lixi "
                "(Table 2; Fig. 1B)."
            ),
            actually_stated_check="mismatch",
            mismatch_bases=["comparator_substitution", "numeric"],
            arithmetic=ArithmeticExtraction(
                operation="ratio",
                claimed_text="~2x",
                claimed_numeric=2.0,
                source_operands=[-1.63, -1.34],
            ),
        )
        prediction = self.verify("C3", result)
        self.assertEqual(prediction["verdict"], "overstated")
        self.assertEqual(prediction["checks"]["figure"], "mismatch")
        self.assertIn("different comparator", prediction["reason"])
        self.assertIn("1.22x", prediction["reason"])

    def test_study_design_mismatch_quotes_the_source_limitation(self):
        result = extraction(
            evidence_source="Tabak2020_PSM.pdf",
            evidence_page=1,
            evidence_span=(
                "We compared these treat-\nments in a post hoc propensity score matched\n"
                "analysis using randomized trial data."
            ),
            actually_stated_check="mismatch",
            mismatch_bases=["study_design"],
        )
        prediction = self.verify("C7", result)
        self.assertEqual(prediction["verdict"], "overstated")
        self.assertEqual(prediction["evidence_page"], 12)
        self.assertIn("hypothesis generating", prediction["evidence_span"])
        self.assertIn("post hoc", prediction["reason"])

    def test_qualitative_claim_keeps_decisive_computed_values(self):
        result = extraction(
            evidence_source="LixiLan-L_Aroda2016_abstract.pdf",
            evidence_page=2,
            evidence_span=LIXILAN_ABSTRACT,
            actually_stated_check="mismatch",
            mismatch_bases=["not_stated"],
            supporting_values=[-0.7, 0.7],
        )
        prediction = self.verify(
            "C5",
            result,
            transcriber=transcript_injector(
                "LixiLan-L_Aroda2016_abstract.pdf", 2, LIXILAN_ABSTRACT
            ),
        )
        self.assertEqual(prediction["verdict"], "overstated")
        self.assertIsNone(prediction["checks"]["figure"])
        self.assertEqual(prediction["computed"]["from_source"], [-0.7, 0.7])
        self.assertIn("-0.7, 0.7", prediction["reason"])

    def test_invented_span_fails_closed(self):
        result = extraction(
            evidence_source="SoliMix_Rosenstock2021.pdf",
            evidence_page=5,
            evidence_span="Superiority was proven beyond any doubt",
        )
        prediction = self.verify("C2", result)
        self.assertEqual(prediction["verdict"], "cannot_verify")
        self.assertIn("not found verbatim", prediction["reason"])

    def test_contradictory_extraction_fails_closed(self):
        result = extraction(
            evidence_source="SoliMix_Rosenstock2021.pdf",
            evidence_page=5,
            evidence_span=(
                "LS mean difference iGlarLixi\n"
                "vs BIAsp 30 (95% CI); p-value\n"
                "-0.2 (-0.4, -0.1) %; <0.001"
            ),
            mismatch_bases=["numeric"],
        )
        prediction = self.verify("C2", result)
        self.assertEqual(prediction["verdict"], "cannot_verify")
        self.assertIn("contradictory", prediction["reason"].lower())

    def test_invented_source_value_is_not_certified(self):
        result = extraction(
            evidence_source="SoliMix_Rosenstock2021.pdf",
            evidence_page=5,
            evidence_span=(
                "LS mean difference iGlarLixi\n"
                "vs BIAsp 30 (95% CI); p-value\n"
                "-0.2 (-0.4, -0.1) %; <0.001"
            ),
            arithmetic=ArithmeticExtraction(
                operation="difference",
                claimed_text="1.3 vs 1.1",
                claimed_numeric=-0.2,
                source_operands=[1.3, 1.1],
                reported_value=-0.2,
            ),
        )
        prediction = self.verify("C2", result)
        self.assertEqual(prediction["verdict"], "cannot_verify")
        self.assertIn("numbers used", prediction["reason"])


if __name__ == "__main__":
    unittest.main()
