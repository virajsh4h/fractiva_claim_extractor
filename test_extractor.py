import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from audit import append_llm_log
from extractor import (
    ArithmeticExtraction,
    ArithmeticExtractionComponent,
    ClaimExtractionResponse,
    _call_openrouter_chat,
    _nvidia_chat_payload,
    _validation_issues,
    extract_claim_evidence,
    parse_structured_json,
    transcribe_scanned_pages,
)


class TestExtractor(unittest.TestCase):
    def test_pydantic_schema_validation(self):
        payload = {
            "population_check": "match",
            "drug_indication_check": "mismatch",
            "actually_stated_check": "mismatch",
            "mismatch_bases": ["comparator_substitution", "numeric"],
            "evidence_page": 13,
            "evidence_span": "changes from baseline to Week 30 in HbA1c were -1.63%",
            "arithmetic": {
                "operation": "ratio",
                "claimed_text": "~2x",
                "claimed_numeric": 2.0,
                "source_operands": [-1.63, -1.34],
            },
            "supporting_values": [],
        }
        res = ClaimExtractionResponse(**payload)
        self.assertEqual(res.population_check, "match")
        self.assertEqual(res.arithmetic.operation, "ratio")
        self.assertEqual(res.arithmetic.source_operands, [-1.63, -1.34])
        self.assertEqual(res.mismatch_bases, ["comparator_substitution", "numeric"])

    def test_schema_rejects_free_text_check_values(self):
        payload = {
            "population_check": "probably fine",
            "drug_indication_check": "match",
            "actually_stated_check": "match",
            "mismatch_bases": [],
            "evidence_page": None,
            "evidence_span": None,
            "arithmetic": {"operation": "none"},
            "supporting_values": [],
        }
        with self.assertRaises(Exception):
            ClaimExtractionResponse(**payload)

    def test_schema_rejects_unknown_mismatch_basis(self):
        payload = {
            "population_check": "mismatch",
            "drug_indication_check": "match",
            "actually_stated_check": "match",
            "mismatch_bases": ["vibes"],
            "evidence_page": None,
            "evidence_span": None,
            "arithmetic": {"operation": "none"},
            "supporting_values": [],
        }
        with self.assertRaises(Exception):
            ClaimExtractionResponse(**payload)

    def test_model_reason_is_not_part_of_the_schema(self):
        payload = {
            "population_check": "match",
            "drug_indication_check": "match",
            "actually_stated_check": "match",
            "mismatch_bases": [],
            "evidence_page": None,
            "evidence_span": None,
            "arithmetic": {"operation": "none"},
            "supporting_values": [],
            "reason": "The model should not own the reviewer reason.",
        }
        res = ClaimExtractionResponse(**payload)
        self.assertFalse(hasattr(res, "reason"))

    def test_append_llm_log(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tf:
            log_path = tf.name

        try:
            append_llm_log(
                log_path=log_path,
                prompt="Verify claim C3",
                response_raw='{"verdict": "overstated"}',
                model="test-model",
                latency_s=0.42,
                metadata={"claim_id": "C3"},
            )
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["prompt"], "Verify claim C3")
            self.assertEqual(entry["model"], "test-model")
            self.assertEqual(entry["metadata"]["claim_id"], "C3")
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_missing_api_key_fails_closed(self):
        claim = {
            "claim_id": "C1",
            "text": "Indicated for adults with type 2 diabetes",
            "population": "adults with type 2 diabetes",
            "molecule": "iGlarLixi",
            "cited_paper": "SOLIQUA_LBL.pdf",
        }
        with patch("extractor.load_local_env"), patch(
            "extractor.get_api_key", return_value=None
        ), patch("extractor.get_nvidia_api_key", return_value=None), patch(
            "extractor.get_openrouter_api_key", return_value=None
        ):
            res = extract_claim_evidence(claim, pages_data={})
            self.assertEqual(res.population_check, "unclear")
            self.assertEqual(res.drug_indication_check, "unclear")
            self.assertEqual(res.actually_stated_check, "unclear")
            self.assertIsNone(res.evidence_span)
            self.assertEqual(res.arithmetic.operation, "none")
            failure = getattr(res, "_failure_reason", "")
            self.assertIn("missing", failure.lower())

    def test_confidence_level_conflict_requests_repair(self):
        page_text = (
            "baseline HbA1c was reduced by 1.3% with iGlarLixi and 1.1% with "
            "BIAsp 30 (LS mean difference -0.2% [97.5% CI -0.4, -0.1]; P < 0.001)."
        )
        extraction = ClaimExtractionResponse(
            population_check="match",
            drug_indication_check="match",
            actually_stated_check="mismatch",
            mismatch_bases=["numeric"],
            evidence_page=1,
            evidence_span=page_text,
            arithmetic=ArithmeticExtraction(
                operation="difference",
                claimed_numeric=-0.2,
                source_operands=[-1.3, -1.1],
                reported_value=-0.2,
            ),
        )
        sources = [
            {
                "label": "trial.pdf",
                "pages": {
                    1: {
                        "text": page_text,
                        "has_native_text": True,
                        "needs_vision": False,
                    }
                },
            }
        ]
        issues = _validation_issues(
            extraction,
            {
                "text": (
                    "Superior HbA1c reduction with iGlarLixi (1.3%) vs BIAsp 30 "
                    "(1.1%); 95% CI -0.4, -0.1"
                )
            },
            sources,
        )
        self.assertTrue(any("97.5%" in issue for issue in issues), issues)
        self.assertTrue(any("95%" in issue for issue in issues), issues)

    def test_composite_with_redundant_arm_values_requests_repair(self):
        page_text = "The change was -1.3% and -1.1%; difference -0.2%."
        extraction = ClaimExtractionResponse(
            population_check="match",
            drug_indication_check="match",
            actually_stated_check="mismatch",
            mismatch_bases=["numeric"],
            evidence_page=1,
            evidence_span=page_text,
            arithmetic=ArithmeticExtraction(
                operation="composite",
                claimed_numeric=-0.2,
                components=[
                    ArithmeticExtractionComponent(
                        operation="reported",
                        claimed_numeric=1.3,
                        source_operands=[-1.3],
                        reported_value=-1.3,
                    ),
                    ArithmeticExtractionComponent(
                        operation="reported",
                        claimed_numeric=1.1,
                        source_operands=[-1.1],
                        reported_value=-1.1,
                    ),
                    ArithmeticExtractionComponent(
                        operation="difference",
                        claimed_numeric=-0.2,
                        source_operands=[-1.3, -1.1],
                        reported_value=-0.2,
                    ),
                ],
            ),
        )
        sources = [
            {
                "label": "trial.pdf",
                "pages": {
                    1: {
                        "text": page_text,
                        "has_native_text": True,
                        "needs_vision": False,
                    }
                },
            }
        ]
        issues = _validation_issues(
            extraction,
            {"text": "1.3% vs 1.1%; difference -0.2%"},
            sources,
        )
        self.assertTrue(
            any("derived comparison" in issue for issue in issues),
            issues,
        )

    def test_nvidia_transcription_retries_transient_500(self):
        page = {
            "page_number": 1,
            "text": "",
            "has_native_text": False,
            "needs_vision": True,
            "image_bytes": b"fake-png",
            "image_mime": "image/png",
        }
        sources = [
            {
                "label": "scan.pdf",
                "path": "scan.pdf",
                "pages": {1: page},
            }
        ]
        calls = []

        def flaky_nvidia(**kwargs):
            calls.append(kwargs["model"])
            if len(calls) < 3:
                raise RuntimeError("NVIDIA HTTP 500: Failed to generate completions")
            return json.dumps(
                {
                    "page_transcripts": [
                        {
                            "source": "scan.pdf",
                            "page_number": 1,
                            "text": "Exact page text.",
                        }
                    ]
                }
            )

        with patch("extractor.load_local_env"), patch(
            "extractor.get_api_key", return_value=None
        ), patch("extractor.get_nvidia_api_key", return_value="nvidia-key"), patch(
            "extractor._call_nvidia_chat", side_effect=flaky_nvidia
        ), patch(
            "extractor.time.sleep"
        ), patch(
            "extractor.append_llm_log"
        ):
            transcripts = transcribe_scanned_pages(
                sources,
                cache_dir=None,
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(page["transcript"], "Exact page text.")
        self.assertEqual(transcripts[0].text, "Exact page text.")

    def test_extraction_uses_vision_model_when_images_remain(self):
        page_text = "The change was -1.3% for iGlarLixi and -1.1% for iGlar."
        response = {
            "population_check": "match",
            "drug_indication_check": "match",
            "actually_stated_check": "match",
            "mismatch_bases": [],
            "evidence_source": "scan.pdf",
            "evidence_page": 1,
            "evidence_span": page_text,
            "arithmetic": {
                "operation": "ratio",
                "claimed_numeric": 2.0,
                "source_operands": [-1.3, -1.1],
            },
            "supporting_values": [],
            "page_transcripts": [],
        }
        sources = [
            {
                "label": "scan.pdf",
                "path": "scan.pdf",
                "pages": {
                    1: {
                        "page_number": 1,
                        "text": page_text,
                        "has_native_text": True,
                        "needs_vision": True,
                        "image_bytes": b"fake-png",
                        "image_mime": "image/png",
                    }
                },
            }
        ]
        models = []

        def nvidia_success(**kwargs):
            models.append(kwargs["model"])
            return json.dumps(response)

        with patch("extractor.load_local_env"), patch(
            "extractor.get_api_key", return_value=None
        ), patch("extractor.get_nvidia_api_key", return_value="nvidia-key"), patch(
            "extractor.get_nvidia_extraction_model", return_value="nvidia/text"
        ), patch(
            "extractor.get_nvidia_vision_model", return_value="nvidia/vision"
        ), patch(
            "extractor._call_nvidia_chat", side_effect=nvidia_success
        ), patch(
            "extractor.append_llm_log"
        ) as logged:
            res = extract_claim_evidence(
                {
                    "claim_id": "C_TEST",
                    "text": "iGlarLixi gave ~2x the HbA1c reduction",
                    "cited_paper": "scan.pdf",
                },
                sources=sources,
                log_path="unused.jsonl",
            )

        self.assertEqual(models, ["nvidia/vision"])
        self.assertEqual(res.evidence_page, 1)
        metadata = logged.call_args.kwargs["metadata"]
        self.assertTrue(metadata["fallback"])
        self.assertIn("GEMINI_API_KEY is missing", metadata["fallback_reason"])

    def test_nvidia_schema_echo_is_retried_as_a_parse_failure(self):
        page = {
            "page_number": 1,
            "text": "",
            "has_native_text": False,
            "needs_vision": True,
            "image_bytes": b"fake-png",
            "image_mime": "image/png",
        }
        sources = [{"label": "scan.pdf", "path": "scan.pdf", "pages": {1: page}}]
        calls = []

        def echo_then_success(**kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                return json.dumps(
                    {
                        "$defs": {"PageTranscript": {"type": "object"}},
                        "properties": {"page_transcripts": {"type": "array"}},
                        "title": "TranscriptionResponse",
                        "type": "object",
                    }
                )
            return json.dumps(
                {
                    "page_transcripts": [
                        {
                            "source": "scan.pdf",
                            "page_number": 1,
                            "text": "Exact page text.",
                        }
                    ]
                }
            )

        with patch("extractor.load_local_env"), patch(
            "extractor.get_api_key", return_value=None
        ), patch("extractor.get_nvidia_api_key", return_value="nvidia-key"), patch(
            "extractor._call_nvidia_chat", side_effect=echo_then_success
        ), patch(
            "extractor.time.sleep"
        ), patch(
            "extractor.append_llm_log"
        ) as logged:
            transcripts = transcribe_scanned_pages(
                sources,
                cache_dir=None,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(page["transcript"], "Exact page text.")
        self.assertEqual(transcripts[0].text, "Exact page text.")
        first = logged.call_args_list[0].kwargs["metadata"]
        self.assertTrue(first.get("error"))
        self.assertIn("no matching page transcript", first["parse_error"])

    def test_nvidia_payload_uses_json_mode_and_data_url(self):
        payload = _nvidia_chat_payload(
            model="nvidia/test-model",
            prompt="Return the audit.",
            system_instruction="Follow the schema.",
            images=[{"bytes": b"fake-png", "mime": "image/png"}],
        )
        self.assertEqual(payload["model"], "nvidia/test-model")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["messages"][0]["content"], "Follow the schema.")
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Return the audit."})
        self.assertTrue(
            content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )

    def test_gemini_quota_failure_falls_back_to_nvidia(self):
        page_text = "The change was -1.63% for iGlarLixi and -1.34% for iGlar."
        response = {
            "population_check": "match",
            "drug_indication_check": "match",
            "actually_stated_check": "mismatch",
            "mismatch_bases": ["not_stated"],
            "evidence_source": "trial.pdf",
            "evidence_page": 1,
            "evidence_span": page_text,
            "arithmetic": {
                "operation": "ratio",
                "claimed_text": "~2x",
                "claimed_numeric": 2.0,
                "source_operands": [-1.63, -1.34],
            },
            "supporting_values": [],
            "page_transcripts": [],
        }
        claim = {
            "claim_id": "C_TEST",
            "text": "iGlarLixi gave ~2x the HbA1c reduction",
            "cited_paper": "trial.pdf",
        }
        sources = [
            {
                "label": "trial.pdf",
                "path": "trial.pdf",
                "pages": {
                    1: {
                        "page_number": 1,
                        "text": page_text,
                        "has_native_text": True,
                        "needs_vision": False,
                    }
                },
            }
        ]
        gemini_calls = []
        nvidia_calls = []

        def gemini_failure(**kwargs):
            gemini_calls.append(kwargs)
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

        def nvidia_success(**kwargs):
            nvidia_calls.append(kwargs)
            return json.dumps(response)

        with patch("extractor.load_local_env"), patch(
            "extractor.get_api_key", return_value="gemini-key"
        ), patch("extractor.get_nvidia_api_key", return_value="nvidia-key"), patch(
            "extractor._call_gemini", side_effect=gemini_failure
        ), patch(
            "extractor._call_nvidia_chat", side_effect=nvidia_success
        ), patch(
            "extractor.append_llm_log"
        ) as logged:
            res = extract_claim_evidence(
                claim,
                sources=sources,
                max_attempts=3,
                log_path="unused.jsonl",
            )

        self.assertEqual(len(gemini_calls), 1)
        self.assertEqual(len(nvidia_calls), 1)
        self.assertEqual(res.evidence_page, 1)
        self.assertEqual(res.arithmetic.source_operands, [-1.63, -1.34])
        first = logged.call_args_list[0].kwargs["metadata"]
        second = logged.call_args_list[1].kwargs["metadata"]
        self.assertEqual(first["provider"], "gemini")
        self.assertTrue(first.get("error"))
        self.assertEqual(second["provider"], "nvidia")
        self.assertTrue(second["fallback"])
        self.assertIn("429", second["fallback_reason"])
        self.assertEqual(
            nvidia_calls[0]["system_instruction"].count("JSON schema"),
            1,
        )


    def test_openrouter_retries_without_response_format_when_rejected(self):
        rejected = MagicMock(
            status_code=400,
            text='{"error":{"message":"response_format is not supported"}}',
        )
        accepted = MagicMock(status_code=200)
        accepted.json.return_value = {
            "choices": [{"message": {"content": '{"ok": true}'}}]
        }

        with patch(
            "extractor.requests.post", side_effect=[rejected, accepted]
        ) as post:
            text = _call_openrouter_chat(
                api_key="openrouter-key",
                model="nvidia/test:free",
                prompt="Return the audit.",
                system_instruction="Follow the schema.",
            )

        self.assertEqual(text, '{"ok": true}')
        first = post.call_args_list[0].kwargs["json"]
        second = post.call_args_list[1].kwargs["json"]
        self.assertIn("response_format", first)
        self.assertNotIn("response_format", second)
        self.assertEqual(
            post.call_args_list[0].kwargs["headers"]["Authorization"],
            "Bearer openrouter-key",
        )

    def test_extraction_falls_back_gemini_nvidia_openrouter_in_order(self):
        page_text = "The change was -1.63% for iGlarLixi and -1.34% for iGlar."
        response = {
            "population_check": "match",
            "drug_indication_check": "match",
            "actually_stated_check": "mismatch",
            "mismatch_bases": ["not_stated"],
            "evidence_source": "trial.pdf",
            "evidence_page": 1,
            "evidence_span": page_text,
            "arithmetic": {
                "operation": "ratio",
                "claimed_text": "~2x",
                "claimed_numeric": 2.0,
                "source_operands": [-1.63, -1.34],
            },
            "supporting_values": [],
            "page_transcripts": [],
        }
        claim = {
            "claim_id": "C_TEST",
            "text": "iGlarLixi gave ~2x the HbA1c reduction",
            "cited_paper": "trial.pdf",
        }
        sources = [
            {
                "label": "trial.pdf",
                "path": "trial.pdf",
                "pages": {
                    1: {
                        "page_number": 1,
                        "text": page_text,
                        "has_native_text": True,
                        "needs_vision": False,
                    }
                },
            }
        ]
        order = []

        def gemini_failure(**kwargs):
            order.append("gemini")
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

        def nvidia_failure(**kwargs):
            order.append("nvidia")
            raise RuntimeError("NVIDIA HTTP 500: upstream failure")

        def openrouter_success(**kwargs):
            order.append("openrouter")
            return json.dumps(response)

        with patch("extractor.load_local_env"), patch(
            "extractor.get_api_key", return_value="gemini-key"
        ), patch(
            "extractor.get_nvidia_api_key", return_value="nvidia-key"
        ), patch(
            "extractor.get_openrouter_api_key", return_value="openrouter-key"
        ), patch(
            "extractor._call_gemini", side_effect=gemini_failure
        ), patch(
            "extractor._call_nvidia_chat", side_effect=nvidia_failure
        ), patch(
            "extractor._call_openrouter_chat", side_effect=openrouter_success
        ), patch(
            "extractor.time.sleep"
        ), patch(
            "extractor.append_llm_log"
        ) as logged:
            res = extract_claim_evidence(
                claim,
                sources=sources,
                max_attempts=1,
                log_path="unused.jsonl",
            )

        self.assertEqual(order, ["gemini", "nvidia", "openrouter"])
        self.assertEqual(res.evidence_page, 1)
        providers = [call.kwargs["metadata"]["provider"] for call in logged.call_args_list]
        self.assertEqual(providers, ["gemini", "nvidia", "openrouter"])
        final = logged.call_args_list[-1].kwargs["metadata"]
        self.assertEqual(final["provider"], "openrouter")
        self.assertTrue(final["fallback"])
        self.assertIn("500", final["fallback_reason"])

    def test_openrouter_transcription_fallback_after_nvidia_failure(self):
        page = {
            "page_number": 1,
            "text": "",
            "has_native_text": False,
            "needs_vision": True,
            "image_bytes": b"fake-png",
            "image_mime": "image/png",
        }
        sources = [{"label": "scan.pdf", "path": "scan.pdf", "pages": {1: page}}]
        nvidia_calls = []
        openrouter_calls = []

        def nvidia_failure(**kwargs):
            nvidia_calls.append(kwargs["model"])
            raise RuntimeError("NVIDIA HTTP 500: upstream failure")

        def openrouter_success(**kwargs):
            openrouter_calls.append(kwargs)
            return json.dumps(
                {
                    "page_transcripts": [
                        {
                            "source": "scan.pdf",
                            "page_number": 1,
                            "text": "Exact page text.",
                        }
                    ]
                }
            )

        with patch("extractor.load_local_env"), patch(
            "extractor.get_api_key", return_value=None
        ), patch(
            "extractor.get_nvidia_api_key", return_value="nvidia-key"
        ), patch(
            "extractor.get_openrouter_api_key", return_value="openrouter-key"
        ), patch(
            "extractor.get_openrouter_vision_model",
            return_value="nvidia/vision:free",
        ), patch(
            "extractor._call_nvidia_chat", side_effect=nvidia_failure
        ), patch(
            "extractor._call_openrouter_chat", side_effect=openrouter_success
        ), patch(
            "extractor.time.sleep"
        ), patch(
            "extractor.append_llm_log"
        ) as logged:
            transcripts = transcribe_scanned_pages(
                sources,
                cache_dir=None,
            )

        self.assertEqual(len(nvidia_calls), 3)
        self.assertEqual(len(openrouter_calls), 1)
        self.assertEqual(
            openrouter_calls[0]["model"],
            "nvidia/vision:free",
        )
        self.assertTrue(openrouter_calls[0]["images"])
        self.assertEqual(page["transcript"], "Exact page text.")
        self.assertEqual(transcripts[0].text, "Exact page text.")
        metadata = [call.kwargs["metadata"] for call in logged.call_args_list]
        self.assertEqual(metadata[0]["provider"], "nvidia")
        self.assertTrue(metadata[0]["error"])
        self.assertEqual(metadata[-1]["provider"], "openrouter")
        self.assertTrue(metadata[-1]["fallback"])
        self.assertIn("nvidia transcription failed", metadata[-1]["fallback_reason"])

    def test_openrouter_bare_transcript_strings_are_normalized(self):
        page = {
            "page_number": 2,
            "text": "",
            "has_native_text": False,
            "needs_vision": True,
            "image_bytes": b"fake-png",
            "image_mime": "image/png",
        }
        sources = [{"label": "scan.pdf", "path": "scan.pdf", "pages": {2: page}}]

        with patch("extractor.load_local_env"), patch(
            "extractor.get_api_key", return_value=None
        ), patch(
            "extractor.get_nvidia_api_key", return_value=None
        ), patch(
            "extractor.get_openrouter_api_key", return_value="openrouter-key"
        ), patch(
            "extractor._call_openrouter_chat",
            return_value=json.dumps(
                {"page_transcripts": ["Bare transcript text."]}
            ),
        ), patch(
            "extractor.append_llm_log"
        ):
            transcripts = transcribe_scanned_pages(sources, cache_dir=None)

        self.assertEqual(page["transcript"], "Bare transcript text.")
        self.assertEqual(transcripts[0].source, "scan.pdf")
        self.assertEqual(transcripts[0].page_number, 2)

    def test_openrouter_alone_can_read_scanned_pages(self):
        page = {
            "page_number": 2,
            "text": "",
            "has_native_text": False,
            "needs_vision": True,
            "image_bytes": b"fake-png",
            "image_mime": "image/png",
        }
        sources = [{"label": "scan.pdf", "path": "scan.pdf", "pages": {2: page}}]
        calls = []

        def openrouter_success(**kwargs):
            calls.append(kwargs)
            return json.dumps(
                {
                    "page_transcripts": [
                        {
                            "source": "scan.pdf",
                            "page_number": 2,
                            "text": "Scanned page text.",
                        }
                    ]
                }
            )

        with patch("extractor.load_local_env"), patch(
            "extractor.get_api_key", return_value=None
        ), patch(
            "extractor.get_nvidia_api_key", return_value=None
        ), patch(
            "extractor.get_openrouter_api_key", return_value="openrouter-key"
        ), patch(
            "extractor.get_openrouter_vision_model",
            return_value="nvidia/vision:free",
        ), patch(
            "extractor._call_openrouter_chat", side_effect=openrouter_success
        ), patch(
            "extractor.append_llm_log"
        ):
            transcripts = transcribe_scanned_pages(sources, cache_dir=None)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "nvidia/vision:free")
        self.assertTrue(calls[0]["images"])
        self.assertEqual(page["transcript"], "Scanned page text.")
        self.assertEqual(transcripts[0].page_number, 2)

    def test_parse_structured_json_repairs_truncated_json(self):
        truncated = '{"page_transcripts": [{"source": "test.pdf", "page_number": 4, "text": "SOLIQUA weight neutral'
        parsed = parse_structured_json(truncated)
        self.assertEqual(parsed["page_transcripts"][0]["source"], "test.pdf")
        self.assertEqual(parsed["page_transcripts"][0]["page_number"], 4)
        self.assertEqual(parsed["page_transcripts"][0]["text"], "SOLIQUA weight neutral")


if __name__ == "__main__":
    unittest.main()
