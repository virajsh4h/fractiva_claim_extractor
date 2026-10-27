"""End-to-end claim verification pipeline.

The CLI is intentionally thin. ``ClaimVerifier`` owns source resolution,
ingestion, extraction, span location and verdict resolution. Dependencies can
be replaced at that seam for deterministic tests.
"""

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from domain import CheckStatus, Verdict
from extractor import (
    ArithmeticExtraction,
    ClaimExtractionResponse,
    _validation_issues,
    extract_claim_evidence,
    fail_closed_extraction,
    parse_structured_json,
    transcribe_scanned_pages,
)
from ingest import extract_pdf_pages, resolve_source_refs
from verification_engine import (
    ArithmeticComponent,
    ArithmeticSpec,
    build_reason,
    check_span_presence,
    claim_direction,
    claim_has_quantitative_assertion,
    deterministic_figure_reason,
    figure_coverage_gaps,
    find_limitation_evidence,
    limitation_hits,
    locate_evidence,
    numeric_qualifiers_match,
    page_evidence_text,
    resolve_verdict,
    span_supports_arithmetic,
    supported_reason,
    verify_arithmetic,
    _display_number,
)


def make_fail_closed_prediction(
    claim_id: str,
    reason: str,
    evidence_page: Optional[int] = None,
    evidence_span: Optional[str] = None,
    claim_text: Optional[str] = None,
) -> Dict[str, Any]:
    figure_status = None
    if claim_text and claim_has_quantitative_assertion(claim_text):
        figure_status = CheckStatus.UNCLEAR.value

    return {
        "claim_id": claim_id,
        "verdict": Verdict.CANNOT_VERIFY.value,
        "checks": {
            "figure": figure_status,
            "population": CheckStatus.UNCLEAR.value,
            "drug_indication": CheckStatus.UNCLEAR.value,
            "actually_stated": CheckStatus.UNCLEAR.value,
        },
        "evidence_page": evidence_page,
        "evidence_span": evidence_span,
        "computed": None,
        "reason": reason,
    }


def _source_key(label: str) -> str:
    return os.path.basename(label or "").strip().lower()


def _match_source(label: Optional[str], sources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not label:
        return None
    wanted = _source_key(label)
    for source in sources:
        if _source_key(source["label"]) == wanted:
            return source
    for source in sources:
        if wanted and (wanted in _source_key(source["label"]) or _source_key(source["label"]) in wanted):
            return source
    return None


def load_logged_extractions(
    log_path: str,
) -> tuple:
    """Return the last successful extraction for each claim in an audit log."""

    responses: Dict[str, ClaimExtractionResponse] = {}
    models: Dict[str, str] = {}
    if not os.path.isfile(log_path):
        return responses, models

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            metadata = row.get("metadata") or {}
            if metadata.get("error") or metadata.get("task") == "transcription":
                continue
            claim_id = metadata.get("claim_id")
            raw = row.get("response_raw", "")
            if not claim_id or not raw:
                continue
            try:
                payload = parse_structured_json(raw)
                if isinstance(payload, dict):
                    responses[claim_id] = ClaimExtractionResponse(**payload)
            except Exception:
                continue
            models[claim_id] = row.get("model", "")
    return responses, models


def load_logged_transcripts(log_path: str) -> Dict[tuple, str]:
    """Index every successful page transcription in an audit log."""

    transcripts: Dict[tuple, str] = {}
    if not os.path.isfile(log_path):
        return transcripts

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            metadata = row.get("metadata") or {}
            if metadata.get("task") != "transcription" or metadata.get("error"):
                continue
            try:
                payload = parse_structured_json(row.get("response_raw", ""))
            except Exception:
                continue
            if isinstance(payload, dict):
                for entry in payload.get("page_transcripts", []):
                    if isinstance(entry, dict):
                        label = _source_key(entry.get("source", ""))
                        page_number = entry.get("page_number")
                        text = entry.get("text", "")
                        if label and isinstance(page_number, int) and text.strip():
                            transcripts[(label, page_number)] = text
    return transcripts


def replay_predictions(
    claims_file: str,
    log_file: str,
    output_file: str = "outputs/predictions.json",
    base_dir: str = ".",
) -> List[Dict[str, Any]]:
    """Rebuild predictions from logged responses without calling a model.

    This makes the arithmetic and span checks auditable: a reviewer can rerun
    the deterministic half of the pipeline against the exact model responses in
    ``outputs/llm_log.jsonl``.
    """

    responses, _ = load_logged_extractions(log_file)
    transcripts = load_logged_transcripts(log_file)

    def extractor(
        claim: Dict[str, Any],
        sources: Optional[List[Dict[str, Any]]] = None,
        **_: Any,
    ) -> ClaimExtractionResponse:
        response = responses.get(claim.get("claim_id", ""))
        if response is None:
            return fail_closed_extraction(
                "No successful extraction for this claim exists in the audit log."
            )
        if sources:
            issues = _validation_issues(response, claim, sources)
            if issues:
                return fail_closed_extraction(
                    "Extraction still failed deterministic validation after one repair: "
                    + "; ".join(issues[:3])
                )
        return response

    def transcriber(sources: List[Dict[str, Any]], **_: Any) -> List[Any]:
        for source in sources:
            label = _source_key(source.get("label", ""))
            for page_number, page in source.get("pages", {}).items():
                if not page.get("needs_vision"):
                    continue
                text = transcripts.get((label, page_number))
                if text:
                    page["transcript"] = text
        return []

    verifier = ClaimVerifier(
        base_dir=base_dir,
        extractor=extractor,
        transcriber=transcriber,
    )
    return verify_claims_pipeline(
        claims_file,
        output_file,
        base_dir=base_dir,
        verifier=verifier,
    )


class ClaimVerifier:
    def __init__(
        self,
        base_dir: str = ".",
        model_name: Optional[str] = None,
        log_path: Optional[str] = None,
        extractor: Optional[Callable[..., ClaimExtractionResponse]] = None,
        transcriber: Optional[Callable[..., List[Any]]] = None,
    ):
        self.base_dir = base_dir
        self.model_name = model_name
        self.log_path = log_path or os.path.join(base_dir, "outputs", "llm_log.jsonl")
        self.extractor = extractor or extract_claim_evidence
        self.transcriber = transcriber or transcribe_scanned_pages

    def _load_sources(self, claim: Dict[str, Any]) -> tuple:
        refs = resolve_source_refs(claim.get("cited_paper", ""), base_dir=self.base_dir)
        sources: List[Dict[str, Any]] = []
        missing: List[str] = []

        for ref in refs:
            if not ref.path or not os.path.isfile(ref.path):
                missing.append(ref.raw)
                continue
            try:
                pages = extract_pdf_pages(ref.path, source_label=ref.label)
            except Exception as exc:
                missing.append(f"{ref.raw}: {str(exc)[:80]}")
                continue
            sources.append({"label": ref.label, "path": ref.path, "pages": pages})

        return sources, missing

    def _inject_transcripts(
        self,
        extraction: ClaimExtractionResponse,
        sources: List[Dict[str, Any]],
    ) -> None:
        for transcript in extraction.page_transcripts:
            source = _match_source(transcript.source, sources)
            if not source:
                continue
            page = source["pages"].get(transcript.page_number)
            if page is None or not transcript.text.strip():
                continue
            # Native text is authoritative. A page that was already transcribed
            # before claim extraction is frozen: the judging model may not
            # rewrite the evidence it is judged against.
            if page.get("has_native_text") and not page.get("needs_vision"):
                continue
            if page.get("transcript"):
                continue
            page["transcript"] = transcript.text

    def _spec_from_extraction(
        self,
        arithmetic: ArithmeticExtraction,
        claim_text: str,
    ) -> ArithmeticSpec:
        direction = claim_direction(claim_text)
        components = [
            ArithmeticComponent(
                operation=component.operation,
                claimed_numeric=component.claimed_numeric,
                source_operands=list(component.source_operands),
                claimed_text=component.claimed_text,
                reported_value=component.reported_value,
                direction=direction,
            )
            for component in arithmetic.components
        ]
        return ArithmeticSpec(
            operation=arithmetic.operation,
            claimed_text=arithmetic.claimed_text or claim_text,
            claimed_numeric=arithmetic.claimed_numeric,
            source_operands=list(arithmetic.source_operands),
            claimed_values=list(arithmetic.claimed_values),
            reported_value=arithmetic.reported_value,
            components=components,
            direction=direction,
        )

    def verify(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        claim_id = claim.get("claim_id", "UNKNOWN")
        claim_text = claim.get("text", "") or ""
        sources, missing = self._load_sources(claim)

        if not sources:
            reason = "Cited source not found or could not be opened."
            if missing:
                reason += " Missing: " + "; ".join(missing)
            return make_fail_closed_prediction(
                claim_id=claim_id,
                reason=reason,
                claim_text=claim_text,
            )

        try:
            # Image-only pages get an independent transcription before the
            # claim-specific extraction. The transcript is the text that the
            # span checker will use, and extraction may not overwrite it.
            self.transcriber(
                sources,
                model_name=self.model_name,
                log_path=self.log_path,
            )
            extraction = self.extractor(
                claim=claim,
                sources=sources,
                model_name=self.model_name,
                log_path=self.log_path,
            )
        except Exception as exc:
            reason = f"Extraction failed: {str(exc)[:100]}. Fails closed to cannot_verify."
            if missing:
                reason += " Missing source(s): " + "; ".join(missing)
            return make_fail_closed_prediction(
                claim_id=claim_id,
                reason=reason,
                claim_text=claim_text,
            )

        self._inject_transcripts(extraction, sources)
        pages_by_source = {source["label"]: source["pages"] for source in sources}

        location = None
        if extraction.evidence_span:
            preferred = _match_source(extraction.evidence_source, sources)
            location = locate_evidence(
                extraction.evidence_span,
                pages_by_source,
                preferred_source=preferred["label"] if preferred else None,
                preferred_page=extraction.evidence_page,
            )

        if location:
            evidence_source = location.source
            evidence_page = location.page
            span_verified = True
        else:
            evidence_source = extraction.evidence_source
            evidence_page = extraction.evidence_page
            span_verified = False

        arithmetic = extraction.arithmetic
        quantitative_claim = claim_has_quantitative_assertion(claim_text)
        spec = self._spec_from_extraction(arithmetic, claim_text)

        limitations_span: Optional[str] = None
        if "study_design" in set(extraction.mismatch_bases):
            preferred = _match_source(extraction.evidence_source, sources)
            limitation = find_limitation_evidence(
                pages_by_source,
                preferred_source=preferred["label"] if preferred else None,
                preferred_page=extraction.evidence_page,
            )
            if limitation:
                limitation_location, limitation_text = limitation
                if limitation_hits(limitation_text) > limitation_hits(
                    extraction.evidence_span or ""
                ):
                    location = limitation_location
                    extraction.evidence_span = limitation_text
                    extraction.evidence_source = limitation_location.source
                    extraction.evidence_page = limitation_location.page
                    evidence_source = limitation_location.source
                    evidence_page = limitation_location.page
                    span_verified = True
                    limitations_span = limitation_text

        page_text = ""
        if location:
            page_record = pages_by_source.get(location.source, {}).get(location.page, {})
            page_text = page_evidence_text(page_record)

        if quantitative_claim:
            arith_status, comp_dict = verify_arithmetic(spec)
            figure_check: Optional[str] = (
                None if arith_status == "not_applicable" else arith_status
            )
            if figure_check is None:
                figure_check = CheckStatus.UNCLEAR.value
                comp_dict = dict(comp_dict)
                comp_dict["result"] = (
                    "claim contains a numeric endpoint but no arithmetic was extracted"
                )
        else:
            # A qualitative claim may still carry decisive numbers. They are
            # context for the reviewer and never create a figure check.
            if (
                arithmetic.operation != "none"
                and (arithmetic.source_operands or arithmetic.reported_value is not None)
            ):
                _, comp_dict = verify_arithmetic(spec)
            elif extraction.supporting_values:
                comp_dict = {
                    "claimed": claim_text,
                    "from_source": list(extraction.supporting_values),
                    "result": "Source values: "
                    + ", ".join(_display_number(float(value)) for value in extraction.supporting_values),
                    "components": [],
                }
            else:
                comp_dict = None
            figure_check = None

        if quantitative_claim and figure_check is not None:
            gaps = figure_coverage_gaps(claim_text, spec)
            if gaps:
                figure_check = CheckStatus.UNCLEAR.value
                comp_dict = dict(comp_dict or {})
                comp_dict["result"] = (
                    f"{comp_dict.get('result', '')} | unaccounted claim figure(s): "
                    + ", ".join(gaps)
                ).strip(" |")

        if (
            quantitative_claim
            and figure_check is not None
            and span_verified
            and extraction.evidence_span
            and not span_supports_arithmetic(
                extraction.evidence_span,
                spec,
                page_text,
                figure_check,
            )
        ):
            figure_check = CheckStatus.UNCLEAR.value
            comp_dict = dict(comp_dict)
            comp_dict["result"] = (
                f"{comp_dict.get('result', '')} | evidence span does not contain the numbers used"
            ).strip(" |")

        bases = [basis for basis in extraction.mismatch_bases if basis != "none"]

        # A model can leave a numeric mismatch behind after the repaired
        # arithmetic reconciles. If "numeric" is the only stated basis, Python
        # has matched the figure, and the claimed CI and p-value occur in the
        # span, the numeric objection is resolved. This is a narrow consistency
        # rule, not a semantic override of any other mismatch.
        if (
            extraction.actually_stated_check == CheckStatus.MISMATCH.value
            and extraction.population_check == CheckStatus.MATCH.value
            and extraction.drug_indication_check == CheckStatus.MATCH.value
            and set(bases) == {"numeric"}
            and figure_check == CheckStatus.MATCH.value
            and span_verified
            and extraction.evidence_span
            and numeric_qualifiers_match(claim_text, extraction.evidence_span)
        ):
            extraction.actually_stated_check = CheckStatus.MATCH.value
            bases = []

        checks = {
            "figure": figure_check,
            "population": extraction.population_check,
            "drug_indication": extraction.drug_indication_check,
            "actually_stated": extraction.actually_stated_check,
        }

        # A mismatch basis with no mismatch check is a contradictory extraction.
        # It cannot support a claim.
        inconsistent = bool(bases) and not any(
            value == CheckStatus.MISMATCH.value for value in checks.values()
        )

        verdict = resolve_verdict(checks, span_verified=span_verified)
        if inconsistent:
            verdict = Verdict.CANNOT_VERIFY.value

        # A claim that asserts evidence across several sources is not fully
        # supported if one of those sources is missing. A contradiction in an
        # available source can still justify "overstated".
        if missing and verdict == Verdict.SUPPORTED.value:
            verdict = Verdict.CANNOT_VERIFY.value

        if span_verified and verdict == Verdict.OVERSTATED.value:
            reason = build_reason(
                claim_text=claim_text,
                checks=checks,
                bases=bases,
                computed=comp_dict,
                limitation_span=limitations_span,
                supporting_values=extraction.supporting_values,
            )
        elif span_verified and verdict == Verdict.SUPPORTED.value:
            reason = supported_reason(claim_text, comp_dict)
        else:
            reason_parts: List[str] = []
            if inconsistent:
                reason_parts.append(
                    "Contradictory extraction: at least one mismatch basis was "
                    "listed while every check was marked match."
                )
            if not span_verified:
                reason_parts.append(
                    "The quoted evidence span was not found verbatim on any supplied page."
                )
            if figure_check == CheckStatus.UNCLEAR.value:
                reason_parts.append(
                    "The arithmetic could not be verified"
                    + (
                        f": {comp_dict.get('result')}"
                        if comp_dict and comp_dict.get("result")
                        else "."
                    )
                )
            unclear_checks = [
                name
                for name, value in checks.items()
                if value == CheckStatus.UNCLEAR.value
            ]
            if unclear_checks:
                reason_parts.append(
                    "Unresolved check(s): " + ", ".join(unclear_checks) + "."
                )
            if not reason_parts:
                reason_parts.append("The claim could not be verified from the supplied source.")
            reason = " ".join(reason_parts)

        failure_reason = getattr(extraction, "_failure_reason", None)
        if failure_reason and verdict == Verdict.CANNOT_VERIFY.value:
            reason = f"{failure_reason} {reason}".strip()

        if len(sources) > 1 and evidence_source and evidence_page:
            reason = f"Source: {evidence_source}, page {evidence_page}. {reason}"
        if missing:
            reason = f"{reason} Missing source(s): {'; '.join(missing)}."

        computed = comp_dict if (figure_check is not None or comp_dict) else None

        return {
            "claim_id": claim_id,
            "verdict": verdict,
            "checks": checks,
            "evidence_page": evidence_page,
            "evidence_span": extraction.evidence_span,
            "computed": computed,
            "reason": reason,
        }


def verify_single_claim(claim: Dict[str, Any], base_dir: str = ".") -> Dict[str, Any]:
    return ClaimVerifier(base_dir=base_dir).verify(claim)


def verify_claims_pipeline(
    claims_file: str,
    output_file: str = "outputs/predictions.json",
    base_dir: str = ".",
    verifier: Optional[ClaimVerifier] = None,
) -> List[Dict[str, Any]]:
    if not os.path.isfile(claims_file):
        raise FileNotFoundError(f"Claims input file not found: {claims_file}")

    with open(claims_file, "r", encoding="utf-8") as handle:
        raw_claims = json.load(handle)

    if isinstance(raw_claims, list):
        claims = raw_claims
    elif isinstance(raw_claims, dict):
        if "claims" in raw_claims and isinstance(raw_claims["claims"], list):
            claims = raw_claims["claims"]
        else:
            claims = [raw_claims]
    else:
        raise ValueError(f"Unexpected JSON structure in {claims_file}")

    pipeline = verifier or ClaimVerifier(base_dir=base_dir)
    predictions: List[Dict[str, Any]] = []
    print(f"--> Verifying {len(claims)} claims from {claims_file}...")

    for index, claim in enumerate(claims, 1):
        claim_id = claim.get("claim_id", f"#{index}")
        print(f"    [{index}/{len(claims)}] Processing {claim_id}...")
        prediction = pipeline.verify(claim)
        predictions.append(prediction)
        print(f"         Verdict: {prediction['verdict'].upper()}")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)

    print(f"--> Done! Predictions saved to {output_file}\n")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify clinical claims against cited sources.")
    parser.add_argument("claims", nargs="?", default="data/claims.json")
    parser.add_argument("output", nargs="?", default="outputs/predictions.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument(
        "--replay",
        default=None,
        help="Rebuild predictions from a JSONL audit log without calling a model.",
    )
    parser.add_argument("--claim", default=None, help="Verify a single claim string directly.")
    parser.add_argument("--paper", default=None, help="Cited paper for direct single claim verification.")
    parser.add_argument("--claim-id", default="NEW_CLAIM", help="Claim ID for direct single claim.")
    parser.add_argument("--population", default=None, help="Claimed population.")
    parser.add_argument("--molecule", default=None, help="Claimed molecule.")
    args = parser.parse_args()

    try:
        verifier = ClaimVerifier(base_dir=args.base_dir, model_name=args.model)
        if args.claim:
            single_claim = {
                "claim_id": args.claim_id,
                "text": args.claim,
                "cited_paper": args.paper or "",
                "population": args.population,
                "molecule": args.molecule,
            }
            pred = verifier.verify(single_claim)
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump([pred], handle, indent=2, ensure_ascii=False)
            print(json.dumps(pred, indent=2))
            return

        if args.replay:
            replay_predictions(
                args.claims,
                args.replay,
                args.output,
                base_dir=args.base_dir,
            )
            return

        verify_claims_pipeline(args.claims, args.output, base_dir=args.base_dir, verifier=verifier)
    except Exception as exc:
        print(f"Error during verification pipeline execution: {str(exc)}", file=sys.stderr)
        try:
            with open(args.claims, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, dict) and isinstance(raw.get("claims"), list):
                claims = raw["claims"]
            elif isinstance(raw, list):
                claims = raw
            else:
                claims = [raw]
            fallback = [
                make_fail_closed_prediction(
                    claim_id=claim.get("claim_id", "UNKNOWN"),
                    reason=f"Fatal pipeline failure: {str(exc)}",
                    claim_text=claim.get("text"),
                )
                for claim in claims
            ]
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(fallback, handle, indent=2)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
