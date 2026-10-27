"""Evaluation against the audited gold set.

A missing prediction is not a prediction. It is reported as ``missing`` and
never earns credit, even when the gold verdict is ``cannot_verify``. The report
compares verdicts, checks, evidence pages, evidence spans and computed
operands, and includes a trivial always-overstated baseline.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from verification_engine import normalize_text

VALID_VERDICTS = ["supported", "overstated", "cannot_verify"]


def _tokens(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalize_text(str(text or "")))


def _numeric_values(text: Any) -> List[float]:
    return [
        abs(float(value))
        for value in re.findall(r"-?\d+(?:\.\d+)?", normalize_text(str(text or "")))
    ]


def _span_containment(predicted: Any, gold: Any) -> bool:
    predicted_text = normalize_text(str(predicted or ""))
    gold_text = normalize_text(str(gold or ""))
    if not predicted_text or not gold_text:
        return not predicted_text and not gold_text
    return predicted_text in gold_text or gold_text in predicted_text


def _span_similarity(predicted: Any, gold: Any) -> float:
    predicted_tokens = set(_tokens(predicted))
    gold_tokens = set(_tokens(gold))
    if not predicted_tokens or not gold_tokens:
        return 0.0
    return len(predicted_tokens & gold_tokens) / float(len(predicted_tokens | gold_tokens))


def _operand_coverage(predicted: Any, gold: Any) -> Optional[float]:
    if not gold:
        return 1.0 if not predicted else 0.0
    predicted_values = [abs(float(value)) for value in (predicted or [])]
    gold_values = [abs(float(value)) for value in gold]
    matched = sum(
        1
        for gold_value in gold_values
        if any(abs(gold_value - value) <= 0.01 for value in predicted_values)
    )
    return matched / float(len(gold_values))


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pred_map = {
        item["claim_id"]: item
        for item in predictions
        if item.get("claim_id")
    }
    gold_map = {
        item["claim_id"]: item
        for item in gold
        if item.get("claim_id")
    }

    total = len(gold_map)
    correct = 0
    present = 0
    false_supported_claims: List[str] = []
    false_overstated_claims: List[str] = []
    false_cannot_verify_claims: List[str] = []
    missing_claims: List[str] = []
    checks_match_count = 0
    evidence_page_match_count = 0
    evidence_span_containment_count = 0
    evidence_span_similarities: List[float] = []
    computed_match_count = 0
    operand_coverages: List[float] = []

    confusion_matrix = {
        actual: {pred: 0 for pred in VALID_VERDICTS}
        for actual in VALID_VERDICTS
    }

    per_claim_results = []

    for claim_id, gold_item in gold_map.items():
        gold_verdict = gold_item.get("verdict")
        actual = gold_verdict if gold_verdict in VALID_VERDICTS else "cannot_verify"
        prediction_item = pred_map.get(claim_id)
        raw_prediction = prediction_item.get("verdict") if prediction_item else None
        if raw_prediction not in VALID_VERDICTS:
            missing_claims.append(claim_id)
            per_claim_results.append({
                "claim_id": claim_id,
                "gold": actual,
                "prediction": None,
                "match": False,
            })
            continue

        present += 1
        prediction = raw_prediction
        confusion_matrix[actual][prediction] += 1
        is_match = actual == prediction
        if is_match:
            correct += 1
        if prediction == "supported" and actual != "supported":
            false_supported_claims.append(claim_id)
        if prediction == "overstated" and actual != "overstated":
            false_overstated_claims.append(claim_id)
        if prediction == "cannot_verify" and actual != "cannot_verify":
            false_cannot_verify_claims.append(claim_id)

        gold_checks = gold_item.get("checks") or {}
        predicted_checks = prediction_item.get("checks") or {}
        checks_match = bool(gold_checks) and all(
            predicted_checks.get(name) == value
            for name, value in gold_checks.items()
        )
        if checks_match:
            checks_match_count += 1

        evidence_page_match = (
            prediction_item.get("evidence_page") == gold_item.get("evidence_page")
        )
        if evidence_page_match:
            evidence_page_match_count += 1

        span_containment = _span_containment(
            prediction_item.get("evidence_span"),
            gold_item.get("evidence_span"),
        )
        if span_containment:
            evidence_span_containment_count += 1
        similarity = _span_similarity(
            prediction_item.get("evidence_span"),
            gold_item.get("evidence_span"),
        )
        evidence_span_similarities.append(similarity)

        gold_computed = gold_item.get("computed")
        predicted_computed = prediction_item.get("computed")
        if gold_computed is None:
            operand_coverage = 1.0 if predicted_computed is None else 0.0
        else:
            operand_coverage = _operand_coverage(
                (predicted_computed or {}).get("from_source"),
                gold_computed.get("from_source"),
            )
        operand_coverages.append(operand_coverage or 0.0)
        computed_match = (
            gold_computed is None and predicted_computed is None
        ) or (
            gold_computed is not None
            and predicted_computed is not None
            and operand_coverage == 1.0
        )
        if computed_match:
            computed_match_count += 1

        per_claim_results.append({
            "claim_id": claim_id,
            "gold": actual,
            "prediction": prediction,
            "match": is_match,
            "checks_match": checks_match,
            "evidence_page_match": evidence_page_match,
            "evidence_span_containment": span_containment,
            "evidence_span_similarity": round(similarity, 3),
            "computed_operand_coverage": round(operand_coverage or 0.0, 3),
        })

    accuracy = correct / total if total else 0.0
    coverage = present / total if total else 0.0

    supported_gold = [cid for cid, item in gold_map.items() if item.get("verdict") == "supported"]
    supported_pred = [
        cid for cid, item in pred_map.items()
        if item.get("verdict") == "supported"
    ]
    supported_true_positive = len(set(supported_gold) & set(supported_pred))
    supported_precision = (
        supported_true_positive / len(supported_pred) if supported_pred else 0.0
    )
    supported_recall = (
        supported_true_positive / len(supported_gold) if supported_gold else 0.0
    )

    baseline_overstated_correct = sum(
        1 for item in gold_map.values() if item.get("verdict") == "overstated"
    )
    baseline_overstated_accuracy = (
        baseline_overstated_correct / total if total else 0.0
    )

    return {
        "total": total,
        "present": present,
        "correct": correct,
        "accuracy": accuracy,
        "coverage": coverage,
        "missing_claims": missing_claims,
        "false_supported_count": len(false_supported_claims),
        "false_supported_claims": false_supported_claims,
        "false_overstated_count": len(false_overstated_claims),
        "false_overstated_claims": false_overstated_claims,
        "false_cannot_verify_count": len(false_cannot_verify_claims),
        "false_cannot_verify_claims": false_cannot_verify_claims,
        "supported_precision": supported_precision,
        "supported_recall": supported_recall,
        "checks_match_count": checks_match_count,
        "checks_match_rate": checks_match_count / present if present else 0.0,
        "evidence_page_match_count": evidence_page_match_count,
        "evidence_page_match_rate": evidence_page_match_count / present if present else 0.0,
        "evidence_span_containment_count": evidence_span_containment_count,
        "evidence_span_containment_rate": (
            evidence_span_containment_count / present if present else 0.0
        ),
        "mean_evidence_span_similarity": (
            sum(evidence_span_similarities) / len(evidence_span_similarities)
            if evidence_span_similarities else 0.0
        ),
        "computed_match_count": computed_match_count,
        "computed_match_rate": computed_match_count / present if present else 0.0,
        "mean_computed_operand_coverage": (
            sum(operand_coverages) / len(operand_coverages)
            if operand_coverages else 0.0
        ),
        "trivial_baseline": {
            "name": "always_overstated",
            "accuracy": baseline_overstated_accuracy,
            "false_supported_count": 0,
            "false_overstated_count": len(supported_gold),
        },
        "confusion_matrix": confusion_matrix,
        "per_claim": per_claim_results,
    }


def summarize_audit_log(log_path: str) -> Dict[str, Any]:
    """Describe a JSONL model log without interpreting its contents."""

    summary: Dict[str, Any] = {
        "requests": 0,
        "extraction_success": 0,
        "transcription_success": 0,
        "failures": 0,
        "models": [],
        "successful_claim_ids": [],
    }
    if not os.path.isfile(log_path):
        return summary

    models = set()
    claims = set()
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            summary["requests"] += 1
            models.add(row.get("model", ""))
            metadata = row.get("metadata") or {}
            task = metadata.get("task", "extraction")
            if metadata.get("error"):
                summary["failures"] += 1
                continue
            if task == "transcription":
                summary["transcription_success"] += 1
            else:
                raw = row.get("response_raw", "")
                if raw.startswith("{"):
                    summary["extraction_success"] += 1
                    if metadata.get("claim_id"):
                        claims.add(metadata["claim_id"])
    summary["models"] = sorted(model for model in models if model)
    summary["successful_claim_ids"] = sorted(claims)
    return summary


def format_confusion_matrix(matrix: Dict[str, Dict[str, int]]) -> str:
    lines = [
        "               Predicted",
        "Actual         supported  overstated  cannot_verify",
        "---------------------------------------------------",
    ]
    for actual in VALID_VERDICTS:
        row = f"{actual:<14} "
        for pred in VALID_VERDICTS:
            row += f"{matrix[actual][pred]:^11} "
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/predictions.json"
    gold_path = sys.argv[2] if len(sys.argv) > 2 else "data/gold.json"
    log_path = sys.argv[3] if len(sys.argv) > 3 else "outputs/llm_log.jsonl"

    if not os.path.exists(gold_path):
        print(f"Error: Gold dataset not found at {gold_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(pred_path):
        print(f"Error: Predictions file not found at {pred_path}", file=sys.stderr)
        sys.exit(1)

    with open(gold_path, "r", encoding="utf-8") as handle:
        gold = json.load(handle)
    with open(pred_path, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    results = evaluate_predictions(predictions, gold)
    audit = summarize_audit_log(log_path)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total gold claims          : {results['total']}")
    print(f"Predictions present        : {results['present']}")
    print(f"Prediction file coverage   : {results['coverage']:.1%}")
    print(f"Correct verdicts           : {results['correct']}")
    print(f"Verdict accuracy           : {results['accuracy']:.1%}")
    print(f"Supported precision/recall : {results['supported_precision']:.1%} / "
          f"{results['supported_recall']:.1%}")
    print("----------------------------------------------------")
    print(f"Checks exactly matching    : {results['checks_match_count']}/{results['present']}")
    print(f"Evidence page matches      : {results['evidence_page_match_count']}/{results['present']}")
    print(f"Span containment           : {results['evidence_span_containment_count']}/{results['present']}")
    print(f"Mean span token similarity : {results['mean_evidence_span_similarity']:.3f}")
    print(f"Computed operands matching : {results['computed_match_count']}/{results['present']}")
    print("----------------------------------------------------")
    print(f"False supported            : {results['false_supported_count']}")
    if results["false_supported_claims"]:
        print(f"  WARNING: {', '.join(results['false_supported_claims'])}")
    print(f"False overstated           : {results['false_overstated_count']}")
    if results["false_overstated_claims"]:
        print(f"  Claims: {', '.join(results['false_overstated_claims'])}")
    print(f"False cannot_verify        : {results['false_cannot_verify_count']}")
    if results["false_cannot_verify_claims"]:
        print(f"  Claims: {', '.join(results['false_cannot_verify_claims'])}")
    if results["missing_claims"]:
        print(f"Missing predictions        : {', '.join(results['missing_claims'])}")
    print("----------------------------------------------------")
    print("Trivial baseline (always overstated):")
    print(f"  Accuracy                 : {results['trivial_baseline']['accuracy']:.1%}")
    print(f"  False supported          : {results['trivial_baseline']['false_supported_count']}")
    print(f"  False overstated         : {results['trivial_baseline']['false_overstated_count']}")
    print("----------------------------------------------------")
    print(f"Audit log                  : {log_path}")
    print(f"  Requests                 : {audit['requests']}")
    print(f"  Extraction successes     : {audit['extraction_success']}")
    print(f"  Transcription successes  : {audit['transcription_success']}")
    print(f"  Fail-closed failures     : {audit['failures']}")
    print(f"  Models                   : {', '.join(audit['models']) or 'none'}")
    print("----------------------------------------------------")
    print("NOTE: Seven development claims are a fixed regression set, not a")
    print("held-out accuracy estimate.")
    print("----------------------------------------------------")
    print("Confusion Matrix:")
    print(format_confusion_matrix(results["confusion_matrix"]))
    print("====================================================\n")

    if results["false_supported_count"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
