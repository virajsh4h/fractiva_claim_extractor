"""Deterministic arithmetic, span validation and verdict resolution.

This module is the safety core. An absent number is never a match. A check that
cannot be computed returns ``unclear``; a check that contradicts the source
returns ``mismatch``. Only a complete set of computed matches can produce
``supported``.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from domain import CheckStatus, EvidenceLocation, Verdict


@dataclass
class ArithmeticComponent:
    operation: str
    claimed_numeric: Optional[float] = None
    source_operands: List[float] = field(default_factory=list)
    claimed_text: Optional[str] = None
    reported_value: Optional[float] = None
    direction: Optional[str] = None


@dataclass
class ArithmeticSpec:
    operation: str
    claimed_text: Optional[str] = None
    claimed_numeric: Optional[float] = None
    source_operands: List[float] = field(default_factory=list)
    claimed_values: List[float] = field(default_factory=list)
    reported_value: Optional[float] = None
    components: List[ArithmeticComponent] = field(default_factory=list)
    direction: Optional[str] = None


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Some PDF fonts encode a minus sign as the control character \x01.
    # Only treat it as a sign when it directly precedes a digit; elsewhere it
    # is a diacritic marker and must not become a dash.
    text = re.sub(r"\x01(?=\d)", "-", text)
    text = text.replace("\x01", "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    text = text.lower()
    text = text.replace("±", "+/-")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def confidence_levels(text: str) -> set:
    """Return confidence levels written as ``95% CI`` or ``95% confidence interval``."""

    return {
        float(match)
        for match in re.findall(
            r"(?<![\w.])(\d{2}(?:\.\d+)?)\s*%\s*(?:CI|confidence\s+interval)",
            normalize_text(text or ""),
            flags=re.IGNORECASE,
        )
    }


def numeric_qualifiers_match(claim_text: str, evidence_span: str) -> bool:
    """Check that a claimed confidence level and p-value appear in the span.

    Used only to reconcile one narrow contradiction: the model says the claim
    is not numerically stated, while Python has already verified the arithmetic.
    """

    claim_levels = confidence_levels(claim_text)
    if claim_levels and not claim_levels.issubset(confidence_levels(evidence_span)):
        return False

    comparison = r"(?:<|>|=|<=|>=)\s*\d+(?:\.\d+)?"
    claim_compact = re.sub(r"\s+", "", normalize_text(claim_text))
    span_compact = re.sub(r"\s+", "", normalize_text(evidence_span))
    claim_values = set(re.findall(comparison, claim_compact))
    span_values = set(re.findall(comparison, span_compact))
    if claim_values and not claim_values.issubset(span_values):
        return False
    return True


def _numeric_tokens(text: str) -> List[str]:
    return re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", normalize_text(text))


def _display_number(value: float) -> str:
    rounded = round(float(value), 6)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def claim_numeric_tokens(text: str) -> List[float]:
    """Return the absolute numbers a claim asserts, excluding p-values."""

    cleaned = re.sub(
        r"\bp\s*[<>=]\s*-?\d+(?:\.\d+)?",
        " ",
        normalize_text(text),
        flags=re.IGNORECASE,
    )
    return [abs(float(token)) for token in _numeric_tokens(cleaned)]


def comparison_anchor_numbers(text: str) -> List[float]:
    """Return the two percentage endpoints of a "X% vs Y%" comparison.

    Drug names often contain digits ("BIAsp 30") and confidence intervals use
    labels ("95% CI"). Only percentage values that sit immediately before and
    after the comparison word are evidence anchors.
    """

    normalized = normalize_text(text)
    if not normalized:
        return []
    before = re.search(
        r"(?<![\d.])(\d+(?:\.\d+)?)(?:\s*(?:\+/-)\s*\d+(?:\.\d+)?)?\s*%\s*\)?\s*(?:vs\.?|versus)",
        normalized,
        flags=re.IGNORECASE,
    )
    after = re.search(
        r"(?:vs\.?|versus)\s+[^.;]{0,60}?(?<![\d.])(\d+(?:\.\d+)?)"
        r"(?:\s*(?:\+/-)\s*\d+(?:\.\d+)?)?\s*%",
        normalized,
        flags=re.IGNORECASE,
    )
    if not before or not after:
        return []
    return [abs(float(before.group(1))), abs(float(after.group(1)))]


def span_covers_numbers(span: str, numbers: Sequence[float]) -> bool:
    """Whether a quote contains every number, within display tolerance."""

    if not numbers:
        return True
    span_numbers = [abs(float(token)) for token in _numeric_tokens(span or "")]
    return all(
        any(abs(value - candidate) <= 0.011 for candidate in span_numbers)
        for value in numbers
    )


def _compact(text: str) -> str:
    """Remove whitespace but preserve word and number order.

    PDF extraction inserts line breaks in the middle of a sentence and sometimes
    after a hyphen. Comparing compacted strings still requires a contiguous
    passage: every character must occur in the same order.
    """

    return re.sub(r"\s+", "", normalize_text(text))


def page_evidence_text(page_info: Dict[str, Any]) -> str:
    """Return the searchable text for one page.

    A frozen transcript is authoritative for a scanned page. Native text and
    extracted tables are combined only when no transcript exists, so a model
    may quote a table cell without bypassing the transcript freeze.
    """

    transcript = page_info.get("transcript")
    if transcript:
        return transcript
    text = page_info.get("text", "") or ""
    tables = page_info.get("tables") or []
    if tables:
        return text + "\n" + "\n".join(tables)
    return text


def check_span_presence(evidence_span: str, page_text: str) -> bool:
    """Accept only a contiguous verbatim passage from the page.

    Case, whitespace, ligatures and dash variants are normalized. Word order is
    never ignored, and an ellipsis is refused because it is an authorial join.
    """

    if not evidence_span or not page_text:
        return False
    if "..." in evidence_span or "\u2026" in evidence_span:
        return False

    norm_span = normalize_text(evidence_span)
    norm_page = normalize_text(page_text)
    if not norm_span:
        return False
    if norm_span in norm_page:
        return True

    compact_span = _compact(evidence_span)
    compact_page = _compact(page_text)
    return len(compact_span) >= 8 and compact_span in compact_page


def validate_evidence_span(
    evidence_span: Optional[str],
    evidence_page: Optional[int],
    pages_data: Dict[int, Dict[str, Any]],
) -> bool:
    if not evidence_span or not evidence_page or evidence_page not in pages_data:
        return False
    page_info = pages_data[evidence_page]
    page_text = page_evidence_text(page_info)
    if not page_text:
        return False
    return check_span_presence(evidence_span, page_text)


def locate_evidence(
    evidence_span: str,
    pages_by_source: Dict[str, Dict[int, Dict[str, Any]]],
    preferred_source: Optional[str] = None,
    preferred_page: Optional[int] = None,
) -> Optional[EvidenceLocation]:
    """Find a span in any ingested source.

    The extraction's own source and page are preferred when the quote really
    occurs there, so the recorded provenance matches the model's reasoning.
    Otherwise the span is searched in document order across every source.
    """

    if (
        preferred_source in pages_by_source
        and preferred_page
        and validate_evidence_span(
            evidence_span,
            preferred_page,
            pages_by_source[preferred_source],
        )
    ):
        return EvidenceLocation(source=preferred_source, page=preferred_page)

    matches: List[EvidenceLocation] = []
    for source, pages in pages_by_source.items():
        for page_number in sorted(pages):
            if validate_evidence_span(evidence_span, page_number, pages):
                matches.append(EvidenceLocation(source=source, page=page_number))
    if not matches:
        return None
    return matches[0]


LIMITATION_MARKERS = (
    "post hoc",
    "post-hoc",
    "posthoc",
    "hypothesis generating",
    "hypothesis-generating",
    "not randomized",
    "not an independent trial",
    "exploratory",
    "ad hoc",
    "retrospective",
)

_LIMITATION_CONTEXT = (
    "prior ",
    "previous ",
    "earlier ",
    "other studies",
    "literature",
    "we conducted",
    "in contrast to",
)


def _limitation_hits(text: str) -> int:
    lowered = (text or "").lower()
    return sum(1 for marker in LIMITATION_MARKERS if marker in lowered)


def limitation_hits(text: str) -> int:
    """Public count used to compare two candidate limitation quotes."""

    return _limitation_hits(text)


def _sentences(text: str) -> List[str]:
    protected = re.sub(
        r"\b(?:et al|e\.g|i\.e|vs|Dr|Fig|No|p)\.(?=\s|$)",
        lambda match: match.group(0).replace(".", "\u0000"),
        text or "",
        flags=re.IGNORECASE,
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", protected)
    return [part.replace("\u0000", ".").strip() for part in parts if part.strip()]


def find_limitation_evidence(
    pages_by_source: Dict[str, Dict[int, Dict[str, Any]]],
    preferred_source: Optional[str] = None,
    preferred_page: Optional[int] = None,
) -> Optional[Tuple[EvidenceLocation, str]]:
    """Find the source's own limitation sentence for a study-design mismatch.

    A model can correctly mark a post-hoc or exploratory result as overstated
    while quoting a descriptive sentence instead of the limitation. This search
    is generic: it looks only for the source's language about study design and
    returns a real sentence from the cited page. It never decides the verdict.

    Sentences that describe earlier literature ("prior retrospective studies")
    are ignored unless they carry at least three markers. The preferred source
    and page are ranked first.
    """

    candidates: List[Tuple[int, int, int, int, EvidenceLocation, str]] = []
    source_order = list(pages_by_source)
    if preferred_source in pages_by_source:
        source_order.remove(preferred_source)
        source_order.insert(0, preferred_source)

    for source_rank, source in enumerate(source_order):
        for page_number in sorted(pages_by_source[source]):
            page = pages_by_source[source][page_number]
            text = page.get("transcript") or page.get("text", "")
            for sentence_index, sentence in enumerate(_sentences(text)):
                hits = _limitation_hits(sentence)
                if not hits:
                    continue
                lowered = sentence.lower()
                if hits < 3 and any(marker in lowered for marker in _LIMITATION_CONTEXT):
                    continue
                if len(sentence) > 700:
                    continue
                page_rank = 0 if preferred_page == page_number else 1
                candidates.append(
                    (
                        source_rank,
                        -hits,
                        page_rank,
                        sentence_index,
                        EvidenceLocation(source=source, page=page_number),
                        sentence,
                    )
                )

    if not candidates:
        return None
    _, _, _, _, location, sentence = min(candidates, key=lambda item: item[:4])
    return location, sentence


def _component_result(
    operation: str,
    claimed_numeric: Optional[float],
    source_operands: Sequence[float],
    claimed_text: Optional[str] = None,
    claimed_values: Optional[Sequence[float]] = None,
    direction: Optional[str] = None,
    reported_value: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    ops = [float(value) for value in source_operands]
    values = [float(value) for value in (claimed_values or [])]
    info: Dict[str, Any] = {
        "claimed": claimed_text,
        "from_source": ops,
        "result": "N/A",
        "components": [],
        "computed_value": None,
    }

    if operation == "none":
        if claimed_numeric is not None or values:
            info["result"] = "claim contains a figure but no arithmetic was extracted"
            return CheckStatus.UNCLEAR.value, info
        return "not_applicable", info

    if claimed_numeric is None and not values:
        info["result"] = "missing claimed value"
        return CheckStatus.UNCLEAR.value, info

    if operation == "ratio":
        if len(ops) < 2 or ops[1] == 0:
            info["result"] = "invalid_operands"
            return CheckStatus.UNCLEAR.value, info
        ratio = abs(ops[0] / ops[1])
        info["result"] = f"{round(ratio, 2)}x"
        info["computed_value"] = ratio
        claimed = float(claimed_numeric if claimed_numeric is not None else values[0])
        rel_diff = abs(ratio - claimed) / abs(claimed) if claimed else abs(ratio)
        return (CheckStatus.MATCH if rel_diff <= 0.15 else CheckStatus.MISMATCH).value, info

    if operation == "difference":
        if len(ops) < 2:
            info["result"] = "insufficient_operands"
            return CheckStatus.UNCLEAR.value, info
        claimed = float(claimed_numeric if claimed_numeric is not None else values[0])
        source_diff = reported_value
        if source_diff is None and len(ops) >= 3:
            # Backward-compatible with older logs that stored the printed
            # difference as a third operand.
            source_diff = ops[2]

        if source_diff is not None:
            # The source prints the difference itself. Its sign is stronger
            # evidence than arm values, which a claim may state as unsigned
            # magnitudes ("1.3% reduction").
            source_diff = float(source_diff)
            info["result"] = _display_number(source_diff)
            info["computed_value"] = source_diff
            if abs(abs(source_diff) - abs(claimed)) > 0.05:
                info["result"] += " (source difference does not match the claim)"
                return CheckStatus.MISMATCH.value, info
            if direction == "decrease" and source_diff > 0.05:
                if not any(value < 0 for value in ops[:2]):
                    info["result"] += " (unsigned source difference cannot settle direction)"
                    return CheckStatus.UNCLEAR.value, info
                info["result"] += " (source reports an increase)"
                return CheckStatus.MISMATCH.value, info
            if direction == "increase" and source_diff < -0.05:
                if not any(value < 0 for value in ops[:2]):
                    info["result"] += " (unsigned source difference cannot settle direction)"
                    return CheckStatus.UNCLEAR.value, info
                info["result"] += " (source reports a decrease)"
                return CheckStatus.MISMATCH.value, info
            if direction is None and source_diff * claimed < 0:
                if not any(value < 0 for value in ops[:2]):
                    info["result"] += " (unsigned source difference cannot settle direction)"
                    return CheckStatus.UNCLEAR.value, info
                info["result"] += " (source direction is opposite to the claim)"
                return CheckStatus.MISMATCH.value, info
            return CheckStatus.MATCH.value, info

        diff = ops[0] - ops[1]
        info["result"] = _display_number(diff)
        info["computed_value"] = diff
        if abs(abs(diff) - abs(claimed)) > 0.05:
            return CheckStatus.MISMATCH.value, info
        if direction in ("decrease", "increase"):
            conflict = (direction == "decrease" and diff > 0.05) or (
                direction == "increase" and diff < -0.05
            )
            if conflict:
                if not any(value < 0 for value in ops[:2]):
                    info["result"] += " (unsigned magnitudes cannot settle direction)"
                    return CheckStatus.UNCLEAR.value, info
                info["result"] += (
                    " (source reports an increase)"
                    if direction == "decrease"
                    else " (source reports a decrease)"
                )
                return CheckStatus.MISMATCH.value, info
        if direction is None and claimed * diff < 0:
            if not any(value < 0 for value in ops[:2]):
                info["result"] += " (unsigned magnitudes cannot settle direction)"
                return CheckStatus.UNCLEAR.value, info
            info["result"] += " (source direction is opposite to the claim)"
            return CheckStatus.MISMATCH.value, info
        return CheckStatus.MATCH.value, info

    if operation == "percentage":
        if len(ops) >= 4:
            percentages = [
                (ops[0] / ops[1]) * 100 if ops[1] else None,
                (ops[2] / ops[3]) * 100 if ops[3] else None,
            ]
            if any(value is None for value in percentages):
                info["result"] = "invalid_operands"
                return CheckStatus.UNCLEAR.value, info
            info["result"] = f"{percentages[0]:.2f}% vs {percentages[1]:.2f}%"
            info["computed_value"] = percentages
            targets = values if len(values) >= 2 else (
                [float(claimed_numeric)] if claimed_numeric is not None else []
            )
            if len(targets) < 2:
                info["result"] += " (second percentage was not supplied for verification)"
                return CheckStatus.UNCLEAR.value, info
            ok = all(abs(computed - target) <= 0.1 for computed, target in zip(percentages, targets))
            return (CheckStatus.MATCH if ok else CheckStatus.MISMATCH).value, info

        if len(ops) < 2 or ops[1] == 0:
            info["result"] = "invalid_operands"
            return CheckStatus.UNCLEAR.value, info
        percentage = (ops[0] / ops[1]) * 100
        info["result"] = f"{percentage:.2f}%"
        info["computed_value"] = percentage
        claimed = float(claimed_numeric if claimed_numeric is not None else values[0])
        return (
            CheckStatus.MATCH if abs(percentage - claimed) <= 0.1 else CheckStatus.MISMATCH
        ).value, info

    if operation == "odds_ratio":
        reported = reported_value if reported_value is not None else (
            ops[4] if len(ops) >= 5 else (ops[0] if len(ops) == 1 else None)
        )
        claimed = float(claimed_numeric if claimed_numeric is not None else values[0])
        if reported is not None:
            info["result"] = f"Reported OR: {_display_number(float(reported))}"
            info["computed_value"] = float(reported)
            return (
                CheckStatus.MATCH
                if abs(float(reported) - claimed) <= 0.05
                else CheckStatus.MISMATCH
            ).value, info

        if len(ops) >= 4:
            a_events, a_total, b_events, b_total = ops[:4]
            a_nonevents = a_total - a_events
            b_nonevents = b_total - b_events
            if a_nonevents <= 0 or b_events <= 0 or b_nonevents <= 0:
                info["result"] = "invalid_denominators"
                return CheckStatus.UNCLEAR.value, info
            crude_or = (a_events * b_nonevents) / (b_events * a_nonevents)
            info["result"] = f"Crude OR: {crude_or:.2f}"
            info["computed_value"] = crude_or
            return (
                CheckStatus.MATCH if abs(crude_or - claimed) <= 0.10 else CheckStatus.MISMATCH
            ).value, info

        info["result"] = "insufficient_operands"
        return CheckStatus.UNCLEAR.value, info

    if operation == "reported":
        reported = reported_value if reported_value is not None else (ops[0] if ops else None)
        if reported is None:
            info["result"] = "missing_source_value"
            return CheckStatus.UNCLEAR.value, info
        claimed = float(claimed_numeric if claimed_numeric is not None else values[0])
        info["result"] = f"Reported value: {_display_number(float(reported))}"
        info["computed_value"] = float(reported)
        return (
            CheckStatus.MATCH
            if abs(float(reported) - claimed) <= 0.05
            else CheckStatus.MISMATCH
        ).value, info

    info["result"] = f"unsupported_operation: {operation}"
    return CheckStatus.UNCLEAR.value, info


def verify_arithmetic(spec: ArithmeticSpec) -> Tuple[str, Dict[str, Any]]:
    """Evaluate every arithmetic component and return one aggregate status."""

    if spec.components:
        statuses: List[str] = []
        details: List[Dict[str, Any]] = []
        results: List[str] = []
        operands: List[float] = []
        for component in spec.components:
            status, info = _component_result(
                component.operation,
                component.claimed_numeric,
                component.source_operands,
                component.claimed_text,
                direction=component.direction or spec.direction,
                reported_value=component.reported_value,
            )
            statuses.append(status)
            details.append(info)
            results.append(str(info.get("result", "")))
            for value in info.get("from_source", []):
                if value not in operands:
                    operands.append(value)

        if CheckStatus.MISMATCH.value in statuses:
            aggregate = CheckStatus.MISMATCH.value
        elif any(status == CheckStatus.UNCLEAR.value for status in statuses):
            aggregate = CheckStatus.UNCLEAR.value
        elif statuses and all(status == CheckStatus.MATCH.value for status in statuses):
            aggregate = CheckStatus.MATCH.value
        else:
            aggregate = CheckStatus.UNCLEAR.value

        return aggregate, {
            "claimed": spec.claimed_text,
            "from_source": operands,
            "result": " | ".join(results),
            "components": details,
        }

    return _component_result(
        spec.operation,
        spec.claimed_numeric,
        spec.source_operands,
        spec.claimed_text,
        spec.claimed_values,
        spec.direction,
        spec.reported_value,
    )


def resolve_verdict(checks: Dict[str, Optional[str]], span_verified: bool) -> str:
    if not span_verified:
        return Verdict.CANNOT_VERIFY.value

    if any(value == CheckStatus.MISMATCH.value for value in checks.values()):
        return Verdict.OVERSTATED.value

    if any(value == CheckStatus.UNCLEAR.value for value in checks.values()):
        return Verdict.CANNOT_VERIFY.value

    active = [
        value
        for value in checks.values()
        if value not in (None, "not_applicable")
    ]
    if active and all(value == CheckStatus.MATCH.value for value in active):
        return Verdict.SUPPORTED.value

    return Verdict.CANNOT_VERIFY.value


def _signed_present(value: float, observed: Sequence[float]) -> bool:
    return any(abs(float(value) - float(candidate)) <= 0.011 for candidate in observed)


def span_supports_arithmetic(
    evidence_span: str,
    spec: ArithmeticSpec,
    page_text: str,
    status: str,
) -> bool:
    """Check that the quote and page contain the numbers the arithmetic used.

    A ``match`` must quote at least one claimed value, while every source operand
    must occur on the cited page. A ``mismatch`` must quote at least one real
    source operand, because the source will not contain the false claimed
    number. Signs are preserved: an invented positive value cannot be certified
    by a negative source value of the same magnitude.
    """

    if status not in (CheckStatus.MATCH.value, CheckStatus.MISMATCH.value):
        return True
    if spec.operation == "none" and not spec.components:
        return True
    if not evidence_span or not page_text:
        return False

    span_numbers = [float(token) for token in _numeric_tokens(evidence_span)]
    page_numbers = [float(token) for token in _numeric_tokens(page_text)]
    claimed: List[float] = []
    operands: List[float] = []

    if spec.components:
        for component in spec.components:
            if component.claimed_numeric is not None:
                claimed.append(float(component.claimed_numeric))
            operands.extend(float(value) for value in component.source_operands)
            if component.reported_value is not None:
                operands.append(float(component.reported_value))
    else:
        if spec.claimed_numeric is not None:
            claimed.append(float(spec.claimed_numeric))
        operands.extend(float(value) for value in spec.source_operands)
        if spec.reported_value is not None:
            operands.append(float(spec.reported_value))

    if operands and not all(_signed_present(value, page_numbers) for value in operands):
        return False

    if status == CheckStatus.MATCH.value:
        return bool(claimed) and any(
            abs(abs(value) - abs(candidate)) <= 0.011
            for value in claimed
            for candidate in span_numbers
        )

    return any(
        abs(abs(value) - abs(candidate)) <= 0.011
        for value in operands
        for candidate in span_numbers
    )


def _component_numbers(spec: ArithmeticSpec) -> Tuple[List[str], List[float], List[float]]:
    """Return operations, operands and computed values for coverage checks."""

    operations: List[str] = []
    operands: List[float] = []
    computed: List[float] = []
    components = spec.components or [
        ArithmeticComponent(
            operation=spec.operation,
            claimed_numeric=spec.claimed_numeric,
            source_operands=list(spec.source_operands),
            claimed_text=spec.claimed_text,
            reported_value=spec.reported_value,
        )
    ]
    for component in components:
        operations.append(component.operation)
        operands.extend(float(value) for value in component.source_operands)
        if component.reported_value is not None:
            operands.append(float(component.reported_value))
        status, info = _component_result(
            component.operation,
            component.claimed_numeric,
            component.source_operands,
            component.claimed_text,
            direction=component.direction or spec.direction,
            reported_value=component.reported_value,
        )
        value = info.get("computed_value")
        if isinstance(value, (int, float)):
            computed.append(float(value))
        elif isinstance(value, list):
            computed.extend(float(item) for item in value if item is not None)
    return operations, operands, computed


def figure_coverage_gaps(claim_text: str, spec: ArithmeticSpec) -> List[str]:
    """Return claimed figures that no extracted component accounts for.

    This is a completeness check, not operation inference. It looks only at the
    numbers the claim itself asserts and asks whether the extraction supplied the
    operands needed to recompute them.
    """

    if not claim_text:
        return []
    operations, operands, computed = _component_numbers(spec)
    observed = operands + computed
    gaps: List[str] = []

    anchors = comparison_anchor_numbers(claim_text)
    if anchors and not all(
        any(abs(abs(value) - abs(candidate)) <= 0.11 for candidate in observed)
        for value in anchors
    ):
        gaps.append("comparison percentages")

    components = spec.components or [ArithmeticComponent(
        operation=spec.operation,
        claimed_numeric=spec.claimed_numeric,
        source_operands=list(spec.source_operands),
        claimed_text=spec.claimed_text,
        reported_value=spec.reported_value,
    )]
    effect_components = [
        component for component in components
        if component.operation in ("odds_ratio", "reported")
        and component.claimed_numeric is not None
    ]
    for label, value_text in re.findall(
        r"\b(OR|HR|RR)\b\s*[=:]?\s*(\d+(?:\.\d+)?)",
        claim_text,
        flags=re.IGNORECASE,
    ):
        value = float(value_text)
        if not any(
            abs(value - float(component.claimed_numeric or 0)) <= 0.05
            for component in effect_components
        ):
            gaps.append(f"{label.upper()} {value_text}")

    ratio_components = [
        component for component in components
        if component.operation == "ratio" and component.claimed_numeric is not None
    ]
    for value_text in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[x×]", claim_text):
        value = float(value_text)
        if not any(
            abs(value - float(component.claimed_numeric or 0))
            <= max(0.15 * abs(value), 0.05)
            for component in ratio_components
        ):
            gaps.append(f"ratio {value_text}x")

    return gaps


def claim_has_quantitative_assertion(text: str) -> bool:
    """Detect a numeric endpoint without treating every digit as a figure."""

    if not text:
        return False
    normalized = normalize_text(text)
    patterns = [
        r"\d+(?:\.\d+)?\s*%",
        r"\b(?:odds|risk|hazard)\s+ratio\b",
        r"\b(?:OR|HR|RR)\b\s*(?:=|of|:)?\s*\d+(?:\.\d+)?",
        r"~?\s*\d+(?:\.\d+)?\s*[x×]",
        r"\bp\s*[<>=]",
        r"\d+(?:\.\d+)?\s*(?:mg|kg|mmol|u/ml)",
        r"\b\d+(?:\.\d+)?\s*(?:vs\.?|versus)\s*\d+(?:\.\d+)?",
        (
            r"\b(?:difference|reduction|increase|decrease|change|"
            r"improvement|lower|higher|greater)\b[^.]{0,50}?"
            r"\d+(?:\.\d+)?"
        ),
    ]
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
        return True
    # Abbreviations are uppercase. A case-insensitive \bOR\b would match the
    # ordinary word "or" in a population description.
    return bool(re.search(r"\b(?:OR|HR|RR|CI)\b", text))


def deterministic_figure_reason(
    claimed_text: Optional[str],
    computed: Dict[str, Any],
) -> str:
    """Build a reviewer reason from code-owned numbers, not model prose."""

    claimed = claimed_text or computed.get("claimed") or "the claimed figure"
    source = computed.get("from_source") or []
    source_text = ", ".join(_display_number(float(value)) for value in source)
    result = computed.get("result", "not computed")
    if source_text:
        return (
            f"The claimed {claimed} does not reconcile with the source values "
            f"({source_text}); computed {result}."
        )
    return f"The claimed {claimed} does not reconcile with the source; computed {result}."


def _short(text: Optional[str], limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def build_reason(
    claim_text: str,
    checks: Dict[str, Optional[str]],
    bases: Sequence[str],
    computed: Optional[Dict[str, Any]],
    limitation_span: Optional[str] = None,
    supporting_values: Optional[Sequence[float]] = None,
) -> str:
    """Build one reviewer reason from code-owned checks and values."""

    basis = {item for item in bases if item and item != "none"}
    parts: List[str] = []

    if checks.get("population") == CheckStatus.MISMATCH.value or "population" in basis:
        parts.append(
            "The cited study population does not match the claimed population "
            "without broadening or dropping a restriction."
        )
    if checks.get("drug_indication") == CheckStatus.MISMATCH.value or basis.intersection(
        {"drug_indication", "indication", "dose", "formulation", "timepoint", "endpoint"}
    ):
        parts.append(
            "The cited molecule, formulation, dose, indication, endpoint or time "
            "point does not match the claim."
        )
    if "comparator_substitution" in basis:
        parts.append(
            "The claim attributes the figure to a different comparator than the "
            "source's reported comparison."
        )
    if "study_design" in basis:
        parts.append(
            "The source describes this result as post hoc, exploratory or a "
            "re-analysis, not independent trial evidence."
        )
    if checks.get("actually_stated") == CheckStatus.MISMATCH.value or "not_stated" in basis:
        parts.append(f'The cited page does not state the claim: "{_short(claim_text)}".')

    if checks.get("figure") == CheckStatus.MISMATCH.value or "numeric" in basis:
        if computed:
            parts.append(
                deterministic_figure_reason(
                    computed.get("claimed") or claim_text,
                    computed,
                )
            )

    if limitation_span:
        parts.append(f'The source\'s own limitation is: "{_short(limitation_span, 260)}".')

    if supporting_values and not (
        checks.get("figure") == CheckStatus.MISMATCH.value or "numeric" in basis
    ):
        values = ", ".join(_display_number(float(value)) for value in supporting_values)
        parts.append(f"The cited page reports {values}.")

    if not parts:
        parts.append("At least one regulatory check contradicts the claim.")

    # Remove duplicate sentences while preserving order.
    unique: List[str] = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    return " ".join(unique)


def supported_reason(claim_text: str, computed: Optional[Dict[str, Any]]) -> str:
    reason = f'The cited page supports the claim: "{_short(claim_text)}".'
    if computed and computed.get("result") and computed.get("result") != "N/A":
        reason += f" Computed {computed['result']}."
    return reason


def claim_direction(text: str) -> Optional[str]:
    """Return the direction a claim asserts, when it is unambiguous.

    Only used for difference checks. A comparative phrase such as "greater
    reduction" describes a reduction, not an increase. A claim that is genuinely
    directionless returns ``None`` and the numeric operands decide.
    """

    if not text:
        return None
    lowered = normalize_text(text)
    decrease = bool(
        re.search(
            r"\b(?:reduc\w*|decreas\w*|lower\w*|less|fewer|declin\w*|drop\w*)\b",
            lowered,
        )
    )
    increase = bool(
        re.search(
            r"\b(?:increas\w*|higher|greater|more|rise|rose|gain\w*)\b",
            lowered,
        )
    )
    if re.search(r"\b(?:greater|more|higher|larger)\s+(?:\w+\s+){0,3}"
                 r"(?:reduc\w*|decreas\w*|declin\w*|lower\w*)\b", lowered):
        decrease, increase = True, False
    if re.search(r"\b(?:lower|less|smaller)\s+(?:\w+\s+){0,3}"
                 r"(?:increas\w*|gain\w*|rise|rose)\b", lowered):
        increase, decrease = True, False
    if decrease == increase:
        return None
    return "decrease" if decrease else "increase"
