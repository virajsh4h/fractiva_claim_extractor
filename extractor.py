"""Structured evidence extraction with a claim-agnostic prompt.

The model is an evidence locator, not a judge. It receives generic instructions,
returns a validated JSON object, and never sees a claim-specific answer key.
All arithmetic is rejected here and performed in ``verification_engine``.
"""

import base64
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Sequence

import requests
from pydantic import BaseModel, Field, PrivateAttr

from audit import append_llm_log
from config import (
    DEFAULT_TIMEOUT_MS,
    NVIDIA_BASE_URL,
    NVIDIA_TIMEOUT_S,
    OPENROUTER_BASE_URL,
    OPENROUTER_TIMEOUT_S,
    get_api_key,
    get_model_name,
    get_nvidia_api_key,
    get_nvidia_extraction_model,
    get_nvidia_vision_model,
    get_openrouter_api_key,
    get_openrouter_text_model,
    get_openrouter_vision_model,
    get_primary_provider,
    load_local_env,
)
from verification_engine import (
    check_span_presence,
    claim_has_quantitative_assertion,
    confidence_levels,
    normalize_text,
    page_evidence_text,
)

CheckValue = Literal["match", "mismatch", "unclear"]
Operation = Literal[
    "none",
    "reported",
    "difference",
    "ratio",
    "percentage",
    "odds_ratio",
    "composite",
]
MismatchBasis = Literal[
    "none",
    "population",
    "drug_indication",
    "dose",
    "formulation",
    "indication",
    "comparator_substitution",
    "not_stated",
    "study_design",
    "numeric",
    "timepoint",
    "endpoint",
]


class ArithmeticExtractionComponent(BaseModel):
    operation: Operation = Field(
        description="One of reported, difference, ratio, percentage, odds_ratio."
    )
    claimed_text: Optional[str] = Field(
        default=None, description="The verbatim claimed figure for this component."
    )
    claimed_numeric: float = Field(description="The claimed numeric value.")
    source_operands: List[float] = Field(
        default_factory=list,
        description=(
            "Raw source numbers only. Percentage: [numerator, denominator]. "
            "Difference: [arm_a, arm_b]. Odds ratio: "
            "[a_events, a_total, b_events, b_total]."
        ),
    )
    reported_value: Optional[float] = Field(
        default=None,
        description=(
            "The source-printed derived value when one exists: the signed "
            "difference for a difference, or the adjusted OR for an odds ratio."
        ),
    )


class ArithmeticExtraction(BaseModel):
    operation: Operation = Field(
        default="none",
        description="none, reported, ratio, difference, percentage, odds_ratio, or composite.",
    )
    claimed_text: Optional[str] = Field(default=None)
    claimed_numeric: Optional[float] = Field(default=None)
    claimed_values: List[float] = Field(
        default_factory=list,
        description="Every endpoint number asserted by the claim.",
    )
    source_operands: List[float] = Field(default_factory=list)
    reported_value: Optional[float] = Field(default=None)
    components: List[ArithmeticExtractionComponent] = Field(
        default_factory=list,
        description=(
            "Use for claims with more than one independent figure, such as two "
            "percentages plus an odds ratio."
        ),
    )


class PageTranscript(BaseModel):
    source: str = Field(description="Exact source label from the prompt.")
    page_number: int = Field(description="1-indexed PDF page number.")
    text: str = Field(description="Exact transcription of the attached page image.")


class TranscriptionResponse(BaseModel):
    page_transcripts: List[PageTranscript] = Field(default_factory=list)


def _repair_truncated_json(raw: str) -> str:
    """Repair JSON truncated mid-stream by token limit, escapes, or socket close."""
    s = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", raw)
    in_str = False
    esc = False
    stack: List[str] = []
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\":
            if in_str:
                esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch in ("{", "["):
                stack.append("}" if ch == "{" else "]")
            elif ch in ("}", "]"):
                if stack and stack[-1] == ch:
                    stack.pop()
    if in_str:
        s += '"'
    while stack:
        s += stack.pop()
    return s


def parse_structured_json(raw_text: str) -> Any:
    """Parse model JSON, repairing malformed escapes or truncated transcripts.

    A long page transcription occasionally hits token limits or contains a
    lone ``\\u`` sequence. That is invalid JSON even though the rest of the
    object is well formed. Repairing the escape and closing truncated strings
    keeps the surrounding text and avoids discarding a usable extraction. A
    fenced JSON block is unwrapped because not every OpenAI-compatible model
    honours JSON mode.
    """

    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        repaired = _repair_truncated_json(text)
        return json.loads(repaired)


def _normalize_page_transcripts(payload: Any, batch: List[Any]) -> Any:
    """Attach source and page to bare transcription strings.

    Some OpenRouter free vision models return
    ``{"page_transcripts": ["text"]}`` instead of the requested object array.
    Each fallback call attaches exactly one page, so the omitted provenance is
    unambiguous and can be restored without guessing.
    """

    if not isinstance(payload, dict) or not batch:
        return payload
    entries = payload.get("page_transcripts")
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return payload

    normalized: List[Any] = []
    for index, entry in enumerate(entries):
        label, page_number, _ = batch[min(index, len(batch) - 1)]
        if isinstance(entry, str):
            if entry.strip():
                normalized.append(
                    {
                        "source": label,
                        "page_number": page_number,
                        "text": entry,
                    }
                )
            continue
        if isinstance(entry, dict):
            item = dict(entry)
            item.setdefault("source", label)
            item.setdefault("page_number", page_number)
            normalized.append(item)
            continue
        normalized.append(entry)
    return {**payload, "page_transcripts": normalized}


def _brief_error(exc: Any) -> str:
    return f"{type(exc).__name__}: {str(exc)[:240]}"


def _nvidia_chat_payload(
    model: str,
    prompt: str,
    system_instruction: str,
    images: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 12000,
) -> Dict[str, Any]:
    """Build one OpenAI-compatible chat request for NVIDIA's endpoint.

    The JSON schema travels in the system message because the endpoint exposes
    ``json_object`` mode rather than Gemini's typed ``response_schema``.
    """

    content: Any = prompt
    if images:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = base64.b64encode(image["bytes"]).decode("ascii")
            mime = image.get("mime", "image/png")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def _call_nvidia_chat(
    api_key: str,
    model: str,
    prompt: str,
    system_instruction: str,
    images: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 12000,
) -> str:
    payload = _nvidia_chat_payload(
        model=model,
        prompt=prompt,
        system_instruction=system_instruction,
        images=images,
        max_tokens=max_tokens,
    )
    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=NVIDIA_TIMEOUT_S,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"NVIDIA HTTP {response.status_code}: {response.text[:300]}"
        )
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"NVIDIA response shape was invalid: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("NVIDIA returned an empty message")
    return content


def _call_openrouter_chat(
    api_key: str,
    model: str,
    prompt: str,
    system_instruction: str,
    images: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 12000,
) -> str:
    """Call OpenRouter's OpenAI-compatible endpoint.

    Free models are the last fallback. Some of them reject ``json_object``
    response formatting, so a 400/422 that names that field is retried once
    without it. The prompt still describes the required JSON object.
    """

    payload = _nvidia_chat_payload(
        model=model,
        prompt=prompt,
        system_instruction=system_instruction,
        images=images,
        max_tokens=max_tokens,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "Clinical Claim Verifier",
    }
    for attempt in (1, 2):
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=OPENROUTER_TIMEOUT_S,
        )
        if (
            response.status_code in (400, 422)
            and attempt == 1
            and "response_format" in response.text.lower()
        ):
            payload = {
                key: value
                for key, value in payload.items()
                if key != "response_format"
            }
            continue
        break
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenRouter HTTP {response.status_code}: {response.text[:300]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"OpenRouter returned invalid JSON: {response.text[:300]}"
        ) from exc
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"OpenRouter error: {data['error']}")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        body = json.dumps(data, ensure_ascii=False)[:300]
        raise RuntimeError(
            f"OpenRouter response shape was invalid: {exc}; body={body}"
        ) from exc
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter returned an empty message")
    return content


def _call_gemini(
    api_key: str,
    model: str,
    prompt: str,
    system_instruction: str,
    schema: Any,
    image_parts: Optional[List[Any]] = None,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=DEFAULT_TIMEOUT_MS),
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt] + list(image_parts or []),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        ),
    )
    return response.text or ""


def _nvidia_schema_instruction(system_instruction: str, schema: Any) -> str:
    schema_name = getattr(schema, "__name__", "")
    if schema_name == "TranscriptionResponse":
        return (
            system_instruction
            + "\n\nReturn only one JSON object containing the actual transcribed text, formatted as:\n"
            '{"page_transcripts": [{"source": "<source_name>", "page_number": <page_number>, "text": "<exact transcribed text>"}]}\n'
            "Return the actual transcribed data instance, never the schema or $defs definitions. Do not wrap it in markdown."
        )
    return (
        system_instruction
        + "\n\nReturn only one JSON object that is an instance conforming to this "
        "JSON schema. Return the instance, never the schema itself. Do not wrap "
        "it in markdown.\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


class ClaimExtractionResponse(BaseModel):
    population_check: CheckValue = Field(
        description="match, mismatch, or unclear. Does the study population match without broadening?"
    )
    drug_indication_check: CheckValue = Field(
        description=(
            "match, mismatch, or unclear. Do disease, molecule, formulation, "
            "dose and time point match? Comparator attribution is judged by "
            "actually_stated_check."
        )
    )
    actually_stated_check: CheckValue = Field(
        description=(
            "match, mismatch, or unclear. Is the claim's comparison explicitly "
            "stated by the source, rather than inferred, generalized, attributed "
            "to a different comparator, or stronger than the source's own "
            "characterization?"
        )
    )
    mismatch_bases: List[MismatchBasis] = Field(
        default_factory=list,
        description=(
            "Every basis that applies when a check is mismatch. Use none only "
            "when every check is match."
        ),
    )
    evidence_source: Optional[str] = Field(
        default=None, description="Exact source label containing the evidence span."
    )
    evidence_page: Optional[int] = Field(
        default=None, description="1-indexed PDF page number, not a journal page number."
    )
    evidence_span: Optional[str] = Field(
        default=None, description="Exact verbatim contiguous substring from the cited page."
    )
    arithmetic: ArithmeticExtraction = Field(default_factory=ArithmeticExtraction)
    supporting_values: List[float] = Field(
        default_factory=list,
        description=(
            "For a claim with no numeric endpoint, raw source numbers that "
            "decisively qualify or contradict the claim."
        ),
    )
    page_transcripts: List[PageTranscript] = Field(
        default_factory=list,
        description=(
            "Only for an attached image page that has no transcript yet. Never "
            "retranscribe a page marked TRANSCRIBED."
        ),
    )
    _failure_reason: Optional[str] = PrivateAttr(default=None)


SYSTEM_INSTRUCTION = """You are a regulatory evidence auditor.
Compare the promotional claim with the supplied source pages and return one structured audit.

RULES:
1. Use only the supplied pages. Never use outside knowledge.
2. Do not calculate. Copy raw source numbers exactly. Python performs every calculation.
3. A check is "match" only when the source explicitly supports that element.
   - population: the study population and every stated restriction match the claim without broadening.
   - drug_indication: disease, molecule, formulation, dose and time point match. Comparator attribution is judged by actually_stated.
   - actually_stated: the source states the claim's comparison and characterization. It is a mismatch when the claim is inferred, generalized, attributed to a different comparator, or stronger than the source's own wording.
   "unclear" is only for a source that cannot be read or does not contain enough information.
4. When a check is mismatch, list every applicable mismatch_bases value: population, drug_indication, dose, formulation, indication, comparator_substitution, not_stated, study_design, numeric, timepoint, endpoint. Do not guess a basis the source does not support. Leave the list empty when every check is match.
5. evidence_span must be one exact contiguous substring of the cited page text or transcript, copied character for character. Do not use an ellipsis. Do not join separate lines, table cells or columns. Do not reorder words. Prefer a passage that contains a row label or sentence as well as the numbers. For a comparison, quote the source's reported comparison and include the claimed figure. When the claim states a confidence interval or p-value, quote the source's own result with that interval or p-value. Do not substitute a 97.5% CI for a claimed 95% CI, or the reverse. Cite the page that reports the claimed result, not a baseline characteristics table, a methods paragraph, or a summary that reports different values.
6. evidence_page is the 1-indexed PDF page, not the journal page.
7. Pages marked TRANSCRIBED are evidence text. Quote them exactly. Do not retranscribe a page that already has a transcript.
8. Arithmetic:
   - none: no numeric endpoint.
   - reported: one number printed directly by the source; put it in reported_value.
   - difference: one component. Put the two source arm values exactly as printed, including any minus sign, in source_operands. Put the source-printed signed difference in reported_value. claimed_numeric is the claimed difference, not an arm value. Never add separate components for the arm values.
   - ratio: [numerator, denominator]. claimed_numeric is the claimed ratio.
   - percentage: [numerator, denominator]. claimed_numeric is the claimed percentage.
   - odds_ratio: [a_events, a_total, b_events, b_total] and the printed adjusted OR in reported_value. claimed_numeric is the claimed OR.
   - composite: one component for each independent figure.
   Use composite only when the claim asserts more than one independent result, such as two percentages plus an odds ratio.
   Preserve the sign of every source number exactly as printed. Put every number the claim asserts in claimed_values. Never put a calculated result in source_operands.
   EXAMPLE (generic): source "arm A fell 1.4% and arm B fell 1.0%; difference -0.4%", claim "1.4% vs 1.0%; difference -0.4%". Correct: one difference component, claimed_numeric=-0.4, source_operands=[-1.4,-1.0], reported_value=-0.4. Wrong: one reported component per number.
9. For a claim with no numeric endpoint, put decisive source numbers in supporting_values when they exist.
10. Do not output a reason or any calculated value.
"""


def build_prompt(
    claim: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> str:
    chunks: List[str] = []
    for source in sources:
        label = source.get("label", "source")
        path = source.get("path", "")
        chunks.append(f"\n===== SOURCE: {label} ({path}) =====")
        for page_number, page in sorted(source.get("pages", {}).items()):
            provenance = page.get("provenance", "native")
            transcript = page.get("transcript")
            if transcript:
                chunks.append(
                    f"\n--- PAGE {page_number} (TRANSCRIBED FROM {provenance.upper()} IMAGE) ---\n"
                    f"{transcript}\n"
                )
            elif page.get("needs_vision"):
                chunks.append(
                    f"\n--- PAGE {page_number} (IMAGE ATTACHED; {provenance}) ---\n"
                    f"{page.get('text', '')}\n"
                    "If this page has no transcript, transcribe the relevant passage exactly "
                    "in page_transcripts.\n"
                )
            else:
                chunks.append(f"\n--- PAGE {page_number} (NATIVE TEXT) ---\n{page.get('text', '')}\n")
            tables = page.get("tables") or []
            if tables:
                chunks.append("TABLES ON THIS PAGE:\n" + "\n\n".join(tables) + "\n")

    return f"""CLAIM TO AUDIT
Claim ID: {claim.get('claim_id') or 'N/A'}
Text: {claim.get('text') or ''}
Claimed population: {claim.get('population') or 'Not specified'}
Claimed molecule: {claim.get('molecule') or 'Not specified'}
Cited source(s): {claim.get('cited_paper') or 'Not specified'}

SOURCE DOCUMENT CONTEXT:
{"".join(chunks)}

Return the structured audit. Quote the exact contiguous page text or transcript that proves your finding.
"""


TRANSCRIPTION_INSTRUCTION = """You are a document transcription engine.
Transcribe every visible word on each attached page exactly as printed.
Do not summarize, correct, shorten, explain or omit anything.
Preserve numbers, punctuation, units and line order.
For a multi-column page, transcribe in natural reading order.
Return one page_transcripts entry for every attached page, using the exact
source label and 1-indexed PDF page number supplied in the prompt.
"""


def _transcript_cache_path(
    cache_dir: str,
    image_bytes: bytes,
    model: str,
) -> str:
    """Return a deterministic cache path for one page image and model."""

    digest = hashlib.sha256(
        b"transcription-v2\0" + model.encode("utf-8") + b"\0" + image_bytes
    ).hexdigest()
    return os.path.join(cache_dir, digest + ".json")


def _load_cached_transcript(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        text = payload.get("text", "")
        return text if isinstance(text, str) and text.strip() else None
    except (OSError, ValueError):
        return None


def _store_cached_transcript(
    path: str,
    label: str,
    page_number: int,
    text: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": label,
                "page_number": page_number,
                "text": text,
            },
            handle,
            ensure_ascii=False,
        )


def _provider_failure_is_fatal(exc: Any) -> bool:
    """A quota or availability failure should go straight to the fallback."""

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "resource_exhausted",
            "quota",
            "503",
            "unavailable",
            "403",
            "permission",
        )
    )


def _provider_retryable(exc: Any) -> bool:
    """Transient endpoint failures are worth one or two retries."""

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "rate limit",
            "timed out",
            "timeout",
            "connection",
            "temporarily",
            "response shape",
            "empty message",
            "invalid json",
        )
    )


def _nvidia_retryable(exc: Any) -> bool:
    return _provider_retryable(exc)


def _openrouter_retryable(exc: Any) -> bool:
    return _provider_retryable(exc)


def transcribe_scanned_pages(
    sources: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    log_path: str = "outputs/llm_log.jsonl",
    max_attempts: int = 2,
    cache_dir: Optional[str] = os.path.join("outputs", "cache", "transcripts"),
) -> List[PageTranscript]:
    """Transcribe every image-only page before claim extraction.

    Gemini is the primary transcriber. A quota, availability, timeout or parse
    failure falls back to NVIDIA's vision model, then to an OpenRouter free
    vision model. Each provider call is logged, including failed attempts.
    Transcripts are cached by image hash and model so a multi-claim run does not
    pay for the same scanned page twice. A failed multi-page request is retried
    one page at a time, which keeps one unreadable page from losing the rest.
    """

    pending = []
    transcripts: List[PageTranscript] = []
    gemini_model = model_name or get_model_name()
    nvidia_model = get_nvidia_vision_model()
    openrouter_model = get_openrouter_vision_model()
    for source in sources:
        for page_number, page in sorted(source.get("pages", {}).items()):
            if not page.get("needs_vision") or not page.get("image_bytes"):
                continue
            if page.get("transcript"):
                continue

            label = source.get("label", "source")
            if cache_dir:
                cached = None
                for cache_model in dict.fromkeys(
                    (gemini_model, nvidia_model, openrouter_model)
                ):
                    cache_path = _transcript_cache_path(
                        cache_dir,
                        page["image_bytes"],
                        cache_model,
                    )
                    cached = _load_cached_transcript(cache_path)
                    if cached:
                        break
                if cached:
                    page["transcript"] = cached
                    transcripts.append(
                        PageTranscript(source=label, page_number=page_number, text=cached)
                    )
                    continue
            pending.append((label, page_number, page))

    if not pending:
        return transcripts

    load_local_env()
    gemini_key = get_api_key()
    nvidia_key = get_nvidia_api_key()
    openrouter_key = get_openrouter_api_key()
    if not gemini_key and not nvidia_key and not openrouter_key:
        return transcripts

    def gemini_batch(batch: List[Any], attempts: int) -> tuple:
        listing = "\n".join(
            f"{index}. source={label!r}, page={page_number}"
            for index, (label, page_number, _) in enumerate(batch, 1)
        )
        prompt_text = f"""TASK: TRANSCRIBE PAGE IMAGES

The attached images are supplied in this order:
{listing}

Transcribe every visible word exactly. Do not summarize or omit fine print.
Return one page_transcripts entry per attached page.
"""

        from google.genai import types

        image_parts = [
            types.Part.from_bytes(
                data=page["image_bytes"],
                mime_type=page.get("image_mime", "image/png"),
            )
            for _, _, page in batch
        ]
        request_parts = [
            {
                "type": "image",
                "source": label,
                "page": page_number,
                "sha256": hashlib.sha256(page["image_bytes"]).hexdigest(),
            }
            for label, page_number, page in batch
        ]

        last_error = ""
        for attempt in range(1, attempts + 1):
            start_time = time.time()
            try:
                raw_text = _call_gemini(
                    api_key=gemini_key,
                    model=gemini_model,
                    prompt=prompt_text,
                    system_instruction=TRANSCRIPTION_INSTRUCTION,
                    schema=TranscriptionResponse,
                    image_parts=image_parts,
                )
            except Exception as exc:
                last_error = _brief_error(exc)
                append_llm_log(
                    log_path=log_path,
                    prompt=prompt_text,
                    response_raw=str(exc),
                    model=gemini_model,
                    latency_s=time.time() - start_time,
                    parameters={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "schema": "TranscriptionResponse",
                        "attempt": attempt,
                        "provider": "gemini",
                    },
                    metadata={
                        "task": "transcription",
                        "pages": len(batch),
                        "provider": "gemini",
                        "fallback": False,
                        "error": True,
                    },
                    system_instruction=TRANSCRIPTION_INSTRUCTION,
                    request_parts=request_parts,
                )
                if _provider_failure_is_fatal(exc):
                    return [], True, last_error
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 8))
                continue

            parsed: Optional[TranscriptionResponse] = None
            parse_error = None
            try:
                parsed = TranscriptionResponse(**parse_structured_json(raw_text))
            except Exception as exc:
                parse_error = _brief_error(exc)

            matched: List[Any] = []
            if parsed is not None:
                for item in parsed.page_transcripts:
                    for label, page_number, page in batch:
                        if item.source != label or item.page_number != page_number:
                            continue
                        if not item.text.strip():
                            continue
                        matched.append((label, page_number, page, item.text))
                        break
                if not matched:
                    parse_error = "response contained no matching page transcript"
                    parsed = None

            append_llm_log(
                log_path=log_path,
                prompt=prompt_text,
                response_raw=raw_text,
                model=gemini_model,
                latency_s=time.time() - start_time,
                parameters={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "schema": "TranscriptionResponse",
                    "attempt": attempt,
                    "provider": "gemini",
                },
                metadata={
                    "task": "transcription",
                    "pages": len(batch),
                    "provider": "gemini",
                    "fallback": False,
                    **({"error": True, "parse_error": parse_error} if parse_error else {}),
                },
                system_instruction=TRANSCRIPTION_INSTRUCTION,
                request_parts=request_parts,
            )
            if parsed is None:
                last_error = parse_error or last_error
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 8))
                continue

            saved: List[PageTranscript] = []
            for label, page_number, page, text in matched:
                page["transcript"] = text
                if cache_dir:
                    _store_cached_transcript(
                        _transcript_cache_path(
                            cache_dir,
                            page["image_bytes"],
                            gemini_model,
                        ),
                        label,
                        page_number,
                        text,
                    )
                saved.append(
                    PageTranscript(
                        source=label,
                        page_number=page_number,
                        text=text,
                    )
                )
            return saved, False, last_error
        return [], False, last_error

    provider_system = _nvidia_schema_instruction(
        TRANSCRIPTION_INSTRUCTION,
        TranscriptionResponse,
    )

    def run_gemini() -> Optional[str]:
        if not gemini_key:
            return "GEMINI_API_KEY is missing; using fallbacks."
        unread_pages = [item for item in pending if not item[2].get("transcript")]
        if not unread_pages:
            return None
        saved, fatal, error_note = gemini_batch(unread_pages, max_attempts)
        transcripts.extend(saved)
        gem_err = f"gemini transcription failed: {error_note}" if error_note else None
        remaining = [item for item in pending if not item[2].get("transcript")]
        if remaining and len(remaining) > 1 and not fatal:
            for item in remaining:
                saved, fatal, err2 = gemini_batch([item], 1)
                transcripts.extend(saved)
                if err2:
                    gem_err = f"gemini transcription failed: {err2}"
                if fatal:
                    break
        return gem_err

    def vision_page(
        item: Any,
        provider: str,
        api_key: str,
        model: str,
        fallback_reason: Optional[str],
    ) -> Optional[PageTranscript]:
        label, page_number, page = item
        prompt_text = f"""TASK: TRANSCRIBE PAGE IMAGE

The attached image is supplied as:
1. source={label!r}, page={page_number}

Transcribe every visible word exactly. Do not summarize or omit fine print.
Return one page_transcripts entry for this page.
"""
        request_parts = [
            {
                "type": "image",
                "source": label,
                "page": page_number,
                "sha256": hashlib.sha256(page["image_bytes"]).hexdigest(),
            }
        ]
        call_chat = (
            _call_nvidia_chat if provider == "nvidia" else _call_openrouter_chat
        )
        retryable = (
            _nvidia_retryable
            if provider == "nvidia"
            else _openrouter_retryable
        )
        base_parameters = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "schema": "TranscriptionResponse",
            "provider": provider,
        }
        base_metadata = {
            "task": "transcription",
            "pages": 1,
            "provider": provider,
            "fallback": True,
            "fallback_reason": fallback_reason,
        }
        for attempt in range(1, 4):
            start_time = time.time()
            try:
                raw_text = call_chat(
                    api_key=api_key,
                    model=model,
                    prompt=prompt_text,
                    system_instruction=provider_system,
                    images=[
                        {
                            "bytes": page["image_bytes"],
                            "mime": page.get("image_mime", "image/png"),
                        }
                    ],
                    max_tokens=8000,
                )
            except Exception as exc:
                append_llm_log(
                    log_path=log_path,
                    prompt=prompt_text,
                    response_raw=str(exc),
                    model=model,
                    latency_s=time.time() - start_time,
                    parameters={**base_parameters, "attempt": attempt},
                    metadata={
                        **base_metadata,
                        "attempt": attempt,
                        "error": True,
                    },
                    system_instruction=provider_system,
                    request_parts=request_parts,
                )
                if retryable(exc) and attempt < 3:
                    time.sleep(3 * attempt)
                    continue
                return None

            parsed: Optional[TranscriptionResponse] = None
            parse_error = None
            try:
                parsed = TranscriptionResponse(
                    **_normalize_page_transcripts(
                        parse_structured_json(raw_text),
                        [item],
                    )
                )
            except Exception as exc:
                parse_error = _brief_error(exc)

            transcript_text: Optional[str] = None
            if parsed is not None:
                for entry in parsed.page_transcripts:
                    if entry.source != label or entry.page_number != page_number:
                        continue
                    if not entry.text.strip():
                        continue
                    transcript_text = entry.text
                    break
                if transcript_text is None:
                    parse_error = "response contained no matching page transcript"
                    parsed = None

            append_llm_log(
                log_path=log_path,
                prompt=prompt_text,
                response_raw=raw_text,
                model=model,
                latency_s=time.time() - start_time,
                parameters={**base_parameters, "attempt": attempt},
                metadata={
                    **base_metadata,
                    "attempt": attempt,
                    **({"error": True, "parse_error": parse_error} if parse_error else {}),
                },
                system_instruction=provider_system,
                request_parts=request_parts,
            )
            if parsed is None or transcript_text is None:
                if attempt < 3:
                    time.sleep(3 * attempt)
                    continue
                return None

            page["transcript"] = transcript_text
            if cache_dir:
                _store_cached_transcript(
                    _transcript_cache_path(
                        cache_dir,
                        page["image_bytes"],
                        model,
                    ),
                    label,
                    page_number,
                    transcript_text,
                )
            return PageTranscript(
                source=label,
                page_number=page_number,
                text=transcript_text,
            )
        return None

    def run_vision(
        provider_name: str,
        key: Optional[str],
        model: str,
        fallback_reason: Optional[str],
    ) -> Optional[str]:
        if not key:
            return f"{provider_name.upper()}_API_KEY is missing"
        unread_pages = [item for item in pending if not item[2].get("transcript")]
        for item in unread_pages:
            saved = vision_page(
                item,
                provider_name,
                key,
                model,
                fallback_reason,
            )
            if saved:
                transcripts.append(saved)
        remaining = [item for item in pending if not item[2].get("transcript")]
        if remaining:
            return f"{provider_name} transcription failed for {len(remaining)} page(s)"
        return None

    pref = get_primary_provider()
    order = (
        ["openrouter", "nvidia", "gemini"]
        if pref == "openrouter"
        else (
            ["nvidia", "openrouter", "gemini"]
            if pref == "nvidia"
            else ["gemini", "nvidia", "openrouter"]
        )
    )

    last_failure: Optional[str] = (
        None if gemini_key else "GEMINI_API_KEY is missing; using fallbacks."
    )
    for prov in order:
        if not any(not item[2].get("transcript") for item in pending):
            break
        if prov == "gemini":
            last_failure = run_gemini() or last_failure
        elif prov == "nvidia":
            nvidia_err = run_vision(
                "nvidia", nvidia_key, nvidia_model, last_failure
            )
            if nvidia_err:
                last_failure = (
                    f"{last_failure}; {nvidia_err}" if last_failure else nvidia_err
                )
        elif prov == "openrouter":
            run_vision(
                "openrouter", openrouter_key, openrouter_model, last_failure
            )

    return transcripts


def fail_closed_extraction(reason: str) -> ClaimExtractionResponse:
    response = ClaimExtractionResponse(
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
    response._failure_reason = reason
    return response


def _signed_numbers(text: str) -> List[float]:
    return [
        float(token)
        for token in re.findall(
            r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])",
            normalize_text(text or ""),
        )
    ]


def _contains_signed(value: float, observed: Sequence[float]) -> bool:
    return any(abs(float(value) - float(candidate)) <= 0.011 for candidate in observed)


def _display_level(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _arithmetic_operands(arithmetic: ArithmeticExtraction) -> List[float]:
    values: List[float] = []
    if arithmetic.components:
        for component in arithmetic.components:
            values.extend(float(item) for item in component.source_operands)
            if component.reported_value is not None:
                values.append(float(component.reported_value))
    else:
        values.extend(float(item) for item in arithmetic.source_operands)
        if arithmetic.reported_value is not None:
            values.append(float(arithmetic.reported_value))
    return values


def _find_verbatim_span(
    span: str,
    sources: List[Dict[str, Any]],
):
    for source in sources:
        for page_number, page in sorted(source.get("pages", {}).items()):
            text = page_evidence_text(page)
            if check_span_presence(span, text):
                return source, page_number, text
    return None


def _validation_issues(
    extraction: ClaimExtractionResponse,
    claim: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> List[str]:
    """Return deterministic structural failures that a repair call can fix.

    This does not judge the claim. It only checks that the evidence is a real
    contiguous passage and that the arithmetic refers to numbers on that page.
    """

    issues: List[str] = []
    span = extraction.evidence_span or ""
    if not span.strip():
        issues.append("evidence_span is empty.")
        found = None
    else:
        found = _find_verbatim_span(span, sources)
        if found is None:
            issues.append(
                "evidence_span is not a verbatim contiguous substring of any "
                "supplied page. Copy one complete passage exactly."
            )

    if found is not None:
        source, page_number, page_text = found
        page_numbers = _signed_numbers(page_text)
        operands = _arithmetic_operands(extraction.arithmetic)
        missing = [
            value
            for value in operands
            if not _contains_signed(value, page_numbers)
        ]
        if missing:
            shown = ", ".join(str(value) for value in missing)
            issues.append(
                f"The evidence page ({source.get('label')} page {page_number}) "
                f"does not contain these arithmetic operands with the printed "
                f"sign: {shown}. Cite the page that reports the result, or copy "
                "the source numbers exactly as printed."
            )

        claim_levels = confidence_levels(claim.get("text", ""))
        span_levels = confidence_levels(extraction.evidence_span or "")
        if claim_levels and span_levels and claim_levels.isdisjoint(span_levels):
            claimed = ", ".join(
                f"{_display_level(value)}%" for value in sorted(claim_levels)
            )
            quoted = ", ".join(
                f"{_display_level(value)}%" for value in sorted(span_levels)
            )
            issues.append(
                f"The claim states a {claimed} confidence interval, but the "
                f"quoted span reports {quoted}. Search every supplied page for "
                "the passage that reports the claimed confidence level and cite it."
            )

    arithmetic = extraction.arithmetic
    if arithmetic.operation == "composite" and len(arithmetic.components) > 1:
        derived_values = set()
        for component in arithmetic.components:
            if component.operation in {"difference", "ratio", "percentage", "odds_ratio"}:
                derived_values.update(
                    round(abs(float(value)), 6) for value in component.source_operands
                )
        redundant = []
        for component in arithmetic.components:
            if component.operation != "reported":
                continue
            values = [float(value) for value in component.source_operands]
            if component.reported_value is not None:
                values.append(float(component.reported_value))
            if any(round(abs(value), 6) in derived_values for value in values):
                redundant.append(component)
        if redundant:
            issues.append(
                "The composite reports an arm value separately and also uses that "
                "same value inside a derived comparison. A two-arm difference is "
                "one difference component, not a composite of the arms plus the "
                "difference. Return only the independent results the claim asserts."
            )

    if (
        arithmetic.operation == "composite"
        and len(arithmetic.components) > 2
        and all(component.operation == "reported" for component in arithmetic.components)
    ):
        issues.append(
            "The arithmetic used one reported component per number. A two-arm "
            "difference is one difference component; do not return a component "
            "for every number."
        )

    if (
        claim_has_quantitative_assertion(claim.get("text", ""))
        and arithmetic.operation == "none"
        and not arithmetic.components
    ):
        issues.append(
            "The claim asserts a numeric endpoint but arithmetic is none. "
            "Extract the calculation or the reported value."
        )

    return issues


def extract_claim_evidence(
    claim: Dict[str, Any],
    pages_data: Optional[Dict[int, Dict[str, Any]]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    model_name: Optional[str] = None,
    log_path: str = "outputs/llm_log.jsonl",
    max_attempts: int = 3,
) -> ClaimExtractionResponse:
    """Return one validated extraction, with Gemini and tested fallbacks.

    Gemini is primary. If it is missing, rate limited, unavailable or returns
    invalid JSON, the same request goes to NVIDIA's OpenAI-compatible endpoint,
    then to OpenRouter's free-model endpoint. A deterministic validation pass
    follows the first response. If the quote is not verbatim, the confidence
    level differs, or the arithmetic points at numbers the cited page does not
    contain, one repair call is made with the exact failures. The pipeline stays
    fail-closed if no provider returns a valid, validated audit.
    """

    load_local_env()
    gemini_key = get_api_key()
    nvidia_key = get_nvidia_api_key()
    openrouter_key = get_openrouter_api_key()
    if not gemini_key and not nvidia_key and not openrouter_key:
        return fail_closed_extraction(
            "LLM API key is missing (GEMINI_API_KEY, NVIDIA_API_KEY and "
            "OPENROUTER_KEY). Fails closed to cannot_verify."
        )

    if sources is None:
        source_label = claim.get("cited_paper", "source")
        sources = [{
            "label": source_label,
            "path": source_label,
            "pages": pages_data or {},
        }]

    prompt_text = build_prompt(claim, sources)
    gemini_image_parts: List[Any] = []
    fallback_images: List[Dict[str, Any]] = []
    request_parts: List[Dict[str, Any]] = []

    for source in sources:
        for page_number, page in sorted(source.get("pages", {}).items()):
            if not page.get("image_bytes") or page.get("transcript"):
                continue
            request_parts.append(
                {
                    "type": "image",
                    "source": source.get("label", "source"),
                    "page": page_number,
                    "sha256": hashlib.sha256(page["image_bytes"]).hexdigest(),
                }
            )
            if gemini_key:
                from google.genai import types

                gemini_image_parts.append(
                    types.Part.from_bytes(
                        data=page["image_bytes"],
                        mime_type=page.get("image_mime", "image/png"),
                    )
                )
            if nvidia_key or openrouter_key:
                fallback_images.append(
                    {
                        "bytes": page["image_bytes"],
                        "mime": page.get("image_mime", "image/png"),
                    }
                )

    gemini_model = model_name or get_model_name()
    # The text model is faster and cheaper. If a page still has no transcript,
    # only a vision model can read the attached image.
    nvidia_model = (
        get_nvidia_vision_model() if fallback_images else get_nvidia_extraction_model()
    )
    openrouter_model = (
        get_openrouter_vision_model()
        if fallback_images
        else get_openrouter_text_model()
    )

    def run_model(
        prompt: str,
        system_instruction: str,
        metadata: Dict[str, Any],
        attempts: Optional[int] = None,
    ) -> Optional[ClaimExtractionResponse]:
        budget = max(1, attempts or max_attempts)
        provider_dict = {}
        if gemini_key:
            provider_dict["gemini"] = ("gemini", gemini_model, budget)
        if nvidia_key:
            provider_dict["nvidia"] = ("nvidia", nvidia_model, 1 if budget <= 1 else 2)
        if openrouter_key:
            provider_dict["openrouter"] = (
                "openrouter",
                openrouter_model,
                1 if budget <= 1 else 2,
            )

        pref = get_primary_provider()
        order = (
            ["openrouter", "nvidia", "gemini"]
            if pref == "openrouter"
            else (
                ["nvidia", "openrouter", "gemini"]
                if pref == "nvidia"
                else ["gemini", "nvidia", "openrouter"]
            )
        )
        providers: List[tuple] = [
            provider_dict[p] for p in order if p in provider_dict
        ]

        fallback_reason: Optional[str] = (
            None if gemini_key else "GEMINI_API_KEY is missing; using fallbacks."
        )
        last_error_text = ""
        nvidia_system = _nvidia_schema_instruction(
            system_instruction,
            ClaimExtractionResponse,
        )

        for provider_index, (provider, provider_model, provider_attempts) in enumerate(
            providers
        ):
            is_fallback = provider_index > 0 or not gemini_key
            system_for_provider = (
                system_instruction if provider == "gemini" else nvidia_system
            )
            for attempt in range(1, provider_attempts + 1):
                start_time = time.time()
                try:
                    if provider == "gemini":
                        raw_text = _call_gemini(
                            api_key=gemini_key,
                            model=provider_model,
                            prompt=prompt,
                            system_instruction=system_instruction,
                            schema=ClaimExtractionResponse,
                            image_parts=gemini_image_parts,
                        )
                    elif provider == "nvidia":
                        raw_text = _call_nvidia_chat(
                            api_key=nvidia_key,
                            model=provider_model,
                            prompt=prompt,
                            system_instruction=system_for_provider,
                            images=fallback_images or None,
                        )
                    else:
                        raw_text = _call_openrouter_chat(
                            api_key=openrouter_key,
                            model=provider_model,
                            prompt=prompt,
                            system_instruction=system_for_provider,
                            images=fallback_images or None,
                        )
                except Exception as exc:
                    note = _brief_error(exc)
                    last_error_text = f"{provider} {note}"
                    if provider == "gemini":
                        fallback_reason = f"gemini failed: {note}"
                    elif provider == "nvidia":
                        fallback_reason = f"nvidia failed: {note}"
                    append_llm_log(
                        log_path=log_path,
                        prompt=prompt,
                        response_raw=str(exc),
                        model=provider_model,
                        latency_s=time.time() - start_time,
                        parameters={
                            "temperature": 0.0,
                            "response_mime_type": "application/json",
                            "schema": "ClaimExtractionResponse",
                            "attempt": attempt,
                            "provider": provider,
                        },
                        metadata={
                            **metadata,
                            "attempt": attempt,
                            "provider": provider,
                            "fallback": is_fallback,
                            "fallback_reason": fallback_reason,
                            "error": True,
                        },
                        system_instruction=system_for_provider,
                        request_parts=request_parts,
                    )
                    if _provider_failure_is_fatal(exc):
                        break
                    if attempt < provider_attempts:
                        time.sleep(min(2 ** attempt, 8))
                    continue

                parsed: Optional[ClaimExtractionResponse] = None
                parse_error: Optional[str] = None
                try:
                    parsed = ClaimExtractionResponse(
                        **parse_structured_json(raw_text)
                    )
                except Exception as exc:
                    parse_error = _brief_error(exc)
                    last_error_text = f"{provider} parse {parse_error}"
                    if provider == "gemini":
                        fallback_reason = f"gemini parse failed: {parse_error}"

                append_llm_log(
                    log_path=log_path,
                    prompt=prompt,
                    response_raw=raw_text,
                    model=provider_model,
                    latency_s=time.time() - start_time,
                    parameters={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "schema": "ClaimExtractionResponse",
                        "attempt": attempt,
                        "provider": provider,
                    },
                    metadata={
                        **metadata,
                        "attempt": attempt,
                        "provider": provider,
                        "fallback": is_fallback,
                        "fallback_reason": (
                            fallback_reason if is_fallback else None
                        ),
                        **({"error": True, "parse_error": parse_error} if parse_error else {}),
                    },
                    system_instruction=system_for_provider,
                    request_parts=request_parts,
                )
                if parsed is not None:
                    return parsed
                if attempt < provider_attempts:
                    time.sleep(min(2 ** attempt, 8))
        if last_error_text:
            return None
        return None

    extraction = run_model(
        prompt_text,
        SYSTEM_INSTRUCTION,
        {"claim_id": claim.get("claim_id")},
    )
    if extraction is None:
        return fail_closed_extraction(
            "Extraction failed on every configured provider. Fails closed to "
            "cannot_verify."
        )

    issues = _validation_issues(extraction, claim, sources)
    if not issues:
        return extraction

    repair_prompt = f"""{prompt_text}

REPAIR REQUIRED
The previous audit failed deterministic validation:
{chr(10).join(f"- {issue}" for issue in issues)}

PREVIOUS AUDIT:
{json.dumps(extraction.model_dump(), ensure_ascii=False)}

Return the complete corrected audit. Fix every failure above. Keep the same
schema and do not explain. The quote must be copied exactly from one page. A
two-arm comparison is one difference component, not one component per number.
Copy source numbers with their printed signs. Cite the page that reports the
claimed result."""

    repaired = run_model(
        repair_prompt,
        SYSTEM_INSTRUCTION,
        {"claim_id": claim.get("claim_id"), "repair": True, "issues": issues},
        attempts=1,
    )
    if repaired is None:
        return fail_closed_extraction(
            "The extraction failed deterministic validation and no provider "
            "returned a valid repair. Fails closed to cannot_verify."
        )

    remaining = _validation_issues(repaired, claim, sources)
    if remaining:
        return fail_closed_extraction(
            "Extraction still failed deterministic validation after one repair: "
            + "; ".join(remaining[:3])
        )

    return repaired
