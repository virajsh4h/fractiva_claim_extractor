# Specification: Clinical Claim Verification Pipeline

## Problem

Pharmaceutical leave-behinds often present trial results with the qualification
removed: a broader population, a renamed comparator, a post-hoc re-analysis
counted as a new trial, or a number rounded in the promotional direction. A
compliance reviewer needs a verdict, a verbatim source quote and the arithmetic
that supports it. The verifier must never mark an unproven claim `supported`.

## Solution

A neuro-symbolic pipeline in `verify.py`. A model locates evidence and extracts
raw numbers under a strict schema. Python performs all arithmetic, validates the
quote against the cited page and resolves the verdict. Sources, page text,
transcripts and model responses are retained so the result can be replayed.

## User stories

1. Run `python verify.py data/claims.json` and write `outputs/predictions.json`.
2. Accept any claims file with the assignment schema, without changing code.
3. Record every model request and raw response in `outputs/llm_log.jsonl`.
4. Read every image-only page before judging it, and cache the transcript.
5. Cite a verbatim span and a 1-indexed PDF page for every non-fail-closed
   verdict.
6. Compute every difference, ratio, percentage and odds ratio in Python.
7. Fail closed to `cannot_verify` on a missing key, missing source, malformed
   response, unverified span or unresolved check.
8. Return `overstated` only when the source contradicts the claim's population,
   comparator, indication or reported strength.
9. Compare predictions with `data/gold.json` and report coverage, accuracy, a
   confusion matrix and false-supported claims.
10. Replay the deterministic pipeline from the audit log without model calls.
11. Keep a working fallback when the primary provider's quota is exhausted,
    without hiding the failed primary calls.

## Pipeline

### 1. Source resolution (`ingest.py`)

- A `cited_paper` field may name several sources, for example
  `Tabak2020_PSM.pdf (plus LixiLan-L abstract and SoliMix)`. The resolver
  extracts the PDF names and page annotations, searches the project root and
  `papers/`, and returns one `SourceRef` per resolvable PDF.
- A citation that names only a page, such as
  `SOLIQUA_LBL.pdf (Prescribing Information, page 4)`, is one source. The page
  number is not a second source.
- If no source can be opened, the claim fails closed.

### 2. Ingestion (`ingest.py`)

- PyMuPDF (`fitz`) extracts per-page native text, text blocks and Markdown
  tables.
- Tables are supplemental context. A vector figure may expose no usable grid
  even when its labels are readable in the text layer, so page text is the
  primary evidence source.
- A page is marked `needs_vision` when an image covers at least 25% of the page,
  native text is under 1,200 characters, and less than 150 characters of native
  text sit inside the image bounds. A full-page raster with a thin native header
  is also marked for vision when the image covers at least 80% of the page.
  Vector figures expose their labels in the text layer and stay native.
- Native control characters are sanitized. A `\x01` immediately before a digit
  is read as a minus sign; elsewhere it is a diacritic marker and is removed.
  A hyphen at a line break between letters is removed because it is a layout
  artifact, not a missing character in a quote.
- Each page records `text`, `tables`, `has_native_text`, `needs_vision`,
  optional `image_bytes`, `provenance` and later an optional `transcript`.

### 3. Two-stage vision and provider fallback (`extractor.py`)

- Before claim extraction, every image page is transcribed by a separate model
  call. The transcript is cached under `outputs/cache/transcripts/` by SHA-256
  of the image bytes and model name.
- Gemini is primary. If `GEMINI_API_KEY` is missing, quota is exhausted, the
  service is unavailable, a request times out or the response is invalid JSON,
  the same request goes to NVIDIA's OpenAI-compatible endpoint, then to an
  OpenRouter free model. Every failed call is logged; each fallback row records
  `provider`, `fallback` and `fallback_reason`.
- A multi-page transcription request that fails or leaves pages unread is
  retried one page at a time, so one bad page does not discard the rest.
- The parser repairs invalid `\uXXXX` escapes in model JSON instead of
  discarding a usable page.
- The extraction response may also carry `page_transcripts`; these are merged
  only into scanned or image-dominant pages, never over native text. A page
  transcribed before extraction is frozen: the extraction model may not rewrite
  the evidence it is judged against.

### 4. Structured extraction (`extractor.py`)

`ClaimExtractionResponse` contains:

- `population_check`, `drug_indication_check`, `actually_stated_check`:
  `match` | `mismatch` | `unclear`.
- `mismatch_bases`: a closed enum of reasons (`population`, `drug_indication`,
  `dose`, `formulation`, `indication`, `comparator_substitution`, `not_stated`,
  `study_design`, `numeric`, `timepoint`, `endpoint`).
- `evidence_source`: the source label chosen by the model.
- `evidence_page`: 1-indexed PDF page.
- `evidence_span`: an exact contiguous substring of the page text, transcript
  or extracted table text.
- `arithmetic`: `operation` (`none`, `reported`, `difference`, `ratio`,
  `percentage`, `odds_ratio`, `composite`), `claimed_text`,
  `claimed_numeric`, `claimed_values`, `source_operands`, optional
  `reported_value` and optional `components`.
- `supporting_values`: decisive source numbers for a qualitative claim.
- `page_transcripts`: exact text for any attached image page.

The model does not write the reviewer reason. The reason is generated in Python
from the checks, mismatch bases, computed values and the source's own limitation
sentence, so the model cannot introduce a number that is absent from the page.

The prompt is claim-agnostic. It contains no gold answer, forbids outside
knowledge and forbids arithmetic. It forbids ellipses, stitched table cells and
reordered words in the evidence span. It asks for the source's own comparison
result when the claim states a confidence interval or p-value. It defines a
two-arm difference as one component, not one component per number.

### 5. Deterministic validation and one repair (`extractor.py`)

After the first response, Python checks:

- the span is a verbatim contiguous substring of a supplied page;
- every arithmetic operand, with its printed sign, occurs on the evidence page;
- a numeric claim did not return `operation=none`;
- a quoted span reports a different confidence level from the claim (for
  example, the claim states 95% CI but the span reports 97.5% CI);
- a composite did not report an arm value separately and also use the same
  value inside a derived comparison;
- a composite did not return one reported component per number.

If any check fails, one repair call receives the original prompt, the previous
JSON and the exact failures. If the repair still fails, the claim is
`cannot_verify`. No model response is edited.

### 6. Deterministic verification (`verification_engine.py`)

- **Span validation.** Normalize case, whitespace, dash variants and ligatures
  (`±`, `−`, `–`, ﬁ). Accept only a contiguous substring after normalization,
  or a whitespace-compacted contiguous substring of at least eight characters.
  If a page has a frozen transcript, only that transcript counts; otherwise
  native text and extracted tables count. Word order is never ignored. Every
  number used by the arithmetic must occur on the cited page. An ellipsis is
  rejected because it is an authorial join, not a verbatim passage.
- **Evidence location.** Prefer the model's source and page when the quote
  really occurs there; otherwise search every source in document order. If no
  page contains the quote, the claim fails closed.
- **Limitation evidence.** When a mismatch basis is `study_design`, search the
  cited source for its own limitation sentence (`post hoc`,
  `hypothesis generating`, `not randomized`, `exploratory`, `retrospective`). A
  matching sentence replaces a merely descriptive quote only when it has more
  limitation markers. It is still a verbatim page substring.
- **Arithmetic.**
  - `difference`: prefer the source-printed signed difference; otherwise
    subtract the arm values. Match within 0.05 of the claimed magnitude. A
    reversed sign is a `mismatch` only when the source operands carry signs.
    With unsigned magnitudes, a direction conflict is `unclear`, not a guessed
    increase.
  - `ratio`: `abs(a / b)`, match within 15% relative error.
  - `percentage`: `(a / b) * 100`, match within 0.1 percentage points; the
    four-operand form checks both arms when both claimed percentages are
    supplied.
  - `odds_ratio`: when the source-printed adjusted OR is supplied, it is
    authoritative and must match within 0.05. Otherwise compute the crude 2×2
    OR from events and totals and match within 0.10.
  - `reported`: a lone printed value, match within 0.05.
  - `composite`: every component must match; one `mismatch` makes the whole
    figure `mismatch`, and any `unclear` makes it `unclear`.
  - A numeric claim with no extracted arithmetic is `unclear`, never absent.
    Detection covers percentages, units, `OR`/`HR`/`RR`, ratios and unitless
    numeric comparisons such as `1.3 vs 1.1`; the ordinary word "or" is not an
    odds ratio.
- **Figure coverage.** A completeness check compares the numbers the claim
  asserts with the operands and computed values the extraction supplied. It is
  not operation inference: it only reports an unaccounted comparison percentage,
  `OR`/`HR`/`RR` value or `Nx` ratio. A gap makes the figure `unclear`.
- **Span-number provenance.** For a `match`, the quote must contain at least one
  claimed value and every source operand must occur on the cited page. For a
  `mismatch`, the quote must contain at least one real source operand. Signs are
  preserved, so an invented positive value cannot be certified by a negative
  source value of the same magnitude.
- **Verdict.** An unverified span is `cannot_verify`. Otherwise any `mismatch`
  is `overstated`, any `unclear` is `cannot_verify`, and a full set of matches
  is `supported`. A sole `numeric` mismatch is reconciled only when the
  arithmetic matches and the claimed confidence level and p-value occur in the
  span; the model's numeric objection is then contradicted by code. If a
  multi-source claim is `supported` but one cited source is missing, the verdict
  is downgraded to `cannot_verify`. A contradiction in an available source can
  still justify `overstated`. A mismatch basis with no mismatch check is a
  contradictory extraction and fails closed.

The public prediction has exactly the assignment's top-level keys: `claim_id`,
`verdict`, `checks`, `evidence_page`, `evidence_span`, `computed` and `reason`.
`checks` holds the four checks. Multi-source provenance is written into `reason`
as `Source: <label>, page <n>.`

### 7. Replay (`verify.py`)

`python verify.py <claims> <output> --replay outputs/llm_log.jsonl` rebuilds
predictions from the last successful extraction per claim and the logged page
transcripts. It re-ingests the PDFs and reruns span, arithmetic and verdict
logic. No model call is made. This is the audit path for a code change.

### 8. Evaluation (`eval.py`)

- Match predictions to gold by `claim_id`.
- A prediction absent from the file is `missing`, not a correct
  `cannot_verify`; it reduces coverage and earns no credit.
- Report total, present, coverage, correct, accuracy, a 3×3 confusion matrix,
  supported precision and recall, false-supported / false-overstated /
  false-cannot-verify claims, check agreement, evidence-page agreement, span
  containment, span similarity, computed-operand coverage, an always-overstated
  baseline and an audit-log summary.
- Exit non-zero if a required file is absent.

## Configuration

- `GEMINI_API_KEY`, optional `NVIDIA_API_KEY` and optional `OPENROUTER_KEY`
  from the environment or a gitignored `.env`.
- `GEMINI_MODEL` or `--model` selects the primary model. The default is
  `gemini-3-flash-preview`.
- `NVIDIA_EXTRACTION_MODEL` and `NVIDIA_VISION_MODEL` select the first
  fallback models. The defaults are `nvidia/nemotron-3-ultra-550b-a55b` and
  `nvidia/nemotron-3.5-super-vl-120b-a12b`.
- `OPENROUTER_TEXT_MODEL` and `OPENROUTER_VISION_MODEL` select the final free
  fallback models.
- Missing all keys: no request, every claim `cannot_verify` with a clear
  missing-key reason.
- `.env`, `outputs/cache/`, `.venv/` and archives are gitignored.

## Testing

The suite is `python -m unittest discover -s . -p "test_*.py"`. It makes no
network calls. It covers:

- arithmetic tolerances, missing operands, signed and unsigned direction
  conflicts, reported versus crude odds ratios, composite components and false
  ratios;
- span matching across case, whitespace, ligatures, invented numbers, ellipsis
  stitching, control-character minus signs, line-break hyphenation and exact
  contiguity;
- unitless numeric claims and the rule that the word "or" is not an OR;
- every branch of the verdict resolver;
- multi-source citation resolution and page annotations;
- transcription caching, per-page retry, JSON escape repair and the rule that a
  transcript cannot overwrite native page text;
- provider fallback selection, NVIDIA request construction, failed-primary
  logging and missing-key fail-closed behaviour;
- audit rows carrying the system instruction and image hashes;
- limitation-sentence selection, code-owned figure reasons and removal of
  internal rule numbers from reviewer prose;
- replay from a synthetic audit log;
- evaluation honesty for missing predictions;
- an end-to-end scripted pipeline for supported, comparator-substitution,
  study-design and invented-span claims;
- a prompt check that no gold answer text appears in the prompt.

The primary seam is the CLI: `verify.py` then `eval.py`. Unit seams are
`verify_arithmetic`, `check_span_presence`, `resolve_verdict` and
`ClaimVerifier` with injected extractor/transcriber dependencies.

## Out of scope

- Literature discovery and paper retrieval.
- Full OCR training or external OCR binaries.
- Statistical re-estimation of covariate-adjusted models; the pipeline compares
  reported values and computes crude checks only.
- A UI. The deliverable is a CLI and JSON artifacts.

## Limits

The checked-in result is seven claims. It is not a general accuracy estimate.
Raster transcripts, model judgement on `supported` verdicts and the model's
choice of evidence remain the main sources of error. The audit log shows which
provider served each call and why a fallback occurred.
