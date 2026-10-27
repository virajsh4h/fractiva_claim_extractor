# Engineering Write-up: Clinical Claim Verification Mini-Pipeline

**Scope.** Seven SOLIQUA claims checked against the cited prescribing
information and trial papers. `verify.py` runs the pipeline. A model locates
evidence and answers three structured checks; Python does every calculation,
validates every quote and resolves every verdict.

## 1. Approach

Promotional claims fail predictably: a population restriction disappears, a
comparator is renamed, a post-hoc analysis becomes a "trial", or a number is
rounded up. A single LLM fact-checker merges separate regulatory questions and
does arithmetic in prose. The pipeline separates judgement from calculation.

1. **Ingest** (`ingest.py`). PyMuPDF extracts text, blocks and tables. Raster
   pages are flagged for vision; vector figures with readable labels stay
   native. Control-character minus signs and line-break hyphens are repaired.
   Multi-source citations expand into independent sources.
2. **Transcribe** (`extractor.py`). Image pages are transcribed once, before
   claim judgement, and cached by image hash and model. The transcript is
   frozen: extraction cannot rewrite the evidence it is judged against.
3. **Extract** (`extractor.py`). A claim-agnostic prompt and Pydantic schema
   return three checks, structured mismatch bases, one verbatim span, page
   provenance and raw operands. Arithmetic, ellipses and stitched cells are
   forbidden. The model does not write the reviewer reason. Every call is
   logged; Gemini is primary, with NVIDIA and OpenRouter free models as tested
   fallbacks.
4. **Validate and repair** (`extractor.py`). Python checks that the quote is a
   contiguous page substring or a verbatim table row, that every operand with
   its printed sign occurs on that page, and that a composite is not a list of
   numbers pretending to be one comparison. One repair call receives the exact
   failures. A failed repair is `cannot_verify`.
5. **Verify and resolve** (`verification_engine.py`, `verify.py`). Python
   recomputes differences, ratios, percentages and odds ratios. A quote must be
   verbatim, contain a claimed value or a real operand, and contain every
   number used. A `mismatch` gives `overstated`; an `unclear`, unverified span
   or missing source gives `cannot_verify`; only all matches gives `supported`.
   Reasons are generated in code. Missing predictions score `missing`.

## 2. Results

One end-to-end fallback-path run made 18 model calls: 6 transcription calls and
12 extraction or repair calls. Gemini was not called in this run; NVIDIA served
one repair and OpenRouter free models served the rest. Deterministic replay of
that log with the final checker produced:

```
Total gold claims          : 7
Correct verdicts           : 7
Verdict accuracy           : 100.0%
Supported precision/recall : 100.0% / 100.0%
False supported            : 0
False overstated           : 0
False cannot_verify        : 0
Checks exactly matching    : 7/7
Evidence page matches      : 6/7
Computed operands matching : 6/7
Trivial baseline           : 57.1% (always overstated)
```

The audit log is `outputs/llm_log.jsonl`; it records each provider, fallback
reason, raw response and failed attempt. The checked-in predictions are the
deterministic replay of that log after a table-evidence fix: table text is now
searchable evidence, so C4's verbatim table row validates. No model response
was edited and no run was stitched together. Seven fixed development claims are
a regression set, not a held-out accuracy estimate.

## 3. The hardest table

SoliMix Figure 1B (p. 5) is a two-arm forest plot drawn as vector graphics.
`find_tables()` returned fragments with repeated headers and empty cells; the
numbers were never in a usable grid. Page text and block order had to be
primary, with tables as supplemental context.

The remaining defect is visible in the C2 quote: the page's native text order
interleaves the superiority row with axis ticks and baseline labels. The span is
verbatim and contains both arm values, but it is ugly. A reviewer can check it;
a promotional reviewer would prefer the row alone. I kept the larger quote
rather than widen or stitch a shorter one, because a stitched quote is not
verbatim.

## 4. One real failure

C2 was first marked `overstated` although the page states the claim. The NVIDIA
fallback returned a composite with one reported component for `-1.3`, one for
`-1.1`, plus a separate difference component. Every number existed on the page,
so a shallow check would pass. Python then summed three independent components
instead of one difference and the figure check failed.

The fix is structural, not claim-specific: a composite that reports an arm
value separately and also uses the same value inside a derived comparison fails
validation and triggers one repair call. NVIDIA's repair returned one
difference component with operands `[-1.3, -1.1]` and reported value `-0.2`.
C2 then became `supported` with the correct page, span and arithmetic. This was
a false negative before the fix, not a false supported, but it showed that
"valid JSON" is not the same as "valid evidence structure".

## 5. Where AI misled me

Two examples. First, an early NVIDIA response for C2 looked authoritative and
passed schema validation, but its arithmetic shape was wrong: it treated each
arm and the difference as independent results. The deterministic composite check
caught it. The lesson was to validate structure, not just JSON types.

Second, the first fallback implementation retried a four-page Gemini
transcription request after a hard quota error, then retried each page, turning
one exhausted key into many minutes of pointless calls. The audit log made the
pattern obvious. The fix treats quota and availability errors as fatal for that
claim and moves immediately to NVIDIA, while still retrying transient NVIDIA
500s per page. Every failed call remains in the log.

## 6. What not to trust

- **Raster transcripts.** The leave-behind and the LixiLan-L abstract are image
  pages. A misread digit or a dropped restriction changes a verdict.
- **LLM judgement on `supported`.** C7 shows a post-hoc re-analysis can look
  like a third trial unless the model is asked what kind of evidence it is.
- **Provider consistency.** Gemini, NVIDIA and OpenRouter do not return
  identical extractions. The checked-in run used the fallback path and
  discloses which provider served each call.
- **Close calls.** C4 is an exploratory endpoint read literally; C5 reads
  "weight neutral" against a measured 1.4 kg difference; C7 reads "three trial
  panels" as an independence claim. A stricter reviewer could move C4 or C5 to
  `cannot_verify`. Gold discloses each choice.
- **The result.** Seven fixed claims, one run. It is not a general accuracy
  estimate.

## 7. Next step

Add an adversarial second pass for every `supported` verdict: a fresh prompt
that looks only for a verbatim limiting sentence (post-hoc, subgroup,
exploratory, comparator substitution, population restriction). If found,
downgrade to `cannot_verify`. Measure false-supported and unnecessary
`cannot_verify` rates on new claims; the first number must stay at zero.
