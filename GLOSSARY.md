# Glossary: Clinical Claim Verification

Vocabulary used by the verifier. The code is the source of truth; this file
names the concepts so code, tests and write-up use one term each.

## Claim and sources

**Claim**
A promotional assertion about a drug's indication, efficacy or safety, with a
population, a molecule and one or more cited sources.

**Cited source**
A PDF named by a claim. A citation may name several sources, for example
`Tabak2020_PSM.pdf (plus LixiLan-L abstract and SoliMix)`. The resolver expands
this to three independent `SourceRef` values.

**Source label**
The basename used to identify a source in a span, transcript or prediction,
for example `SoliMix_Rosenstock2021.pdf`.

**Page provenance**
The 1-indexed PDF page on which evidence was found. It is not a journal page
number. `EvidenceLocation` pairs a source label with this page.

## Evidence

**Transcript**
Text produced by a vision model from an image-only page. A transcript is stored
on the page and used by the span checker exactly like native text. Transcripts
are cached by image hash and model.

**Verbatim evidence span**
An exact contiguous substring of a page's text or transcript. The span checker
normalizes case, whitespace and typographic characters, but it never reorders
words, accepts a token bag, inserts an ellipsis or widens the quote. Every
number used by the arithmetic must occur on the cited page.

**Limitation evidence**
A sentence in the cited source that says the analysis is post-hoc, exploratory,
not randomized or hypothesis-generating. When the model marks a study-design
mismatch, the verifier prefers this sentence over a descriptive quote. It is
still a verbatim page substring.

**Source operand**
A raw number read from the source and passed to the arithmetic engine. The
model may extract operands; it may not calculate with them.

**Composite figure**
A claim that asserts several independent results at once. Each result is an
`ArithmeticComponent`. A two-arm difference is one component, not one component
per number. A composite that reports the same arm value separately and inside a
derived comparison fails deterministic validation and triggers one repair call.

## Checks and verdicts

**Check**
One regulatory question about a claim. The pipeline reports four:
`figure`, `population`, `drug_indication` and `actually_stated`. `figure` is
`null` for a non-numeric claim.

**CheckStatus**
One of `match`, `mismatch` or `unclear`. `match` means the supplied source
explicitly supports the claim element. `mismatch` means the source contradicts
it, narrows it, or supports only a weaker statement. `unclear` means the source
cannot decide it.

**Verdict**
One of `supported`, `overstated` or `cannot_verify`.
- `supported`: every applicable check matches, the span is verbatim, every
  arithmetic component matches, and the span contains the numbers used.
- `overstated`: at least one check is `mismatch` and the span is verbatim.
- `cannot_verify`: the span is unverified, a check is `unclear`, a source is
  missing, extraction failed, or deterministic validation failed.

`overstated` is an affirmative finding. `cannot_verify` is an absence of proof.
They are not interchangeable.

**False supported**
A prediction of `supported` for a claim whose audited verdict is not
`supported`. This is the critical safety error; `eval.py` counts it separately.

**Coverage**
Predictions present divided by gold claims. A missing prediction is scored as
`missing` and earns no credit, even when the gold verdict is `cannot_verify`.

## Pipeline properties

**Fail-closed**
A design rule: missing evidence, malformed output, an unverified span, an
unresolved check or a failed provider call produces `cannot_verify`, never
`supported`.

**Provider fallback**
Gemini is the primary model. If its key is missing, its quota is exhausted, the
service is unavailable, it times out, or it returns invalid JSON, the same
request is sent to NVIDIA. Every failed primary call stays in the audit log,
and each fallback row records `provider`, `fallback` and `fallback_reason`.

**Repair pass**
One additional model call after deterministic validation finds a structurally
invalid answer: a span that is not verbatim, arithmetic operands absent from the
cited page, a quoted confidence level that differs from the claim, a numeric
claim with no arithmetic, or a malformed composite. The repair prompt contains
the exact failures. If the repair also fails validation, the claim is
`cannot_verify`.

**Numeric reconciliation**
A narrow code-owned rule. If the model's only mismatch basis is `numeric`, the
computed figure matches, and the claimed confidence level and p-value occur in
the span, the numeric objection is contradicted by code and the claim can be
`supported`. Any other mismatch basis keeps the verdict unchanged.

**Replay**
Rebuilding predictions from `outputs/llm_log.jsonl` without calling a model.
Replay re-ingests the PDFs, restores logged transcripts, and reruns the
deterministic span, arithmetic and verdict code against the recorded model
responses.

**Audit log**
The JSONL file `outputs/llm_log.jsonl`. Each row records the model, verbatim
prompt, prompt SHA-256, system instruction and its SHA-256, image part hashes,
raw response, parameters, latency and metadata such as claim ID, task, provider,
fallback reason and error flag.

**Code-owned reason**
The reviewer reason is generated by Python from the checks, mismatch bases,
computed values and the source's limitation sentence. The model does not write
free-text prose, so it cannot introduce a number or a claim that the evidence
does not support.

**Neuro-symbolic verifier**
The architecture used here: an LLM locates evidence, extracts raw operands and
answers three structured judgement checks; Python performs arithmetic, span
validation and verdict resolution.
