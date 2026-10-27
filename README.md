# Clinical Claim Verification Pipeline

A fail-closed neuro-symbolic pipeline that audits pharmaceutical promotional
claims against the cited prescribing information and trial publications. A model
locates evidence, quotes one verbatim span, extracts raw numbers and answers
three structured judgement checks. Python does every calculation, validates
every quote and resolves every verdict.

## Quickstart (5 commands)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GEMINI_API_KEY and optional NVIDIA_API_KEY
python verify.py data/claims.json
python eval.py outputs/predictions.json data/gold.json
```

`verify.py` reads any claims file with the same schema and writes
`outputs/predictions.json`. The default primary model is
`gemini-3-flash-preview`. Override it with `--model` or `GEMINI_MODEL`.

If Gemini is missing, rate limited, unavailable or returns invalid JSON, the
same request goes to NVIDIA, then to OpenRouter's free models. The fallback
models are configured in `config.py` and can be overridden with
`NVIDIA_EXTRACTION_MODEL`, `NVIDIA_VISION_MODEL`, `OPENROUTER_TEXT_MODEL` and
`OPENROUTER_VISION_MODEL`. Every failed primary call remains in the audit log
with its provider and fallback reason.

## Audit replay (no model calls)

Every request and raw response is appended to `outputs/llm_log.jsonl`. The
deterministic half of the pipeline can be rerun from that log:

```bash
python verify.py data/claims.json outputs/predictions.json \
  --replay outputs/llm_log.jsonl
```

Replay re-ingests the cited PDFs, restores logged page transcripts, then
recomputes span checks, arithmetic and verdicts. It is the fastest way to audit
a code change without spending model quota.

## Tests

```bash
python -m unittest discover -s . -p "test_*.py"
```

The suite makes no network calls. It covers ingestion, exact span matching,
arithmetic tolerances and signs, multi-source resolution, provider fallback and
request construction, fail-closed verdicts, limitation evidence, replay and
evaluation honesty. See `docs/SPEC.md` for the full behavioural contract.

## Credentials

- `GEMINI_API_KEY` is primary; `NVIDIA_API_KEY` and `OPENROUTER_KEY` enable
  the fallback chain.
- Keys are read from the environment or a local `.env` file.
- `.env`, `outputs/cache/` and archives are gitignored. No key is written to
  source, predictions or the audit log.
- Missing all keys: every claim resolves to `cannot_verify`; no network
  request is made.

## Fail-closed rules

- A claim is never `supported` unless its quoted span is found verbatim on a
  cited page, every relevant check is `match`, every arithmetic component
  matches, and the span contains the numbers used.
- Missing operands, a missing part of a composite figure, a numeric claim with
  no extracted arithmetic, or a span containing numbers that do not occur on
  the page resolves to `unclear`/`cannot_verify`.
- A claim citing several sources is downgraded to `cannot_verify` if any cited
  source cannot be opened, unless an available source already proves the claim
  is `overstated`.
- A `cannot_verify` check is never converted into a match.
- `eval.py` counts a missing prediction as `missing`, not as a correct
  `cannot_verify`.

## Architecture

1. **Ingestion (`ingest.py`)** - PyMuPDF extracts native text, block layout and
   Markdown tables. A page is marked for vision when a raster image dominates
   it and its native text sits inside the image; vector figures with readable
   labels stay native. Control-character minus signs and line-break hyphens are
   repaired. Multi-source citations such as `A.pdf (plus B and C)` resolve to
   independent sources.
2. **Two-stage vision (`extractor.py`)** - Image pages are transcribed first,
   in a separate model call, and the transcript is cached by image hash and
   model. Gemini is primary; a missing key, quota failure, timeout or invalid
   response falls back to NVIDIA. Claim extraction then reads text and frozen
   transcripts. The extraction schema also accepts page transcripts, so a
   missed first pass can still be recovered.
3. **Structured extraction (`extractor.py`)** - A claim-agnostic prompt with a
   Pydantic response schema returns three judgement checks, structured mismatch
   bases, one verbatim span, page provenance and raw operands. The prompt
   contains no answer key and forbids arithmetic. It also forbids ellipses and
   stitched table cells. The model does not write the reviewer reason.
4. **Validation and repair (`extractor.py`)** - Python checks that the span is
   contiguous and verbatim, that every operand occurs on the evidence page,
   that a quoted span does not report a different confidence level from the
   claim, and that a composite is not a list of numbers pretending to be one
   comparison. One repair call receives the exact failures. A failed repair is
   `cannot_verify`.
5. **Deterministic verification (`verification_engine.py`)** - Python recomputes
   ratios, differences, percentages and odds ratios. The reported adjusted OR
   is preferred over the crude 2x2 calculation when both are supplied. The span
   checker is case-, whitespace- and ligature-tolerant, but word order and
   numeric provenance are enforced; an ellipsis is rejected. A match must quote
   a claimed value or a real source operand, and every operand must occur on the
   cited page. Signed values cannot be replaced by unsigned magnitudes.
6. **Verdict resolution (`verify.py`)** - A `mismatch` produces `overstated`; an
   `unclear` or unverified span produces `cannot_verify`; only a full set of
   matches produces `supported`. A sole `numeric` mismatch is cleared when the
   arithmetic matches and the claimed confidence level and p-value occur in the
   span, because code has then contradicted the model's only stated objection.
   A study-design mismatch prefers the source's own limitation sentence. The
   reviewer reason is generated in code from the checks, mismatch bases,
   computed values and limitation text. `domain.py` holds the enums and
   provenance types. `audit.py` writes the verbatim prompt, SHA-256 hashes,
   system instruction, image hashes, raw response, provider and fallback
   metadata.

## Results

See `WRITEUP.md` for the actual evaluation output and the close calls. The
checked-in `outputs/llm_log.jsonl` comes from one end-to-end run over all seven
claims. `outputs/predictions.json` is the deterministic replay of that log with
the final checker; no model response was edited and no run was stitched
together.

## Repository layout

```
README.md                  Setup, replay and architecture
requirements.txt           Pinned dependencies
.env.example               Empty key template
verify.py                  Pipeline, CLI and replay
eval.py                    Accuracy, coverage and confusion matrix
ingest.py                  PDF text, tables, vision detection, source refs
extractor.py               Prompt, schema, transcription, provider fallback, audit logging
verification_engine.py     Arithmetic, span checks, verdict resolver
domain.py                  CheckStatus, Verdict, EvidenceLocation, SourceRef
config.py                  Model and API-key configuration
audit.py                   JSONL request/response log
papers/                    Three cited trial publications
SOLIQUA_LBL.pdf            Scanned promotional leave-behind
data/claims.json           Provided claims
data/gold.json             Audited reference verdicts
outputs/predictions.json   Final predictions
outputs/llm_log.jsonl      Every model call, including failed primary calls
docs/SPEC.md               Behavioural specification
docs/adr/                  Architecture decision records
GLOSSARY.md                Domain vocabulary
WRITEUP.md                 Approach, results, failures, next step
```
