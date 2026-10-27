# ADR 0001: Neuro-Symbolic Verification Instead of a Decision Model

## Context

A regulatory reviewer needs a verdict, a verbatim source quote and the
arithmetic behind the verdict. A classifier that returns only a label cannot
supply that evidence. A generative model that calculates in prose can invent a
number and cannot be audited mechanically.

## Decision

Use a hybrid neuro-symbolic pipeline:

- The model is an evidence locator and a structured judge. It returns three
  `match` / `mismatch` / `unclear` checks, one verbatim evidence span, page
  provenance, raw source operands and structured mismatch bases.
- Python performs every difference, ratio, percentage and odds-ratio
  calculation, validates the span against the cited page, resolves the verdict
  and writes the reviewer reason.
- The response is validated with a Pydantic schema. A deterministic repair pass
  handles a non-verbatim span or arithmetic that does not match the cited page.
- Every model call is logged with its raw response. Gemini is primary; a
  missing key, quota failure, service failure or invalid response falls back to
  NVIDIA, and every failed primary call remains in the log.

## Rejected options

- **Non-generative decision model.** Fast for classification, but it cannot
  produce a verbatim span or the raw operands a reviewer can check.
- **Generative model with chain-of-thought arithmetic.** Flexible, but it puts
  calculation in prose, which the assignment forbids and which cannot fail
  closed on a wrong number.
- **LLM-written reviewer prose.** Plausible but unauditable. The model can
  introduce a number that is absent from the page. The final reason is therefore
  generated in code.

## Consequences

- The pipeline needs a model that can follow a JSON schema and quote text.
- Arithmetic and verdict resolution are deterministic and replayable from the
  audit log.
- An invented or misplaced quote fails closed instead of becoming evidence.
- Provider fallback keeps the pipeline usable when the primary free tier is
  exhausted, at the cost of a second provider path that must be tested and
  disclosed.
- A model judgement can still be wrong. The design limits the damage: a wrong
  `match` cannot produce `supported` unless the quote is verbatim and the
  arithmetic reconciles.
