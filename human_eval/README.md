# Human reply-quality evaluation (Phase 6F)

**Rater: read "Rating procedure" and "What you must not open" before starting.**

## Purpose

The deterministic G1–G7 checks detect specific patterns (URLs, numbers, action claims and
so on). They do not tell whether a reply is relevant, helpful, grounded, asks for the right
information, or avoids unsupported claims. The LLM judge meant to measure this failed its
pre-declared validation (docs/DECISION_LOG.md D46) and is not used. This directory holds a
**single human annotator's** ratings of reply quality instead.

These ratings are evidence about the frozen system. **They never feed back into it**: no
training, retrieval, prompt, classifier, threshold or escalation change may use them. They
are kept apart from `golden/` and are never joined to the gold intent labels.

## Population and sample

| | |
| --- | --- |
| Source | the 248 frozen Phase 6E evaluation rows (`data/processed/phase6/evaluation_rows.jsonl`) |
| Excluded before sampling | `b01_0001`, `b01_0019`, `b01_0057`, `b01_0074`, `b02_0012` (their message or reply text was shown during development) |
| Population | 243 rows |
| Sample | 40 rows, **40 items × 3 responses = 120 responses** |
| Method | proportional stratified random sampling by batch × escalation decision, seed **46** |
| Strata (population → sample) | b01 auto-handled 99 → **16**; b01 escalated 45 → **7**; b02 auto-handled 66 → **11**; b02 escalated 33 → **6** |

Selection used only the batch and the frozen escalation decision. It used no gold or
predicted intent, reply quality, G1–G7 result or similarity score. The sampled row ids and a
selection hash are in `reply_rating_manifest.json`.

The three responses per item are:
- the LLM system's reply;
- Baseline A (the top historical reply, sanitised);
- Baseline B (a fixed generic reply).

No reply was regenerated.

## Blinding

- **One row per response.** All 120 rows are shuffled (seed 46) so that no two neighbouring
  rows belong to the same item.
- **Opaque ids.** Items are `I01`–`I40`; responses are `R001`–`R120`. Both are assigned after
  shuffling.
- **Shown to the rater:**
  - response id and item id;
  - the customer message (with @mentions removed);
  - the five historical examples exactly as the generator saw them;
  - the candidate reply.
- **Never shown:**
  - which system wrote the reply;
  - annotation or conversation ids;
  - gold or predicted intent, and classifier confidence;
  - the escalation decision, G1–G7 flags, similarities, parse status, or any automated score.
- **The system mapping is not stored anywhere.** It is rebuilt from the frozen inputs and
  the seed at analysis time, and the manifest holds only its SHA-256.
- **One sampled LLM reply is empty.** It was not replaced. Its row reads `[NOT RATED - ...]`
  and is recorded as `not_rated_empty_reply`.

## Rating procedure

1. Open `reply_rating_rubric.md` and keep it beside you.
2. Copy `reply_rating_r01_blank.csv` to **`reply_rating_r01_rated.csv`** and rate only the
   copy. Never edit the blank file.
3. For every row, enter whole numbers 1–5 in `relevance`, `helpfulness`, `groundedness`,
   `information_request` and `claim_safety`. `notes` is optional.
4. Rate each row **on its own**, against the rubric. Do not compare the three responses to
   the same item.
5. Leave all scores blank on `[NOT RATED - ...]` rows.
6. Do not change any other column, reorder rows, or add columns.
7. Two sittings are fine. Write down the date and time you start and finish each one.
8. When finished, say so. Round 1 is then only **checked for completeness**; no scores are
   computed or shown before the retest is done.

## What you must not open while rating

- `reply_rating_manifest.json`
- `src/build_reply_rating_batch.py` or anything that rebuilds the mapping
- `data/processed/phase6/`, `reports/`, `golden/`
- during the retest: your round-1 file

## Test-retest (batch built)

- **Subset:** 36 of the rated responses, 12 per system, seed **47**. The empty reply is
  never included.
- **Blinding:** new opaque ids (`T01`–`T36`), a new order, and no round-1 scores shown.
- **Gap:** round 1 was completed at **2026-09-16T21:39:47+05:30**.
  - **Do not start before 2026-09-19T21:39:47+05:30** (72 hours).
  - **7 days, from 2026-09-23T21:39:47+05:30, is preferred.**
  - Record the time you start and finish.
- **Procedure:**
  - Copy `reply_rating_retest_blank.csv` to **`reply_rating_retest_rated.csv`** and rate
    only the copy, using the same rubric and rules as round 1.
  - **Every row must get all five scores**, as plain whole numbers (`4`, not `4.0`).
  - **Do not open your round-1 file or either manifest while rating.**
- **Analysis:** agreement per dimension (exact, within one point, mean absolute difference,
  quadratic-weighted Cohen's κ with a bootstrap interval) and a pooled summary.

This measures the **same annotator's** consistency: **intra-annotator short-gap test-retest
agreement**. It is not inter-annotator agreement and not independent corroboration. It is
also separate from the 45-item intent-label retest in Phase 5.

## Limitations

- **One annotator, who also built the system.** Bias toward it cannot be excluded.
- **Weak blinding.** Baseline B is one fixed sentence, and Baseline A repeats evidence item 1
  word for word, so both are recognisable and the LLM reply can be identified by elimination.
  Ratings are therefore absolute ratings against the rubric, and comparisons between systems
  are descriptive only: no ranking and no winner.
- **Familiar messages.** The rater labelled these messages in Phase 5 and may remember them.
  The intent is not shown, and the task rates the reply, not the intent.
- **Small sample.** Forty items give wide intervals; results describe this sample only.
- **Other effects.** Fatigue, rating order, and drift in how the rubric is applied between
  sittings.
- **No external check.** These ratings are not validated ground truth.

## Files

| File | What it is |
| --- | --- |
| `reply_rating_rubric.md` | the 1–5 rubric |
| `reply_rating_r01_blank.csv` | the blank round-1 batch (do not edit) |
| `reply_rating_r01_rated.csv` | your completed round-1 ratings (you create this) |
| `reply_rating_manifest.json` | provenance: seeds, population, sample, hashes. No mapping. **Do not open while rating.** |
| `reply_rating_retest_blank.csv` | the blank retest batch, 36 rows (do not edit) |
| `reply_rating_retest_rated.csv` | your completed retest ratings (you create this) |
| `reply_rating_retest_manifest.json` | round-1 validation facts, retest timing, seed and hashes. No ratings, no mapping. **Do not open while rating.** |

The batch is built by `python -m src.build_reply_rating_batch --round r01`, and can be
checked against disk with `--verify`.
