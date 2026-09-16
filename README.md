# Hiver Support Agent

An AI customer-support agent for **@AmericanAir**, built on the *Customer Support on
Twitter* dataset. For each incoming customer message it:

1. classifies the message into one of 13 intents;
2. retrieves how the brand handled similar messages and drafts a reply grounded in them;
3. decides **auto-handle or escalate, with auditable reason codes**.

The project is **evaluation-first**. **Start with [`REPORT.md`](REPORT.md)**: results,
failure modes, what the headline numbers do not show, and key decisions.

| Document | Purpose |
| --- | --- |
| [`REPORT.md`](REPORT.md) | Final report (5 A4 pages) |
| [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) | 49 non-obvious decisions, with alternatives and tradeoffs |
| [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md) | What each file does, and which contain core logic |
| [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md) | Every step, error and fix, in order |

---

## Headline results

All results are on 248 held-out golden messages labelled by one annotator. **b02 (100 rows)
is the cleaner, independently sampled subset.** Full tables and caveats are in `REPORT.md`.

| Component | Result (all 248 / b02) | Baseline | What it is **not** |
| --- | --- | --- | --- |
| Intent: TF-IDF + LR on weak labels | accuracy **0.472** / 0.450; macro-F1 0.358 / 0.322 | majority class 0.117 / 0.100 | not an upper bound: 41 gold rows use labels the classifier cannot predict |
| Retrieval, k=5 | gold-intent vs weak-label agreement **39.83%** / 40.80% | random evidence 12.13% / 12.04% | not retrieval accuracy (evidence has only weak labels) |
| Generation, `llama3.1:8b` | parsed 95.97%; **all G1–G7 checks passed 95.16%** | fixed reply 100%; top-1 echo 0.4% | not correctness |
| Escalation | **32.3%** escalated / 34.0%, each with reason codes | — | not an error rate (no outcome labels) |
| Intent-label reliability | intra-annotator retest: raw 84.4%, κ **0.8281** [0.7039, 0.9254] | — | not inter-annotator agreement |
| Reply quality | one round, 119 ratings by the developer, AI-assisted; retest **not completed** | two baselines rated alongside | not independent or validated human evaluation |
| LLM-as-judge | **failed** pre-declared validation (4 configurations) and is not used | — | — |

## Why AmericanAir

The 25 largest brands went through four gates (volume, public resolution, language,
support purity) before a weighted score (D22, D23).

| | AmericanAir | Delta | AmazonHelp |
| --- | --- | --- | --- |
| Clean conversations | 24,429 | 24,799 | 78,763 |
| DM deflection | **21.1%** | 25.5% | 1.5% |
| URL rate | **8.3%** | 23.5% | 65.6% |
| English (heuristic) | 99.2% | 99.8% | **81.9%** |

- **AmericanAir answers in prose, in public**, which is good material to ground replies in.
- **AmazonHelp** is 18.1% non-English, and most of its replies are links.
- **AppleSupport** pushes 64.4% of conversations to DMs, where resolutions are invisible.
- **The weighted score ranked AmericanAir sixth,** and the weights were not changed to hide
  that (D23).
- **Almost all the data (99.8%) falls in Oct–Dec 2017.**

## Architecture

```text
raw tweets (2.81M) ─► conversation reconstruction (798,197 components; 741,110 clean dyads)
                   ─► AmericanAir corpus (24,178) ─► leakage exclusion ─► 23,722 conversations
                                                                           │
customer message ─► intent classifier (TF-IDF + LR, trained on 9,423 weakly labelled messages)
                 ─► retrieval (TF-IDF over customer openings, k = 5, own conversation and customer excluded)
                 ─► generation (Ollama llama3.1:8b, temperature 0, seed 42, strict JSON)
                 ─► G1–G7 deterministic checks
                 ─► escalation esc-v1 ─► auto_handle | escalate + reason codes
```

| Stage | Module | Key facts |
| --- | --- | --- |
| Profiling | `src/profile_raw.py` | reads every column as text; cycle detection on a functional graph |
| Reconstruction | `src/reconstruct.py` | edges only from `in_response_to_tweet_id`; pointer doubling; broadcasts removed by participant structure |
| Brand selection | `src/analyze_brands.py` | gates, then score |
| Intent discovery | `src/discover_intents.py` | TF-IDF clustering, **reported as inadequate** (silhouette ≤ 0.054) |
| Corpus and weak labels | `src/build_corpus.py`, `src/taxonomy.py`, `src/leakage.py` | shared golden exclusion: conversation, customer, exact text, near-duplicate |
| Intent classifier | `src/classify_intent.py` | TF-IDF (1–2-grams) + balanced logistic regression, seed 42 |
| Retrieval | `src/retrieve.py` | deterministic tie-breaking; a random baseline over the same pool |
| Generation and checks | `src/generate_reply.py`, `src/llm.py` | `urllib` only; disk cache keyed by prompt, model, temperature, seed and format |
| Escalation | `src/escalate.py`, `src/calibrate_escalation.py` | 7 ordered rules; every reason recorded; thresholds are golden-free 20th percentiles |
| Evaluation | `src/evaluate.py` | the only code that reads gold labels, and only with `--confirm-golden-run` |
| Reply-quality ratings | `src/build_reply_rating_batch.py`, `src/analyse_reply_ratings.py` | blinded batch; analysis written and tested on synthetic data only |

**The seven escalation rules**, in order:

1. generation unusable;
2. G2–G7 flag;
3. invalid evidence citation;
4. fewer than 2 evidence items;
5. explicit human-review policy (`staff_and_service_complaint`, `general_dissatisfaction`;
   a stated policy, D44);
6. intent confidence < 0.1913;
7. top-1 similarity < 0.1981 together with a substantive reply.

A reply that only asks for details gets the informational code `awaiting_customer_details`
and is **not** escalated for weak evidence (D45).

## Intents (taxonomy v2, frozen)

| Intents | Escape labels |
| --- | --- |
| `flight_delay`, `flight_cancellation_rebooking`, `baggage`, `seating_and_upgrade`, `boarding_and_gate`, `booking_fees_and_fare_rules`, `staff_and_service_complaint`, `loyalty_and_lounge`, `praise_and_compliment`, `general_dissatisfaction`, `non_support_commentary` | `OTHER`, `UNCLEAR` |

- **Definitions, boundaries and tie-breakers:** `golden/taxonomy_v2.md`.
- **Weak-label rules** exist for the first ten intents only. `non_support_commentary`,
  `OTHER` and `UNCLEAR` are therefore never predicted (D41).

## Evaluation methodology

- **Golden set** (`golden/golden_set_v1.csv`, frozen):
  - 148 pilot rows (b01: 120 random + 28 targeted), relabelled under taxonomy v2;
  - 100 fresh rows (b02) labelled under the frozen v2;
  - one annotator.
- **Isolation:**
  - golden conversations, **all conversations by golden customers**, exact duplicates and
    near-duplicates are excluded from retrieval and training;
  - `src/evaluate.py` re-checks this and stops on any violation;
  - gold labels never train, tune or select anything. The one exception, the choice of k in
    Phase 6B, is disclosed in `REPORT.md` §7.
- **Run of record:** the recorded Phase 6C generations. The evaluation reproduces the
  classifier predictions, retrieved evidence, G1–G7 flags and both baselines **exactly**
  before scoring.
- **Statistics:** 95% percentile bootstrap intervals (2,000 resamples, seed 42) for accuracy
  and macro-F1. Every table reports all 248 rows and b02.
- **Baselines:**
  - majority-class intent;
  - random evidence from the same pool with the same exclusions (seed `42 + query index`);
  - **Baseline B**, a fixed generic reply;
  - **Baseline A**, the sanitised top-1 historical reply (its G7 failures follow from its
    definition).
- **Reply quality** (Phase 6F):
  - 40 golden items × 3 systems, blinded and shuffled, rated on a 1–5 rubric
    (`human_eval/`);
  - **the rater is the system's developer, and the ratings were AI-assisted;**
  - blinding was weak, and the retest was built but **not completed** before submission
    (D47, D48).
- **LLM-as-judge:** `src/judge.py` is **experimental and unused**. It failed its
  pre-declared validation (D46).

## Reproduction

### Environment

```bash
conda env create -f environment.yml
conda activate hiver
```

Python 3.11, pandas, numpy, pyarrow and scikit-learn. No other packages are needed. Ollama
is only needed for Level 2.

### Level 0: tests (no data, about 1 minute)

```bash
python -m unittest discover -s tests
```

391 tests. They use synthetic fixtures plus the tracked `golden/` files, and pass on a clean
clone with no `data/` directory.

### Level 1: headline deterministic evaluation (no LLM, about 5 minutes)

1. Download [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
   and place the CSV at `data/raw/twcs/twcs.csv`. Check its SHA-256:
   `CD297FCFA1BF6F99938BE242E8E578980BC6D1B96ADC8691ABEC9A39175B03C0`.
2. Run:

   ```bash
   python -m src.reproduce_headline
   ```

The runner does the following, in order:

1. profiles and caches the CSV;
2. reconstructs conversations;
3. builds the corpus and weak labels;
4. **checks `artifacts/phase6/SHA256SUMS` and copies the recorded generations** into
   `data/processed/phase6/`;
5. calibrates the escalation thresholds;
6. rebuilds the Phase 5 retest key and agreement report;
7. runs `python -m src.evaluate --confirm-golden-run`;
8. runs the round-1 reply-rating analysis;
9. prints per-step timings.

It exits with status 0 only if `evaluation_rows.jsonl` is byte-identical to the recorded
run.

**Measured on a clean copy of the repository** (no `data/`, no caches, Windows laptop): 295 s in total. That breaks down as profiling 123 s, reconstruction 51 s, corpus 10 s, calibration 39 s, evaluation 64 s, and the remaining steps under 10 s together. The evaluation statistics matched the original run except for the recorded git commit.

Reports are written to `reports/`, which Git ignores; the main ones are
`phase6_evaluation.md` and `phase6f_reply_ratings_round1.md`.

### Level 2: regenerate replies (optional, about 1 hour, needs Ollama)

```bash
ollama pull llama3.1:8b
python -m src.run_golden_generation      # writes data/processed/phase6/golden_generation_records.jsonl
python -m src.retry_golden_generation    # retries rows that failed for infrastructure reasons, once
```

**These are the historical Phase 6C run scripts, kept unchanged.**

- They run as soon as they are imported.
- They contain a hard-coded `sys.path` entry, which is harmless when run with `python -m`
  from the repository root.
- The first pass and the retry were merged into `golden_generation_final.jsonl` by hand;
  `src/evaluate.py` verifies that merge.
- **Regenerated text may differ byte for byte from the record** (b01_0001, `REPORT.md` §8).
  The evaluation uses the tracked record.

### Other entry points

```bash
python -m src.profile_raw --source sample          # 93-row correctness check, seconds
python -m src.analyze_brands --source full         # brand screening
python -m src.discover_intents --source full      # Phase 4 clustering report
python -m src.classify_intent                      # intent baselines
python -m src.retrieve                             # retrieval report and k selection
python -m src.build_reply_rating_batch --verify    # needs the local generation cache
```

## Repository layout

```text
REPORT.md                 final report
src/                      pipeline, evaluation and rating tools (see docs/FILE_GUIDE.md)
tests/                    391 unit tests (synthetic fixtures and tracked golden files)
golden/                   frozen taxonomy, guidelines, golden labels, exclusion ids, Phase 5 retest labels
human_eval/               Phase 6F rubric, blinded batches, round-1 ratings, manifests (no system mapping)
artifacts/phase6/         recorded Phase 6C generations + SHA256SUMS (tracked; D49)
docs/                     decision log, file guide, development log
data/                     raw data, caches, corpus and LLM caches (ignored)
reports/                  generated reports (ignored)
```

**What each part is for:**

| | Needed to run the agent | Needed to reproduce the evaluation |
| --- | --- | --- |
| `data/raw/twcs/twcs.csv` (download) | yes, to build the corpus | yes |
| `src/`, `environment.yml` | yes | yes |
| Ollama + `llama3.1:8b` | yes, for generation | no (Level 2 only) |
| `golden/` | only for its exclusion ids | yes |
| `artifacts/phase6/` | no | yes (the run of record) |
| `human_eval/` | no | reply-quality analysis only |

**Git:**

- **Data files are stored byte-exact** (`.gitattributes`), so the recorded SHA-256 values
  hold on any platform.
- **`data/` and `reports/` are ignored.** The raw CSV (493 MB) is never committed.

## Limitations

- **One annotator.** Every gold label comes from one person. Label reliability evidence is
  intra-annotator only (κ 0.8281, a 12–14-hour gap), and 148 golden rows were relabelled
  under definitions written after reading them.
- **Weak supervision.** The classifier learns keyword rules, covers 10 of 13 labels, and
  cannot predict 41 golden rows.
- **Lexical retrieval.**
  - Similarity is low (median top-1 0.26).
  - Weak labels cover 38% of evidence.
  - k was chosen with golden results.
- **Safety checks are regular expressions.** Soft promises and invented procedures pass
  (two such auto-handled replies were found in the reply ratings). **A 95% pass rate is not
  correctness.**
- **Escalation.** The thresholds are traffic sizes, and no outcome labels exist.
- **Reply quality.**
  - One round of developer-made, AI-assisted ratings, with weak blinding.
  - The retest was not completed.
  - No agreement statistic exists.
  - The LLM judge failed validation.
- **Reproducibility of generation.** 49 of the 248 generations needed an infrastructure
  retry, and one of five probed rows did not regenerate byte for byte.
- **Scope.** One brand, English, Oct–Dec 2017, a local 8B model, and laptop latency (median
  12.8 s per reply).
