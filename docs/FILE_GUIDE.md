# File Guide

What every meaningful file in this project does, and which ones contain logic that
matters for the interview.

> **Where explanations live.** Source files carry only short one-line comments
> explaining *why* something non-obvious is done. The detailed reasoning —
> methodology, alternatives, tradeoffs — lives here and in `DECISION_LOG.md`.
> Read this file alongside the code, not the code alone.

**Review priority legend**

- 🔴 **Core logic — understand this properly.** Methodology lives here.
- 🟡 **Worth reading once.** Supporting logic, few surprises.
- ⚪ **Straightforward.** Configuration, paths, generated output.

---

## Source code

### `src/config.py` ⚪

**Purpose:** Every path and tunable constant in one place.

**Responsibility:** Derive all paths from the project root so nothing depends on the
current working directory and no personal absolute path is ever written into code.

**Key detail:** `PROJECT_ROOT = Path(__file__).resolve().parents[1]`. This is what
makes the project reproducible on someone else's machine — a requirement for
Phase 19.

**Constants that affect results:**

| Constant | Effect |
| --- | --- |
| `CHUNK_SIZE` | Rows per CSV chunk. Memory vs speed |
| `RANDOM_SEED` | Fixes the chronology sample, so re-runs are identical |
| `TWITTER_TIME_FORMAT` | Twitter's non-ISO format. Parsing is explicit so failures are counted, not silently coerced |
| `TOP_BRAND_COUNT` | How many brands reach the comparison table and text pass |
| `BRAND_TEXT_SAMPLE` | Tweets sampled per brand for the type-token ratio |
| `MIN_FREE_DISK_GB` | Below this, the Parquet write fails loudly instead of filling the disk |

**Inputs:** none. **Outputs:** none. **Dependencies:** `pathlib` only.

---

### `src/profile_raw.py` 🔴

**Purpose:** Phase 1. Profile the raw dataset and write a report plus a JSON dump
of every statistic.

**Inputs:** `data/raw/twcs/twcs.csv` or `data/raw/sample.csv` (opened read-only).

**Outputs:**
- `data/interim/twcs.parquet` — faithful cache
- `reports/phase1_profile.md` — human-readable report
- `reports/phase1_stats.json` — machine-readable statistics

**Dependencies:** pandas, numpy, pyarrow, `src.config`.

**How it runs:** four passes, so that the largest column (`text`) is never held in
memory alongside everything else.

| Pass | Function | What it does |
| --- | --- | --- |
| 1 | `scan_and_cache` | Streams the CSV in chunks, accumulates global statistics, writes the Parquet cache |
| 2 | `analyze_relationships` | Graph structure: parents, children, dangling refs, branching, cycles, chronology |
| 3 | `analyze_authors` + `compare_brands` | Author roles and the brand comparison table |
| 4 | `analyze_brand_text` | Per-brand text characteristics for shortlisted brands only |

#### Functions that matter

**`detect_cycles` 🔴 — the one to study first.**
Every tweet has at most one parent, which makes the parent graph a *functional
graph*. That property allows cycle detection in a single linear scan using
three-state marking (unvisited / in-progress / done) instead of a general
graph algorithm. It is written iteratively on purpose: recursion would overflow
the Python stack on millions of nodes. Be ready to explain why the graph is
functional and why that makes the problem easy.

**`analyze_chronology` 🔴 — the methodological heart of Phase 1.**
Answers "does `tweet_id` order mean anything?" with three *independent* measures:
Spearman correlation of `tweet_id` against time, the share of parent→child edges
where the parent is earlier in time, and the share where the parent has a lower ID.
If these disagree, the disagreement is reported rather than resolved by preference.
Note the Spearman is computed as Pearson-on-ranks specifically to avoid adding a
scipy dependency.

**`analyze_authors` 🔴 — tests an assumption instead of trusting it.**
Cross-tabulates "`author_id` looks numeric" against `inbound`. The point is to
*verify* whether numeric IDs really mean customers rather than assume it. Also
counts authors appearing with both inbound directions, which would break any
simple role rule.

**High fan-out detection (inside `analyze_relationships`) 🔴.**
Counts tweets with 10+ and 50+ direct replies and lists the largest. These turn out
to be promotions and outage notices, not support threads — 472 tweets carrying
77,411 replies between them. Phase 2 must exclude or special-case them, or
reconstruction will invent enormous fake conversations. Know this one; it is a
concrete example of profiling preventing a downstream bug.

**`compare_brands` 🟡 — deliberately approximate.**
Builds the per-brand table from 1-hop reply links only, because real conversations
do not exist until Phase 3. The conversation-shaped columns are for **shortlisting
only**; the report says so in-line. Know this limitation — it is an obvious
interview question.

**`scan_and_cache` 🟡 — pass 1.** Streams the CSV in chunks, accumulates global
statistics, and writes the Parquet cache in the same read.

Two details worth knowing. First, `RAW_DTYPES` reads **every** column as text.
Declaring `tweet_id` as an integer would make pandas raise on the first malformed
row and kill the whole run; reading as text lets malformed values be *counted* and
sampled instead. Nothing is silently dropped. Second, `count_physical_lines` counts
newline bytes and compares that against the parsed row count — the gap (190,749 on
the full file) is tweets containing embedded newlines. Any line-based reader would
corrupt those rows; the CSV parser handles them correctly, and this check proves it.

**`verify_cache` 🟡.** Confirms the Parquet cache faithfully reproduces the CSV:
row count equality plus a first-1000 `tweet_id` comparison. `main` raises if the row
counts disagree, so a truncated cache cannot silently reach later phases.

**`analyze_brand_text` 🟡 — pass 4.** Streams the `text` column separately and keeps
only the shortlisted brands in memory. `text` is by far the largest column, so
loading it alongside the relationship analysis would risk exhausting RAM. Produces
the DM-deflection rate, which turned out to be the most important brand-selection
signal in the whole phase.

**`markdown_table` ⚪.** Nine lines. Exists only because pandas' `to_markdown()`
pulls in `tabulate`, and printing a table does not justify a dependency.

**`markdown_table` ⚪.** Small helper. Exists only because pandas' `to_markdown()`
requires the `tabulate` package, and this phase does not otherwise need it.

---

### `src/reconstruct.py` 🔴

**Purpose:** Phase 2. Turn tweet-level reply links into conversations.

**Inputs:** `data/interim/twcs.parquet` (Phase 1 cache).

**Outputs:** `data/processed/conversation_turns.parquet`,
`data/processed/conversations.parquet`, `reports/phase2_reconstruction.{md,json}`.

**Dependencies:** pandas, numpy, pyarrow, `src.config`. No graph library.

**Flow:** `reconstruct()` runs the sequence — parent array → roots → depth →
participant classification → conversation IDs → turn ordering → conversation stats.
`main()` then writes, validates and reports. Deliberately linear; no abstraction
layer.

#### Functions that matter

**`find_roots` 🔴 — the one non-obvious algorithm.**
Pointer doubling. Each tweet's slot holds its parent's slot; roots point at
themselves. `pointer = pointer[pointer]` means "take my parent's pointer as my own",
so reach **doubles** every round — after ~log₂(depth) rounds everyone points at their
root. Max depth is 649, so about 10 rounds. Roots being self-pointing is what halts
it. A cycle would never stabilise, so the loop caps iterations and raises rather than
hanging; Phase 1 measured 0 cycles, which is what makes this safe. Be ready to
explain why the graph is a forest and why that makes this valid.

**`classify_participants` 🔴 — the broadcast decision, and the likeliest interview
question.** Counts distinct customers and distinct brands per component, then assigns
status via `np.select` with explicit precedence. `clean` means strictly one customer
and one brand. Uses `np.select` rather than sequential overwrites because a chain of
overwrites left a gap that mislabelled a non-dyad as clean — see D21. Also computes
`customer_thread_count`, recorded raw with no threshold (D18).

**`assign_turn_index` 🔴 — why `created_at`, never `tweet_id`.**
`np.lexsort((tweet_id, timestamp, root))` sorts by conversation, then time, with
`tweet_id` breaking exact ties only. The scatter-back (`turn_index[order] = position`)
assigns positions without reordering the underlying rows, which keeps the batched
Parquet write aligned by position.

**`validate` 🔴 — where correctness is actually proven.**
Twelve invariants; `main` raises if any fails. The ones that carry real weight:
every tweet in exactly one conversation, every parent in the *same* conversation,
parent never later than its child, and clean conversations being strict dyads.

**`build_parent_index` 🟡.** Maps parent tweet IDs to array positions. Returns three
arrays: resolved parent index, whether a parent was *declared*, and whether it
*resolved*. The gap between the last two is the 3,862 truncated roots. Asserts no
duplicate tweet IDs, since that would make the lookup ambiguous.

**`cross_check_response_field` 🟡.** Compares the built edges against
`response_tweet_id` *after* reconstruction. Never used to build anything — this is
independent corroboration, and it returned 100.0%.

**`compute_depth` ⚪.** Walks parent pointers on a shrinking active set.

---

### `src/analyze_brands.py` 🔴

**Purpose:** Phase 3. Select one brand from measured evidence. Intent discovery is
**not** part of this file.

**Inputs:** `data/processed/conversations.parquet`,
`data/processed/conversation_turns.parquet`.

**Outputs:** `reports/phase3_brand_analysis.md`, `reports/phase3_brand_stats.json`.

**Dependencies:** pandas, numpy, pyarrow, `src.config`.

#### Functions that matter

**`collect_turn_signals` 🔴 — where the decisive measurements come from.**
Streams the turns table once and produces two things: per-conversation flags (does
**any** brand turn ask for a DM, does any contain a URL, mean reply length) and
per-brand opening-message signals (language, vocabulary). The DM flag is
**conversation-level** — one redirect anywhere makes the whole resolution invisible,
which is why this differs so much from Phase 1's per-tweet figure. Note the fixed
`TTR_SAMPLE_SIZE` budget: type-token ratio falls as sample size grows, so brands with
very different volumes would not otherwise be comparable.

**`apply_gates` 🔴 — screening before scoring.**
Four boolean gates. Deliberately separate from scoring, because some weaknesses are
disqualifying no matter how strong a brand looks elsewhere. AmazonHelp is the proof:
it would top any weighted score while being unusable at 18.1% non-English.

**`score_candidates` 🔴 — and why it must not be trusted alone.**
Min-max normalises each criterion **within the surviving set**, then applies weights.
Normalising within survivors means the score rewards whichever brand is most
*extreme* on the heaviest criterion — which is exactly why it ranks GWRHelp first and
the selected brand sixth. Read D23 before quoting this score; the disagreement is
recorded rather than tuned away.

**`brand_metrics` 🟡.** Assembles the per-brand table. Note it filters conversations
to the analysed brand set first — without that, brands outside the top 25 join to
null signals and produce meaningless rows.

**`summarise_selected_brand` 🟡.** Produces the corpus handed to the next phase,
including the monthly distribution that reveals the temporal concentration.

**Constants worth knowing:** `SELECTED_BRAND`, the four gate thresholds,
`SCORING_WEIGHTS`, `DM_PATTERN`, `TTR_SAMPLE_SIZE`.

---

### `src/discover_intents.py` 🔴

**Purpose:** Phase 4. Discover issue structure in AmericanAir opening messages.
Produces **discovery artifacts, not intents**.

**Inputs:** `data/processed/conversations.parquet`,
`data/processed/conversation_turns.parquet`.

**Outputs:** `reports/phase4_intent_discovery.md`, `reports/phase4_intent_stats.json`.

**Dependencies:** pandas, numpy, pyarrow, **scikit-learn**, `src.config`.

**Note the first three lines of the file:** `OMP_NUM_THREADS` is set *before* the
sklearn import. It has no effect afterwards. It suppresses the MKL KMeans leak on
Windows and stabilises float summation order across runs.

#### Functions that matter

**`normalise_message` 🔴 — every downstream number depends on it.**
HTML unescape, URL removal, mention stripping, hashtag-word retention, whitespace
collapse, lowercase. No stemming. Small function, large consequences: the
normalisation-sensitivity check shows ARI of only **0.37** between this and a
variant, so these choices materially change the cluster structure.

**`describe_cluster` 🔴 — where a cluster becomes interpretable.**
Returns size, cohesion, distinctive terms, 8 centroid-nearest messages **and 3
seeded-random members**. The random sample exists so representative examples cannot
be cherry-picked — and it earned its place: cluster 11's nearest-centroid messages
turned out to contradict its own distinctive terms.

**`distinctive_terms` 🔴.** Ranks terms by *lift* — mean weight inside the cluster
minus outside — rather than raw weight. Raw weight would surface corpus-wide common
words in every cluster and make them all look alike.

**`sweep_k` 🟡.** Runs K ∈ {6,8,10,12,15,20} and reports inertia, silhouette, size
distribution and max centroid similarity. **Silhouette on sparse text is weak** —
it never exceeded 0.054 here — so it is reported for comparison across K, never as
evidence that clusters are meaningful.

**`quality_checks` 🟡.** Screens for generic, URL-driven and format-driven clusters,
over-broad and tiny clusters, and agent signatures.

**`normalisation_sensitivity` 🟡.** Re-clusters with different preprocessing and
reports ARI against the main partition. This produced one of the phase's most
important numbers.

**`pick_k` ⚪.** A default starting point only — prefers a K without near-duplicate
centroids or a dominant cluster. `--k` overrides it. K is ultimately a human call.

---

### `src/build_golden_set.py` 🔴

**Purpose:** Merge both labelled batches into the frozen 248-row golden set, attach
provenance, build both exclusion lists, and run leakage validation.

**Key functions:** `build_golden_frame` (deterministic order by `batch`,
`annotation_id`; renames `text_display`->`text` and `is_ambiguous`->`ambiguous`) ·
`jaccard_overlaps` 🔴 (exact near-duplicate search over 248 x 24,239, pruned by a
length band derived from the Jaccard threshold - **no sampling**, so the single
cross-customer duplicate could not be missed) · `leakage_report` ·
`write_exclusion_list` / `write_customer_exclusion_list`.

**Read this for the interview:** the length-band pruning in `jaccard_overlaps`. A
Jaccard of at least *t* forces the two token sets' sizes to sit within
`[t*|A|, |A|/t]`, which is what makes an exact all-pairs check affordable.

---

### `src/build_retest_batch.py` 🔴

**Purpose:** Draw 45 rows from the frozen golden set for intra-annotator test-retest.

**Why it matters:** this file's whole job is *withholding* information. `select_rows`
shuffles **before** assigning `rt_001...rt_045`, so ordinal position cannot leak
batch or original order. `write_blank` emits only `retest_id` and `text`. `verify`
asserts text identity 45/45 against the mapped golden rows - the check that proves
the retest is being run against the right examples. The builder refuses to overwrite
a frozen batch.

---

### `src/build_annotation_batch.py` 🔴

**Purpose:** Phase 5a. Draw a stratified sample of AmericanAir opening messages and
write a **blank** CSV for manual labelling.

**Inputs:** `data/processed/conversations.parquet`,
`data/processed/conversation_turns.parquet`.

**Outputs:** `golden/b01_pilot_blank.csv`,
`reports/phase5_b01_sampling_manifest.json`.

**Dependencies:** pandas, numpy, pyarrow, `src.config`. **No sklearn, no LLM.**

#### Functions that matter

**`sample_random_stratified` 🔴 — the honest half of the sample.**
Uniform random, allocated **month-proportionally**. AmericanAir's traffic is 99.8%
concentrated in Oct-Dec 2017, so without this one busy week could dominate. This
stratum alone supports frequency estimates.

**`sample_targeted` 🔴 — the deliberately biased half.**
Keyword probes guarantee rare intents appear; pure random might return zero
`loyalty_and_lounge`. Probes come from Phase 4 **term** evidence, never from cluster
assignments (D28). Drawn *after* the random pool and excluding it, so the strata never
overlap. Rows are flagged `targeted` and excluded from frequency claims.

**`main`'s guard clauses 🔴 — the "no automatic labels" guarantee.**
Refuses any path ending `_labelled.csv`; refuses to overwrite an existing blank batch.
Both tested, not assumed. This is where the promise becomes a property of the code.

**`verify_blank_batch` 🟡.** Re-reads the written file and asserts every label column
is empty, IDs are unique, no text is lost, and the text survives the CSV round trip.
Catches encoding and quoting damage before a human wastes time on a corrupted file.

**`build_rows` 🟡.** Note `text_display` collapses newlines to ` / ` so one message
occupies one spreadsheet row. This is a **rendering, not an edit** — the original is
untouched in Parquet and joinable on `conversation_id`.

---

### `src/analyse_annotations.py` 🔴

**Purpose:** Phase 5d. Analyse a completed annotation batch and evaluate the
pre-declared taxonomy triggers.

**Inputs:** `golden/b01_pilot_labelled.csv` (**read-only**),
`data/processed/conversation_turns.parquet`, the sampling manifest.

**Outputs:** `reports/phase5_pilot_analysis.md`, `reports/phase5_pilot_stats.json`,
`reports/phase5_ambiguous_cases.csv`.

**Dependencies:** pandas, pyarrow, `src.config`. **No sklearn, no LLM, no network.**

#### Functions that matter

**`validate` 🔴 — the check that actually earned its place.**
Compares every row's `text_display` against the Phase 2 source instead of trusting
the returned file. Caught `b01_0066`, whose label had been applied to text belonging
to no message in the batch — a row that looked entirely well-formed. Without this,
a mislabelled example would have entered the golden set silently. See D31.

**`normalise` 🔴 — corrections that never touch the file.**
Applies a deterministic map for typos and conventions, and **logs every affected
`annotation_id`**. The annotator's CSV is opened read-only and never written. Keeps
an author's judgement distinguishable from a tool's correction (D32).

**`evaluate_triggers` 🔴.** Evaluates the five thresholds declared in D29 **before
any label existed**, against the **random stratum only** — the targeted stratum
over-samples rare intents and cannot support prevalence claims (D28). Note the
report presents trigger results as evidence; the interpretation is authored
separately, because a fired trigger says *that* something is wrong, not *what* (D33).

**`intent_table` 🟡.** Per-intent counts with confidence breakdown, ambiguity and
discussion rates. The `pct_not_high` column proved more useful than `low` confidence
alone, which the annotator barely used.

**`build_review_cases` 🟡.** Extracts every row flagged ambiguous, flagged for
discussion, or labelled below high confidence, with **90-character excerpts only** —
reports deliberately avoid dumping full tweet text.

---

### `src/escalate.py` 🔴

**Purpose:** Phase 6D. Decide `auto_handle` or `escalate` for one message, with
reason codes. Runs after classification, retrieval, generation and G1–G7.

**Inputs:** an `EscalationSignals` value and a `Thresholds` value. **Outputs:** a dict
with `decision`, `reason_codes`, `informational_codes`, `reason`, the signals and
thresholds used, and `policy_version`.

**Dependencies:** `src.config`, `src.taxonomy`. **No model, no file I/O, no network,
no golden data.** A test enforces this.

#### Functions that matter

**`decide` 🔴.** Evaluates all seven rules without short-circuiting, so the record
lists every reason. Codes are emitted in the fixed `RULE_ORDER`, and the reason text is
built from fixed templates, so identical input gives identical output.

**`EscalationSignals` 🔴 — validation is part of the method.** It rejects intents the
classifier cannot predict (`OTHER`, `UNCLEAR`, `non_support_commentary`) and unknown
grounding codes. It also forbids `needs_more_information` and `evidence_used` on an
unparsed reply, so rules that need parsed fields cannot run on defaults. It has no
gold-label field.

**`build_signals` 🟡.** Converts a retrieval list and a `generate_llm` result into
signals. It drops the `needs_more_information=False` that `generate_llm` fills in on a
parse failure; otherwise that value would read as "the reply gave a substantive answer".

**`HUMAN_REVIEW_POLICY_INTENTS` 🔴.** A deliberate system policy, not learned from the
data. It is kept local to this module, and `taxonomy.ALWAYS_ESCALATE` is not used.

---

### `src/calibrate_escalation.py` 🔴

**Purpose:** Phase 6D. Produce `T_conf` and `T_sim` by the pre-declared rule: the
20th percentile of each on a fixed development pool.

**Inputs:** retrieval corpus, weak training set, and golden **identifier columns only**
(read to prove they are absent). **Output:** `reports/phase6_escalation_calibration.json`.

**Dependencies:** numpy, pandas, sklearn, `src.classify_intent`, `src.retrieve`,
`src.escalate`, `src.leakage`. **No LLM, no network.**

#### Functions that matter

**`sample_pool` 🟡.** Draws 2,000 conversations uniformly from the golden-free corpus
with seed 42. A SHA-256 of the sorted IDs is recorded so the pool can be reproduced.

**`intent_confidence` 🔴.** Pool rows that are in the weak training set get
out-of-fold probabilities (5 stratified folds). Otherwise the classifier would be scoring its own training data and look
more confident than it is. All other rows use the fitted model. The two populations
have very different confidence distributions, and both are reported.

**`verify_no_golden` 🔴.** Checks the corpus, training set and pool against the golden
set's conversation and customer IDs and both exclusion files, and raises an error on any match.

**`confirm_strict_comparison` 🟡.** Calls `decide()` exactly at each threshold and just
below it. This confirms equality does not fire, instead of assuming it.

**The thresholds are traffic sizes, not tuned values.** No escalation outcome labels
exist.

---

### `src/judge.py` 🟡 — experimental, failed validation, not used

**Status:** **Experimental. It failed its pre-declared non-golden validation (D46) and is
NOT used by the production or evaluation path.** No module imports it, and it has no
golden-judging entry point. No judge scores exist for the golden set.

**Purpose:** Phase 6E LLM-as-judge for reply quality, kept so the negative result can be
inspected and reproduced.

**Inputs:** golden-free corpus messages with synthetic replies, through `llm.complete`
(local Ollama). **Outputs:** `reports/phase6_judge_sanity*.json` and
`reports/phase6_judge_probe*.json`, written into isolated caches under
`data/llm_cache_judge_sanity/` and `data/llm_cache_judge_probe/`, never into the production cache.

**Dependencies:** numpy, `src.llm`, `src.generate_reply` (formatting only), `src.retrieve`
(sanity messages only). Run only with `--sanity-check` or `--determinism-probe`.

#### What it preserves

- **Prompt history.** j3 is the live design: five per-dimension templates, each with its
  own SHA-256. j1 and j2 are kept as retired constants with their hashes, together with
  their whole-item parser, so their results can be reproduced.
- **Validation logic.** The j2 cases and rule, j3 sets 1 and 2 with their C1–C5
  specifications, and the message-selection filters.
- **Evaluation machinery that was never used on golden data.** Blinded, shuffled pairing
  (`build_judge_items`); per-dimension summaries and paired comparisons; the strict parser;
  one infrastructure retry per call; no retry for malformed output.

#### Functions worth reading

**`isolated_cache` 🟡.** Temporarily points `config.CACHE_DIR` elsewhere.
`llm.complete` writes a cache entry even when the cache is bypassed, so without this a
probe would overwrite production entries.

**`evaluate_j3_criteria` 🟡.** The pre-declared C1–C5 rule, identical for every case set.
A dimension that failed to parse counts as failing the criterion that uses it.

---

### `src/build_reply_rating_batch.py` 🔴

**Purpose:** Phase 6F. Build the blinded human reply-quality batch (D47).

**Inputs:** `data/processed/phase6/evaluation_rows.jsonl` and `retrieval_corpus.parquet`,
both frozen. The production LLM cache is **read only**, to check the evidence.
**Outputs:** `human_eval/reply_rating_r01_blank.csv` and `human_eval/reply_rating_manifest.json`.

**Dependencies:** numpy, pandas, `src.generate_reply` (formatting only) and
`src.llm._cache_key` (a hash function; nothing calls a model). **No retrieval rerun, no
generation, no network, no gold labels.** `--verify` rebuilds the batch and compares it
byte for byte with disk.

#### Functions that matter

**`restrict_row` 🔴.** Each frozen row is reduced to the fields that sampling, display and
the evidence check need. The `gold` field is never read, and a test enforces this.

**`build_batch` 🔴.** Builds the population and the largest-remainder allocation, draws
the sample, assigns shuffled item ids, and shuffles responses so no two neighbours share an
item. It returns the rows for the CSV, the hidden key and the manifest core. The key is
never written; only its SHA-256 is.

**`verify_evidence_against_generation` 🟡.** Rebuilds each generation prompt from the
rebuilt evidence and checks that a matching production cache entry exists. This proves the
rater sees the evidence the generator saw.

---

### `src/analyse_reply_ratings.py` 🔴

**Purpose:** Phase 6F. Analyses the human reply ratings exactly as declared in D47.

**Inputs:** rating files given as explicit paths, the frozen blank batches, the hidden keys
(rebuilt by `src/build_reply_rating_batch.py`), and the frozen G-flags,
`needs_more_information` and escalation decisions from `evaluation_rows.jsonl` (the `gold`
field is never read). **Outputs:** `reports/phase6f_reply_ratings.{json,md}`.

**Dependencies:** numpy, sklearn (`cohen_kappa_score`), `src.build_reply_rating_batch`.
**No model, no network.**

**Safety.**
- `read_ratings` refuses to read the real rating files unless `allow_real=True` is passed.
- The command line needs explicit `--round1` and `--retest` paths plus
  `--confirm-real-ratings`, so round-1 results cannot be computed before the retest exists.
- The module never searches for rating files.
- It was written and tested on synthetic ratings only; the real round-1 file was not read
  while building it.

#### Functions that matter

**`require_valid_round1` / `validate_retest` 🔴.**
- Round 1 goes through the unchanged strict validator in the batch builder, plus a check
  that the system mapping has exactly one response per system per item.
- The retest must be fully rated, with its ids, text and columns unchanged.

**`item_resamples` and `bootstrap_mean_interval` 🔴.**
- Percentile bootstrap (2,000 resamples, seed 42) over **whole items**, so an item's three
  responses always stay together.
- One draw matrix serves every system and every paired difference.

**`paired_comparisons` 🟡.** LLM − Baseline B and LLM − Baseline A for each dimension:
first higher, tie, second higher, mean difference and an item-level interval. These are
descriptive only.

**`claim_flag_crosstab` 🟡.** G2–G7 flags against human claim safety ≤2, per system. G1
(empty reply) is not a claim flag.

**`retest_agreement` 🔴.**
- Per dimension: exact agreement, agreement within one point, mean absolute difference,
  quadratic-weighted κ on the fixed 1–5 scale, and a bootstrap over retest units (seed 44).
- κ is reported as undefined when both rounds use a single shared value.
- A pooled figure is also given, labelled non-independent, alongside the score
  distributions.

---

## Hand-authored files, not generated

| File | Nature |
| --- | --- |
| `reports/phase4_candidate_taxonomy.md` | **Hand-written.** Not produced by any script. The human-interpretation layer, kept separate so an authored judgement is never mistaken for a computed result. Marked UNVALIDATED. See D26. |
| `golden/taxonomy_v1.md` | **Hand-written.** Candidate taxonomy used for the pilot. Superseded by v2; kept as the historical record of what the pilot was actually labelled against |
| `golden/taxonomy_v2.md` | **Hand-written.** Minimal revision: 12 v1 labels retained, `non_support_commentary` added, `loyalty_and_lounge` broadened, 7 definitions sharpened. **Unvalidated** — see D38 |
| `reports/phase5_v2_relabel_queue.csv` | **Hand-authored proposals** joined to real excerpts. 22 rows needing human reconsideration under v2. `proposed_primary_intent` is a suggestion, never a decision |
| `reports/phase5_v2_impact.md` | **Hand-written.** What v2 changes and why each contested call went the way it did |
| `golden/annotation_guidelines.md` | **Hand-written.** Labelling rules and pre-declared revision triggers |
| `golden/README.md` | **Hand-written.** Why `golden/` is tracked in Git and how labels may be used |
| `golden/b01_pilot_labelled.csv` | **Written by the annotator.** No code writes to this path |
| `golden/b02_fresh_labelled.csv` | **Written by the annotator.** 100 independent rows under frozen v2 |
| `golden/golden_set_v1.csv` | Derived merge of both batches, 248 rows. **Frozen** |
| `golden/golden_exclusion_ids.txt` | 248 conversation IDs - layer 1 of downstream exclusion |
| `golden/golden_customer_exclusion_ids.txt` | 248 customer IDs - layer 2. Neither layer catches cross-customer duplicates; see D39 |
| `golden/retest_r01_blank.csv` | 45 opaque rows for intra-annotator retest. **Frozen**; builder refuses to overwrite |
| `human_eval/README.md` | **Hand-written.** Phase 6F rating procedure, blinding, limitations and rules for the rater (D47) |
| `human_eval/reply_rating_rubric.md` | **Hand-written.** The 1–5 reply-quality rubric, including the claim-safety rules |
| `human_eval/reply_rating_r01_blank.csv` | 120 blinded responses (40 items × 3 systems). Written by `src/build_reply_rating_batch.py`, which refuses to overwrite it with different content |
| `human_eval/reply_rating_manifest.json` | Sampling, blinding and input provenance with hashes. **Contains no system mapping.** Not to be opened while rating |
| `human_eval/reply_rating_r01_rated.csv` | **Written by the annotator.** No code writes to this path. Round-1 ratings; validated, not yet analysed |
| `human_eval/reply_rating_retest_blank.csv` | 36 blinded retest rows (`T01`–`T36`), written by `src/build_reply_rating_batch.py --retest` |
| `human_eval/reply_rating_retest_manifest.json` | Round-1 structural validation facts, retest timing, seed and hashes. **No ratings and no system mapping** |

`golden/` is the one directory deliberately **not** gitignored — hand labels cannot
be regenerated from code. See D27.

`human_eval/` (Phase 6F) is also tracked, for the same reason: human ratings cannot be
regenerated from code. It is kept apart from `golden/`. See D47.

---

## Documentation

| File | Purpose | Priority |
| --- | --- | --- |
| `docs/DEVELOPMENT_LOG.md` | Living record: every step, error, root cause, fix | 🟡 |
| `docs/FILE_GUIDE.md` | This file | ⚪ |
| `docs/DECISION_LOG.md` | Non-obvious engineering decisions and their tradeoffs | 🔴 |
| `README.md` | Project overview and reproduction instructions | 🟡 |

---

## Data and output

| Path | Purpose | Notes |
| --- | --- | --- |
| `data/raw/twcs/twcs.csv` | Raw dataset, ~493 MB | **Immutable.** Read-only at OS level, Git-ignored |
| `data/raw/sample.csv` | 93-row sample | Used as the correctness harness |
| `data/interim/twcs.parquet` | Cache of the raw CSV | Faithful copy, no cleaning. Git-ignored, regenerable |
| `reports/phase1_profile.md` | Phase 1 report | Generated, Git-ignored |
| `reports/phase1_stats.json` | Phase 1 statistics | Generated, Git-ignored |
| `reports/phase1_*_sample.md/json` | Same, for the sample harness | Generated, Git-ignored |
| `data/processed/conversation_turns.parquet` | One row per tweet with conversation context | Generated, Git-ignored |
| `data/processed/conversations.parquet` | One row per conversation | Generated, Git-ignored |
| `reports/phase2_reconstruction.{md,json}` | Phase 2 report | Generated, Git-ignored |

---

## Configuration

| File | Purpose |
| --- | --- |
| `.gitignore` | Keeps `data/` and secrets out of Git. Written before the data was placed, deliberately |
| `environment.yml` | The single conda environment, `hiver` |
