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

## Hand-authored files, not generated

| File | Nature |
| --- | --- |
| `reports/phase4_candidate_taxonomy.md` | **Hand-written.** Not produced by any script. The human-interpretation layer, kept separate so an authored judgement is never mistaken for a computed result. Marked UNVALIDATED. See D26. |
| `golden/taxonomy_v1.md` | **Hand-written.** Candidate taxonomy frozen for the pilot. Not final |
| `golden/annotation_guidelines.md` | **Hand-written.** Labelling rules and pre-declared revision triggers |
| `golden/README.md` | **Hand-written.** Why `golden/` is tracked in Git and how labels may be used |
| `golden/b01_pilot_labelled.csv` | **Written by the annotator.** No code writes to this path |

`golden/` is the one directory deliberately **not** gitignored — hand labels cannot
be regenerated from code. See D27.

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
