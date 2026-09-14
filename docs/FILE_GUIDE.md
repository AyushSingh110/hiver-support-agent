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

---

## Configuration

| File | Purpose |
| --- | --- |
| `.gitignore` | Keeps `data/` and secrets out of Git. Written before the data was placed, deliberately |
| `environment.yml` | The single conda environment, `hiver` |
