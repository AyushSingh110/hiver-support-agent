# Hiver Support Agent

An AI customer-support agent built on the *Customer Support on Twitter* dataset. For
one selected brand it classifies an incoming customer message into an intent, drafts
a reply grounded in how that brand historically resolved similar issues, and decides
whether to auto-handle or escalate — with a stated reason.

The project is built **evaluation-first**: the proof matters more than the system.

> **Status: Phase 1 complete (raw dataset profiling).**
> The brand has **not** been selected and conversations have **not** been
> reconstructed. Sections below are filled in as phases complete.

---

## 1. Project overview

*(Filled in once the system exists.)*

## 2. Problem framing

*(Filled in at Phase 9.)*

## 3. Selected brand

**@AmericanAir** — 24,429 clean conversations, 24,178 usable for retrieval.

Chosen by four screening gates (volume, public resolution, language, support purity)
applied to the 25 largest brands, then a weighted score over the 11 survivors.

| | AmericanAir | Delta | AmazonHelp |
| --- | --- | --- | --- |
| Clean conversations | 24,429 | 24,799 | 78,763 |
| DM deflection | **21.1%** | 25.5% | 1.5% |
| URL rate | **8.3%** | 23.5% | 65.6% |
| English (heuristic) | 99.2% | 99.8% | **81.9%** |

**Why not AmazonHelp, despite being three times larger?** It leads on nearly every
headline metric — but **18.1% of its opening messages are not English** (it is a
global handle), against under 2% for every other candidate. That would confound
intent classification, retrieval and LLM-judging at once. Its 65.6% URL rate also
means many replies are links rather than answers.

**Why not Delta?** Very close, and defensible. AmericanAir wins on DM deflection
(21.1% vs 25.5%) and decisively on URL rate (8.3% vs 23.5%) — it answers in prose,
which is better material to ground a generated reply in.

**Honest note:** the weighted score does *not* rank AmericanAir first — it ranks
GWRHelp first and AmericanAir sixth. The weights were not adjusted to change that.
GWRHelp's near-zero DM deflection dominates the heaviest criterion, but it is a
regional train operator with 9,577 conversations and roughly four distinguishable
intents. The full reasoning is in `docs/DECISION_LOG.md` (D22, D23).

**Dataset size alone is the wrong criterion**: AppleSupport is the second-largest
brand in the corpus and is unusable, because 64.4% of its conversations get pushed
to DM where the resolution is invisible.

## 4. Dataset

[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(Kaggle). One row is **one tweet**, not one conversation.

| Property | Value |
| --- | --- |
| Rows | 2,811,774 |
| File size | 516,508,641 bytes (~493 MB) |
| Columns | `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id` |
| Time range | 2008-05-08 to 2017-12-03 |
| Unique authors | 702,777 |
| Support accounts | 108 |

**Checksums** (SHA-256), so you can confirm you have the identical file:

```
sample.csv       22A2ABA84EF3B19CEB0AA452161474E9DB541B9C706F205645412A735E1F7F38
twcs/twcs.csv    CD297FCFA1BF6F99938BE242E8E578980BC6D1B96ADC8691ABEC9A39175B03C0
```

The dataset is **not committed** — it exceeds GitHub's 100 MB file limit. Download it
from Kaggle and extract it to `data/raw/` so the layout is:

```
data/raw/sample.csv
data/raw/twcs/twcs.csv
```

`data/raw/` is treated as immutable and is set read-only.

## 5. Dataset processing

Phase 1 writes a faithful Parquet cache at `data/interim/twcs.parquet` (~254 MB).
It is a **copy, not a transformation** — same rows, same values, no cleaning — so it
can be deleted and regenerated without changing any result.

## 6. Conversation reconstruction

**Done.** Reply links become conversations by following `in_response_to_tweet_id`
only; components are found by pointer doubling; turns are ordered by `created_at`.

| Status | Conversations | Tweets |
| --- | --- | --- |
| `clean` (1 customer + 1 brand) | **741,110** (92.85%) | 2,360,317 (83.94%) |
| `multi_customer` | 54,642 | 437,899 |
| `multi_brand` | 2,358 | 13,351 |
| `no_customer` | 87 | 207 |

Clean conversations: median 2 turns, mean 3.18. All 798,197 components are kept and
classified — downstream phases choose which statuses to use.

**Broadcasts are excluded by participant structure, not a fan-out threshold.** A
support interaction is a dyad; the largest component is an outage notice with 973
distinct authors. A fan-out rule would have been worse than useless here: replies
under high fan-out tweets are *more* likely to develop into real exchanges (78.8% vs
a 48.0% baseline), so a threshold would have deleted genuine support.

Reconstruction runs in ~1 minute and is byte-for-byte deterministic.

## 7. Intent taxonomy

**Discovery done; taxonomy proposed but UNVALIDATED.** 24,190 opening messages
clustered (49 of the 24,239 normalise to empty — mention-plus-URL only).

**The clustering is weak, and that is reported rather than tuned away.** TF-IDF +
SVD + KMeans gave 16.56% explained variance, silhouette of 0.037–0.054 across every
K from 6 to 20, one cluster holding 38.3% of the corpus, and an ARI of only 0.37
between preprocessing variants. Only ~37% of messages land in plausibly coherent
clusters. TF-IDF matches word overlap, and *"bag never showed up"* shares no tokens
with *"luggage missing"*.

What *is* reliable is the recurring issue vocabulary — baggage, delays,
cancellations, boarding, seats and upgrades, fare rules, staff conduct. The candidate
taxonomy is authored from that evidence and lives in
`reports/phase4_candidate_taxonomy.md`, hand-written and explicitly marked
unvalidated.

Next decision: accept a vocabulary-derived taxonomy, or invest in embeddings for a
better partition. Evidence in `docs/DECISION_LOG.md` D25.

**Pilot labelled and analysed; taxonomy decision pending.** 148 messages hand-labelled
by one annotator; 147 analysed after one row was excluded for a text/source mismatch
caught by validation. Prevalence is reported from the 119-row random stratum only.

Two of five pre-declared triggers fired: `OTHER` at 11.8% (threshold 10%), and
non-high confidence above 30% in six intents — worst is
`flight_cancellation_rebooking` at 67%, exactly the intent Phase 4 flagged as
suspect. Reading the `OTHER` rows showed they are **not** one missing category: six
are non-support travel commentary, seven are a genuine long tail of distinct
operational issues. Recommendation is minimal revision, not redesign.

**No agreement score is reported.** One annotator, one pass, so neither inter- nor
intra-annotator agreement is computable, and none was invented.

`golden/` is the one directory tracked in Git rather than ignored — hand labels are
the only artifact in this project that cannot be regenerated from code. Golden labels
are used for **evaluation only**: never for training, retrieval, prompt examples or
threshold tuning.

```
Raw tweets (2.81M)
    -> Conversation reconstruction   [done]  798,197 components, 741,110 clean
    -> Brand selection               [done]  AmericanAir, 24,429 conversations
    -> Intent discovery              [next]
    -> Historical resolution corpus
    -> Response generation + escalation
    -> Evaluation
```

**One constraint discovered in Phase 3:** despite 17 months of nominal coverage,
99.8% of AmericanAir conversations fall in Oct–Dec 2017. Any temporal split will
span weeks, not months.

## 8. Architecture

*(Phase 9. Not started.)*

## 9–16. Baselines, retrieval, escalation, evaluation, judge

*(Phases 8–14. Not started.)*

---

## Key findings from Phase 1

These drive the design of later phases.

**`tweet_id` is not chronological — timestamps are authoritative.**
Across 2,013,577 parent→child edges, the parent is **never** later in time than its
child (0 violations; 56 same-second ties). Yet only 44.5% of edges have a parent with
a lower `tweet_id`, and the Spearman correlation between `tweet_id` and time is just
0.34. Conversation ordering must use `created_at`, never `tweet_id`.

**`in_response_to_tweet_id` is far more reliable than `response_tweet_id`.**
The two columns agree on 91.95% of edges, but the disagreement is lopsided:
172,500 dangling child references versus only 3,862 dangling parent references.
Reconstruction should follow the parent column.

**Broadcast tweets are not conversations.**
472 tweets have 50+ direct replies — contests, product announcements and outage
notices (one AldiUK competition drew 1,755 replies). Treating these as conversations
would fabricate enormous fake threads.

**The data is effectively a three-month snapshot.**
Despite spanning 2008–2017, ~99.5% of tweets fall in Oct–Dec 2017. The long tail
back to 2008 is a negligible trickle. This constrains what a temporal split can mean
(Phase 6).

**Brands differ enormously in whether they resolve publicly.**
DM-deflection rate ranges from 0.5% (hulu_support) to 81.8% (TMobileHelp). A brand
that pushes conversations to private DMs leaves no visible resolution to ground a
reply in — this is the single most important brand-selection criterion.

**Tweet text contains embedded newlines.** 190,749 more physical lines than parsed
rows. Any line-based reader would corrupt the data; the CSV parser handles it.

---

## Reproducibility

### Environment

One conda environment for the whole project:

```bash
conda env create -f environment.yml
conda activate hiver
```

Dependencies are added in the phase that needs them, not upfront. Phase 1 needs only
pandas, numpy and pyarrow.

### Running Phase 1

```bash
# Fast correctness check on the 93-row sample (seconds)
python -m src.profile_raw --source sample

# Full profile (~2 minutes, 2.8M rows)
python -m src.profile_raw --source full
```

Outputs:

| Path | Contents |
| --- | --- |
| `reports/phase1_profile.md` | Human-readable profile |
| `reports/phase1_stats.json` | Every statistic, machine-readable |
| `data/interim/twcs.parquet` | Regenerable cache |

`reports/` and `data/` are **not tracked in Git** — they are generated output, fully
reproducible from the code plus the dataset. The findings that matter are summarised
above; run the profiler to regenerate the full detail.

The run is deterministic (fixed seed in `src/config.py`); re-running reproduces
identical statistics.

---

## File structure

```
data/raw/           immutable source data (git-ignored, read-only)
data/interim/       regenerable cache (git-ignored)
src/config.py       paths and constants
src/profile_raw.py  Phase 1 profiling
reports/            generated reports and statistics
docs/               development log, file guide, decision log
```

See `docs/FILE_GUIDE.md` for what each file does and which contain core logic.

## Documentation

| Document | Purpose |
| --- | --- |
| `docs/DEVELOPMENT_LOG.md` | Every step, error, root cause and fix, in order |
| `docs/FILE_GUIDE.md` | Per-file responsibilities and review priority |
| `docs/DECISION_LOG.md` | Non-obvious decisions, alternatives and tradeoffs |
