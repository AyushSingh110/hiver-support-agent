# Decision Log

Non-obvious engineering decisions, with the alternatives that were rejected and
what each choice cost. Trivial decisions are deliberately left out.

---

## D1 — Write `.gitignore` before placing the dataset, not after

**Problem:** The raw CSV is ~493 MB. GitHub rejects files over 100 MB, and anything
committed once stays in Git history forever, even if a later commit deletes it.

**Alternatives:** (a) place data first, add the ignore rule afterwards;
(b) use Git LFS; (c) keep data outside the project entirely.

**Chosen:** Write the ignore rule first, verify it with `git check-ignore`, then
extract the data.

**Why:** Ordering removes the window in which a careless `git add .` could stage the
file. Verifying with `git check-ignore` gives real evidence — it prints the exact
rule and line number that matched — rather than an assumption. LFS was rejected
because it consumes quota for a file that anyone can re-download from Kaggle.

**Tradeoff:** Contributors must download the dataset themselves. Accepted; the
README documents it, and SHA-256 checksums let them confirm they have the same file.

**Consequence:** The repository stays small. Reproduction depends on documented
download instructions.

---

## D2 — Enforce raw-data immutability with OS permissions, not convention

**Problem:** The brief requires the raw dataset to stay untouched. A comment saying
"do not modify" prevents nothing.

**Alternatives:** (a) rely on code review and discipline; (b) checksum-only
verification after the fact; (c) OS-level read-only flag.

**Chosen:** Set the read-only flag on both raw files, and record SHA-256 checksums.

**Why:** Accidental writes now fail loudly at the operating-system level instead of
corrupting data silently. Checksums give an independent after-the-fact proof.

**Tradeoff:** Any legitimate future write to `data/raw/` needs the flag cleared
first — a deliberate, visible act.

**Consequence:** Immutability is enforced rather than intended. Verified twice with
two different tools, which caught nothing but established the standard.

---

## D3 — Read every CSV column as text, then validate

**Problem:** `tweet_id` looks like an integer, but declaring it as one means pandas
raises on the first malformed row and the whole run dies.

**Alternatives:** (a) declare real dtypes and let it crash; (b) declare dtypes with
`errors="coerce"` and accept silent nulls; (c) read everything as text and validate
explicitly.

**Chosen:** Read all seven columns as text, then convert and **count** failures,
keeping example values.

**Why:** The brief requires not silently discarding difficult cases. Malformed rows
become a reported number instead of a crash or an invisible null.

**Tradeoff:** More memory during the scan, and conversion code in the analysis.

**Consequence:** Malformed `tweet_id` and unparsed timestamp counts are reported
rather than guessed at.

---

## D4 — Cache the raw CSV as Parquet, as a copy and not a transformation

**Problem:** Re-parsing a 493 MB CSV in every later phase is slow.

**Alternatives:** (a) re-read the CSV every time; (b) cache a *cleaned* version;
(c) cache a faithful copy.

**Chosen:** A faithful Parquet copy — same rows, same values, no filtering, no
cleaning. `created_at` is even kept as its original text.

**Why:** A cleaned cache would quietly become the real dataset, and any cleaning bug
would silently propagate into every later phase. A faithful copy is only a speed
optimisation, so it can be deleted and regenerated without changing any result.

**Tradeoff:** ~254 MB of disk, on a machine with limited space. Mitigated by a
free-space check that fails clearly before writing.

**Consequence:** Later phases load in seconds. The raw CSV remains the single
source of truth.

---

## D5 — Test the "numeric `author_id` means customer" assumption instead of using it

**Problem:** In the first rows, brands look like text (`sprintcare`) and customers
look like numbers (`115712`). It is tempting to treat that as the rule.

**Alternatives:** (a) assume numeric means customer; (b) assume the `inbound` column
is authoritative; (c) cross-tabulate the two and measure.

**Chosen:** Cross-tabulate "looks numeric" against `inbound`, and separately count
authors that appear with **both** inbound values.

**Why:** It was an observation drawn from five rows. Building role detection on an
unverified pattern would put a silent bug in the foundation of every later phase.
Measuring costs one line and either confirms the rule or exposes the exceptions.

**Tradeoff:** None worth mentioning.

**Consequence:** Role detection rests on measured evidence.

---

## D6 — Measure chronology three independent ways

**Problem:** Early inspection suggested tweet IDs descend as a conversation
progresses. If reconstruction assumed the wrong direction, every conversation would
be built inside-out — and would still *look* plausible.

**Alternatives:** (a) assume IDs are chronological; (b) assume timestamps are
authoritative; (c) measure several ways and compare.

**Chosen:** Three measures — Spearman correlation of `tweet_id` against time; the
share of parent→child edges where the parent is earlier **in time**; the share where
the parent has a lower **ID**. Plus a hand check of a real thread.

**Why:** This is the single highest-risk assumption in the project. Independent
measures that must agree is the cheapest protection. Where they disagree, the
disagreement is the finding.

**Tradeoff:** Slightly more code than one correlation.

**Consequence:** Hand-checking thread `8 → 6 → 5 → 4 → 3 → 1 → 2` showed time
strictly increasing while IDs descend. Edges run **forward in time**; `tweet_id`
carries no chronological meaning. Phase 2 will order by timestamp, not by ID.

---

## D7 — Exploit the functional-graph property for cycle detection

**Problem:** Detecting cycles among ~2.8M nodes without exhausting time or stack.

**Alternatives:** (a) NetworkX; (b) recursive depth-first search; (c) a linear scan
using the structure of this specific graph.

**Chosen:** Each tweet has **at most one parent**, which makes the parent graph a
*functional graph*. That permits cycle detection in one linear pass with three-state
marking, written iteratively.

**Why:** NetworkX would mean a dependency and a large in-memory graph for a question
answerable with two arrays. Recursion would overflow the stack at this depth.

**Tradeoff:** Requires explaining why the graph is functional — but that explanation
is itself worth having.

**Consequence:** Cycle detection is O(n), dependency-free, and cannot blow the stack.

---

## D8 — Compute Spearman as Pearson-on-ranks to avoid adding scipy

**Problem:** `Series.corr(method="spearman")` in pandas routes through scipy.

**Alternatives:** (a) add scipy; (b) rank both series and take the Pearson
correlation, which is the definition of Spearman.

**Chosen:** `series.rank().corr(other.rank())`.

**Why:** Phase 1 needs exactly one statistic from scipy. Adding a large dependency
for one number contradicts the rule that dependencies arrive when genuinely needed.
The identity is exact, not an approximation.

**Tradeoff:** A reader must know the identity — noted in a comment.

**Consequence:** Phase 1 runs on pandas, numpy and pyarrow alone.

---

## D9 — Four passes rather than three, to keep the text column isolated

**Problem:** `text` is by far the largest column. Holding it in memory alongside the
relationship analysis risks exhausting RAM on a machine that is already short on
disk.

**Alternatives:** (a) load everything once and accept the memory cost; (b) sample
rows and lose exactness on global counts; (c) separate the text work into its own
streaming pass.

**Chosen:** Four passes — global scan and cache; relationships without text; authors
and brands without text; then a text pass that keeps only shortlisted brands.

**Why:** Relationship facts such as dangling references and cycles are *global*
properties, so sampling would give wrong answers; they must be exhaustive. Text
statistics are descriptive and tolerate being restricted to shortlisted brands.

**Tradeoff:** One more read of the cached data. This is a refinement of the approved
three-pass plan; it changes memory behaviour only, and produces the same statistics.

**Consequence:** Peak memory stays bounded. Recorded here because it deviates from
the plan as approved.

---

## D10 — Hand-curate `environment.yml` instead of using `conda env export`

**Problem:** `conda env export` produced 78 pinned packages and ended with
`prefix: C:\Users\ASUS\anaconda3\envs\hiver`.

**Alternatives:** (a) commit the full export; (b) commit the export with the prefix
line stripped; (c) hand-write a file listing only direct dependencies.

**Chosen:** A curated file with the four direct dependencies, no `prefix:` line.

**Why:** The exported prefix is a hardcoded personal path, which the Phase 19
reproducibility audit explicitly forbids. Pinning every transitive package also
pins Windows-specific builds such as `vs2015_runtime`, so the environment would fail
to build on Linux or macOS — exactly where a reviewer is likely to try it.

**Tradeoff:** Transitive versions are not locked, so a future solver run could
resolve slightly differently. Acceptable: direct dependencies are pinned to minor
versions, which is where behaviour changes would actually come from.

**Consequence:** The environment builds cross-platform and leaks no local paths.

---

## D11 — Run the profiler on `sample.csv` before the full dataset

**Problem:** A logic error discovered eight minutes into a full run wastes the run.

**Alternatives:** (a) go straight to the full file; (b) build a synthetic fixture;
(c) use the 93-row `sample.csv` that ships with the dataset.

**Chosen:** Run the identical code path on `sample.csv` first.

**Why:** Same code, same output files, seconds instead of minutes. Real data, so no
fixture to keep in sync.

**Tradeoff:** The sample is too small for meaningful statistics — it is a
*correctness* harness, not a measurement.

**Consequence:** It immediately caught a regex bug (a capturing group where a
non-capturing group was intended, which pandas warned about) and a counting bug in
how tweets-without-parent was totalled when duplicate IDs are present. Both were
fixed before the full run.

---

## D12 — Code is implementation, documentation is explanation

**Problem:** The Phase 1 source had drifted into carrying its own documentation:
multi-paragraph docstrings above most functions, separator banners, and a module
docstring with run instructions. The methodology was readable, but the control flow
was buried under prose.

**Alternatives:** (a) keep rich docstrings so the code is self-contained;
(b) strip comments entirely and rely on naming; (c) keep only one-line *why*
comments in code and move reasoning into the docs.

**Chosen:** (c). Source files keep single-line comments that explain something the
code cannot say itself. Everything longer lives in `FILE_GUIDE.md` (what a file
does), `DECISION_LOG.md` (why a choice was made) or `DEVELOPMENT_LOG.md` (what
happened).

**Why:** Run instructions inside a source file duplicate the README, and duplicated
instructions drift apart. Long docstrings restating methodology make the flow harder
to follow, not easier. And explanation living in docs can be ordered, cross-linked
and read end-to-end before an interview, which explanation scattered across
docstrings cannot.

**Tradeoff:** The code is no longer self-contained — a reader needs `FILE_GUIDE.md`
alongside it. Accepted, and stated at the top of that file.

**Consequence:** 9 docstrings, 5 separator banners and 1 module docstring removed;
13 one-line *why* comments kept. Behaviour verified unchanged by diffing the full
2.8M-row output against a pre-cleanup baseline.

---

## D13 — Keep generated reports out of Git

**Problem:** `reports/` holds generated Markdown and JSON that is rewritten on every
run. Tracking it means every re-run produces a large, noisy diff of numbers nobody
reviews line by line.

**Alternatives:** (a) track reports so results are visible on GitHub; (b) ignore
them and let the README carry results; (c) track only a trimmed summary.

**Chosen:** Ignore `reports/`, and make the README the public presentation of
results.

**Why:** Generated output is reproducible from the code plus the dataset, so it is
derived state rather than source. An interviewer wants the *conclusions* — which
belong in a curated README — not a 200-line auto-generated month-by-month table.
The reports remain on disk for our own analysis; they are untracked, not deleted.

**Tradeoff:** Someone cloning the repository must re-run the profiler (about two
minutes) to see the full detail. Acceptable, and documented in the README.

**Consequence:** The repository stays small and readable. The README must now be
kept genuinely current, since it is the only place results appear.

---

## D14 — `in_response_to_tweet_id` is the only relationship used to build edges

**Problem:** The dataset describes the same reply relationship twice, and the two
columns disagree.

**Alternatives:** (a) use `response_tweet_id`, which declares more edges;
(b) use both and merge; (c) use `in_response_to_tweet_id` alone and treat the other
as a cross-check.

**Chosen:** (c).

**Why:** Phase 1 measured 172,500 dangling child references against 3,862 dangling
parent references — `response_tweet_id` points at tweets that are not in the dataset
roughly 45 times more often. Merging both would import those broken edges. The
decisive confirmation came in Phase 2: **100.0% of the 2,013,577 edges built from
`in_response_to_tweet_id` are also declared by `response_tweet_id`**, while 172,500
declared edges cannot be built at all. The parent column is a strict, reliable subset.

**Tradeoff:** We build no edge for a reply whose parent is missing. Those become
truncated roots rather than being invented.

**Consequence:** Every edge in the output is corroborated by both columns.

---

## D15 — Broadcast detection by participant structure, not a fan-out threshold

**Problem:** Announcements and outage notices attract hundreds of unrelated replies.
Treating those as conversations would fabricate enormous fake threads.

**Alternatives:** (a) drop tweets above a fan-out threshold; (b) drop components
above a size threshold; (c) classify by how many distinct people are involved.

**Chosen:** (c). A component is `clean` only when it contains exactly one customer
and one brand.

**Why:** Fan-out is a proxy and a misleading one. Three measurements killed it:

1. Replies under high fan-out hubs are *more* likely to develop into real exchanges
   (78.8% have their own reply at fan-out ≥10, versus a 48.0% baseline), so a
   threshold would delete genuine support.
2. Cutting hubs barely helps structurally — the largest component shrinks only from
   1,390 to 650 tweets.
3. Participant count measures the thing itself. The largest component is an
   ATVIAssist outage notice with **973 distinct authors**. A support interaction is a
   dyad; that plainly is not one.

**Tradeoff:** 16.06% of tweets fall outside `clean`. Deliberate: we would rather lose
recall than let one fabricated conversation into the retrieval corpus.

**Consequence:** 741,110 clean conversations covering 83.94% of tweets. The seeded
examples confirm the rule behaves as intended — it separated a hulu_support dyad from
two customers chatting about M&S, from a customer complaining to Amazon *and* UPS
about one package, from LondonMidland's self-threaded outage updates.

---

## D16 — Keep multi-customer components intact rather than splitting them

**Problem:** A multi-customer component may hide several real dyads inside it.

**Alternatives:** (a) split into per-customer sub-conversations; (b) discard;
(c) keep intact, classify, and let later phases filter.

**Chosen:** (c).

**Why:** Splitting means deciding which brand turn answers which customer when a
brand replies to several people under one announcement. That is a second inference
stacked on top of reconstruction, and an error would silently manufacture a
conversation — precisely the failure this phase exists to prevent. With 741,110 clean
dyads already available, the extra ~5% is not needed.

**Tradeoff:** Some genuine exchanges stay locked inside multi-customer components.

**Consequence:** Nothing is deleted, so the decision is auditable and reversible.
Revisit in Phase 4 only if the selected brand proves data-poor.

---

## D17 — Pointer doubling for component discovery

**Problem:** Assign each of 2.81M tweets to its conversation root.

**Alternatives:** (a) NetworkX connected components; (b) walk parent pointers per
tweet with memoisation; (c) pointer doubling.

**Chosen:** (c). `pointer = pointer[pointer]` repeated until stable.

**Why:** Each round replaces a pointer with its parent's pointer, so reach doubles;
roots point at themselves, which halts the walk. Max depth is 649, so ~10 rounds
suffice. It is ~6 lines of vectorised numpy. NetworkX means a dependency and a large
in-memory graph object. The memoised walk is a Python loop over 2.81M nodes — both
slower *and* longer, once path-compression bookkeeping is included.

**Tradeoff:** Requires understanding the doubling idea. It is not chosen for
cleverness — the alternative is worse on both axes.

**Consequence:** Whole reconstruction runs in 0.9 minutes. The loop caps iterations
and raises rather than spinning forever, so a cycle would fail loudly.

---

## D18 — Record `customer_thread_count` without applying a threshold

**Problem:** Some brand accounts are anonymised as numeric IDs and are therefore
labelled customers. Measured: at least 179 such authors, touching 1,279 of 741,110
clean conversations (0.173%).

**Alternatives:** (a) ignore it; (b) emit a `suspect_customer_role` boolean using a
cutoff; (c) record the raw count and let later phases decide.

**Chosen:** (c).

**Why:** Any cutoff is arbitrary, and the measurement shows why it would misfire:
author `169172` received 447 replies but appears in only 2 components — a real
customer who went viral, not a brand. Freezing a threshold into the data would bake
that error in permanently. Recording the raw count keeps the judgement visible and
deferrable at the cost of one groupby.

**Tradeoff:** Downstream phases must decide for themselves.

**Consequence:** Role inference is honest about its uncertainty instead of hiding it
behind a boolean.

---

## D19 — `conversation_id = conv_{root_tweet_id}`

**Problem:** Conversations need stable identifiers.

**Alternatives:** (a) sequential counter; (b) hash of members; (c) derive from the
root tweet.

**Chosen:** (c).

**Why:** A counter depends on row order and changes between runs. A hash is stable
but opaque. The root tweet ID is already unique per component, traceable straight
back to the raw CSV, and needs no extra machinery.

**Tradeoff:** If a component were ever split, IDs would change — acceptable, since
D16 means we do not split.

**Consequence:** Re-running produces byte-identical output, verified by SHA-256.

---

## D20 — Preserve Phase 1's fan-out measurement rather than restate it

**Problem:** Phase 1 reported 472 tweets with 50+ replies. Phase 2 measures 62.

**Alternatives:** (a) correct the Phase 1 report; (b) keep both and explain.

**Chosen:** (b).

**Why:** Neither number is wrong. Phase 1 counted **declared** children from
`response_tweet_id`; Phase 2 counts **resolvable** children from actual parent edges.
The gap is not an error — it *is* the evidence that the two relationship columns have
different reliability, which is what justifies D14. Overwriting the earlier figure
would destroy the reasoning that led to the decision.

**Tradeoff:** A reader meeting both numbers needs the explanation.

**Consequence:** Both are reported, with the distinction stated wherever either
appears.

---

## D21 — A fifth status, `no_brand`, to make classification a total partition

**Problem:** The sample harness failed the `clean_conversations_are_dyads` invariant.
A component with one customer and **zero** brands was falling through sequential
overwrites and landing on `clean`.

**Alternatives:** (a) special-case it into an existing status; (b) leave it, since it
never occurs in the full dataset; (c) add a fifth status and make the classification
an explicit total partition.

**Chosen:** (c), using `np.select` with stated precedence.

**Why:** The approved design named four statuses, but those four did not cover the
whole space. Leaving a gap is how the bug happened. `no_brand` occurs **0 times** in
the full dataset — but it occurs in the sample, and a classification that is only
correct on one input is not correct.

**Tradeoff:** One status more than the approved design. Flagged rather than applied
silently.

**Consequence:** Bug caught by the harness before the full run. Full dataset:
`no_brand` = 0, as predicted.

---

## D22 — AmericanAir is the selected brand

**Decision:** Build the support agent for **@AmericanAir**. Brand evaluation stops
here; later phases use this corpus.

**Problem:** One brand must be chosen from 108 support accounts, and dataset size is
the obvious but wrong criterion.

**Alternatives considered:** AmazonHelp, AppleSupport, Delta, British_Airways,
SouthwestAir, GWRHelp, and 19 others measured in full.

**Chosen:** AmericanAir — 24,429 clean conversations, 24,178 usable for retrieval.

**Why:**

- **Public resolution.** 21.1% conversation-level DM deflection, the lowest of the
  large English candidates. The assignment requires replies grounded in how the brand
  *historically resolved* issues; if the fix happened in DM that evidence does not
  exist.
- **Lowest URL rate in the shortlist — 8.3%.** AmericanAir answers in prose rather
  than deflecting to a help page. Delta is 23.5%, AmazonHelp 65.6%. Prose replies are
  far better grounding material than a link.
- **Volume.** 24,429 conversations supports a 150-250 golden set, a disjoint
  retrieval corpus and held-out evaluation.
- **Diversity.** Airline support spans delays, baggage, seating, booking changes,
  check-in, refunds and loyalty — enough for a meaningful 8-12 intent taxonomy.
- **Language.** 99.2% English by the heuristic.

**Why AmazonHelp was rejected despite being three times larger:** it leads on
almost every headline metric — most conversations (78,763), lowest DM deflection
(1.5%), deepest threads (46.3% with 4+ turns), highest measured diversity. But
**18.1% of its opening messages are not English** (8.4% CJK, 9.7% non-English
function words) against ≤1.7% for every other candidate. @AmazonHelp is a global
multilingual handle. That would confound intent classification, retrieval and
LLM-judging simultaneously. It also has a 65.6% URL rate, so many replies are links
rather than answers. Its apparent diversity advantage is partly an artifact of
foreign vocabulary inflating the type-token ratio, not richer support topics.

**Why Delta was not selected:** genuinely close — 24,799 conversations, 99.8%
English, and *more* brand engagement (32.3% of conversations have 2+ brand turns
against AmericanAir's 25.8%). It loses on two measurements: DM deflection 25.5% vs
21.1%, and URL rate 23.5% vs 8.3%. The URL gap decided it. Delta remains a
defensible alternative and the two sit within noise of each other.

**Tradeoffs and limitations:**

- Median conversation is 2 turns; only 27.4% reach 4+ turns. Many exchanges are a
  single question and a single reply.
- 21.1% of conversations still deflect to DM, so those resolutions are invisible.
- **Temporal concentration:** despite 17 months of nominal coverage, 99.8% of
  conversations fall in Oct-Dec 2017. A temporal split will span weeks, not months.
- Airline support is a specific domain; conclusions will not transfer to, say,
  software support.
- Language measurement is a heuristic, not a detector.

**Consequence:** Phases 4 onward operate on 24,429 clean AmericanAir conversations —
24,239 customer-rooted opening messages and a 24,178-conversation retrieval pool.

---

## D23 — Gates before scoring, and the score is not the decision

**Problem:** A single weighted score can be dragged upward by strengths that do not
compensate for a disqualifying weakness.

**Alternatives:** (a) rank purely by weighted score; (b) pick by judgement;
(c) hard screening gates first, then score only the survivors.

**Chosen:** (c). Four gates — volume, DM deflection, language, support purity — then
a weighted score over survivors, with all raw metrics published.

**Why:** AmazonHelp is the proof. It would rank at or near the top of any weighted
score built from headline metrics, while failing a gate that makes it unusable. Gates
express "this is disqualifying" in a way a weighted average cannot.

**The honest result, recorded because it matters:** the computed score does **not**
rank AmericanAir first. It ranks **GWRHelp first (0.634)**, then VirginTrains
(0.604), Delta (0.602), and **AmericanAir sixth (0.569)**.

The score was not adjusted to make the chosen brand win. The reason to override it:
min-max normalisation within the surviving set rewards whichever brand is most
*extreme* on the heaviest criterion. GWRHelp's 2.6% DM deflection nearly maxes the
30% public-resolution weight on its own. But GWRHelp is a regional UK train operator
with 9,577 conversations and a narrow issue space — its opening vocabulary is
`train, paddington, late, ticket, delayed, cancelled`. That yields perhaps four
intents and a trivial classifier. The type-token ratio, at 15%, does not capture
"narrow domain" well enough to offset this.

**Tradeoff:** Overriding a computed ranking requires justification, which is why it
is written here in full rather than buried.

**Consequence:** The decision rests on the gates plus named measurements (volume,
URL rate, domain breadth), with the score as a cross-check that visibly disagreed.
This is the concrete example behind "what is misleading about my headline number?"

---

## D24 — scikit-learn added; embeddings deliberately not

**Problem:** Intent discovery needs TF-IDF, dimensionality reduction and clustering.
The environment had only pandas, numpy and pyarrow.

**Alternatives:** (a) hand-roll TF-IDF and Lloyd's algorithm in numpy;
(b) add scikit-learn; (c) jump straight to sentence-transformers.

**Chosen:** (b). scikit-learn 1.9.0 (pulls scipy 1.17.1) into the existing `hiver`
environment. No second environment, no torch, no embedding model, no LLM.

**Why:** Hand-rolling means reimplementing sublinear TF scaling, sparse handling,
k-means++ init and randomized SVD — more code and more bug surface than importing a
standard library, which contradicts the simplicity rule rather than serving it.
Embeddings were not chosen up front because the cheap method had not yet been shown
to fail; paying 2-3 GB of a constrained disk before evidence would be premature.

**Tradeoff:** ~1.7 GB of disk (8.7 GB free before, 7.0 GB after).

**Consequence:** Phase 4 runs in 0.64 minutes on 24,190 messages.

---

## D25 — TF-IDF clustering reported as inadequate rather than escalated

**Problem:** The approved plan said that if TF-IDF proved inadequate I should stop
and show the evidence before proposing embeddings. It did prove inadequate.

**The evidence:**

| Measure | Result |
| --- | --- |
| SVD explained variance, 100 components | **16.56%** |
| Silhouette, every K from 6 to 20 | **0.037 - 0.054** |
| Largest cluster at the best K (20) | **38.3%** |
| Largest cluster at K=6 | 53.5% |
| ARI between preprocessing variants | **0.37** |
| Corpus in plausibly coherent clusters | **37.0%** |
| Corpus in `fly`/`flying`/`travel`/`way` clusters | **11.0%** |

Plus three qualitative failures: the same intent split across clusters (delays in 5
and 17; complaints in 8 and 14), a language appearing as if it were an intent
(cluster 18 is Spanish), and cluster 11 holding 15.2% of the corpus while its
distinctive terms disagree with its own centroid-nearest messages.

**Alternatives:** (a) tune preprocessing and K until the numbers look better;
(b) quietly switch to embeddings; (c) report the failure and stop.

**Chosen:** (c).

**Why:** (a) is metric-gaming — the instruction was explicit not to optimise for a
nicer-looking result, and a tuned partition would still be lexical. (b) would hide
the most useful finding in the phase. TF-IDF measures **word overlap**, and the
corpus is short, informal, emoji-laden and paraphrase-heavy: *"bag never showed up"*
and *"luggage missing"* share no tokens. That is a property of the data, not a
tuning problem.

**Tradeoff:** Phase 4 ends without a validated taxonomy. That was never promised.

**Consequence:** The taxonomy is authored from **recurring term evidence**, which is
reliable, rather than from the cluster partition, which is not. The embeddings
decision is deferred to a human with the evidence in hand.

---

## D26 — Taxonomy kept in a separate hand-authored file

**Problem:** The three layers (unsupervised cluster, human interpretation, proposed
intent) must stay distinguishable. Putting all three in one generated report makes
the boundary cosmetic — and a re-run would overwrite the human work.

**Alternatives:** (a) hardcode the taxonomy into the script so it appears in the
generated report; (b) append it to the generated report after each run; (c) keep it
in a separate hand-authored file.

**Chosen:** (c). `reports/phase4_intent_discovery.md` is machine output;
`reports/phase4_candidate_taxonomy.md` is hand-written and states so at the top.

**Why:** Hardcoding interpretation into a script would make an authored judgement
look like a computed result — the exact confusion this phase is meant to avoid. A
separate file makes the boundary structural: one file is reproducible from code, the
other is a person's reading of it.

**Tradeoff:** Two files instead of one; the taxonomy is not regenerated on re-run,
which is the intended behaviour.

**Consequence:** The taxonomy is explicitly marked UNVALIDATED and carries its own
open questions, so no later phase can mistake it for a validated result.

---

*Further decisions are appended as later phases are implemented.*
