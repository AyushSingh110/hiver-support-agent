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

*Further decisions are appended as later phases are implemented.*
